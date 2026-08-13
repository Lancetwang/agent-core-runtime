from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from openai import OpenAI

Message = Mapping[str, Any]
"""One OpenAI-style chat message: ``{"role": ..., "content": ..., ...}``."""


class ChatModel(Protocol):
    """Runtime model protocol normalized to the OpenAI wire format.

    This is a thin seam, not a provider abstraction: messages, tool schemas,
    ``tool_choice``, and streaming options keep their OpenAI shapes, and the
    assistant message it returns is an OpenAI-style dict
    (``{"role": "assistant", "content": str, "tool_calls": [...]?, "usage": {...}?}``).
    Adapters for non-OpenAI providers translate their native schema at this
    boundary; "replaceable" therefore means "another OpenAI-compatible
    endpoint, or a one-method adapter", not pluggable native protocols.
    """

    def chat_message(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...


class LLM:
    """Default OpenAI-compatible chat model.

    If values are not passed explicitly, they are read from the process environment:
    `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, or the compatible
    `OPENAI_*` / `DEEPSEEK_*` aliases. The aliases and the optional
    `LLM_THINKING` -> ``extra_body["thinking"]`` convenience exist because
    OpenAI-compatible deployments are the common case; provider-specific
    knobs still travel through ``extra_body``/``chat_kwargs`` as plain data.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: OpenAI | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url or _env("LLM_BASE_URL", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL")
        self.model = model or _env("LLM_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL")
        if not self.model:
            raise RuntimeError("Pass model= or set LLM_MODEL, OPENAI_MODEL, or DEEPSEEK_MODEL.")
        thinking = _env("LLM_THINKING", "OPENAI_THINKING", "DEEPSEEK_THINKING")
        self.extra_body = {
            **({"thinking": {"type": thinking}} if thinking else {}),
            **dict(extra_body or {}),
        }

        api_key = api_key or _env("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
        if client is None and not api_key:
            raise RuntimeError(
                "Pass api_key= or set LLM_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY."
            )
        client_kwargs = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = client if client is not None else OpenAI(**client_kwargs)

    def chat_message(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send one chat completion request and return the assistant message.

        Pass ``stream=True`` to stream; text chunks are forwarded to the
        optional ``on_delta`` / ``on_reasoning_delta`` callbacks and usage is
        still captured via ``stream_options``. Remaining kwargs go to the
        provider unchanged.
        """
        on_delta = kwargs.pop("on_delta", None)
        on_reasoning_delta = kwargs.pop("on_reasoning_delta", None)
        stream = bool(kwargs.pop("stream", False))
        request = {
            "model": self.model,
            "messages": list(messages),
            "stream": stream,
            **kwargs,
        }
        if stream:
            request["stream_options"] = {
                "include_usage": True,
                **dict(request.get("stream_options") or {}),
            }
        if tools:
            request["tools"] = list(tools)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        extra_body = {**self.extra_body, **dict(request.pop("extra_body", {}) or {})}
        if extra_body:
            request["extra_body"] = extra_body

        response = self.client.chat.completions.create(**request)
        if stream:
            return _stream_message(
                response,
                on_delta=on_delta,
                on_reasoning_delta=on_reasoning_delta,
            )
        return _message_to_dict(response.choices[0].message, getattr(response, "usage", None))


def _message_to_dict(message: Any, usage: Any = None) -> dict[str, Any]:
    result = {"role": "assistant", "content": message.content or ""}
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        result["reasoning_content"] = reasoning_content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in tool_calls
        ]
    if usage is not None:
        result["usage"] = _usage_to_dict(usage)
    return result


def _stream_message(
    chunks: Any,
    on_delta: Callable[[str], None] | None,
    on_reasoning_delta: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: Any = None
    for chunk in chunks:
        usage = _get(chunk, "usage", usage)
        for choice in getattr(chunk, "choices", []):
            delta = getattr(choice, "delta", None)
            text = _get(delta, "content")
            if text:
                parts.append(text)
                if on_delta:
                    on_delta(text)
            if reasoning := _get(delta, "reasoning_content"):
                reasoning_parts.append(reasoning)
                if on_reasoning_delta:
                    on_reasoning_delta(reasoning)
            for position, item in enumerate(_get(delta, "tool_calls") or []):
                _merge_stream_tool_call(tool_calls, item, position)

    message = {"role": "assistant", "content": "".join(parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    if usage is not None:
        message["usage"] = _usage_to_dict(usage)
    return message


def _usage_to_dict(usage: Any) -> Any:
    return usage.model_dump() if hasattr(usage, "model_dump") else usage


def _merge_stream_tool_call(
    tool_calls: dict[int, dict[str, Any]],
    item: Any,
    fallback_index: int,
) -> None:
    index = _get(item, "index", fallback_index)
    current = tool_calls.setdefault(
        int(index),
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if item_id := _get(item, "id"):
        current["id"] = item_id
    if item_type := _get(item, "type"):
        current["type"] = item_type

    function = _get(item, "function") or {}
    if name := _get(function, "name"):
        current["function"]["name"] += name
    if arguments := _get(function, "arguments"):
        current["function"]["arguments"] += arguments


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _env(*names: str) -> str | None:
    return next((value for name in names if (value := os.getenv(name))), None)
