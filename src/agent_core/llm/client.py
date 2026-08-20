from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from openai import AsyncOpenAI, OpenAI

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
    ) -> dict[str, Any]: ...


@runtime_checkable
class AsyncChatModel(Protocol):
    """Optional async model capability consumed by ``ModelNode.aexec``."""

    async def achat_message(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


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
        async_client: AsyncOpenAI | None = None,
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
        if client is None and async_client is None and not api_key:
            raise RuntimeError(
                "Pass api_key= or set LLM_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY."
            )
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = (
            client if client is not None else (OpenAI(**client_kwargs) if api_key else None)
        )
        self.async_client = (
            async_client
            if async_client is not None
            else (AsyncOpenAI(**client_kwargs) if api_key else None)
        )

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
        if self.client is None:
            raise RuntimeError("This LLM has only an async client; use achat_message().")
        request, stream, on_delta, on_reasoning_delta = self._request(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            kwargs=kwargs,
        )
        response = self.client.chat.completions.create(**request)
        if stream:
            return _stream_message(
                response,
                on_delta=on_delta,
                on_reasoning_delta=on_reasoning_delta,
            )
        return _message_to_dict(response.choices[0].message, getattr(response, "usage", None))

    async def achat_message(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async counterpart of :meth:`chat_message` using ``AsyncOpenAI``."""
        if self.async_client is None:
            return await asyncio.to_thread(
                self.chat_message,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
        request, stream, on_delta, on_reasoning_delta = self._request(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            kwargs=kwargs,
        )
        response = await self.async_client.chat.completions.create(**request)
        if stream:
            return await _astream_message(
                response,
                on_delta=on_delta,
                on_reasoning_delta=on_reasoning_delta,
            )
        return _message_to_dict(response.choices[0].message, getattr(response, "usage", None))

    def _request(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: str | Mapping[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        bool,
        Callable[[str], None] | None,
        Callable[[str], None] | None,
    ]:
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
        return request, stream, on_delta, on_reasoning_delta


def _message_to_dict(message: Any, usage: Any = None) -> dict[str, Any]:
    result = {"role": "assistant", "content": message.content or ""}
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        result["reasoning_content"] = reasoning_content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [
            item.model_dump() if hasattr(item, "model_dump") else item for item in tool_calls
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
    tool_calls = _StreamToolCallMerger()
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
                tool_calls.merge(item, position)

    message: dict[str, Any] = {"role": "assistant", "content": "".join(parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    merged_calls = tool_calls.result()
    if merged_calls:
        message["tool_calls"] = merged_calls
    if usage is not None:
        message["usage"] = _usage_to_dict(usage)
    return message


async def _astream_message(
    chunks: Any,
    on_delta: Callable[[str], None] | None,
    on_reasoning_delta: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls = _StreamToolCallMerger()
    usage: Any = None
    async for chunk in chunks:
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
                tool_calls.merge(item, position)

    message: dict[str, Any] = {"role": "assistant", "content": "".join(parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    merged_calls = tool_calls.result()
    if merged_calls:
        message["tool_calls"] = merged_calls
    if usage is not None:
        message["usage"] = _usage_to_dict(usage)
    return message


def _usage_to_dict(usage: Any) -> Any:
    return usage.model_dump() if hasattr(usage, "model_dump") else usage


class _StreamToolCallMerger:
    """Merge standard and indexless OpenAI-compatible tool-call deltas."""

    def __init__(self) -> None:
        self.calls: dict[int, dict[str, Any]] = {}
        self.last_slot = -1
        # Indexless providers usually preserve the ordinal position of each
        # call inside a delta even when they omit the OpenAI ``index`` field.
        # Remember that position so a later chunk containing several
        # argument fragments cannot append all of them to ``last_slot``.
        self.position_slots: dict[int, int] = {}

    def merge(self, item: Any, fallback_index: int) -> None:
        item_id = str(_get(item, "id") or "")
        raw_index = _get(item, "index")
        if isinstance(raw_index, int):
            slot = raw_index
        elif item_id:
            mapped_slot = self.position_slots.get(fallback_index)
            if mapped_slot is not None and self.calls[mapped_slot].get("id") == item_id:
                slot = mapped_slot
            elif mapped_slot is not None or fallback_index > 0:
                # A different ID at an established position starts a new
                # call. ``fallback_index > 0`` also keeps duplicate IDs in
                # the same indexless delta separate so ``result`` can assign
                # them unique IDs.
                slot = max(self.calls, default=-1) + 1
            else:
                slot = next(
                    (index for index, call in self.calls.items() if call.get("id") == item_id),
                    max(self.calls, default=-1) + 1,
                )
        elif fallback_index in self.position_slots:
            slot = self.position_slots[fallback_index]
        elif fallback_index > 0:
            slot = max(self.calls, default=-1) + 1
        elif self.last_slot >= 0:
            slot = self.last_slot
        else:
            slot = fallback_index
        self.last_slot = slot
        self.position_slots[fallback_index] = slot

        current = self.calls.setdefault(
            slot,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if item_id:
            current["id"] = item_id
        if item_type := _get(item, "type"):
            current["type"] = item_type

        function = _get(item, "function") or {}
        current_function = current["function"]
        if name := _get(function, "name"):
            current_function["name"] = _merge_fragment(current_function["name"], str(name))
        if arguments := _get(function, "arguments"):
            current_function["arguments"] = _merge_fragment(
                current_function["arguments"],
                str(arguments),
            )

    def result(self) -> list[dict[str, Any]]:
        calls = [self.calls[index] for index in sorted(self.calls)]
        seen: set[str] = set()
        for index, call in enumerate(calls):
            call_id = str(call.get("id") or "")
            if not call_id or call_id in seen:
                call_id = _unique_tool_call_id(call_id, index, seen)
                call["id"] = call_id
            seen.add(call_id)
        return calls


def _merge_fragment(current: str, fragment: str) -> str:
    if not fragment or fragment == current:
        return current
    if fragment.startswith(current):
        return fragment
    return current + fragment


def _unique_tool_call_id(base: str, index: int, seen: set[str]) -> str:
    candidate = f"{base}_{index}" if base else f"call_{index}"
    while candidate in seen:
        candidate += "x"
    return candidate


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _env(*names: str) -> str | None:
    return next((value for name in names if (value := os.getenv(name))), None)
