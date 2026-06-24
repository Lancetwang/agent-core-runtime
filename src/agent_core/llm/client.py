from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import OpenAI

Message = Mapping[str, Any]


class ChatModel(Protocol):
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

    If values are not passed explicitly, they are read from `.env`:
    `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, or the compatible
    `OPENAI_*` / `DEEPSEEK_*` aliases.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        env_file: str | os.PathLike[str] | None = None,
        client: OpenAI | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        load_dotenv(dotenv_path=env_file)
        self.base_url = base_url or _env("LLM_BASE_URL", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL")
        self.base_url = self.base_url or "https://api.deepseek.com"
        self.model = model or _env("LLM_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL")
        self.model = self.model or "deepseek-v4-flash"
        self.extra_body = {
            **_default_extra_body(self.base_url),
            **dict(extra_body or {}),
        }

        api_key = api_key or _env("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
        if client is None and not api_key:
            raise RuntimeError(
                "Set LLM_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY in .env."
            )
        self.client = client or OpenAI(api_key=api_key, base_url=self.base_url)

    def chat_message(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        on_delta = kwargs.pop("on_delta", None)
        stream = bool(kwargs.pop("stream", False))
        request = {
            "model": self.model,
            "messages": list(messages),
            "stream": stream,
            **kwargs,
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
            return _stream_message(response, on_delta=on_delta)
        return _message_to_dict(response.choices[0].message, getattr(response, "usage", None))


def _message_to_dict(message: Any, usage: Any = None) -> dict[str, Any]:
    result = {"role": "assistant", "content": message.content or ""}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in tool_calls
        ]
    if usage is not None:
        result["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else usage
    return result


def _stream_message(chunks: Any, on_delta: Callable[[str], None] | None) -> dict[str, Any]:
    parts: list[str] = []
    for chunk in chunks:
        for choice in getattr(chunk, "choices", []):
            text = getattr(getattr(choice, "delta", None), "content", None)
            if text:
                parts.append(text)
                if on_delta:
                    on_delta(text)
    return {"role": "assistant", "content": "".join(parts)}


def _env(*names: str) -> str | None:
    return next((value for name in names if (value := os.getenv(name))), None)


def _default_extra_body(base_url: str) -> dict[str, Any]:
    thinking = os.getenv("LLM_THINKING") or os.getenv("OPENAI_THINKING") or os.getenv("DEEPSEEK_THINKING")
    if thinking:
        return {"thinking": {"type": thinking}}
    if "deepseek" in base_url.lower():
        return {"thinking": {"type": "disabled"}}
    return {}
