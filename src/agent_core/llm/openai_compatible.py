from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agent_core.llm.models import Message


class OpenAICompatibleChatModel:
    """Small ChatModel adapter around the OpenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        client: OpenAI | None = None,
        default_extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_url = base_url
        self.default_extra_body = dict(default_extra_body or {})

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
        request = self._request(messages, tools, tool_choice, kwargs)
        request["stream"] = stream
        if request["stream"]:
            return _stream_message(
                self.client.chat.completions.create(**request),
                on_delta=on_delta,
            )

        response = self.client.chat.completions.create(**request)
        return _message_to_dict(response.choices[0].message, getattr(response, "usage", None))

    def _request(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: str | Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = {
            "model": self.model,
            "messages": list(messages),
            **kwargs,
        }
        if tools:
            request["tools"] = list(tools)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        extra_body = {**self.default_extra_body, **dict(request.pop("extra_body", {}) or {})}
        if extra_body:
            request["extra_body"] = extra_body
        return request


def build_model_from_env(
    *,
    env_file: str | os.PathLike[str] | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    model_env: str = "OPENAI_MODEL",
    api_key_aliases: Sequence[str] = ("DEEPSEEK_API_KEY",),
    base_url_aliases: Sequence[str] = ("DEEPSEEK_BASE_URL",),
    model_aliases: Sequence[str] = ("DEEPSEEK_MODEL",),
    default_base_url: str = "https://api.deepseek.com",
    default_model: str = "deepseek-v4-flash",
    disable_deepseek_thinking: bool = True,
) -> OpenAICompatibleChatModel:
    load_dotenv(dotenv_path=env_file)
    api_key = _first_env(api_key_env, *api_key_aliases)
    if not api_key:
        raise RuntimeError(
            f"Set one of these environment variables in .env: "
            f"{', '.join([api_key_env, *api_key_aliases])}."
        )

    base_url = _first_env(base_url_env, *base_url_aliases) or default_base_url
    extra_body = _deepseek_extra_body(base_url, disable_deepseek_thinking)
    return OpenAICompatibleChatModel(
        api_key=api_key,
        base_url=base_url,
        model=_first_env(model_env, *model_aliases) or default_model,
        default_extra_body=extra_body,
    )


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


def _first_env(*names: str) -> str | None:
    return next((value for name in names if (value := os.getenv(name))), None)


def _deepseek_extra_body(base_url: str, disabled: bool) -> dict[str, Any]:
    thinking = os.getenv("OPENAI_THINKING") or os.getenv("DEEPSEEK_THINKING")
    if thinking:
        return {"thinking": {"type": thinking}}
    return {"thinking": {"type": "disabled"}} if disabled and "deepseek" in base_url.lower() else {}
