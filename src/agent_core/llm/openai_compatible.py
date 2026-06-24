from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agent_core.models import Message


class OpenAICompatibleChatModel:
    """ChatModel adapter for OpenAI-compatible chat completion APIs."""

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
        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": stream,
            **kwargs,
        }
        if tools is not None:
            request["tools"] = list(tools)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        extra_body = dict(self.default_extra_body)
        extra_body.update(dict(request.pop("extra_body", {}) or {}))
        if extra_body:
            request["extra_body"] = extra_body

        if stream:
            return self._chat_message_stream(request, on_delta=on_delta)

        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        result: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            result["tool_calls"] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in tool_calls
            ]

        usage = getattr(response, "usage", None)
        if usage is not None:
            result["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else usage
        return result

    def _chat_message_stream(
        self,
        request: dict[str, Any],
        *,
        on_delta: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        response = self.client.chat.completions.create(**request)

        for chunk in response:
            choices = getattr(chunk, "choices", [])
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)
                if on_delta is not None:
                    on_delta(content)

            for tool_call_delta in getattr(delta, "tool_calls", None) or []:
                _merge_tool_call_delta(tool_calls, tool_call_delta)

        result: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if tool_calls:
            result["tool_calls"] = [
                tool_calls[index]
                for index in sorted(tool_calls)
                if tool_calls[index].get("function", {}).get("name")
            ]
        return result


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
    """Build an OpenAI-compatible model from environment variables."""

    load_dotenv(dotenv_path=env_file)
    api_key = _get_env(api_key_env, api_key_aliases)
    if not api_key:
        aliases = ", ".join([api_key_env, *api_key_aliases])
        raise RuntimeError(f"Set one of these environment variables in .env: {aliases}.")

    base_url = _get_env(base_url_env, base_url_aliases) or default_base_url
    model = _get_env(model_env, model_aliases) or default_model
    extra_body = _default_extra_body(
        base_url=base_url,
        disable_deepseek_thinking=disable_deepseek_thinking,
    )
    return OpenAICompatibleChatModel(
        api_key=api_key,
        base_url=base_url,
        model=model,
        default_extra_body=extra_body,
    )


def _get_env(primary: str, aliases: Sequence[str]) -> str | None:
    for key in [primary, *aliases]:
        value = os.getenv(key)
        if value:
            return value
    return None


def _default_extra_body(
    *,
    base_url: str,
    disable_deepseek_thinking: bool,
) -> dict[str, Any]:
    thinking = os.getenv("OPENAI_THINKING") or os.getenv("DEEPSEEK_THINKING")
    if thinking:
        return {"thinking": {"type": thinking}}
    if disable_deepseek_thinking and "deepseek" in base_url.lower():
        return {"thinking": {"type": "disabled"}}
    return {}


def _merge_tool_call_delta(
    tool_calls: dict[int, dict[str, Any]],
    delta: Any,
) -> None:
    index = int(getattr(delta, "index", 0) or 0)
    item = tool_calls.setdefault(
        index,
        {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )

    delta_id = getattr(delta, "id", None)
    if delta_id:
        item["id"] = delta_id
    delta_type = getattr(delta, "type", None)
    if delta_type:
        item["type"] = delta_type

    function = getattr(delta, "function", None)
    if function is None:
        return
    name = getattr(function, "name", None)
    if name:
        item["function"]["name"] += name
    arguments = getattr(function, "arguments", None)
    if arguments:
        item["function"]["arguments"] += arguments
