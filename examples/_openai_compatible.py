from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


class OpenAICompatibleChatModel:
    """ChatModel adapter for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat_message(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            **kwargs,
        }
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        extra_body = dict(request.pop("extra_body", {}) or {})
        thinking = os.getenv("OPENAI_THINKING") or os.getenv("DEEPSEEK_THINKING")
        if thinking:
            extra_body.setdefault("thinking", {"type": thinking})
        elif _is_deepseek_url(str(self.client.base_url)):
            extra_body.setdefault("thinking", {"type": "disabled"})
        if extra_body:
            request["extra_body"] = extra_body

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


def build_model_from_env() -> OpenAICompatibleChatModel:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env.")
    return OpenAICompatibleChatModel(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com",
        model=os.getenv("OPENAI_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-v4-flash",
    )


def safe_print(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


def _is_deepseek_url(base_url: str) -> bool:
    return "deepseek" in base_url.lower()

