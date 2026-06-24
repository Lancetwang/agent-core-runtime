from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

Message = Mapping[str, Any]


class ChatModel(Protocol):
    """Minimal protocol required by LLM nodes."""

    def chat_message(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...
