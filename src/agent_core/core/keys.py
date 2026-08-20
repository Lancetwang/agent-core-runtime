"""Canonical payload state key names shared by the built-in nodes.

Custom flows read and write the flow payload through these keys. Centralizing
the built-in names reduces repeated magic strings, but arbitrary custom
payload mappings remain dynamically typed.
"""

from __future__ import annotations


class PayloadKeys:
    """Canonical keys of the built-in flow payload state."""

    INPUT = "input"
    """Business text passed to the flow by ``Agent.chat``."""

    ANSWER = "answer"
    """Final answer text stored by ``ToolRouterNode``."""

    ASSISTANT_MESSAGE = "assistant_message"
    """Last raw assistant message stored by ``ModelNode``."""

    HISTORY = "history"
    """Message history a custom flow may seed and carry in the payload."""

    CHAT_KWARGS = "chat_kwargs"
    """Per-call model request overrides carried in the payload."""

    TOOL_RESULTS = "tool_results"
    """``ToolResult`` list stored by ``ToolCallNode``."""
