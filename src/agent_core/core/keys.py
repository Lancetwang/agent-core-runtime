"""Canonical payload state key names shared by the built-in nodes.

Custom flows read and write the flow payload through these keys. Centralizing
them removes typo-driven silent failures: a misspelled ``answer`` key, for
example, silently yields an empty chat response instead of an error.
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
