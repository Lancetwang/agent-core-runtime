from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_core.tools.base import Tool


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai_item(cls, item: dict[str, Any]) -> ToolCall:
        """Parse one OpenAI-style ``tool_calls`` entry.

        Parsing is deliberately lenient: models sometimes emit malformed
        argument JSON, and a hard failure here would kill the whole run.
        Unparseable arguments become ``{}`` so the tool itself can report a
        precise validation error back to the model.
        """
        function = item.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = _safe_json_loads(arguments)
        if not isinstance(arguments, dict):
            arguments = {}

        return cls(
            id=item.get("id", ""),
            name=function.get("name", ""),
            arguments=arguments,
        )


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool execution, ready to feed back to the model."""

    tool_call_id: str
    content: str
    is_error: bool = False

    def to_message(self) -> dict[str, str]:
        """Return the OpenAI-style ``role: tool`` message for this result."""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


class ToolExecutor:
    """Execute model tool calls against a fixed set of tools.

    Execution never raises for model-caused failures: unknown tools and tool
    exceptions come back as :class:`ToolResult` with ``is_error=True`` so the
    model can read the error text and correct itself.
    """

    def __init__(self, tools: Sequence[Tool] | None = None) -> None:
        self.tools = list(tools or [])
        names = [tool.name for tool in self.tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate tool names: {', '.join(duplicates)}")
        self.tool_map = {tool.name: tool for tool in self.tools}

    def parse_tool_calls(self, assistant_message: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from an assistant message; empty list when there are none."""
        openai_calls = assistant_message.get("tool_calls")
        if not isinstance(openai_calls, list):
            return []
        return [ToolCall.from_openai_item(item) for item in openai_calls if isinstance(item, dict)]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Run one tool call and return its result or a readable error."""
        tool = self.tool_map.get(tool_call.name)
        if tool is None:
            known = ", ".join(sorted(self.tool_map)) or "none"
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Tool '{tool_call.name}' not found. Available tools: {known}.",
                is_error=True,
            )

        try:
            result = tool.execute(**tool_call.arguments)
            content = _stringify_result(result)
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Tool '{tool_call.name}' failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            content=content,
            is_error=False,
        )

    def execute_all(self, tool_calls: Sequence[ToolCall]) -> list[ToolResult]:
        """Run every tool call in order and collect the results."""
        return [self.execute(tool_call) for tool_call in tool_calls]


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _stringify_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
