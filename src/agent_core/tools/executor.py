from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_core.tools.base import Tool


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai_item(cls, item: dict[str, Any]) -> ToolCall:
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
    tool_call_id: str
    content: str
    is_error: bool = False

    def to_message(self) -> dict[str, str]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


class ToolExecutor:
    def __init__(self, tools: Sequence[Tool] | None = None) -> None:
        self.tools = list(tools or [])
        self.tool_map = {tool.name: tool for tool in self.tools}

    def parse_tool_calls(self, assistant_message: dict[str, Any]) -> list[ToolCall]:
        openai_calls = assistant_message.get("tool_calls")
        if not isinstance(openai_calls, list):
            return []
        return [ToolCall.from_openai_item(item) for item in openai_calls]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        tool = self.tool_map.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Tool '{tool_call.name}' not found",
                is_error=True,
            )

        try:
            result = tool.execute(**tool_call.arguments)
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: {exc}",
                is_error=True,
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            content=_stringify_result(result),
            is_error=False,
        )

    def execute_all(self, tool_calls: Sequence[ToolCall]) -> list[ToolResult]:
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
