from __future__ import annotations

import contextvars
import json
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
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


_CURRENT_TOOL_CALL: contextvars.ContextVar[ToolCall | None] = contextvars.ContextVar(
    "agent_core_current_tool_call",
    default=None,
)


def get_current_tool_call() -> ToolCall | None:
    """Return the tool call executing in this thread or async context."""
    return _CURRENT_TOOL_CALL.get()


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool execution, ready to feed back to the model."""

    tool_call_id: str
    content: str
    is_error: bool = False
    #: Wall-clock execution time in milliseconds, measured inside ``execute``.
    elapsed_ms: float | None = None

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

    def __init__(
        self,
        tools: Sequence[Tool] | None = None,
        *,
        max_workers: int = 4,
    ) -> None:
        self.tools = list(tools or [])
        names = [tool.name for tool in self.tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate tool names: {', '.join(duplicates)}")
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.max_workers = max_workers

    def parse_tool_calls(self, assistant_message: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from an assistant message; empty list when there are none."""
        openai_calls = assistant_message.get("tool_calls")
        if not isinstance(openai_calls, list):
            return []
        return [ToolCall.from_openai_item(item) for item in openai_calls if isinstance(item, dict)]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Run one tool call and return its result or a readable error."""
        started = time.monotonic()
        tool = self.tool_map.get(tool_call.name)
        if tool is None:
            known = ", ".join(sorted(self.tool_map)) or "none"
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Tool '{tool_call.name}' not found. Available tools: {known}.",
                is_error=True,
                elapsed_ms=(time.monotonic() - started) * 1000,
            )

        token = _CURRENT_TOOL_CALL.set(tool_call)
        try:
            try:
                result = tool.execute(**tool_call.arguments)
                content = _stringify_result(result)
            except Exception as exc:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"Tool '{tool_call.name}' failed: {type(exc).__name__}: {exc}",
                    is_error=True,
                    elapsed_ms=(time.monotonic() - started) * 1000,
                )

            return ToolResult(
                tool_call_id=tool_call.id,
                content=content,
                is_error=False,
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
        finally:
            _CURRENT_TOOL_CALL.reset(token)

    def execute_all(self, tool_calls: Sequence[ToolCall]) -> list[ToolResult]:
        """Run every tool call and collect the results in the original order.

        Calls to tools marked ``parallel`` run concurrently on a bounded thread
        pool; everything else runs one at a time in declaration order. Results
        are returned in ``tool_calls`` order either way, so hosts can append
        them to the conversation without re-sorting.
        """
        if not tool_calls:
            return []
        futures: dict[int, Future[ToolResult]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for index, call in enumerate(tool_calls):
                tool = self.tool_map.get(call.name)
                if tool is not None and tool.parallel:
                    # Worker threads do not inherit contextvars, so each call
                    # runs inside a snapshot of the submitting context: tools
                    # that read get_current_context() keep working concurrently.
                    futures[index] = pool.submit(contextvars.copy_context().run, self.execute, call)
            results: list[ToolResult] = []
            for index, call in enumerate(tool_calls):
                future = futures.get(index)
                if future is not None:
                    results.append(future.result())
                else:
                    results.append(self.execute(call))
        return results


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _stringify_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
