from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from agent_core.core.node import _FLOW_CANCEL, FlowCancelled
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

_CURRENT_TOOL_PROGRESS: contextvars.ContextVar[Callable[[str], None] | None] = (
    contextvars.ContextVar(
        "agent_core_current_tool_progress",
        default=None,
    )
)


def get_current_tool_call() -> ToolCall | None:
    """Return the tool call executing in this thread or async context."""
    return _CURRENT_TOOL_CALL.get()


def report_tool_progress(content: Any) -> bool:
    """Publish transient progress from the currently executing tool.

    Returns ``True`` when an executor supplied a progress subscriber and
    ``False`` when the tool is running without one. Progress is deliberately
    live-only; callers that need persistence should store an artifact.
    """
    callback = _CURRENT_TOOL_PROGRESS.get()
    if callback is None:
        return False
    try:
        serialized = _stringify_result(content)
    except (TypeError, ValueError):
        serialized = str(content)
    callback(serialized)
    return True


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
        if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
            raise ValueError(f"max_workers must be a positive integer, got {max_workers!r}.")
        self.max_workers = max_workers

    def parse_tool_calls(self, assistant_message: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from an assistant message; empty list when there are none."""
        openai_calls = assistant_message.get("tool_calls")
        if not isinstance(openai_calls, list):
            return []
        calls: list[ToolCall] = []
        seen: set[str] = set()
        for index, item in enumerate(openai_calls):
            if not isinstance(item, dict):
                continue
            call = ToolCall.from_openai_item(item)
            call_id = call.id
            if not call_id or call_id in seen:
                call_id = _unique_tool_call_id(call_id, index, seen)
                item["id"] = call_id
                call = ToolCall(id=call_id, name=call.name, arguments=call.arguments)
            seen.add(call_id)
            calls.append(call)
        return calls

    def execute(
        self,
        tool_call: ToolCall,
        *,
        on_progress: Callable[[ToolCall, str], None] | None = None,
    ) -> ToolResult:
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
        progress_token = _CURRENT_TOOL_PROGRESS.set(
            (lambda content: on_progress(tool_call, content)) if on_progress is not None else None
        )
        try:
            try:
                result = tool.execute(**tool_call.arguments)
                if inspect.isawaitable(result):
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        result = asyncio.run(_await_value(result))
                    else:
                        if inspect.iscoroutine(result):
                            result.close()
                        raise RuntimeError(
                            "Async tool called through ToolExecutor.execute(); "
                            "use await ToolExecutor.aexecute() instead."
                        )
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
            _CURRENT_TOOL_PROGRESS.reset(progress_token)
            _CURRENT_TOOL_CALL.reset(token)

    async def aexecute(
        self,
        tool_call: ToolCall,
        *,
        on_progress: Callable[[ToolCall, str], None] | None = None,
    ) -> ToolResult:
        """Execute one sync or async tool without blocking the event loop."""
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
        progress_token = _CURRENT_TOOL_PROGRESS.set(
            (lambda content: on_progress(tool_call, content)) if on_progress is not None else None
        )
        try:
            try:
                if inspect.iscoroutinefunction(tool.fn):
                    result = tool.execute(**tool_call.arguments)
                else:
                    result = await asyncio.to_thread(tool.execute, **tool_call.arguments)
                if inspect.isawaitable(result):
                    result = await result
                content = _stringify_result(result)
            except asyncio.CancelledError:
                raise
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
            _CURRENT_TOOL_PROGRESS.reset(progress_token)
            _CURRENT_TOOL_CALL.reset(token)

    def execute_all(
        self,
        tool_calls: Sequence[ToolCall],
        *,
        cancel: threading.Event | None = None,
        on_progress: Callable[[ToolCall, str], None] | None = None,
    ) -> list[ToolResult]:
        """Run every tool call and collect the results in the original order.

        Consecutive calls to tools marked ``parallel`` run concurrently on a
        bounded thread pool. A serial call is an exclusive barrier between
        parallel batches. Results preserve ``tool_calls`` order either way.

        ``cancel`` is checked cooperatively between calls (and before each
        parallel batch); when set, a :class:`FlowCancelled` is raised. Inside
        a flow run the enclosing run's cancel event applies automatically.
        """
        if not tool_calls:
            return []
        cancel_event = cancel if cancel is not None else _FLOW_CANCEL.get()

        def check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise FlowCancelled("Tool execution cancelled.")

        results: list[ToolResult | None] = [None] * len(tool_calls)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            batch: list[tuple[int, ToolCall]] = []

            def flush_parallel_batch() -> None:
                check_cancelled()
                futures: dict[int, Future[ToolResult]] = {
                    index: pool.submit(
                        contextvars.copy_context().run,
                        self.execute,
                        call,
                        on_progress=on_progress,
                    )
                    for index, call in batch
                }
                for index, future in futures.items():
                    results[index] = future.result()
                batch.clear()

            for index, call in enumerate(tool_calls):
                tool = self.tool_map.get(call.name)
                if tool is not None and tool.parallel:
                    batch.append((index, call))
                else:
                    flush_parallel_batch()
                    check_cancelled()
                    results[index] = self.execute(call, on_progress=on_progress)
            flush_parallel_batch()

        return [result for result in results if result is not None]

    async def aexecute_all(
        self,
        tool_calls: Sequence[ToolCall],
        *,
        cancel: threading.Event | asyncio.Event | None = None,
        on_progress: Callable[[ToolCall, str], None] | None = None,
    ) -> list[ToolResult]:
        """Async batch execution with the same parallel/serial barriers."""
        if not tool_calls:
            return []
        cancel_event = cancel if cancel is not None else _FLOW_CANCEL.get()

        def check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise FlowCancelled("Tool execution cancelled.")

        results: list[ToolResult | None] = [None] * len(tool_calls)
        batch: list[tuple[int, ToolCall]] = []

        async def flush_parallel_batch() -> None:
            check_cancelled()
            for start in range(0, len(batch), self.max_workers):
                current = batch[start : start + self.max_workers]
                values = await asyncio.gather(
                    *(self.aexecute(call, on_progress=on_progress) for _, call in current)
                )
                for (index, _), result in zip(current, values, strict=True):
                    results[index] = result
            batch.clear()

        for index, call in enumerate(tool_calls):
            tool = self.tool_map.get(call.name)
            if tool is not None and tool.parallel:
                batch.append((index, call))
            else:
                await flush_parallel_batch()
                check_cancelled()
                results[index] = await self.aexecute(call, on_progress=on_progress)
        await flush_parallel_batch()
        return [result for result in results if result is not None]


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _stringify_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


async def _await_value(value: Awaitable[Any]) -> Any:
    return await value


def _unique_tool_call_id(base: str, index: int, seen: set[str]) -> str:
    candidate = f"{base}_{index}" if base else f"call_{index}"
    while candidate in seen:
        candidate += "x"
    return candidate
