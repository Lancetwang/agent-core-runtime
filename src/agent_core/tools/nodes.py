from collections.abc import Callable
from typing import Any

from agent_core.core import ExecResult, Node, PayloadKeys
from agent_core.core.context import get_current_context
from agent_core.tools.executor import ToolCall, ToolExecutor, ToolResult


class ToolCallNode(Node):
    """Execute the pending tool calls and append their results as tool messages.

    Reads the assistant message from ``state[assistant_key]``, runs every tool
    call through the executor, stores results under ``state[results_key]``,
    appends ``role: tool`` messages to the active context scope (the
    canonical history; or to ``state[messages_key]`` when no context is
    active), and emits ``tool.call`` / ``tool.result`` events. By default it
    retains metadata only and sends full arguments/results through the
    observation channel; ``retain_event_payloads=True`` is the explicit
    compatibility opt-in for retaining full event payloads.
    """

    def __init__(
        self,
        *,
        executor: ToolExecutor | None = None,
        assistant_key: str = PayloadKeys.ASSISTANT_MESSAGE,
        messages_key: str = PayloadKeys.HISTORY,
        results_key: str = PayloadKeys.TOOL_RESULTS,
        next_action: str = "chat",
        retain_event_payloads: bool = False,
    ) -> None:
        super().__init__()
        self.executor = executor if executor is not None else ToolExecutor()
        self.assistant_key = assistant_key
        self.messages_key = messages_key
        self.results_key = results_key
        self.next_action = next_action
        self.retain_event_payloads = retain_event_payloads

    def exec(self, payload: Any) -> ExecResult:
        state, tool_calls, context = self._prepare(payload)
        self._emit_calls(context, tool_calls)
        results = self.executor.execute_all(
            tool_calls,
            on_progress=self._progress_callback(context),
        )
        return self._complete(state, context, tool_calls, results)

    async def aexec(self, payload: Any) -> ExecResult:
        state, tool_calls, context = self._prepare(payload)
        self._emit_calls(context, tool_calls)
        results = await self.executor.aexecute_all(
            tool_calls,
            on_progress=self._progress_callback(context),
        )
        return self._complete(state, context, tool_calls, results)

    def _prepare(self, payload: Any) -> tuple[dict[str, Any], list[ToolCall], Any]:
        state: dict[str, Any] = dict(payload or {})
        assistant_message = state.get(self.assistant_key, {})
        tool_calls = self.executor.parse_tool_calls(assistant_message)
        context = get_current_context()
        return state, tool_calls, context

    def _emit_calls(self, context: Any, tool_calls: list[ToolCall]) -> None:
        # Announce every call first so hosts see the batch the model asked
        # for; results are then emitted in the same order, whether the calls
        # ran concurrently or one after another. Metadata-only retention is
        # the safe default; full payload events remain an explicit opt-in.
        # The canonical message history still retains data required for the
        # next model request.
        for tool_call in tool_calls:
            if context is not None:
                full_data = {
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                if self.retain_event_payloads:
                    context.emit("tool.call", category="tool", data=full_data)
                else:
                    context.emit(
                        "tool.call",
                        category="tool",
                        data={
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                        },
                    )
                    context.observe(
                        "tool.call.payload",
                        category="tool",
                        data=full_data,
                    )

    def _complete(
        self,
        state: dict[str, Any],
        context: Any,
        tool_calls: list[ToolCall],
        results: list[ToolResult],
    ) -> ExecResult:
        names = {call.id: call.name for call in tool_calls}
        for result in results:
            if context is not None:
                full_data = {
                    "tool_call_id": result.tool_call_id,
                    "name": names.get(result.tool_call_id),
                    "content": result.content,
                    "is_error": result.is_error,
                    "elapsed_ms": result.elapsed_ms,
                }
                if self.retain_event_payloads:
                    context.emit("tool.result", category="tool", data=full_data)
                else:
                    context.emit(
                        "tool.result",
                        category="tool",
                        data={
                            "tool_call_id": result.tool_call_id,
                            "name": names.get(result.tool_call_id),
                            "is_error": result.is_error,
                            "elapsed_ms": result.elapsed_ms,
                            "content_length": len(result.content),
                        },
                    )
                    context.observe(
                        "tool.result.payload",
                        category="tool",
                        data=full_data,
                    )

        state[self.results_key] = results
        if context is not None:
            for result in results:
                context.add_message(
                    "tool",
                    result.content,
                    tool_call_id=result.tool_call_id,
                )
        else:
            messages = list(state.get(self.messages_key, []))
            for result in results:
                messages.append(result.to_message())
            state[self.messages_key] = messages

        return ExecResult(self.next_action, state)

    @staticmethod
    def _progress_callback(
        context: Any,
    ) -> Callable[[ToolCall, str], None] | None:
        if context is None:
            return None

        def on_progress(tool_call: ToolCall, content: str) -> None:
            context.notify(
                "tool.progress",
                category="tool",
                data={
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": content,
                },
            )

        return on_progress
