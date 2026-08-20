from typing import Any

from agent_core.core import ExecResult, Node, PayloadKeys
from agent_core.core.context import get_current_context
from agent_core.tools.executor import ToolExecutor


class ToolCallNode(Node):
    """Execute the pending tool calls and append their results as tool messages.

    Reads the assistant message from ``state[assistant_key]``, runs every tool
    call through the executor, stores results under ``state[results_key]``,
    appends ``role: tool`` messages to the active context scope (the
    canonical history; or to ``state[messages_key]`` when no context is
    active), and emits ``tool.call`` / ``tool.result`` events. Set
    ``retain_event_payloads=False`` to retain metadata only and send full
    arguments/results through the observation channel instead.
    """

    def __init__(
        self,
        *,
        executor: ToolExecutor | None = None,
        assistant_key: str = PayloadKeys.ASSISTANT_MESSAGE,
        messages_key: str = PayloadKeys.HISTORY,
        results_key: str = PayloadKeys.TOOL_RESULTS,
        next_action: str = "chat",
        retain_event_payloads: bool = True,
    ) -> None:
        super().__init__()
        self.executor = executor if executor is not None else ToolExecutor()
        self.assistant_key = assistant_key
        self.messages_key = messages_key
        self.results_key = results_key
        self.next_action = next_action
        self.retain_event_payloads = retain_event_payloads

    def exec(self, payload: Any) -> ExecResult:
        state: dict[str, Any] = dict(payload or {})
        assistant_message = state.get(self.assistant_key, {})
        tool_calls = self.executor.parse_tool_calls(assistant_message)
        context = get_current_context()
        # Announce every call first so hosts see the batch the model asked
        # for; results are then emitted in the same order, whether the calls
        # ran concurrently or one after another. The compatibility default
        # retains full payloads; the built-in Agent loop opts into metadata-
        # only tool events to avoid duplicating large or sensitive content.
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
        results = self.executor.execute_all(tool_calls)
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
