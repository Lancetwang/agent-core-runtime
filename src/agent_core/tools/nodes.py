from typing import Any

from agent_core.core import ExecResult, Node
from agent_core.core.context import get_current_context
from agent_core.tools.executor import ToolExecutor


class ToolCallNode(Node):
    """Execute the pending tool calls and append their results as tool messages.

    Reads the assistant message from ``state[assistant_key]``, runs every tool
    call through the executor, stores results under ``state[results_key]``,
    appends ``role: tool`` messages to both ``state[messages_key]`` and the
    active context, and emits ``tool.call`` / ``tool.result`` events.
    """

    def __init__(
        self,
        *,
        executor: ToolExecutor | None = None,
        assistant_key: str = "assistant_message",
        messages_key: str = "history",
        results_key: str = "tool_results",
        next_action: str = "chat",
    ) -> None:
        super().__init__()
        self.executor = executor or ToolExecutor()
        self.assistant_key = assistant_key
        self.messages_key = messages_key
        self.results_key = results_key
        self.next_action = next_action

    def exec(self, payload: Any) -> ExecResult:
        state: dict[str, Any] = dict(payload or {})
        assistant_message = state.get(self.assistant_key, {})
        tool_calls = self.executor.parse_tool_calls(assistant_message)
        context = get_current_context()
        # Announce every call first so hosts see the batch the model asked
        # for; results are then emitted in the same order, whether the calls
        # ran concurrently or one after another.
        for tool_call in tool_calls:
            if context is not None:
                context.emit(
                    "tool.call",
                    category="tool",
                    data={
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                )
        results = self.executor.execute_all(tool_calls)
        for result in results:
            if context is not None:
                context.emit(
                    "tool.result",
                    category="tool",
                    data={
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                        "elapsed_ms": result.elapsed_ms,
                    },
                )

        state[self.results_key] = results
        messages = state.setdefault(self.messages_key, [])
        for result in results:
            message = result.to_message()
            messages.append(message)
            if context is not None:
                context.add_message(
                    "tool",
                    result.content,
                    tool_call_id=result.tool_call_id,
                )

        return self.next_action, state
