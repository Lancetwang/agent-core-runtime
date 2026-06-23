from typing import Any

from agent_core.core import ExecResult, Node, Payload
from agent_core.core.context import get_current_context
from agent_core.core.trace import get_trace_recorder
from agent_core.tools.executor import ToolExecutor


class ToolCallNode(Node):
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

    def exec(self, payload: Payload) -> ExecResult:
        state: dict[str, Any] = dict(payload or {})
        assistant_message = state.get(self.assistant_key, {})
        tool_calls = self.executor.parse_tool_calls(assistant_message)
        recorder = get_trace_recorder(state)
        context = get_current_context()
        results = []
        for tool_call in tool_calls:
            if recorder is not None:
                recorder.emit(
                    "tool.call",
                    category="tool",
                    data={
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                )
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
            result = self.executor.execute(tool_call)
            results.append(result)
            if recorder is not None:
                recorder.emit(
                    "tool.result",
                    category="tool",
                    data={
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    },
                )
            if context is not None:
                context.emit(
                    "tool.result",
                    category="tool",
                    data={
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                        "is_error": result.is_error,
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
                    is_error=result.is_error,
                )

        return self.next_action, state
