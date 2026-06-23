from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_core.core import ExecResult, Flow, Node, Payload
from agent_core.core.context import get_current_context
from agent_core.models import ChatModel, Message
from agent_core.tools import Tool, ToolCallNode, ToolExecutor

MessageBuilder = Callable[[Payload], Sequence[Message]]
ToolSpec = Tool | Mapping[str, Any]
ToolProvider = Callable[[Payload], Sequence[ToolSpec]] | Sequence[ToolSpec]


class ModelNode(Node):
    """Call a chat model and store the returned assistant message."""

    def __init__(
        self,
        *,
        model: ChatModel,
        messages: MessageBuilder | None = None,
        tools: ToolProvider | None = None,
        assistant_key: str = "assistant_message",
        messages_key: str = "history",
        action: str = "default",
        chat_kwargs: Mapping[str, Any] | None = None,
        chat_kwargs_key: str = "chat_kwargs",
        append_message: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.messages = messages
        self.tools = tools
        self.assistant_key = assistant_key
        self.messages_key = messages_key
        self.action = action
        self.chat_kwargs = dict(chat_kwargs or {})
        self.chat_kwargs_key = chat_kwargs_key
        self.append_message = append_message

    def exec(self, payload: Payload) -> ExecResult:
        state: dict[str, Any] = dict(payload or {})
        context = get_current_context()
        messages = self._build_messages(state)
        tools = self._build_tools(state)
        chat_kwargs = dict(self.chat_kwargs)
        chat_kwargs.update(state.get(self.chat_kwargs_key, {}))

        if context is not None:
            context.emit(
                "model.request",
                category="model",
                data={
                    "message_count": len(messages),
                    "tool_count": len(tools),
                    "tool_names": _tool_names(tools),
                },
            )

        assistant_message = self.model.chat_message(
            messages,
            tools=tools or None,
            **chat_kwargs,
        )
        state[self.assistant_key] = assistant_message
        if self.append_message:
            state.setdefault(self.messages_key, []).append(assistant_message)
            if context is not None:
                context.add_message(
                    "assistant",
                    str(assistant_message.get("content", "")),
                    tool_calls=assistant_message.get("tool_calls", []),
                )

        if context is not None:
            context.emit(
                "model.response",
                category="model",
                data={
                    "has_tool_calls": bool(assistant_message.get("tool_calls")),
                    "content_length": len(str(assistant_message.get("content", ""))),
                    "usage": assistant_message.get("usage", {}),
                },
            )

        return self.action, state

    def _build_messages(self, state: dict[str, Any]) -> list[Message]:
        if self.messages is not None:
            return list(self.messages(state))
        context = get_current_context()
        if context is not None and context.messages:
            return list(context.messages)
        return list(state.get(self.messages_key, []))

    def _build_tools(self, state: dict[str, Any]) -> list[Mapping[str, Any]]:
        if self.tools is None:
            return []
        tools = self.tools(state) if callable(self.tools) else self.tools
        return [_to_tool_schema(tool) for tool in tools]


class ToolRouterNode(Node):
    """Route an assistant message to tool execution or final output."""

    def __init__(
        self,
        *,
        assistant_key: str = "assistant_message",
        output_key: str = "answer",
        tool_action: str = "tool_call",
        done_action: str = "final",
    ) -> None:
        super().__init__()
        self.assistant_key = assistant_key
        self.output_key = output_key
        self.tool_action = tool_action
        self.done_action = done_action

    def exec(self, payload: Payload) -> ExecResult:
        state: dict[str, Any] = dict(payload or {})
        assistant_message = state.get(self.assistant_key, {})
        tool_calls = (
            assistant_message.get("tool_calls")
            if isinstance(assistant_message, dict)
            else None
        )
        context = get_current_context()

        if tool_calls:
            if context is not None:
                context.emit(
                    "tool.route",
                    category="tool",
                    action=self.tool_action,
                    data={"tool_call_count": len(tool_calls)},
                )
            return self.tool_action, state

        if isinstance(assistant_message, dict):
            state[self.output_key] = assistant_message.get("content", "")
        if context is not None:
            context.emit(
                "tool.route",
                category="tool",
                action=self.done_action,
                data={"tool_call_count": 0},
            )
        return self.done_action, state


def build_tool_agent_flow(
    *,
    model: ChatModel,
    messages: MessageBuilder | None = None,
    tools: Sequence[Tool],
    chat_kwargs: Mapping[str, Any] | None = None,
    assistant_key: str = "assistant_message",
    messages_key: str = "history",
    output_key: str = "answer",
) -> Flow:
    model_node = ModelNode(
        model=model,
        messages=messages,
        tools=tools,
        assistant_key=assistant_key,
        messages_key=messages_key,
        action="route",
        chat_kwargs=chat_kwargs,
    )
    router_node = ToolRouterNode(
        assistant_key=assistant_key,
        output_key=output_key,
        tool_action="tool_call",
        done_action="final",
    )
    tool_node = ToolCallNode(
        executor=ToolExecutor(tools),
        assistant_key=assistant_key,
        messages_key=messages_key,
        next_action="chat",
    )

    model_node - "route" >> router_node
    router_node - "tool_call" >> tool_node
    tool_node - "chat" >> model_node

    return Flow(model_node)


def _to_tool_schema(tool: ToolSpec) -> Mapping[str, Any]:
    if isinstance(tool, Tool):
        return tool.to_llm_format()
    return tool


def _tool_names(tools: Sequence[Mapping[str, Any]]) -> list[str]:
    names = []
    for tool in tools:
        function = tool.get("function", {})
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str):
                names.append(name)
    return names
