from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_core.core import ExecResult, Flow, Node
from agent_core.core.context import get_current_context
from agent_core.llm.client import ChatModel, LLM, Message
from agent_core.tools import Tool, ToolCallNode, ToolExecutor

MessageBuilder = Callable[[Any], Sequence[Message]]
ToolSpec = Tool | Mapping[str, Any]
ToolProvider = Callable[[Any], Sequence[ToolSpec]] | Sequence[ToolSpec]


class ModelNode(Node):
    def __init__(
        self,
        *,
        model: ChatModel | None = None,
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
        self.model = model or LLM()
        self.messages = messages
        self.tools = tools
        self.assistant_key = assistant_key
        self.messages_key = messages_key
        self.action = action
        self.chat_kwargs = dict(chat_kwargs or {})
        self.chat_kwargs_key = chat_kwargs_key
        self.append_message = append_message

    def exec(self, payload: Any) -> ExecResult:
        state = dict(payload or {})
        context = get_current_context()
        messages = self._messages(state)
        tools = self._tools(state)
        chat_kwargs = self._chat_kwargs(state)

        if context:
            context.emit(
                "model.request",
                category="model",
                data={"message_count": len(messages), "tool_names": _tool_names(tools)},
            )

        message = self.model.chat_message(messages, tools=tools or None, **chat_kwargs)
        state[self.assistant_key] = message
        if self.append_message:
            state.setdefault(self.messages_key, []).append(message)
            if context:
                extra = {"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {}
                context.add_message("assistant", str(message.get("content", "")), **extra)

        if context:
            context.emit(
                "model.response",
                category="model",
                data={
                    "has_tool_calls": bool(message.get("tool_calls")),
                    "content_length": len(str(message.get("content", ""))),
                    "usage": message.get("usage", {}),
                },
            )
        return self.action, state

    def _messages(self, state: dict[str, Any]) -> list[Message]:
        if self.messages:
            return list(self.messages(state))
        context = get_current_context()
        if context:
            scoped_messages = context.get_messages()
            if scoped_messages:
                return list(scoped_messages)
        return list(state.get(self.messages_key, []))

    def _tools(self, state: dict[str, Any]) -> list[Mapping[str, Any]]:
        if self.tools is None:
            return []
        tools = self.tools(state) if callable(self.tools) else self.tools
        return [tool.to_llm_format() if isinstance(tool, Tool) else tool for tool in tools]

    def _chat_kwargs(self, state: dict[str, Any]) -> dict[str, Any]:
        context = get_current_context()
        kwargs = {**self.chat_kwargs, **state.get(self.chat_kwargs_key, {})}
        on_delta = kwargs.pop("on_delta", None)
        if context:
            kwargs["on_delta"] = _delta_callback(context, on_delta)
        elif on_delta:
            kwargs["on_delta"] = on_delta
        return kwargs


class ToolRouterNode(Node):
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

    def exec(self, payload: Any) -> ExecResult:
        state = dict(payload or {})
        message = state.get(self.assistant_key, {})
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        context = get_current_context()
        action = self.tool_action if tool_calls else self.done_action

        if isinstance(message, dict) and not tool_calls:
            state[self.output_key] = message.get("content", "")
        if context:
            context.emit(
                "tool.observe",
                category="tool",
                action=action,
                data={"tool_call_count": len(tool_calls or [])},
            )
        return action, state


def _minimal_agent_loop(
    *,
    model: ChatModel | None = None,
    messages: MessageBuilder | None = None,
    tools: Sequence[Tool],
    chat_kwargs: Mapping[str, Any] | None = None,
    assistant_key: str = "assistant_message",
    messages_key: str = "history",
    output_key: str = "answer",
) -> Flow:
    chat_kwargs = {"stream": True, **dict(chat_kwargs or {})}
    model_node = ModelNode(
        model=model,
        messages=messages,
        tools=tools,
        assistant_key=assistant_key,
        messages_key=messages_key,
        action="observe",
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

    model_node - "observe" >> router_node
    router_node - "tool_call" >> tool_node
    tool_node - "chat" >> model_node
    return Flow(model_node)


def _delta_callback(context: Any, callback: Callable[[str], None] | None) -> Callable[[str], None]:
    def on_delta(text: str) -> None:
        if text:
            context.emit("model.delta", category="model", data={"content": text})
            if callback:
                callback(text)

    return on_delta


def _tool_names(tools: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function", {})
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if isinstance(name, str):
            names.append(name)
    return names
