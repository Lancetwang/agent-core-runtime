from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_core.core import ExecResult, Flow, Node, PayloadKeys
from agent_core.core.context import get_current_context
from agent_core.llm.client import LLM, ChatModel, Message
from agent_core.tools import Tool, ToolCallNode, ToolExecutor

MessageBuilder = Callable[[Any], Sequence[Message]]
ToolSpec = Tool | Mapping[str, Any]
ToolProvider = Callable[[Any], Sequence[ToolSpec]] | Sequence[ToolSpec]


class ModelNode(Node):
    """Call the chat model once and store the assistant message.

    Messages come from an explicit ``messages`` builder when provided.
    Otherwise the active context scope is the single canonical history: an
    unscoped ambient conversation is adopted into the agent scope at run
    start, and an initial ``state[messages_key]`` history is imported into an
    empty scope once. The response lands in ``state[assistant_key]``;
    per-call overrides can be passed through ``state[chat_kwargs_key]``.
    Emits ``model.request`` / ``model.response`` events and records usage on
    the active context.
    """

    def __init__(
        self,
        *,
        model: ChatModel | None = None,
        messages: MessageBuilder | None = None,
        tools: ToolProvider | None = None,
        assistant_key: str = PayloadKeys.ASSISTANT_MESSAGE,
        messages_key: str = PayloadKeys.HISTORY,
        action: str = "default",
        chat_kwargs: Mapping[str, Any] | None = None,
        chat_kwargs_key: str = PayloadKeys.CHAT_KWARGS,
        append_message: bool = True,
    ) -> None:
        super().__init__()
        self.model: ChatModel | None = model
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
        model = self._get_model()
        messages = self._messages(state)
        tools = self._tools(state)
        chat_kwargs = self._chat_kwargs(state, model)

        if context:
            context.observe(
                "model.request.payload",
                category="model",
                data={
                    "messages": messages,
                    "tools": tools,
                    "chat_kwargs": {
                        key: value
                        for key, value in chat_kwargs.items()
                        if key not in {"on_delta", "on_reasoning_delta"}
                    },
                },
            )
            context.emit(
                "model.request",
                category="model",
                data={"message_count": len(messages), "tool_names": _tool_names(tools)},
            )

        message = model.chat_message(messages, tools=tools or None, **chat_kwargs)
        state[self.assistant_key] = message
        if self.append_message:
            if context is not None:
                # The active scope is the canonical history; do not mirror
                # into state[messages_key], which is only an import seed for
                # custom flows without a context.
                tool_calls = message.get("tool_calls")
                extra = {"tool_calls": tool_calls} if tool_calls else {}
                if tool_calls and message.get("reasoning_content"):
                    extra["reasoning_content"] = message["reasoning_content"]
                context.add_message("assistant", str(message.get("content", "")), **extra)
            else:
                history = list(state.get(self.messages_key, []))
                history.append(_history_copy(message))
                state[self.messages_key] = history

        if context:
            context.record_model_usage(message.get("usage"))
            context.observe(
                "model.response.payload",
                category="model",
                data={"message": message},
            )
            context.emit(
                "model.response",
                category="model",
                data={
                    "has_tool_calls": bool(message.get("tool_calls")),
                    "has_reasoning": bool(message.get("reasoning_content")),
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
            for message in state.get(self.messages_key, []):
                if not isinstance(message, Mapping) or not isinstance(message.get("role"), str):
                    continue
                context.add_message(
                    message["role"],
                    message.get("content", ""),
                    **{
                        key: value
                        for key, value in message.items()
                        if key not in {"role", "content"}
                    },
                )
            return list(context.get_messages())
        return list(state.get(self.messages_key, []))

    def _tools(self, state: dict[str, Any]) -> list[Mapping[str, Any]]:
        if self.tools is None:
            return []
        tools = self.tools(state) if callable(self.tools) else self.tools
        return [tool.to_llm_format() if isinstance(tool, Tool) else tool for tool in tools]

    def _chat_kwargs(self, state: dict[str, Any], model: ChatModel) -> dict[str, Any]:
        context = get_current_context()
        kwargs = {**self.chat_kwargs, **state.get(self.chat_kwargs_key, {})}
        on_delta = kwargs.pop("on_delta", None)
        on_reasoning_delta = kwargs.pop("on_reasoning_delta", None)
        if context:
            kwargs["on_delta"] = _delta_callback(context, on_delta)
            if on_reasoning_delta is not None or isinstance(model, LLM):
                kwargs["on_reasoning_delta"] = _reasoning_delta_callback(
                    context, on_reasoning_delta
                )
        elif on_delta:
            kwargs["on_delta"] = on_delta
        if not context and on_reasoning_delta:
            kwargs["on_reasoning_delta"] = on_reasoning_delta
        return kwargs

    def _get_model(self) -> ChatModel:
        if self.model is None:
            self.model = LLM()
        return self.model


class ToolRouterNode(Node):
    """Route on the last assistant message: tool calls pending or final answer.

    Returns ``tool_action`` when the message carries tool calls, otherwise
    stores the message content under ``state[output_key]`` and returns
    ``done_action``.
    """

    def __init__(
        self,
        *,
        assistant_key: str = PayloadKeys.ASSISTANT_MESSAGE,
        output_key: str = PayloadKeys.ANSWER,
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
    assistant_key: str = PayloadKeys.ASSISTANT_MESSAGE,
    messages_key: str = PayloadKeys.HISTORY,
    output_key: str = PayloadKeys.ANSWER,
) -> Flow:
    """Build the standard model -> router -> tools -> model chat loop."""
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
            # Live-only: one event per streamed chunk is transient UI
            # progress, not retained history. Tracing still captures it
            # through the live subscriber.
            context.notify("model.delta", category="model", data={"content": text})
            if callback:
                callback(text)

    return on_delta


def _reasoning_delta_callback(
    context: Any,
    callback: Callable[[str], None] | None,
) -> Callable[[str], None]:
    def on_delta(text: str) -> None:
        if text:
            context.notify("model.reasoning.delta", category="model", data={"content": text})
            if callback:
                callback(text)

    return on_delta


def _history_copy(message: Mapping[str, Any]) -> dict[str, Any]:
    """History copy with provider reasoning retained only for tool-call turns.

    Matches the retention rule used by the context store, so the two stores
    never diverge: reasoning is needed to continue tool-call conversations
    but is dead weight (and context cost) on final answers.
    """
    if message.get("tool_calls") or "reasoning_content" not in message:
        return dict(message)
    return {key: value for key, value in message.items() if key != "reasoning_content"}


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
