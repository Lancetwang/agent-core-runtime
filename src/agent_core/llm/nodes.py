from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from agent_core.core import ExecResult, Flow, Node, PayloadKeys
from agent_core.core.context import get_current_context
from agent_core.llm.client import LLM, AsyncChatModel, ChatModel, Message
from agent_core.tools import Tool, ToolCall, ToolCallNode, ToolExecutor, ToolResult

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
        model: ChatModel | AsyncChatModel | None = None,
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
        self.model: ChatModel | AsyncChatModel | None = model
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
        self._emit_request(context, messages, tools, chat_kwargs)
        chat_message = getattr(model, "chat_message", None)
        if not callable(chat_message):
            raise RuntimeError(
                f"{type(model).__name__} implements only AsyncChatModel; "
                "use Flow.arun() or Agent.achat()."
            )
        message = cast(
            dict[str, Any],
            chat_message(messages, tools=tools or None, **chat_kwargs),
        )
        return self._finish(state, context, message)

    async def aexec(self, payload: Any) -> ExecResult:
        state = dict(payload or {})
        context = get_current_context()
        model = self._get_model()
        messages = self._messages(state)
        tools = self._tools(state)
        chat_kwargs = self._chat_kwargs(state, model)
        self._emit_request(context, messages, tools, chat_kwargs)
        if isinstance(model, AsyncChatModel):
            message = await model.achat_message(
                messages,
                tools=tools or None,
                **chat_kwargs,
            )
        else:
            message = await asyncio.to_thread(
                model.chat_message,
                messages,
                tools=tools or None,
                **chat_kwargs,
            )
        return self._finish(state, context, message)

    def _finish(
        self,
        state: dict[str, Any],
        context: Any,
        message: dict[str, Any],
    ) -> ExecResult:
        if state.get(PayloadKeys.TOOLS_ENABLED) is False and message.get("tool_calls"):
            message = dict(message)
            message.pop("tool_calls", None)
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
        return ExecResult(self.action, state)

    @staticmethod
    def _emit_request(
        context: Any,
        messages: list[Message],
        tools: list[Mapping[str, Any]],
        chat_kwargs: Mapping[str, Any],
    ) -> None:
        if context is None:
            return
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

    def _messages(self, state: dict[str, Any]) -> list[Message]:
        context = get_current_context()
        if context is not None:
            scoped_messages = context.get_messages()
            if not scoped_messages:
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
                scoped_messages = context.get_messages()
            if self.messages is not None:
                # A custom builder remains in control of the final prompt, but
                # receives the canonical context history under messages_key.
                # This keeps assistant/tool turns visible on later loop passes
                # without restoring a second mutable history store.
                builder_state = {**state, self.messages_key: list(scoped_messages)}
                return list(self.messages(builder_state))
            return list(scoped_messages)
        if self.messages is not None:
            return list(self.messages(state))
        return list(state.get(self.messages_key, []))

    def _tools(self, state: dict[str, Any]) -> list[Mapping[str, Any]]:
        if self.tools is None:
            return []
        tools = self.tools(state) if callable(self.tools) else self.tools
        return [tool.to_llm_format() if isinstance(tool, Tool) else tool for tool in tools]

    def _chat_kwargs(
        self,
        state: dict[str, Any],
        model: ChatModel | AsyncChatModel,
    ) -> dict[str, Any]:
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

    def _get_model(self) -> ChatModel | AsyncChatModel:
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
        return ExecResult(action, state)


class ToolLoopGuardNode(Node):
    """Detect tool loops that repeat the same call with the same result.

    The guard is a regular node: wire its ``continue``, ``warn``, and ``halt``
    actions wherever the workflow needs. On warning it appends an internal
    system message; on halt it additionally sets ``TOOLS_ENABLED=False`` so a
    dynamic ``ModelNode`` tool provider can force one final text-only answer.
    Guard history lives in the flow payload, not hidden process state.
    """

    def __init__(
        self,
        *,
        assistant_key: str = PayloadKeys.ASSISTANT_MESSAGE,
        results_key: str = PayloadKeys.TOOL_RESULTS,
        messages_key: str = PayloadKeys.HISTORY,
        guard_key: str = PayloadKeys.LOOP_GUARD,
        tools_enabled_key: str = PayloadKeys.TOOLS_ENABLED,
        continue_action: str = "continue",
        warn_action: str = "warn",
        halt_action: str = "halt",
        window: int = 3,
    ) -> None:
        super().__init__()
        if window < 2:
            raise ValueError("ToolLoopGuardNode window must be at least 2.")
        self.assistant_key = assistant_key
        self.results_key = results_key
        self.messages_key = messages_key
        self.guard_key = guard_key
        self.tools_enabled_key = tools_enabled_key
        self.continue_action = continue_action
        self.warn_action = warn_action
        self.halt_action = halt_action
        self.window = window

    def exec(self, payload: Any) -> ExecResult:
        state: dict[str, Any] = dict(payload or {})
        action, reason, guard_state = self._evaluate(state)
        state[self.guard_key] = guard_state
        context = get_current_context()

        if action == self.warn_action:
            message = (
                "Exact tool calls repeated without a changed result. Do not repeat "
                "them; change approach or report the concrete blocker."
            )
            self._add_internal_message(state, context, message)
            if context is not None:
                context.emit("loop.warning", category="runtime", data={"reason": reason})
        elif action == self.halt_action:
            state[self.tools_enabled_key] = False
            message = (
                "Loop guard: exact tool calls kept returning the same result after "
                "a warning. Do not call more tools. Return the best supported answer, "
                "state unresolved items, and stop."
            )
            self._add_internal_message(state, context, message)
            if context is not None:
                context.emit("loop.guard", category="runtime", data={"reason": reason})
        return ExecResult(action, state)

    def _evaluate(self, state: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        previous = state.get(self.guard_key)
        previous = previous if isinstance(previous, Mapping) else {}
        previous_rounds = previous.get("rounds", [])
        previous_rounds = (
            previous_rounds
            if isinstance(previous_rounds, Sequence)
            and not isinstance(previous_rounds, (str, bytes))
            else []
        )
        rounds = [dict(item) for item in previous_rounds if isinstance(item, Mapping)]
        previous_warned = previous.get("warned", {})
        previous_warned = previous_warned if isinstance(previous_warned, Mapping) else {}
        warned = {str(key): str(value) for key, value in previous_warned.items()}
        current = _tool_round(state.get(self.assistant_key), state.get(self.results_key))
        if not current:
            return self.continue_action, "", {"rounds": [], "warned": {}}

        rounds.append(current)
        rounds = rounds[-self.window :]
        if any(
            item.get("result") and warned.get(signature) == item.get("result")
            for signature, item in current.items()
        ):
            return (
                self.halt_action,
                "exact tool calls repeated after a no-progress warning",
                {"rounds": rounds, "warned": warned},
            )

        stalled = {
            signature: item
            for signature, item in current.items()
            if int(item.get("count", 0)) >= self.window
        }
        if len(rounds) == self.window:
            for signature, item in current.items():
                outcome = item.get("result")
                if outcome and all(
                    isinstance(round_item.get(signature), Mapping)
                    and round_item[signature].get("result") == outcome
                    for round_item in rounds
                ):
                    stalled[signature] = item
        if not stalled:
            return self.continue_action, "", {"rounds": rounds, "warned": {}}
        next_warned = {
            signature: str(item.get("result", "")) for signature, item in stalled.items()
        }
        return (
            self.warn_action,
            "exact tool calls repeated without a changed result",
            {"rounds": rounds, "warned": next_warned},
        )

    def _add_internal_message(self, state: dict[str, Any], context: Any, content: str) -> None:
        if context is not None:
            context.add_message("system", content, agent_internal=True)
            return
        messages = list(state.get(self.messages_key, []))
        messages.append({"role": "system", "content": content, "agent_internal": True})
        state[self.messages_key] = messages


def _minimal_agent_loop(
    *,
    model: ChatModel | AsyncChatModel | None = None,
    messages: MessageBuilder | None = None,
    tools: Sequence[Tool],
    chat_kwargs: Mapping[str, Any] | None = None,
    assistant_key: str = PayloadKeys.ASSISTANT_MESSAGE,
    messages_key: str = PayloadKeys.HISTORY,
    output_key: str = PayloadKeys.ANSWER,
    loop_guard: bool = True,
) -> Flow:
    """Build the standard model -> router -> tools -> model chat loop."""
    chat_kwargs = {"stream": True, **dict(chat_kwargs or {})}
    model_node = ModelNode(
        model=model,
        messages=messages,
        tools=lambda state: (
            tools
            if not isinstance(state, Mapping) or state.get(PayloadKeys.TOOLS_ENABLED, True)
            else []
        ),
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
    if loop_guard and tools:
        guard_node = ToolLoopGuardNode(
            assistant_key=assistant_key,
            results_key=PayloadKeys.TOOL_RESULTS,
            messages_key=messages_key,
        )
        tool_node.next_action = "guard"
        tool_node - "guard" >> guard_node
        guard_node - "continue" >> model_node
        guard_node - "warn" >> model_node
        guard_node - "halt" >> model_node
    else:
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


def _tool_round(assistant_message: Any, raw_results: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(assistant_message, Mapping):
        return {}
    raw_calls = assistant_message.get("tool_calls")
    if not isinstance(raw_calls, list) or not isinstance(raw_results, list):
        return {}
    results: dict[str, tuple[str, bool]] = {}
    for result in raw_results:
        if isinstance(result, ToolResult):
            results[result.tool_call_id] = (result.content, result.is_error)
        elif isinstance(result, Mapping):
            tool_call_id = result.get("tool_call_id")
            if isinstance(tool_call_id, str):
                results[tool_call_id] = (
                    str(result.get("content", "")),
                    bool(result.get("is_error", False)),
                )

    current: dict[str, dict[str, Any]] = {}
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        call = ToolCall.from_openai_item(item)
        result = results.get(call.id)
        if result is None:
            continue
        signature = _digest({"name": call.name, "arguments": call.arguments})
        outcome = _digest({"is_error": result[1], "content": result[0]})
        prior = current.get(signature)
        current[signature] = {
            "result": outcome if prior is None or prior.get("result") == outcome else "",
            "count": int(prior.get("count", 0) if prior else 0) + 1,
        }
    return current


def _digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()
