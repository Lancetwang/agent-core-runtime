from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_core.core import (
    Action,
    ExecResult,
    Flow,
    FlowRunResult,
    Node,
    RunContext,
    TraceOptions,
    get_current_context,
)
from agent_core.llm import ChatModel
from agent_core.llm.nodes import _minimal_agent_loop
from agent_core.tools import Tool

_INHERIT_ACTION = object()


class Agent(Node):
    def __init__(
        self,
        flow: Flow | None = None,
        *,
        model: ChatModel | None = None,
        instructions: str | None = None,
        tools: Sequence[Tool] | None = None,
        chat_kwargs: Mapping[str, Any] | None = None,
        stream: bool = True,
        action: Action | object = _INHERIT_ACTION,
        max_steps: int = 100,
        max_retries: int = 1,
        wait: float = 0,
    ) -> None:
        super().__init__(max_retries=max_retries, wait=wait)
        if flow is None:
            flow = _minimal_agent_loop(
                model=model,
                tools=list(tools or []),
                chat_kwargs={"stream": stream, **dict(chat_kwargs or {})},
            )
        elif model is not None:
            raise ValueError("Pass either flow or model, not both.")

        self.flow = flow
        self.instructions = instructions
        self._message_scope = f"agent:{id(self)}"
        self._instruction_marker = f"agent_core.instructions.{self._message_scope}"
        if action is None or (action is not _INHERIT_ACTION and not isinstance(action, str)):
            raise TypeError("action must be a string when provided.")
        self.action = action
        self.max_steps = max_steps

    def new_context(self) -> RunContext:
        context = RunContext(active_message_scope=self._message_scope)
        return self._prepare_context(context)

    def chat(
        self,
        text: str,
        *,
        context: RunContext | None = None,
        max_steps: int = 100,
        trace: TraceOptions | bool | None = None,
        stream: bool | None = None,
        on_delta: Any = None,
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        run_context = self._prepare_context(context) or RunContext()
        run_context.add_message("user", text, scope=self._message_scope)
        state = {"input": text, **dict(payload or {})}
        chat_kwargs = dict(state.get("chat_kwargs", {}) or {})
        if stream is not None:
            chat_kwargs["stream"] = stream
        if on_delta is not None:
            chat_kwargs["on_delta"] = on_delta
        if chat_kwargs:
            state["chat_kwargs"] = chat_kwargs
        result = self.run(
            state,
            max_steps=max_steps,
            trace=trace,
            context=run_context,
        )
        return str(result.payload.get("answer", ""))

    def run(
        self,
        payload: Any = None,
        *,
        max_steps: int = 100,
        trace: TraceOptions | bool | None = None,
        context: RunContext | None = None,
    ) -> FlowRunResult:
        context = self._prepare_context(context) or RunContext()
        with context.use_message_scope(self._message_scope):
            return self.flow.run(
                payload,
                max_steps=max_steps,
                trace=trace,
                context=context,
            )

    def exec(self, payload: Any) -> ExecResult:
        context = self._prepare_context(get_current_context()) or RunContext()
        with context.use_message_scope(self._message_scope):
            result = self.flow.run(
                payload,
                max_steps=self.max_steps,
                trace=None,
                context=context,
            )
        action = (result.action or "default") if self.action is _INHERIT_ACTION else self.action
        return action, result.payload

    def _prepare_context(self, context: RunContext | None) -> RunContext | None:
        if self.instructions is None:
            return context
        if context is None:
            context = RunContext()
        if not context.metadata.get(self._instruction_marker):
            context.add_message("system", self.instructions, scope=self._message_scope)
            context.metadata[self._instruction_marker] = True
        return context
