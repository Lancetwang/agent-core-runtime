from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_core.core import (
    Action,
    ExecResult,
    Flow,
    FlowRunResult,
    Node,
    Payload,
    RunContext,
    TraceOptions,
    get_current_context,
)
from agent_core.llm import ChatModel
from agent_core.llm.nodes import _minimal_agent_loop
from agent_core.tools import Tool


class Agent(Node):
    def __init__(
        self,
        flow: Flow | None = None,
        *,
        model: ChatModel | None = None,
        instructions: str | None = None,
        tools: Sequence[Tool] | None = None,
        chat_kwargs: Mapping[str, Any] | None = None,
        action: Action | None = "default",
        max_steps: int = 100,
        max_retries: int = 1,
        wait: float = 0,
    ) -> None:
        super().__init__(max_retries=max_retries, wait=wait)
        if flow is None:
            flow = _minimal_agent_loop(
                model=model,
                tools=list(tools or []),
                chat_kwargs=chat_kwargs,
            )
        elif model is not None:
            raise ValueError("Pass either flow or model, not both.")

        self.flow = flow
        self.instructions = instructions
        self._instruction_marker = f"agent_core.instructions.{id(self)}"
        self.action = action
        self.max_steps = max_steps

    def new_context(self) -> RunContext:
        return self._prepare_context(RunContext())

    def chat(
        self,
        text: str,
        *,
        context: RunContext | None = None,
        max_steps: int = 100,
        trace: TraceOptions | bool | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        run_context = self._prepare_context(context) or RunContext()
        run_context.add_message("user", text)
        state = {"input": text, **dict(payload or {})}
        result = self.run(
            state,
            max_steps=max_steps,
            trace=trace,
            context=run_context,
        )
        return str(result.payload.get("answer", ""))

    def run(
        self,
        payload: Payload = None,
        *,
        max_steps: int = 100,
        trace: TraceOptions | bool | None = None,
        context: RunContext | None = None,
    ) -> FlowRunResult:
        context = self._prepare_context(context)
        return self.flow.run(
            payload,
            max_steps=max_steps,
            trace=trace,
            context=context,
        )

    def exec(self, payload: Payload) -> ExecResult:
        context = self._prepare_context(get_current_context())
        result = self.flow.run(
            payload,
            max_steps=self.max_steps,
            trace=None,
            context=context,
        )
        return ((result.action or "default") if self.action is None else self.action), result.payload

    def _prepare_context(self, context: RunContext | None) -> RunContext | None:
        if self.instructions is None:
            return context
        if context is None:
            context = RunContext()
        if not context.metadata.get(self._instruction_marker):
            context.add_message("system", self.instructions)
            context.metadata[self._instruction_marker] = True
        return context
