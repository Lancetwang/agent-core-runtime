from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast

from agent_core.core import (
    Action,
    ExecResult,
    Flow,
    FlowError,
    FlowRunResult,
    Node,
    PayloadKeys,
    RunContext,
    TraceOptions,
    get_current_context,
)
from agent_core.llm import AsyncChatModel, ChatModel
from agent_core.llm.nodes import _minimal_agent_loop
from agent_core.tools import Tool

_INHERIT_ACTION = object()


class Agent(Node):
    """An agent that runs a flow — and is itself a :class:`Node`.

    Two construction styles:

    - ``Agent(model=..., instructions=..., tools=[...])`` builds the standard
      model → tool → model chat loop.
    - ``Agent(Flow(...), instructions=...)`` wraps a custom flow when the loop
      needs different wiring.

    Because ``Agent`` is a node, agents compose: wire one agent's action to
    another agent and wrap the pair in an outer ``Flow``. Each agent keeps an
    isolated message scope inside the shared :class:`RunContext`, so prompts
    never leak between agents while events and usage stay unified.
    """

    def __init__(
        self,
        flow: Flow | None = None,
        *,
        model: ChatModel | AsyncChatModel | None = None,
        instructions: str | None = None,
        tools: Sequence[Tool] | None = None,
        chat_kwargs: Mapping[str, Any] | None = None,
        stream: bool = True,
        loop_guard: bool = True,
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
                loop_guard=loop_guard,
            )
        elif model is not None:
            raise ValueError(
                "Pass either flow or model, not both. A custom flow already owns its model nodes."
            )

        self.flow = flow
        self.instructions = instructions
        # A random, address-free scope name: id(self) leaked memory addresses
        # into message_scopes keys and to_dict() output and could be reused
        # after garbage collection, letting two successive agents share a scope.
        self._message_scope = f"agent:{uuid.uuid4().hex}"
        self._instruction_marker = f"agent_core.instructions.{self._message_scope}"
        if action is None or (action is not _INHERIT_ACTION and not isinstance(action, str)):
            raise TypeError(
                "action must be a string when provided; omit it to inherit "
                "the inner flow's final action."
            )
        self.action = action
        self.max_steps = max_steps

    def new_context(self) -> RunContext:
        """Create a fresh context with this agent's message scope and instructions."""
        context = RunContext(active_message_scope=self._message_scope)
        prepared = self._prepare_context(context)
        assert prepared is not None
        return prepared

    def chat(
        self,
        text: str,
        *,
        content: Any | None = None,
        context: RunContext | None = None,
        max_steps: int | None = None,
        trace: TraceOptions | bool | None = None,
        stream: bool | None = None,
        on_delta: Any = None,
        payload: Mapping[str, Any] | None = None,
        cancel: threading.Event | None = None,
    ) -> str:
        """Send one user message through the flow and return the final answer text.

        Reuse the same ``context`` across calls to hold a conversation.
        ``stream``/``on_delta`` override streaming for this call only;
        ``payload`` seeds extra business state for the flow; ``max_steps``
        overrides the agent's constructor budget for this call only.
        """
        run_context = self._prepare_context(context)
        if run_context is None:
            run_context = RunContext()
        run_context.add_message(
            "user",
            text if content is None else content,
            scope=self._message_scope,
        )
        state = {PayloadKeys.INPUT: text, **dict(payload or {})}
        chat_kwargs = dict(state.get(PayloadKeys.CHAT_KWARGS, {}) or {})
        if stream is not None:
            chat_kwargs["stream"] = stream
        if on_delta is not None:
            chat_kwargs["on_delta"] = on_delta
        if chat_kwargs:
            state[PayloadKeys.CHAT_KWARGS] = chat_kwargs
        result = self.run(
            state,
            max_steps=max_steps,
            trace=trace,
            context=run_context,
            cancel=cancel,
        )
        if not isinstance(result.payload, Mapping) or PayloadKeys.ANSWER not in result.payload:
            raise FlowError(
                f"Agent.chat expected the flow payload to contain "
                f"{PayloadKeys.ANSWER!r}. Custom chat flows must set that key; "
                "use Agent.run() when the result has a different shape."
            )
        return str(result.payload[PayloadKeys.ANSWER])

    async def achat(
        self,
        text: str,
        *,
        content: Any | None = None,
        context: RunContext | None = None,
        max_steps: int | None = None,
        trace: TraceOptions | bool | None = None,
        stream: bool | None = None,
        on_delta: Any = None,
        payload: Mapping[str, Any] | None = None,
        cancel: threading.Event | asyncio.Event | None = None,
    ) -> str:
        """Async counterpart of :meth:`chat` for async models and tools."""
        run_context = self._prepare_context(context)
        if run_context is None:
            run_context = RunContext()
        run_context.add_message(
            "user",
            text if content is None else content,
            scope=self._message_scope,
        )
        state = {PayloadKeys.INPUT: text, **dict(payload or {})}
        chat_kwargs = dict(state.get(PayloadKeys.CHAT_KWARGS, {}) or {})
        if stream is not None:
            chat_kwargs["stream"] = stream
        if on_delta is not None:
            chat_kwargs["on_delta"] = on_delta
        if chat_kwargs:
            state[PayloadKeys.CHAT_KWARGS] = chat_kwargs
        result = await self.arun(
            state,
            max_steps=max_steps,
            trace=trace,
            context=run_context,
            cancel=cancel,
        )
        if not isinstance(result.payload, Mapping) or PayloadKeys.ANSWER not in result.payload:
            raise FlowError(
                f"Agent.achat expected the flow payload to contain "
                f"{PayloadKeys.ANSWER!r}. Custom chat flows must set that key; "
                "use Agent.arun() when the result has a different shape."
            )
        return str(result.payload[PayloadKeys.ANSWER])

    def run(
        self,
        payload: Any = None,
        *,
        max_steps: int | None = None,
        trace: TraceOptions | bool | None = None,
        context: RunContext | None = None,
        cancel: threading.Event | None = None,
    ) -> FlowRunResult:
        """Run the inner flow on a payload and return the full :class:`FlowRunResult`.

        ``max_steps`` defaults to the agent's constructor budget. Inside an
        outer flow, every inner node visit is debited from the shared outer
        budget. ``cancel`` accepts a ``threading.Event`` checked cooperatively
        between steps.
        """
        if max_steps is None:
            max_steps = self.max_steps
        context = self._prepare_context(context)
        if context is None:
            context = RunContext()
        with context.use_message_scope(self._message_scope):
            self._adopt_global_messages(context)
            return self.flow.run(
                payload,
                max_steps=max_steps,
                trace=trace,
                context=context,
                cancel=cancel,
            )

    async def arun(
        self,
        payload: Any = None,
        *,
        max_steps: int | None = None,
        trace: TraceOptions | bool | None = None,
        context: RunContext | None = None,
        cancel: threading.Event | asyncio.Event | None = None,
    ) -> FlowRunResult:
        """Asynchronously run the inner node flow."""
        if max_steps is None:
            max_steps = self.max_steps
        context = self._prepare_context(context)
        if context is None:
            context = RunContext()
        with context.use_message_scope(self._message_scope):
            self._adopt_global_messages(context)
            return await self.flow.arun(
                payload,
                max_steps=max_steps,
                trace=trace,
                context=context,
                cancel=cancel,
            )

    def exec(self, payload: Any) -> ExecResult:
        """Run as a node inside an outer flow, exposing the inner flow's final action."""
        context = self._prepare_context(get_current_context())
        if context is None:
            context = RunContext()
        with context.use_message_scope(self._message_scope):
            self._adopt_global_messages(context)
            result = self.flow.run(
                payload,
                max_steps=self.max_steps,
                trace=None,
                context=context,
            )
        action = (
            result.action or "default"
            if self.action is _INHERIT_ACTION
            else cast(Action, self.action)
        )
        return ExecResult(action, result.payload)

    async def aexec(self, payload: Any) -> ExecResult:
        """Run this agent as an async node inside an outer async flow."""
        context = self._prepare_context(get_current_context())
        if context is None:
            context = RunContext()
        with context.use_message_scope(self._message_scope):
            self._adopt_global_messages(context)
            result = await self.flow.arun(
                payload,
                max_steps=self.max_steps,
                trace=None,
                context=context,
            )
        action = (
            result.action or "default"
            if self.action is _INHERIT_ACTION
            else cast(Action, self.action)
        )
        return ExecResult(action, result.payload)

    def _adopt_global_messages(self, context: RunContext) -> None:
        """Merge unscoped ambient messages into this agent's scope once each.

        Harnesses may seed or continue a conversation through unscoped
        ``context.add_message`` calls (see examples/04). Entering the agent
        scope adopts that ambient history so the model sees the full
        conversation, while messages other agents wrote to their own scopes
        stay invisible.
        """
        scoped = context.message_scopes.setdefault(self._message_scope, [])
        known = {id(message) for message in scoped}
        foreign = {
            id(message)
            for name, messages in context.message_scopes.items()
            if name != self._message_scope
            for message in messages
        }
        for message in context.messages:
            if id(message) in known or id(message) in foreign:
                continue
            scoped.append(message)
            known.add(id(message))

    def _prepare_context(self, context: RunContext | None) -> RunContext | None:
        if self.instructions is None:
            return context
        if context is None:
            context = RunContext()
        if not context.metadata.get(self._instruction_marker):
            context.add_message("system", self.instructions, scope=self._message_scope)
            context.metadata[self._instruction_marker] = True
        return context
