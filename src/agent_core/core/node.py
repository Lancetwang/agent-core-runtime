from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
import time

from agent_core.core.context import (
    RunContext,
    RunUsage,
    reset_current_context,
    set_current_context,
)
from agent_core.core.trace import (
    TraceEvent,
    TraceOptions,
)

Action = str


class ExecResult(tuple):
    """Explicit ``(action, payload)`` routing result of one node execution.

    Subclasses ``tuple``, so existing destructuring (``action, payload =
    node.exec(...)``) keeps working, but instances are distinguishable from
    plain data tuples. ``CallableNode`` treats a returned :class:`ExecResult`
    as routing; an ordinary ``(str, value)`` tuple stays business payload.
    """

    __slots__ = ()

    def __new__(cls, action: Action, payload: Any) -> ExecResult:
        return super().__new__(cls, (action, payload))

    @property
    def action(self) -> Action:
        return self[0]

    @property
    def payload(self) -> Any:
        return self[1]


class FlowError(RuntimeError):
    """Raised when a flow cannot run or exceeds its step budget."""


class Node:
    """One unit of work in a flow.

    Subclasses implement ``exec(payload)`` and return ``(action, payload)``.
    The action string selects the next node; an action with no wired
    successor ends the flow. Edges are declared with the ``-``/``>>`` DSL::

        classify - "question" >> answer_node
        classify - "statement" >> summary_node

    ``max_retries`` and ``wait`` retry ``exec`` on exception before giving up.
    """

    def __init__(self, *, max_retries: int = 1, wait: float = 0) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be at least 1, got {max_retries}.")
        self.successors: dict[Action, Node] = {}
        self.max_retries = max_retries
        self.wait = wait

    def exec(self, payload: Any) -> ExecResult:
        """Do this node's work and return ``(action, payload)``."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement exec(payload) and return (action, payload)."
        )

    def _exec(self, payload: Any) -> ExecResult:
        for attempt in range(self.max_retries):
            try:
                return self.exec(payload)
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                if self.wait > 0:
                    time.sleep(self.wait)
        raise RuntimeError("Unexpected error in Node._exec")

    def add_edge(self, action: Action, successor: Node) -> Node:
        """Wire ``action`` to ``successor`` programmatically and return it.

        This is the explicit, stateless form of the ``node - "action" >>
        next`` DSL, useful when flows are built from data instead of literal
        wiring. Each action selects exactly one successor; wiring the same
        action to a different node raises ``ValueError``.
        """
        if not isinstance(successor, Node):
            raise TypeError(
                f"The right side of >> must be a Node, got {type(successor).__name__}. "
                'Write edges as: node - "action" >> next_node.'
            )
        existing = self.successors.get(action)
        if existing is not None and existing is not successor:
            raise ValueError(
                f"{type(self).__name__} already routes action '{action}' to "
                f"{type(existing).__name__}. Each action selects exactly one successor; "
                "use a distinct action name for the new edge."
            )
        self.successors[action] = successor
        return successor

    def __rshift__(self, other: Node) -> Node:
        """Shorthand for ``self - "default" >> other``; returns ``other`` for chaining."""
        return self.add_edge("default", other)

    def __sub__(self, action: Action) -> _ActionBinding:
        """Select the action name for the next ``>>`` edge without mutating the node."""
        if not isinstance(action, str):
            raise TypeError(f"action must be a string, got {type(action).__name__}.")
        return _ActionBinding(self, action or "default")


class _ActionBinding:
    """Pending ``node - "action"`` selection for the next ``>>`` edge.

    Carries the pending action instead of mutating the source node, so wiring
    is stateless: graph construction no longer depends on hidden node state
    and concurrent wiring of distinct nodes stays safe.
    """

    __slots__ = ("source", "action")

    def __init__(self, source: Node, action: Action) -> None:
        self.source = source
        self.action = action

    def __rshift__(self, other: Node) -> Node:
        return self.source.add_edge(self.action, other)


class CallableNode(Node):
    """Adapt a plain function into a node.

    The function receives the payload. If it returns ``ExecResult(action,
    payload)`` that pair is used as-is; any other return value — including a
    plain ``(str, value)`` tuple — is treated as business payload and wrapped
    as ``("default", value)``.
    """

    def __init__(
        self,
        fn: Callable[[Any], ExecResult | Any],
        *,
        max_retries: int = 1,
        wait: float = 0,
    ) -> None:
        super().__init__(max_retries=max_retries, wait=wait)
        if not callable(fn):
            raise TypeError(f"CallableNode requires a callable, got {type(fn).__name__}.")
        self.fn = fn

    def exec(self, payload: Any) -> ExecResult:
        result = self.fn(payload)
        if isinstance(result, ExecResult):
            return result
        return ExecResult("default", result)


@dataclass(frozen=True)
class FlowRunResult:
    """Outcome of one ``Flow.run``.

    ``action`` is the last action returned before the flow ended; ``payload``
    is the final business data; ``path`` lists visited node class names;
    ``trace`` holds the events selected by the trace options; ``context`` is
    the run's :class:`RunContext`; ``usage`` is the model usage delta for this
    invocation only.
    """

    action: Action | None
    payload: Any
    path: list[str]
    trace: list[TraceEvent] = field(default_factory=list)
    context: RunContext | None = None
    usage: RunUsage = field(default_factory=RunUsage)


class Flow:
    """Route payloads between nodes by action name until no successor matches.

    A flow ends when the current node returns an action with no wired
    successor. ``Flow.run`` raises :class:`FlowError` if the flow has no start
    node or does not finish within ``max_steps``.
    """

    def __init__(self, start: Node | None = None) -> None:
        self.start = start

    def run(
        self,
        payload: Any = None,
        *,
        max_steps: int = 100,
        trace: TraceOptions | bool | None = None,
        context: RunContext | None = None,
    ) -> FlowRunResult:
        if self.start is None:
            raise FlowError("Flow has no start node. Construct it as Flow(start_node).")
        current = self.start
        last_action: Action | None = None
        path: list[str] = []
        run_context = context if context is not None else RunContext()
        trace_options = TraceOptions.from_value(trace)
        event_start = len(run_context.events)
        usage_start = run_context.usage.snapshot()
        previous_on_event = run_context.on_event
        if trace_options.enabled:
            def on_event(event: TraceEvent) -> None:
                if previous_on_event is not None:
                    previous_on_event(event)
                trace_options.dispatch(event)

            run_context.on_event = on_event
        context_token = set_current_context(run_context)

        # Nested flows share one step budget: an inner flow (for example an
        # Agent used as a node) may not burn more steps than the outer run
        # has left. The remaining budget travels in a ContextVar so nesting
        # composes without threading parameters through Node.exec.
        outer_budget = _STEP_BUDGET.get()
        budget = max_steps if outer_budget is None else min(max_steps, outer_budget)
        budget_token = _STEP_BUDGET.set(budget)

        try:
            for step in range(1, budget + 1):
                _STEP_BUDGET.set(budget - step)
                node_name = current.__class__.__name__
                path.append(node_name)
                run_context.set_execution_context(step=step, node=node_name)
                run_context.emit("node.start", category="node")
                try:
                    last_action, payload = current._exec(payload)
                except Exception as exc:
                    error = {"error_type": type(exc).__name__, "message": str(exc)}
                    run_context.emit("node.error", category="node", data=error)
                    run_context.emit("flow.error", category="flow", data=error)
                    raise
                # Align with the wiring DSL, which normalizes empty action
                # names to "default": a node returning "" routes to the
                # "default" successor instead of silently ending the flow.
                last_action = last_action or "default"
                next_node = current.successors.get(last_action)
                run_context.set_execution_context(step=step, node=node_name)
                run_context.emit(
                    "node.end",
                    category="node",
                    action=last_action,
                    data={"next_node": next_node.__class__.__name__ if next_node else None},
                )
                if next_node is None:
                    run_context.set_execution_context(step=step, node=None)
                    run_context.emit("flow.end", category="flow", step=step, node=None)
                    events = run_context.events[event_start:]
                    return FlowRunResult(
                        action=last_action,
                        payload=payload,
                        path=path,
                        trace=[event for event in events if trace_options.includes(event.category)],
                        context=run_context,
                        usage=run_context.usage.since(usage_start),
                    )
                current = next_node

            nested_note = (
                " The remaining budget was reduced by an enclosing flow run."
                if outer_budget is not None
                else ""
            )
            error = FlowError(
                f"Flow exceeded max_steps={budget}.{nested_note} "
                "Raise max_steps for long runs, or check the graph for an action cycle that never ends."
            )
            run_context.emit(
                "flow.error",
                category="flow",
                data={"error_type": type(error).__name__, "message": str(error)},
            )
            raise error
        finally:
            _STEP_BUDGET.reset(budget_token)
            if trace_options.enabled:
                run_context.on_event = previous_on_event
            reset_current_context(context_token)


_STEP_BUDGET: ContextVar[int | None] = ContextVar(
    "agent_core_step_budget",
    default=None,
)
