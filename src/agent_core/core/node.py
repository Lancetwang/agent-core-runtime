from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

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
    plain data tuples. ``CallableNode`` always treats an :class:`ExecResult`
    as routing; tuple-shape inference remains available as a 0.1.x
    compatibility mode.
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


class FlowCancelled(FlowError):
    """Raised when a run is cancelled through the cooperative cancel event."""


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

    def exec(self, payload: Any) -> tuple[Action, Any]:
        """Do this node's work and return ``(action, payload)``."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement exec(payload) and return (action, payload)."
        )

    def _exec(self, payload: Any) -> tuple[Action, Any]:
        for attempt in range(self.max_retries):
            try:
                return self.exec(payload)
            except FlowCancelled:
                raise
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
        if not isinstance(action, str):
            raise TypeError(f"action must be a string, got {type(action).__name__}.")
        action = action or "default"
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
    payload)`` that pair is used as-is. For 0.1.x compatibility, a plain
    ``(str, value)`` tuple is also routed by default; pass
    ``route_plain_tuples=False`` when tuples are business payloads. Any other
    return value is wrapped as ``("default", value)``.
    """

    def __init__(
        self,
        fn: Callable[[Any], ExecResult | Any],
        *,
        route_plain_tuples: bool = True,
        max_retries: int = 1,
        wait: float = 0,
    ) -> None:
        super().__init__(max_retries=max_retries, wait=wait)
        if not callable(fn):
            raise TypeError(f"CallableNode requires a callable, got {type(fn).__name__}.")
        self.fn = fn
        self.route_plain_tuples = route_plain_tuples

    def exec(self, payload: Any) -> ExecResult:
        result = self.fn(payload)
        if isinstance(result, ExecResult):
            return result
        # Keep the 0.1.x CallableNode contract available while callers migrate
        # routing functions to explicit ExecResult values. Tuple-shaped
        # business data can opt out immediately with route_plain_tuples=False.
        if self.route_plain_tuples and self._is_legacy_exec_result(result):
            warnings.warn(
                "Returning a plain (action, payload) tuple from CallableNode is "
                "deprecated; return ExecResult(action, payload) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return ExecResult(result[0], result[1])
        return ExecResult("default", result)

    @staticmethod
    def _is_legacy_exec_result(value: Any) -> bool:
        return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str)


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
    context: RunContext = field(default_factory=RunContext)
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
        cancel: threading.Event | None = None,
    ) -> FlowRunResult:
        if self.start is None:
            raise FlowError("Flow has no start node. Construct it as Flow(start_node).")
        current = self.start
        last_action: Action | None = None
        path: list[str] = []
        run_context = context if context is not None else RunContext()
        trace_options = TraceOptions.from_value(trace)
        usage_start = run_context.usage.snapshot()
        previous_on_event = run_context.on_event
        # Trace collection is independent of retention: it captures every
        # live event of this run, including non-retained notify events such
        # as streamed model deltas.
        trace_events: list[TraceEvent] = []
        if trace_options.enabled:
            def on_event(event: TraceEvent) -> None:
                if previous_on_event is not None:
                    previous_on_event(event)
                trace_events.append(event)
                trace_options.dispatch(event)

            run_context.on_event = on_event
        context_token = set_current_context(run_context)

        # Every visited node, including nodes inside nested Agent flows,
        # consumes from one shared run budget. Each nested Flow still keeps
        # its own max_steps cap as a local guard.
        shared_budget = _STEP_BUDGET.get()
        if shared_budget is None:
            shared_budget = _StepBudget(max_steps)
        budget_token = _STEP_BUDGET.set(shared_budget)

        # Cooperative cancellation: an explicit event wins; otherwise nested
        # runs inherit the enclosing run's event through the ContextVar.
        cancel_event = cancel if cancel is not None else _FLOW_CANCEL.get()
        cancel_token = _FLOW_CANCEL.set(cancel_event)

        try:
            for step in range(1, max_steps + 1):
                node_name = current.__class__.__name__
                run_context.set_execution_context(step=step, node=node_name)
                if cancel_event is not None and cancel_event.is_set():
                    run_context.emit(
                        "flow.cancel",
                        category="flow",
                        step=step,
                        data={"step": step},
                    )
                    raise FlowCancelled(f"Flow cancelled at step {step}.")
                if not shared_budget.consume():
                    error = FlowError(
                        f"Flow exhausted shared max_steps={shared_budget.limit}. "
                        "Nested flow steps count toward the enclosing run budget."
                    )
                    run_context.emit(
                        "flow.error",
                        category="flow",
                        data={"error_type": type(error).__name__, "message": str(error)},
                    )
                    raise error
                path.append(node_name)
                run_context.emit("node.start", category="node")
                try:
                    last_action, payload = current._exec(payload)
                    if not isinstance(last_action, str):
                        raise TypeError(
                            f"{node_name}.exec(payload) returned a non-string action: "
                            f"{last_action!r}."
                        )
                    if last_action == "":
                        last_action = "default"
                except Exception as exc:
                    error = {"error_type": type(exc).__name__, "message": str(exc)}
                    run_context.set_execution_context(step=step, node=node_name)
                    run_context.emit("node.error", category="node", data=error)
                    run_context.emit("flow.error", category="flow", data=error)
                    raise
                # Align with the wiring DSL, which normalizes empty action
                # names to "default": a node returning "" routes to the
                # "default" successor instead of silently ending the flow.
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
                    return FlowRunResult(
                        action=last_action,
                        payload=payload,
                        path=path,
                        trace=[
                            event
                            for event in trace_events
                            if trace_options.includes(event.category)
                        ],
                        context=run_context,
                        usage=run_context.usage.since(usage_start),
                    )
                current = next_node

            error = FlowError(
                f"Flow exceeded max_steps={max_steps}. "
                "Raise max_steps for long runs, or check the graph for "
                "an action cycle that never ends."
            )
            run_context.emit(
                "flow.error",
                category="flow",
                data={"error_type": type(error).__name__, "message": str(error)},
            )
            raise error
        finally:
            _FLOW_CANCEL.reset(cancel_token)
            _STEP_BUDGET.reset(budget_token)
            if trace_options.enabled:
                run_context.on_event = previous_on_event
            reset_current_context(context_token)


class _StepBudget:
    """Thread-safe counter shared by nested flow runs in one context."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.remaining = limit
        self._lock = threading.Lock()

    def consume(self) -> bool:
        with self._lock:
            if self.remaining <= 0:
                return False
            self.remaining -= 1
            return True


_STEP_BUDGET: ContextVar[_StepBudget | None] = ContextVar(
    "agent_core_step_budget",
    default=None,
)

_FLOW_CANCEL: ContextVar[threading.Event | None] = ContextVar(
    "agent_core_flow_cancel",
    default=None,
)
