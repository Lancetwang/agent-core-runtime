from __future__ import annotations

from collections.abc import Callable
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
ExecResult = tuple[Action, Any]


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
        self._action: Action = "default"
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

    def __rshift__(self, other: Node) -> Node:
        """Wire ``self - "action" >> other`` and return ``other`` for chaining."""
        if not isinstance(other, Node):
            raise TypeError(
                f"The right side of >> must be a Node, got {type(other).__name__}. "
                'Write edges as: node - "action" >> next_node.'
            )
        existing = self.successors.get(self._action)
        if existing is not None and existing is not other:
            raise ValueError(
                f"{type(self).__name__} already routes action '{self._action}' to "
                f"{type(existing).__name__}. Each action selects exactly one successor; "
                "use a distinct action name for the new edge."
            )
        self.successors[self._action] = other
        self._action = "default"
        return other

    def __sub__(self, action: Action) -> Node:
        """Select the action name for the next ``>>`` edge."""
        if not isinstance(action, str):
            raise TypeError(f"action must be a string, got {type(action).__name__}.")
        self._action = action or "default"
        return self


class CallableNode(Node):
    """Adapt a plain function into a node.

    The function receives the payload. If it returns ``(action, payload)``
    that pair is used as-is; any other return value is wrapped as
    ``("default", value)``.
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
        if self._is_exec_result(result):
            return result
        return "default", result

    @staticmethod
    def _is_exec_result(value: Any) -> bool:
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], str)
        )


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

        try:
            for step in range(1, max_steps + 1):
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

            error = FlowError(
                f"Flow exceeded max_steps={max_steps}. "
                "Raise max_steps for long runs, or check the graph for an action cycle that never ends."
            )
            run_context.emit(
                "flow.error",
                category="flow",
                data={"error_type": type(error).__name__, "message": str(error)},
            )
            raise error
        finally:
            if trace_options.enabled:
                run_context.on_event = previous_on_event
            reset_current_context(context_token)
