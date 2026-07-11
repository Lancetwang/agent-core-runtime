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
    pass


class Node:
    def __init__(self, *, max_retries: int = 1, wait: float = 0) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        self.successors: dict[Action, Node] = {}
        self._action: Action = "default"
        self.max_retries = max_retries
        self.wait = wait

    def exec(self, payload: Any) -> ExecResult:
        raise NotImplementedError

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
        self.successors[self._action] = other
        self._action = "default"
        return other

    def __sub__(self, action: Action) -> Node:
        if not isinstance(action, str):
            raise TypeError("action must be a string.")
        self._action = action or "default"
        return self


class CallableNode(Node):
    def __init__(
        self,
        fn: Callable[[Any], ExecResult | Any],
        *,
        max_retries: int = 1,
        wait: float = 0,
    ) -> None:
        super().__init__(max_retries=max_retries, wait=wait)
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
    action: Action | None
    payload: Any
    path: list[str]
    trace: list[TraceEvent] = field(default_factory=list)
    context: RunContext | None = None
    usage: RunUsage = field(default_factory=RunUsage)


class Flow:
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
        current = self.start
        last_action: Action | None = None
        path: list[str] = []
        run_context = context or RunContext()
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
                if current is None:
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

                node_name = current.__class__.__name__
                path.append(node_name)
                run_context.set_execution_context(step=step, node=node_name)
                run_context.emit("node.start", category="node")
                last_action, payload = current._exec(payload)
                next_node = current.successors.get(last_action)
                run_context.set_execution_context(step=step, node=node_name)
                run_context.emit(
                    "node.end",
                    category="node",
                    action=last_action,
                    data={"next_node": next_node.__class__.__name__ if next_node else None},
                )
                current = next_node
        finally:
            if trace_options.enabled:
                run_context.on_event = previous_on_event
            reset_current_context(context_token)

        raise FlowError(f"Flow exceeded max_steps={max_steps}.")
