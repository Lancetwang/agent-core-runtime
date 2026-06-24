from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
import time

from agent_core.core.context import (
    RunContext,
    reset_current_context,
    set_current_context,
)
from agent_core.core.trace import (
    TraceEvent,
    TraceOptions,
    TraceRecorder,
    get_trace_recorder,
    reset_current_trace_recorder,
    set_current_trace_recorder,
)

Action = str
Payload = Any
ExecResult = tuple[Action, Payload]


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

    def exec(self, payload: Payload) -> ExecResult:
        raise NotImplementedError

    def _exec(self, payload: Payload) -> ExecResult:
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
        fn: Callable[[Payload], ExecResult | Payload],
        *,
        max_retries: int = 1,
        wait: float = 0,
    ) -> None:
        super().__init__(max_retries=max_retries, wait=wait)
        self.fn = fn

    def exec(self, payload: Payload) -> ExecResult:
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
    payload: Payload
    path: list[str]
    trace: list[TraceEvent] = field(default_factory=list)
    context: RunContext | None = None


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
        run_context = context or RunContext.from_payload(payload)
        inherited_recorder = get_trace_recorder(payload) if trace is None else None
        recorder = inherited_recorder or TraceRecorder(trace)
        context_token = set_current_context(run_context)
        token = None if inherited_recorder else set_current_trace_recorder(recorder)

        try:
            for step in range(1, max_steps + 1):
                if current is None:
                    recorder.set_context(step=step, node=None)
                    recorder.emit("flow.end", category="flow", step=step, node=None)
                    run_context.set_execution_context(step=step, node=None)
                    run_context.sync_payload(payload)
                    run_context.emit("flow.end", category="flow", step=step, node=None)
                    return FlowRunResult(
                        action=last_action,
                        payload=payload,
                        path=path,
                        trace=list(recorder.events),
                        context=run_context,
                    )

                node_name = current.__class__.__name__
                path.append(node_name)
                recorder.set_context(step=step, node=node_name)
                recorder.emit("node.start", category="node")
                run_context.set_execution_context(step=step, node=node_name)
                run_context.sync_payload(payload)
                run_context.emit("node.start", category="node")
                last_action, payload = current._exec(payload)
                next_node = current.successors.get(last_action)
                recorder.set_context(step=step, node=node_name)
                run_context.set_execution_context(step=step, node=node_name)
                recorder.emit(
                    "node.end",
                    category="node",
                    action=last_action,
                    data={"next_node": next_node.__class__.__name__ if next_node else None},
                )
                run_context.sync_payload(payload)
                run_context.emit(
                    "node.end",
                    category="node",
                    action=last_action,
                    data={"next_node": next_node.__class__.__name__ if next_node else None},
                )
                current = next_node
        finally:
            if token is not None:
                reset_current_trace_recorder(token)
            reset_current_context(context_token)

        raise FlowError(f"Flow exceeded max_steps={max_steps}.")
