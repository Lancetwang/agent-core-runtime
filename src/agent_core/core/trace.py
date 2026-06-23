from __future__ import annotations

from collections.abc import Callable, Iterable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any
import time

TRACE_KEY = "_agent_core_trace"
DEFAULT_TRACE_CATEGORIES = frozenset({"flow", "node", "tool", "model"})


@dataclass(frozen=True)
class TraceEvent:
    event: str
    category: str
    step: int | None = None
    node: str | None = None
    action: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "category": self.category,
            "step": self.step,
            "node": self.node,
            "action": self.action,
            "data": self.data,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class TraceOptions:
    enabled: bool = True
    include: frozenset[str] = DEFAULT_TRACE_CATEGORIES
    print_to_console: bool = False
    printer: Callable[[str], None] = print
    on_event: Callable[[TraceEvent], None] | None = None
    trace_key: str = TRACE_KEY

    @classmethod
    def from_value(cls, value: TraceOptions | bool | None) -> TraceOptions:
        if isinstance(value, TraceOptions):
            return value
        return cls(enabled=bool(value))

    @classmethod
    def disabled(cls) -> TraceOptions:
        return cls(enabled=False)

    def includes(self, category: str) -> bool:
        return self.enabled and category in self.include


class TraceRecorder:
    def __init__(self, options: TraceOptions | bool | None = None) -> None:
        self.options = TraceOptions.from_value(options)
        self.events: list[TraceEvent] = []
        self.step: int | None = None
        self.node: str | None = None

    def set_context(self, *, step: int | None, node: str | None) -> None:
        self.step = step
        self.node = node

    def emit(
        self,
        event: str,
        *,
        category: str,
        step: int | None = None,
        node: str | None = None,
        action: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if not self.options.includes(category):
            return

        trace_event = TraceEvent(
            event=event,
            category=category,
            step=self.step if step is None else step,
            node=self.node if node is None else node,
            action=action,
            data=data or {},
        )
        self.events.append(trace_event)

        if self.options.on_event is not None:
            self.options.on_event(trace_event)

        if self.options.print_to_console:
            self.options.printer(format_trace_event(trace_event))

    def to_dicts(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]


_CURRENT_TRACE_RECORDER: ContextVar[TraceRecorder | None] = ContextVar(
    "agent_core_trace_recorder",
    default=None,
)


def make_trace_options(
    *,
    enabled: bool = True,
    include: Iterable[str] | None = None,
    print_to_console: bool = False,
    printer: Callable[[str], None] = print,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> TraceOptions:
    categories = DEFAULT_TRACE_CATEGORIES if include is None else frozenset(include)
    return TraceOptions(
        enabled=enabled,
        include=categories,
        print_to_console=print_to_console,
        printer=printer,
        on_event=on_event,
    )


def get_trace_recorder(payload: Any, trace_key: str = TRACE_KEY) -> TraceRecorder | None:
    current = _CURRENT_TRACE_RECORDER.get()
    if current is not None:
        return current
    if not isinstance(payload, dict):
        return None
    recorder = payload.get(trace_key)
    if isinstance(recorder, TraceRecorder):
        return recorder
    for value in payload.values():
        if isinstance(value, TraceRecorder):
            return value
    return None


def clear_trace_recorder(payload: Any, trace_key: str = TRACE_KEY) -> None:
    if isinstance(payload, dict):
        payload.pop(trace_key, None)


def set_current_trace_recorder(recorder: TraceRecorder) -> Token[TraceRecorder | None]:
    return _CURRENT_TRACE_RECORDER.set(recorder)


def reset_current_trace_recorder(token: Token[TraceRecorder | None]) -> None:
    _CURRENT_TRACE_RECORDER.reset(token)


def format_trace_event(event: TraceEvent) -> str:
    parts = [f"[trace:{event.category}]", event.event]
    if event.step is not None:
        parts.append(f"step={event.step}")
    if event.node:
        parts.append(f"node={event.node}")
    if event.action:
        parts.append(f"action={event.action}")
    if event.data:
        parts.append(f"data={event.data}")
    return " ".join(parts)
