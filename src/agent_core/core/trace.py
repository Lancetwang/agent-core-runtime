from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from agent_core.core.context import AgentEvent

DEFAULT_TRACE_CATEGORIES = frozenset({"flow", "node", "tool", "model", "llm", "plan"})
TraceEvent = AgentEvent


@dataclass(frozen=True)
class TraceOptions:
    enabled: bool = True
    include: frozenset[str] = DEFAULT_TRACE_CATEGORIES
    print_to_console: bool = False
    printer: Callable[[str], None] = print
    on_event: Callable[[TraceEvent], None] | None = None

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

    def dispatch(self, event: TraceEvent) -> None:
        if not self.includes(event.category):
            return
        if self.on_event is not None:
            self.on_event(event)
        if self.print_to_console:
            self.printer(format_trace_event(event))


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


def format_trace_event(event: TraceEvent) -> str:
    parts = [f"[trace:{event.category}]", event.type]
    if event.step is not None:
        parts.append(f"step={event.step}")
    if event.node:
        parts.append(f"node={event.node}")
    if event.action:
        parts.append(f"action={event.action}")
    if event.data:
        parts.append(f"data={event.data}")
    return " ".join(parts)
