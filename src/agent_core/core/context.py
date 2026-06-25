from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Callable, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any
import time
import uuid


@dataclass(frozen=True)
class AgentEvent:
    """Product-level runtime event emitted during an agent run."""

    type: str
    category: str = "runtime"
    run_id: str = ""
    step: int | None = None
    node: str | None = None
    action: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "category": self.category,
            "run_id": self.run_id,
            "step": self.step,
            "node": self.node,
            "action": self.action,
            "data": self.data,
            "timestamp": self.timestamp,
        }


@dataclass
class RunContext:
    """Runtime context and event stream for one flow execution.

    Business data moves through ``Node.exec(payload)`` and is returned as the
    flow result payload. The run context is intentionally separate: it carries
    conversation messages, artifacts, metadata, and UI/runtime events.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[dict[str, Any]] = field(default_factory=list)
    message_scopes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    active_message_scope: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[AgentEvent] = field(default_factory=list)
    on_event: Callable[[AgentEvent], None] | None = None
    step: int | None = None
    node: str | None = None

    def set_execution_context(
        self,
        *,
        step: int | None,
        node: str | None,
    ) -> None:
        self.step = step
        self.node = node

    def emit(
        self,
        type: str,
        *,
        category: str = "runtime",
        step: int | None = None,
        node: str | None = None,
        action: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> AgentEvent:
        event = AgentEvent(
            type=type,
            category=category,
            run_id=self.run_id,
            step=self.step if step is None else step,
            node=self.node if node is None else node,
            action=action,
            data=dict(data or {}),
        )
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)
        return event

    def add_message(self, role: str, content: str, **extra: Any) -> dict[str, Any]:
        scope = extra.pop("scope", self.active_message_scope)
        message = {"role": role, "content": content, **extra}
        self.messages.append(message)
        if scope is not None:
            self.message_scopes.setdefault(scope, []).append(message)
        self.emit(
            "message.add",
            category="message",
            data={"role": role, "content": content, "scope": scope, **extra},
        )
        return message

    def get_messages(self, scope: str | None = None) -> list[dict[str, Any]]:
        scope = self.active_message_scope if scope is None else scope
        if scope is None:
            return self.messages
        return self.message_scopes.setdefault(scope, [])

    @contextmanager
    def use_message_scope(self, scope: str):
        previous = self.active_message_scope
        self.active_message_scope = scope
        try:
            yield
        finally:
            self.active_message_scope = previous

    def set_artifact(self, name: str, value: Any) -> None:
        self.artifacts[name] = value
        self.emit("artifact.set", category="artifact", data={"name": name})

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "messages": list(self.messages),
            "message_scopes": {
                name: list(messages)
                for name, messages in self.message_scopes.items()
            },
            "active_message_scope": self.active_message_scope,
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
            "events": [event.to_dict() for event in self.events],
        }


_CURRENT_RUN_CONTEXT: ContextVar[RunContext | None] = ContextVar(
    "agent_core_run_context",
    default=None,
)


def get_current_context() -> RunContext | None:
    return _CURRENT_RUN_CONTEXT.get()


def set_current_context(context: RunContext) -> Token[RunContext | None]:
    return _CURRENT_RUN_CONTEXT.set(context)


def reset_current_context(token: Token[RunContext | None]) -> None:
    _CURRENT_RUN_CONTEXT.reset(token)

