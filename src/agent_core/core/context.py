from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any


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
class RunUsage:
    """Cumulative model usage for a context or a delta between two snapshots."""

    requests: int = 0
    usage_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record(self, usage: Mapping[str, Any] | None) -> None:
        self.requests += 1
        if not isinstance(usage, Mapping):
            return
        input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
        if input_tokens is None or output_tokens is None:
            return
        self.usage_requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def snapshot(self) -> RunUsage:
        return RunUsage(
            requests=self.requests,
            usage_requests=self.usage_requests,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    def since(self, previous: RunUsage) -> RunUsage:
        return RunUsage(
            requests=self.requests - previous.requests,
            usage_requests=self.usage_requests - previous.usage_requests,
            input_tokens=self.input_tokens - previous.input_tokens,
            output_tokens=self.output_tokens - previous.output_tokens,
        )

    def to_dict(self) -> dict[str, int | None]:
        exact = self.requests == self.usage_requests
        input_tokens = self.input_tokens if exact else None
        output_tokens = self.output_tokens if exact else None
        return {
            "requests": self.requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens if exact else None,
        }


@dataclass
class RunContext:
    """Caller-owned runtime context and event stream.

    Business data moves through ``Node.exec(payload)`` and is returned as the
    flow result payload. The run context is intentionally separate: it carries
    conversation messages, artifacts, metadata, and UI/runtime events. Reuse
    one context across turns for a conversation, or create a fresh context for
    an isolated execution.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[dict[str, Any]] = field(default_factory=list)
    message_scopes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    active_message_scope: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[AgentEvent] = field(default_factory=list)
    usage: RunUsage = field(default_factory=RunUsage)
    on_event: Callable[[AgentEvent], None] | None = None
    on_observation: Callable[[AgentEvent], None] | None = None
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
        """Record an event in ``events`` and notify both subscribers.

        Use for events the run should remember (they drive UIs and traces).
        For large payloads that should not stay in memory, use :meth:`observe`.
        """
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
        if self.on_observation is not None:
            self.on_observation(event)
        if self.on_event is not None:
            self.on_event(event)
        return event

    def observe(
        self,
        type: str,
        *,
        category: str = "runtime",
        step: int | None = None,
        node: str | None = None,
        action: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> AgentEvent | None:
        """Send a detailed observation without retaining it in runtime memory."""
        if self.on_observation is None:
            return None
        event = AgentEvent(
            type=type,
            category=category,
            run_id=self.run_id,
            step=self.step if step is None else step,
            node=self.node if node is None else node,
            action=action,
            data=dict(data or {}),
        )
        self.on_observation(event)
        return event

    def notify(
        self,
        type: str,
        *,
        category: str = "runtime",
        step: int | None = None,
        node: str | None = None,
        action: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> AgentEvent | None:
        """Notify the live host without retaining the event or writing it to trace."""
        if self.on_event is None:
            return None
        event = AgentEvent(
            type=type,
            category=category,
            run_id=self.run_id,
            step=self.step if step is None else step,
            node=self.node if node is None else node,
            action=action,
            data=dict(data or {}),
        )
        self.on_event(event)
        return event

    def add_message(self, role: str, content: Any, **extra: Any) -> dict[str, Any]:
        """Append a chat message to the context and emit ``message.add``.

        ``extra`` keys (e.g. ``tool_calls``, ``tool_call_id``) are stored on
        the message. ``scope`` targets a specific message scope; by default
        the active scope is used, keeping each agent's prompt isolated in
        multi-agent flows.
        """
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
        """Return the messages for a scope (default: the active scope, else all)."""
        scope = self.active_message_scope if scope is None else scope
        if scope is None:
            return self.messages
        return self.message_scopes.setdefault(scope, [])

    @contextmanager
    def use_message_scope(self, scope: str):
        """Temporarily switch the active message scope for the enclosed block."""
        previous = self.active_message_scope
        self.active_message_scope = scope
        try:
            yield
        finally:
            self.active_message_scope = previous

    def set_artifact(self, name: str, value: Any) -> None:
        """Store a named artifact and emit ``artifact.set``."""
        self.artifacts[name] = value
        self.emit("artifact.set", category="artifact", data={"name": name})

    def record_model_usage(self, usage: Mapping[str, Any] | None) -> None:
        """Accumulate one model response's usage into :attr:`usage`."""
        self.usage.record(usage)

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
            "usage": self.usage.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }


def _usage_int(usage: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int):
            return value
    return None


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
