# API Reference

Everything documented here is importable from the top-level `agent_core`
package and covered by `agent_core.__all__`. Anything else is internal.

`agent_core.__version__` holds the installed distribution version.

## Core

### `Node`

One unit of work. Subclass and implement `exec(payload) -> (action, payload)`.
Wire edges with the DSL: `node - "action" >> next_node`. An action with no
successor ends the flow. Constructor options: `max_retries`, `wait` (retry
`exec` on exception).

- Wiring the same action twice raises `ValueError`.
- The right side of `>>` must be a `Node`, otherwise `TypeError`.

### `CallableNode(fn)`

Adapts a plain function into a node. If `fn(payload)` returns
`(action, payload)` it is used as-is; any other value becomes
`("default", value)`.

### `Flow(start)`

Routes payloads between nodes by action name.

```python
result = Flow(start_node).run(payload, max_steps=100, trace=None, context=None)
```

Raises `FlowError` when the flow has no start node or exceeds `max_steps`.

### `FlowRunResult`

`action` (last action), `payload` (final business data), `path` (visited node
class names), `trace` (selected events), `context` (the `RunContext`),
`usage` (model usage delta for this invocation).

### `RunContext`

Runtime state for one execution: `messages` and per-agent `message_scopes`,
`events`, `artifacts`, `metadata`, cumulative `usage`, and the
`on_event` / `on_observation` subscriber hooks.

Key methods:

- `add_message(role, content, **extra)` — append a chat message (extra keys
  such as `tool_calls` are stored on the message; `scope=` targets a scope).
- `get_messages(scope=None)` — messages for the active scope, or all.
- `use_message_scope(scope)` — context manager to switch the active scope.
- `emit(type, category=..., data=...)` — record an event and notify
  subscribers; `observe(...)` notifies `on_observation` only, without
  retaining the event in memory (for large payloads).
- `set_artifact(name, value)`, `record_model_usage(usage)`, `to_dict()`.

Inside a node, `get_current_context()` returns the active run's context (or
`None` outside a run). `set_current_context` / `reset_current_context` are
low-level hooks for embedding the runtime.

### `RunUsage`

Cumulative model usage. `snapshot()` copies the current counters;
`since(previous)` returns a delta; `to_dict()` reports exact token totals
only when every response carried usage, otherwise `None` instead of a
misleading partial sum.

### `AgentEvent` / `TraceEvent`

Frozen event record: `type`, `category`, `run_id`, `step`, `node`, `action`,
`data`, `timestamp`. `TraceEvent` is an alias of `AgentEvent`.

## Agent

### `Agent`

An agent that runs a flow — and is itself a `Node`, so agents compose into
larger flows.

```python
Agent(model=..., instructions=..., tools=[...])   # standard chat loop
Agent(Flow(...), instructions=...)                # custom loop
```

- `chat(text, *, context=None, max_steps=100, trace=None, stream=None,
  on_delta=None, payload=None) -> str` — one user turn; reuse `context` to
  hold a conversation.
- `run(payload, *, max_steps, trace, context) -> FlowRunResult` — run the
  inner flow on a payload.
- `new_context() -> RunContext` — fresh context carrying this agent's
  message scope and instructions.
- As a node, an agent exposes its inner flow's final action; pass
  `action="..."` to force a fixed outward action instead.

Each `Agent` keeps an isolated message scope inside a shared `RunContext`:
events, usage, artifacts, and metadata are unified across a multi-agent flow
while prompts never leak between agents.

## Model layer

### `ChatModel` (protocol)

Implement one method to plug in any provider:

```python
def chat_message(messages, *, tools=None, tool_choice=None, **kwargs) -> dict
```

It must return an OpenAI-style assistant message:
`{"role": "assistant", "content": str, "tool_calls": [...]?, "usage": {...}?}`.
`Message` is the input message mapping type.

### `LLM`

Default OpenAI-compatible adapter. Explicit arguments win; otherwise
configuration comes from `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` (or the
`OPENAI_*` / `DEEPSEEK_*` aliases). Supports `stream=True` with an
`on_delta` text callback, and captures usage in both modes.

### `ModelNode`

Calls the model once and stores the assistant message in
`state[assistant_key]`. Messages come from an explicit builder, the active
context's scope, or `state[messages_key]` — in that order. Per-call overrides
travel in `state[chat_kwargs_key]`. Emits `model.request` / `model.response`
and records usage.

### `ToolRouterNode`

Routes on the assistant message: `tool_action` when tool calls are pending,
otherwise stores the text under `state[output_key]` and returns
`done_action`.

## Tools

### `@tool(description, *, name=None)`

Converts a typed Python function into a `Tool` with an OpenAI-compatible
JSON schema. Parameter types map to JSON types; `Annotated[T, "text"]` adds
per-parameter descriptions; defaults mark parameters optional. Raises
`ToolDefinitionError` (naming the function and parameter) when a signature
cannot be converted.

### `Tool`

`name`, `description`, `parameters` (JSON schema), `fn`. `to_llm_format()`
returns the OpenAI `tools` entry; `execute(**kwargs)` invokes the function;
the instance itself stays callable.

### `ToolExecutor(tools)`

Executes model tool calls. Model-caused failures never raise: unknown tools
(the error lists available names) and tool exceptions come back as
`ToolResult(is_error=True)` so the model can self-correct.
`parse_tool_calls(message)` extracts `ToolCall`s; `execute` / `execute_all`
run them.

### `ToolCallNode`

Runs pending tool calls inside a flow: executes them, stores results under
`state[results_key]`, appends `role: tool` messages to the history and the
active context, and emits `tool.call` / `tool.result` events.

### `ToolCall` / `ToolResult`

Frozen records for one requested invocation and its outcome.
`ToolCall.from_openai_item` parses leniently — malformed argument JSON
becomes `{}` so the tool can report a precise validation error back to the
model. `ToolResult.to_message()` yields the OpenAI-style tool message.

## Tracing

### `TraceOptions` / `make_trace_options(...)`

Select which run events are collected, forwarded to `on_event`, or printed.
Pass `trace=True` for defaults, or build options with categories from
`{"flow", "node", "tool", "model", "llm", "plan"}`.
`format_trace_event(event)` renders one event as a log line.

## Events emitted by the runtime

| Event | Category | Emitted by |
| --- | --- | --- |
| `node.start` / `node.end` | `node` | `Flow` |
| `flow.end` | `flow` | `Flow` |
| `model.request` / `model.response` | `model` | `ModelNode` |
| `model.delta` | `model` | streaming callback |
| `model.request.payload` / `model.response.payload` | `model` | `ModelNode` (observe-only) |
| `tool.observe` | `tool` | `ToolRouterNode` |
| `tool.call` / `tool.result` | `tool` | `ToolCallNode` |
| `message.add` | `message` | `RunContext.add_message` |
| `artifact.set` | `artifact` | `RunContext.set_artifact` |
