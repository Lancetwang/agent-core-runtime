# API Reference

Everything documented here is importable from the top-level `agent_core`
package and covered by `agent_core.__all__`. Anything else is internal.

`agent_core.__version__` holds the installed distribution version.

Package diagnostics:

```powershell
python -m agent_core --version
python -m agent_core check
```

The local check exercises `CallableNode`, `Flow`, and `RunContext` without
credentials or a model request.

## Core

### `Node`

One unit of work. Subclass and implement `exec(payload) -> (action, payload)`.
Wire edges with the DSL: `node - "action" >> next_node`, or programmatically
with `node.add_edge("action", next_node)` / `node >> next_node` for the
`"default"` action. An action with no successor ends the flow. Constructor
options: `max_retries`, `wait` (retry `exec` on exception).

- Action selection (`node - "action"`) is stateless: it produces a binding
  used by the next `>>` and never mutates the node.
- Wiring the same action twice raises `ValueError`.
- The right side of `>>` must be a `Node`, otherwise `TypeError`.

### `CallableNode(fn)`

Adapts a plain function into a node. If `fn(payload)` returns
`ExecResult(action, payload)` it is used as-is; any other value — including
a plain `(str, value)` tuple — becomes `("default", value)`.

### `ExecResult`

Explicit `(action, payload)` routing result. It is a `tuple` subclass, so
`action, payload = result` keeps working, but only `ExecResult` instances
returned from a `CallableNode` function are treated as routing; ordinary
tuples stay business payload.

### `Flow(start)`

Routes payloads between nodes by action name.

```python
result = Flow(start_node).run(payload, max_steps=100, trace=None, context=None)
```

Raises `FlowError` when the flow has no start node or exceeds `max_steps`.
Nested flows share one step budget: an inner flow may not burn more steps
than the enclosing run has left.

### `PayloadKeys`

Canonical payload state keys used by the built-in nodes: `INPUT`, `ANSWER`,
`ASSISTANT_MESSAGE`, `HISTORY`, `CHAT_KWARGS`, `TOOL_RESULTS`. Custom flows
should use these names instead of literal strings when touching the built-in
contract, so a typo cannot silently produce an empty answer.

### `FlowRunResult`

`action` (last action), `payload` (final business data), `path` (visited node
class names), `trace` (selected events), `context` (the `RunContext`),
`usage` (model usage delta for this invocation).

### `RunContext`

Caller-owned runtime/session state: `messages` and per-agent `message_scopes`,
`events`, `artifacts`, `metadata`, cumulative `usage`, and the
`on_event` / `on_observation` subscriber hooks.

Reuse one context across `Agent.chat` calls to preserve a conversation. Do not
drive the same context concurrently from multiple flows.

Key methods:

- `add_message(role, content, **extra)` — append a chat message (extra keys
  such as `tool_calls` are stored on the message; `scope=` targets a scope).
- `get_messages(scope=None)` — messages for the active scope, or all.
- `use_message_scope(scope)` — context manager to switch the active scope.
- `emit(type, category=..., data=...)` — record an event and notify
  subscribers; `observe(...)` sends non-retained detail to `on_observation`,
  while `notify(...)` sends transient UI progress to `on_event` only.

Retention policy: retained events (`emit`) carry small metadata, so the
event stream stays bounded over long sessions. Large or sensitive payloads
travel through `observe` (never retained), and transient per-chunk streams
such as model deltas travel through `notify` (live only). Trace collection
captures live events of the run regardless of retention.
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

- `chat(text, *, content=None, context=None, max_steps=None, trace=None,
  stream=None, on_delta=None, payload=None) -> str` — one user turn; reuse
  `context` to hold a conversation. Pass OpenAI-style multimodal `content`
  while keeping `text` as the flow's business input.
- `run(payload, *, max_steps=None, trace, context) -> FlowRunResult` — run the
  inner flow on a payload. `max_steps` defaults to the constructor budget.
  Nested flows share one step budget: an `Agent` running inside an outer
  flow may not burn more steps than the outer run has left.
- `new_context() -> RunContext` — fresh context carrying this agent's
  message scope and instructions.
- As a node, an agent exposes its inner flow's final action; pass
  `action="..."` to force a fixed outward action instead.

Each `Agent` keeps an isolated message scope inside a shared `RunContext`:
events, usage, artifacts, and metadata are unified across a multi-agent flow
while prompts never leak between agents.

## Model layer

### `ChatModel` (protocol)

Implement one normalized OpenAI-style method, translating a provider's native
request and response schema when needed:

```python
def chat_message(messages, *, tools=None, tool_choice=None, **kwargs) -> dict
```

It must return an OpenAI-style assistant message:
`{"role": "assistant", "content": str, "reasoning_content": str?, "tool_calls": [...]?, "usage": {...}?}`.
`Message` is the input message mapping type.

### `LLM`

Default OpenAI-compatible adapter. Explicit arguments win; otherwise
configuration comes from `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` (or the
`OPENAI_*` / `DEEPSEEK_*` aliases). Supports `stream=True` with an
`on_delta` text callback, and captures usage in both modes.

### `ModelNode`

Calls the model once and stores the assistant message in
`state[assistant_key]`. Messages come from an explicit builder, the active
context's scope, or `state[messages_key]` — in that order. The context scope
is the single canonical history during a flow run; `state[messages_key]` is
only an import seed for custom flows and no longer mirrors new messages.
Per-call overrides travel in `state[chat_kwargs_key]`. Emits
`model.request` / `model.response` and records usage.

An `Agent` adopts unscoped ambient messages (added to the context outside
any scope) into its own scope at run start, so a conversation seeded through
`context.add_message` without a scope is visible to the model while other
agents' scoped messages stay isolated.

### `ToolRouterNode`

Routes on the assistant message: `tool_action` when tool calls are pending,
otherwise stores the text under `state[output_key]` and returns
`done_action`.

## Tools

### `@tool(description, *, name=None, parallel=False)`

Converts a typed Python function into a `Tool` with an OpenAI-compatible
JSON schema. Parameter types map to JSON types; `Annotated[T, "text"]` adds
per-parameter descriptions; `TypedDict` describes structured objects inside
lists; defaults mark parameters optional. Raises
`ToolDefinitionError` (naming the function and parameter) when a signature
cannot be converted.

### `Tool`

`name`, `description`, `parameters` (JSON schema), `fn`, `parallel`. `to_llm_format()`
returns the OpenAI `tools` entry; `execute(**kwargs)` invokes the function;
the instance itself stays callable.

### `ToolExecutor(tools, *, max_workers=4)`

Executes model tool calls. Model-caused failures never raise: unknown tools
(the error lists available names) and tool exceptions come back as
`ToolResult(is_error=True)` so the model can self-correct.
`parse_tool_calls(message)` extracts `ToolCall`s; `execute` / `execute_all`
run them. Consecutive `parallel=True` tools form concurrent batches on the
bounded worker pool; serial tools are exclusive barriers. Results retain the
model's original call order. During execution, `get_current_tool_call()` exposes the active call
to tools that need to correlate transient progress with a UI entry.

### `ToolCallNode`

Runs pending tool calls inside a flow: executes them, stores results under
`state[results_key]`, appends `role: tool` messages to the active context
scope (or to `state[messages_key]` when no context is active), and emits
`tool.call` / `tool.result` events.

### `ToolCall` / `ToolResult`

Frozen records for one requested invocation and its outcome.
`ToolCall.from_openai_item` parses leniently — malformed argument JSON
becomes `{}` so the tool can report a precise validation error back to the
model. `ToolResult.to_message()` yields the OpenAI-style tool message.

## Tracing

### `TraceOptions` / `make_trace_options(...)`

Select which run events are collected, forwarded to `on_event`, or printed.
Pass `trace=True` for defaults, or build options with categories from
`{"flow", "node", "tool", "model"}`.
`format_trace_event(event)` renders one event as a log line.

## Events emitted by the runtime

| Event | Category | Emitted by |
| --- | --- | --- |
| `node.start` / `node.end` / `node.error` | `node` | `Flow` |
| `flow.end` / `flow.error` | `flow` | `Flow` |
| `model.request` / `model.response` | `model` | `ModelNode` |
| `model.delta` / `model.reasoning.delta` | `model` | streaming callback (live only, not retained) |
| `model.request.payload` / `model.response.payload` | `model` | `ModelNode` (observe-only) |
| `tool.observe` | `tool` | `ToolRouterNode` |
| `tool.call` / `tool.result` | `tool` | `ToolCallNode` (metadata only) |
| `tool.call.payload` / `tool.result.payload` | `tool` | `ToolCallNode` (observe-only) |
| `message.add` | `message` | `RunContext.add_message` |
| `artifact.set` | `artifact` | `RunContext.set_artifact` |
