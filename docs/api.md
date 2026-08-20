# API Reference

Everything documented here is importable from the top-level `agent_core`
package and covered by `agent_core.__all__`. Anything else is internal.

`agent_core.__version__` holds the installed distribution version.

Package diagnostics:

```powershell
python -m agent_core --version
python -m agent_core check
python -m agent_core chat [--instructions TEXT] [--max-steps N] [--no-stream]
```

The local check exercises `CallableNode`, `Flow`, and `RunContext` without
credentials or a model request. `chat` starts an interactive REPL built from
`Agent` + `LLM`; model configuration comes from the `LLM_*` environment
variables, exactly as with the library API.

## Core

### `Node`

One unit of work. Subclass and implement `exec(payload) -> (action, payload)`.
For native async work, override `aexec(payload)`; the default `aexec` runs
`exec` in a worker thread so existing nodes also work under `Flow.arun`.
Wire edges with the DSL: `node - "action" >> next_node`, or programmatically
with `node.add_edge("action", next_node)` / `node >> next_node` for the
`"default"` action. An action with no successor ends the flow. Constructor
options: `max_retries`, `wait` (retry `exec` on exception).

- Action selection (`node - "action"`) is stateless: it produces a binding
  used by the next `>>` and never mutates the node.
- Wiring the same action twice raises `ValueError`.
- The right side of `>>` must be a `Node`, otherwise `TypeError`.

### `CallableNode(fn, *, route_plain_tuples=False)`

Adapts a plain function into a node. If `fn(payload)` returns
`ExecResult(action, payload)` it is used as-is. Plain two-item tuples are
business payloads and become `("default", value)`; this avoids guessing that
arbitrary tuple-shaped data is routing control. Pass
`route_plain_tuples=True` only as a temporary 0.1.x migration switch. Legacy
tuple routing emits `DeprecationWarning`.

Both synchronous and `async def` functions are supported by `Flow.arun`.
Calling a `CallableNode` backed by an async function through `Flow.run` raises
a clear error instead of leaking an unawaited coroutine.

### `ExecResult`

Explicit `(action, payload)` routing result. It is a `tuple` subclass, so
`action, payload = result` keeps working while routing intent is unambiguous.

### `Flow(start)`

Routes payloads between nodes by action name.

```python
result = Flow(start_node).run(payload, max_steps=100, trace=None, context=None, cancel=None)
result = await Flow(start_node).arun(payload, max_steps=100, trace=None, context=None, cancel=None)
```

Raises `FlowError` when the flow has no start node or exceeds `max_steps`.
Nested flows share one counter: every visited inner node is debited from the
enclosing run's `max_steps` budget (in addition to each flow's local cap).

`run(cancel=...)` accepts a `threading.Event`; `arun(cancel=...)` accepts a
`threading.Event` or `asyncio.Event`. They are checked cooperatively between
steps (and tool calls), and a cancelled run emits `flow.cancel` and raises
`FlowCancelled`. Nested runs inherit the enclosing event. Cancelling an
`arun` task propagates through native async nodes immediately. A synchronous
body already executing in a worker thread cannot be forcibly interrupted.

### `PayloadKeys`

Canonical payload state keys used by the built-in nodes: `INPUT`, `ANSWER`,
`ASSISTANT_MESSAGE`, `HISTORY`, `CHAT_KWARGS`, `TOOL_RESULTS`,
`TOOLS_ENABLED`, and `LOOP_GUARD`. Custom flows can use these names instead
of repeating literal strings when touching the built-in contract. They
centralize names but do not validate arbitrary custom payload mappings.
`Agent.chat` does validate that its final payload contains `ANSWER` and raises
`FlowError` with a migration hint instead of silently returning an empty
string; use `Agent.run` for other result shapes.

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

Retention policy: transient streams such as model deltas and tool progress
travel through `notify` (live only). `message.add`, `tool.call`, and
`tool.result` retain metadata by default; full model/tool payloads travel
through `observe`. Pass `ToolCallNode(retain_event_payloads=True)` only when a
host explicitly needs complete arguments/results retained in events. Trace
collection captures notify events of the run regardless of retention. Tool
arguments/results still live in canonical conversation messages when needed
for model continuity; this avoids duplicate copies rather than promising
end-to-end secret redaction.
- `set_artifact(name, value)`, `record_model_usage(usage)`, `to_dict()`.

Inside a node, `get_current_context()` returns the active run's context (or
`None` outside a run). `set_current_context` / `reset_current_context` are
low-level hooks for embedding the runtime.

### `RunUsage`

Cumulative model usage. `snapshot()` copies the current counters;
`since(previous)` returns a delta; `to_dict()` reports exact token totals
only when every response carried usage, otherwise `None` instead of a
misleading partial sum. `cached_tokens` is normalized from common
OpenAI-compatible and Anthropic cache-usage fields when available.

### `AgentEvent` / `TraceEvent`

Frozen event record: `type`, `category`, `run_id`, `seq`, `step`, `node`,
`action`, `data`, `timestamp`. `seq` is monotonic within one `RunContext`.
`TraceEvent` is an alias of `AgentEvent`.

## Agent

### `Agent`

An agent that runs a flow — and is itself a `Node`, so agents compose into
larger flows.

```python
Agent(model=..., instructions=..., tools=[...], loop_guard=True)  # standard chat loop
Agent(Flow(...), instructions=...)                # custom loop
```

- `chat(text, *, content=None, context=None, max_steps=None, trace=None,
  stream=None, on_delta=None, payload=None, cancel=None) -> str` — one user turn; reuse
  `context` to hold a conversation. Pass OpenAI-style multimodal `content`
  while keeping `text` as the flow's business input.
- `achat(...) -> str` — async counterpart that uses native async model/tool
  paths and accepts either a `threading.Event` or `asyncio.Event` for
  cooperative cancellation.
- `run(payload, *, max_steps=None, trace, context, cancel=None) -> FlowRunResult` — run the
  inner flow on a payload. `max_steps` defaults to the constructor budget.
  An `Agent` running inside an outer flow shares its counter: each inner node
  visit consumes one of the outer run's remaining steps.
- `arun(...) -> FlowRunResult` — asynchronous counterpart of `run`.
- `new_context() -> RunContext` — fresh context carrying this agent's
  message scope and instructions.
- As a node, an agent exposes its inner flow's final action; pass
  `action="..."` to force a fixed outward action instead.

Each `Agent` keeps an isolated message scope inside a shared `RunContext`:
events, usage, artifacts, and metadata are unified across a multi-agent flow
while prompts never leak between agents.

The standard tool loop enables `ToolLoopGuardNode` by default. Repeated exact
call/result rounds first produce a warning message, then disable tools for a
final text-only model response. Set `loop_guard=False` only when the enclosing
workflow supplies its own no-progress policy.

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

### `AsyncChatModel` (protocol)

Optional runtime-checkable capability consumed by `ModelNode.aexec`:

```python
async def achat_message(messages, *, tools=None, tool_choice=None, **kwargs) -> dict
```

`Agent` and `ModelNode` accept either protocol, or an adapter implementing
both. Under `Flow.arun`, a sync-only `ChatModel` is automatically called in a
worker thread. Calling a purely async adapter through `Flow.run` reports that
`arun` / `achat` is required.

### `LLM`

Default OpenAI-compatible adapter. Explicit arguments win; otherwise
configuration comes from `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` (or the
`OPENAI_*` / `DEEPSEEK_*` aliases). Supports `stream=True` with an
`on_delta` text callback, and captures usage in both modes. It implements
both `chat_message` and `achat_message`, using `OpenAI` / `AsyncOpenAI`
clients respectively. Streaming tool-call fragments are merged by explicit
index when present and by call ID boundaries otherwise; missing or duplicate
IDs are normalized before execution.

### `ModelNode`

Calls the model once and stores the assistant message in
`state[assistant_key]`. Messages come from an explicit builder, the active
context's scope, or `state[messages_key]` — in that order. The context scope
is the single canonical history during a flow run; `state[messages_key]` is
only an import seed for custom flows and no longer mirrors new messages. A
custom builder receives the canonical scope projected under `messages_key`,
so later tool-loop calls still include assistant and tool messages.
Per-call overrides travel in `state[chat_kwargs_key]`. Emits
`model.request` / `model.response` and records usage. `aexec` uses
`AsyncChatModel` when available and otherwise offloads the synchronous model
call. A dynamic tool provider can inspect payload state; the standard agent
loop uses `PayloadKeys.TOOLS_ENABLED` to force its final text-only request.

An `Agent` adopts unscoped ambient messages (added to the context outside
any scope) into its own scope at run start, so a conversation seeded through
`context.add_message` without a scope is visible to the model while other
agents' scoped messages stay isolated.

### `ToolRouterNode`

Routes on the assistant message: `tool_action` when tool calls are pending,
otherwise stores the text under `state[output_key]` and returns
`done_action`.

### `ToolLoopGuardNode`

A regular routable node that detects repeated exact tool call/result rounds.
Its `continue`, `warn`, and `halt` actions can be wired anywhere in a custom
flow. The default window is three rounds; warning appends an internal system
message, and a repeated call after warning also sets
`state[PayloadKeys.TOOLS_ENABLED] = False`. Guard history is explicit under
`state[PayloadKeys.LOOP_GUARD]`, not hidden process state.

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
model's original call order. `aexecute` / `aexecute_all` await async tools and
offload synchronous tools while preserving the same batching rules. Missing
or duplicate model call IDs are normalized back into the assistant message so
tool-result messages remain pairable. During execution,
`get_current_tool_call()` exposes the active call to tools that need to
correlate progress with a UI entry.

`report_tool_progress(value) -> bool` publishes progress through the current
executor without adding an argument to the tool schema. `ToolCallNode`
forwards it as live-only `tool.progress`; the return value is false when no
progress subscriber exists.

### `ToolCallNode`

Runs pending tool calls inside a flow: executes them, stores results under
`state[results_key]`, appends `role: tool` messages to the active context
scope (or to `state[messages_key]` when no context is active), and emits
`tool.call` / `tool.result` events. Metadata-only retained events and
observe-only `tool.call.payload` / `tool.result.payload` details are the safe
default. `retain_event_payloads=True` explicitly restores the 0.1.x full
retained event data. Its `aexec` path uses `ToolExecutor.aexecute_all`.

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
| `node.start` / `node.end` / `node.error` / `node.cancel` | `node` | `Flow` |
| `flow.end` / `flow.error` / `flow.cancel` | `flow` | `Flow` |
| `model.request` / `model.response` | `model` | `ModelNode` |
| `model.delta` / `model.reasoning.delta` | `model` | streaming callback (live only, not retained) |
| `model.request.payload` / `model.response.payload` | `model` | `ModelNode` (observe-only) |
| `tool.observe` | `tool` | `ToolRouterNode` |
| `tool.call` / `tool.result` | `tool` | `ToolCallNode` (metadata-only by default) |
| `tool.call.payload` / `tool.result.payload` | `tool` | `ToolCallNode` (observe-only with the default retention policy) |
| `tool.progress` | `tool` | `report_tool_progress` through `ToolCallNode` (live only) |
| `loop.warning` / `loop.guard` | `runtime` | `ToolLoopGuardNode` |
| `message.add` | `message` | `RunContext.add_message` (metadata-only) |
| `artifact.set` | `artifact` | `RunContext.set_artifact` |
