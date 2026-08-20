# Agent Core Runtime Design

`agent_core` is a generic runtime package. It should not contain application pipelines, domain prompts, domain tools, web UI code, or storage choices.

The package includes a small OpenAI-compatible `LLM`. Applications own configuration loading and pass constructor arguments or process environment variables; the runtime converts messages, tools, streaming deltas, and usage into plain dictionaries.

## Runtime Model

- `Node` owns one unit of work and returns `(action, payload)`. A node may
  implement synchronous `exec`, asynchronous `aexec`, or both.
- `Flow` routes each action to at most one next node and runs the same graph
  through `run` or `arun`.
- `Agent` wraps a flow and is itself a node.
- `RunContext` is caller-owned and carries scoped messages, one event stream, cumulative model usage, artifacts, and metadata. Reuse it across turns for a conversation, or create a new one for an isolated invocation.

The payload and context are deliberately separate:

- The payload is the explicit business data passed from node to node and returned in `FlowRunResult.payload`.
- `RunContext` is the runtime surface for conversation messages, emitted events, model usage, artifacts, metadata, and current execution position.

Use payload for routing and domain state. Use `RunContext` for things a UI, CLI, logger, or chat session needs to observe or preserve. Large artifacts should live outside the runtime payload, with payload/context carrying paths, IDs, summaries, or metadata.

In nested or multi-agent flows, the run context is shared for events, usage, artifacts, and metadata. LLM input messages are scoped per `Agent`, so one role's prompt and chat history do not leak into another role's model call.

## Built-In Tool Loop

The common chat loop is:

```text
ModelNode -- observe --> ToolRouterNode
                         | tool_call -> ToolCallNode -> ToolLoopGuardNode
                                                        | continue/warn/halt
                                                        v
                                                    ModelNode
                         | final -> flow end
```

`Agent(model=..., instructions=..., tools=...)` builds that loop for the common
case. The loop guard is itself an ordinary node whose state lives in the
payload. It warns after repeated identical call/result rounds, then disables
the dynamic tool provider and routes to one final text-only model request.
Applications with a different policy can pass `loop_guard=False` or wire a
`ToolLoopGuardNode` into a custom graph.

If an application needs a different loop, it can create a `Flow` directly and pass it to `Agent(Flow(...))`.

## Model Boundary

`ChatModel` is a thin seam normalized to the OpenAI wire format — messages,
tool schemas, `tool_choice`, and streaming options keep their OpenAI shapes,
and the assistant message it returns is an OpenAI-style dict:

```python
{
    "role": "assistant",
    "content": "...",
    "tool_calls": [...],
    "usage": {...},
}
```

This is deliberately not a provider abstraction layer: "replaceable" means
another OpenAI-compatible endpoint, or a one-method adapter that translates
a native provider schema at this boundary. Provider-specific knobs travel
as plain data through `extra_body` / `chat_kwargs`.

The default OpenAI-compatible adapter requests usage in streaming mode and normalizes it into the assistant message. `ModelNode` records each response in `RunContext.usage`, while `FlowRunResult.usage` exposes the delta for that flow invocation. Applications do not need to rescan trace events to calculate turn totals.

`AsyncChatModel` is an optional capability and may be implemented on its own
for async-only applications. `ModelNode.aexec` awaits it when present and
otherwise moves a synchronous `ChatModel` request to a worker thread. A sync
entry point reports a clear error for an async-only adapter. The bundled
`LLM` implements both paths using `OpenAI` and `AsyncOpenAI` clients.

## Runtime Signals

Runtime signals have three explicit delivery semantics:

- `emit` retains compact events and delivers them to both subscribers.
- `observe` delivers potentially sensitive or large details without retaining
  another copy in runtime memory.
- `notify` delivers transient UI progress, such as model deltas and
  `tool.progress`, without retention.

Every delivered event receives a monotonic `seq` inside its `RunContext`.
Canonical conversation messages retain the data needed by the next model
request; their `message.add` events retain metadata only. Tool arguments and
results follow the same rule unless full retained event payloads are
explicitly requested.

## Ownership Boundary

The runtime deliberately does not load `.env`, choose a provider, define
domain tools, persist sessions, compact conversations, or implement product
approval policy. Applications own those decisions and pass ordinary model
configuration, tools, payloads, and contexts into the runtime.

Core provides equivalent synchronous and asynchronous graph execution.
Native async nodes/models/tools stay on the event loop; synchronous work is
offloaded under `arun`, and parallel tool batches remain bounded. Cancellation
is cooperative at node/tool boundaries, while task cancellation propagates
through native async work. A synchronous body already running in a worker
thread cannot be forcibly stopped. The same `RunContext` must not be driven
concurrently by multiple flows.
