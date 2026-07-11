# Agent Core Runtime Design

`agent_core` is a generic runtime package. It should not contain application pipelines, domain prompts, domain tools, web UI code, or storage choices.

The package includes a small OpenAI-compatible `LLM`. Applications own configuration loading and pass constructor arguments or process environment variables; the runtime converts messages, tools, streaming deltas, and usage into plain dictionaries.

## Runtime Model

- `Node` owns one unit of work and returns `(action, payload)`.
- `Flow` routes each action to at most one next node.
- `Agent` wraps a flow and is itself a node.
- `RunContext` belongs to one execution and carries scoped messages, one event stream, cumulative model usage, artifacts, and metadata.

The payload and context are deliberately separate:

- The payload is the explicit business data passed from node to node and returned in `FlowRunResult.payload`.
- `RunContext` is the runtime surface for conversation messages, emitted events, model usage, artifacts, metadata, and current execution position.

Use payload for routing and domain state. Use `RunContext` for things a UI, CLI, logger, or chat session needs to observe or preserve. Large artifacts should live outside the runtime payload, with payload/context carrying paths, IDs, summaries, or metadata.

In nested or multi-agent flows, the run context is shared for events, usage, artifacts, and metadata. LLM input messages are scoped per `Agent`, so one role's prompt and chat history do not leak into another role's model call.

## Built-In Tool Loop

The common chat loop is:

```text
ModelNode -- observe --> ToolRouterNode
                         | tool_call -> ToolCallNode -- chat --> ModelNode
                         | final     -> flow end
```

`Agent(model=..., instructions=..., tools=...)` builds that loop for the common case.

If an application needs a different loop, it can create a `Flow` directly and pass it to `Agent(Flow(...))`.

## Model Boundary

`ChatModel` is the provider-neutral protocol. It returns assistant messages in an OpenAI-style shape:

```python
{
    "role": "assistant",
    "content": "...",
    "tool_calls": [...],
    "usage": {...},
}
```

Any model provider can be used by implementing this small protocol.

The default OpenAI-compatible adapter requests usage in streaming mode and normalizes it into the assistant message. `ModelNode` records each response in `RunContext.usage`, while `FlowRunResult.usage` exposes the delta for that flow invocation. Applications do not need to rescan trace events to calculate turn totals.

## Ownership Boundary

The runtime deliberately does not load `.env`, choose a provider, define domain tools, persist sessions, compact conversations, or implement product approval policy. Applications own those decisions and pass ordinary model configuration, tools, payloads, and contexts into the runtime.
