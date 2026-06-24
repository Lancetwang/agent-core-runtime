# Agent Core Runtime Design

`agent_core` is a generic runtime package. It should not contain application pipelines, domain prompts, domain tools, web UI code, or storage choices.

The package does include a small OpenAI-compatible adapter because most users expect a cloned runtime to run after they add an API key. The adapter lives in `agent_core.llm` and stays thin: it converts messages, tools, streaming deltas, and usage into plain dictionaries.

## Runtime Model

- `Node` owns one unit of work and returns `(action, payload)`.
- `Flow` routes each action to at most one next node.
- `Agent` wraps a flow and is itself a node.
- `RunContext` belongs to one execution and carries messages, artifacts, metadata, and UI-friendly events.

Plain payload dictionaries are still useful for node-to-node arguments. Agent state that needs to survive across nodes or turns should live in `RunContext`.

## Built-In Tool Loop

The common chat loop is:

```text
ModelNode -> ToolRouterNode
              | tool_call -> ToolCallNode -> ModelNode
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
