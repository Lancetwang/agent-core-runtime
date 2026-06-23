# Agent Core Runtime Design

This branch is a standalone runtime branch. It contains only generic agent runtime primitives under `agent_core`.

## Package Boundary

`agent_core` must stay application-agnostic:

- no application pipeline code
- no model-provider SDK dependency
- no environment configuration management
- no web server or frontend code
- no domain-specific prompts or tools

Applications should import `agent_core` and provide their own model adapters, prompts, tools, storage, and UI.

## Runtime Model

The runtime keeps a node-flow shape:

- `Node` owns one unit of work and returns `(action, payload)`.
- `Flow` routes each action to at most one next node.
- `Agent` runs a flow.
- `RunContext` belongs to one execution and carries runtime state that should not be hidden inside a business payload.

Existing nodes can use plain payload dictionaries. Richer nodes can call `get_current_context()` while running to add messages, artifacts, metadata, or UI-friendly events.

`TraceEvent` is the debug/logging layer. `AgentEvent` is the product-level event stream intended for observers and frontends.

## Standard Tool Loop

The runtime includes a provider-neutral tool-agent loop:

```text
ModelNode -> ToolRouterNode
              | tool_call -> ToolCallNode -> ModelNode
              | final     -> flow end
```

`ModelNode` only requires a `ChatModel` implementation. The protocol expects `chat_message(...)` to return an assistant message in an OpenAI-style shape:

```python
{
    "role": "assistant",
    "content": "...",
    "tool_calls": [...],
}
```

This shape is intentionally generic and can be produced by any model adapter.

`build_tool_agent_flow(...)` is a convenience helper for the common loop. Users can still wire nodes manually when they need a different structure.
