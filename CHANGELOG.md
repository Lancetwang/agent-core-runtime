# Changelog

All notable changes to `friday-agent-core` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.5] - 2026-07-31

### Changed

- Retain `reasoning_content` in `RunContext` only for assistant tool-call
  messages that need it on the next request, avoiding needless context growth
  after final answers.

## [0.1.4] - 2026-07-31

### Fixed

- Preserve provider `reasoning_content` in streamed and non-streamed assistant
  messages so thinking models can continue tool-call conversations correctly.

## [0.1.3] - 2026-07-29

### Added

- OpenAI-compatible multimodal user-message content support.

## [0.1.2] - 2026-07-29

### Added

- `python -m agent_core --version` for installed-version inspection.
- `python -m agent_core check` for a credential-free local runtime smoke test.

## [0.1.1] - 2026-07-26

### Added

- `agent_core.__version__`, resolved from the installed distribution.
- `ChatModel` and `Message` are now exported from the top-level `agent_core` package.
- PEP 561 `py.typed` marker: consumers get full type checking for the public API.
- Docstrings across the public API (`Node`, `Flow`, `Agent`, `ModelNode`,
  `ToolRouterNode`, `ToolCallNode`, `Tool`, `tool`, `ToolExecutor`,
  `RunContext` methods, `TraceOptions`).
- CI workflow running tests on every push and pull request.

### Changed

- Actionable error messages throughout: unknown tools list the available tool
  names, tool definition errors name the offending function and parameter,
  `max_steps` overrun suggests the likely causes, and tool execution failures
  include the exception type.
- `Flow.run` on a flow without a start node now raises `FlowError` instead of
  silently returning an empty result.
- Wiring the same action of a node to a second successor now raises
  `ValueError` instead of silently replacing the first edge.
- `Node >> other` validates that `other` is a `Node`.

## [0.1.0] - 2026-07-26

### Added

- First PyPI release as `friday-agent-core` (import name `agent_core`).
- Runtime primitives: `Node`, `Flow`, `Agent` (an agent is a node),
  `RunContext` with scoped messages, events, artifacts, metadata, and exact
  usage accounting.
- Built-in model/tool chat loop: `ModelNode`, `ToolRouterNode`,
  `ToolCallNode`, `ToolExecutor`, and the `@tool` schema decorator.
- OpenAI-compatible `LLM` adapter with streaming and usage capture.
- Trace options for filtering, forwarding, and printing run events.
