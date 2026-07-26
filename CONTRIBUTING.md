# Contributing

Thanks for your interest in `friday-agent-core` (import name `agent_core`).

## Development setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/Lancetwang/agent-core-runtime.git
cd agent-core-runtime
uv sync
```

Model-backed examples need credentials in the environment or a local `.env`
(see `.env.example`). The test suite uses fakes and needs no API key.

## Running checks

```powershell
uv run python -m unittest discover -s tests
uv run python -m compileall src tests examples
```

Both must pass before a change is merged. CI runs the same commands on every
push and pull request.

## Design ground rules

Read `docs/agent-core.md` and the "Scope and Boundary" section of the README
before proposing features. In short:

- The runtime owns **execution**: nodes, flows, model/tool calls, and the
  per-run `RunContext`. It stays small enough to read in one sitting.
- Sessions, memory, compaction, permissions, prompts, and UI belong to
  harnesses built on top (see [Friday](https://github.com/Lancetwang/friday)),
  not here. Features that pull product policy into the runtime will be
  declined.
- Business data moves through node payloads; runtime state (messages, events,
  usage, artifacts) lives in `RunContext`. Keep the two separate.
- Every public API needs a docstring, a test, and an error message that tells
  the user what to do, not just what went wrong.

## Compatibility

The public API is everything exported from `agent_core.__all__`. Behavior
changes to those names require a version bump and a CHANGELOG entry. Keep
error message *meanings* stable — downstream tests may assert on key phrases.

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml`, update `CHANGELOG.md`, commit.
2. Tag and push: `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`.
3. The `release.yml` workflow tests, builds, and publishes to PyPI via
   trusted publishing (no tokens involved).
4. Update the pin in downstream consumers (friday, ft-agent) and run their
   test suites.
