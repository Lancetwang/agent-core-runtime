from __future__ import annotations

import argparse

from agent_core import CallableNode, Flow, __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_core")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("command", nargs="?", choices=["check"], help="Run a local runtime smoke test.")
    args = parser.parse_args(argv)
    if args.command != "check":
        parser.print_help()
        return 0

    result = Flow(CallableNode(lambda payload: {**payload, "ok": True})).run({})
    if result.payload != {"ok": True} or not result.context.run_id:
        print(f"friday-agent-core {__version__}: FAILED")
        return 1
    print(f"friday-agent-core {__version__}: OK (import, node, flow, context)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
