from __future__ import annotations

import argparse
import sys

from agent_core import Agent, CallableNode, Flow, FlowError, LLM, __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_core")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    check_parser = subparsers.add_parser(
        "check",
        help="Run a local runtime smoke test.",
    )
    check_parser.set_defaults(func=_run_check)

    chat_parser = subparsers.add_parser(
        "chat",
        description="Start an interactive chat REPL using LLM_* environment configuration.",
        help="Start an interactive chat REPL.",
    )
    chat_parser.add_argument(
        "--instructions",
        default=None,
        help="System prompt for the agent (default: a short built-in prompt).",
    )
    chat_parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum flow steps per turn (default: 20).",
    )
    chat_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output.",
    )
    chat_parser.set_defaults(func=_run_chat)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)


def _run_check(args: argparse.Namespace) -> int:
    result = Flow(CallableNode(lambda payload: {**payload, "ok": True})).run({})
    if result.payload != {"ok": True} or not result.context.run_id:
        print(f"friday-agent-core {__version__}: FAILED")
        return 1
    print(f"friday-agent-core {__version__}: OK (import, node, flow, context)")
    return 0


def _run_chat(args: argparse.Namespace) -> int:
    try:
        model = LLM()
    except RuntimeError as exc:
        print(f"agent_core chat: {exc}")
        return 1

    instructions = args.instructions or "You are a helpful assistant. Answer concisely."
    stream = not args.no_stream

    def print_delta(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    agent = Agent(
        model=model,
        instructions=instructions,
        stream=stream,
        max_steps=args.max_steps,
    )
    context = agent.new_context()
    print(f"agent_core chat using model '{model.model}'. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if user_input.lower() in {"exit", "quit", "q"}:
            print("bye")
            return 0
        if not user_input:
            continue
        try:
            answer = agent.chat(
                user_input,
                context=context,
                on_delta=print_delta if stream else None,
            )
        except KeyboardInterrupt:
            print("\n(cancelled)")
            continue
        except FlowError as exc:
            print(f"\nerror: {exc}")
            continue
        if stream:
            print()
        else:
            print(answer)


if __name__ == "__main__":
    raise SystemExit(main())
