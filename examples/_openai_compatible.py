import sys

from agent_core import build_model_from_env


def safe_print(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
