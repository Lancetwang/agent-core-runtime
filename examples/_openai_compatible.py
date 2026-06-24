from __future__ import annotations

import sys
from pathlib import Path

from agent_core import OpenAICompatibleChatModel, build_model_from_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def build_demo_model() -> OpenAICompatibleChatModel:
    """Build the default OpenAI-compatible model used by real examples."""

    return build_model_from_env(env_file=ENV_FILE)


def safe_print(text: object = "") -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
