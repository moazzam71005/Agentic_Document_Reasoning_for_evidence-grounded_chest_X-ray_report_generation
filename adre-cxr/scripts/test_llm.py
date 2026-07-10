#!/usr/bin/env python3
"""Quick test that LLM cloud API is configured."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.client import LLMClient


def main() -> None:
    llm = LLMClient()
    reply = llm.chat(
        [{"role": "user", "content": "Reply with exactly: ADRE-CXR OK"}],
        temperature=0.0,
    )
    print(f"Provider: {llm.provider} | Model: {llm.model}")
    print(f"Response: {reply.strip()}")


if __name__ == "__main__":
    main()
