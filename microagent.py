#!/usr/bin/env python3
"""
microagent — test-first AI coding agent

Usage:
    python microagent.py "Write a function that reverses a list"
    python microagent.py "Write a Fibonacci function" --task-dir ./fib-task
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import anthropic

from agent import AgentLoop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test-first AI coding agent powered by Claude"
    )
    parser.add_argument("prompt", help="Description of the function/task to implement")
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Directory for task files (default: ./task-YYYYMMDD-HHMMSS/)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Max implementation attempts before giving up (default: 10)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # Resolve task directory
    if args.task_dir:
        task_dir = Path(args.task_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_dir = Path(f"task-{timestamp}")

    task_dir.mkdir(parents=True, exist_ok=True)
    print(f"Task directory: {task_dir.resolve()}")

    # Copy .prompt.md from cwd if present
    prompt_md_src = Path(".prompt.md")
    if prompt_md_src.exists():
        dest = task_dir / "solution.prompt.md"
        shutil.copy(prompt_md_src, dest)
        print(f"Copied .prompt.md → {dest}")

    client = anthropic.Anthropic(api_key=api_key)
    loop = AgentLoop(
        client=client,
        task_dir=task_dir,
        model=args.model,
        max_iterations=args.max_iterations,
    )

    try:
        loop.run(args.prompt)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
