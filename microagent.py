#!/usr/bin/env python3
"""
microagent — test-first AI coding agent

Usage:
    python microagent.py "Write a function that reverses a list"
    python microagent.py "Write a Fibonacci function" --task-dir ./fib-task
    python microagent.py "Write a reverse string function" --prompts v1
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

import db
from agent import AgentLoop, AgentConfig
from config import DEFAULT_MODEL, DEFAULT_PROMPTS_VERSION, DEFAULT_MAX_ITERATIONS
from logger import setup_logging


def _render_event(event: dict) -> None:
    """Render a single AgentEvent to the terminal (non-interactive events only)."""
    match event["type"]:
        case "done":
            print("\n" + "=" * 60)
            if event["success"]:
                print("SUCCESS — all tests passed!")
            else:
                print("FAILED — could not pass all tests.")
            print("=" * 60)
            if event.get("message"):
                print(event["message"])


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
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Max implementation attempts before giving up (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--prompts",
        default=DEFAULT_PROMPTS_VERSION,
        metavar="VERSION",
        help=f"Prompts version to load from prompts/<VERSION>.toml (default: {DEFAULT_PROMPTS_VERSION})",
    )
    parser.add_argument(
        "--allow-test-revision",
        action="store_true",
        help="When the agent stops without passing tests, offer it a chance to revise solution_test.py (requires user approval unless --auto-approve-revision is set)",
    )
    parser.add_argument(
        "--auto-approve-revision",
        action="store_true",
        help="Automatically approve agent test revisions without prompting (use with --allow-test-revision)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Minimum test coverage %% required to pass (0 = disabled, default: 0)",
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
    logger = setup_logging(task_dir)
    logger.info(f"Task directory: {task_dir.resolve()}")

    # Copy .prompt.md from cwd if present
    prompt_md_src = Path(".prompt.md")
    if prompt_md_src.exists():
        dest = task_dir / "solution.prompt.md"
        shutil.copy(prompt_md_src, dest)
        logger.info(f"Copied .prompt.md → {dest}")

    conn = db.get_db()
    db.init_db(conn)
    db.seed_if_empty(conn)

    client = anthropic.Anthropic(api_key=api_key)
    config = AgentConfig(
        model=args.model,
        max_iterations=args.max_iterations,
        prompts_version=args.prompts,
        allow_test_revision=args.allow_test_revision,
        auto_approve_revision=args.auto_approve_revision,
        min_coverage=args.min_coverage,
    )
    loop = AgentLoop(
        client=client,
        task_dir=task_dir,
        config=config,
        logger=logger,
        db_conn=conn,
    )

    try:
        for event in loop.run(args.prompt):
            if event["type"] == "awaiting_approval":
                print("\n" + "=" * 60)
                print("GENERATED TESTS (solution_test.py):")
                print("=" * 60)
                print(event["content"])
                print("=" * 60)
                try:
                    input("\nPress Enter to start implementation, or Ctrl+C to abort...\n")
                except KeyboardInterrupt:
                    print("\nAborted.")
                    sys.exit(0)
            else:
                _render_event(event)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
