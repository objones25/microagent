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
from logger import setup_logging


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
    parser.add_argument(
        "--prompts",
        default="v2.6",
        metavar="VERSION",
        help="Prompts version to load from prompts/<VERSION>.toml (default: v2.6)",
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
        loop.run(args.prompt)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
