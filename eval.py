#!/usr/bin/env python3
"""
eval.py — Evaluation harness for microagent.

Runs a suite of harder coding tasks through the agent, collects structured results,
then passes everything to a Claude judge for analysis and prompt improvement suggestions.

Usage:
    python eval.py                       # run V2 prompts on all tasks
    python eval.py --ab-test             # run V1 vs V2 on all tasks and compare
    python eval.py --tasks 5             # run first N tasks only
    python eval.py --max-iter 5          # limit implementation iterations per task
    python eval.py --out results.json    # save raw results to file
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import anthropic

from agent import AgentLoop

# ------------------------------------------------------------------
# Harder task suite (designed to stress-test the agent)
# ------------------------------------------------------------------

TASKS = [
    # Algorithms
    "Write a function that finds the longest common subsequence of two strings and returns the subsequence itself",
    "Write a function that solves the 0/1 knapsack problem given weights, values, and a capacity",
    "Write a function that finds all valid parenthesizations of n pairs of parentheses",
    # Data structures
    "Write an LRUCache class with get and put methods using only a dict and a doubly-linked list — no OrderedDict",
    "Write a function that serializes a binary tree to a string and deserializes it back to a tree",
    # String/text processing
    "Write a function that implements run-length encoding and a companion function that decodes it",
    "Write a function that groups anagrams together from a list of strings",
    # Concurrency / correctness
    "Write a TokenBucket rate limiter class with an allow() method that returns True if a request is allowed",
    # Math / number theory
    "Write a function that finds the longest increasing subsequence of a list of integers and returns its length",
    # Parsing
    "Write a function that evaluates simple arithmetic expressions given as strings (supporting +, -, *, / and parentheses) without using eval()",
]

# ------------------------------------------------------------------
# Judge prompts
# ------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an expert AI systems evaluator specialising in LLM-powered coding agents \
and prompt engineering. You give specific, direct, actionable feedback.
"""

JUDGE_SINGLE_TEMPLATE = """\
Below are results from running a two-phase AI coding agent on {n} tasks.

## Agent workflow
1. A first LLM call generates a locked pytest test file from the task description
2. A second LLM (with tools) iteratively writes solution.py and runs pytest until all tests pass
3. Max {max_iter} write→run cycles allowed per task

## System prompts used

**TEST_GENERATION_SYSTEM:**
```
{test_gen_system}
```

**IMPLEMENTATION_SYSTEM:**
```
{impl_system}
```

## Task results

{results_block}

---

Evaluate these dimensions and give specific, actionable recommendations:

### 1. Test generation quality
- Do tests cover happy path, edge cases, error handling, and return types?
- Are any tests tautological (always pass regardless of implementation)?
- Systematic gaps across tasks?

### 2. Implementation quality
- Are solutions general-purpose or do any hardcode values to pass specific tests?
- For named algorithms/techniques, was the technique implemented correctly?

### 3. Iteration efficiency
- How many iterations on average? What caused retries?

### 4. Failure analysis
- For failed tasks, what went wrong? What would have fixed it?

### 5. Concrete prompt improvements
For each change, quote the **exact current text** then show the **full replacement**.
Focus on highest-leverage changes first.
"""

JUDGE_AB_TEMPLATE = """\
Below are A/B test results comparing two versions of prompts for a two-phase AI coding agent.

## Agent workflow
1. A first LLM call generates a locked pytest test file from the task description
2. A second LLM (with tools) iteratively writes solution.py and runs pytest until all tests pass
3. Max {max_iter} write→run cycles allowed per task

## Prompt versions

### V1 (baseline) — TEST_GENERATION_SYSTEM:
```
{v1_test_gen}
```

### V1 (baseline) — IMPLEMENTATION_SYSTEM:
```
{v1_impl}
```

### V2 (improved) — TEST_GENERATION_SYSTEM:
```
{v2_test_gen}
```

### V2 (improved) — IMPLEMENTATION_SYSTEM:
```
{v2_impl}
```

## V1 Results

{v1_results_block}

## V2 Results

{v2_results_block}

---

Provide a direct A/B comparison:

### 1. Pass rate comparison
V1 vs V2 — which passed more tasks? Why?

### 2. Test quality comparison
Did V2 generate better tests? Were the added rules (falsifiability, single contract,
breadth guidelines, memoization checks) effective? Cite specific examples.

### 3. Implementation quality comparison
Did V2 produce better implementations? Did the memoization/named-algorithm rule matter?

### 4. Iteration efficiency
Which version needed fewer retries? Why?

### 5. Verdict
Is V2 a clear improvement? Which changes had the most impact? Which had little or no effect?

### 6. Next iteration — prompt improvements for V3
Based on the V2 results and any remaining failure modes, what specific changes would you
make to V2 to create V3? Quote exact text and show replacements.
"""

# ------------------------------------------------------------------
# Result collection
# ------------------------------------------------------------------

def run_task(
    client: anthropic.Anthropic,
    task: str,
    task_dir: Path,
    max_iterations: int,
    prompts_module,
) -> dict:
    result = {
        "task": task,
        "task_dir": str(task_dir),
        "passed": False,
        "iterations": 0,
        "test_file": "",
        "solution_file": "",
        "pytest_output": "",
        "error": "",
        "duration_seconds": 0.0,
    }

    start = time.time()
    captured = StringIO()
    original_stdout = sys.stdout

    try:
        loop = AgentLoop(
            client=client,
            task_dir=task_dir,
            max_iterations=max_iterations,
            prompts_module=prompts_module,
        )

        sys.stdout = captured
        test_content = loop.generate_tests(task)
        result["test_file"] = test_content

        success, message = loop.run_implementation_loop(task, test_content)
        result["passed"] = success
        result["pytest_output"] = message

    except Exception as e:
        result["error"] = str(e)
    finally:
        sys.stdout = original_stdout
        result["duration_seconds"] = round(time.time() - start, 1)

    captured_text = captured.getvalue()
    result["iterations"] = captured_text.count("pytest →")

    sol = task_dir / "solution.py"
    if sol.exists():
        result["solution_file"] = sol.read_text()

    return result


def format_results_block(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        status = "PASSED" if r["passed"] else "FAILED"
        error_note = f"\n  error: {r['error']}" if r["error"] else ""

        test_snippet = ""
        if r["test_file"]:
            test_snippet = f"\n\n**solution_test.py:**\n```python\n{r['test_file']}\n```"

        sol_snippet = ""
        if r["solution_file"]:
            lines = r["solution_file"].splitlines()
            preview = "\n".join(lines[:40])
            if len(lines) > 40:
                preview += f"\n  ... ({len(lines) - 40} more lines)"
            sol_snippet = f"\n\n**solution.py:**\n```python\n{preview}\n```"

        pytest_snippet = ""
        if r["pytest_output"]:
            lines = r["pytest_output"].splitlines()
            trimmed = "\n".join(lines[-50:])
            pytest_snippet = f"\n\n**pytest output (last 50 lines):**\n```\n{trimmed}\n```"

        blocks.append(
            f"### Task {i}: {r['task']}\n"
            f"- **Status**: {status}\n"
            f"- **Iterations**: {r['iterations']}\n"
            f"- **Duration**: {r['duration_seconds']}s{error_note}"
            f"{test_snippet}{sol_snippet}{pytest_snippet}"
        )
    return "\n\n---\n\n".join(blocks)


def summary_line(results: list[dict]) -> str:
    passed = sum(1 for r in results if r["passed"])
    total_iter = sum(r["iterations"] for r in results)
    avg = total_iter / len(results) if results else 0
    return f"{passed}/{len(results)} passed, avg {avg:.1f} iterations"


# ------------------------------------------------------------------
# Judges
# ------------------------------------------------------------------

def run_judge_single(
    client: anthropic.Anthropic,
    results: list[dict],
    max_iterations: int,
    prompts_module,
) -> str:
    user_content = JUDGE_SINGLE_TEMPLATE.format(
        n=len(results),
        max_iter=max_iterations,
        test_gen_system=prompts_module.TEST_GENERATION_SYSTEM,
        impl_system=prompts_module.IMPLEMENTATION_SYSTEM,
        results_block=format_results_block(results),
    )
    print("\nCalling judge...", flush=True)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


def run_judge_ab(
    client: anthropic.Anthropic,
    v1_results: list[dict],
    v2_results: list[dict],
    max_iterations: int,
    v1_prompts,
    v2_prompts,
) -> str:
    user_content = JUDGE_AB_TEMPLATE.format(
        max_iter=max_iterations,
        v1_test_gen=v1_prompts.TEST_GENERATION_SYSTEM,
        v1_impl=v1_prompts.IMPLEMENTATION_SYSTEM,
        v2_test_gen=v2_prompts.TEST_GENERATION_SYSTEM,
        v2_impl=v2_prompts.IMPLEMENTATION_SYSTEM,
        v1_results_block=format_results_block(v1_results),
        v2_results_block=format_results_block(v2_results),
    )
    print("\nCalling A/B judge...", flush=True)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_suite(
    client: anthropic.Anthropic,
    tasks: list[str],
    base_dir: Path,
    max_iterations: int,
    prompts_module,
    label: str = "",
) -> list[dict]:
    results = []
    prefix = f"[{label}] " if label else ""
    for i, task in enumerate(tasks, 1):
        task_dir = base_dir / f"task-{i:02d}"
        task_dir.mkdir(parents=True)
        print(f"{prefix}[{i}/{len(tasks)}] {task}", flush=True)
        result = run_task(client, task, task_dir, max_iterations, prompts_module)
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"  → {status} in {result['iterations']} iter(s), {result['duration_seconds']}s\n")
        results.append(result)
    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate microagent on a suite of harder coding tasks")
    parser.add_argument("--ab-test", action="store_true", help="Run V1 vs V2 prompts and compare")
    parser.add_argument("--tasks", type=int, default=len(TASKS), help="Number of tasks to run (default: all)")
    parser.add_argument("--max-iter", type=int, default=5, help="Max implementation iterations per task (default: 5)")
    parser.add_argument("--out", default=None, help="Save raw results JSON to this file")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    tasks = TASKS[: args.tasks]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_dir = Path(f"eval-{timestamp}")
    eval_dir.mkdir()

    import prompts as v2_prompts
    import prompts_v1 as v1_prompts

    if args.ab_test:
        print(f"A/B test: {len(tasks)} tasks × 2 prompt versions → {eval_dir}/")
        print(f"Max iterations per task: {args.max_iter}\n")

        print("=== Running V1 (baseline) ===")
        v1_dir = eval_dir / "v1"
        v1_dir.mkdir()
        v1_results = run_suite(client, tasks, v1_dir, args.max_iter, v1_prompts, label="V1")

        print("=== Running V2 (improved) ===")
        v2_dir = eval_dir / "v2"
        v2_dir.mkdir()
        v2_results = run_suite(client, tasks, v2_dir, args.max_iter, v2_prompts, label="V2")

        print("=" * 60)
        print(f"V1: {summary_line(v1_results)}")
        print(f"V2: {summary_line(v2_results)}")
        print("=" * 60)

        all_results = {"v1": v1_results, "v2": v2_results}
        out_path = Path(args.out) if args.out else eval_dir / "results.json"
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nRaw results saved to: {out_path}")

        judgment = run_judge_ab(client, v1_results, v2_results, args.max_iter, v1_prompts, v2_prompts)

    else:
        print(f"Running {len(tasks)} tasks → {eval_dir}/")
        print(f"Max iterations per task: {args.max_iter}\n")

        results = run_suite(client, tasks, eval_dir, args.max_iter, v2_prompts)

        passed = sum(1 for r in results if r["passed"])
        total_iter = sum(r["iterations"] for r in results)
        print("=" * 60)
        print(f"RESULTS: {passed}/{len(results)} passed")
        print(f"Total iterations: {total_iter} (avg {total_iter/len(results):.1f} per task)")
        print("=" * 60)

        out_path = Path(args.out) if args.out else eval_dir / "results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nRaw results saved to: {out_path}")

        judgment = run_judge_single(client, results, args.max_iter, v2_prompts)

    judge_path = eval_dir / "judgment.md"
    judge_path.write_text(judgment)
    print(f"Judgment saved to: {judge_path}\n")
    print("=" * 60)
    print("JUDGE EVALUATION")
    print("=" * 60)
    print(judgment)


if __name__ == "__main__":
    main()
