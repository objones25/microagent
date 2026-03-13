#!/usr/bin/env python3
"""
eval.py — Evaluation harness for microagent.

Runs a suite of harder coding tasks through the agent, collects structured results,
then passes everything to a Claude judge for analysis and prompt improvement suggestions.

Usage:
    python eval.py                           # run v1 prompts on all tasks
    python eval.py --compare v2              # A/B test v1 vs v2 prompts
    python eval.py --tasks 5                 # run first N tasks only
    python eval.py --max-iter 5              # limit implementation iterations per task
    python eval.py --out results.json        # save raw results to file
    python eval.py --optimize                # save improved prompt TOML after judging
    python eval.py --meta-judge              # also evaluate the judge's output quality
    python eval.py --eval-prompts eval-v2    # use a different judge/optimizer prompt version
    python eval.py --tasks-file v2           # use a different task list
"""

import argparse
import json
import os
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import anthropic

from agent import AgentLoop, load_prompts
from logger import RunMetrics, setup_logging, save_metrics


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_eval_prompts(version: str = "eval-v1") -> dict:
    path = Path(__file__).parent / "prompts" / f"{version}.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_tasks(version: str = "v1") -> list[str]:
    path = Path(__file__).parent / "evals" / f"tasks-{version}.txt"
    lines = path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


# ------------------------------------------------------------------
# Result collection
# ------------------------------------------------------------------

def run_task(
    client: anthropic.Anthropic,
    task: str,
    task_dir: Path,
    max_iterations: int,
    prompts_dict: dict,
    prompts_version: str,
    allow_test_revision: bool = False,
    auto_approve_revision: bool = False,
) -> RunMetrics:
    task_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(task_dir)

    loop = AgentLoop(
        client=client,
        task_dir=task_dir,
        max_iterations=max_iterations,
        prompts=prompts_dict,
        prompts_version=prompts_version,
        logger=logger,
        allow_test_revision=allow_test_revision,
        auto_approve_revision=auto_approve_revision,
    )

    try:
        test_content = loop.generate_tests(task)
        loop.run_implementation_loop(task, test_content)
    except Exception as e:
        loop.metrics.failure_reason = str(e)

    save_metrics(loop.metrics, task_dir)
    return loop.metrics


def format_results_block(metrics_list: list[RunMetrics]) -> str:
    blocks = []
    for i, m in enumerate(metrics_list, 1):
        status = "PASSED" if m.success else "FAILED"
        error_note = f"\n  error: {m.failure_reason}" if m.failure_reason and not m.success else ""

        task_dir = Path(m.task_dir)
        test_snippet = ""
        test_path = task_dir / "solution_test.py"
        if test_path.exists():
            test_snippet = f"\n\n**solution_test.py:**\n```python\n{test_path.read_text()}\n```"

        sol_snippet = ""
        sol_path = task_dir / "solution.py"
        if sol_path.exists():
            lines = sol_path.read_text().splitlines()
            preview = "\n".join(lines[:40])
            if len(lines) > 40:
                preview += f"\n  ... ({len(lines) - 40} more lines)"
            sol_snippet = f"\n\n**solution.py:**\n```python\n{preview}\n```"

        pytest_snippet = ""
        log_path = task_dir / "run.log"
        if log_path.exists():
            log_lines = log_path.read_text().splitlines()
            trimmed = "\n".join(log_lines[-20:])
            pytest_snippet = f"\n\n**run.log (last 20 lines):**\n```\n{trimmed}\n```"

        blocks.append(
            f"### Task {i}: {m.task_prompt}\n"
            f"- **Status**: {status}\n"
            f"- **Iterations**: {m.impl_iterations}\n"
            f"- **Duration**: {m.total_duration_s:.1f}s{error_note}"
            f"{test_snippet}{sol_snippet}{pytest_snippet}"
        )
    return "\n\n---\n\n".join(blocks)


def summary_line(metrics_list: list[RunMetrics]) -> str:
    passed = sum(1 for m in metrics_list if m.success)
    total_iter = sum(m.impl_iterations for m in metrics_list)
    avg = total_iter / len(metrics_list) if metrics_list else 0
    return f"{passed}/{len(metrics_list)} passed, avg {avg:.1f} iterations"


def build_eval_metrics(metrics_list: list[RunMetrics]) -> dict:
    n = len(metrics_list)
    passed = sum(1 for m in metrics_list if m.success)
    tool_totals: dict[str, int] = {}
    for m in metrics_list:
        for tool, count in m.tool_calls.items():
            tool_totals[tool] = tool_totals.get(tool, 0) + count

    summary = {
        "tasks": n,
        "passed": passed,
        "failed": n - passed,
        "avg_iterations": round(sum(m.impl_iterations for m in metrics_list) / n, 2) if n else 0,
        "avg_total_duration_s": round(sum(m.total_duration_s for m in metrics_list) / n, 2) if n else 0,
        "avg_test_gen_duration_s": round(sum(m.test_gen_duration_s for m in metrics_list) / n, 2) if n else 0,
        "avg_impl_duration_s": round(sum(m.impl_duration_s for m in metrics_list) / n, 2) if n else 0,
        "total_tool_calls_by_type": tool_totals,
        "avg_impl_llm_calls": round(sum(m.impl_llm_calls for m in metrics_list) / n, 2) if n else 0,
    }
    return {
        "summary": summary,
        "runs": [m.to_dict() for m in metrics_list],
    }


# ------------------------------------------------------------------
# Judges
# ------------------------------------------------------------------

def run_judge_single(
    client: anthropic.Anthropic,
    metrics_list: list[RunMetrics],
    max_iterations: int,
    prompts_dict: dict,
    eval_prompts: dict,
) -> str:
    user_content = eval_prompts["judge_single"]["template"].format(
        n=len(metrics_list),
        max_iter=max_iterations,
        test_gen_system=prompts_dict["test_generation"]["system"],
        impl_system=prompts_dict["implementation"]["system"],
        results_block=format_results_block(metrics_list),
    )
    print("\nCalling judge...", flush=True)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=eval_prompts["judge"]["system"],
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


def run_judge_ab(
    client: anthropic.Anthropic,
    v1_metrics: list[RunMetrics],
    v2_metrics: list[RunMetrics],
    max_iterations: int,
    v1_prompts: dict,
    v2_prompts: dict,
    eval_prompts: dict,
) -> str:
    user_content = eval_prompts["judge_ab"]["template"].format(
        max_iter=max_iterations,
        v1_test_gen=v1_prompts["test_generation"]["system"],
        v1_impl=v1_prompts["implementation"]["system"],
        v2_test_gen=v2_prompts["test_generation"]["system"],
        v2_impl=v2_prompts["implementation"]["system"],
        v1_results_block=format_results_block(v1_metrics),
        v2_results_block=format_results_block(v2_metrics),
    )
    print("\nCalling A/B judge...", flush=True)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=eval_prompts["judge"]["system"],
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


def run_meta_judge(
    client: anthropic.Anthropic,
    metrics_list: list[RunMetrics],
    judgment_text: str,
    eval_prompts: dict,
    judge_template_key: str = "judge_single",
) -> str:
    n_tasks = len(metrics_list)
    n_passed = sum(1 for m in metrics_list if m.success)
    n_failed = n_tasks - n_passed
    avg_iter = sum(m.impl_iterations for m in metrics_list) / n_tasks if n_tasks else 0

    user_content = eval_prompts["meta_judge"]["template"].format(
        judge_system=eval_prompts["judge"]["system"],
        judge_template=eval_prompts[judge_template_key]["template"],
        n_tasks=n_tasks,
        n_passed=n_passed,
        n_failed=n_failed,
        avg_iter=avg_iter,
        judgment_text=judgment_text,
    )
    print("\nCalling meta-judge...", flush=True)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=eval_prompts["meta_judge"]["system"],
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


def optimize_prompts(
    client: anthropic.Anthropic,
    current_toml_path: Path,
    judge_analysis: str,
    eval_timestamp: str,
    eval_prompts: dict,
) -> None:
    current_toml = current_toml_path.read_text()
    user_message = current_toml + "\n\n---\n\n" + judge_analysis
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=eval_prompts["prompt_optimizer"]["system"],
        messages=[{"role": "user", "content": user_message}],
    )
    new_toml_text = response.content[0].text.strip()
    try:
        tomllib.loads(new_toml_text)
    except tomllib.TOMLDecodeError as e:
        print(f"Warning: optimized output is not valid TOML ({e}). Not saving.")
        return
    out_path = Path(__file__).parent / "prompts" / f"{eval_timestamp}.toml"
    out_path.write_text(new_toml_text)
    print(f"Optimized prompts saved → prompts/{eval_timestamp}.toml")
    print(f"Test with: uv run python eval.py --prompts {eval_timestamp}")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_suite(
    client: anthropic.Anthropic,
    tasks: list[str],
    base_dir: Path,
    max_iterations: int,
    prompts_dict: dict,
    prompts_version: str,
    label: str = "",
    allow_test_revision: bool = False,
    auto_approve_revision: bool = False,
) -> list[RunMetrics]:
    results = []
    prefix = f"[{label}] " if label else ""
    for i, task in enumerate(tasks, 1):
        task_dir = base_dir / f"task-{i:02d}"
        print(f"{prefix}[{i}/{len(tasks)}] {task}", flush=True)
        metrics = run_task(
            client, task, task_dir, max_iterations, prompts_dict, prompts_version,
            allow_test_revision=allow_test_revision,
            auto_approve_revision=auto_approve_revision,
        )
        status = "✓ PASS" if metrics.success else "✗ FAIL"
        print(f"  → {status} in {metrics.impl_iterations} iter(s), {metrics.total_duration_s:.1f}s\n")
        results.append(metrics)
    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate microagent on a suite of harder coding tasks")
    parser.add_argument("--prompts", default="v1", metavar="VERSION",
                        help="Agent prompts version to use (default: v1)")
    parser.add_argument("--compare", default=None, metavar="VERSION",
                        help="Second agent prompts version to A/B test against --prompts")
    parser.add_argument("--tasks", type=int, default=None,
                        help="Number of tasks to run (default: all)")
    parser.add_argument("--tasks-file", default="v1", metavar="VERSION",
                        help="Task list version to load from evals/tasks-<VERSION>.txt (default: v1)")
    parser.add_argument("--max-iter", type=int, default=5,
                        help="Max implementation iterations per task (default: 5)")
    parser.add_argument("--out", default=None,
                        help="Save raw results JSON to this file")
    parser.add_argument("--optimize", action="store_true",
                        help="After judge analysis, call Claude to produce an improved TOML prompt file")
    parser.add_argument("--meta-judge", action="store_true",
                        help="After judging, call Claude to evaluate the quality of the judge's output")
    parser.add_argument("--eval-prompts", default="eval-v1", metavar="VERSION",
                        help="Eval prompts version (judge, optimizer, meta-judge) from prompts/<VERSION>.toml (default: eval-v1)")
    parser.add_argument("--allow-test-revision", action="store_true",
                        help="When the agent stops without passing tests, offer it a chance to revise tests (requires --auto-approve-revision for non-interactive use)")
    parser.add_argument("--auto-approve-revision", action="store_true",
                        help="Automatically approve agent test revisions without prompting (use with --allow-test-revision)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    all_tasks = load_tasks(args.tasks_file)
    tasks = all_tasks[: args.tasks] if args.tasks is not None else all_tasks
    eval_prompts = load_eval_prompts(args.eval_prompts)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_dir = Path(f"eval-{timestamp}")
    eval_dir.mkdir()

    v1_prompts = load_prompts(args.prompts)

    if args.compare:
        v2_prompts = load_prompts(args.compare)

        print(f"A/B test: {len(tasks)} tasks × 2 prompt versions → {eval_dir}/")
        print(f"V1: {args.prompts}  V2: {args.compare}")
        print(f"Max iterations per task: {args.max_iter}\n")

        print(f"=== Running {args.prompts} ===")
        v1_dir = eval_dir / args.prompts
        v1_dir.mkdir()
        v1_results = run_suite(client, tasks, v1_dir, args.max_iter, v1_prompts, args.prompts,
                               label=args.prompts, allow_test_revision=args.allow_test_revision,
                               auto_approve_revision=args.auto_approve_revision)

        print(f"=== Running {args.compare} ===")
        v2_dir = eval_dir / args.compare
        v2_dir.mkdir()
        v2_results = run_suite(client, tasks, v2_dir, args.max_iter, v2_prompts, args.compare,
                               label=args.compare, allow_test_revision=args.allow_test_revision,
                               auto_approve_revision=args.auto_approve_revision)

        print("=" * 60)
        print(f"{args.prompts}: {summary_line(v1_results)}")
        print(f"{args.compare}: {summary_line(v2_results)}")
        print("=" * 60)

        all_results = {
            args.prompts: build_eval_metrics(v1_results),
            args.compare: build_eval_metrics(v2_results),
        }
        out_path = Path(args.out) if args.out else eval_dir / "results.json"
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nRaw results saved to: {out_path}")

        eval_metrics_path = eval_dir / "eval_metrics.json"
        all_results["_meta"] = {
            "eval_prompts_version": args.eval_prompts,
            "tasks_version": args.tasks_file,
        }
        eval_metrics_path.write_text(json.dumps(all_results, indent=2))
        print(f"Eval metrics saved to: {eval_metrics_path}")

        judgment = run_judge_ab(
            client, v1_results, v2_results, args.max_iter, v1_prompts, v2_prompts, eval_prompts
        )
        judge_template_key = "judge_ab"
        primary_metrics = v1_results + v2_results

    else:
        print(f"Running {len(tasks)} tasks → {eval_dir}/")
        print(f"Prompts: {args.prompts}")
        print(f"Max iterations per task: {args.max_iter}\n")

        results = run_suite(client, tasks, eval_dir, args.max_iter, v1_prompts, args.prompts,
                            allow_test_revision=args.allow_test_revision,
                            auto_approve_revision=args.auto_approve_revision)

        passed = sum(1 for m in results if m.success)
        total_iter = sum(m.impl_iterations for m in results)
        print("=" * 60)
        print(f"RESULTS: {passed}/{len(results)} passed")
        print(f"Total iterations: {total_iter} (avg {total_iter/len(results):.1f} per task)")
        print("=" * 60)

        eval_data = build_eval_metrics(results)
        eval_data["_meta"] = {
            "eval_prompts_version": args.eval_prompts,
            "tasks_version": args.tasks_file,
        }
        out_path = Path(args.out) if args.out else eval_dir / "results.json"
        out_path.write_text(json.dumps(eval_data, indent=2))
        print(f"\nRaw results saved to: {out_path}")

        eval_metrics_path = eval_dir / "eval_metrics.json"
        eval_metrics_path.write_text(json.dumps(eval_data, indent=2))
        print(f"Eval metrics saved to: {eval_metrics_path}")

        judgment = run_judge_single(client, results, args.max_iter, v1_prompts, eval_prompts)
        judge_template_key = "judge_single"
        primary_metrics = results

    judge_path = eval_dir / "judgment.md"
    judge_path.write_text(judgment)
    print(f"Judgment saved to: {judge_path}\n")
    print("=" * 60)
    print("JUDGE EVALUATION")
    print("=" * 60)
    print(judgment)

    if args.meta_judge:
        meta_judgment = run_meta_judge(
            client, primary_metrics, judgment, eval_prompts, judge_template_key
        )
        meta_path = eval_dir / "meta_judgment.md"
        meta_path.write_text(meta_judgment)
        print(f"\nMeta-judgment saved to: {meta_path}\n")
        print("=" * 60)
        print("META-JUDGE EVALUATION")
        print("=" * 60)
        print(meta_judgment)

    if args.optimize:
        print("\nOptimizing prompts...")
        current_toml_path = Path(__file__).parent / "prompts" / f"{args.prompts}.toml"
        optimize_prompts(client, current_toml_path, judgment, timestamp, eval_prompts)


if __name__ == "__main__":
    main()
