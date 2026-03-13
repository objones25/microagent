import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

from logger import RunMetrics, setup_logging, save_metrics
from tools import TOOL_SCHEMAS, dispatch_tool

DEFAULT_MODEL = "claude-sonnet-4-6"


def load_prompts(version: str = "v1", path: Optional[Path] = None) -> dict:
    target = path or (Path(__file__).parent / "prompts" / f"{version}.toml")
    with open(target, "rb") as f:
        return tomllib.load(f)


def _parse_pytest_result(output: str) -> tuple[str, list[str]]:
    """Returns (summary_line, list_of_failed_test_names)."""
    summary = ""
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if ("passed" in stripped or "failed" in stripped or "error" in stripped) and (
            "==" in stripped or stripped.startswith("PASSED") or stripped.startswith("FAILED")
        ):
            summary = stripped
            break
    failed = []
    for line in output.splitlines():
        if line.startswith("FAILED "):
            test_name = line.split("::")[-1].split(" ")[0]
            failed.append(test_name)
    return summary, failed


class AgentLoop:
    def __init__(
        self,
        client: anthropic.Anthropic,
        task_dir: Path,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 10,
        prompts: Optional[dict] = None,
        prompts_version: str = "v1",
        logger=None,
        allow_test_revision: int = 0,
    ) -> None:
        self.client = client
        self.task_dir = task_dir
        self.model = model
        self.max_iterations = max_iterations
        self.prompts_version = prompts_version
        self.allow_test_revision = allow_test_revision
        self._prompts = prompts if prompts is not None else load_prompts(prompts_version)
        self._logger = logger if logger is not None else setup_logging(task_dir)
        self.metrics = RunMetrics(
            task_prompt="",
            task_dir=str(task_dir),
            prompts_version=prompts_version,
            model=model,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Phase 1: generate locked test file
    # ------------------------------------------------------------------

    def generate_tests(self, user_prompt: str) -> str:
        self.metrics.task_prompt = user_prompt
        self._logger.info("Generating test file...")
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self._prompts["test_generation"]["system"],
            messages=[
                {
                    "role": "user",
                    "content": self._prompts["test_generation"]["user"].format(
                        user_prompt=user_prompt
                    ),
                }
            ],
        )
        self.metrics.test_gen_duration_s = time.perf_counter() - start
        self.metrics.test_gen_input_tokens = response.usage.input_tokens
        self.metrics.test_gen_output_tokens = response.usage.output_tokens
        self._logger.debug(
            f"Test gen: {self.metrics.test_gen_duration_s:.1f}s, "
            f"{self.metrics.test_gen_input_tokens} in / {self.metrics.test_gen_output_tokens} out tokens"
        )

        test_content = response.content[0].text.strip()
        test_path = self.task_dir / "solution_test.py"
        test_path.write_text(test_content)
        test_count = test_content.count("def test_")
        self._logger.info(f"Test file written: solution_test.py ({test_count} tests)")
        return test_content

    # ------------------------------------------------------------------
    # Phase 2: iterative implementation loop
    # ------------------------------------------------------------------

    def run_implementation_loop(self, user_prompt: str, test_content: str) -> tuple[bool, str]:
        prompt_md_path = self.task_dir / "solution.prompt.md"
        prompt_md_section = ""
        if prompt_md_path.exists():
            prompt_md_section = self._prompts["prompt_md_section"]["template"].format(
                prompt_md=prompt_md_path.read_text()
            )

        user_content = self._prompts["implementation"]["user"].format(
            user_prompt=user_prompt,
            test_content=test_content,
            prompt_md_section=prompt_md_section,
        )

        messages: list[dict] = [{"role": "user", "content": user_content}]
        iteration = 0
        last_output = ""
        impl_start = time.perf_counter()
        pending_write = False

        # Test revision state
        _revision_offered = False
        _original_test_content = None

        while iteration < self.max_iterations:
            # Offer test revision if enabled and threshold reached
            if (
                self.allow_test_revision > 0
                and iteration >= self.allow_test_revision
                and not _revision_offered
            ):
                _revision_offered = True
                _original_test_content = (self.task_dir / "solution_test.py").read_text()
                messages.append({
                    "role": "user",
                    "content": self._prompts["test_revision"]["user"].format(n=iteration),
                })
                self.metrics.test_revisions_attempted += 1
                self._logger.info(f"  [test revision offered after {iteration} failing iterations]")

            self.metrics.impl_llm_calls += 1
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8096,
                system=self._prompts["implementation"]["system"],
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "end_turn":
                # Ground-truth check: run pytest ourselves regardless of what the LLM ran
                solution_path = self.task_dir / "solution.py"
                if solution_path.exists():
                    result = subprocess.run(
                        ["pytest", "solution_test.py", "-v"],
                        cwd=str(self.task_dir),
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    final_output = result.stdout + result.stderr
                    if _tests_passed(final_output):
                        self.metrics.impl_duration_s = time.perf_counter() - impl_start
                        self.metrics.impl_iterations = iteration
                        self.metrics.success = True
                        return True, final_output
                    last_output = final_output

                final_text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                self.metrics.impl_duration_s = time.perf_counter() - impl_start
                self.metrics.impl_iterations = iteration
                self.metrics.success = False
                self.metrics.failure_reason = "Agent stopped without passing tests."
                return False, f"Agent stopped without passing tests.\n{final_text}\n\n{last_output}"

            if not tool_calls:
                self.metrics.impl_duration_s = time.perf_counter() - impl_start
                self.metrics.impl_iterations = iteration
                self.metrics.success = False
                self.metrics.failure_reason = "Agent stopped requesting tools before tests passed."
                return False, "Agent stopped requesting tools before tests passed."

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # Dispatch all tool calls, build tool_result list
            tool_results = []
            for tc in tool_calls:
                self.metrics.tool_calls[tc.name] = self.metrics.tool_calls.get(tc.name, 0) + 1
                result_str = dispatch_tool(tc.name, tc.input, self.task_dir)

                # Granular per-tool logging
                if tc.name == "read_file":
                    path = tc.input.get("path", "?")
                    self._logger.info(f"  → read_file: {path}")

                elif tc.name == "write_file":
                    path = tc.input.get("path", "?")
                    content = tc.input.get("content", "")
                    line_count = content.count("\n") + 1 if content else 0
                    self._logger.info(f"  → write_file: {path} ({line_count} lines)")
                    self.metrics.impl_write_count += 1
                    pending_write = True

                elif tc.name == "run_subprocess":
                    cmd = tc.input.get("command", [])
                    cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
                    self._logger.info(f"  → run_subprocess: {cmd_str}")
                    last_output = result_str
                    self.metrics.impl_pytest_runs += 1
                    summary, failed_names = _parse_pytest_result(result_str)
                    if failed_names:
                        failing_str = ", ".join(failed_names[:5])
                        self._logger.info(f"    ✗ {summary} | failing: {failing_str}")
                    elif summary:
                        self._logger.info(f"    ✓ {summary}")
                    # Increment iteration on write→run pair
                    if pending_write:
                        iteration += 1
                        pending_write = False

                elif tc.name == "context7_docs":
                    library = tc.input.get("library", "?")
                    query = tc.input.get("query", "")
                    self._logger.info(f"  → context7_docs: {library} — {query[:60]}")

                elif tc.name == "firecrawl_search":
                    query = tc.input.get("query", "")
                    self._logger.info(f'  → firecrawl_search: "{query[:60]}"')

                elif tc.name == "firecrawl_scrape":
                    url = tc.input.get("url", "?")
                    self._logger.info(f"  → firecrawl_scrape: {url}")

                elif tc.name == "calculator":
                    expr = tc.input.get("expression", "?")
                    self._logger.info(f"  → calculator: {expr} = {result_str}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_str,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

            # Check if test file was rewritten after a revision offer
            if (
                _revision_offered
                and _original_test_content is not None
            ):
                current_test_content = (self.task_dir / "solution_test.py").read_text()
                if current_test_content != _original_test_content:
                    agent_reasoning = next(
                        (b.text for b in response.content if b.type == "text"), ""
                    )
                    approved = self._prompt_test_revision_approval(
                        agent_reasoning, _original_test_content, current_test_content
                    )
                    self.metrics.test_revision_reasoning = agent_reasoning

                    if approved:
                        self.metrics.test_revisions_approved += 1
                        iteration = 0
                        pending_write = False
                        _revision_offered = False
                        _original_test_content = None
                        self._logger.info("  [test revision approved — resetting iteration counter]")
                    else:
                        (self.task_dir / "solution_test.py").write_text(_original_test_content)
                        self._logger.info("  [test revision denied — original tests restored]")
                        _original_test_content = None

        self.metrics.impl_duration_s = time.perf_counter() - impl_start
        self.metrics.impl_iterations = iteration
        self.metrics.success = False
        self.metrics.failure_reason = f"Max iterations ({self.max_iterations}) reached."
        return False, f"Max iterations ({self.max_iterations}) reached.\n{last_output}"

    def _prompt_test_revision_approval(
        self, reasoning: str, old: str, new: str
    ) -> bool:
        print("\n" + "=" * 60)
        print("AGENT PROPOSES TEST REVISION")
        print("=" * 60)
        if reasoning:
            print("\nAgent reasoning:")
            print(reasoning)
        print("\n--- OLD solution_test.py ---")
        print(old)
        print("\n--- NEW solution_test.py ---")
        print(new)
        print("=" * 60)
        try:
            answer = input("Approve this test revision? [y/N] ").strip().lower()
            return answer == "y"
        except KeyboardInterrupt:
            return False

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self, user_prompt: str, auto_approve: bool = False) -> None:
        self.task_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1
        test_content = self.generate_tests(user_prompt)

        # Show tests, optionally wait for approval
        print("\n" + "=" * 60)
        print("GENERATED TESTS (solution_test.py):")
        print("=" * 60)
        print(test_content)
        print("=" * 60)
        if not auto_approve:
            try:
                input("\nPress Enter to start implementation, or Ctrl+C to abort...\n")
            except KeyboardInterrupt:
                print("\nAborted.")
                sys.exit(0)

        # Phase 2
        self._logger.info("Starting implementation loop...")
        success, message = self.run_implementation_loop(user_prompt, test_content)

        # Save metrics
        save_metrics(self.metrics, self.task_dir)
        self._logger.debug(
            f"Metrics: {self.metrics.impl_iterations} iterations, "
            f"{self.metrics.impl_llm_calls} LLM calls, "
            f"{self.metrics.total_tool_calls} tool calls, "
            f"{self.metrics.total_duration_s:.1f}s total"
        )

        print("\n" + "=" * 60)
        if success:
            print("SUCCESS — all tests passed!")
        else:
            print("FAILED — could not pass all tests.")
        print("=" * 60)
        print(message)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _tests_passed(pytest_output: str) -> bool:
    """Check the pytest summary line, e.g. '12 passed in 0.03s'."""
    for line in reversed(pytest_output.splitlines()):
        line = line.strip()
        if "passed" in line and ("==" in line or line.startswith("PASSED")):
            return "failed" not in line and "error" not in line
    return False
