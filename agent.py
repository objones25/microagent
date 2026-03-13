import sqlite3
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

from logger import RunMetrics, setup_logging, save_metrics
from tools import TOOL_SCHEMAS, dispatch_tool

DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass
class _LoopState:
    """Mutable state threaded through the implementation loop."""
    messages: list
    iteration: int = 0          # resets on approved test revision
    total_iterations: int = 0   # never resets — used for metrics
    last_output: str = ""
    pending_write: bool = False
    revision_offered: bool = False        # stays True after denial to prevent re-offering
    original_test_content: Optional[str] = None  # set when revision is active
    impl_start: float = field(default_factory=time.perf_counter)

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    max_iterations: int = 10
    max_retries: int = 3
    prompts_version: str = "v2.4"
    allow_test_revision: bool = False
    auto_approve_revision: bool = False
    min_coverage: float = 0.0


def load_prompts(
    version: str = "v2.4",
    conn: Optional[sqlite3.Connection] = None,
    path: Optional[Path] = None,
) -> dict:
    if conn is not None:
        import db
        return db.load_prompts(conn, version)
    target = path or (Path(__file__).parent / "prompts" / f"{version}.toml")
    with open(target, "rb") as f:
        return tomllib.load(f)


def _parse_coverage_pct(output: str) -> float:
    """Parse solution.py coverage % from pytest-cov term-missing output.

    Looks for a line starting with 'solution.py' or 'solution ' and extracts
    the percentage column, e.g. 'solution.py   45   0   100%' → 100.0.
    Returns 0.0 if not found or unparseable.
    """
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("solution.py") or stripped.startswith("solution "):
            parts = stripped.split()
            for part in reversed(parts):
                if part.endswith("%"):
                    try:
                        return float(part[:-1])
                    except ValueError:
                        pass
    return 0.0


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
        config: Optional[AgentConfig] = None,
        *,
        prompts: Optional[dict] = None,
        logger=None,
        db_conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        cfg = config if config is not None else AgentConfig()
        self.client = client
        self.task_dir = task_dir
        self.model = cfg.model
        self.max_iterations = cfg.max_iterations
        self.max_retries = cfg.max_retries
        self.prompts_version = cfg.prompts_version
        self.allow_test_revision = cfg.allow_test_revision
        self.auto_approve_revision = cfg.auto_approve_revision
        self.min_coverage = cfg.min_coverage
        self._db_conn = db_conn
        self._prompts = (
            prompts if prompts is not None
            else load_prompts(cfg.prompts_version, conn=db_conn)
        )
        self._logger = logger if logger is not None else setup_logging(task_dir)
        self.metrics = RunMetrics(
            task_prompt="",
            task_dir=str(task_dir),
            prompts_version=cfg.prompts_version,
            model=cfg.model,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # API call with exponential backoff retry
    # ------------------------------------------------------------------

    def _call_api_with_retry(self, **kwargs):
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.messages.create(**kwargs)
            except _RETRYABLE as e:
                if attempt == self.max_retries:
                    raise
                self.metrics.api_retries += 1
                self._logger.warning(
                    f"  API error ({type(e).__name__}), retrying in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})..."
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)

    # ------------------------------------------------------------------
    # Phase 1: generate locked test file
    # ------------------------------------------------------------------

    def generate_tests(self, user_prompt: str) -> str:
        self.metrics.task_prompt = user_prompt
        self._logger.info("Generating test file...")
        start = time.perf_counter()
        response = self._call_api_with_retry(
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
        state = _LoopState(
            messages=[{"role": "user", "content": self._build_user_message(user_prompt, test_content)}]
        )
        while state.iteration < self.max_iterations:
            response = self._llm_call(state)
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "end_turn":
                result = self._handle_end_turn(response, state)
                if result is not None:
                    return result
                continue

            if not tool_calls:
                return self._fail(state, "Agent stopped requesting tools before tests passed.", "test_failure")

            state.messages.append({"role": "assistant", "content": response.content})
            state.messages.append({"role": "user", "content": self._run_tools(tool_calls, state)})
            self._maybe_apply_revision(response, state)

        return self._fail(
            state, f"Max iterations ({self.max_iterations}) reached.", "max_iterations",
            message=f"Max iterations ({self.max_iterations}) reached.\n{state.last_output}",
        )

    def _build_user_message(self, user_prompt: str, test_content: str) -> str:
        prompt_md_path = self.task_dir / "solution.prompt.md"
        prompt_md_section = ""
        if prompt_md_path.exists():
            prompt_md_section = self._prompts["prompt_md_section"]["template"].format(
                prompt_md=prompt_md_path.read_text()
            )
        return self._prompts["implementation"]["user"].format(
            user_prompt=user_prompt,
            test_content=test_content,
            prompt_md_section=prompt_md_section,
        )

    def _llm_call(self, state: _LoopState):
        self.metrics.impl_llm_calls += 1
        response = self._call_api_with_retry(
            model=self.model,
            max_tokens=8096,
            system=self._prompts["implementation"]["system"],
            tools=TOOL_SCHEMAS,
            messages=state.messages,
        )
        self.metrics.impl_input_tokens += response.usage.input_tokens
        self.metrics.impl_output_tokens += response.usage.output_tokens
        return response

    def _handle_end_turn(self, response, state: _LoopState) -> tuple[bool, str] | None:
        """Return (success, message) to stop the loop, or None to continue."""
        solution_path = self.task_dir / "solution.py"
        if solution_path.exists():
            result = subprocess.run(
                ["pytest", "solution_test.py", "-v", "--override-ini=addopts="],
                cwd=str(self.task_dir), capture_output=True, text=True, timeout=60,
            )
            final_output = result.stdout + result.stderr
            if _tests_passed(final_output):
                self.metrics.impl_duration_s = time.perf_counter() - state.impl_start
                self.metrics.impl_iterations = state.total_iterations
                cov = self._measure_coverage()
                self.metrics.test_coverage_pct = cov
                self._logger.info(f"  → coverage: {cov:.0f}%" if cov > 0 else "  → coverage: n/a")
                if self.min_coverage > 0 and cov < self.min_coverage:
                    reason = f"Test coverage {cov:.0f}% below minimum {self.min_coverage:.0f}%."
                    return self._fail(state, reason, "coverage")
                self.metrics.success = True
                return True, final_output
            state.last_output = final_output

        if self.allow_test_revision and not state.revision_offered:
            state.revision_offered = True
            state.original_test_content = (self.task_dir / "solution_test.py").read_text()
            state.messages.append({"role": "assistant", "content": response.content})
            state.messages.append({"role": "user", "content": self._prompts["test_revision"]["user"]})
            self.metrics.test_revisions_attempted += 1
            self._logger.info("  [test revision offered — agent stopped without passing]")
            return None

        final_text = next((b.text for b in response.content if b.type == "text"), "")
        return self._fail(
            state, "Agent stopped without passing tests.", "test_failure",
            message=f"Agent stopped without passing tests.\n{final_text}\n\n{state.last_output}",
        )

    def _run_tools(self, tool_calls, state: _LoopState) -> list[dict]:
        tool_results = []
        for tc in tool_calls:
            self.metrics.tool_calls[tc.name] = self.metrics.tool_calls.get(tc.name, 0) + 1
            result_str = dispatch_tool(tc.name, tc.input, self.task_dir)
            self._log_tool_call(tc, result_str)

            if tc.name == "write_file":
                self.metrics.impl_write_count += 1
                state.pending_write = True
            elif tc.name == "run_subprocess":
                state.last_output = result_str
                self.metrics.impl_pytest_runs += 1
                if state.pending_write:
                    state.iteration += 1
                    state.total_iterations += 1
                    state.pending_write = False

            tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result_str})
        return tool_results

    def _maybe_apply_revision(self, response, state: _LoopState) -> None:
        if state.original_test_content is None:
            return
        current = (self.task_dir / "solution_test.py").read_text()
        if current == state.original_test_content:
            return

        original = state.original_test_content
        state.original_test_content = None
        agent_reasoning = next((b.text for b in response.content if b.type == "text"), "")
        approved = self._prompt_test_revision_approval(agent_reasoning, original, current)
        self.metrics.test_revision_reasoning = agent_reasoning

        if approved:
            self.metrics.test_revisions_approved += 1
            state.revision_offered = False
            state.iteration = 0
            state.pending_write = False
            self._logger.info("  [test revision approved — resetting iteration counter]")
        else:
            (self.task_dir / "solution_test.py").write_text(original)
            self._logger.info("  [test revision denied — original tests restored]")

    def _fail(self, state: _LoopState, reason: str, category: str, *, message: str = "") -> tuple[bool, str]:
        self.metrics.impl_duration_s = time.perf_counter() - state.impl_start
        self.metrics.impl_iterations = state.total_iterations
        self.metrics.success = False
        self.metrics.failure_reason = reason
        self.metrics.failure_category = category
        return False, message or reason

    def _log_tool_call(self, tc, result_str: str) -> None:
        """Emit one structured log line per tool call."""
        inp = tc.input
        match tc.name:
            case "read_file":
                self._logger.info(f"  → read_file: {inp.get('path', '?')}")
            case "write_file":
                content = inp.get("content", "")
                lines = content.count("\n") + 1 if content else 0
                self._logger.info(f"  → write_file: {inp.get('path', '?')} ({lines} lines)")
            case "run_subprocess":
                cmd = inp.get("command", [])
                self._logger.info(f"  → run_subprocess: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
                summary, failed_names = _parse_pytest_result(result_str)
                if failed_names:
                    self._logger.info(f"    ✗ {summary} | failing: {', '.join(failed_names[:5])}")
                elif summary:
                    self._logger.info(f"    ✓ {summary}")
            case "context7_docs":
                self._logger.info(f"  → context7_docs: {inp.get('library', '?')} — {inp.get('query', '')[:60]}")
            case "firecrawl_search":
                self._logger.info(f'  → firecrawl_search: "{inp.get("query", "")[:60]}"')
            case "firecrawl_scrape":
                self._logger.info(f"  → firecrawl_scrape: {inp.get('url', '?')}")
            case "run_python":
                code = inp.get("code", "")
                self._logger.info(f"  → run_python: {code.splitlines()[0][:60] if code else '?'}")
            case "calculator":
                self._logger.info(f"  → calculator: {inp.get('expression', '?')} = {result_str}")

    def _measure_coverage(self) -> float:
        """Run pytest-cov on solution.py and return coverage percentage (0–100).

        Returns 0.0 if pytest-cov is not installed or coverage cannot be parsed.
        """
        result = subprocess.run(
            ["pytest", "solution_test.py", "--cov=solution",
             "--cov-report=term-missing", "--tb=no", "-q",
             "--override-ini=addopts="],
            cwd=str(self.task_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        return _parse_coverage_pct(output)

    def _prompt_test_revision_approval(
        self, reasoning: str, old: str, new: str
    ) -> bool:
        if self.auto_approve_revision:
            self._logger.info("  [test revision auto-approved]")
            return True
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
        save_metrics(self.metrics, self.task_dir, conn=self._db_conn)
        total_in = self.metrics.test_gen_input_tokens + self.metrics.impl_input_tokens
        total_out = self.metrics.test_gen_output_tokens + self.metrics.impl_output_tokens
        self._logger.info(
            f"Tokens — test gen: {self.metrics.test_gen_input_tokens}in/"
            f"{self.metrics.test_gen_output_tokens}out  "
            f"impl: {self.metrics.impl_input_tokens}in/"
            f"{self.metrics.impl_output_tokens}out  "
            f"total: {total_in}in/{total_out}out"
        )
        if self.metrics.api_retries:
            self._logger.info(f"API retries: {self.metrics.api_retries}")
        self._logger.debug(
            f"Metrics: {self.metrics.impl_iterations} iterations, "
            f"{self.metrics.impl_llm_calls} LLM calls, "
            f"{self.metrics.total_tool_calls} tool calls, "
            f"{self.metrics.total_duration_s:.1f}s total, "
            f"failure_category={self.metrics.failure_category!r}"
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
