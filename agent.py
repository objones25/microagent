import ast
import re
import sqlite3
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import anthropic

from config import DEFAULT_MODEL, DEFAULT_MAX_ITERATIONS, DEFAULT_PROMPTS_VERSION
from logger import RunMetrics, setup_logging, save_metrics
from tools import TOOL_SCHEMAS, dispatch_tool

def _extract_write_lines(snapshot: str, prev_count: int) -> tuple[str | None, list[str], int]:
    """Extract path and new complete lines from a partial write_file JSON snapshot.

    snapshot:   raw JSON text accumulated by the streaming API (not parsed)
    prev_count: number of complete lines already emitted
    Returns:    (path_or_None, new_decoded_lines, new_total_count)
    """
    # Extract path (file paths don't normally need escape handling)
    path: str | None = None
    m = re.search(r'"path"\s*:\s*"([^"]*)"', snapshot)
    if m:
        path = m.group(1)

    # Find the start of the content string value
    cm = re.search(r'"content"\s*:\s*"', snapshot)
    if not cm:
        return path, [], prev_count

    content_raw = snapshot[cm.end():]

    # The JSON escape \n (2-char backslash+n) marks each newline in the file.
    # In Python source '\\n' is the 2-char string \+n, which is what we want.
    parts = content_raw.split("\\n")
    total_complete = len(parts) - 1  # last segment is the incomplete current line

    if total_complete <= prev_count:
        return path, [], prev_count

    new_lines: list[str] = []
    for raw in parts[prev_count:total_complete]:
        # Decode the most common JSON string escapes
        line = (
            raw.replace("\\\\", "\\")
               .replace('\\"', '"')
               .replace("\\t", "\t")
               .replace("\\r", "\r")
        )
        new_lines.append(line)

    return path, new_lines, total_complete


_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


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


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_retries: int = 3
    prompts_version: str = DEFAULT_PROMPTS_VERSION
    allow_test_revision: bool = False
    auto_approve_revision: bool = False
    min_coverage: float = 0.0


def load_prompts(
    version: str = DEFAULT_PROMPTS_VERSION,
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
        try:
            ast.parse(test_content)
        except SyntaxError as e:
            raise ValueError(f"Generated test file is not valid Python: {e}") from e
        test_path = self.task_dir / "solution_test.py"
        test_path.write_text(test_content)
        test_count = test_content.count("def test_")
        self._logger.info(f"Test file written: solution_test.py ({test_count} tests)")
        return test_content

    # ------------------------------------------------------------------
    # Phase 2: iterative implementation loop
    # ------------------------------------------------------------------

    def _implementation_gen(self, user_prompt: str, test_content: str, hint: str = "") -> Iterator[dict]:
        """Generator that drives the implementation loop, yielding AgentEvents."""
        state = _LoopState(
            messages=[{"role": "user", "content": self._build_user_message(user_prompt, test_content, hint)}]
        )
        while state.iteration < self.max_iterations:
            response = None
            for event in self._llm_call(state):
                if event["type"] == "_response":
                    response = event["response"]
                else:
                    yield event
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "end_turn":
                result = self._handle_end_turn(response, state)
                if result is not None:
                    success, message = result
                    if self.metrics.test_coverage_pct > 0:
                        yield {"type": "coverage", "pct": self.metrics.test_coverage_pct}
                    yield {
                        "type": "done",
                        "success": success,
                        "message": message,
                        "failure_reason": self.metrics.failure_reason,
                        "failure_category": self.metrics.failure_category,
                    }
                    return
                continue

            if not tool_calls:
                success, message = self._fail(
                    state, "Agent stopped requesting tools before tests passed.", "test_failure"
                )
                yield {
                    "type": "done",
                    "success": success,
                    "message": message,
                    "failure_reason": self.metrics.failure_reason,
                    "failure_category": self.metrics.failure_category,
                }
                return

            state.messages.append({"role": "assistant", "content": response.content})
            tool_results, events = self._run_tools(tool_calls, state)
            yield from events
            state.messages.append({"role": "user", "content": tool_results})
            self._maybe_apply_revision(response, state)

        success, message = self._fail(
            state,
            f"Max iterations ({self.max_iterations}) reached.",
            "max_iterations",
            message=f"Max iterations ({self.max_iterations}) reached.\n{state.last_output}",
        )
        yield {
            "type": "done",
            "success": success,
            "message": message,
            "failure_reason": self.metrics.failure_reason,
            "failure_category": self.metrics.failure_category,
        }

    def run_implementation_loop(self, user_prompt: str, test_content: str, hint: str = "") -> tuple[bool, str]:
        """Sync wrapper — drives _implementation_gen and returns the terminal (success, msg)."""
        result = (False, "")
        for event in self._implementation_gen(user_prompt, test_content, hint=hint):
            if event["type"] == "done":
                result = (event["success"], event["message"])
        return result

    def _build_user_message(self, user_prompt: str, test_content: str, hint: str = "") -> str:
        prompt_md_path = self.task_dir / "solution.prompt.md"
        prompt_md_section = ""
        if prompt_md_path.exists():
            prompt_md_section = self._prompts["prompt_md_section"]["template"].format(
                prompt_md=prompt_md_path.read_text()
            )
        message = self._prompts["implementation"]["user"].format(
            user_prompt=user_prompt,
            test_content=test_content,
            prompt_md_section=prompt_md_section,
        )
        if hint:
            message += f"\n\n<user_hint>{hint}</user_hint>"
        return message

    def _llm_call(self, state: _LoopState) -> Iterator[dict]:
        """Stream one LLM call.

        Yields:
          {"type": "text_delta", "text": str}            — agent reasoning tokens
          {"type": "write_line", "path": str,
           "line": str, "line_num": int}                 — write_file lines as generated
          {"type": "_response", "response": Message}    — sentinel; always last
        """
        self.metrics.impl_llm_calls += 1
        current_tool: str | None = None
        write_path: str | None = None
        write_lines_emitted: int = 0
        write_json_accum: str = ""  # accumulated partial_json for write_file

        with self.client.messages.stream(
            model=self.model,
            max_tokens=8096,
            system=self._prompts["implementation"]["system"],
            tools=TOOL_SCHEMAS,
            messages=state.messages,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = block.name
                        write_path = None
                        write_lines_emitted = 0
                        write_json_accum = ""
                    else:
                        current_tool = None
                elif event.type == "content_block_stop":
                    current_tool = None
                elif event.type == "text":
                    if event.text:
                        yield {"type": "text_delta", "text": event.text}
                elif event.type == "input_json" and current_tool == "write_file":
                    # event.snapshot is a parsed dict — use partial_json to build
                    # our own raw JSON accumulator for line extraction
                    write_json_accum += event.partial_json
                    path, new_lines, write_lines_emitted = _extract_write_lines(
                        write_json_accum, write_lines_emitted
                    )
                    if path and write_path is None:
                        write_path = path
                    start = write_lines_emitted - len(new_lines) + 1
                    for i, line in enumerate(new_lines, start=start):
                        yield {"type": "write_line", "path": write_path or "?",
                               "line": line, "line_num": i}
            final = stream.get_final_message()

        self.metrics.impl_input_tokens += final.usage.input_tokens
        self.metrics.impl_output_tokens += final.usage.output_tokens
        yield {"type": "_response", "response": final}

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

    def _run_tools(self, tool_calls, state: _LoopState) -> tuple[list[dict], list[dict]]:
        """Dispatch tool calls, returning (tool_results, agent_events)."""
        tool_results = []
        events = []
        for tc in tool_calls:
            self.metrics.tool_calls[tc.name] = self.metrics.tool_calls.get(tc.name, 0) + 1
            result_str = dispatch_tool(tc.name, tc.input, self.task_dir)
            self._log_tool_call(tc, result_str)
            events.append(self._make_tool_event(tc, result_str))

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
        return tool_results, events

    def _make_tool_event(self, tc, result_str: str) -> dict:
        """Build a structured AgentEvent dict for a completed tool call."""
        inp = tc.input
        match tc.name:
            case "read_file":
                return {"type": "tool_call", "tool": "read_file", "path": inp.get("path", "?")}
            case "write_file":
                content = inp.get("content", "")
                lines = content.count("\n") + 1 if content else 0
                return {"type": "tool_call", "tool": "write_file",
                        "path": inp.get("path", "?"), "lines": lines}
            case "run_subprocess":
                summary, failing = _parse_pytest_result(result_str)
                passed = _tests_passed(result_str)
                cmd = inp.get("command", [])
                return {"type": "tool_call", "tool": "run_subprocess",
                        "command": " ".join(cmd) if isinstance(cmd, list) else str(cmd),
                        "passed": passed, "summary": summary, "failing": failing[:5]}
            case "context7_docs":
                return {"type": "tool_call", "tool": "context7_docs",
                        "library": inp.get("library", "?"),
                        "query": inp.get("query", "")[:60]}
            case "firecrawl_search":
                return {"type": "tool_call", "tool": "firecrawl_search",
                        "query": inp.get("query", "")[:60]}
            case "firecrawl_scrape":
                return {"type": "tool_call", "tool": "firecrawl_scrape",
                        "url": inp.get("url", "?")}
            case "run_python":
                code = inp.get("code", "")
                return {"type": "tool_call", "tool": "run_python",
                        "code": code.splitlines()[0][:60] if code else "?"}
            case "calculator":
                return {"type": "tool_call", "tool": "calculator",
                        "expression": inp.get("expression", "?"), "result": result_str}
            case _:
                return {"type": "tool_call", "tool": tc.name}

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

    def run(self, user_prompt: str, auto_approve: bool = False, hint: str = "") -> Iterator[dict]:
        """Run the agent, yielding structured AgentEvent dicts.

        Event types emitted in order:
          {"type": "phase", "phase": "test_generation"}
          {"type": "test_generated", "content": str, "test_count": int}
          {"type": "awaiting_approval", "content": str}   # only when auto_approve=False;
                                                           # caller may send a hint string back
          {"type": "phase", "phase": "implementation"}
          {"type": "tool_call", "tool": str, ...}         # one per tool call
          {"type": "coverage", "pct": float}              # only on success with coverage > 0
          {"type": "done", "success": bool, "message": str,
           "failure_reason": str, "failure_category": str}

        For CLI use: when auto_approve=False, send a hint string via generator.send() in
        response to the "awaiting_approval" event to inject it into the implementation prompt.
        For API use: pass hint= directly; auto_approve=True bypasses the pause entirely.
        """
        self.task_dir.mkdir(parents=True, exist_ok=True)

        yield {"type": "phase", "phase": "test_generation"}
        test_content = self.generate_tests(user_prompt)
        yield {"type": "test_generated", "content": test_content,
               "test_count": test_content.count("def test_")}

        if not auto_approve:
            sent = yield {"type": "awaiting_approval", "content": test_content}
            if isinstance(sent, str) and sent:
                hint = sent

        self._logger.info("Starting implementation loop...")
        yield {"type": "phase", "phase": "implementation"}
        yield from self._implementation_gen(user_prompt, test_content, hint=hint)

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
