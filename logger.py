import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import db as db_module


@dataclass
class RunMetrics:
    task_prompt: str
    task_dir: str
    prompts_version: str
    model: str
    started_at: str  # ISO 8601

    # Phase 1 — test generation
    test_gen_duration_s: float = 0.0
    test_gen_input_tokens: int = 0
    test_gen_output_tokens: int = 0

    # Phase 2 — implementation loop
    impl_duration_s: float = 0.0
    impl_llm_calls: int = 0
    impl_iterations: int = 0
    impl_pytest_runs: int = 0
    impl_write_count: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)

    # Test revision
    test_revisions_attempted: int = 0
    test_revisions_approved: int = 0
    test_revision_reasoning: str = ""

    # Test coverage (measured after successful ground-truth run)
    test_coverage_pct: float = 0.0

    # API retry tracking
    api_retries: int = 0

    # Outcome
    success: bool = False
    failure_reason: str = ""
    failure_category: str = ""   # "api_error" | "max_iterations" | "coverage" | "test_failure" | ""

    @property
    def total_duration_s(self) -> float:
        return self.test_gen_duration_s + self.impl_duration_s

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())

    def to_dict(self) -> dict:
        return {
            "task_prompt": self.task_prompt,
            "task_dir": self.task_dir,
            "prompts_version": self.prompts_version,
            "model": self.model,
            "started_at": self.started_at,
            "test_gen_duration_s": round(self.test_gen_duration_s, 3),
            "test_gen_input_tokens": self.test_gen_input_tokens,
            "test_gen_output_tokens": self.test_gen_output_tokens,
            "impl_duration_s": round(self.impl_duration_s, 3),
            "impl_llm_calls": self.impl_llm_calls,
            "impl_iterations": self.impl_iterations,
            "impl_pytest_runs": self.impl_pytest_runs,
            "impl_write_count": self.impl_write_count,
            "tool_calls": self.tool_calls,
            "test_revisions_attempted": self.test_revisions_attempted,
            "test_revisions_approved": self.test_revisions_approved,
            "test_revision_reasoning": self.test_revision_reasoning,
            "test_coverage_pct": round(self.test_coverage_pct, 1),
            "api_retries": self.api_retries,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "failure_category": self.failure_category,
            "total_duration_s": round(self.total_duration_s, 3),
            "total_tool_calls": self.total_tool_calls,
        }


def setup_logging(task_dir: Path) -> logging.Logger:
    """
    Returns a logger named 'microagent' with two handlers:
    - StreamHandler (console): INFO level, plain text
    - FileHandler (task_dir/run.log): DEBUG level, timestamped
    """
    logger = logging.getLogger("microagent")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(task_dir / "run.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def save_metrics(
    metrics: RunMetrics,
    task_dir: Path,
    conn: sqlite3.Connection | None = None,
    eval_run_id: int | None = None,
) -> None:
    if conn is not None:
        import db
        db.save_task_result(conn, metrics, eval_run_id=eval_run_id)

