"""SQLite database layer for microagent.

Replaces file-based storage (TOML prompts, task text files, JSON metrics,
markdown judgments) with a single SQLite database.
"""

import json
import random
import sqlite3
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logger import RunMetrics

DB_PATH = Path(__file__).parent / "microagent.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_sections (
    id INTEGER PRIMARY KEY,
    version_id INTEGER NOT NULL REFERENCES prompt_versions(id),
    section TEXT NOT NULL,
    key TEXT NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(version_id, section, key)
);

CREATE TABLE IF NOT EXISTS eval_prompt_versions (
    id INTEGER PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_prompt_sections (
    id INTEGER PRIMARY KEY,
    version_id INTEGER NOT NULL REFERENCES eval_prompt_versions(id),
    section TEXT NOT NULL,
    key TEXT NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(version_id, section, key)
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL UNIQUE,
    difficulty TEXT NOT NULL DEFAULT 'standard',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL UNIQUE,
    prompts_version TEXT NOT NULL,
    eval_prompts_version TEXT NOT NULL,
    model TEXT NOT NULL,
    max_iterations INTEGER NOT NULL,
    allow_test_revision INTEGER NOT NULL DEFAULT 0,
    task_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_results (
    id INTEGER PRIMARY KEY,
    eval_run_id INTEGER REFERENCES eval_runs(id),
    task_id INTEGER REFERENCES tasks(id),
    task_prompt TEXT NOT NULL,
    task_dir TEXT NOT NULL,
    prompts_version TEXT NOT NULL,
    model TEXT NOT NULL,
    started_at TEXT NOT NULL,
    test_gen_duration_s REAL,
    test_gen_input_tokens INTEGER,
    test_gen_output_tokens INTEGER,
    impl_duration_s REAL,
    impl_llm_calls INTEGER,
    impl_iterations INTEGER,
    impl_pytest_runs INTEGER,
    impl_write_count INTEGER,
    tool_calls TEXT NOT NULL,
    test_revisions_attempted INTEGER,
    test_revisions_approved INTEGER,
    test_revision_reasoning TEXT,
    test_coverage_pct REAL,
    success INTEGER NOT NULL,
    failure_reason TEXT,
    failure_category TEXT,
    api_retries INTEGER,
    total_duration_s REAL,
    total_tool_calls INTEGER
);

CREATE TABLE IF NOT EXISTS eval_judgments (
    id INTEGER PRIMARY KEY,
    eval_run_id INTEGER REFERENCES eval_runs(id),
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Which TOML files to seed as agent prompts and eval prompts on first run.
_SEED_PROMPT_FILES: list[str] = ["v1", "v2", "v2.1", "v2.2", "v2.3", "v2.4"]
_SEED_EVAL_PROMPT_FILES: list[str] = ["eval-v1", "eval-v1.1", "eval-v1.2"]


def _load_toml_prompts(version: str) -> dict:
    """Load a prompts TOML file from the prompts/ directory."""
    path = _PROMPTS_DIR / f"{version}.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


_TASKS_FILE = Path(__file__).parent / "evals" / "tasks.txt"


def _load_tasks_file(path: Path = _TASKS_FILE) -> list[tuple[str, str]]:
    """Parse evals/tasks.txt into a list of (content, difficulty) tuples.

    Format: lines starting with '# <difficulty>' set the current difficulty tier;
    blank lines and lines starting with '#' that aren't difficulty markers are skipped;
    all other lines are task content.
    """
    tasks: list[tuple[str, str]] = []
    difficulty = "standard"
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            tag = line.lstrip("# ").lower()
            if tag in ("easy", "standard", "hard"):
                difficulty = tag
            continue
        tasks.append((line, difficulty))
    return tasks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_db(path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not exist."""
    conn.executescript(_SCHEMA)
    # Migrate: add columns to existing databases
    for col, typedef in [
        ("test_coverage_pct", "REAL"),
        ("api_retries", "INTEGER"),
        ("failure_category", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE task_results ADD COLUMN {col} {typedef}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists in new databases created by _SCHEMA


def seed_if_empty(conn: sqlite3.Connection) -> None:
    """Seed prompts and tasks from bundled data if the DB is empty."""
    row = conn.execute("SELECT COUNT(*) FROM prompt_versions").fetchone()
    if row[0] > 0:
        return  # already seeded

    now = datetime.now(timezone.utc).isoformat()

    # Agent prompts — loaded from prompts/*.toml
    for version in _SEED_PROMPT_FILES:
        save_prompt_version(conn, version, _load_toml_prompts(version), created_at=now)

    # Eval prompts — loaded from prompts/*.toml
    for version in _SEED_EVAL_PROMPT_FILES:
        save_eval_prompt_version(conn, version, _load_toml_prompts(version), created_at=now)

    # Tasks — loaded from evals/tasks.txt
    for content, difficulty in _load_tasks_file():
        conn.execute(
            "INSERT OR IGNORE INTO tasks (content, difficulty, created_at) VALUES (?, ?, ?)",
            (content, difficulty, now),
        )

    conn.commit()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def load_prompts(conn: sqlite3.Connection, version: str) -> dict:
    """Load agent prompts for a given version from the DB."""
    row = conn.execute(
        "SELECT id FROM prompt_versions WHERE version = ?", (version,)
    ).fetchone()
    if row is None:
        raise KeyError(f"Prompt version '{version}' not found in database.")
    version_id = row["id"]
    rows = conn.execute(
        "SELECT section, key, content FROM prompt_sections WHERE version_id = ?",
        (version_id,),
    ).fetchall()
    result: dict = {}
    for r in rows:
        result.setdefault(r["section"], {})[r["key"]] = r["content"]
    return result


def load_eval_prompts(conn: sqlite3.Connection, version: str) -> dict:
    """Load eval prompts for a given version from the DB."""
    row = conn.execute(
        "SELECT id FROM eval_prompt_versions WHERE version = ?", (version,)
    ).fetchone()
    if row is None:
        raise KeyError(f"Eval prompt version '{version}' not found in database.")
    version_id = row["id"]
    rows = conn.execute(
        "SELECT section, key, content FROM eval_prompt_sections WHERE version_id = ?",
        (version_id,),
    ).fetchall()
    result: dict = {}
    for r in rows:
        result.setdefault(r["section"], {})[r["key"]] = r["content"]
    return result


def save_prompt_version(
    conn: sqlite3.Connection,
    version: str,
    prompts: dict,
    created_at: str | None = None,
) -> None:
    """Save (or replace) an agent prompt version in the DB."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO prompt_versions (version, created_at) VALUES (?, ?)",
        (version, created_at),
    )
    row = conn.execute(
        "SELECT id FROM prompt_versions WHERE version = ?", (version,)
    ).fetchone()
    version_id = row["id"]
    conn.execute(
        "DELETE FROM prompt_sections WHERE version_id = ?", (version_id,)
    )
    for section, keys in prompts.items():
        if isinstance(keys, dict):
            for key, content in keys.items():
                conn.execute(
                    "INSERT INTO prompt_sections (version_id, section, key, content) VALUES (?, ?, ?, ?)",
                    (version_id, section, key, content),
                )
    conn.commit()


def save_eval_prompt_version(
    conn: sqlite3.Connection,
    version: str,
    prompts: dict,
    created_at: str | None = None,
) -> None:
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO eval_prompt_versions (version, created_at) VALUES (?, ?)",
        (version, created_at),
    )
    row = conn.execute(
        "SELECT id FROM eval_prompt_versions WHERE version = ?", (version,)
    ).fetchone()
    version_id = row["id"]
    conn.execute(
        "DELETE FROM eval_prompt_sections WHERE version_id = ?", (version_id,)
    )
    for section, keys in prompts.items():
        if isinstance(keys, dict):
            for key, content in keys.items():
                conn.execute(
                    "INSERT INTO eval_prompt_sections (version_id, section, key, content) VALUES (?, ?, ?, ?)",
                    (version_id, section, key, content),
                )
    conn.commit()


def prompts_to_toml_text(prompts: dict, version: str) -> str:
    """Reconstruct a TOML-formatted string from a prompts dict (for prompt optimizer)."""
    lines = [f'version = "{version}"\n']
    for section, keys in prompts.items():
        lines.append(f"[{section}]")
        if isinstance(keys, dict):
            for key, value in keys.items():
                escaped = str(value).replace('"""', r'\"\"\"')
                lines.append(f'{key} = """\n{escaped}\n"""')
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def get_random_tasks(conn: sqlite3.Connection, n: int) -> list[str]:
    """Return n randomly sampled task content strings from the DB."""
    rows = conn.execute("SELECT content FROM tasks").fetchall()
    contents = [r["content"] for r in rows]
    if n >= len(contents):
        return contents
    return random.sample(contents, n)


def add_task(
    conn: sqlite3.Connection, content: str, difficulty: str = "standard"
) -> None:
    """Add a new task to the DB (ignored if content already exists)."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO tasks (content, difficulty, created_at) VALUES (?, ?, ?)",
        (content, difficulty, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Eval runs
# ---------------------------------------------------------------------------


def save_eval_run(
    conn: sqlite3.Connection,
    timestamp: str,
    prompts_version: str,
    eval_prompts_version: str,
    model: str,
    max_iterations: int,
    allow_test_revision: bool,
    task_count: int,
) -> int:
    """Insert an eval run record and return its id."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO eval_runs
           (timestamp, prompts_version, eval_prompts_version, model,
            max_iterations, allow_test_revision, task_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            timestamp,
            prompts_version,
            eval_prompts_version,
            model,
            max_iterations,
            int(allow_test_revision),
            task_count,
            now,
        ),
    )
    conn.commit()
    return cur.lastrowid


def save_task_result(
    conn: sqlite3.Connection,
    metrics: "RunMetrics",
    eval_run_id: int | None = None,
    task_id: int | None = None,
) -> int:
    """Insert a task result row from a RunMetrics object and return its id."""
    cur = conn.execute(
        """INSERT INTO task_results
           (eval_run_id, task_id, task_prompt, task_dir, prompts_version, model,
            started_at, test_gen_duration_s, test_gen_input_tokens, test_gen_output_tokens,
            impl_duration_s, impl_llm_calls, impl_iterations, impl_pytest_runs,
            impl_write_count, tool_calls, test_revisions_attempted,
            test_revisions_approved, test_revision_reasoning,
            test_coverage_pct, success, failure_reason, failure_category, api_retries,
            total_duration_s, total_tool_calls)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            eval_run_id,
            task_id,
            metrics.task_prompt,
            metrics.task_dir,
            metrics.prompts_version,
            metrics.model,
            metrics.started_at,
            round(metrics.test_gen_duration_s, 3),
            metrics.test_gen_input_tokens,
            metrics.test_gen_output_tokens,
            round(metrics.impl_duration_s, 3),
            metrics.impl_llm_calls,
            metrics.impl_iterations,
            metrics.impl_pytest_runs,
            metrics.impl_write_count,
            json.dumps(metrics.tool_calls),
            metrics.test_revisions_attempted,
            metrics.test_revisions_approved,
            metrics.test_revision_reasoning,
            round(metrics.test_coverage_pct, 1),
            int(metrics.success),
            metrics.failure_reason,
            metrics.failure_category,
            metrics.api_retries,
            round(metrics.total_duration_s, 3),
            metrics.total_tool_calls,
        ),
    )
    conn.commit()
    return cur.lastrowid


def save_judgment(
    conn: sqlite3.Connection,
    eval_run_id: int,
    type: str,
    content: str,
) -> None:
    """Insert a judgment record."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO eval_judgments (eval_run_id, type, content, created_at) VALUES (?, ?, ?, ?)",
        (eval_run_id, type, content, now),
    )
    conn.commit()


def get_eval_summary(conn: sqlite3.Connection, eval_run_id: int) -> dict:
    """Return a summary dict for a completed eval run."""
    run = conn.execute(
        "SELECT * FROM eval_runs WHERE id = ?", (eval_run_id,)
    ).fetchone()
    results = conn.execute(
        "SELECT * FROM task_results WHERE eval_run_id = ?", (eval_run_id,)
    ).fetchall()
    passed = sum(1 for r in results if r["success"])
    return {
        "eval_run_id": eval_run_id,
        "timestamp": run["timestamp"] if run else None,
        "task_count": run["task_count"] if run else 0,
        "passed": passed,
        "failed": len(results) - passed,
    }
