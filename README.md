# microagent

A test-first AI coding agent powered by Claude. Given a task description, it generates a locked pytest test file, then iteratively writes and debugs an implementation until all tests pass.

## How it works

```
Phase 1: Test generation
  User prompt → Claude → solution_test.py (locked)

Phase 2: Implementation loop
  Claude (with tools) → solution.py → pytest → reflect → repeat
```

The test file is generated once and never modified (unless `--allow-test-revision` is set — see below). The implementation loop runs until pytest passes or `--max-iterations` is hit. A final ground-truth pytest run verifies the result regardless of what the agent reports.

## Setup

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
FIRECRAWL_API_KEY=fc-...       # optional — needed for web search/scrape tools
CONTEXT7_API_KEY=...           # optional — needed for library docs tool
```

## Usage

### Single task

```bash
uv run python microagent.py "Write a function that reverses a string"
uv run python microagent.py "Write a Fibonacci function" --task-dir ./fib-task
uv run python microagent.py "Write a reverse string function" --prompts v1
uv run python microagent.py "Write RLE encode/decode" --allow-test-revision 3
```

**Arguments:**

| Flag                    | Default                 | Description                                                               |
| ----------------------- | ----------------------- | ------------------------------------------------------------------------- |
| `prompt`                | _(required)_            | Task description                                                          |
| `--task-dir`            | `task-YYYYMMDD-HHMMSS/` | Directory for task files                                                  |
| `--model`               | `claude-sonnet-4-6`     | Claude model to use                                                       |
| `--max-iterations`      | `10`                    | Max write→run cycles before giving up                                     |
| `--prompts`             | `v1`                    | Agent prompts version (`prompts/<version>.toml`)                          |
| `--allow-test-revision` | `0` (disabled)          | After N failing iterations, offer the agent a chance to revise the tests  |

After generating tests, the agent pauses and shows `solution_test.py` before starting implementation. Press Enter to continue or Ctrl+C to abort.

#### Test revision

When `--allow-test-revision N` is set, after N consecutive failing iterations the agent is shown the test file and asked whether it believes the tests contain an error. If it writes a revised `solution_test.py`, **the terminal shows a full diff and prompts for explicit approval** before the new tests take effect. Denied revisions restore the original file. The iteration counter resets to 0 on approval.

**Task directory output:**

```
task-20240613-120000/
├── solution_test.py    # generated test file (locked)
├── solution.py         # final implementation
├── run.log             # timestamped DEBUG-level log
└── metrics.json        # timing, token usage, tool call counts
```

Console output shows a structured log of every tool call:

```
  → read_file: solution_test.py
  → write_file: solution.py (45 lines)
  → run_subprocess: pytest solution_test.py -v
    ✗ == 2 failed in 0.12s == | failing: test_encode, test_roundtrip
  → run_subprocess: pytest solution_test.py -v
    ✓ == 3 passed in 0.09s ==
```

---

### Evaluation harness

```bash
uv run python eval.py                                  # run v1 prompts on all 10 tasks
uv run python eval.py --tasks 3                        # run first 3 tasks
uv run python eval.py --prompts v2                     # use a different agent prompt version
uv run python eval.py --compare v2                     # A/B test v1 vs v2
uv run python eval.py --max-iter 5                     # limit iterations per task
uv run python eval.py --optimize                       # save an improved prompt TOML after judging
uv run python eval.py --meta-judge                     # also evaluate the judge's output quality
uv run python eval.py --eval-prompts eval-v2           # use a different judge/optimizer prompt version
uv run python eval.py --tasks-file v2                  # use a different task list
```

The eval harness runs a suite of 10 harder coding tasks (algorithms, data structures, parsing, etc.), collects `RunMetrics` for each, then calls a Claude judge for analysis and prompt improvement suggestions.

**`--optimize`** — After the judge runs, calls Claude with a prompt-engineer system prompt to produce an improved agent TOML. Validates it parses as TOML, then saves it to `prompts/YYYYMMDD-HHMMSS.toml`. These timestamped files are gitignored by default.

```
Optimized prompts saved → prompts/20260313-143022.toml
Test with: uv run python eval.py --prompts 20260313-143022
```

**`--meta-judge`** — After the primary judge runs, calls a second Claude instance to evaluate the judge's own output quality: coverage of all template sections, specificity of citations, actionability of prompt improvement suggestions (quoting exact text vs. vague direction), calibration against actual pass rates, and structural weaknesses in the judge prompts themselves. Saves `meta_judgment.md` alongside `judgment.md`.

**Eval output:**

```
eval-20240613-120000/
├── task-01/ ... task-10/   # per-task dirs (solution_test.py, solution.py, run.log, metrics.json)
├── results.json             # all run metrics + _meta (eval_prompts_version, tasks_version)
├── eval_metrics.json        # per-task + aggregate summary
├── judgment.md              # Claude judge evaluation
└── meta_judgment.md         # meta-judge evaluation of the judge (--meta-judge only)
```

`results.json` structure:

```json
{
  "_meta": { "eval_prompts_version": "eval-v1", "tasks_version": "v1" },
  "summary": {
    "tasks": 10,
    "passed": 9,
    "failed": 1,
    "avg_iterations": 1.2,
    "avg_total_duration_s": 28.4,
    "avg_test_gen_duration_s": 4.1,
    "avg_impl_duration_s": 24.3,
    "total_tool_calls_by_type": {"read_file": 10, "write_file": 11, "run_subprocess": 12},
    "avg_impl_llm_calls": 2.1
  },
  "runs": [...]
}
```

---

## File structure

```
microagent/
├── agent.py          # AgentLoop: test generation + implementation loop + metrics
├── eval.py           # Evaluation harness: task suite, judge, A/B comparison, meta-judge
├── logger.py         # RunMetrics dataclass, setup_logging(), save_metrics()
├── microagent.py     # CLI entry point
├── tools.py          # Tool schemas + implementations (read, write, pytest, docs, search, calc)
├── evals/
│   └── tasks-v1.txt  # Eval task list (one task per line, versioned)
├── prompts/
│   ├── v1.toml       # Agent prompt templates (test generation + implementation + test revision)
│   └── eval-v1.toml  # Eval prompt templates (judge, A/B judge, optimizer, meta-judge)
└── tests/
    ├── conftest.py   # Shared fixtures and mock helpers
    ├── test_agent.py
    ├── test_eval.py
    └── test_microagent.py
```

---

## Prompts

### Agent prompts (`prompts/v1.toml`)

Loaded at runtime using Python's stdlib `tomllib`. To iterate:

```bash
cp prompts/v1.toml prompts/v2.toml
# edit prompts/v2.toml
uv run python microagent.py "your task" --prompts v2
uv run python eval.py --compare v2     # A/B test v1 vs v2
```

| Section               | Key        | Used as                                                                  |
| --------------------- | ---------- | ------------------------------------------------------------------------ |
| `[test_generation]`   | `system`   | System prompt for Phase 1                                                |
| `[test_generation]`   | `user`     | User message — `{user_prompt}`                                           |
| `[implementation]`    | `system`   | System prompt for Phase 2                                                |
| `[implementation]`    | `user`     | User message — `{user_prompt}`, `{test_content}`, `{prompt_md_section}` |
| `[prompt_md_section]` | `template` | Injected when `.prompt.md` exists — `{prompt_md}`                       |
| `[test_revision]`     | `user`     | Injected after N failures — `{n}` (iteration count)                     |

### Eval prompts (`prompts/eval-v1.toml`)

Controls the judge, A/B judge, prompt optimizer, and meta-judge. Can be versioned and iterated independently of agent prompts:

```bash
cp prompts/eval-v1.toml prompts/eval-v2.toml
# edit prompts/eval-v2.toml
uv run python eval.py --eval-prompts eval-v2
```

| Section              | Keys              | Used as                                              |
| -------------------- | ----------------- | ---------------------------------------------------- |
| `[judge]`            | `system`          | System prompt for the judge call                     |
| `[judge_single]`     | `template`        | User message for single-version eval                 |
| `[judge_ab]`         | `template`        | User message for A/B comparison                      |
| `[prompt_optimizer]` | `system`          | System prompt for `--optimize` prompt generation     |
| `[meta_judge]`       | `system`,`template` | System + user prompts for `--meta-judge` call      |

### Task lists (`evals/tasks-v1.txt`)

One task per line. To add a new task set:

```bash
cp evals/tasks-v1.txt evals/tasks-v2.txt
# edit evals/tasks-v2.txt
uv run python eval.py --tasks-file v2
```

---

## Agent tools

| Tool               | Description                                         |
| ------------------ | --------------------------------------------------- |
| `read_file`        | Read a file in the task directory                   |
| `write_file`       | Write `solution.py` (only file the agent may write) |
| `run_subprocess`   | Run pytest — only pytest commands allowed           |
| `context7_docs`    | Fetch structured library documentation              |
| `firecrawl_scrape` | Scrape a URL as markdown                            |
| `firecrawl_search` | Web search with content snippets                    |
| `calculator`       | Safe AST-based math expression evaluator            |

---

## `.prompt.md` — additional context

Place a `.prompt.md` file in the working directory before running and it will be copied into the task dir as `solution.prompt.md`. Its contents are injected into the implementation prompt as `<additional_context>`. Useful for passing constraints, style guides, or library preferences.

---

## Metrics

Each run produces `metrics.json` in the task directory:

```json
{
  "task_prompt": "Write a function that reverses a string",
  "task_dir": "task-20240613-120000",
  "prompts_version": "v1",
  "model": "claude-sonnet-4-6",
  "started_at": "2024-06-13T12:00:00+00:00",
  "test_gen_duration_s": 3.2,
  "test_gen_input_tokens": 412,
  "test_gen_output_tokens": 187,
  "impl_duration_s": 18.5,
  "impl_llm_calls": 3,
  "impl_iterations": 2,
  "impl_pytest_runs": 2,
  "impl_write_count": 2,
  "tool_calls": { "read_file": 1, "write_file": 2, "run_subprocess": 2 },
  "test_revisions_attempted": 0,
  "test_revisions_approved": 0,
  "test_revision_reasoning": "",
  "success": true,
  "failure_reason": "",
  "total_duration_s": 21.7,
  "total_tool_calls": 5
}
```

---

## Tests

```bash
uv run pytest               # runs with coverage (configured in pyproject.toml)
uv run pytest -q            # quiet output
uv run pytest tests/test_agent.py   # single file
```

Coverage is enforced at 98% (`--cov-fail-under=98`). Current coverage: **99.6%** across `agent.py`, `eval.py`, and `microagent.py`.

---

## Logging

Console output is INFO-level with structured per-tool lines. Each task directory contains `run.log` with DEBUG-level timestamped entries including token counts and final metrics summary.
