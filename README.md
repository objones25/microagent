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

The database (`microagent.db`) is created automatically on first run and seeded with all prompt versions and 100 tasks.

## Usage

### Single task

```bash
uv run python microagent.py "Write a function that reverses a string"
uv run python microagent.py "Write a Fibonacci function" --task-dir ./fib-task
uv run python microagent.py "Write a reverse string function" --prompts v2.2
uv run python microagent.py "Write RLE encode/decode" --allow-test-revision
```

**Arguments:**

| Flag                      | Default                 | Description                                                                    |
| ------------------------- | ----------------------- | ------------------------------------------------------------------------------ |
| `prompt`                  | _(required)_            | Task description                                                               |
| `--task-dir`              | `task-YYYYMMDD-HHMMSS/` | Directory for task files                                                       |
| `--model`                 | `claude-sonnet-4-6`     | Claude model to use                                                            |
| `--max-iterations`        | `10`                    | Max write→run cycles before giving up                                          |
| `--prompts`               | `v2.8`                  | Agent prompts version (key in `microagent.db`, seeded from `prompts/<v>.toml`) |
| `--allow-test-revision`   | off                     | When the agent stops without passing, offer it a chance to revise the tests    |
| `--auto-approve-revision` | off                     | Automatically approve test revisions without prompting                         |

After generating tests, the agent pauses and shows `solution_test.py` before starting implementation. Press Enter to continue or Ctrl+C to abort.

#### Test revision

When `--allow-test-revision` is set, if the agent stops without passing all tests, it is shown the test file and asked whether the tests contain a factual error. The agent must use the `calculator` tool to verify any expected values before rewriting — it cannot modify tests without proof. If it writes a revised `solution_test.py`, **the terminal shows a full diff and prompts for explicit approval** before the new tests take effect. Denied revisions restore the original file. The iteration counter resets on approval.

Add `--auto-approve-revision` to skip the interactive prompt (useful for eval runs).

**Task directory output:**

```
task-20240613-120000/
├── solution_test.py    # generated test file (locked)
├── solution.py         # final implementation
└── run.log             # timestamped DEBUG-level log
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
uv run python eval.py                                                    # 10 random tasks, v1 prompts
uv run python eval.py --tasks 5                                          # 5 randomly-sampled tasks
uv run python eval.py --prompts v2.2                                     # use a different prompt version
uv run python eval.py --compare v2.2                                     # A/B test v1 vs v2.2
uv run python eval.py --max-iter 5                                       # limit iterations per task
uv run python eval.py --out results.json                                 # save raw results to file
uv run python eval.py --optimize                                         # save an improved prompt version to DB after judging
uv run python eval.py --meta-judge                                       # also evaluate the judge's output quality
uv run python eval.py --eval-prompts eval-v1.2                           # use a different judge/optimizer prompt version
uv run python eval.py --allow-test-revision --auto-approve-revision      # enable non-interactive test revision
```

Tasks are randomly sampled from the DB pool (100 tasks across easy/standard/hard tiers) each run.

**Arguments:**

| Flag                      | Default   | Description                                                          |
| ------------------------- | --------- | -------------------------------------------------------------------- |
| `--prompts`               | `v2.8`    | Agent prompts version to evaluate                                    |
| `--compare`               | —         | Second prompts version for A/B test                                  |
| `--tasks`                 | `10`      | Number of tasks to randomly sample from DB                           |
| `--max-iter`              | `5`       | Max implementation iterations per task                               |
| `--out`                   | —         | Write raw results JSON to this path                                  |
| `--optimize`              | off       | After judging, generate an improved prompt version and save it to DB |
| `--meta-judge`            | off       | After judging, evaluate the judge's own output quality               |
| `--eval-prompts`          | `eval-v1.4` | Judge/optimizer/meta-judge prompts version                         |
| `--allow-test-revision`   | off       | Allow agent to revise tests when stuck                               |
| `--auto-approve-revision` | off       | Auto-approve test revisions (required for non-interactive eval use)  |

**`--optimize`** — After the judge runs, calls Claude with a prompt-engineer system prompt to produce an improved agent prompt set. Validates it parses as valid TOML, then saves it as a new version in the DB (key = `YYYYMMDD-HHMMSS`).

```
Optimized prompts saved → version: 20260313-143022
Test with: uv run python eval.py --prompts 20260313-143022
```

**`--meta-judge`** — After the primary judge runs, calls a second Claude instance to evaluate the judge's own output quality: coverage, specificity of citations, actionability of prompt improvements, calibration against actual pass rates, and structural weaknesses in the judge prompts.

**Eval output:**

```
eval-20240613-120000/
└── task-01/ ... task-10/   # per-task dirs (solution_test.py, solution.py, run.log)
```

Results, judgments, and metrics are stored in `microagent.db` (`eval_runs`, `task_results`, `eval_judgments` tables). Pass `--out results.json` to also write a JSON snapshot.

---

## Database

All persistent state lives in `microagent.db` (SQLite, auto-created on first run):

| Table                  | Contents                                                    |
| ---------------------- | ----------------------------------------------------------- |
| `prompt_versions`      | Agent prompt version names and creation timestamps          |
| `prompt_sections`      | Section/key/content rows for each prompt version            |
| `eval_prompt_versions` | Eval prompt version names                                   |
| `eval_prompt_sections` | Section/key/content rows for each eval prompt version       |
| `tasks`                | 100 coding tasks with difficulty tier (easy/standard/hard)  |
| `eval_runs`            | One row per eval invocation (config, timestamp, task count) |
| `task_results`         | One row per task run (all RunMetrics fields)                |
| `eval_judgments`       | Judge/meta-judge text outputs, typed `single`/`ab`/`meta`   |

The DB is seeded on first run from `prompts/*.toml` (prompt versions) and `evals/tasks.txt` (task pool). To add tasks, edit `evals/tasks.txt` and delete `microagent.db` to reseed.

---

## File structure

```
microagent/
├── agent.py          # AgentLoop + AgentConfig: test generation + implementation loop + metrics
├── db.py             # SQLite schema, seed, CRUD layer + Task dataclass
├── eval.py           # Evaluation harness: task suite, judge, A/B comparison, meta-judge
├── logger.py         # RunMetrics dataclass, setup_logging(), save_metrics()
├── microagent.py     # CLI entry point
├── tools.py          # Tool schemas + implementations (read, write, pytest, docs, search, run_python, calc)
├── evals/
│   ├── tasks.txt     # Task pool — 100 tasks across easy/standard/hard tiers
│   ├── tasks-v1.txt  # Legacy task list (v1, 10 tasks)
│   └── tasks-v2.txt  # Legacy task list (v2, 10 tasks)
├── prompts/
│   ├── v1.toml       # Agent prompts v1 (baseline)
│   ├── v2.toml       # Agent prompts v2
│   ├── v2.1.toml     # Agent prompts v2.1
│   ├── v2.2.toml     # Agent prompts v2.2
│   ├── v2.3.toml     # Agent prompts v2.3
│   ├── v2.4.toml     # Agent prompts v2.4
│   ├── v2.5.toml     # Agent prompts v2.5
│   ├── v2.6.toml     # Agent prompts v2.6
│   ├── v2.7.toml     # Agent prompts v2.7
│   ├── v2.8.toml     # Agent prompts v2.8 (current best)
│   ├── eval-v1.toml  # Eval prompts v1
│   ├── eval-v1.1.toml # Eval prompts v1.1
│   ├── eval-v1.2.toml # Eval prompts v1.2
│   ├── eval-v1.3.toml # Eval prompts v1.3
│   └── eval-v1.4.toml # Eval prompts v1.4 (current best)
└── tests/
    ├── conftest.py   # Shared fixtures and mock helpers
    ├── test_agent.py
    ├── test_eval.py
    └── test_microagent.py
```

---

## Prompts

### Agent prompts

Loaded from `microagent.db` at runtime (seeded from `prompts/*.toml` on first run). To iterate:

```bash
cp prompts/v2.7.toml prompts/v3.toml
# edit prompts/v3.toml, then reseed:
rm microagent.db
uv run python eval.py --compare v3     # A/B test v2.7 vs v3
```

| Section               | Key        | Used as                                                                 |
| --------------------- | ---------- | ----------------------------------------------------------------------- |
| `[test_generation]`   | `system`   | System prompt for Phase 1                                               |
| `[test_generation]`   | `user`     | User message — `{user_prompt}`                                          |
| `[implementation]`    | `system`   | System prompt for Phase 2                                               |
| `[implementation]`    | `user`     | User message — `{user_prompt}`, `{test_content}`, `{prompt_md_section}` |
| `[prompt_md_section]` | `template` | Injected when `.prompt.md` exists — `{prompt_md}`                       |
| `[test_revision]`     | `user`     | Injected when agent stops without passing (if `--allow-test-revision`)  |

**Prompt version history:**

| Version | Key changes                                                                                                                                        |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v1`    | Baseline                                                                                                                                           |
| `v2`    | `from solution import MyClass` rule; forbid `or` in asserts; round-trip testing for encodings; conservative timing tests                           |
| `v2.1`  | Derivation comments for numeric assertions; concrete always-True anti-pattern list; algorithm variant verification step; function name consistency |
| `v2.2`  | Strengthened `[test_revision]`: requires calculator proof before rewriting tests; defaults to fixing implementation                                |
| `v2.3`  | Tautological test rules: ban captured-but-unasserted return values; separate in-place mutation vs return value tests; empty-argument boundary tests; precondition violation tests; derivation comments extended to graph algorithms; implementation Python version comments |
| `v2.4`  | Based on auto-optimized `20260313-125033`: membership-assertion verification before writing `not in` tests; combinatorial formula verification with small-n enumeration; type-only assertions must be in same function as value assertion; `n=0` constructor tests for data structures; minimal-instance interface contract tests; `TEST_BUG:` must cite exact assertion line and correct value |
| `v2.5`  | Complexity guard tests for tasks with specified O-complexity; strengthened general DP cross-check (restored from stock-trading-specific narrowing); stricter palindrome membership brute-force rule with index-position verification |
| `v2.6`  | Best-of-breed from v2.5 + auto-optimized variants: bare `try/except: pass` prohibition extended to all nesting levels; `time.sleep`-based thread pool tests explicitly forbidden when a future/result blocking interface is available; palindrome substring membership requires index enumeration; restored general DP cross-check rule |
| `v2.7`  | Agent forbidden from requesting test modification; complexity guard n-values calibrated per complexity class (100k/10k/50k); `or`-prohibition extended to set-membership assertions and search return values; segment-tree loser-propagation adversarial test required; implementation must declare complexity class in comment before writing code |
| `v2.8`  | `or`-prohibition extended to tuple-valued returns with implementation-defined elements; bare-pass prohibition extended to "empty state" and "optional feature" patterns (`assert result == expected or result is None` banned); mandatory zero/boundary parameter tests made explicit standalone rule; loop-variable assertion prohibition (iterating result without asserting is forbidden); token budget safety rule: output validity gate requires syntactically complete last function |

### Eval prompts

Controls the judge, A/B judge, prompt optimizer, and meta-judge:

```bash
cp prompts/eval-v1.4.toml prompts/eval-v2.toml
# edit, reseed, then use:
rm microagent.db && uv run python eval.py --eval-prompts eval-v2
```

| Section              | Keys                 | Used as                                          |
| -------------------- | -------------------- | ------------------------------------------------ |
| `[judge]`            | `system`             | System prompt for the judge call                 |
| `[judge_single]`     | `template`           | User message for single-version eval             |
| `[judge_ab]`         | `template`           | User message for A/B comparison                  |
| `[prompt_optimizer]` | `system`             | System prompt for `--optimize` prompt generation |
| `[meta_judge]`       | `system`, `template` | System + user prompts for `--meta-judge` call    |

**Eval prompt version history:**

| Version     | Key changes                                                                                                                                                                                                                            |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eval-v1`   | Baseline                                                                                                                                                                                                                               |
| `eval-v1.1` | Verification constraints (no fabricating text/tools); placeholder guards (EVALUATION BLOCKED if vars unfilled); implementation depth requirements; data reconciliation; system-level design section; REPLACEMENT/NEW ADDITION labeling |
| `eval-v1.2` | Adds `api_retries` and `failure_category` to judge context; `failure_category_counts` breakdown in A/B summary; judge instructed to distinguish API errors from agent failures |
| `eval-v1.3` | Strengthened placeholder guards (`{v1_results_block}`, `{v2_results_block}` added to block list); OBSERVED/INFERRED labels required per-claim with source location; task-count range must be stated before verification; thread-pool and complexity-guard evaluation criteria added |
| `eval-v1.4` | Adds test-revision mechanism description to agent workflow; judge required to flag revision-assisted passes separately, compare original vs. revised test counts, and assess whether revision was legitimate |

### Task pool

100 tasks in `evals/tasks.txt`, one per line, grouped by difficulty tier (`# easy`, `# standard`, `# hard`). Tasks are randomly sampled each eval run. To add tasks, edit the file and reseed the DB.

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
| `run_python`       | Execute a Python snippet and return stdout/stderr   |
| `calculator`       | Safe AST-based math evaluator (supports string literals: `len("hello")`) |

---

## `.prompt.md` — additional context

Place a `.prompt.md` file in the working directory before running and it will be copied into the task dir as `solution.prompt.md`. Its contents are injected into the implementation prompt as `<additional_context>`. Useful for passing constraints, style guides, or library preferences.

---

## Tests

```bash
uv run pytest               # runs with coverage (configured in pyproject.toml)
uv run pytest -q            # quiet output
uv run pytest tests/test_agent.py   # single file
```

Coverage is enforced at 98% (`--cov-fail-under=98`). Current coverage: **99%** across `agent.py`, `eval.py`, and `microagent.py`.

---

## Logging

Console output is INFO-level with structured per-tool lines. Each task directory contains `run.log` with DEBUG-level timestamped entries including token counts and a final metrics summary.
