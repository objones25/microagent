# microagent

A test-first AI coding agent powered by Claude. Given a task description, it generates a locked pytest test file, then iteratively writes and debugs an implementation until all tests pass.

## How it works

```
Phase 1: Test generation
  User prompt → Claude → solution_test.py (locked)

Phase 2: Implementation loop
  Claude (with tools) → solution.py → pytest → reflect → repeat
```

The test file is generated once and never modified. The implementation loop runs until pytest passes or `--max-iterations` is hit. A final ground-truth pytest run verifies the result regardless of what the agent reports.

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
```

**Arguments:**

| Flag               | Default                 | Description                                |
| ------------------ | ----------------------- | ------------------------------------------ |
| `prompt`           | _(required)_            | Task description                           |
| `--task-dir`       | `task-YYYYMMDD-HHMMSS/` | Directory for task files                   |
| `--model`          | `claude-sonnet-4-6`     | Claude model to use                        |
| `--max-iterations` | `10`                    | Max write→run cycles before giving up      |
| `--prompts`        | `v1`                    | Prompts version (`prompts/<version>.toml`) |

After generating tests, the agent pauses and shows `solution_test.py` before starting implementation. Press Enter to continue or Ctrl+C to abort.

**Task directory output:**

```
task-20240613-120000/
├── solution_test.py    # generated test file (locked)
├── solution.py         # final implementation
├── run.log             # timestamped DEBUG-level log
└── metrics.json        # timing, token usage, tool call counts
```

### Evaluation harness

```bash
uv run python eval.py                        # run v1 prompts on all 10 tasks
uv run python eval.py --tasks 3              # run first 3 tasks
uv run python eval.py --prompts v2           # use a different prompt version
uv run python eval.py --compare v2           # A/B test v1 vs v2
uv run python eval.py --max-iter 5           # limit iterations per task
```

The eval harness runs a suite of 10 harder coding tasks (algorithms, data structures, parsing, etc.), collects `RunMetrics` for each, then calls a Claude judge for analysis and prompt improvement suggestions.

**Eval output:**

```
eval-20240613-120000/
├── task-01/ ... task-10/   # per-task dirs (solution_test.py, solution.py, run.log, metrics.json)
├── results.json             # all run metrics
├── eval_metrics.json        # per-task + aggregate summary
└── judgment.md              # Claude judge evaluation
```

`eval_metrics.json` structure:

```json
{
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

## File structure

```
microagent/
├── agent.py          # AgentLoop: test generation + implementation loop + metrics
├── eval.py           # Evaluation harness: run task suite, judge, A/B comparison
├── logger.py         # RunMetrics dataclass, setup_logging(), save_metrics()
├── microagent.py     # CLI entry point
├── tools.py          # Tool schemas + implementations (read, write, pytest, docs, search, calc)
└── prompts/
    └── v1.toml       # Prompt templates (test generation + implementation system/user prompts)
```

## Prompts

Prompts live in `prompts/<version>.toml` and are loaded at runtime using Python's stdlib `tomllib`. To create a new prompt version:

```bash
cp prompts/v1.toml prompts/v2.toml
# edit prompts/v2.toml
uv run python microagent.py "your task" --prompts v2
uv run python eval.py --compare v2   # A/B test v1 vs v2
```

The TOML file has four sections:

| Section               | Key        | Used as                                                                 |
| --------------------- | ---------- | ----------------------------------------------------------------------- |
| `[test_generation]`   | `system`   | System prompt for Phase 1                                               |
| `[test_generation]`   | `user`     | User message — `{user_prompt}` placeholder                              |
| `[implementation]`    | `system`   | System prompt for Phase 2                                               |
| `[implementation]`    | `user`     | User message — `{user_prompt}`, `{test_content}`, `{prompt_md_section}` |
| `[prompt_md_section]` | `template` | Injected into impl user prompt when `.prompt.md` exists — `{prompt_md}` |

## Agent tools

The implementation agent has access to:

| Tool               | Description                                         |
| ------------------ | --------------------------------------------------- |
| `read_file`        | Read a file in the task directory                   |
| `write_file`       | Write `solution.py` (only file the agent may write) |
| `run_subprocess`   | Run pytest — only pytest commands allowed           |
| `context7_docs`    | Fetch structured library documentation              |
| `firecrawl_scrape` | Scrape a URL as markdown                            |
| `firecrawl_search` | Web search with content snippets                    |
| `calculator`       | Safe AST-based math expression evaluator            |

## `.prompt.md` — additional context

Place a `.prompt.md` file in the working directory before running and it will be copied into the task dir as `solution.prompt.md`. Its contents are injected into the implementation prompt as `<additional_context>`. Useful for passing constraints, style guides, or library preferences.

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
  "impl_llm_calls": 2,
  "impl_iterations": 1,
  "tool_calls": { "read_file": 1, "write_file": 1, "run_subprocess": 1 },
  "success": true,
  "failure_reason": "",
  "total_duration_s": 21.7,
  "total_tool_calls": 3
}
```

## Logging

Console output is INFO-level (clean, same as before). Each task directory also contains `run.log` with DEBUG-level timestamped entries including token counts, tool dispatch details, and final metrics.
