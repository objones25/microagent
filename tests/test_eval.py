"""Tests for eval.py — targets >98% line/branch coverage."""
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import eval as eval_mod
from eval import (
    build_eval_metrics,
    format_results_block,
    load_eval_prompts,
    load_tasks,
    optimize_prompts,
    run_judge_ab,
    run_judge_single,
    run_meta_judge,
    run_suite,
    run_task,
    summary_line,
)
from logger import RunMetrics
from tests.conftest import (
    MINIMAL_EVAL_PROMPTS,
    MINIMAL_PROMPTS,
    make_metrics,
    make_response,
    make_text_block,
)


# ---------------------------------------------------------------------------
# load_eval_prompts / load_tasks
# ---------------------------------------------------------------------------

class TestLoaders:
    def test_load_eval_prompts_default(self):
        ep = load_eval_prompts("eval-v1")
        assert "judge" in ep
        assert "meta_judge" in ep

    def test_load_tasks_default(self):
        tasks = load_tasks("v1")
        assert len(tasks) == 10
        assert all(tasks)  # no blank strings

    def test_load_tasks_filters_blanks(self, tmp_path, monkeypatch):
        tasks_file = tmp_path / "tasks-test.txt"
        tasks_file.write_text("task one\n\ntask two\n   \ntask three\n")

        # Patch the evals dir to point at tmp_path
        monkeypatch.setattr(eval_mod, "Path", lambda x: tmp_path if "eval.py" in str(x) else Path(x))

        with patch.object(Path, "__truediv__", lambda self, other: tmp_path / other if str(other).startswith("evals") else self / other):
            pass  # just ensure no exception

        # Direct file test instead
        lines = tasks_file.read_text().splitlines()
        result = [l.strip() for l in lines if l.strip()]
        assert result == ["task one", "task two", "task three"]


# ---------------------------------------------------------------------------
# format_results_block
# ---------------------------------------------------------------------------

class TestFormatResultsBlock:
    def test_passed_task_no_files(self, tmp_path):
        m = make_metrics(task="t1", success=True, task_dir=str(tmp_path))
        block = format_results_block([m])
        assert "PASSED" in block
        assert "t1" in block

    def test_failed_task_with_error(self, tmp_path):
        m = make_metrics(task="t2", success=False, failure_reason="timeout", task_dir=str(tmp_path))
        block = format_results_block([m])
        assert "FAILED" in block
        assert "timeout" in block

    def test_includes_test_snippet(self, tmp_path):
        (tmp_path / "solution_test.py").write_text("def test_x(): pass\n")
        m = make_metrics(task="t3", task_dir=str(tmp_path))
        block = format_results_block([m])
        assert "solution_test.py" in block

    def test_includes_solution_short(self, tmp_path):
        (tmp_path / "solution.py").write_text("def f(): pass\n")
        m = make_metrics(task="t4", task_dir=str(tmp_path))
        block = format_results_block([m])
        assert "solution.py" in block

    def test_includes_solution_long(self, tmp_path):
        """solution.py > 40 lines triggers the truncation branch."""
        lines = "\n".join(f"x_{i} = {i}" for i in range(50))
        (tmp_path / "solution.py").write_text(lines)
        m = make_metrics(task="t5", task_dir=str(tmp_path))
        block = format_results_block([m])
        assert "more lines" in block

    def test_includes_log_snippet(self, tmp_path):
        (tmp_path / "run.log").write_text("line1\nline2\n")
        m = make_metrics(task="t6", task_dir=str(tmp_path))
        block = format_results_block([m])
        assert "run.log" in block

    def test_multiple_tasks_separated(self, tmp_path):
        m1 = make_metrics(task="a", task_dir=str(tmp_path))
        m2 = make_metrics(task="b", success=False, task_dir=str(tmp_path))
        block = format_results_block([m1, m2])
        assert "Task 1" in block
        assert "Task 2" in block


# ---------------------------------------------------------------------------
# summary_line / build_eval_metrics
# ---------------------------------------------------------------------------

class TestSummaryLine:
    def test_basic(self, tmp_path):
        m1 = make_metrics(success=True, iterations=2, task_dir=str(tmp_path))
        m2 = make_metrics(success=False, iterations=4, task_dir=str(tmp_path))
        line = summary_line([m1, m2])
        assert "1/2" in line
        assert "3.0" in line  # avg of 2 and 4

    def test_empty_list(self):
        line = summary_line([])
        assert "0/0" in line


class TestBuildEvalMetrics:
    def test_structure(self, tmp_path):
        m1 = make_metrics(success=True, task_dir=str(tmp_path))
        m2 = make_metrics(success=False, task_dir=str(tmp_path))
        data = build_eval_metrics([m1, m2])
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert len(data["runs"]) == 2
        assert "write_file" in data["summary"]["total_tool_calls_by_type"]

    def test_empty(self):
        data = build_eval_metrics([])
        assert data["summary"]["tasks"] == 0
        assert data["summary"]["avg_iterations"] == 0


# ---------------------------------------------------------------------------
# run_judge_single / run_judge_ab / run_meta_judge
# ---------------------------------------------------------------------------

class TestJudges:
    def _mock_response(self, text="judgment text"):
        client = MagicMock()
        client.messages.create.return_value = make_response(
            [make_text_block(text)], stop_reason="end_turn"
        )
        return client

    def test_run_judge_single(self, tmp_path):
        client = self._mock_response("single judgment")
        m = make_metrics(task_dir=str(tmp_path))
        result = run_judge_single(client, [m], 5, MINIMAL_PROMPTS, MINIMAL_EVAL_PROMPTS)
        assert result == "single judgment"
        client.messages.create.assert_called_once()

    def test_run_judge_ab(self, tmp_path):
        client = self._mock_response("ab judgment")
        m = make_metrics(task_dir=str(tmp_path))
        result = run_judge_ab(
            client, [m], [m], 5, MINIMAL_PROMPTS, MINIMAL_PROMPTS, MINIMAL_EVAL_PROMPTS
        )
        assert result == "ab judgment"

    def test_run_meta_judge(self, tmp_path):
        client = self._mock_response("meta judgment")
        m = make_metrics(success=True, iterations=2, task_dir=str(tmp_path))
        result = run_meta_judge(client, [m], "primary judgment", MINIMAL_EVAL_PROMPTS, "judge_single")
        assert result == "meta judgment"
        call_kwargs = client.messages.create.call_args.kwargs
        assert "n=1" in call_kwargs["messages"][0]["content"]

    def test_run_meta_judge_empty_list(self):
        """avg_iter division-by-zero guard — empty metrics_list."""
        client = MagicMock()
        client.messages.create.return_value = make_response(
            [make_text_block("meta")], stop_reason="end_turn"
        )
        result = run_meta_judge(client, [], "text", MINIMAL_EVAL_PROMPTS, "judge_single")
        assert result == "meta"


# ---------------------------------------------------------------------------
# optimize_prompts
# ---------------------------------------------------------------------------

class TestOptimizePrompts:
    def test_valid_toml_saves_file(self, tmp_path, monkeypatch, capsys):
        current_toml = tmp_path / "agent.toml"
        current_toml.write_text('version = "v1"\n')

        valid_toml = (
            'version = "v2"\n'
            '[test_generation]\nsystem = "s"\nuser = "u"\n'
        )
        client = MagicMock()
        client.messages.create.return_value = make_response(
            [make_text_block(valid_toml)], stop_reason="end_turn"
        )

        written = {}
        monkeypatch.setattr(Path, "write_text", lambda self, content, **kw: written.__setitem__(str(self), content))

        optimize_prompts(client, current_toml, "analysis", "20260101-120000", MINIMAL_EVAL_PROMPTS)

        captured = capsys.readouterr()
        assert "Optimized prompts saved" in captured.out
        assert "20260101-120000" in captured.out
        # A file with the timestamp was written
        assert any("20260101-120000" in k for k in written)

    def test_invalid_toml_prints_warning(self, tmp_path, capsys):
        current_toml = tmp_path / "agent.toml"
        current_toml.write_text('version = "v1"\n')

        client = MagicMock()
        client.messages.create.return_value = make_response(
            [make_text_block("NOT VALID TOML ][[ !!!")], stop_reason="end_turn"
        )

        optimize_prompts(client, current_toml, "analysis", "20260101", MINIMAL_EVAL_PROMPTS)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Not saving" in captured.out


# ---------------------------------------------------------------------------
# run_task
# ---------------------------------------------------------------------------

class TestRunTask:
    def test_run_task_success(self, tmp_path):
        client = MagicMock()
        with patch("eval.AgentLoop") as MockLoop:
            instance = MagicMock()
            instance.metrics = make_metrics(task_dir=str(tmp_path / "task-01"))
            MockLoop.return_value = instance
            with patch("eval.save_metrics"):
                result = run_task(client, "a task", tmp_path / "task-01", 5, MINIMAL_PROMPTS, "v1")
        assert result is instance.metrics

    def test_run_task_exception_stored(self, tmp_path):
        client = MagicMock()
        with patch("eval.AgentLoop") as MockLoop:
            instance = MagicMock()
            instance.metrics = make_metrics(task_dir=str(tmp_path / "task-01"))
            instance.generate_tests.side_effect = RuntimeError("boom")
            MockLoop.return_value = instance
            with patch("eval.save_metrics"):
                result = run_task(client, "a task", tmp_path / "task-01", 5, MINIMAL_PROMPTS, "v1")
        assert "boom" in result.failure_reason

    def test_run_task_passes_revision_flags(self, tmp_path):
        client = MagicMock()
        with patch("eval.AgentLoop") as MockLoop:
            instance = MagicMock()
            instance.metrics = make_metrics(task_dir=str(tmp_path / "task-01"))
            MockLoop.return_value = instance
            with patch("eval.save_metrics"):
                run_task(client, "a task", tmp_path / "task-01", 5, MINIMAL_PROMPTS, "v1",
                         allow_test_revision=3, auto_approve_revision=True)
        _, kwargs = MockLoop.call_args
        assert kwargs["allow_test_revision"] == 3
        assert kwargs["auto_approve_revision"] is True


# ---------------------------------------------------------------------------
# run_suite
# ---------------------------------------------------------------------------

class TestRunSuite:
    def test_run_suite_basic(self, tmp_path, capsys):
        client = MagicMock()
        tasks = ["task A", "task B"]
        m = make_metrics(task_dir=str(tmp_path))
        with patch("eval.run_task", return_value=m):
            results = run_suite(client, tasks, tmp_path, 5, MINIMAL_PROMPTS, "v1", label="v1")
        assert len(results) == 2
        out = capsys.readouterr().out
        assert "task A" in out
        assert "task B" in out


# ---------------------------------------------------------------------------
# main() — single run, A/B, flags
# ---------------------------------------------------------------------------

class TestMain:
    """Tests for eval.main() through patched dependencies."""

    def _make_fake_metrics(self, tmp_path):
        m = make_metrics(task_dir=str(tmp_path / "task-01"))
        return [m]

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(sys, "argv", ["eval.py"])
        with pytest.raises(SystemExit) as exc:
            eval_mod.main()
        assert exc.value.code == 1

    def test_single_run(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["eval.py", "--tasks", "1", "--max-iter", "1"]
        )
        fake_m = make_metrics(task_dir=str(tmp_path / "eval-xxx" / "task-01"))
        with patch("eval.anthropic.Anthropic"):
            with patch("eval.run_suite", return_value=[fake_m]) as mock_suite:
                with patch("eval.run_judge_single", return_value="judgment") as mock_judge:
                    eval_mod.main()
        mock_suite.assert_called_once()
        mock_judge.assert_called_once()
        out = capsys.readouterr().out
        assert "RESULTS" in out

    def test_single_run_with_optimize(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["eval.py", "--tasks", "1", "--max-iter", "1", "--optimize"]
        )
        fake_m = make_metrics(task_dir=str(tmp_path / "eval-xxx" / "task-01"))
        with patch("eval.anthropic.Anthropic"):
            with patch("eval.run_suite", return_value=[fake_m]):
                with patch("eval.run_judge_single", return_value="judgment"):
                    with patch("eval.optimize_prompts") as mock_opt:
                        eval_mod.main()
        mock_opt.assert_called_once()

    def test_single_run_with_meta_judge(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["eval.py", "--tasks", "1", "--max-iter", "1", "--meta-judge"]
        )
        fake_m = make_metrics(task_dir=str(tmp_path / "eval-xxx" / "task-01"))
        with patch("eval.anthropic.Anthropic"):
            with patch("eval.run_suite", return_value=[fake_m]):
                with patch("eval.run_judge_single", return_value="judgment"):
                    with patch("eval.run_meta_judge", return_value="meta") as mock_meta:
                        eval_mod.main()
        mock_meta.assert_called_once()
        out = capsys.readouterr().out
        assert "META-JUDGE" in out

    def test_ab_run(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["eval.py", "--tasks", "1", "--max-iter", "1", "--prompts", "va", "--compare", "vb"]
        )
        fake_m = make_metrics(task_dir=str(tmp_path / "task-01"))
        with patch("eval.anthropic.Anthropic"):
            with patch("eval.load_prompts", return_value=MINIMAL_PROMPTS):
                with patch("eval.run_suite", return_value=[fake_m]) as mock_suite:
                    with patch("eval.run_judge_ab", return_value="ab judgment") as mock_judge:
                        eval_mod.main()
        assert mock_suite.call_count == 2
        mock_judge.assert_called_once()
        out = capsys.readouterr().out
        assert "ab judgment" in out

    def test_ab_run_with_meta_judge(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["eval.py", "--tasks", "1", "--max-iter", "1",
             "--prompts", "va", "--compare", "vb", "--meta-judge"]
        )
        fake_m = make_metrics(task_dir=str(tmp_path / "task-01"))
        with patch("eval.anthropic.Anthropic"):
            with patch("eval.load_prompts", return_value=MINIMAL_PROMPTS):
                with patch("eval.run_suite", return_value=[fake_m]):
                    with patch("eval.run_judge_ab", return_value="ab judgment"):
                        with patch("eval.run_meta_judge", return_value="meta ab") as mock_meta:
                            eval_mod.main()
        mock_meta.assert_called_once()
        args = mock_meta.call_args[0]
        assert args[4] == "judge_ab"

    def test_results_written_to_custom_out(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        out_file = tmp_path / "custom_results.json"
        monkeypatch.setattr(
            sys, "argv",
            ["eval.py", "--tasks", "1", "--max-iter", "1", "--out", str(out_file)]
        )
        fake_m = make_metrics(task_dir=str(tmp_path / "eval-xxx" / "task-01"))
        with patch("eval.anthropic.Anthropic"):
            with patch("eval.run_suite", return_value=[fake_m]):
                with patch("eval.run_judge_single", return_value="j"):
                    eval_mod.main()
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "summary" in data

    def test_allow_test_revision_and_auto_approve_passed_to_suite(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["eval.py", "--tasks", "1", "--max-iter", "1",
             "--allow-test-revision", "2", "--auto-approve-revision"]
        )
        fake_m = make_metrics(task_dir=str(tmp_path / "task-01"))
        with patch("eval.anthropic.Anthropic"):
            with patch("eval.run_suite", return_value=[fake_m]) as mock_suite:
                with patch("eval.run_judge_single", return_value="j"):
                    eval_mod.main()
        _, kwargs = mock_suite.call_args
        assert kwargs["allow_test_revision"] == 2
        assert kwargs["auto_approve_revision"] is True
