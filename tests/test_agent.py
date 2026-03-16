"""Tests for agent.py — targets >98% line/branch coverage."""
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from agent import (
    AgentConfig,
    AgentLoop,
    DEFAULT_MODEL,
    _RETRYABLE,
    _extract_write_lines,
    _parse_coverage_pct,
    _parse_pytest_result,
    _tests_passed,
    load_prompts,
)
from tests.conftest import (
    MINIMAL_PROMPTS,
    make_response,
    make_stream_mock,
    make_text_block,
    make_tool_block,
)


# ---------------------------------------------------------------------------
# load_prompts
# ---------------------------------------------------------------------------

class TestLoadPrompts:
    def test_load_default_version(self):
        """Loads v1.toml from the prompts directory."""
        prompts = load_prompts("v1")
        assert "test_generation" in prompts
        assert "implementation" in prompts

    def test_load_explicit_path(self, tmp_path):
        """Loads from an explicit Path argument."""
        toml_text = '[test_generation]\nsystem = "s"\nuser = "u"\n'
        p = tmp_path / "custom.toml"
        p.write_text(toml_text)
        result = load_prompts(path=p)
        assert result["test_generation"]["system"] == "s"


# ---------------------------------------------------------------------------
# _parse_coverage_pct
# ---------------------------------------------------------------------------

class TestParseCoveragePct:
    def test_full_coverage(self):
        output = (
            "Name          Stmts   Miss  Cover\n"
            "solution.py      45      0   100%\n"
            "TOTAL            45      0   100%\n"
        )
        assert _parse_coverage_pct(output) == 100.0

    def test_partial_coverage(self):
        output = "solution.py      45     10    78%\n"
        assert _parse_coverage_pct(output) == 78.0

    def test_no_solution_line(self):
        output = "== 5 passed in 0.10s ==\n"
        assert _parse_coverage_pct(output) == 0.0

    def test_empty_string(self):
        assert _parse_coverage_pct("") == 0.0


# ---------------------------------------------------------------------------
# _parse_pytest_result
# ---------------------------------------------------------------------------

class TestParsePytestResult:
    def test_empty_string(self):
        assert _parse_pytest_result("") == ("", [])

    def test_passed_summary_line(self):
        out = "some output\n== 3 passed in 0.12s ==\n"
        summary, failed = _parse_pytest_result(out)
        assert "3 passed" in summary
        assert failed == []

    def test_failed_summary_with_failed_lines(self):
        out = (
            "FAILED tests/test_f.py::test_alpha FAILED\n"
            "FAILED tests/test_f.py::test_beta FAILED\n"
            "== 2 failed in 0.20s ==\n"
        )
        summary, failed = _parse_pytest_result(out)
        assert "failed" in summary
        assert "test_alpha" in failed
        assert "test_beta" in failed

    def test_summary_starting_with_FAILED(self):
        # Must contain lowercase "failed" AND start with "FAILED" to match
        out = "FAILED 2 failed tests\n"
        summary, failed = _parse_pytest_result(out)
        assert summary == "FAILED 2 failed tests"

    def test_no_matching_summary_line(self):
        out = "collected 0 items\n"
        summary, failed = _parse_pytest_result(out)
        assert summary == ""
        assert failed == []

    def test_error_in_summary(self):
        out = "== 1 error in 0.05s ==\n"
        summary, failed = _parse_pytest_result(out)
        assert "error" in summary

    def test_failed_line_multiple_colons(self):
        out = "FAILED tests/sub/test_f.py::MyClass::test_method FAILED\n"
        _, failed = _parse_pytest_result(out)
        assert "test_method" in failed


# ---------------------------------------------------------------------------
# _tests_passed
# ---------------------------------------------------------------------------

class TestTestsPassed:
    def test_all_passed(self):
        assert _tests_passed("== 5 passed in 0.10s ==") is True

    def test_mixed_failed(self):
        assert _tests_passed("== 2 failed, 3 passed in 0.20s ==") is False

    def test_error_present(self):
        assert _tests_passed("== 1 passed, 1 error in 0.10s ==") is False

    def test_empty(self):
        assert _tests_passed("") is False

    def test_no_summary_line(self):
        assert _tests_passed("collected 3 items\n") is False

    def test_summary_only_failed(self):
        assert _tests_passed("== 2 failed in 0.15s ==") is False


# ---------------------------------------------------------------------------
# _extract_write_lines
# ---------------------------------------------------------------------------

class TestExtractWriteLines:
    def test_no_content_key(self):
        path, lines, count = _extract_write_lines('{"path": "f.py"', 0)
        assert path == "f.py"
        assert lines == []
        assert count == 0

    def test_no_path_key(self):
        path, lines, count = _extract_write_lines('{"content": "hi\\n"', 0)
        assert path is None
        assert lines == ["hi"]
        assert count == 1

    def test_partial_snapshot_no_lines_yet(self):
        snapshot = '{"path": "sol.py", "content": "def foo'
        path, lines, count = _extract_write_lines(snapshot, 0)
        assert path == "sol.py"
        assert lines == []
        assert count == 0

    def test_one_complete_line(self):
        snapshot = '{"path": "sol.py", "content": "def foo():\\n    pass'
        path, lines, count = _extract_write_lines(snapshot, 0)
        assert path == "sol.py"
        assert lines == ["def foo():"]
        assert count == 1

    def test_incremental_prev_count(self):
        """prev_count=1 → only emit lines after the first."""
        snapshot = '{"path": "sol.py", "content": "line1\\nline2\\nline3'
        path, lines, count = _extract_write_lines(snapshot, 1)
        assert lines == ["line2"]
        assert count == 2

    def test_decodes_json_escapes(self):
        # JSON \t (backslash+t in raw JSON) should decode to real tab
        snapshot = '{"path": "f.py", "content": "a\\tb\\n'
        path, lines, count = _extract_write_lines(snapshot, 0)
        assert lines == ["a\tb"]

    def test_decodes_escaped_backslash(self):
        snapshot = '{"path": "f.py", "content": "a\\\\b\\n'
        path, lines, count = _extract_write_lines(snapshot, 0)
        assert lines == ["a\\b"]

    def test_decodes_escaped_quote(self):
        snapshot = '{"path": "f.py", "content": "say \\"hi\\"\\n'
        path, lines, count = _extract_write_lines(snapshot, 0)
        assert lines == ['say "hi"']

    def test_prev_count_already_caught_up(self):
        """No new lines since last call → returns empty."""
        snapshot = '{"path": "f.py", "content": "line1\\n'
        path, lines, count = _extract_write_lines(snapshot, 1)
        assert lines == []
        assert count == 1


# ---------------------------------------------------------------------------
# _llm_call streaming — unit tests via a mock stream
# ---------------------------------------------------------------------------

class TestLlmCallStreaming:
    """Test _llm_call generator using a mock stream that emits specific events."""

    def _make_loop(self, tmp_task_dir, mock_client):
        return AgentLoop(
            client=mock_client,
            task_dir=tmp_task_dir,
            prompts=MINIMAL_PROMPTS,
            logger=MagicMock(),
        )

    def _make_stream(self, events, response):
        """Build a mock context manager whose __iter__ yields SimpleNamespace events."""
        class _Stream:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def __iter__(self):
                return iter(events)
            def get_final_message(self):
                return response
        return _Stream()

    def _run(self, loop, stream):
        from agent import _LoopState
        state = _LoopState(messages=[{"role": "user", "content": "go"}])
        loop.client.messages.stream.return_value = stream
        return list(loop._llm_call(state))

    def test_text_delta_emitted(self, tmp_task_dir, mock_client):
        resp = make_response([make_text_block("done")], stop_reason="end_turn")
        stream = self._make_stream([
            SimpleNamespace(type="text", text="think"),
            SimpleNamespace(type="text", text=""),  # empty text skipped
        ], resp)
        loop = self._make_loop(tmp_task_dir, mock_client)
        events = self._run(loop, stream)
        text_events = [e for e in events if e["type"] == "text_delta"]
        assert len(text_events) == 1
        assert text_events[0]["text"] == "think"

    def test_write_line_emitted(self, tmp_task_dir, mock_client):
        # partial_json events are incremental deltas that we accumulate ourselves
        resp = make_response([make_text_block("")], stop_reason="end_turn")
        stream = self._make_stream([
            SimpleNamespace(type="content_block_start",
                            content_block=SimpleNamespace(type="tool_use", name="write_file")),
            SimpleNamespace(type="input_json",
                            partial_json='{"path": "sol.py", "content": "def foo():\\n    pass'),
            SimpleNamespace(type="input_json",
                            partial_json='\\n"}'),
            SimpleNamespace(type="content_block_stop"),
        ], resp)
        loop = self._make_loop(tmp_task_dir, mock_client)
        events = self._run(loop, stream)
        write_events = [e for e in events if e["type"] == "write_line"]
        assert len(write_events) == 2
        assert write_events[0]["line"] == "def foo():"
        assert write_events[0]["line_num"] == 1
        assert write_events[0]["path"] == "sol.py"
        assert write_events[1]["line"] == "    pass"
        assert write_events[1]["line_num"] == 2

    def test_non_write_file_tool_no_write_lines(self, tmp_task_dir, mock_client):
        """input_json for a non-write_file tool emits nothing."""
        resp = make_response([], stop_reason="end_turn")
        stream = self._make_stream([
            SimpleNamespace(type="content_block_start",
                            content_block=SimpleNamespace(type="tool_use", name="read_file")),
            SimpleNamespace(type="input_json", partial_json='{"path": "x.py"}'),
        ], resp)
        loop = self._make_loop(tmp_task_dir, mock_client)
        events = self._run(loop, stream)
        assert not any(e["type"] == "write_line" for e in events)

    def test_text_block_resets_current_tool(self, tmp_task_dir, mock_client):
        """content_block_start with a text block sets current_tool to None."""
        resp = make_response([], stop_reason="end_turn")
        stream = self._make_stream([
            SimpleNamespace(type="content_block_start",
                            content_block=SimpleNamespace(type="text")),
            SimpleNamespace(type="input_json", partial_json='{"path": "x.py"}'),
        ], resp)
        loop = self._make_loop(tmp_task_dir, mock_client)
        events = self._run(loop, stream)
        assert not any(e["type"] == "write_line" for e in events)

    def test_response_sentinel_carries_final_message(self, tmp_task_dir, mock_client):
        resp = make_response([make_text_block("ok")], stop_reason="end_turn",
                             in_tok=10, out_tok=5)
        stream = self._make_stream([], resp)
        loop = self._make_loop(tmp_task_dir, mock_client)
        events = self._run(loop, stream)
        sentinel = next(e for e in events if e["type"] == "_response")
        assert sentinel["response"] is resp
        assert loop.metrics.impl_input_tokens == 10
        assert loop.metrics.impl_output_tokens == 5

    def test_content_block_stop_clears_tool(self, tmp_task_dir, mock_client):
        """content_block_stop resets current_tool so subsequent input_json is ignored."""
        resp = make_response([], stop_reason="end_turn")
        stream = self._make_stream([
            SimpleNamespace(type="content_block_start",
                            content_block=SimpleNamespace(type="tool_use", name="write_file")),
            SimpleNamespace(type="content_block_stop"),
            SimpleNamespace(type="input_json",
                            partial_json='{"path": "f.py", "content": "x\\n"}'),
        ], resp)
        loop = self._make_loop(tmp_task_dir, mock_client)
        events = self._run(loop, stream)
        assert not any(e["type"] == "write_line" for e in events)


# ---------------------------------------------------------------------------
# AgentLoop.__init__
# ---------------------------------------------------------------------------

class TestAgentLoopInit:
    def test_explicit_prompts_and_logger(self, tmp_task_dir, mock_client):
        logger = MagicMock()
        loop = AgentLoop(
            client=mock_client,
            task_dir=tmp_task_dir,
            prompts=MINIMAL_PROMPTS,
            logger=logger,
        )
        assert loop.model == DEFAULT_MODEL
        assert loop.max_iterations == 10
        assert loop.allow_test_revision == False
        assert loop._logger is logger
        assert loop._prompts is MINIMAL_PROMPTS

    def test_auto_load_prompts_from_version(self, tmp_task_dir, mock_client):
        """When prompts=None, loads from disk using prompts_version."""
        loop = AgentLoop(
            client=mock_client,
            task_dir=tmp_task_dir,
            config=AgentConfig(prompts_version="v1"),
            logger=MagicMock(),
        )
        assert "test_generation" in loop._prompts

    def test_auto_create_logger(self, tmp_task_dir, mock_client):
        """When logger=None, setup_logging is called."""
        loop = AgentLoop(
            client=mock_client,
            task_dir=tmp_task_dir,
            prompts=MINIMAL_PROMPTS,
        )
        assert loop._logger is not None


# ---------------------------------------------------------------------------
# AgentLoop.generate_tests
# ---------------------------------------------------------------------------

class TestGenerateTests:
    def _make_loop(self, tmp_task_dir, mock_client):
        return AgentLoop(
            client=mock_client,
            task_dir=tmp_task_dir,
            prompts=MINIMAL_PROMPTS,
            logger=MagicMock(),
        )

    def test_writes_test_file_and_returns_content(self, tmp_task_dir, mock_client):
        mock_client.messages.create.return_value = make_response(
            [make_text_block("def test_a():\n    pass\ndef test_b():\n    pass\n")],
            stop_reason="end_turn",
            in_tok=50,
            out_tok=30,
        )
        loop = self._make_loop(tmp_task_dir, mock_client)
        result = loop.generate_tests("write a fn")

        assert "def test_a" in result
        assert (tmp_task_dir / "solution_test.py").read_text() == result
        assert loop.metrics.test_gen_input_tokens == 50
        assert loop.metrics.test_gen_output_tokens == 30
        assert loop.metrics.task_prompt == "write a fn"

    def test_test_count_logged(self, tmp_task_dir, mock_client):
        mock_client.messages.create.return_value = make_response(
            [make_text_block("def test_one():\n    pass\n")],
            stop_reason="end_turn",
        )
        loop = self._make_loop(tmp_task_dir, mock_client)
        loop.generate_tests("task")
        # Logger is a MagicMock; verify info was called with the test count message
        info_calls = [str(c) for c in loop._logger.info.call_args_list]
        assert any("1 tests" in c for c in info_calls)


# ---------------------------------------------------------------------------
# Helpers for implementation loop tests
# ---------------------------------------------------------------------------

def _make_dispatch_side_effect(task_dir, run_output="1 failed in 0.1s"):
    """Dispatch mock that actually writes files to disk."""
    def dispatch(name, inputs, td):
        if name == "write_file":
            p = td / inputs["path"]
            p.write_text(inputs.get("content", ""))
            return f"written {inputs['path']}"
        elif name == "run_subprocess":
            return run_output
        return f"ok:{name}"
    return dispatch


def _make_loop(tmp_task_dir, mock_client, config=None):
    return AgentLoop(
        client=mock_client,
        task_dir=tmp_task_dir,
        config=config or AgentConfig(max_iterations=5),
        prompts=MINIMAL_PROMPTS,
        logger=MagicMock(),
    )


# ---------------------------------------------------------------------------
# AgentLoop.run_implementation_loop — end_turn paths
# ---------------------------------------------------------------------------

class TestRunImplementationLoopEndTurn:
    def test_end_turn_no_solution_py(self, tmp_task_dir, mock_client):
        mock_client.messages.stream.return_value = make_stream_mock(make_response(
            [make_text_block("done")], stop_reason="end_turn"
        ))
        loop = _make_loop(tmp_task_dir, mock_client)
        success, msg = loop.run_implementation_loop("task", "tests")
        assert not success
        assert "stopped" in msg

    def test_end_turn_tests_pass(self, tmp_task_dir, mock_client):
        (tmp_task_dir / "solution.py").write_text("def f(): pass\n")
        mock_client.messages.stream.return_value = make_stream_mock(make_response(
            [make_text_block("done")], stop_reason="end_turn"
        ))
        with patch("agent.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="== 3 passed in 0.05s ==\n", stderr=""
            )
            loop = _make_loop(tmp_task_dir, mock_client)
            success, msg = loop.run_implementation_loop("task", "tests")
        assert success
        assert loop.metrics.success is True

    def test_end_turn_tests_fail(self, tmp_task_dir, mock_client):
        (tmp_task_dir / "solution.py").write_text("def f(): pass\n")
        mock_client.messages.stream.return_value = make_stream_mock(make_response(
            [make_text_block("gave up")], stop_reason="end_turn"
        ))
        with patch("agent.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="== 1 failed in 0.05s ==\n", stderr=""
            )
            loop = _make_loop(tmp_task_dir, mock_client)
            success, msg = loop.run_implementation_loop("task", "tests")
        assert not success
        assert loop.metrics.failure_reason == "Agent stopped without passing tests."

    def test_end_turn_coverage_stored_in_metrics(self, tmp_task_dir, mock_client):
        (tmp_task_dir / "solution.py").write_text("def f(): pass\n")
        mock_client.messages.stream.return_value = make_stream_mock(make_response(
            [make_text_block("done")], stop_reason="end_turn"
        ))
        cov_output = "solution.py      20      0   100%\nTOTAL  20  0  100%\n"
        with patch("agent.subprocess.run") as mock_run:
            mock_run.side_effect = [
                SimpleNamespace(stdout="== 3 passed in 0.05s ==\n", stderr=""),
                SimpleNamespace(stdout=cov_output, stderr=""),
            ]
            loop = _make_loop(tmp_task_dir, mock_client)
            success, _ = loop.run_implementation_loop("task", "tests")
        assert success
        assert loop.metrics.test_coverage_pct == 100.0

    def test_end_turn_coverage_below_min_fails(self, tmp_task_dir, mock_client):
        (tmp_task_dir / "solution.py").write_text("def f(): pass\n")
        mock_client.messages.stream.return_value = make_stream_mock(make_response(
            [make_text_block("done")], stop_reason="end_turn"
        ))
        cov_output = "solution.py      20      6    70%\n"
        with patch("agent.subprocess.run") as mock_run:
            mock_run.side_effect = [
                SimpleNamespace(stdout="== 3 passed in 0.05s ==\n", stderr=""),
                SimpleNamespace(stdout=cov_output, stderr=""),
            ]
            loop = _make_loop(tmp_task_dir, mock_client, AgentConfig(max_iterations=5, min_coverage=80.0))
            success, msg = loop.run_implementation_loop("task", "tests")
        assert not success
        assert "coverage" in loop.metrics.failure_reason.lower()
        assert loop.metrics.test_coverage_pct == 70.0

    def test_end_turn_no_text_block(self, tmp_task_dir, mock_client):
        """end_turn with no text block in content → final_text is empty."""
        (tmp_task_dir / "solution.py").write_text("def f(): pass\n")
        mock_client.messages.stream.return_value = make_stream_mock(make_response(
            [], stop_reason="end_turn"
        ))
        with patch("agent.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="== 1 failed ==\n", stderr=""
            )
            loop = _make_loop(tmp_task_dir, mock_client)
            success, _ = loop.run_implementation_loop("task", "tests")
        assert not success

    def test_no_tool_calls_not_end_turn(self, tmp_task_dir, mock_client):
        mock_client.messages.stream.return_value = make_stream_mock(make_response(
            [], stop_reason="tool_use"  # not end_turn, but also no tools
        ))
        loop = _make_loop(tmp_task_dir, mock_client)
        success, msg = loop.run_implementation_loop("task", "tests")
        assert not success
        assert "stopped requesting tools" in msg


# ---------------------------------------------------------------------------
# AgentLoop.run_implementation_loop — tool logging branches
# ---------------------------------------------------------------------------

class TestRunImplementationLoopToolLogging:
    """Each branch of the per-tool logging block."""

    def _run_single_tool(self, tmp_task_dir, mock_client, tool_block, dispatch_return="ok"):
        """Helper: run one tool-use turn then end_turn."""
        mock_client.messages.stream.side_effect = [
            make_stream_mock(make_response([tool_block])),
            make_stream_mock(make_response([make_text_block("done")], stop_reason="end_turn")),
        ]
        with patch("agent.dispatch_tool", return_value=dispatch_return) as m:
            loop = _make_loop(tmp_task_dir, mock_client)
            loop.run_implementation_loop("task", "tests")
        return loop, m

    def test_read_file_tool(self, tmp_task_dir, mock_client):
        tb = make_tool_block("read_file", {"path": "solution_test.py"})
        loop, _ = self._run_single_tool(tmp_task_dir, mock_client, tb)
        assert loop.metrics.tool_calls.get("read_file", 0) == 1

    def test_write_file_tool_with_content(self, tmp_task_dir, mock_client):
        tb = make_tool_block("write_file", {"path": "solution.py", "content": "a\nb\nc\n"})
        loop, _ = self._run_single_tool(tmp_task_dir, mock_client, tb)
        assert loop.metrics.impl_write_count == 1

    def test_write_file_tool_empty_content(self, tmp_task_dir, mock_client):
        """Empty content → line_count = 0."""
        tb = make_tool_block("write_file", {"path": "solution.py", "content": ""})
        loop, _ = self._run_single_tool(tmp_task_dir, mock_client, tb)
        assert loop.metrics.impl_write_count == 1

    def test_run_subprocess_with_failed_names(self, tmp_task_dir, mock_client):
        output = (
            "FAILED tests/test_f.py::test_one FAILED\n"
            "== 1 failed in 0.1s ==\n"
        )
        tb = make_tool_block("run_subprocess", {"command": ["pytest", "solution_test.py"]})
        mock_client.messages.stream.side_effect = [
            make_stream_mock(make_response([tb])),
            make_stream_mock(make_response([make_text_block("done")], stop_reason="end_turn")),
        ]
        with patch("agent.dispatch_tool", return_value=output):
            loop = _make_loop(tmp_task_dir, mock_client)
            loop.run_implementation_loop("task", "tests")
        info_calls = [str(c) for c in loop._logger.info.call_args_list]
        assert any("test_one" in c for c in info_calls)
        assert loop.metrics.impl_pytest_runs == 1

    def test_run_subprocess_summary_only(self, tmp_task_dir, mock_client):
        output = "== 2 passed in 0.1s ==\n"
        tb = make_tool_block("run_subprocess", {"command": ["pytest"]})
        mock_client.messages.stream.side_effect = [
            make_stream_mock(make_response([tb])),
            make_stream_mock(make_response([make_text_block("done")], stop_reason="end_turn")),
        ]
        with patch("agent.dispatch_tool", return_value=output):
            loop = _make_loop(tmp_task_dir, mock_client)
            loop.run_implementation_loop("task", "tests")
        info_calls = [str(c) for c in loop._logger.info.call_args_list]
        assert any("✓" in c for c in info_calls)

    def test_run_subprocess_no_summary_no_failed(self, tmp_task_dir, mock_client):
        """Output with no summary and no FAILED lines — neither log branch taken."""
        output = "collected 0 items\n"
        tb = make_tool_block("run_subprocess", {"command": ["pytest"]})
        mock_client.messages.stream.side_effect = [
            make_stream_mock(make_response([tb])),
            make_stream_mock(make_response([make_text_block("done")], stop_reason="end_turn")),
        ]
        with patch("agent.dispatch_tool", return_value=output):
            loop = _make_loop(tmp_task_dir, mock_client)
            loop.run_implementation_loop("task", "tests")
        assert loop.metrics.impl_pytest_runs == 1

    def test_run_subprocess_string_command(self, tmp_task_dir, mock_client):
        """Command as a plain string (not a list) hits the str() branch."""
        tb = make_tool_block("run_subprocess", {"command": "pytest solution_test.py"})
        self._run_single_tool(tmp_task_dir, mock_client, tb)

    def test_context7_docs_tool(self, tmp_task_dir, mock_client):
        tb = make_tool_block("context7_docs", {"library": "requests", "query": "headers"})
        loop, _ = self._run_single_tool(tmp_task_dir, mock_client, tb)
        assert loop.metrics.tool_calls.get("context7_docs", 0) == 1

    def test_firecrawl_search_tool(self, tmp_task_dir, mock_client):
        tb = make_tool_block("firecrawl_search", {"query": "python rle encoding"})
        loop, _ = self._run_single_tool(tmp_task_dir, mock_client, tb)
        assert loop.metrics.tool_calls.get("firecrawl_search", 0) == 1

    def test_firecrawl_scrape_tool(self, tmp_task_dir, mock_client):
        tb = make_tool_block("firecrawl_scrape", {"url": "https://example.com"})
        loop, _ = self._run_single_tool(tmp_task_dir, mock_client, tb)
        assert loop.metrics.tool_calls.get("firecrawl_scrape", 0) == 1

    def test_run_python_tool(self, tmp_task_dir, mock_client):
        tb = make_tool_block("run_python", {"code": "print(len('hello'))"})
        loop, _ = self._run_single_tool(tmp_task_dir, mock_client, tb, dispatch_return="5")
        assert loop.metrics.tool_calls.get("run_python", 0) == 1

    def test_calculator_tool(self, tmp_task_dir, mock_client):
        tb = make_tool_block("calculator", {"expression": "2 + 2"})
        loop, _ = self._run_single_tool(tmp_task_dir, mock_client, tb, dispatch_return="4")
        assert loop.metrics.tool_calls.get("calculator", 0) == 1

    def test_unknown_tool_fallback(self, tmp_task_dir, mock_client):
        """Unknown tool name hits the case _ fallback in _make_tool_event."""
        tb = make_tool_block("mystery_tool", {"foo": "bar"})
        loop, _ = self._run_single_tool(tmp_task_dir, mock_client, tb)
        assert loop.metrics.tool_calls.get("mystery_tool", 0) == 1


# ---------------------------------------------------------------------------
# AgentLoop.run_implementation_loop — iteration counter logic
# ---------------------------------------------------------------------------

class TestIterationCounter:
    def test_write_then_run_increments(self, tmp_task_dir, mock_client):
        """write_file followed by run_subprocess in same turn increments iteration."""
        tb_write = make_tool_block("write_file", {"path": "solution.py", "content": "x"}, "tc1")
        tb_run = make_tool_block("run_subprocess", {"command": ["pytest"]}, "tc2")
        mock_client.messages.stream.side_effect = [
            make_stream_mock(make_response([tb_write, tb_run])),
            make_stream_mock(make_response([make_text_block("done")], stop_reason="end_turn")),
        ]
        with patch("agent.dispatch_tool", return_value="1 passed in 0.1s"):
            loop = _make_loop(tmp_task_dir, mock_client)
            loop.run_implementation_loop("task", "tests")
        assert loop.metrics.impl_iterations == 1

    def test_run_without_prior_write_no_increment(self, tmp_task_dir, mock_client):
        """run_subprocess without a preceding write_file does NOT increment."""
        tb_run = make_tool_block("run_subprocess", {"command": ["pytest"]})
        mock_client.messages.stream.side_effect = [
            make_stream_mock(make_response([tb_run])),
            make_stream_mock(make_response([make_text_block("done")], stop_reason="end_turn")),
        ]
        with patch("agent.dispatch_tool", return_value="0 passed in 0.1s"):
            loop = _make_loop(tmp_task_dir, mock_client)
            loop.run_implementation_loop("task", "tests")
        assert loop.metrics.impl_iterations == 0

    def test_max_iterations_reached(self, tmp_task_dir, mock_client):
        """Loop exits when iteration reaches max_iterations."""
        tb_write = make_tool_block("write_file", {"path": "solution.py", "content": "x"}, "tw")
        tb_run = make_tool_block("run_subprocess", {"command": ["pytest"]}, "tr")
        # 2 turns each with write+run → 2 iterations; max_iterations=2 → exits
        mock_client.messages.stream.side_effect = [
            make_stream_mock(make_response([tb_write, tb_run])),
            make_stream_mock(make_response([tb_write, tb_run])),
        ]
        with patch("agent.dispatch_tool", return_value="1 failed in 0.1s"):
            loop = _make_loop(tmp_task_dir, mock_client, AgentConfig(max_iterations=2))
            success, msg = loop.run_implementation_loop("task", "tests")
        assert not success
        assert "Max iterations" in msg
        assert loop.metrics.impl_iterations == 2


# ---------------------------------------------------------------------------
# AgentLoop.run_implementation_loop — prompt_md_section
# ---------------------------------------------------------------------------

class TestPromptMdSection:
    def test_prompt_md_injected(self, tmp_task_dir, mock_client):
        (tmp_task_dir / "solution.prompt.md").write_text("Use numpy.")
        mock_client.messages.stream.return_value = make_stream_mock(make_response(
            [make_text_block("done")], stop_reason="end_turn"
        ))
        loop = _make_loop(tmp_task_dir, mock_client)
        loop.run_implementation_loop("task", "tests")
        # The user message sent to the model should contain prompt_md content
        call_args = mock_client.messages.stream.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "Use numpy." in user_msg


# ---------------------------------------------------------------------------
# AgentLoop.run_implementation_loop — test revision
# ---------------------------------------------------------------------------

class TestTestRevision:
    def _make_dispatch(self, task_dir):
        def dispatch(name, inputs, td):
            if name == "write_file":
                (td / inputs["path"]).write_text(inputs.get("content", ""))
                return f"written {inputs['path']}"
            return "1 failed in 0.1s"
        return dispatch

    def test_revision_offered_file_changed_approved(self, tmp_task_dir, mock_client):
        initial = "def test_old(): assert True\n"
        revised = "def test_new(): assert True\n"
        (tmp_task_dir / "solution_test.py").write_text(initial)

        # Turn 1: write solution.py + run → iteration=1, tests still fail
        # Turn 2: end_turn (agent stops) → ground truth fails → revision offered → continue
        # Turn 3: agent writes solution_test.py
        # Turn 4: end_turn → ground truth passes
        tb_w = make_tool_block("write_file", {"path": "solution.py", "content": "def f(): pass\n"}, "tw")
        tb_r = make_tool_block("run_subprocess", {"command": ["pytest"]}, "tr")
        tb_rev = make_tool_block("write_file", {"path": "solution_test.py", "content": revised}, "tv")

        mock_client.messages.stream.side_effect = [
            make_stream_mock(make_response([tb_w, tb_r])),
            make_stream_mock(make_response([make_text_block("stopping")], stop_reason="end_turn")),
            make_stream_mock(make_response([make_text_block("I think tests are wrong"), tb_rev])),
            make_stream_mock(make_response([make_text_block("done")], stop_reason="end_turn")),
        ]

        with patch("agent.dispatch_tool", side_effect=self._make_dispatch(tmp_task_dir)):
            with patch("agent.subprocess.run") as mock_run:
                mock_run.side_effect = [
                    SimpleNamespace(stdout="== 1 failed in 0.1s ==\n", stderr=""),  # turn 2 ground-truth
                    SimpleNamespace(stdout="== 1 passed in 0.1s ==\n", stderr=""),  # turn 4 ground-truth
                    SimpleNamespace(stdout="solution.py  10  0  100%\n", stderr=""),  # coverage run
                ]
                loop = _make_loop(tmp_task_dir, mock_client,
                                  AgentConfig(max_iterations=5, allow_test_revision=True, auto_approve_revision=True))
                success, _ = loop.run_implementation_loop("task", initial)

        assert success
        assert loop.metrics.test_revisions_attempted == 1
        assert loop.metrics.test_revisions_approved == 1
        assert loop.metrics.test_revision_reasoning == "I think tests are wrong"

    def test_revision_offered_file_changed_denied(self, tmp_task_dir, mock_client):
        initial = "def test_old(): assert True\n"
        revised = "def test_new(): assert True\n"
        (tmp_task_dir / "solution_test.py").write_text(initial)

        # Turn 1: write+run fails, Turn 2: end_turn → revision offered
        # Turn 3: agent writes test file → denied → original restored
        # Turn 4: end_turn → _revision_offered=True → return failure
        tb_w = make_tool_block("write_file", {"path": "solution.py", "content": "def f(): pass\n"}, "tw")
        tb_r = make_tool_block("run_subprocess", {"command": ["pytest"]}, "tr")
        tb_rev = make_tool_block("write_file", {"path": "solution_test.py", "content": revised}, "tv")

        mock_client.messages.stream.side_effect = [
            make_stream_mock(make_response([tb_w, tb_r])),
            make_stream_mock(make_response([make_text_block("stopping")], stop_reason="end_turn")),
            make_stream_mock(make_response([tb_rev])),
            make_stream_mock(make_response([make_text_block("done")], stop_reason="end_turn")),
        ]

        with patch("agent.dispatch_tool", side_effect=self._make_dispatch(tmp_task_dir)):
            with patch("agent.subprocess.run") as mock_run:
                mock_run.return_value = SimpleNamespace(
                    stdout="== 1 failed in 0.1s ==\n", stderr=""
                )
                with patch("builtins.input", return_value="N"):
                    loop = _make_loop(tmp_task_dir, mock_client, AgentConfig(max_iterations=5, allow_test_revision=True))
                    success, _ = loop.run_implementation_loop("task", initial)

        assert not success
        assert (tmp_task_dir / "solution_test.py").read_text() == initial
        assert loop.metrics.test_revisions_attempted == 1
        assert loop.metrics.test_revisions_approved == 0

    def test_revision_offered_file_not_changed(self, tmp_task_dir, mock_client):
        """Revision is offered but agent only writes solution.py — no approval prompt."""
        initial = "def test_old(): assert True\n"
        (tmp_task_dir / "solution_test.py").write_text(initial)

        # Turn 1: write+run fails, Turn 2: end_turn → revision offered
        # Turn 3: agent writes solution.py (not test file), Turn 4: end_turn → passes
        tb_w = make_tool_block("write_file", {"path": "solution.py", "content": "def f(): pass\n"}, "tw")
        tb_r = make_tool_block("run_subprocess", {"command": ["pytest"]}, "tr")
        tb_sol = make_tool_block("write_file", {"path": "solution.py", "content": "def f(): return 1\n"}, "tw2")

        mock_client.messages.stream.side_effect = [
            make_stream_mock(make_response([tb_w, tb_r])),
            make_stream_mock(make_response([make_text_block("stopping")], stop_reason="end_turn")),
            make_stream_mock(make_response([tb_sol])),
            make_stream_mock(make_response([make_text_block("done")], stop_reason="end_turn")),
        ]

        with patch("agent.dispatch_tool", side_effect=self._make_dispatch(tmp_task_dir)):
            with patch("agent.subprocess.run") as mock_run:
                mock_run.side_effect = [
                    SimpleNamespace(stdout="== 1 failed in 0.1s ==\n", stderr=""),  # turn 2
                    SimpleNamespace(stdout="== 1 passed in 0.1s ==\n", stderr=""),  # turn 4
                    SimpleNamespace(stdout="solution.py  10  0  100%\n", stderr=""),  # coverage run
                ]
                loop = _make_loop(tmp_task_dir, mock_client,
                                  AgentConfig(max_iterations=5, allow_test_revision=True, auto_approve_revision=True))
                loop.run_implementation_loop("task", initial)

        # test file was NOT changed → no revision approved
        assert loop.metrics.test_revisions_approved == 0
        assert loop.metrics.test_revisions_attempted == 1


# ---------------------------------------------------------------------------
# AgentLoop._prompt_test_revision_approval
# ---------------------------------------------------------------------------

class TestPromptTestRevisionApproval:
    def _make_loop(self, tmp_task_dir, mock_client):
        return AgentLoop(
            client=mock_client,
            task_dir=tmp_task_dir,
            prompts=MINIMAL_PROMPTS,
            logger=MagicMock(),
        )

    def test_auto_approve(self, tmp_task_dir, mock_client):
        loop = AgentLoop(
            client=mock_client,
            task_dir=tmp_task_dir,
            config=AgentConfig(auto_approve_revision=True),
            prompts=MINIMAL_PROMPTS,
            logger=MagicMock(),
        )
        assert loop._prompt_test_revision_approval("reasoning", "old", "new") is True

    def test_approved_y(self, tmp_task_dir, mock_client):
        loop = self._make_loop(tmp_task_dir, mock_client)
        with patch("builtins.input", return_value="y"):
            assert loop._prompt_test_revision_approval("reasoning", "old", "new") is True

    def test_rejected_n(self, tmp_task_dir, mock_client):
        loop = self._make_loop(tmp_task_dir, mock_client)
        with patch("builtins.input", return_value="N"):
            assert loop._prompt_test_revision_approval("reasoning", "old", "new") is False

    def test_rejected_empty(self, tmp_task_dir, mock_client):
        loop = self._make_loop(tmp_task_dir, mock_client)
        with patch("builtins.input", return_value=""):
            assert loop._prompt_test_revision_approval("", "old", "new") is False

    def test_keyboard_interrupt(self, tmp_task_dir, mock_client):
        loop = self._make_loop(tmp_task_dir, mock_client)
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert loop._prompt_test_revision_approval("", "old", "new") is False

    def test_no_reasoning(self, tmp_task_dir, mock_client, capsys):
        loop = self._make_loop(tmp_task_dir, mock_client)
        with patch("builtins.input", return_value="y"):
            loop._prompt_test_revision_approval("", "old", "new")
        captured = capsys.readouterr()
        # "Agent reasoning:" block should not appear
        assert "Agent reasoning:" not in captured.out


# ---------------------------------------------------------------------------
# AgentLoop.run (orchestrator) — generator interface
# ---------------------------------------------------------------------------

def _done_event(success: bool, message: str = "") -> dict:
    return {"type": "done", "success": success, "message": message,
            "failure_reason": "", "failure_category": ""}


class TestAgentLoopRun:
    def _make_loop(self, tmp_task_dir, mock_client):
        return AgentLoop(
            client=mock_client,
            task_dir=tmp_task_dir,
            prompts=MINIMAL_PROMPTS,
            logger=MagicMock(),
        )

    def test_auto_approve_success(self, tmp_task_dir, mock_client):
        loop = self._make_loop(tmp_task_dir, mock_client)
        loop.metrics.api_retries = 2  # exercises the api_retries log branch in run()
        with patch.object(loop, "generate_tests", return_value="def test_foo(): pass"):
            with patch.object(loop, "_implementation_gen",
                              return_value=iter([_done_event(True, "all passed")])):
                events = list(loop.run("task", auto_approve=True))
        done = next(e for e in events if e["type"] == "done")
        assert done["success"] is True
        # no awaiting_approval event when auto_approve=True
        assert not any(e["type"] == "awaiting_approval" for e in events)

    def test_auto_approve_failure(self, tmp_task_dir, mock_client):
        loop = self._make_loop(tmp_task_dir, mock_client)
        with patch.object(loop, "generate_tests", return_value="def test_foo(): pass"):
            with patch.object(loop, "_implementation_gen",
                              return_value=iter([_done_event(False, "failed")])):
                events = list(loop.run("task", auto_approve=True))
        done = next(e for e in events if e["type"] == "done")
        assert done["success"] is False

    def test_awaiting_approval_emitted_when_not_auto(self, tmp_task_dir, mock_client):
        loop = self._make_loop(tmp_task_dir, mock_client)
        with patch.object(loop, "generate_tests", return_value="def test_foo(): pass"):
            with patch.object(loop, "_implementation_gen",
                              return_value=iter([_done_event(True)])):
                events = list(loop.run("task", auto_approve=False))
        types = [e["type"] for e in events]
        assert "awaiting_approval" in types
        assert events[types.index("awaiting_approval")]["content"] == "def test_foo(): pass"

    def test_hint_sent_via_generator_send(self, tmp_task_dir, mock_client):
        """Hint sent via gen.send() is forwarded to _implementation_gen."""
        loop = self._make_loop(tmp_task_dir, mock_client)
        captured: list[str] = []

        def fake_impl_gen(user_prompt, test_content, hint=""):
            captured.append(hint)
            yield _done_event(True)

        with patch.object(loop, "generate_tests", return_value="def test_foo(): pass"):
            with patch.object(loop, "_implementation_gen", side_effect=fake_impl_gen):
                gen = loop.run("task", auto_approve=False)
                event = next(gen)
                while event["type"] != "awaiting_approval":
                    event = next(gen)
                # send the hint back into the generator
                try:
                    while True:
                        event = gen.send("use a heap")
                except StopIteration:
                    pass

        assert captured == ["use a heap"]

    def test_hint_param_used_when_auto_approve(self, tmp_task_dir, mock_client):
        """hint= parameter is forwarded when auto_approve=True (no send needed)."""
        loop = self._make_loop(tmp_task_dir, mock_client)
        captured: list[str] = []

        def fake_impl_gen(user_prompt, test_content, hint=""):
            captured.append(hint)
            yield _done_event(True)

        with patch.object(loop, "generate_tests", return_value="def test_foo(): pass"):
            with patch.object(loop, "_implementation_gen", side_effect=fake_impl_gen):
                list(loop.run("task", auto_approve=True, hint="use a heap"))

        assert captured == ["use a heap"]

    def test_streaming_events_forwarded_through_run(self, tmp_task_dir, mock_client):
        """text_delta and write_line events from _llm_call propagate through run()."""
        class _StreamWithText:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __iter__(self):
                yield SimpleNamespace(type="text", text="thinking...")
            def get_final_message(self):
                return make_response([make_text_block("done")], stop_reason="end_turn")

        loop = self._make_loop(tmp_task_dir, mock_client)
        mock_client.messages.stream.return_value = _StreamWithText()
        with patch.object(loop, "generate_tests", return_value="def test_foo(): pass"):
            events = list(loop.run("task", auto_approve=True))
        text_events = [e for e in events if e["type"] == "text_delta"]
        assert len(text_events) == 1
        assert text_events[0]["text"] == "thinking..."

    def test_phase_events_in_order(self, tmp_task_dir, mock_client):
        loop = self._make_loop(tmp_task_dir, mock_client)
        with patch.object(loop, "generate_tests", return_value="def test_foo(): pass"):
            with patch.object(loop, "_implementation_gen",
                              return_value=iter([_done_event(True)])):
                events = list(loop.run("task", auto_approve=True))
        phase_events = [e for e in events if e["type"] == "phase"]
        assert phase_events[0]["phase"] == "test_generation"
        assert phase_events[1]["phase"] == "implementation"


# ---------------------------------------------------------------------------
# _call_api_with_retry
# ---------------------------------------------------------------------------

class TestCallApiWithRetry:
    def _make_loop(self, tmp_path, max_retries=3):
        client = MagicMock()
        loop = AgentLoop(
            client=client,
            task_dir=tmp_path,
            config=AgentConfig(max_retries=max_retries),
            prompts=MINIMAL_PROMPTS,
            logger=MagicMock(),
        )
        return loop

    def test_succeeds_on_first_attempt(self, tmp_path):
        loop = self._make_loop(tmp_path)
        fake_response = MagicMock()
        loop.client.messages.create.return_value = fake_response

        result = loop._call_api_with_retry(model="m", max_tokens=10, messages=[])

        assert result is fake_response
        assert loop.metrics.api_retries == 0
        loop.client.messages.create.assert_called_once()

    @patch("agent.time.sleep")
    def test_retries_on_rate_limit_error(self, mock_sleep, tmp_path):
        loop = self._make_loop(tmp_path)
        fake_response = MagicMock()
        rate_limit_err = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
        loop.client.messages.create.side_effect = [
            rate_limit_err,
            rate_limit_err,
            fake_response,
        ]

        result = loop._call_api_with_retry(model="m", max_tokens=10, messages=[])

        assert result is fake_response
        assert loop.metrics.api_retries == 2
        assert mock_sleep.call_count == 2

    @patch("agent.time.sleep")
    def test_retries_on_internal_server_error(self, mock_sleep, tmp_path):
        loop = self._make_loop(tmp_path)
        fake_response = MagicMock()
        server_err = anthropic.InternalServerError.__new__(anthropic.InternalServerError)
        loop.client.messages.create.side_effect = [
            server_err,
            fake_response,
        ]

        result = loop._call_api_with_retry(model="m", max_tokens=10, messages=[])

        assert result is fake_response
        assert loop.metrics.api_retries == 1
        assert mock_sleep.call_count == 1

    @patch("agent.time.sleep")
    def test_retries_exhausted_raises(self, mock_sleep, tmp_path):
        loop = self._make_loop(tmp_path, max_retries=3)
        server_err = anthropic.InternalServerError.__new__(anthropic.InternalServerError)
        loop.client.messages.create.side_effect = server_err

        with pytest.raises(anthropic.InternalServerError):
            loop._call_api_with_retry(model="m", max_tokens=10, messages=[])

        assert loop.metrics.api_retries == 3
        assert mock_sleep.call_count == 3

    def test_non_retryable_error_not_retried(self, tmp_path):
        loop = self._make_loop(tmp_path)
        bad_req = anthropic.BadRequestError.__new__(anthropic.BadRequestError)
        loop.client.messages.create.side_effect = bad_req

        with pytest.raises(anthropic.BadRequestError):
            loop._call_api_with_retry(model="m", max_tokens=10, messages=[])

        assert loop.metrics.api_retries == 0
        loop.client.messages.create.assert_called_once()
