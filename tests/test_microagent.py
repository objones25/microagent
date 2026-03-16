"""Tests for microagent.py — targets >98% line/branch coverage."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import microagent


def _db_patches():
    """Patch all DB calls in microagent.main()."""
    return [
        patch("microagent.db.get_db", return_value=MagicMock()),
        patch("microagent.db.init_db"),
        patch("microagent.db.seed_if_empty"),
    ]


class TestMain:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(sys, "argv", ["microagent.py", "write a function"])
        with pytest.raises(SystemExit) as exc:
            microagent.main()
        assert exc.value.code == 1

    def _done_event(self, success=True, message=""):
        return {"type": "done", "success": success, "message": message,
                "failure_reason": "", "failure_category": ""}

    def _mock_loop(self, MockLoop, events):
        def _gen():
            for event in events:
                yield event
        instance = MagicMock()
        instance.run.return_value = _gen()
        MockLoop.return_value = instance
        return instance

    def test_basic_run(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            sys, "argv",
            ["microagent.py", "write a function", "--task-dir", str(tmp_path)]
        )
        with patch("microagent.db.get_db", return_value=MagicMock()):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            instance = self._mock_loop(
                                MockLoop, [self._done_event(True, "3 passed")]
                            )
                            microagent.main()
        instance.run.assert_called_once_with("write a function")
        out = capsys.readouterr().out
        assert "SUCCESS" in out

    def test_render_failure_event(self, tmp_path, monkeypatch, capsys):
        """_render_event covers FAILED branch and message print."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            sys, "argv",
            ["microagent.py", "do thing", "--task-dir", str(tmp_path)]
        )
        with patch("microagent.db.get_db", return_value=MagicMock()):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            self._mock_loop(
                                MockLoop, [self._done_event(False, "tests failed")]
                            )
                            microagent.main()
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "tests failed" in out

    def test_awaiting_approval_handled(self, tmp_path, monkeypatch, capsys):
        """awaiting_approval event shows tests, accepts hint, then waits for Enter."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            sys, "argv",
            ["microagent.py", "do thing", "--task-dir", str(tmp_path)]
        )
        events = [
            {"type": "awaiting_approval", "content": "def test_x(): pass"},
            self._done_event(True),
        ]
        with patch("microagent.db.get_db", return_value=MagicMock()):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            self._mock_loop(MockLoop, events)
                            # two input() calls: hint prompt, then Enter to start
                            with patch("builtins.input", side_effect=["use a heap", ""]):
                                microagent.main()
        out = capsys.readouterr().out
        assert "GENERATED TESTS" in out
        assert "def test_x(): pass" in out

    def test_awaiting_approval_keyboard_interrupt(self, tmp_path, monkeypatch):
        """KI during the hint prompt aborts cleanly."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            sys, "argv",
            ["microagent.py", "do thing", "--task-dir", str(tmp_path)]
        )
        events = [{"type": "awaiting_approval", "content": "tests"}]
        with patch("microagent.db.get_db", return_value=MagicMock()):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            self._mock_loop(MockLoop, events)
                            # KI on the first input() (hint prompt)
                            with patch("builtins.input", side_effect=KeyboardInterrupt):
                                with pytest.raises(SystemExit):
                                    microagent.main()

    def test_default_task_dir_created(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["microagent.py", "write something"])
        with patch("microagent.db.get_db", return_value=MagicMock()):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            self._mock_loop(MockLoop, [])
                            microagent.main()
        # A task-YYYYMMDD-HHMMSS dir should have been created
        created = [d for d in tmp_path.iterdir() if d.name.startswith("task-")]
        assert len(created) == 1

    def test_copies_prompt_md(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".prompt.md").write_text("Use recursion.")
        monkeypatch.setattr(sys, "argv", ["microagent.py", "do thing"])
        with patch("microagent.db.get_db", return_value=MagicMock()):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            self._mock_loop(MockLoop, [])
                            microagent.main()
        task_dirs = [d for d in tmp_path.iterdir() if d.name.startswith("task-")]
        assert len(task_dirs) == 1
        assert (task_dirs[0] / "solution.prompt.md").exists()

    def test_allow_test_revision_passed_to_loop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            sys, "argv",
            ["microagent.py", "do thing", "--task-dir", str(tmp_path),
             "--allow-test-revision"]
        )
        with patch("microagent.db.get_db", return_value=MagicMock()):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            self._mock_loop(MockLoop, [])
                            microagent.main()
        _, kwargs = MockLoop.call_args
        assert kwargs["config"].allow_test_revision is True

    def test_db_conn_passed_to_agent_loop(self, tmp_path, monkeypatch):
        """db_conn kwarg is forwarded to AgentLoop."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            sys, "argv",
            ["microagent.py", "do thing", "--task-dir", str(tmp_path)]
        )
        fake_conn = MagicMock()
        with patch("microagent.db.get_db", return_value=fake_conn):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            self._mock_loop(MockLoop, [])
                            microagent.main()
        _, kwargs = MockLoop.call_args
        assert kwargs["db_conn"] is fake_conn

    def test_custom_model(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            sys, "argv",
            ["microagent.py", "do thing", "--task-dir", str(tmp_path),
             "--model", "claude-opus-4-6"]
        )
        with patch("microagent.db.get_db", return_value=MagicMock()):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            self._mock_loop(MockLoop, [])
                            microagent.main()
        _, kwargs = MockLoop.call_args
        assert kwargs["config"].model == "claude-opus-4-6"

    def test_keyboard_interrupt(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            sys, "argv",
            ["microagent.py", "do thing", "--task-dir", str(tmp_path)]
        )
        with patch("microagent.db.get_db", return_value=MagicMock()):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            instance = MagicMock()
                            instance.run.side_effect = KeyboardInterrupt
                            MockLoop.return_value = instance
                            with pytest.raises(SystemExit):
                                microagent.main()
