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
                            instance = MagicMock()
                            MockLoop.return_value = instance
                            microagent.main()
        instance.run.assert_called_once_with("write a function")

    def test_default_task_dir_created(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["microagent.py", "write something"])
        with patch("microagent.db.get_db", return_value=MagicMock()):
            with patch("microagent.db.init_db"):
                with patch("microagent.db.seed_if_empty"):
                    with patch("microagent.anthropic.Anthropic"):
                        with patch("microagent.AgentLoop") as MockLoop:
                            MockLoop.return_value = MagicMock()
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
                            MockLoop.return_value = MagicMock()
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
                            MockLoop.return_value = MagicMock()
                            microagent.main()
        _, kwargs = MockLoop.call_args
        assert kwargs["allow_test_revision"] is True

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
                            MockLoop.return_value = MagicMock()
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
                            MockLoop.return_value = MagicMock()
                            microagent.main()
        _, kwargs = MockLoop.call_args
        assert kwargs["model"] == "claude-opus-4-6"

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
