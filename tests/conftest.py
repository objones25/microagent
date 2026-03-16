"""Shared fixtures and helpers for the microagent test suite."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Minimal prompt dicts used by agent tests
# ---------------------------------------------------------------------------

MINIMAL_PROMPTS = {
    "test_generation": {
        "system": "tg-sys",
        "user": "tg-user: {user_prompt}",
    },
    "implementation": {
        "system": "impl-sys",
        "user": "impl: {user_prompt} | {test_content} | {prompt_md_section}",
    },
    "prompt_md_section": {
        "template": "<ctx>{prompt_md}</ctx>\n",
    },
    "test_revision": {
        "user": "tried {n} times. revise?",
    },
}

MINIMAL_EVAL_PROMPTS = {
    "judge": {"system": "j-sys"},
    "judge_single": {
        "template": (
            "single n={n} mi={max_iter} tgs={test_gen_system} "
            "is={impl_system} rb={results_block}"
        ),
    },
    "judge_ab": {
        "template": (
            "ab mi={max_iter} v1tg={v1_test_gen} v1i={v1_impl} "
            "v2tg={v2_test_gen} v2i={v2_impl} "
            "v1rb={v1_results_block} v2rb={v2_results_block}"
        ),
    },
    "prompt_optimizer": {"system": "opt-sys"},
    "meta_judge": {
        "system": "mj-sys",
        "template": (
            "mj js={judge_system} jt={judge_template} "
            "n={n_tasks} p={n_passed} f={n_failed} avg={avg_iter:.1f} "
            "jtext={judgment_text}"
        ),
    },
}


# ---------------------------------------------------------------------------
# Mock response builders
# ---------------------------------------------------------------------------

def make_text_block(text: str = ""):
    return SimpleNamespace(type="text", text=text)


def make_tool_block(name: str, inputs: dict, tool_id: str = "tc_1"):
    return SimpleNamespace(type="tool_use", name=name, input=inputs, id=tool_id)


def make_response(content, stop_reason="tool_use", in_tok=100, out_tok=50):
    usage = SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)


def make_stream_mock(response):
    """Wrap a make_response() result in a mock context manager for client.messages.stream().

    Emits no intermediate stream events — only returns the final message.
    Use this to replace messages.create.return_value in implementation loop tests.
    """
    class _StreamMock:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def __iter__(self):
            return iter([])
        def get_final_message(self):
            return response
    return _StreamMock()


# ---------------------------------------------------------------------------
# RunMetrics factory
# ---------------------------------------------------------------------------

def make_metrics(
    task="do something",
    success=True,
    iterations=1,
    impl_dur=5.0,
    test_dur=1.0,
    task_dir="/tmp/t",
    failure_reason="",
):
    from logger import RunMetrics

    m = RunMetrics(
        task_prompt=task,
        task_dir=task_dir,
        prompts_version="v1",
        model="claude-sonnet-4-6",
        started_at="2026-01-01T00:00:00+00:00",
    )
    m.success = success
    m.impl_iterations = iterations
    m.impl_duration_s = impl_dur
    m.test_gen_duration_s = test_dur
    m.failure_reason = failure_reason
    m.tool_calls = {"write_file": 1, "run_subprocess": 1}
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_task_dir(tmp_path):
    (tmp_path / "solution_test.py").write_text(
        "def test_foo():\n    assert 1 == 1\n"
    )
    return tmp_path


@pytest.fixture
def mock_client():
    return MagicMock()
