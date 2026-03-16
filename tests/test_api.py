"""Tests for api.py — REST endpoints and WebSocket /ws/run."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

class TestLifespan:
    def test_startup_and_shutdown(self):
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop_mock(gen_fn):
    mock_loop = MagicMock()
    mock_loop.run.side_effect = gen_fn
    return mock_loop


def _done_gen(*extra):
    """Generator that optionally yields extra events then a done event."""
    def _run(prompt, auto_approve=False, hint=""):
        yield from extra
        yield {"type": "done", "success": True, "message": "ok", "solution": ""}
    return _run


def _recv_all(ws):
    """Collect all messages until done or error."""
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg["type"] in ("done", "error"):
            break
    return msgs


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestConfigDefaults:
    def test_returns_all_fields(self):
        resp = client.get("/config/defaults")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "max_iterations" in data
        assert "prompts_version" in data

    def test_max_iterations_is_int(self):
        data = client.get("/config/defaults").json()
        assert isinstance(data["max_iterations"], int)


# ---------------------------------------------------------------------------
# WebSocket — input validation
# ---------------------------------------------------------------------------

class TestWebSocketValidation:
    def test_missing_prompt_sends_error(self):
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({})
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "prompt" in msg["message"]

    def test_blank_prompt_sends_error(self):
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({"prompt": "   "})
            msg = ws.receive_json()
        assert msg["type"] == "error"

    def test_bad_max_iterations_sends_error(self):
        with client.websocket_connect("/ws/run") as ws:
            ws.send_json({"prompt": "do it", "config": {"max_iterations": "bad"}})
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "config" in msg["message"]


# ---------------------------------------------------------------------------
# WebSocket — event streaming
# ---------------------------------------------------------------------------

class TestWebSocketStreaming:
    def _run(self, gen_fn, payload=None):
        payload = payload or {"prompt": "Write fibonacci"}
        loop_patch = patch("api.AgentLoop", return_value=_make_loop_mock(gen_fn))
        client_patch = patch("api.anthropic.Anthropic")
        with loop_patch, client_patch:
            with client.websocket_connect("/ws/run") as ws:
                ws.send_json(payload)
                return _recv_all(ws)

    def test_done_event_received(self):
        msgs = self._run(_done_gen())
        assert msgs[-1]["type"] == "done"
        assert msgs[-1]["success"] is True

    def test_intermediate_events_forwarded(self):
        phase = {"type": "phase", "phase": "test_generation"}
        test_gen = {"type": "test_generated", "content": "...", "test_count": 2}
        msgs = self._run(_done_gen(phase, test_gen))
        types = [m["type"] for m in msgs]
        assert "phase" in types
        assert "test_generated" in types
        assert "done" in types

    def test_solution_in_done_event(self):
        def gen_fn(prompt, auto_approve=False, hint=""):
            yield {"type": "done", "success": True, "message": "", "solution": "def f(): pass"}

        msgs = self._run(gen_fn)
        assert msgs[-1]["solution"] == "def f(): pass"

    def test_error_event_on_agent_exception(self):
        def gen_fn(prompt, auto_approve=False, hint=""):
            raise RuntimeError("boom")
            yield  # makes it a generator

        msgs = self._run(gen_fn)
        assert msgs[-1]["type"] == "error"
        assert "boom" in msgs[-1]["message"]

    def test_config_fields_forwarded_to_agent_loop(self):
        captured = {}

        def gen_fn(prompt, auto_approve=False, hint=""):
            yield {"type": "done", "success": True, "message": "", "solution": ""}

        def capture_loop(client, task_dir, config):
            captured["config"] = config
            return _make_loop_mock(gen_fn)

        loop_patch = patch("api.AgentLoop", side_effect=capture_loop)
        client_patch = patch("api.anthropic.Anthropic")
        with loop_patch, client_patch:
            with client.websocket_connect("/ws/run") as ws:
                ws.send_json({
                    "prompt": "do it",
                    "config": {
                        "model": "claude-haiku-4-5-20251001",
                        "max_iterations": 3,
                        "min_coverage": 80.0,
                        "allow_test_revision": True,
                        "auto_approve_revision": True,
                    },
                })
                _recv_all(ws)

        cfg = captured["config"]
        assert cfg.model == "claude-haiku-4-5-20251001"
        assert cfg.max_iterations == 3
        assert cfg.min_coverage == 80.0
        assert cfg.allow_test_revision is True
        assert cfg.auto_approve_revision is True

    def test_missing_config_uses_defaults(self):
        captured = {}

        def gen_fn(prompt, auto_approve=False, hint=""):
            yield {"type": "done", "success": True, "message": "", "solution": ""}

        def capture_loop(client, task_dir, config):
            captured["config"] = config
            return _make_loop_mock(gen_fn)

        loop_patch = patch("api.AgentLoop", side_effect=capture_loop)
        client_patch = patch("api.anthropic.Anthropic")
        with loop_patch, client_patch:
            with client.websocket_connect("/ws/run") as ws:
                ws.send_json({"prompt": "do it"})
                _recv_all(ws)

        from config import DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL
        assert captured["config"].model == DEFAULT_MODEL
        assert captured["config"].max_iterations == DEFAULT_MAX_ITERATIONS


# ---------------------------------------------------------------------------
# WebSocket — awaiting_approval pause/resume
# ---------------------------------------------------------------------------

class TestWebSocketApproval:
    def test_awaiting_approval_pause_and_resume(self):
        def gen_fn(prompt, auto_approve=False, hint=""):
            yield {"type": "phase", "phase": "test_generation"}
            yield {"type": "awaiting_approval", "content": "test content here"}
            yield {"type": "done", "success": True, "message": "ok", "solution": ""}

        loop_patch = patch("api.AgentLoop", return_value=_make_loop_mock(gen_fn))
        client_patch = patch("api.anthropic.Anthropic")
        with loop_patch, client_patch:
            with client.websocket_connect("/ws/run") as ws:
                ws.send_json({"prompt": "do it", "config": {"allow_test_revision": True}})
                msgs = []
                while True:
                    msg = ws.receive_json()
                    msgs.append(msg)
                    if msg["type"] == "awaiting_approval":
                        ws.send_json({"type": "hint", "hint": "use a heap"})
                    elif msg["type"] in ("done", "error"):
                        break

        types = [m["type"] for m in msgs]
        assert "awaiting_approval" in types
        assert msgs[-1]["type"] == "done"

    def test_null_hint_continues_without_error(self):
        def gen_fn(prompt, auto_approve=False, hint=""):
            yield {"type": "awaiting_approval", "content": "tests"}
            yield {"type": "done", "success": True, "message": "ok", "solution": ""}

        loop_patch = patch("api.AgentLoop", return_value=_make_loop_mock(gen_fn))
        client_patch = patch("api.anthropic.Anthropic")
        with loop_patch, client_patch:
            with client.websocket_connect("/ws/run") as ws:
                ws.send_json({"prompt": "do it"})
                msgs = []
                while True:
                    msg = ws.receive_json()
                    msgs.append(msg)
                    if msg["type"] == "awaiting_approval":
                        ws.send_json({"type": "hint", "hint": None})
                    elif msg["type"] in ("done", "error"):
                        break

        assert msgs[-1]["type"] == "done"

    def test_disconnect_after_awaiting_approval(self):
        """Client disconnects without responding to awaiting_approval."""
        def gen_fn(prompt, auto_approve=False, hint=""):
            yield {"type": "awaiting_approval", "content": "tests"}
            yield {"type": "done", "success": True, "message": "ok", "solution": ""}

        loop_patch = patch("api.AgentLoop", return_value=_make_loop_mock(gen_fn))
        client_patch = patch("api.anthropic.Anthropic")
        with loop_patch, client_patch:
            with client.websocket_connect("/ws/run") as ws:
                ws.send_json({"prompt": "do it"})
                msg = ws.receive_json()
                assert msg["type"] == "awaiting_approval"
                # Close without responding — triggers the disconnect branch

    def test_hint_value_sent_to_generator(self):
        received = {}

        def gen_fn(prompt, auto_approve=False, hint=""):
            sent = yield {"type": "awaiting_approval", "content": "tests"}
            received["hint"] = sent
            yield {"type": "done", "success": True, "message": "ok", "solution": ""}

        loop_patch = patch("api.AgentLoop", return_value=_make_loop_mock(gen_fn))
        client_patch = patch("api.anthropic.Anthropic")
        with loop_patch, client_patch:
            with client.websocket_connect("/ws/run") as ws:
                ws.send_json({"prompt": "do it"})
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "awaiting_approval":
                        ws.send_json({"type": "hint", "hint": "use a trie"})
                    elif msg["type"] in ("done", "error"):
                        break

        assert received["hint"] == "use a trie"


# ---------------------------------------------------------------------------
# WebSocket — edge cases
# ---------------------------------------------------------------------------

class TestWebSocketEdgeCases:
    def test_disconnect_before_sending_prompt(self):
        """Client connects but disconnects before sending the start message."""
        with client.websocket_connect("/ws/run") as ws:
            pass  # exit immediately without sending anything

    def test_generator_exhausted_without_done(self):
        """Generator runs out of events without emitting done — sentinel terminates the loop."""
        def gen_fn(prompt, auto_approve=False, hint=""):
            yield {"type": "phase", "phase": "test_generation"}
            # Generator exhausts here — no done event

        loop_patch = patch("api.AgentLoop", return_value=_make_loop_mock(gen_fn))
        client_patch = patch("api.anthropic.Anthropic")
        with loop_patch, client_patch:
            with client.websocket_connect("/ws/run") as ws:
                ws.send_json({"prompt": "do it"})
                msgs = []
                # Receive until the server closes — no done event so we read until disconnect
                import contextlib
                with contextlib.suppress(Exception):
                    for _ in range(10):
                        msgs.append(ws.receive_json())
        types = [m["type"] for m in msgs]
        assert "phase" in types
