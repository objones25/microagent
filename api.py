"""FastAPI + WebSocket API for microagent.

WebSocket protocol
------------------
Client → Server (once, to start):
  {"prompt": "Write fizzbuzz", "config": {...optional...}}

Server → Client (streaming):
  {"type": "phase", "phase": "test_generation"|"implementation"}
  {"type": "test_generated", "content": str, "test_count": int}
  {"type": "text_delta", "text": str}
  {"type": "write_line", "path": str, "line": str, "line_num": int}
  {"type": "tool_call", "tool": str, ...}
  {"type": "coverage", "pct": float}
  {"type": "awaiting_approval", "content": str}   ← pause for test approval
  {"type": "done", "success": bool, "message": str, "solution": str}
  {"type": "error", "message": str}

Client → Server (resume after awaiting_approval):
  {"type": "hint", "hint": "use a heap"}   or   {"type": "hint", "hint": null}
"""

import asyncio
import os
import queue
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent import AgentConfig, AgentLoop
from config import DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL, DEFAULT_PROMPTS_VERSION

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="microagent API", lifespan=lifespan)

cors_origins = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in cors_origins.split(",")] if cors_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config/defaults")
async def config_defaults():
    return {
        "model": DEFAULT_MODEL,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "prompts_version": DEFAULT_PROMPTS_VERSION,
    }


@app.websocket("/ws/run")
async def websocket_run(websocket: WebSocket):
    await websocket.accept()

    # ── Parse start message ──────────────────────────────────────────
    try:
        start_msg = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    prompt = start_msg.get("prompt", "").strip()
    if not prompt:
        await websocket.send_json({"type": "error", "message": "prompt is required"})
        await websocket.close()
        return

    cfg_data = start_msg.get("config") or {}
    try:
        config = AgentConfig(
            model=cfg_data.get("model", DEFAULT_MODEL),
            max_iterations=int(cfg_data.get("max_iterations", DEFAULT_MAX_ITERATIONS)),
            prompts_version=cfg_data.get("prompts_version", DEFAULT_PROMPTS_VERSION),
            allow_test_revision=bool(cfg_data.get("allow_test_revision", False)),
            auto_approve_revision=bool(cfg_data.get("auto_approve_revision", False)),
            min_coverage=float(cfg_data.get("min_coverage", 0.0)),
        )
    except (TypeError, ValueError) as e:
        await websocket.send_json({"type": "error", "message": f"invalid config: {e}"})
        await websocket.close()
        return

    # ── Queues bridging sync generator ↔ async handler ───────────────
    event_loop = asyncio.get_event_loop()
    event_queue: asyncio.Queue = asyncio.Queue()
    hint_queue: queue.Queue = queue.Queue()

    # ── Generator thread ─────────────────────────────────────────────
    def run_agent():
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            client = anthropic.Anthropic(api_key=api_key)
            with tempfile.TemporaryDirectory(prefix="task-") as tmpdir:
                task_dir = Path(tmpdir)
                loop = AgentLoop(client=client, task_dir=task_dir, config=config)
                gen = loop.run(prompt, auto_approve=config.allow_test_revision)

                event = next(gen)
                while True:
                    # Forward event to async handler
                    if event.get("type") != "_response":
                        event_loop.call_soon_threadsafe(event_queue.put_nowait, event)

                    if event.get("type") == "awaiting_approval":
                        # Block until the async handler sends a hint back
                        hint = hint_queue.get()
                        event = gen.send(hint)
                    elif event.get("type") == "done":
                        break
                    else:
                        event = next(gen)

        except StopIteration:
            pass
        except Exception as e:
            event_loop.call_soon_threadsafe(
                event_queue.put_nowait, {"type": "error", "message": str(e)}
            )
        finally:
            event_loop.call_soon_threadsafe(event_queue.put_nowait, None)  # sentinel

    executor_thread = threading.Thread(target=run_agent, daemon=True)
    executor_thread.start()

    # ── Async relay loop ─────────────────────────────────────────────
    try:
        while True:
            event = await event_queue.get()

            if event is None:  # sentinel — generator finished
                break

            await websocket.send_json(event)

            if event.get("type") == "awaiting_approval":
                # Wait for client approval/hint message
                try:
                    client_msg = await websocket.receive_json()
                    hint = client_msg.get("hint") or ""
                except WebSocketDisconnect:
                    hint_queue.put("")
                    break
                hint_queue.put(hint)

            elif event.get("type") in ("done", "error"):
                break

    except WebSocketDisconnect:  # pragma: no cover
        pass  # pragma: no cover
    finally:
        try:
            await websocket.close()
        except Exception:  # pragma: no cover
            pass  # pragma: no cover
