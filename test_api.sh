#!/usr/bin/env bash
set -euo pipefail

BASE="http://localhost:8000"

echo "=== GET /health ==="
curl -s "$BASE/health" | python3 -m json.tool

echo ""
echo "=== GET /config/defaults ==="
curl -s "$BASE/config/defaults" | python3 -m json.tool

echo ""
echo "=== WebSocket /ws/run ==="
uv run python3 - <<'EOF'
import asyncio
import json
import websockets

async def test():
    uri = "ws://localhost:8000/ws/run"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "prompt": "Write a function that returns the nth fibonacci number",
            "config": {"auto_approve_revision": True}
        }))
        print("Streaming events:\n")
        while True:
            msg = json.loads(await ws.recv())
            t = msg.get("type")
            if t == "phase":
                print(f"\n--- PHASE: {msg['phase']} ---")
            elif t == "test_generated":
                print(f"[test_generated] {msg['test_count']} tests")
            elif t == "text_delta":
                print(msg["text"], end="", flush=True)
            elif t == "write_line":
                print(f"[write] {msg['path']}:{msg['line_num']}  {msg['line']}")
            elif t == "tool_call":
                tool = msg["tool"]
                if tool == "run_subprocess":
                    status = "PASS" if msg.get("passed") else "FAIL"
                    print(f"\n[tool] {tool} [{status}] {msg.get('summary', '')}")
                else:
                    print(f"\n[tool] {tool} {msg.get('path', '')}")
            elif t == "coverage":
                print(f"\n[coverage] {msg['pct']}%")
            elif t == "awaiting_approval":
                print("\n[awaiting_approval] sending empty hint (auto-continue)")
                await ws.send(json.dumps({"type": "hint", "hint": None}))
            elif t == "done":
                print(f"\n\n[done] success={msg['success']}")
                if msg.get("solution"):
                    print("\n--- solution.py ---")
                    print(msg["solution"])
                break
            elif t == "error":
                print(f"\n[error] {msg['message']}")
                break

asyncio.run(test())
EOF
