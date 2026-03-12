import ast
import math
import operator
import os
import subprocess
from pathlib import Path

import requests
from firecrawl import FirecrawlApp

TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": (
            "Read the full text content of a file inside the task directory. "
            "Use this to read solution_test.py before writing any implementation, "
            "or to re-read solution.py when debugging failures. "
            "Returns an error string if the file does not exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File path relative to the task directory, e.g. 'solution_test.py' or 'solution.py'. "
                        "Absolute paths are also accepted as long as they resolve inside the task directory."
                    ),
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write (or overwrite) a file in the task directory with the given content. "
            "Use this to write your implementation to solution.py. "
            "Do NOT use this to write to solution_test.py — tests are locked. "
            "Always provide the complete file content; partial updates are not supported."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to task directory, e.g. 'solution.py'.",
                },
                "content": {
                    "type": "string",
                    "description": "Complete file content to write. Must be valid, importable Python.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_subprocess",
        "description": (
            "Run pytest against the test file and return the full output (stdout + stderr). "
            "Always run this after writing solution.py to verify correctness. "
            "Only pytest commands are allowed — any other command will be rejected. "
            "Example: ['pytest', 'solution_test.py', '-v']"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Command as a list of strings. Must start with 'pytest'. "
                        "Example: ['pytest', 'solution_test.py', '-v', '--tb=short']"
                    ),
                }
            },
            "required": ["command"],
        },
    },
    {
        "name": "context7_docs",
        "description": (
            "Fetch up-to-date documentation and code examples for a programming library. "
            "Use this when you need accurate API signatures, method names, or usage patterns "
            "before writing code that uses an external library. "
            "Prefer this over firecrawl_search for library-specific API questions. "
            "Example: library='requests', query='how to set headers and timeout on a GET request'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "library": {
                    "type": "string",
                    "description": "The library package name as it appears on PyPI, e.g. 'requests', 'httpx', 'pytest'.",
                },
                "query": {
                    "type": "string",
                    "description": "A specific question or topic to look up, e.g. 'session authentication with cookies'.",
                },
            },
            "required": ["library", "query"],
        },
    },
    {
        "name": "firecrawl_scrape",
        "description": (
            "Scrape a specific URL and return its content as markdown. "
            "Use when you have a known URL containing documentation, an API reference, or a code example. "
            "Prefer context7_docs for library API questions. "
            "Use firecrawl_search instead when you don't have a specific URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to scrape, e.g. 'https://docs.python.org/3/library/pathlib.html'.",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "firecrawl_search",
        "description": (
            "Search the web and return results with content snippets. "
            "Use to find current information, code examples, or solutions when you don't have a specific URL. "
            "Prefer context7_docs for library API questions — it returns more structured docs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'python requests library retry on failure example'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return. Default 5, max 10.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a safe mathematical expression. Supports: +, -, *, /, **, %, //, "
            "abs(), round(), min(), max(), sum(), math.* functions. "
            "Use for precise arithmetic instead of estimating."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '2 ** 32' or 'math.sqrt(144)'",
                },
            },
            "required": ["expression"],
        },
    },
]


# ------------------------------------------------------------------
# Filesystem tools
# ------------------------------------------------------------------

def _resolve_path(path: str, task_dir: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = task_dir / p
    resolved = p.resolve()
    task_dir_resolved = task_dir.resolve()
    if not str(resolved).startswith(str(task_dir_resolved)):
        raise PermissionError(f"Path '{path}' is outside task directory")
    return resolved


def tool_read_file(inputs: dict, task_dir: Path) -> str:
    path = _resolve_path(inputs["path"], task_dir)
    if not path.exists():
        return f"Error: file not found: {path}"
    return path.read_text()


def tool_write_file(inputs: dict, task_dir: Path) -> str:
    path = _resolve_path(inputs["path"], task_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inputs["content"])
    return f"Written {len(inputs['content'])} bytes to {path.name}"


def tool_run_subprocess(inputs: dict, task_dir: Path) -> str:
    command: list[str] = inputs["command"]
    if not command or command[0] != "pytest":
        return "Error: only 'pytest' commands are allowed"
    result = subprocess.run(
        command,
        cwd=str(task_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    return output or f"(no output, exit code {result.returncode})"


# ------------------------------------------------------------------
# Context7 docs lookup
# ------------------------------------------------------------------

_CONTEXT7_BASE = "https://context7.com/api/v2"


def tool_context7_docs(inputs: dict) -> str:
    api_key = os.environ.get("CONTEXT7_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # Step 1: resolve library name → library ID
    try:
        search_resp = requests.get(
            f"{_CONTEXT7_BASE}/libs/search",
            headers=headers,
            params={"libraryName": inputs["library"], "query": inputs["query"]},
            timeout=15,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("results", [])
        if not results:
            return f"No library found matching '{inputs['library']}'"
        library_id = results[0]["id"]
        library_title = results[0].get("title", library_id)
    except Exception as e:
        return f"Context7 library resolve error: {e}"

    # Step 2: fetch documentation context
    try:
        ctx_resp = requests.get(
            f"{_CONTEXT7_BASE}/context",
            headers=headers,
            params={"libraryId": library_id, "query": inputs["query"], "type": "txt"},
            timeout=20,
        )
        ctx_resp.raise_for_status()
        text = ctx_resp.text.strip()
        return f"# {library_title} docs ({library_id})\n\n{text}" if text else "No documentation found."
    except Exception as e:
        return f"Context7 docs fetch error: {e}"


# ------------------------------------------------------------------
# Firecrawl tools
# ------------------------------------------------------------------

def _firecrawl_app() -> FirecrawlApp:
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    return FirecrawlApp(api_key=api_key)


def tool_firecrawl_scrape(inputs: dict) -> str:
    try:
        app = _firecrawl_app()
        result = app.scrape_url(inputs["url"], formats=["markdown"])
        return result.markdown or "(no content)"
    except Exception as e:
        return f"Firecrawl scrape error: {e}"


def tool_firecrawl_search(inputs: dict) -> str:
    try:
        app = _firecrawl_app()
        limit = inputs.get("limit", 5)
        results = app.search(inputs["query"], limit=limit)
        lines = []
        for item in results.data:
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = (item.get("markdown") or item.get("description") or "")[:300]
            lines.append(f"### {title}\n{url}\n{snippet}")
        return "\n\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Firecrawl search error: {e}"


# ------------------------------------------------------------------
# Calculator
# ------------------------------------------------------------------

_SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "math": math,
}
_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.expr) -> float | int:
    match node:
        case ast.Constant(value=v) if isinstance(v, int | float):
            return v
        case ast.BinOp(left=l, op=op, right=r):
            fn = _SAFE_OPS.get(type(op))
            if fn is None:
                raise ValueError(f"Unsupported operator: {type(op).__name__}")
            return fn(_safe_eval(l), _safe_eval(r))
        case ast.UnaryOp(op=op, operand=o):
            fn = _SAFE_OPS.get(type(op))
            if fn is None:
                raise ValueError(f"Unsupported unary operator: {type(op).__name__}")
            return fn(_safe_eval(o))
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]):
            if name not in _SAFE_NAMES:
                raise ValueError(f"Function not allowed: {name}")
            return _SAFE_NAMES[name](*[_safe_eval(a) for a in args])
        case ast.Call(func=ast.Attribute(value=ast.Name(id="math"), attr=attr), args=args, keywords=[]):
            fn = getattr(math, attr, None)
            if fn is None:
                raise ValueError(f"math.{attr} not found")
            return fn(*[_safe_eval(a) for a in args])
        case ast.Name(id=name):
            if name in _SAFE_NAMES:
                return _SAFE_NAMES[name]  # type: ignore[return-value]
            raise ValueError(f"Name not allowed: {name}")
        case _:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def tool_calculator(inputs: dict) -> str:
    expr = inputs["expression"].strip()
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------

def dispatch_tool(name: str, inputs: dict, task_dir: Path) -> str:
    match name:
        case "read_file":
            return tool_read_file(inputs, task_dir)
        case "write_file":
            return tool_write_file(inputs, task_dir)
        case "run_subprocess":
            return tool_run_subprocess(inputs, task_dir)
        case "context7_docs":
            return tool_context7_docs(inputs)
        case "firecrawl_scrape":
            return tool_firecrawl_scrape(inputs)
        case "firecrawl_search":
            return tool_firecrawl_search(inputs)
        case "calculator":
            return tool_calculator(inputs)
        case _:
            return f"Error: unknown tool '{name}'"
