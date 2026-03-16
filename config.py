"""Centralized configuration — single source of truth for shared defaults.

Values can be overridden via environment variables (loaded from .env by the
CLI entry points before this module is imported).
"""

import os

DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-6")
DEFAULT_PROMPTS_VERSION: str = os.getenv("DEFAULT_PROMPTS_VERSION", "v2.8")
DEFAULT_EVAL_PROMPTS_VERSION: str = os.getenv("DEFAULT_EVAL_PROMPTS_VERSION", "eval-v1.4")
DEFAULT_MAX_ITERATIONS: int = int(os.getenv("DEFAULT_MAX_ITERATIONS", "10"))
DEFAULT_MAX_ITERATIONS_EVAL: int = int(os.getenv("DEFAULT_MAX_ITERATIONS_EVAL", "5"))
