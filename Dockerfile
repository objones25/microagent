FROM python:3.13-slim AS builder
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files and install (no dev extras, uses main deps only)
COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --no-dev --frozen

# ── final stage ──────────────────────────────────────────────────────────────
FROM python:3.13-slim
WORKDIR /app

# System packages the agent's subprocesses may need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy installed packages and uv binary from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy source code (excludes items in .dockerignore)
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300"]
