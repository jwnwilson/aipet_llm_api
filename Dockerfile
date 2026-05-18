# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

# GGML_NATIVE=0 produces a portable ARM64 binary without native CPU tuning
ENV CMAKE_ARGS="-DGGML_NATIVE=0"
RUN uv sync --extra inference --no-dev --frozen --no-install-project

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

COPY alembic.ini ./
COPY src/ src/

ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

# Model is downloaded at startup — do not bake it into the image.
# Set one of:
#   MODEL_S3_KEY  — S3 object key to download at startup (requires AWS_S3_BUCKET + credentials)
#   MODEL_PATH    — local path for dev/testing with a volume-mounted model
# Or seed an active model in the database with a gguf_path pointing to S3.

RUN mkdir -p /app/data /app/models

EXPOSE 8000

CMD ["sh", "-c", "python -m interactors.cli.db.db_migrate && uvicorn interactors.api.app:app --host 0.0.0.0 --port 8000"]
