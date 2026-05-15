FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

# Install runtime deps into the system Python (no train/dev extras)
# GGML_NATIVE=0 produces a portable ARM64 binary without native CPU tuning
ENV CMAKE_ARGS="-DGGML_NATIVE=0"
ENV UV_SYSTEM_PYTHON=1
RUN uv sync --no-dev --frozen --no-install-project

COPY src/ src/

ENV PYTHONPATH=/app/src

# Model is downloaded at startup — do not bake it into the image.
# Set one of:
#   MODEL_S3_KEY  — S3 object key to download at startup (requires AWS_S3_BUCKET + credentials)
#   MODEL_PATH    — local path for dev/testing with a volume-mounted model
# Or seed an active model in the database with a gguf_path pointing to S3.

RUN mkdir -p /app/data /app/models

EXPOSE 8000

CMD ["sh", "-c", "python -m interactors.cli.db.db_migrate && uvicorn interactors.api.app:app --host 0.0.0.0 --port 8000"]
