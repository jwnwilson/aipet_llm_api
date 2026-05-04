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
COPY models/aipet.gguf models/aipet.gguf

ENV MODEL_PATH=/app/models/aipet.gguf
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
