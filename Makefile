MODEL_PATH ?= models/aipet.gguf
HOST      ?= 0.0.0.0
PORT      ?= 8000

.PHONY: serve test test-unit test-integration data train evaluate export infer help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

serve: ## Start the FastAPI server  (MODEL_PATH=... make serve)
	MODEL_PATH=$(MODEL_PATH) uv run uvicorn src.api.app:app \
		--host $(HOST) --port $(PORT) --reload

test: ## Run all tests
	uv run pytest tests/ -v

test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	uv run pytest tests/integration/ -v

data: ## Generate synthetic training + eval data  → data/train.jsonl, data/eval.jsonl
	uv run python scripts/generate_dataset.py

train: ## Fine-tune the model on data/train.jsonl  (add DRY_RUN=1 for a smoke test)
	uv run python scripts/train.py $(if $(DRY_RUN),--dry-run)

evaluate: ## Evaluate schema-valid response rate (must pass ≥ 95%)
	uv run python scripts/evaluate.py \
		--model-path $(MODEL_PATH) --eval-data data/eval.jsonl

export: ## Convert HF checkpoint → GGUF Q4_K_M  → models/aipet.gguf
	uv run python scripts/export.py

infer: ## Run a single inference from the CLI  (MODEL_PATH=... make infer)
	uv run python -c "\
import json, sys; \
from src.infrastructure.inference import LlamaCppInferenceAdapter; \
from src.domain.models import InferenceRequest; \
req = InferenceRequest.model_validate(json.load(sys.stdin)); \
adapter = LlamaCppInferenceAdapter(model_path='$(MODEL_PATH)'); \
print(adapter.infer(req).model_dump_json(indent=2))" < $(or $(INPUT),/dev/stdin)
