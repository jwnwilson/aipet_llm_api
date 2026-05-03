MODEL_PATH  ?= models/aipet.gguf
CHECKPOINT  ?= models/checkpoints
HOST        ?= 0.0.0.0
PORT        ?= 8000
DATA_DIR    ?= data
OUTPUT_DIR  ?= models/checkpoints

.PHONY: serve test test-unit test-integration test-cli data train evaluate evaluate-gguf export infer setup-llama help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

serve: ## Start the FastAPI server  (MODEL_PATH=... make serve)
	MODEL_PATH=$(MODEL_PATH) PYTHONPATH=src uv run uvicorn api.app:app \
		--host $(HOST) --port $(PORT) --reload

test: ## Run all tests
	uv run pytest tests/ -v

test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	uv run pytest tests/integration/ -v

test-cli: ## Run CLI tests only
	uv run pytest tests/cli/ -v

data: ## Generate synthetic training + eval data  (DATA_DIR=... to override output path)
	PYTHONPATH=src uv run python src/cli/generate_dataset.py --data-dir $(DATA_DIR)

train: ## Fine-tune the model  (DRY_RUN=1 for smoke test, DATA_DIR/OUTPUT_DIR to override paths)
	PYTHONPATH=src uv run python src/cli/train.py \
		$(if $(DRY_RUN),--dry-run) \
		--train-data $(DATA_DIR)/train.jsonl \
		--eval-data $(DATA_DIR)/eval.jsonl \
		--output-dir $(OUTPUT_DIR)

evaluate: ## Evaluate HF checkpoint response rate  (CHECKPOINT=... to override)
	PYTHONPATH=src uv run python src/cli/evaluate.py \
		--checkpoint $(CHECKPOINT) --eval-data $(DATA_DIR)/eval.jsonl

evaluate-gguf: ## Evaluate quantised GGUF model  (MODEL_PATH=... to override)
	PYTHONPATH=src uv run python src/cli/evaluate.py \
		--model-path $(MODEL_PATH) --eval-data $(DATA_DIR)/eval.jsonl

setup-llama: ## Clone and build llama.cpp (required for make export)
	@if [ -d llama.cpp ]; then \
		echo "llama.cpp already exists, skipping clone."; \
	else \
		git clone https://github.com/ggerganov/llama.cpp.git llama.cpp; \
	fi
	cmake -B llama.cpp/build llama.cpp
	cmake --build llama.cpp/build --config Release -j
	@echo "\nllama.cpp ready — run 'make export' to convert your checkpoint."

export: ## Convert HF checkpoint → GGUF Q4_K_M  → models/aipet.gguf
	PYTHONPATH=src uv run python src/cli/export.py

infer: ## Run a single inference from the CLI  (MODEL_PATH=... make infer)
	PYTHONPATH=src uv run python src/cli/infer.py --model-path $(MODEL_PATH) < $(or $(INPUT),/dev/stdin)
