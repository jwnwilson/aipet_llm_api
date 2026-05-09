MODEL_PATH  ?= models/aipet.gguf
CHECKPOINT  ?= models/checkpoints
HOST        ?= 0.0.0.0
PORT        ?= 8000
DATA_DIR    ?= data
OUTPUT_DIR  ?= models/checkpoints
IMAGE       ?= aipet-llm
RPI_HOST    ?= raspberrypi.local

EXPERIMENT      ?= experiment-01
EPOCHS          ?= 5
PATIENCE        ?= 3
REMOTE_BACKEND  ?= kaggle
MODEL           ?= HuggingFaceTB/SmolLM2-1.7B

.PHONY: serve test test-unit test-integration test-cli data train evaluate evaluate-gguf export infer setup-llama docker-build docker-run docker-export docker-deploy temporal-up temporal-down temporal-worker temporal-trigger kaggle-train help

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

docker-build: ## Build the ARM64 Docker image  (IMAGE=... to override tag)
	docker buildx build --platform linux/arm64 -t $(IMAGE):latest --load .

docker-run: ## Run the image locally for smoke-testing  (MODEL_PATH/PORT to override)
	docker run --rm -p $(PORT):8000 \
		-v "$(PWD)/models:/app/models:ro" \
		-e MODEL_PATH=/app/models/aipet.gguf \
		$(IMAGE):latest

docker-export: ## Save the ARM64 image as a tarball for transfer to the RPi
	docker save $(IMAGE):latest | gzip > $(IMAGE).tar.gz
	@echo "Saved to $(IMAGE).tar.gz — transfer with: scp $(IMAGE).tar.gz pi@$(RPI_HOST):~/"

docker-deploy: docker-export ## Build, export, and copy the image to the RPi (RPI_HOST=... to override)
	scp $(IMAGE).tar.gz pi@$(RPI_HOST):~/
	ssh pi@$(RPI_HOST) "docker load -i ~/$(IMAGE).tar.gz && docker compose up -d"

temporal-up: ## Start Temporal server + web UI (localhost:8233)
	docker compose up temporal temporal-db temporal-ui -d

temporal-down: ## Stop Temporal server and worker
	docker compose down temporal temporal-ui temporal-db temporal-worker

temporal-worker: ## Run the Temporal activity worker locally  (requires Temporal server)
	KAGGLE_REPO_URL=$(KAGGLE_REPO_URL) PYTHONPATH=src uv run python -m temporal.worker

temporal-trigger: ## Trigger a training pipeline workflow  (EXPERIMENT / EPOCHS / PATIENCE / REMOTE_BACKEND / MODEL / SKIP_GENERATE=1)
	KAGGLE_REPO_URL=$(KAGGLE_REPO_URL) PYTHONPATH=src uv run python src/cli/trigger_training.py \
		--experiment-name $(EXPERIMENT) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--remote-backend $(REMOTE_BACKEND) \
		--model $(MODEL) \
		$(if $(SKIP_GENERATE),--skip-generate)

kaggle-train: ## Trigger a Kaggle GPU training run  (EXPERIMENT / EPOCHS / PATIENCE / MODEL)
	PYTHONPATH=src uv run python src/cli/trigger_training.py \
		--experiment-name $(EXPERIMENT) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--remote-backend kaggle \
		--model $(MODEL) \
		$(if $(SKIP_GENERATE),--skip-generate)

kaggle-notebook-local: ## Simulate full Kaggle notebook locally: stage dataset then run all cells
	@echo "--- Staging dataset: build wheel + copy data ---"
	rm -rf /tmp/kaggle-sim/input/$(EXPERIMENT)-data
	mkdir -p /tmp/kaggle-sim/input/$(EXPERIMENT)-data
	uv build --wheel --out-dir /tmp/kaggle-sim/input/$(EXPERIMENT)-data 2>&1 | tail -1
	cp $(DATA_DIR)/train.jsonl $(DATA_DIR)/eval.jsonl /tmp/kaggle-sim/input/$(EXPERIMENT)-data/
	@echo "--- Running all notebook cells locally ---"
	EXPERIMENT=$(EXPERIMENT) MODEL=$(MODEL) \
	KAGGLE_INPUT_BASE=/tmp/kaggle-sim/input/$(EXPERIMENT)-data \
		uv run python src/cli/run_notebook_local.py
	@echo "--- Local Kaggle notebook simulation passed ---"

request: ## Send a test /infer request to the running API server  (HOST/PORT to override)
	curl -s -X POST http://$(HOST):$(PORT)/infer \
		-H "Content-Type: application/json" \
		-d '{"scene":{"objects":[{"id":"bowl1","type":"bowl","distance":2.5},{"id":"toy1","type":"toy","distance":5.0},{"id":"bed1","type":"bed","distance":8.0}],"tick":42},"pet_stats":{"hunger":0.8,"boredom":0.3,"social":0.2,"toilet":0.1,"tiredness":0.2}}' \
		| python3 -m json.tool
