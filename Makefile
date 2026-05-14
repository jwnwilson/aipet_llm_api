MODEL_PATH  ?= models/aipet.gguf
CHECKPOINT  ?= models/checkpoints
HOST        ?= 0.0.0.0
PORT        ?= 8000
DATA_DIR    ?= data
OUTPUT_DIR  ?= models/checkpoints
IMAGE       ?= aipet-llm
RPI_HOST    ?= raspberrypi.local

EXPERIMENT      ?= aipet-v3
EPOCHS          ?= 5
PATIENCE        ?= 3
REMOTE_BACKEND  ?= kaggle
REMOTE_RUN_ID   ?=
MODEL           ?= HuggingFaceTB/SmolLM2-1.7B
# MODEL           ?= HuggingFaceTB/SmolLM2-360M
FAST_MODEL      ?= HuggingFaceTB/SmolLM2-135M
FAST_DATA_DIR   ?= data/fast
GITHUB_REPO     ?= jwnwilson/aipet_llm_api
TF_DIR          ?= infra/terraform

.PHONY: serve sync test test-unit test-integration test-cli test-all data data-fast train train-fast evaluate evaluate-gguf evaluate-remote export export-remote evaluate-export-remote infer setup-llama docker-build docker-run docker-export docker-deploy temporal-up temporal-down temporal-worker temporal-trigger temporal-trigger-fast kaggle-train runpod-train vastai-train db-migrate db-revision seed-models tf-init tf-plan tf-apply tf-deploy aws-env help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

.venv:
	uv sync --extra dev --reinstall-package kaggle

sync: ## Install / sync all dependencies including dev groups
	uv sync --extra dev --reinstall-package kaggle

serve: .venv ## Start the FastAPI server  (MODEL_PATH=... make serve)
	MODEL_PATH=$(MODEL_PATH) PYTHONPATH=src uv run --env-file .env python -m uvicorn interactors.api.app:app \
		--host $(HOST) --port $(PORT) --reload

test: .venv ## Run unit + integration tests (fast; excludes e2e)
	uv run python -m pytest tests/unit/ tests/integration/ -v

test-unit: .venv ## Run unit tests only
	uv run python -m pytest tests/unit/ -v

test-integration: .venv ## Run integration tests only (requires models/aipet.gguf)
	uv run python -m pytest tests/integration/ -v

test-e2e: .venv ## Run end-to-end tests only (tests training on platforms with GPU access; requires AWS_S3_BUCKET + VAST_API_KEY or RUNPOD_API_KEY)
	uv run python -m pytest tests/e2e/ -v

test-all: .venv ## Run all tests including slow integration tests
	uv run python -m pytest tests/ -v

data: ## Generate synthetic training + eval data  (DATA_DIR=... to override output path)
	PYTHONPATH=src uv run python src/interactors/cli/generate_dataset.py --data-dir $(DATA_DIR)

data-fast: ## Generate tiny dataset (20 train / 10 eval) into data/fast/
	PYTHONPATH=src uv run python src/interactors/cli/generate_dataset.py \
		--data-dir $(FAST_DATA_DIR) --train-size 20 --eval-size 10

train-fast: data-fast ## Smoke-test: tiny model + 20-example dataset + 1 training step  (FAST_MODEL=... to override)
	PYTHONPATH=src uv run python src/interactors/cli/train.py \
		--dry-run \
		--model $(FAST_MODEL) \
		--train-data $(FAST_DATA_DIR)/train.jsonl \
		--eval-data $(FAST_DATA_DIR)/eval.jsonl \
		--output-dir models/checkpoints-test

train: ## Fine-tune the model  (DRY_RUN=1 for smoke test, DATA_DIR/OUTPUT_DIR to override paths)
	PYTHONPATH=src uv run python src/interactors/cli/train.py \
		$(if $(DRY_RUN),--dry-run) \
		--train-data $(DATA_DIR)/train.jsonl \
		--eval-data $(DATA_DIR)/eval.jsonl \
		--output-dir $(OUTPUT_DIR) \
		--model $(MODEL)

evaluate: ## Evaluate HF checkpoint response rate  (CHECKPOINT=... to override)
	PYTHONPATH=src uv run python src/interactors/cli/evaluate.py \
		--checkpoint $(CHECKPOINT) --eval-data $(DATA_DIR)/eval.jsonl

evaluate-gguf: ## Evaluate quantised GGUF model  (MODEL_PATH=... to override)
	PYTHONPATH=src uv run python src/interactors/cli/evaluate.py \
		--model-path $(MODEL_PATH) --eval-data $(DATA_DIR)/eval.jsonl

setup-llama: ## Clone and build llama.cpp (required for make export)
	@if [ -d llama.cpp ]; then \
		echo "llama.cpp already exists, skipping clone."; \
	else \
		git clone https://github.com/ggerganov/llama.cpp.git llama.cpp; \
	fi
	cmake -B llama.cpp/build llama.cpp
	cmake --build llama.cpp/build --target llama-quantize --config Release -j$$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
	@echo "\nllama.cpp ready — run 'make export' to convert your checkpoint."

export: ## Convert HF checkpoint → GGUF Q4_K_M  → models/aipet.gguf
	PYTHONPATH=src uv run python src/interactors/cli/export.py

evaluate-remote: ## Download checkpoint from remote and evaluate  (REMOTE_BACKEND / REMOTE_RUN_ID)
	REMOTE_BACKEND=$(REMOTE_BACKEND) REMOTE_RUN_ID=$(REMOTE_RUN_ID) \
	PYTHONPATH=src uv run python src/interactors/cli/evaluate.py \
		--eval-data $(DATA_DIR)/eval.jsonl

export-remote: ## Download checkpoint from remote and export to GGUF  (REMOTE_BACKEND / REMOTE_RUN_ID)
	REMOTE_BACKEND=$(REMOTE_BACKEND) REMOTE_RUN_ID=$(REMOTE_RUN_ID) \
	PYTHONPATH=src uv run python src/interactors/cli/export.py \
		--output $(MODEL_PATH)

evaluate-export-remote: ## Download, evaluate, then export in sequence  (REMOTE_BACKEND / REMOTE_RUN_ID)
	REMOTE_BACKEND=$(REMOTE_BACKEND) REMOTE_RUN_ID=$(REMOTE_RUN_ID) \
	PYTHONPATH=src uv run python src/interactors/cli/evaluate.py \
		--eval-data $(DATA_DIR)/eval.jsonl && \
	REMOTE_BACKEND=$(REMOTE_BACKEND) REMOTE_RUN_ID=$(REMOTE_RUN_ID) \
	PYTHONPATH=src uv run python src/interactors/cli/export.py \
		--output $(MODEL_PATH)

infer: ## Run a single inference from the CLI  (MODEL_PATH=... make infer)
	PYTHONPATH=src uv run python src/interactors/cli/infer.py --model-path $(MODEL_PATH) < $(or $(INPUT),/dev/stdin)

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
	KAGGLE_REPO_URL=$(KAGGLE_REPO_URL) PYTHONPATH=src uv run python -m interactors.temporal.worker

temporal-trigger-fast: ## Trigger a fast smoke-test pipeline via Temporal  (tiny model + 20 examples + 1 step, local backend)
	KAGGLE_REPO_URL=$(KAGGLE_REPO_URL) PYTHONPATH=src uv run python src/interactors/cli/trigger_training.py \
		--experiment-name aipet-fast-test \
		--model $(FAST_MODEL) \
		--train-size 20 \
		--eval-size 10 \
		--dry-run \
		--remote-backend kaggle

temporal-trigger: ## Trigger a training pipeline workflow  (EXPERIMENT / EPOCHS / PATIENCE / REMOTE_BACKEND / MODEL / SKIP_GENERATE=1)
	KAGGLE_REPO_URL=$(KAGGLE_REPO_URL) PYTHONPATH=src uv run python src/interactors/cli/trigger_training.py \
		--experiment-name $(EXPERIMENT) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--remote-backend $(REMOTE_BACKEND) \
		--model $(MODEL) \
		$(if $(SKIP_GENERATE),--skip-generate)

kaggle-train: ## Trigger a Kaggle GPU training run  (EXPERIMENT / EPOCHS / PATIENCE / MODEL)
	PYTHONPATH=src uv run python src/interactors/cli/trigger_training.py \
		--experiment-name $(EXPERIMENT) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--remote-backend kaggle \
		--model $(MODEL) \
		$(if $(SKIP_GENERATE),--skip-generate)

vastai-train: ## Trigger a Vast.ai GPU training run  (EXPERIMENT / EPOCHS / PATIENCE / MODEL; requires AWS_S3_BUCKET + VAST_API_KEY)
	PYTHONPATH=src uv run python src/interactors/cli/trigger_training.py \
		--experiment-name $(EXPERIMENT) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--remote-backend vastai \
		--model $(MODEL) \
		$(if $(SKIP_GENERATE),--skip-generate)

runpod-train: ## Trigger a RunPod GPU training run  (EXPERIMENT / EPOCHS / PATIENCE / MODEL; requires AWS_S3_BUCKET + RUNPOD_API_KEY)
	PYTHONPATH=src uv run python src/interactors/cli/trigger_training.py \
		--experiment-name $(EXPERIMENT) \
		--epochs $(EPOCHS) \
		--patience $(PATIENCE) \
		--remote-backend runpod \
		--model $(MODEL) \
		$(if $(SKIP_GENERATE),--skip-generate)

# Note: Colab training requires manual setup of a Colab notebook with the same codebase, plus Google OAuth credentials for Drive access. The `google-auth` target can be used to perform the one-time OAuth login and token caching on your local machine, which you can then copy to the Colab environment.
# Out of scope for now
# colab-train: ## Trigger a Google Colab training run  (EXPERIMENT / EPOCHS / PATIENCE / MODEL)
# 	PYTHONPATH=src uv run python src/interactors/cli/trigger_training.py \
# 		--experiment-name $(EXPERIMENT) \
# 		--epochs $(EPOCHS) \
# 		--patience $(PATIENCE) \
# 		--remote-backend colab \
# 		--model $(MODEL) \
# 		$(if $(SKIP_GENERATE),--skip-generate)

# Out of scope for now
# google-auth: ## One-time Google OAuth login for Colab Drive access  (GOOGLE_OAUTH_CLIENT_SECRETS=path/to/client_secrets.json)
# 	PYTHONPATH=src uv run python src/interactors/cli/setup_google_auth.py \
# 		$(if $(GOOGLE_OAUTH_CLIENT_SECRETS),--client-secrets $(GOOGLE_OAUTH_CLIENT_SECRETS))

# Out of scope for now
# colab-train-fast: ## Trigger a fast smoke-test Colab run  (tiny model + 20 examples + 1 step)
# 	PYTHONPATH=src uv run python src/interactors/cli/trigger_training.py \
# 		--experiment-name aipet-colab-fast \
# 		--model $(FAST_MODEL) \
# 		--train-size 20 \
# 		--eval-size 10 \
# 		--dry-run \
# 		--remote-backend colab

kaggle-notebook-local: ## Simulate full Kaggle notebook locally: stage dataset then run all cells
	@echo "--- Staging dataset: build wheel + copy data ---"
	rm -rf /tmp/kaggle-sim/input/$(EXPERIMENT)-data
	mkdir -p /tmp/kaggle-sim/input/$(EXPERIMENT)-data
	uv build --wheel --out-dir /tmp/kaggle-sim/input/$(EXPERIMENT)-data 2>&1 | tail -1
	cp $(DATA_DIR)/train.jsonl $(DATA_DIR)/eval.jsonl /tmp/kaggle-sim/input/$(EXPERIMENT)-data/
	@echo "--- Running all notebook cells locally ---"
	EXPERIMENT=$(EXPERIMENT) MODEL=$(MODEL) \
	KAGGLE_INPUT_BASE=/tmp/kaggle-sim/input/$(EXPERIMENT)-data \
		uv run python src/interactors/cli/run_notebook_local.py
	@echo "--- Local Kaggle notebook simulation passed ---"

db-migrate: .venv ## Apply all pending Alembic migrations to data/aipet.db (auto-stamps pre-Alembic DBs)
	PYTHONPATH=src uv run python src/interactors/cli/db_migrate.py

db-revision: .venv ## Generate a new Alembic migration  (MSG="describe the change")
	PYTHONPATH=src uv run alembic revision --autogenerate -m "$(MSG)"

seed-models: ## Seed the database with default training model configurations
	PYTHONPATH=src uv run python -m interactors.cli.seed_models

aws-env: ## Refresh AWS credentials in .env from the current AWS profile
	uv run scripts/update_aws_env.py

tf-init: ## Initialise Terraform working directory
	set -a && . ./.env && set +a && terraform -chdir=$(TF_DIR) init

tf-plan: ## Preview infrastructure changes  (GITHUB_REPO=owner/repo to override)
	set -a && . ./.env && set +a && terraform -chdir=$(TF_DIR) plan -var="github_repo=$(GITHUB_REPO)"

tf-apply: ## Apply infrastructure changes  (GITHUB_REPO=owner/repo to override)
	set -a && . ./.env && set +a && terraform -chdir=$(TF_DIR) apply -var="github_repo=$(GITHUB_REPO)"

tf-deploy: tf-apply ## Apply infra then set AWS_ROLE_ARN secret on GitHub  (requires gh CLI)
	set -a && . ./.env && set +a && gh secret set AWS_ROLE_ARN \
		--repo $(GITHUB_REPO) \
		--body "$$(terraform -chdir=$(TF_DIR) output -raw github_actions_role_arn)"

request: ## Send a test /infer request to the running API server  (HOST/PORT to override)
	curl -s -X POST http://$(HOST):$(PORT)/infer \
		-H "Content-Type: application/json" \
		-d '{"scene":{"objects":[{"id":"bowl1","type":"bowl","distance":2.5},{"id":"toy1","type":"toy","distance":5.0},{"id":"bed1","type":"bed","distance":8.0}],"tick":42},"pet_stats":{"hunger":0.8,"boredom":0.3,"social":0.2,"toilet":0.1,"tiredness":0.2}}' \
		| python3 -m json.tool
