# Completed Work

> Archived from plan.md. All items below have corresponding implementation files.

---

## Phase 1: Foundation

### Task 1.1 — Project structure & dependencies
Hexagonal directory layout established; all packages installed via `uv`.
**Outputs:** `pyproject.toml`, `src/` tree with `__init__.py` files.

### Task 1.2 — Pydantic schemas
All input/output data contracts defined.
**Outputs:** `src/domain/models.py`, `src/domain/actions.py`

### Task 1.3 — Domain ports (interfaces)
Abstract `InferencePort` interface defined with unit tests.
**Outputs:** `src/domain/ports.py`, `tests/unit/test_ports.py`

---

## Phase 2: Core Implementation

### Task 2.1 — Inference adapter (llama.cpp)
`LlamaCppInferenceAdapter` implemented; parse failures return `Action.IDLE`.
**Outputs:** `src/infrastructure/inference.py`

### Task 2.2 — Prompt template & response parser
`build_prompt()` and `parse_response()` implemented; prompt stays under 300 tokens.
**Outputs:** `src/infrastructure/prompt.py`

### Task 2.3 — Synthetic training dataset generator
5 000 train / 500 eval examples generated in JSONL format.
**Outputs:** `src/domain/train/dataset.py`, `src/cli/generate_dataset.py`, `data/train.jsonl`, `data/eval.jsonl`

---

## Phase 3: API Layer

### Task 3.1 — FastAPI application
`POST /infer` and `GET /health` endpoints wired with DI for `InferencePort`.
**Outputs:** `src/api/app.py`, `src/api/routes.py`

### Task 3.2 — API integration tests
Full request/response cycle verified with stub adapter.
**Outputs:** `tests/integration/test_api.py`

---

## Phase 4: Training Pipeline

### Task 4.1 — Fine-tuning script
HuggingFace `Trainer` fine-tune on prompt+completion pairs; supports `--dry-run`.
**Outputs:** `src/domain/train/trainer.py`, `src/cli/train.py`

### Task 4.2 — Evaluation & export script
Schema-valid response rate measured; GGUF export via `llama.cpp` converter.
**Outputs:** `src/domain/train/evaluate.py`, `src/cli/evaluate.py`, `src/domain/train/export.py`, `src/cli/export.py`

---

## Phase 5: Deployment

### Task 5.1 — Docker deployment config
Multi-arch ARM64 `Dockerfile` and `docker-compose.yml` for RPi 5.
**Outputs:** `Dockerfile`, `docker-compose.yml`

---

## Phase 6: Model Quality Improvements

> Root-cause fixes for EAT/SLEEP bias and wrong target-object selection.

### Task 6.1 — Statistical quality test suite
Per-stat accuracy report and action-distribution histogram; integration tests gating CI.
**Outputs:** `src/domain/train/quality_report.py`, `tests/integration/test_model_quality.py`

### Task 6.2 — Dataset regeneration
Stratified sampling, tick-parity fix, richer multi-target scenes, 5k/500 dataset.
**Outputs:** Updated `src/domain/train/dataset.py`, regenerated `data/train.jsonl`, `data/eval.jsonl`

### Task 6.3 — Prompt engineering improvements
Stats sorted high→low with `(highest)` label; explicit decision rule; objects sorted by distance.
**Outputs:** Updated `src/infrastructure/prompt.py`, updated `tests/unit/test_prompt.py`

### Task 6.4 — Training improvements
Weighted sampler, cosine LR schedule with warmup, early stopping (`--patience`), per-action eval logging, `--base-model` arg (default SmolLM2-1.7B).
**Outputs:** Updated `src/domain/train/trainer.py`, updated `src/cli/train.py`

---

## Post V1 — Completed

### P.2 — Kubernetes deployment
`Deployment`, `Service`, and `HPA` manifests for multi-node cluster.
**Outputs:** `infra/k8s/deployment.yaml`, `infra/k8s/service.yaml`, `infra/k8s/hpa.yaml`

### P.3 — Temporal training pipeline
Full training lifecycle orchestrated as a Temporal workflow with retry semantics.
**Outputs:** `src/temporal/workflows.py`, `src/temporal/activities.py`, `src/temporal/worker.py`, `src/cli/trigger_training.py`

### P.4 — Remote GPU training (Kaggle / SSH)
`KaggleTrainingAdapter` and `SshTrainingAdapter` implement `RemoteTrainingPort`; `train_activity` routes via `--remote-backend`.
**Outputs:** `src/adapters/kaggle_adapter.py`, `src/adapters/ssh_adapter.py`, `src/adapters/notebook_template.ipynb`

### P.5 — ECR Terraform provisioning
ECR repository, lifecycle policy, GitHub OIDC IAM role, and push policy provisioned via Terraform.
**Outputs:** `infra/terraform/main.tf`, `infra/terraform/github_actions.tf`, `infra/terraform/variables.tf`, `infra/terraform/outputs.tf`, `infra/terraform/versions.tf`
