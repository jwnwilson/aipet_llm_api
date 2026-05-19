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

---

## EPIC-1: Kaggle Training Pipeline (Operational)

> End-to-end Kaggle GPU training pipeline is running. Validation of model quality (Feature 1.5) remains pending in plan.md.

### Feature 1.1 — Kaggle credentials
`~/.kaggle/kaggle.json` provisioned; `KAGGLE_USERNAME` and `KAGGLE_KEY` set in shell profile.

### Feature 1.2 — Temporal server (local)
`docker-compose.yml` runs `temporal`, `temporal-db`, and `temporal-ui`; Temporal UI at http://localhost:8233.

### Feature 1.3 — Temporal worker (local)
Worker runs outside Docker via `uv run python -m src.temporal.worker`; handles task queue `aipet-training`.

### Feature 1.4 — Dataset generation and training trigger
- `src/cli/generate_dataset.py` produces 5 000 train / 500 eval examples.
- `src/cli/trigger_training.py` submits the Temporal workflow with `--remote-backend kaggle`.
- `evaluate_activity` and `export_activity` wired end-to-end; GGUF written to `models/aipet.gguf` only when eval ≥ 95%.
- Async API endpoints added for workflow triggering: `POST /workflows/training`, `POST /workflows/evaluate`, `POST /workflows/export`.
- Alembic migrations in place for workflow run tracking.

---

## EPIC-3: CI/CD Automation

### Feature 3.1 — GitHub Actions deploy pipeline

#### TASK-3.1.1 — `.github/workflows/deploy.yml`
Triggers on successful `Test` workflow run against `main`; OIDC via `secrets.AWS_ROLE_ARN`; builds linux/arm64 image with GHA layer cache; tags `:<sha>` and `:latest`; applies k8s manifests and waits for rollout with `--timeout=600s`.
**Outputs:** `.github/workflows/deploy.yml`

#### TASK-3.1.2 — GitHub secrets seeded
`AWS_ROLE_ARN` and `KUBECONFIG` (and additional secrets for DB, ECR, Auth0, Kaggle, RunPod, Vast) set via `gh secret set` after `terraform apply`.

#### TASK-3.1.3 — k8s deployment uses ECR URL
Deploy pipeline does `sed -i "s|<ECR_REPOSITORY_URL>:latest|$IMAGE|g"` at deploy time; static manifests keep the placeholder intentionally.
**Outputs:** `infra/k8s/aipet-llm/deployment.yaml`, `infra/k8s/temporal/worker.yaml`

#### TASK-3.1.4 — Terraform state files in `.gitignore`
`.gitignore` entries: `infra/terraform/**/.terraform/`, `infra/terraform/*.tfstate*`, `infra/terraform/**/.terraform.lock.hcl`.

---

## EPIC-4: Production Hardening

### Feature 4.1 — Early stopping verification

#### TASK-4.1.1 — `--patience` smoke-test
`--patience` flag is implemented in `src/interactors/cli/training/train.py`; `EarlyStoppingCallback` wired in trainer. Verified via `uv run python -m src.cli.train --dry-run --patience 1`.

#### TASK-4.1.2 — Training flags documented in `README.md`
`--patience`, `--warmup-ratio`, `--base-model`, and `--remote-backend` documented with example invocations. Auth0 and CORS env vars also documented.
**Outputs:** Updated `README.md`

---

## EPIC-5: Auto Deployment & Model Availability

> Implemented with **AWS S3** instead of GCP GCS. All functionality is equivalent.

### Feature 5.1 — Cloud Storage Adapter (AWS S3)
`S3StorageAdapter` in `src/adapters/storage/s3.py` implements `StoragePort`; uploads/downloads GGUF artifacts keyed by run ID. Config via `AWS_S3_BUCKET` and standard boto3 credential chain.
**Outputs:** `src/adapters/storage/s3.py`, `tests/unit/test_s3_storage.py`

### Feature 5.2 — Upload wired into `export_activity`
After GGUF is written, `export_activity` calls `upload_model()` when `AWS_S3_BUCKET` is set; S3 key logged so the Temporal UI shows the artifact location.
**Outputs:** Updated `src/interactors/temporal/activities.py`

### Feature 5.3 — Model management API endpoints
- `GET /api/models` — list all registered models
- `GET /api/models/{model_id}` — get model by ID
- `POST /api/models/{model_id}/activate` — download GGUF from S3, hot-swap the inference adapter, mark as active
- `POST /api/models` — register a new model record
**Outputs:** `src/interactors/api/routes/models.py`, `tests/integration/test_model_workflow_integration.py`

### Feature 5.4 — Hot-swap support in `LlamaCppInferenceAdapter`
`release()` method unloads the current model from RAM; `activate_model` route acquires new GGUF from S3, calls `release()` on the old adapter, and loads the new one. No explicit lock needed — FastAPI handles request concurrency.
**Outputs:** Updated `src/adapters/inference.py`

---

## EPIC-6: Authentication for Public Access

> Implemented with **Auth0 JWT authentication** instead of static API keys. Provides stronger security and user identity without managing key distribution.

### Feature 6.1 — Auth0 JWT middleware
`Auth0Adapter` in `src/adapters/auth/auth0.py` validates JWTs against the Auth0 JWKS endpoint. `FakeAuthAdapter` in `src/adapters/auth/fake.py` used in local dev (`APP_ENV=development`). `require_auth`, `get_current_user`, `require_approved`, `require_admin` dependencies in `src/interactors/api/auth.py`.
**Outputs:** `src/adapters/auth/auth0.py`, `src/adapters/auth/fake.py`, `src/interactors/api/auth.py`

### Feature 6.2 — Auth applied to all routers
All routers use `require_approved` or `require_admin` as router-level dependency; `GET /health` remains unauthenticated.
**Outputs:** Updated `src/interactors/api/app.py` and all route files

### Feature 6.3 — CORS configured
`CORSMiddleware` reads `CORS_ORIGINS` env var; defaults to `[]` in production, `[localhost:*]` when `APP_ENV=development`.
**Outputs:** Updated `src/interactors/api/app.py`

### Feature 6.4 — Auth integration tests
Full request cycle tested: unauthenticated → 401, invalid token → 401, valid token → 200, `GET /health` → 200 without token.
**Outputs:** `tests/integration/test_auth.py`
