# Plan

> Completed work → [complete.md](complete.md)

---

## EPIC-7: Project Consolidation

> Rename the project to "llm-api" and make it a generic training platform usable beyond aipet.

### Feature 7.1 — Rename project to llm-api

#### TASK-7.1.1 — Remove aipet references
Rename all `aipet`-prefixed identifiers, strings, and config values to `llm-api` equivalents throughout the codebase. Update `pyproject.toml`, `docker-compose.yml`, k8s manifests, Terraform outputs, and any hardcoded strings.

**Outputs:** Updated `pyproject.toml`, `docker-compose.yml`, `infra/k8s/`, `infra/terraform/`, source files

#### TASK-7.1.2 — Integrate llm-ui into this repo
Add the llm-ui frontend as a sub-project (e.g. `ui/` directory) or as a git submodule. Wire up the UI build into the Docker image or serve it via a separate container alongside the API.

**Outputs:** `ui/` directory or submodule, updated `docker-compose.yml`, updated `Dockerfile` or new `ui/Dockerfile`

---

## EPIC-8: LLM Training Pipeline

> Improve reliability, observability, and user control over the training pipeline.

### Feature 8.1 — Error handling in workflows

#### TASK-8.1.1 — Update runs to "error" status with error message
When a Temporal activity raises an unhandled exception, catch it in the workflow and call `RunStore.update()` to set `status="error"` and populate `error_msg`. Ensure the error message is surfaced in the llm-ui run list.

**Outputs:** Updated `src/interactors/temporal/workflows.py`, updated `src/interactors/temporal/activities.py`

### Feature 8.2 — Run overrides flowing to the pipeline

#### TASK-8.2.1 — Investigate and fix run overrides not reaching the pipeline
Trace the override fields from the trigger CLI / API through the Temporal workflow input to each activity. Add unit and integration tests that assert overrides (e.g. `epochs`, `base_model`, `remote_backend`) arrive correctly at the training activity.

**Outputs:** Bug fix in `src/interactors/temporal/workflows.py` or `src/interactors/temporal/activities.py`, new tests in `tests/unit/` or `tests/integration/`

### Feature 8.3 — User-controlled training via UI

#### TASK-8.3.1 — Upload training and eval data via UI
Add API endpoints (`POST /api/datasets/train`, `POST /api/datasets/eval`) that accept JSONL file uploads and store them via `StoragePort`. Wire up a file-upload form in llm-ui.

**Outputs:** `src/interactors/api/routes/datasets.py`, updated `src/domain/ports.py`, UI upload component

#### TASK-8.3.2 — Select base model via UI
Add a model selector to the training trigger form in llm-ui, backed by `GET /api/models` (list of registered models). Pass the selected model ID as `base_model` in the workflow trigger payload.

**Outputs:** UI model selector component, updated trigger form

#### TASK-8.3.3 — Select training platform via UI
Add a platform dropdown to the training trigger form (Kaggle, RunPod, Vast.ai, SSH). Pass the selection as `remote_backend` in the workflow trigger payload.

**Outputs:** UI platform selector component, updated trigger form

### Feature 8.4 — Eval improvements

#### TASK-8.4.1 — Improve eval metrics and expose via API
Extend `evaluate_activity` to produce richer per-action metrics (precision, recall, confusion matrix). Persist eval results to the DB and expose them via `GET /api/runs/{run_id}/eval`.

**Outputs:** Updated `src/domain/train/evaluate.py`, updated `src/interactors/temporal/activities.py`, new route in `src/interactors/api/routes/runs.py`

#### TASK-8.4.2 — Display eval results in llm-ui
Add an eval results panel to the run detail page in llm-ui showing per-action accuracy and the overall pass/fail gate result.

**Outputs:** UI eval results component

---

## EPIC-9: Better LLM API Architecture

> Decouple model loading from startup, support multiple simultaneously active models, and add per-model auto-scaling.

### Feature 9.1 — Decouple model loading from API startup

#### TASK-9.1.1 — Load model lazily on first inference request
Remove blocking model load from the FastAPI lifespan. Instead, load the model on the first call to `POST /infer` and return HTTP 503 (`Retry-After`) while loading is in progress. API `/health` must return 200 immediately.

**Outputs:** Updated `src/interactors/api/app.py`, updated `src/adapters/inference.py`

### Feature 9.2 — Per-model container architecture

#### TASK-9.2.1 — Spin up a dedicated container per active model
Design a controller (k8s operator or simple reconciler) that, when a model is activated, creates a k8s `Deployment` for that model requesting appropriate memory based on GGUF size. The main API proxies inference requests to the correct model pod.

**Outputs:** `infra/k8s/model-deployment-template.yaml`, controller logic in `src/interactors/` or as a separate service

#### TASK-9.2.2 — Scale to zero after 1 hour of inactivity
Configure k8s HPA (or KEDA) to scale each model `Deployment` to zero replicas when it receives no requests for 1 hour. The main API returns HTTP 503 while the pod scales back up.

**Outputs:** `infra/k8s/model-hpa.yaml` or KEDA `ScaledObject`, updated proxy logic

### Feature 9.3 — Model status tracking

#### TASK-9.3.1 — Track and expose active model status
Add a `status` field to the model record (`loading`, `ready`, `scaling_down`, `offline`). Update status from the controller/reconciler and expose it via `GET /api/models` and `GET /api/models/{model_id}`.

**Outputs:** Updated `src/domain/models.py`, updated `src/adapters/database/model_store.py`, updated routes

#### TASK-9.3.2 — Display model status in llm-ui
Show model status badges on the model list page (ready / loading / offline) with auto-refresh polling.

**Outputs:** UI model status component

### Feature 9.4 — Request handling for loading models

#### TASK-9.4.1 — Return well-formed response when model is not ready
When a request arrives for a model that is loading or scaled to zero, return HTTP 503 with `{"status": "not_ready_yet", "retry_after": <seconds>}` so clients can back off gracefully.

**Outputs:** Updated proxy/routing logic, updated API docs

---

## EPIC-10: LLM API — Inference & API Keys

> Expose per-model inference via the API with per-user API keys and rate limiting.

### Feature 10.1 — Per-model inference endpoint

#### TASK-10.1.1 — `POST /api/models/{model_id}/infer` endpoint
Add an inference endpoint that routes requests to the model's pod (or adapter) without changing the active model. Show this endpoint in the llm-ui API tab for each model.

**Outputs:** New route in `src/interactors/api/routes/models.py`, UI API tab component

### Feature 10.2 — Per-user API keys

#### TASK-10.2.1 — Issue API keys per Auth0 user
Add `POST /api/keys` (create key) and `GET /api/keys` (list user's keys) endpoints. Store hashed keys in the DB linked to the Auth0 user ID. Keys are presented once on creation.

**Outputs:** `src/interactors/api/routes/keys.py`, `src/adapters/database/key_store.py`, DB migration

#### TASK-10.2.2 — Accept API key as `Authorization: Bearer` on inference endpoints
Allow inference endpoints to authenticate via either a JWT (Auth0) or a raw API key. Add key lookup to the `require_auth` dependency path.

**Outputs:** Updated `src/interactors/api/auth.py`

### Feature 10.3 — Rate limiting

#### TASK-10.3.1 — Rate limit inference requests per user
Add rate limiting middleware (e.g. `slowapi`) to cap inference requests per user per minute. Return HTTP 429 with `Retry-After` when the limit is exceeded.

**Outputs:** Updated `src/interactors/api/app.py`, new rate limit config

---

## EPIC-11: Fast E2E Tests

> Re-enable the E2E test suite on CI/CD without slowing down every PR.

### Feature 11.1 — Scheduled E2E test run

#### TASK-11.1.1 — Add scheduled E2E workflow
Add `.github/workflows/e2e.yml` triggered on `schedule: cron` (e.g. once daily at 02:00 UTC) and on `workflow_dispatch`. Run `pytest tests/e2e/` against the deployed environment with appropriate secrets.

**Outputs:** `.github/workflows/e2e.yml`

#### TASK-11.1.2 — Fix or skip currently broken E2E tests
Audit `tests/e2e/` and either fix broken tests or mark them `@pytest.mark.skip(reason="...")` with a tracking note. Ensure the suite passes cleanly in the scheduled run.

**Outputs:** Updated `tests/e2e/` files
