# Plan

> Completed work → [complete.md](complete.md)

---

## EPIC-3: CI/CD Automation

> Automate build and deploy so every push to `main` ships a new image.

**Prerequisites:** AWS account with ECR + OIDC role provisioned via `infra/terraform/` (done).

---

### Feature 3.1 — GitHub Actions deploy pipeline

#### TASK-3.1.1 — Write `.github/workflows/deploy.yml`
- Trigger: `push` to `main`; permissions `id-token: write`, `contents: read`
- Steps: checkout → `configure-aws-credentials` (OIDC via `secrets.AWS_ROLE_ARN`, no static keys) → `amazon-ecr-login` → `docker/build-push-action` (linux/arm64, tags `:<sha>` and `:latest`, GHA layer cache) → `kubectl set image` + `kubectl rollout status --timeout=300s`
- Read kubeconfig from `secrets.KUBECONFIG` (base64-encoded)

#### TASK-3.1.2 — First-time GitHub secrets setup
```bash
cd infra/terraform
terraform init && terraform apply -var="github_repo=<owner>/aipet-llm"

gh secret set AWS_ROLE_ARN --body "$(terraform output -raw github_actions_role_arn)"
gh secret set KUBECONFIG   --body "$(cat ~/.kube/config | base64)"
```

#### TASK-3.1.3 — Update k8s deployment with real ECR URL
```bash
REPO=$(terraform output -raw repository_url)
sed -i "s|<ECR_REPOSITORY_URL>|$REPO|g" ../k8s/deployment.yaml
kubectl apply -f ../k8s/
```

#### TASK-3.1.4 — Add Terraform state files to `.gitignore`
```
infra/terraform/.terraform/
infra/terraform/*.tfstate*
infra/terraform/.terraform.lock.hcl
```

---

## EPIC-4: Production Hardening

### Feature 4.1 — Early stopping verification

> `--patience` was added in Task 6.4. Verify it fires correctly and is documented.

#### TASK-4.1.1 — Smoke-test
```bash
uv run python -m src.cli.train --dry-run --patience 1
# Confirm EarlyStoppingCallback log line appears in output
```

#### TASK-4.1.2 — Document training flags in `README.md`
Cover `--patience`, `--warmup-ratio`, `--base-model`, `--remote-backend` with example invocations.

---

## EPIC-5: Auto Deployment & Model Availability

> When a model passes the ≥ 95% eval gate it should be automatically saved to cloud storage, registered in a model registry, and made testable via the API — without manual steps.

**Goals (from TODO):**
- Save successful models to cloud storage (GCP GCS)
- Register model metadata (eval score, run ID, base model, timestamp)
- Let anyone hit an API to list, inspect, and test-infer against any registered model
- Support activating a model (hot-swap the running inference adapter) via the API

**Prerequisites:** EPIC-1 validation complete (a ≥ 95% GGUF exists), GCP project with a GCS bucket.

---

### Feature 5.1 — Cloud Storage Adapter (GCP GCS)

**Fill in this section with:** bucket name/path scheme, auth approach (service account vs. ADC), and whether checkpoints as well as GGUFs should be stored.

#### TASK-5.1.1 — `GcpStorageAdapter`
Implement `GcpStorageAdapter` in `src/adapters/storage/gcp_storage.py` implementing `StoragePort`:
- `upload_model(run_id, gguf_path) → str` — uploads to `gs://<GCS_BUCKET>/models/<run_id>/aipet.gguf`, returns the GCS URI
- `download_model(run_id, dest_path) → Path` — downloads GGUF to a local path
- `list_models() → list[str]` — returns run IDs with a stored GGUF
- Config via env vars: `GCS_BUCKET`, `GOOGLE_APPLICATION_CREDENTIALS`

**Outputs:** `src/adapters/storage/gcp_storage.py`, `tests/unit/test_gcp_storage.py`

#### TASK-5.1.2 — Wire upload into `export_activity`
After the GGUF is written, call `GcpStorageAdapter.upload_model()` when `GCS_BUCKET` is set. Log the returned GCS URI so the Temporal UI displays the artifact location.

**Outputs:** Updated `src/interactors/temporal/activities.py`

---

### Feature 5.2 — Model Registry

**Fill in this section with:** whether to use the existing SQLAlchemy DB or a separate store (e.g. GCS metadata JSON), and what fields matter most for filtering/sorting in the UI.

#### TASK-5.2.1 — `ModelRecord` schema and DB table
Add to `src/domain/models.py`:
```python
class ModelRecord(BaseModel):
    run_id: str
    eval_score: float
    base_model: str       # e.g. "HuggingFaceTB/SmolLM2-1.7B"
    epochs: int
    created_at: datetime
    gcs_uri: str          # gs:// path to the GGUF
    is_active: bool       # currently loaded by the inference adapter
```
Add an Alembic migration for the `model_records` table.

**Outputs:** Updated `src/domain/models.py`, new Alembic migration

#### TASK-5.2.2 — `ModelRegistryPort` + SQL adapter
Add to `src/domain/ports.py`:
- `register(record: ModelRecord) → None`
- `list_models() → list[ModelRecord]`
- `set_active(run_id: str) → ModelRecord`
- `get_active() → ModelRecord | None`

Implement `SqlModelRegistryAdapter` in `src/adapters/database/model_registry.py`.

**Outputs:** Updated `src/domain/ports.py`, `src/adapters/database/model_registry.py`, `tests/unit/test_model_registry.py`

---

### Feature 5.3 — Model Management API Endpoints

**Fill in this section with:** auth requirements (open or gated), whether the activate endpoint should be synchronous or kick off a background task, and desired response shape for `GET /models`.

#### TASK-5.3.1 — Model management routes
Add to `src/interactors/api/`:
- `GET /models` — list all registered models (run ID, eval score, base model, GCS URI, is_active)
- `GET /models/active` — return the currently loaded model record
- `POST /models/{run_id}/activate` — download GGUF from GCS, hot-swap the inference adapter, mark `is_active`
- `POST /models/{run_id}/infer` — run a one-off inference with the specified model *without* making it active (useful for A/B testing)

**Outputs:** `src/interactors/api/routes_models.py`, `tests/integration/test_model_routes.py`

#### TASK-5.3.2 — Hot-swap support in `LlamaCppInferenceAdapter`
Add `reload(gguf_path: str) → None` that unloads the current model and loads the new one. Wrap the swap in a lock so in-flight requests drain before the model switches.

**Outputs:** Updated `src/adapters/inference.py`

---

### Feature 5.4 — Auto-Register on Eval Pass (Temporal)

**Fill in this section with:** whether auto-activate should be the default or opt-in, and any notification hook (Slack/email) wanted on successful registration.

#### TASK-5.4.1 — `register_model_activity`
After `export_activity` succeeds, add `register_model_activity` to `src/interactors/temporal/activities.py`:
1. Calls `GcpStorageAdapter.upload_model()` → GCS URI
2. Calls `ModelRegistryPort.register()` with eval score and metadata
3. Calls `ModelRegistryPort.set_active()` if the `auto_activate` workflow param is `True`

**Outputs:** Updated `src/interactors/temporal/activities.py`, `src/interactors/temporal/workflows.py`

#### TASK-5.4.2 — `--auto-activate` flag on `trigger_training` CLI
```bash
uv run python -m src.cli.trigger_training \
  --experiment-name aipet-v2 \
  --remote-backend kaggle \
  --auto-activate   # activates the model immediately after eval passes
```

**Outputs:** Updated `src/interactors/cli/trigger_training.py`

---

## EPIC-6: Authentication for Public Access

> Add API key authentication so the FastAPI backend can be safely exposed to the internet and called from a public-facing React app.

**Goals:**
- Protect all API endpoints with a simple, low-friction auth mechanism suitable for a single-team React client
- No user accounts required — one or more pre-issued API keys is sufficient for now
- CORS configured so the React app's origin can reach the API
- Easy to extend to per-user tokens later if needed

**Fill in this section with:** the React app's domain/origin (needed for CORS), whether the API key should be embedded in the React app (public, read-only) or kept server-side only, and whether any endpoints (e.g. `GET /health`) should remain unauthenticated.

---

### Feature 6.1 — API Key Middleware

#### TASK-6.1.1 — API key store and domain model
Add `ApiKey` to `src/domain/models.py`:
```python
class ApiKey(BaseModel):
    key_hash: str       # sha256 of the raw key — never store plaintext
    label: str          # human-readable name, e.g. "react-app-prod"
    created_at: datetime
    is_active: bool
```
Store keys in the existing DB via an Alembic migration. Seed at least one key on first startup from the `API_KEYS` env var (comma-separated raw keys; hashed on write).

**Outputs:** Updated `src/domain/models.py`, new Alembic migration, `src/adapters/database/api_key_store.py`

#### TASK-6.1.2 — FastAPI dependency for key validation
Add `src/interactors/api/auth.py` with a `require_api_key` dependency:
- Reads the `X-Api-Key` header (or `Authorization: Bearer <key>`)
- Hashes the incoming value and compares against active keys in the DB
- Returns HTTP 401 if missing or unrecognised, HTTP 403 if the key exists but `is_active=False`
- Excluded routes: `GET /health` (unauthenticated liveness probe)

**Outputs:** `src/interactors/api/auth.py`, `tests/unit/test_auth.py`

#### TASK-6.1.3 — Apply the dependency to all routers
Pass `dependencies=[Depends(require_api_key)]` at the router level so new routes are protected by default without per-endpoint decoration.

**Outputs:** Updated `src/interactors/api/app.py`

---

### Feature 6.2 — CORS Configuration

#### TASK-6.2.1 — Add `CORSMiddleware`
Add `fastapi.middleware.cors.CORSMiddleware` to `src/interactors/api/app.py`:
- `allow_origins` driven by `CORS_ORIGINS` env var (comma-separated list; defaults to `[]` in production, `["*"]` only in local dev when `APP_ENV=development`)
- `allow_methods=["GET", "POST"]`
- `allow_headers=["X-Api-Key", "Content-Type"]`

**Outputs:** Updated `src/interactors/api/app.py`

#### TASK-6.2.2 — Document env vars
Add to `README.md` (or a new `docs/configuration.md`):
| Env var | Required | Example | Purpose |
|---|---|---|---|
| `API_KEYS` | Yes (prod) | `key1,key2` | Comma-separated raw API keys seeded on startup |
| `CORS_ORIGINS` | Yes (prod) | `https://app.example.com` | Allowed React app origins |
| `APP_ENV` | No | `development` | Set to `development` to allow wildcard CORS locally |

---

### Feature 6.3 — Key Management CLI

#### TASK-6.3.1 — `src/interactors/cli/manage_keys.py`
Thin CLI for ops use:
```bash
uv run python -m src.cli.manage_keys create --label "react-app-prod"
# Prints the raw key once — store it in your secrets manager

uv run python -m src.cli.manage_keys list
# Shows label, created_at, is_active (never shows raw key)

uv run python -m src.cli.manage_keys revoke --label "react-app-prod"
```

**Outputs:** `src/interactors/cli/manage_keys.py`, `tests/cli/test_manage_keys.py`

---

### Feature 6.4 — Integration Tests

#### TASK-6.4.1 — Auth integration tests
Cover the full request cycle with a real test DB:
- Unauthenticated request → 401
- Wrong key → 401
- Revoked key → 403
- Valid key → 200 on a protected endpoint
- `GET /health` → 200 with no key

**Outputs:** `tests/integration/test_auth.py`
