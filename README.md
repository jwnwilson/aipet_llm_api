# aipet-llm

AI pet companion inference service. Takes a simplified 3D scene and pet stats (hunger, boredom, social, toilet, tiredness) and returns a valid action + optional target object to drive a browser game character.

Runs a quantised GGUF model via llama-cpp-python — designed for a Raspberry Pi 5 (8GB, ARM64, no GPU).

## Quick start (local dev)

```bash
uv sync
make serve                        # starts uvicorn on :8000 with hot-reload
make request                      # sends a test /infer request
```

## Available commands

```bash
make help
```

## Temporal training pipeline (orchestrated)

Run the full dataset → train → evaluate → export lifecycle as a durable Temporal workflow.

### Start infrastructure

```bash
docker compose up temporal -d       # Temporal server + web UI on :8233
docker compose up temporal-worker   # activity worker (connects to Temporal)
```

### Trigger an experiment

```bash
# Generate a fresh dataset, train for 10 epochs, export GGUF if eval passes
python -m src.cli.trigger_training --experiment-name run-001 --epochs 10 --patience 3

# Reuse the existing dataset (hyperparameter sweep — skips generate step)
python -m src.cli.trigger_training --experiment-name sweep-lr --epochs 5 --skip-generate
```

The CLI prints the workflow ID and a direct link to the Temporal Web UI:

```
Workflow started
  ID     : training-run-001-a1b2c3d4
  Run ID : <uuid>
  UI     : http://localhost:8233/namespaces/default/workflows/training-run-001-a1b2c3d4
```

### Inspect workflow history

Open http://localhost:8233 in your browser. Each pipeline stage appears as a separate activity event with its inputs, outputs, and retry history. A failed evaluation is surfaced in the workflow result (`passed=False`) without terminating with an exception — the GGUF export is simply skipped.

### Pipeline stages

| Stage | Activity | Timeout | Retries |
|-------|----------|---------|---------|
| Generate dataset | `generate_dataset_activity` | 30 min | 3 |
| Fine-tune | `train_activity` | 6 h | 1 |
| Evaluate | `evaluate_activity` | 30 min | 3 |
| Export GGUF | `export_activity` | 1 h | 1 (only if eval passes) |

---

## Training pipeline (manual)

```bash
make data                         # generate 2000 train + 200 eval examples
make train                        # fine-tune SmolLM-360M (3 epochs, ~2h on M1)
make train DRY_RUN=1              # 1-step smoke test
make evaluate                     # score HF checkpoint (target: ≥ 95% parse rate)
make setup-llama                  # clone + build llama.cpp (required for export)
make export                       # convert checkpoint → models/aipet.gguf (Q4_K_M)
make evaluate-gguf                # score the GGUF model
```

## CI/CD (GitHub Actions)

The deploy workflow builds an ARM64 image, pushes it to ECR, and applies k8s manifests. It reads credentials from GitHub Actions secrets.

### Required secrets

| Secret | Description |
|--------|-------------|
| `AWS_ROLE_ARN` | IAM role for OIDC auth — `terraform -chdir=infra/terraform output -raw github_actions_role_arn` |
| `AWS_S3_BUCKET` | S3 bucket for training artefacts |
| `AIPET_AWS_ACCESS_KEY_ID` | AWS access key for the aipet service account |
| `AIPET_AWS_SECRET_ACCESS_KEY` | AWS secret key for the aipet service account |
| `AUTH0_DOMAIN` | Auth0 tenant domain (e.g. `yourapp.auth0.com`) |
| `AUTH0_AUDIENCE` | Auth0 API audience identifier |
| `AUTH0_CLIENT_ID` | Auth0 application client ID |
| `CORS_ORIGINS` | Comma-separated allowed origins (e.g. `https://yourapp.com`) |
| `KUBE_CONFIG` | Base64-encoded kubeconfig for the cluster — `base64 -i ~/.kube/config.yaml` |

### Setting secrets for a new repo

1. Copy `.env.example` to `.env` and fill in your values.
2. Run the helper script:
   ```bash
   ./scripts/set_github_secrets.sh
   # Or for a fork / different repo:
   ./scripts/set_github_secrets.sh --repo owner/repo
   ```
   This sets all secrets that can be read from `.env`. `AWS_ROLE_ARN` and `KUBE_CONFIG` must be set manually (they are not in `.env`):
   ```bash
   gh secret set AWS_ROLE_ARN --body "arn:aws:iam::123456789:role/your-role"
   gh secret set KUBE_CONFIG < <(base64 -i ~/.kube/config.yaml)
   ```
3. Verify:
   ```bash
   gh secret list
   ```
4. Trigger a deploy:
   ```bash
   gh workflow run deploy.yml
   ```

---

## Deployment (Raspberry Pi 5)

### Prerequisites

- Docker with `buildx` and QEMU support for ARM64 cross-compilation:
  ```bash
  docker buildx create --use
  docker run --privileged --rm tonistiigi/binfmt --install arm64
  ```
- SSH access to the RPi as user `pi` (default host: `raspberrypi.local`).
- Docker installed on the RPi.

### Deploy

```bash
# Build the ARM64 image and push it to the RPi in one step:
make docker-deploy RPI_HOST=raspberrypi.local

# Or step by step:
make docker-build               # build linux/arm64 image locally
make docker-export              # save as aipet-llm.tar.gz
scp aipet-llm.tar.gz pi@raspberrypi.local:~/

# On the RPi:
docker load -i ~/aipet-llm.tar.gz
docker compose up -d
```

### Verify

```bash
curl http://raspberrypi.local:8000/health
```

### Swap the model without rebuilding

The `models/` directory is mounted as a volume, so you can update the GGUF without rebuilding the image:

```bash
scp models/aipet.gguf pi@raspberrypi.local:~/models/
ssh pi@raspberrypi.local "docker compose restart"
```
