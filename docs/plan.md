# Plan

> Completed work → [complete.md](complete.md)

---

## EPIC-1: Run Training Locally with Kaggle GPU

> The Temporal workflow and Kaggle adapter are built. This epic covers standing up the local environment and running an end-to-end training experiment on a free Kaggle GPU.

**Prerequisites:**
- Kaggle account with phone verification (required for GPU access)
- `kaggle` CLI: `uv add kaggle` or `pip install kaggle`
- Docker Desktop running (for Temporal server)

---

### Feature 1.1 — Kaggle credentials

#### TASK-1.1.1 — Create Kaggle API key
1. Go to kaggle.com → Settings → API → **Create New Token** — downloads `kaggle.json`
2. Place it at `~/.kaggle/kaggle.json` and `chmod 600 ~/.kaggle/kaggle.json`
3. Verify: `kaggle datasets list` — should return results without auth errors

#### TASK-1.1.2 — Set environment variables
```bash
export KAGGLE_USERNAME=<your-kaggle-username>
export KAGGLE_KEY=<your-kaggle-key>
```
Add both to your shell profile (`.zshrc` / `.bashrc`) so they persist across sessions.

---

### Feature 1.2 — Start Temporal server locally

#### TASK-1.2.1 — Bring up Temporal via docker-compose
The `docker-compose.yml` includes `temporal`, `temporal-db`, and `temporal-ui` services. Start only these — the worker runs locally (outside Docker) so it can reach your local filesystem and Kaggle credentials directly:
```bash
docker compose up temporal temporal-db temporal-ui -d
```
Wait ~30 seconds for Temporal to finish its auto-setup, then verify:
```bash
docker compose ps   # temporal should be "healthy"
```
Temporal UI is now at **http://localhost:8233**.

---

### Feature 1.3 — Run the Temporal worker locally

The worker must run outside Docker so it can write checkpoints to `models/` and read `data/` from the host filesystem:
```bash
KAGGLE_USERNAME=$KAGGLE_USERNAME \
KAGGLE_KEY=$KAGGLE_KEY \
uv run python -m src.temporal.worker
```
Expected output:
```
Worker started — task_queue=aipet-training  host=localhost:7233
```
Leave this terminal running.

---

### Feature 1.4 — Generate dataset and trigger training

#### TASK-1.4.1 — Generate training data (if not already current)
```bash
uv run python -m src.cli.generate_dataset
```
Verify `data/train.jsonl` has 5 000 lines and `data/eval.jsonl` has 500.

#### TASK-1.4.2 — Trigger the Kaggle training workflow
Open a second terminal and run:
```bash
uv run python -m src.cli.trigger_training \
  --experiment-name aipet-v1 \
  --remote-backend kaggle \
  --model HuggingFaceTB/SmolLM2-1.7B \
  --epochs 5 \
  --patience 3
```
The CLI prints a workflow ID and a direct link to the Temporal UI for that run.

What happens internally:
1. `generate_dataset_activity` — skipped if `--skip-generate` is passed
2. `train_activity` — pushes `data/` to a Kaggle Dataset, renders and pushes the notebook, polls status every 60 s via `kaggle kernels status`
3. `evaluate_activity` — runs eval on the downloaded checkpoint
4. `export_activity` — converts to GGUF at `models/aipet.gguf` only if eval passes (≥ 95%)

#### TASK-1.4.3 — Monitor progress
- **Temporal UI:** http://localhost:8233 → find the workflow by experiment name; the activity timeline shows which step is running
- **Kaggle kernel:** https://kaggle.com → Code → Your Work → find the kernel matching `<KAGGLE_USERNAME>/aipet-v1`
- **Worker logs:** the terminal running `temporal.worker` streams `Remote status: kaggle … running`

---

### Feature 1.5 — Validate results

#### TASK-1.5.1 — Confirm checkpoint downloaded
```bash
ls -lh models/checkpoints/     # HuggingFace checkpoint files
ls -lh models/aipet.gguf       # GGUF export (~1 GB for SmolLM2-1.7B Q4_K_M)
```

#### TASK-1.5.2 — Run quality report
```bash
uv run python -m src.cli.evaluate --quality
# Expected: ≥ 95% schema-valid, per-stat accuracy ≥ 0.90
```

#### TASK-1.5.3 — Run quality integration tests
```bash
pytest tests/integration/test_model_quality.py -v
```
All four assertions must pass: per-stat accuracy, target accuracy, no dominant action, priority conflict resolution.

---

## EPIC-2: Deploy to Raspberry Pi 5

> Get the inference API running on the physical RPi using the trained GGUF model.

**Prerequisites:** EPIC-1 complete (`models/aipet.gguf` exists), Docker with `buildx` and QEMU support for ARM64 cross-compilation.

---

### Feature 2.1 — Build ARM64 image

#### TASK-2.1.1 — Enable ARM64 builds on dev machine (one-time)
```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx create --use --name rpi-builder
```

#### TASK-2.1.2 — Build and export image tarball
```bash
docker buildx build \
  --platform linux/arm64 \
  --load \
  -t aipet-llm:latest .

docker save aipet-llm:latest | gzip > /tmp/aipet-llm.tar.gz
```

---

### Feature 2.2 — Write RPi deploy script

#### TASK-2.2.1 — Create `src/cli/deploy.py`
Thin CLI that does the full transfer in one command:
- `rsync` the image tarball and `models/aipet.gguf` to the RPi
- SSH in and run `docker load`, then restart with `docker compose up -d`
- Config via env vars: `RPI_HOST`, `RPI_USER`, `RPI_KEY_PATH`

```bash
RPI_HOST=raspberrypi.local RPI_USER=pi uv run python -m src.cli.deploy
```

---

### Feature 2.3 — First-time RPi setup

#### TASK-2.3.1 — Install Docker on RPi
```bash
ssh pi@raspberrypi.local
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker pi
```

#### TASK-2.3.2 — Copy and start the service
```bash
rsync /tmp/aipet-llm.tar.gz pi@raspberrypi.local:~/
rsync models/aipet.gguf pi@raspberrypi.local:~/models/
ssh pi@raspberrypi.local "docker load < ~/aipet-llm.tar.gz && docker compose up -d"
```

#### TASK-2.3.3 — Validate the endpoint
```bash
curl http://raspberrypi.local:8000/health
# Expected: {"status": "ok", "model": "/app/models/aipet.gguf"}
```

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
