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

## Training pipeline

```bash
make data                         # generate 2000 train + 200 eval examples
make train                        # fine-tune SmolLM-360M (3 epochs, ~2h on M1)
make train DRY_RUN=1              # 1-step smoke test
make evaluate                     # score HF checkpoint (target: ≥ 95% parse rate)
make setup-llama                  # clone + build llama.cpp (required for export)
make export                       # convert checkpoint → models/aipet.gguf (Q4_K_M)
make evaluate-gguf                # score the GGUF model
```

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
