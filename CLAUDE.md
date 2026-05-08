# aipet-llm

An AI pet companion that runs a lightweight LLM on a Raspberry Pi 5 (8GB). It takes simplified 3D scene data and pet stats (hunger, boredom, social, toilet, tiredness) and returns a valid action + optional target object to drive a 3D browser game character.

Full requirements: [docs/prd.md](docs/prd.md) | Implementation plan: [docs/plan.md](docs/plan.md)

## Stack

- Python ≥ 3.12, package manager: `uv`
- FastAPI + uvicorn (API layer)
- llama-cpp-python with GGUF quantised model (inference, no GPU, ARM64)
- HuggingFace transformers + datasets + torch (training only, dev dep)
- pytest + httpx + pytest-asyncio (tests)
- Target hardware: Raspberry Pi 5 (8GB), Docker ARM64 container

## Architecture

Hexagonal architecture — domain logic has no I/O dependencies:

```
src/
  domain/          # pure business logic, no I/O
    models.py      # Pydantic schemas: SceneObject, SceneData, PetStats, InferenceRequest/Response
    actions.py     # Action enum: EAT, DRINK, PLAY, FETCH, SLEEP, SOCIAL, FOLLOW, TOILET, IDLE, EXPLORE
    ports.py       # abstract ports: InferencePort, RemoteTrainingPort
    train/         # training domain logic (no CLI, no argparse)
      dataset.py   # generate(), label(), make_example()
      trainer.py   # train(), build_hf_dataset(), load_jsonl()
      evaluate.py  # evaluate(), load_hf_pipeline(), load_llama_cpp_adapter()
      export.py    # export() — HF checkpoint → GGUF
  infrastructure/  # local infrastructure adapters, implements ports
    inference.py   # LlamaCppInferenceAdapter
    prompt.py      # build_prompt() + parse_response()
  adapters/        # 3rd-party / remote adapters, implements ports
    kaggle_adapter.py   # KaggleTrainingAdapter(RemoteTrainingPort)
    ssh_adapter.py      # SshTrainingAdapter(RemoteTrainingPort)
  api/             # FastAPI adapter (primary/driving)
    app.py
    routes.py      # POST /infer, GET /health
  cli/             # thin CLI wrappers (argparse + sys.exit only)
    generate_dataset.py
    train.py
    evaluate.py
    export.py
    infer.py
tests/
  unit/
  integration/
data/
  train.jsonl      # 5000 synthetic examples
  eval.jsonl       # 500 synthetic examples
models/
  checkpoints/     # HuggingFace fine-tune output
  aipet.gguf       # quantised Q4_K_M export for RPi
```

> **Do not use a `scripts/` folder.** CLI entrypoints live in `src/cli/`; training domain logic lives in `src/domain/train/`.
> **Adapter placement rule:** Ports (interfaces) belong in `src/domain/ports.py`. Local infrastructure (llama.cpp) goes in `src/infrastructure/`. Any adapter that communicates with a 3rd-party service (Kaggle, SSH remotes, external APIs) goes in `src/adapters/`.

## Domain rules

- Valid actions and their target requirements:

  | Action  | Target required | Valid target types |
  |---------|-----------------|--------------------|
  | EAT     | Yes             | bowl               |
  | DRINK   | Yes             | bowl               |
  | PLAY    | Yes             | toy                |
  | FETCH   | Yes             | toy                |
  | SLEEP   | Yes             | bed                |
  | SOCIAL  | Yes             | player, pet        |
  | FOLLOW  | Yes             | player, pet        |
  | TOILET  | No              | —                  |
  | IDLE    | No              | —                  |
  | EXPLORE | No              | —                  |

- Only actions whose target type is present in the scene are valid — the prompt filters available actions before inference.
- Scene objects: `{type: bowl|bed|toy|player|pet, id: str, distance: float}` — no position coordinates.
- On parse failure, adapters must return `Action.IDLE` (never raise).
- Prompt must stay under 300 tokens for RPi-friendly context windows.

## Success metric

> **≥ 95%** of model responses must parse as a valid `InferenceResponse` on the 200-example eval set.

## Implementation phases

| Phase | Tasks | Gate |
|-------|-------|------|
| 1 — Foundation | 1.1 project structure → then 1.2 schemas + 1.3 ports in parallel | `pytest tests/unit/` passes |
| 2 — Core implementation | 2.1 inference adapter, 2.2 prompt/parser, 2.3 dataset generator (all parallel) | `pytest tests/unit/` passes |
| 3 — API layer | 3.1 FastAPI app → 3.2 integration tests | `pytest tests/integration/` passes |
| 4 — Training pipeline | 4.1 fine-tune script + 4.2 eval/export (parallel; runs alongside Phase 3) | `scripts/evaluate.py` reports ≥ 95% |
| 5 — Deployment | 5.1 Docker ARM64 config | `GET /health` returns 200 on ARM64 image |

## Workflow

### Running a task with an agent

Hand each task block from [docs/plan.md](docs/plan.md) to a sub-agent:
- Provide the task block, the files listed under **Inputs**, and the instruction: *"Complete this task. Write your outputs to the paths listed."*
- Tasks within the same phase are independent — run them in parallel.
- Tasks in later phases depend on all earlier phases completing first (except Phase 4, which runs alongside Phase 3).

