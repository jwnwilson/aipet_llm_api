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
  domain/          # pure business logic
    models.py      # Pydantic schemas: SceneObject, SceneData, PetStats, InferenceRequest/Response
    actions.py     # Action enum: EAT, DRINK, PLAY, FETCH, SLEEP, SOCIAL, FOLLOW, TOILET, IDLE, EXPLORE
    ports.py       # abstract InferencePort interface
  infrastructure/  # LLM adapter, implements ports
    inference.py   # LlamaCppInferenceAdapter
    prompt.py      # build_prompt() + parse_response()
  api/             # FastAPI adapter
    app.py
    routes.py      # POST /infer, GET /health
tests/
  unit/
  integration/
scripts/
  generate_dataset.py
  train.py
  evaluate.py
  export.py
data/
  train.jsonl      # 2000 synthetic examples
  eval.jsonl       # 200 synthetic examples
models/
  checkpoints/     # HuggingFace fine-tune output
  aipet.gguf       # quantised Q4_K_M export for RPi
```

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

### Regenerating the plan

Ask: *"Read the PRD and regenerate the plan in docs/plan.md"* using this structure:

```
## Phase N: <name>
> Prerequisite: Phase N-1 complete (or "none" for Phase 1)

### Task N.1 — <title>
**Goal:** One sentence.
**Inputs:** Files/data this task reads.
**Outputs:** Files/data this task produces.
**Steps:** Numbered list of concrete actions.
---
```
