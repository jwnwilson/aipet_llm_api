# Implementation Plan

> Generated from [prd.md](prd.md). Regenerate by asking: *"Read the PRD and regenerate the plan."*
>
> Phases run sequentially unless noted. Tasks within a phase run in parallel.

---

## Phase 1: Foundation
> Prerequisite: none
> Note: Task 1.1 must complete before 1.2 and 1.3, which can then run in parallel.

### Task 1.1 — Project structure & dependencies
**Goal:** Establish hexagonal directory layout and install all required packages.
**Inputs:** `pyproject.toml`
**Outputs:** Updated `pyproject.toml`, `src/` tree skeleton with `__init__.py` files.
**Steps:**
1. Create the following directory structure:
   ```
   src/
     domain/          # pure business logic, no I/O
       models.py      # Pydantic schemas (SceneData, PetStats, PetResponse)
       ports.py       # abstract InferencePort interface
       actions.py     # Action enum / presets
     infrastructure/  # LLM adapter, implements ports
       inference.py
       prompt.py
     api/             # FastAPI adapter
       app.py
       routes.py
   tests/
     unit/
     integration/
   ```
2. Add dependencies via `uv add`: `fastapi`, `uvicorn`, `pydantic`, `llama-cpp-python` (for RPi GGUF inference), `transformers`, `datasets`, `torch` (for training only, dev dep).
3. Add dev dependencies: `pytest`, `httpx`, `pytest-asyncio`.
---

### Task 1.2 — Pydantic schemas
**Goal:** Define all input/output data contracts used across the system.
**Inputs:** PRD "Must have" section, `src/domain/` skeleton.
**Outputs:** `src/domain/models.py`, `src/domain/actions.py`
**Steps:**
1. Define `Action` enum with presets: `EAT`, `DRINK`, `PLAY`, `SLEEP`, `TOILET`, `IDLE`, `SOCIAL`.
2. Define `PetStats` model: `hunger: float`, `boredom: float`, `social: float`, `toilet: float`, `tiredness: float` (all 0.0–1.0).
3. Define `SceneObject` model: `id: str`, `type: Literal["bowl", "bed", "toy", "player", "pet"]`, `distance: float`.
4. Define `SceneData` model: `objects: list[SceneObject]`, `tick: int`.
5. Define `InferenceRequest` model: `scene: SceneData`, `pet_stats: PetStats`.
6. Define `InferenceResponse` model: `action: Action`, `target_object_id: str | None`, `confidence: float | None`.
7. Add JSON schema export helper for use in prompt engineering.
---

### Task 1.3 — Domain ports (interfaces)
**Goal:** Define the abstract interface the domain layer expects from the LLM infrastructure.
**Inputs:** `src/domain/` skeleton, Task 1.2 schemas.
**Outputs:** `src/domain/ports.py`
**Steps:**
1. Define abstract base class `InferencePort` with method `infer(request: InferenceRequest) -> InferenceResponse`.
2. Add docstring specifying the contract: must always return a valid `InferenceResponse`; must never raise on recoverable LLM errors (return `IDLE` action instead).
3. Write a unit test in `tests/unit/test_ports.py` that creates a mock implementation and verifies the interface contract.
---

## Phase 2: Core Implementation
> Prerequisite: Phase 1 complete. All three tasks run in parallel.

### Task 2.1 — Inference adapter (llama.cpp)
**Goal:** Implement `InferencePort` using a GGUF-quantised model via llama-cpp-python.
**Inputs:** `src/domain/ports.py`, `src/domain/models.py`, `src/infrastructure/inference.py`
**Outputs:** `src/infrastructure/inference.py` (complete implementation)
**Steps:**
1. Implement `LlamaCppInferenceAdapter(InferencePort)` that loads a GGUF model from a configurable path.
2. Accept model path and context size via constructor (default context 512 tokens — RPi friendly).
3. Call `src/infrastructure/prompt.py` (Task 2.2) to build the prompt, run inference, and parse the response.
4. On parse failure, log a warning and return `InferenceResponse(action=Action.IDLE, target_object_id=None)`.
5. Write unit test with a stub model (mock `Llama` call) to verify adapter error handling.
---

### Task 2.2 — Prompt template & response parser
**Goal:** Build and validate the prompt sent to the LLM and parse its JSON output reliably.
**Inputs:** `src/domain/models.py`, `src/infrastructure/prompt.py`
**Outputs:** `src/infrastructure/prompt.py` (complete)
**Steps:**
1. Write `build_prompt(request: InferenceRequest) -> str` that serialises scene + pet stats into a compact system prompt instructing the model to respond with JSON matching `InferenceResponse` schema.
2. Include in the prompt only the actions valid for the current scene (filter by action-object mapping) — reduces output space and invalid responses.
3. Include the JSON schema in the prompt so the model knows the exact expected output format.
3. Keep prompt under 300 tokens to stay within RPi-friendly context windows.
4. Write `parse_response(raw: str) -> InferenceResponse` that extracts the first JSON block from the model output and validates it against `InferenceResponse`.
5. Write unit tests covering: valid response, response with extra text, malformed JSON, missing required fields.
---

### Task 2.3 — Synthetic training dataset generator
**Goal:** Generate a labelled dataset of (scene + pet stats) → action pairs for fine-tuning.
**Inputs:** `src/domain/models.py`
**Outputs:** `scripts/generate_dataset.py`, `data/train.jsonl`, `data/eval.jsonl`
**Steps:**
1. Write a script that generates random `InferenceRequest` instances with varied pet stat profiles (e.g. high hunger → EAT, high tiredness → SLEEP).
2. Implement a deterministic rule-based labeller to assign ground-truth `Action` based on which stat is highest.
3. Generate 2000 training examples and 200 eval examples in JSONL format (each line: `{"prompt": "...", "completion": "..."}`).
4. Include a mix of scenes with 0–10 objects so the model learns to optionally select a `target_object_id`.
5. Write a validation script that checks all entries parse as valid `InferenceRequest` + `InferenceResponse`.
---

## Phase 3: API Layer
> Prerequisite: Phase 2 complete. Tasks 3.1 and 3.2 run sequentially (tests require the app).

### Task 3.1 — FastAPI application
**Goal:** Expose a single inference endpoint that accepts scene + pet stats and returns a pet action.
**Inputs:** `src/domain/ports.py`, `src/domain/models.py`, `src/api/`
**Outputs:** `src/api/app.py`, `src/api/routes.py`
**Steps:**
1. Create FastAPI app in `src/api/app.py`; inject `InferencePort` via dependency injection (not imported directly — preserves hexagonal boundary).
2. Define `POST /infer` route accepting `InferenceRequest`, returning `InferenceResponse`.
3. Add `GET /health` returning `{"status": "ok", "model": "<model_path>"}`.
4. Add a startup event that loads the `LlamaCppInferenceAdapter` and wires it into the DI container.
5. Return HTTP 422 with a descriptive message if schema validation fails; HTTP 500 with `{"error": "inference_failed"}` on unrecoverable errors.
---

### Task 3.2 — API integration tests
**Goal:** Verify the full request/response cycle works end-to-end with a stub inference adapter.
**Inputs:** `src/api/app.py`, `tests/integration/`
**Outputs:** `tests/integration/test_api.py`
**Steps:**
1. Use `httpx.AsyncClient` with `app` mounted directly (no real server needed).
2. Inject a `FakeInferenceAdapter` that always returns `Action.IDLE`.
3. Test: valid request returns 200 with valid `InferenceResponse` schema.
4. Test: malformed request body returns 422.
5. Test: `GET /health` returns 200.
6. Test: adapter raising an exception returns 500.
---

## Phase 4: Training Pipeline
> Prerequisite: Phase 2 complete. Runs in parallel with Phase 3. Tasks 4.1 and 4.2 run in parallel.

### Task 4.1 — Fine-tuning script
**Goal:** Fine-tune a small base model on the synthetic dataset to output valid structured actions.
**Inputs:** `data/train.jsonl`, `data/eval.jsonl`, `scripts/train.py`
**Outputs:** `scripts/train.py`, trained model checkpoint in `models/checkpoints/`
**Steps:**
1. Use HuggingFace `transformers` + `Trainer` to fine-tune a small base model (default: TinyLlama-1.1B or SmolLM-360M).
2. Use causal LM fine-tuning on the prompt+completion pairs from the dataset.
3. Train for 3 epochs, eval every 200 steps, save best checkpoint by eval loss.
4. Log training metrics (loss, eval loss) to stdout in a format parseable by the eval script.
5. Add a `--dry-run` flag that trains for 1 step to verify the pipeline works without a GPU.
---

### Task 4.2 — Evaluation & export script
**Goal:** Measure schema-valid response rate and export the model to GGUF for RPi deployment.
**Inputs:** `models/checkpoints/`, `data/eval.jsonl`, `scripts/evaluate.py`, `scripts/export.py`
**Outputs:** `scripts/evaluate.py`, `scripts/export.py`, `models/aipet.gguf`
**Steps:**
1. Write `scripts/evaluate.py`: load checkpoint, run inference on all 200 eval examples, compute % responses that parse as valid `InferenceResponse` (target: >95%).
2. Print a breakdown of action distribution to catch degenerate models (e.g. always predicting `IDLE`).
3. Write `scripts/export.py`: convert HuggingFace checkpoint to GGUF format using `llama.cpp`'s `convert_hf_to_gguf.py`, then quantise to Q4_K_M (good RPi balance of size/quality).
4. Verify the exported GGUF loads correctly with `LlamaCppInferenceAdapter` and passes the eval suite.
---

## Phase 5: Deployment
> Prerequisite: Phases 3 and 4 complete.

### Task 5.1 — Docker deployment config
**Goal:** Package the API and model into a Docker image that runs on a Raspberry Pi 5 (ARM64).
**Inputs:** `src/`, `models/aipet.gguf`, `Dockerfile`
**Outputs:** `Dockerfile`, `docker-compose.yml`, `scripts/deploy.sh`
**Steps:**
1. Write a multi-arch `Dockerfile` (`linux/arm64`) based on `python:3.12-slim`; install `llama-cpp-python` compiled for ARM (no GPU).
2. Copy `src/` and `models/aipet.gguf` into the image; set `MODEL_PATH=/app/models/aipet.gguf` as env var.
3. Expose port 8000; entrypoint: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`.
4. Write `docker-compose.yml` for local development and single-node RPi 5 deployment.
5. Write `scripts/deploy.sh` that builds the image for `linux/arm64` and exports as a tarball for transfer to the RPi.
6. Document the deploy steps in `README.md`.
7. Note: Kubernetes deployment is deferred to post-v1.
---
