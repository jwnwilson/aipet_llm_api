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
**Outputs:** `src/domain/train/dataset.py`, `src/cli/generate_dataset.py`, `data/train.jsonl`, `data/eval.jsonl`
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
**Inputs:** `data/train.jsonl`, `data/eval.jsonl`
**Outputs:** `src/domain/train/trainer.py`, `src/cli/train.py`, trained model checkpoint in `models/checkpoints/`
**Steps:**
1. Use HuggingFace `transformers` + `Trainer` to fine-tune a small base model (default: TinyLlama-1.1B or SmolLM-360M).
2. Use causal LM fine-tuning on the prompt+completion pairs from the dataset.
3. Train for 3 epochs, eval every 200 steps, save best checkpoint by eval loss.
4. Log training metrics (loss, eval loss) to stdout in a format parseable by the eval script.
5. Add a `--dry-run` flag that trains for 1 step to verify the pipeline works without a GPU.
---

### Task 4.2 — Evaluation & export script
**Goal:** Measure schema-valid response rate and export the model to GGUF for RPi deployment.
**Inputs:** `models/checkpoints/`, `data/eval.jsonl`
**Outputs:** `src/domain/train/evaluate.py`, `src/cli/evaluate.py`, `src/domain/train/export.py`, `src/cli/export.py`, `models/aipet.gguf`
**Steps:**
1. Write `src/domain/train/evaluate.py`: load checkpoint, run inference on all 200 eval examples, compute % responses that parse as valid `InferenceResponse` (target: >95%). Thin CLI wrapper at `src/cli/evaluate.py`.
2. Print a breakdown of action distribution to catch degenerate models (e.g. always predicting `IDLE`).
3. Write `src/domain/train/export.py`: convert HuggingFace checkpoint to GGUF format using `llama.cpp`'s `convert_hf_to_gguf.py`, then quantise to Q4_K_M. Thin CLI wrapper at `src/cli/export.py`.
4. Verify the exported GGUF loads correctly with `LlamaCppInferenceAdapter` and passes the eval suite.

> Do not add scripts to a `scripts/` folder — use `src/cli/` for CLI entrypoints and `src/domain/train/` for training logic.
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
---

## Phase 6: Model Quality Improvements
> Prerequisite: Phase 4 complete. All four tasks can run in parallel after diagnosis (6.1) is complete.
> **Context:** Post-testing revealed the model returns EAT/SLEEP disproportionately and selects wrong target objects.
> Root causes: (1) severe class imbalance in the training set — TOILET 20%, SLEEP 15%, all other targeted actions ~7%; (2) tick-parity labelling creates contradictory signal where identical prompts produce different labels (EAT vs DRINK) because tick is not in the prompt; (3) the prompt gives no explicit decision rule, does not sort stats by value, and does not sort objects by distance; (4) evaluation measures only schema validity, never action-selection accuracy.

### Task 6.1 — Statistical quality test suite
**Goal:** Replace the single schema-validity pass/fail with a per-stat accuracy report and action-distribution histogram so we can measure real model quality and regression-test improvements.
**Inputs:** `tests/integration/`, `src/domain/train/evaluate.py`, `data/eval.jsonl`, `models/aipet.gguf`
**Outputs:** `src/domain/train/quality_report.py`, `tests/integration/test_model_quality.py`
**Steps:**
1. Create `src/domain/train/quality_report.py` with a `run_quality_report(adapter, n_per_stat=40) -> dict` function that:
   - For each of the 5 stats, generates `n_per_stat` synthetic requests with that stat at 0.9 and all others at 0.1, always including the required scene object.
   - Runs inference on every request and records the predicted action.
   - Computes **per-stat accuracy**: the fraction of responses that match the expected action category (e.g. EAT or DRINK for hunger).
   - For every action that returned a targeted response, computes **target accuracy**: the fraction where `target_object_id` equals the closest valid object's id.
   - Generates 200 uniformly-random requests and computes the **action frequency distribution** as a dict mapping action name → count.
   - Returns a JSON-serialisable report with all metrics plus a pass/fail flag (per-stat accuracy ≥ 0.90, target accuracy ≥ 0.90).
2. Expose the report via the existing `src/cli/evaluate.py` CLI with a `--quality` flag; print a summary table and write `data/quality_report.json`.
3. Create `tests/integration/test_model_quality.py` — skip if `models/aipet.gguf` is absent:
   - `test_per_stat_accuracy_meets_threshold`: each stat's accuracy ≥ 0.90.
   - `test_target_accuracy_meets_threshold`: target accuracy across all targeted actions ≥ 0.90.
   - `test_no_action_dominates`: no single action exceeds 30% of the uniform-random distribution (catches the TOILET/SLEEP/EAT bias).
   - `test_priority_conflict`: 20 examples with two stats both at 0.7+ (one at 0.9, one at 0.7) — verify model picks the higher stat's action ≥ 80% of the time.
   - `test_fallback_when_object_absent`: 20 examples where required object is absent — verify IDLE or EXPLORE is returned ≥ 90% of the time.
4. Wire `test_model_quality.py` into `pytest tests/integration/` so CI automatically catches quality regressions.
---

### Task 6.2 — Dataset regeneration
**Goal:** Fix class imbalance, eliminate inconsistent tick-parity labels, and produce richer multi-target scenes that teach distance-based selection.
**Inputs:** `src/domain/train/dataset.py`
**Outputs:** Updated `src/domain/train/dataset.py`, regenerated `data/train.jsonl`, `data/eval.jsonl`
**Steps:**
1. **Fix class imbalance** — replace `rng.choice(STAT_NAMES)` with stratified sampling:
   - Divide each dataset into 5 equal tranches, one per dominant stat, so every action category appears at equal frequency in the labelled output.
   - After stratification, shuffle the full set to avoid ordering bias.
2. **Eliminate tick-parity label inconsistency** — instead of selecting EAT-vs-DRINK (or PLAY-vs-FETCH, SOCIAL-vs-FOLLOW) by tick parity, make a single random per-example choice and record it in the completion. This prevents the model from seeing the same prompt paired with different labels. Remove tick from the prompt entirely (it was already absent from `build_prompt`; verify `SceneData.tick` is not forwarded).
3. **Richer target-selection scenes** — when generating a "required object present" example, always add 2–4 extra objects of the same valid type at varied distances (e.g. three bowls at 2 m, 8 m, 25 m for a hungry pet). This teaches the model to distinguish closest vs. far rather than defaulting to the first object id.
4. **Add priority-conflict examples** (15% of dataset) — generate examples where two stats are high (one at 0.80–1.0, another at 0.60–0.79) with both required objects present. The labeller already handles this correctly (picks max); these examples teach the model to compare values.
5. **Increase dataset size**: 5 000 train / 500 eval.
6. **Add `check_dataset_distribution(path)`** to `dataset.py` that prints per-action counts and raises `AssertionError` if any action accounts for fewer than 5% or more than 25% of labelled examples. Call it at the end of `generate()`.
7. Update unit tests in `tests/unit/test_dataset.py` to cover the new stratified generator, the richer multi-target scene, and the distribution check.
---

### Task 6.3 — Prompt engineering improvements
**Goal:** Give the model explicit decision rules and sort the context so it can find the highest stat and closest object without needing to internally search unsorted lists.
**Inputs:** `src/infrastructure/prompt.py`, `tests/unit/test_prompt.py`
**Outputs:** Updated `src/infrastructure/prompt.py`, updated tests
**Steps:**
1. **Sort stats high → low** in the prompt so the dominant stat is always first; add a `(highest)` label to the top entry:
   ```
   Stats (highest first): tiredness=0.92 (highest), hunger=0.31, boredom=0.18, social=0.09, toilet=0.05
   ```
2. **Add explicit decision rule** immediately after the stats line:
   ```
   Rule: choose the action that satisfies the highest stat. If a target object is required, select the closest one.
   ```
3. **Sort scene objects by distance** (nearest first) and group same-type objects together so the model can read off the nearest valid target without comparing scattered values:
   ```
   Scene (nearest first): bowl(id=obj_2,dist=2.1), bowl(id=obj_0,dist=8.4), bed(id=obj_1,dist=15.0)
   ```
4. Verify the updated prompt stays under 300 tokens for all plausible inputs (add a token-count assertion in the test).
5. Update `tests/unit/test_prompt.py` to assert: (a) stats appear in descending order, (b) the `(highest)` label appears on the top stat, (c) the explicit rule line is present, (d) scene objects are in ascending distance order.
---

### Task 6.4 — Training improvements
**Goal:** Prevent action-frequency bias during fine-tuning and improve convergence with a larger training set.
**Inputs:** `src/domain/train/trainer.py`, `src/cli/train.py`
**Outputs:** Updated `src/domain/train/trainer.py`, updated `src/cli/train.py`
**Steps:**
1. **Weighted random sampler** — parse each example's completion JSON at dataset-load time to extract the action label; compute inverse-frequency weights so every action class is sampled with equal probability per batch, eliminating the residual frequency bias even if the dataset is already stratified.
2. **Learning rate schedule** — add linear warmup for the first 5% of total training steps followed by cosine annealing decay to 0. Expose `--warmup-ratio` CLI arg (default 0.05).
3. **Increase epochs to 5** with early stopping at patience = 3 (use `EarlyStoppingCallback`). Expose `--patience` CLI arg.
4. **Per-action eval logging** — at each eval step, run the quality report on the eval set (not just loss) and log per-stat accuracy to stdout so training progress on the real metric is visible. This replaces the loss-only logging.
5. **Expose `--base-model` CLI arg** (default `HuggingFaceTB/SmolLM2-1.7B`) — SmolLM2-1.7B has more capacity to learn numeric comparison; at Q4_K_M quantisation it uses ≈ 1 GB on the RPi (well within the 8 GB budget). Keep 360M as a fast-test option.
6. Update `--dry-run` to exercise the full pipeline (sampler, scheduler, quality logging) in 1 step.

---

## Post V1

### Task P.1 — Early stopping to protect against overfitting
**Goal:** Automatically halt training when eval loss stops improving, preventing the model from memorising the synthetic dataset.
**Inputs:** `scripts/train.py`
**Outputs:** Updated `scripts/train.py`
**Steps:**
1. Import `EarlyStoppingCallback` from `transformers`.
2. Add a `--patience` CLI argument (default: 3) — number of consecutive eval checkpoints with no improvement before stopping.
3. Pass `callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]` to `Trainer`.
4. Log a clear message when early stopping triggers, including the best checkpoint path and best eval loss.
5. Update the `--dry-run` path to set `patience=1` so it can be exercised in a single step.
---

### Task P.2 — Kubernetes deployment
**Goal:** Deploy the inference service on a multi-node Kubernetes cluster for production scaling beyond a single RPi.
**Inputs:** `Dockerfile`, `docker-compose.yml`
**Outputs:** `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/hpa.yaml`
**Steps:**
1. Write a `Deployment` manifest using the ARM64 Docker image; set resource requests/limits appropriate for RPi-class nodes.
2. Write a `Service` manifest (ClusterIP) exposing port 8000.
3. Write a `HorizontalPodAutoscaler` targeting CPU utilisation at 70%.
4. Document cluster setup and `kubectl apply` steps in `README.md`.
---

### Task P.3 — Temporal training pipeline
**Goal:** Orchestrate the full LLM training lifecycle (dataset generation → fine-tuning → evaluation → export) as a Temporal workflow, enabling triggered and scheduled re-training experiments with full visibility and retry semantics.
**Inputs:** `src/domain/train/`, `src/cli/`, `data/`, `models/`
**Outputs:** `src/temporal/workflows.py`, `src/temporal/activities.py`, `src/temporal/worker.py`, `src/cli/trigger_training.py`, `docker-compose.yml` (updated with Temporal server)
**Steps:**
1. Add `temporalio` as a dependency via `uv add temporalio`.
2. Write `src/temporal/activities.py` — one `@activity.defn` per pipeline stage, each wrapping the existing domain function:
   - `generate_dataset_activity(config: DatasetConfig) -> DatasetPaths` — calls `src/domain/train/dataset.py:generate()`
   - `train_activity(config: TrainConfig) -> CheckpointPath` — calls `src/domain/train/trainer.py:train()`
   - `evaluate_activity(config: EvalConfig) -> EvalResult` — calls `src/domain/train/evaluate.py:evaluate()`; attaches pass/fail flag (`result.valid_pct >= 0.95`)
   - `export_activity(checkpoint: CheckpointPath) -> GGUFPath` — calls `src/domain/train/export.py:export()`
3. Write `src/temporal/workflows.py` — a single `@workflow.defn` class `TrainingPipelineWorkflow`:
   - Accept `ExperimentConfig` (dataset params, training hyperparameters, experiment name/tag).
   - Run activities in sequence: generate → train → evaluate → (export only if eval passes).
   - Emit a `WorkflowFailed` signal and surface `EvalResult` in the workflow result so failed experiments are visible without raising.
   - Support a `--skip-generate` flag via workflow input to reuse an existing dataset for hyperparameter experiments.
4. Write `src/temporal/worker.py` — registers all activities and the workflow; reads `TEMPORAL_HOST` env var (default `localhost:7233`) and task queue name `aipet-training`.
5. Write `src/cli/trigger_training.py` — thin CLI that accepts `--experiment-name`, `--epochs`, `--patience`, `--skip-generate`, connects to Temporal, and starts a `TrainingPipelineWorkflow` execution; prints the workflow ID for tracking.
6. Update `docker-compose.yml` to add a `temporal` service (using `temporalio/auto-setup` image) and a `temporal-worker` service that runs `python -m src.temporal.worker`; wire `TEMPORAL_HOST=temporal:7233`.
7. Write unit tests in `tests/unit/test_temporal_activities.py` mocking the domain functions to verify each activity delegates correctly and surfaces errors as `ApplicationError`.
8. Document experiment triggering and how to inspect workflow history via the Temporal UI (`localhost:8233`) in `README.md`.
---
