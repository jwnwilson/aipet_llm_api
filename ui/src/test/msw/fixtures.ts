// apps/llm-ui/src/test/msw/fixtures.ts
import type { TrainingModel, RunRecord, UserContext } from '@/types'

export const MODEL_FIXTURE: TrainingModel = {
  id: 'test-id-1',
  name: 'test-model',
  description: 'A test model',
  base_model: 'HuggingFaceTB/SmolLM2-360M',
  train_data: 'data/train.jsonl',
  eval_data: 'data/eval.jsonl',
  epochs: 5,
  patience: 3,
  warmup_ratio: 0.05,
  remote_backend: 'local',
  skip_generate: false,
  gguf_path: '',
  is_active: false,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
}

export const RUN_FIXTURE: RunRecord = {
  id: 'run-uuid',
  workflow_id: 'training-test-model-abc12345',
  model_id: 'test-id-1',
  status: 'running',
  eval_valid_pct: null,
  progress: null,
  progress_detail: null,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
}

export const PENDING_USER_FIXTURE: UserContext = {
  user_id: 'auth0|pending-user',
  email: 'pending@example.com',
  status: 'pending',
}

export const APPROVED_USER_FIXTURE: UserContext = {
  user_id: 'auth0|approved-user',
  email: 'approved@example.com',
  status: 'approved',
}
