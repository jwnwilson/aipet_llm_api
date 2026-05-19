// src/types/index.ts
export interface TrainingModelConfig {
  name: string
  description: string
  base_model: string
  train_data: string
  eval_data: string
  epochs: number
  patience: number
  warmup_ratio: number
  remote_backend: string
  skip_generate: boolean
  gguf_path?: string   // optional — backend defaults to ''
  is_active?: boolean  // optional — backend defaults to false
}

export interface TrainingModel extends TrainingModelConfig {
  id: string
  created_at: string
  updated_at: string
}

export type RunStatus =
  | 'pending'
  | 'generating'
  | 'training'
  | 'evaluating'
  | 'exporting'
  | 'running'
  | 'completed'
  | 'failed'

export interface RunRecord {
  id: string
  workflow_id: string
  model_id: string
  status: RunStatus
  eval_valid_pct: number | null
  progress: number | null
  progress_detail: string | null
  created_at: string
  updated_at: string
}

export interface TriggerRunRequest {
  model_id: string
  epochs?: number | null
  patience?: number | null
  warmup_ratio?: number | null
  skip_generate?: boolean | null
  remote_backend?: string | null
  base_model?: string | null
  num_train_samples?: number | null
  num_eval_samples?: number | null
}

export interface UserContext {
  user_id: string
  email: string | null
  status: 'pending' | 'approved'
}
