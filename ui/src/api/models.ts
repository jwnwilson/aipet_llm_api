import type { TrainingModel, TrainingModelConfig } from '@/types'
import { apiClient } from './client'

export async function listModels(): Promise<TrainingModel[]> {
  const { data } = await apiClient.get<TrainingModel[]>('/api/models')
  return data
}

export async function getModel(id: string): Promise<TrainingModel> {
  const { data } = await apiClient.get<TrainingModel>(`/api/models/${id}`)
  return data
}

export async function createModel(config: TrainingModelConfig): Promise<TrainingModel> {
  const { data } = await apiClient.post<TrainingModel>('/api/models', config)
  return data
}

export async function updateModel(id: string, config: TrainingModelConfig): Promise<TrainingModel> {
  const { data } = await apiClient.put<TrainingModel>(`/api/models/${id}`, config)
  return data
}

export async function deleteModel(id: string): Promise<void> {
  await apiClient.delete(`/api/models/${id}`)
}
