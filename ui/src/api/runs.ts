import type { RunRecord, TriggerRunRequest } from '@/types'
import { apiClient } from './client'

export async function listRuns(): Promise<RunRecord[]> {
  const { data } = await apiClient.get<RunRecord[]>('/api/runs')
  return data
}

export async function getRun(id: string): Promise<RunRecord> {
  const { data } = await apiClient.get<RunRecord>(`/api/runs/${id}`)
  return data
}

export async function triggerRun(req: TriggerRunRequest): Promise<{ run_id: string }> {
  const { data } = await apiClient.post<{ run_id: string }>('/api/runs/trigger', req)
  return data
}

export async function deleteRun(id: string): Promise<void> {
  await apiClient.delete(`/api/runs/${id}`)
}

export function isRunActive(run: RunRecord): boolean {
  return run.status === 'running'
}
