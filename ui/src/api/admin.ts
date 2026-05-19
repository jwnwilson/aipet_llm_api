import type { UserContext } from '@/types'
import { apiClient } from './client'

export async function listUsers(status: 'approved' | 'pending' = 'approved'): Promise<UserContext[]> {
  const { data } = await apiClient.get<UserContext[]>('/api/admin/users', { params: { status } })
  return data
}

export async function approveUser(user_id: string, email?: string | null): Promise<void> {
  await apiClient.post('/api/admin/users', { user_id, email })
}

export async function revokeUser(user_id: string): Promise<void> {
  await apiClient.delete(`/api/admin/users/${encodeURIComponent(user_id)}`)
}
