// apps/llm-ui/src/test/msw/handlers.ts
import { http, HttpResponse } from 'msw'
import type { TrainingModel, TrainingModelConfig, TriggerRunRequest, UserContext } from '@/types'
import { MODEL_FIXTURE, RUN_FIXTURE, PENDING_USER_FIXTURE, APPROVED_USER_FIXTURE } from './fixtures'

const BASE = 'http://localhost:8000'

let models: TrainingModel[] = [MODEL_FIXTURE]
let pendingUsers: UserContext[] = [PENDING_USER_FIXTURE]
let approvedUsers: UserContext[] = [APPROVED_USER_FIXTURE]

export const handlers = [
  http.get(`${BASE}/api/models`, () => HttpResponse.json(models)),

  http.post(`${BASE}/api/models`, async ({ request }) => {
    const config = await request.json() as TrainingModelConfig
    const created: TrainingModel = {
      ...config,
      id: 'new-id',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }
    models = [...models, created]
    return HttpResponse.json(created, { status: 201 })
  }),

  http.get(`${BASE}/api/models/:id`, ({ params }) => {
    const model = models.find(m => m.id === params.id)
    if (!model) return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    return HttpResponse.json(model)
  }),

  http.put(`${BASE}/api/models/:id`, async ({ params, request }) => {
    const config = await request.json() as TrainingModelConfig
    const idx = models.findIndex(m => m.id === params.id)
    if (idx === -1) return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    const updated = { ...models[idx], ...config, updated_at: new Date().toISOString() }
    models = [...models.slice(0, idx), updated, ...models.slice(idx + 1)]
    return HttpResponse.json(updated)
  }),

  http.delete(`${BASE}/api/models/:id`, ({ params }) => {
    const idx = models.findIndex(m => m.id === params.id)
    if (idx === -1) return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    models = models.filter(m => m.id !== params.id)
    return new HttpResponse(null, { status: 204 })
  }),

  http.post(`${BASE}/api/runs/trigger`, async ({ request }) => {
    const body = await request.json() as TriggerRunRequest
    const model = models.find(m => m.id === body.model_id)
    if (!model) return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
    return HttpResponse.json({ run_id: RUN_FIXTURE.id }, { status: 202 })
  }),

  http.get(`${BASE}/api/runs`, () => HttpResponse.json([RUN_FIXTURE])),

  http.get(`${BASE}/api/runs/:id`, ({ params }) => {
    if (params.id === RUN_FIXTURE.id) return HttpResponse.json(RUN_FIXTURE)
    return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  }),

  http.delete(`${BASE}/api/runs/:id`, ({ params }) => {
    if (params.id === RUN_FIXTURE.id) return new HttpResponse(null, { status: 204 })
    return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  }),

  http.get(`${BASE}/api/admin/users`, ({ request }) => {
    const url = new URL(request.url)
    const status = url.searchParams.get('status') ?? 'approved'
    return HttpResponse.json(status === 'pending' ? pendingUsers : approvedUsers)
  }),

  http.post(`${BASE}/api/admin/users`, async ({ request }) => {
    const body = await request.json() as { user_id: string; email?: string | null }
    const user: UserContext = { user_id: body.user_id, email: body.email ?? null, status: 'approved' }
    approvedUsers = [...approvedUsers, user]
    pendingUsers = pendingUsers.filter(u => u.user_id !== body.user_id)
    return HttpResponse.json({ approved: body.user_id }, { status: 201 })
  }),

  http.delete(`${BASE}/api/admin/users/:userId`, ({ params }) => {
    approvedUsers = approvedUsers.filter(
      u => u.user_id !== decodeURIComponent(params.userId as string)
    )
    return new HttpResponse(null, { status: 204 })
  }),
]

export function resetHandlerState() {
  models = [MODEL_FIXTURE]
  pendingUsers = [PENDING_USER_FIXTURE]
  approvedUsers = [APPROVED_USER_FIXTURE]
}
