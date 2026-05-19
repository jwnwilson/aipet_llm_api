import { describe, it, expect } from 'vitest'
import { listRuns, getRun, isRunActive, triggerRun, deleteRun } from '@/api/runs'
import { MODEL_FIXTURE, RUN_FIXTURE } from '../msw/fixtures'

describe('listRuns', () => {
  it('returns array of RunRecords with id field', async () => {
    const runs = await listRuns()
    expect(Array.isArray(runs)).toBe(true)
    expect(runs[0].id).toBe(RUN_FIXTURE.id)
  })
})

describe('getRun', () => {
  it('returns run by id', async () => {
    const run = await getRun(RUN_FIXTURE.id)
    expect(run.status).toBe('running')
    expect(run.model_id).toBe(MODEL_FIXTURE.id)
  })

  it('throws on unknown id', async () => {
    await expect(getRun('does-not-exist')).rejects.toThrow()
  })
})

describe('triggerRun', () => {
  it('posts to /api/runs/trigger and returns run_id', async () => {
    const result = await triggerRun({ model_id: MODEL_FIXTURE.id })
    expect(result.run_id).toBe(RUN_FIXTURE.id)
  })
})

describe('isRunActive', () => {
  it('returns true for running status', () => {
    expect(isRunActive({ ...RUN_FIXTURE, status: 'running' })).toBe(true)
  })

  it('returns false for completed status', () => {
    expect(isRunActive({ ...RUN_FIXTURE, status: 'completed' })).toBe(false)
  })

  it('returns false for failed status', () => {
    expect(isRunActive({ ...RUN_FIXTURE, status: 'failed' })).toBe(false)
  })
})

describe('deleteRun', () => {
  it('resolves for an existing run id', async () => {
    await expect(deleteRun(RUN_FIXTURE.id)).resolves.toBeUndefined()
  })

  it('throws for an unknown run id', async () => {
    await expect(deleteRun('does-not-exist')).rejects.toThrow()
  })
})
