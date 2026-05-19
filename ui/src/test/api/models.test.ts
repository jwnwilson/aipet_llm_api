import { describe, it, expect } from 'vitest'
import { listModels, getModel, createModel, updateModel, deleteModel } from '@/api/models'
import { MODEL_FIXTURE } from '../msw/fixtures'

describe('listModels', () => {
  it('returns array of models', async () => {
    const models = await listModels()
    expect(Array.isArray(models)).toBe(true)
    expect(models[0].id).toBe(MODEL_FIXTURE.id)
  })
})

describe('getModel', () => {
  it('returns model by id', async () => {
    const model = await getModel(MODEL_FIXTURE.id)
    expect(model.name).toBe(MODEL_FIXTURE.name)
  })

  it('throws on unknown id', async () => {
    await expect(getModel('does-not-exist')).rejects.toThrow()
  })
})

describe('createModel', () => {
  it('creates and returns model with id', async () => {
    const config = { ...MODEL_FIXTURE, name: 'new-model' }
    const model = await createModel(config)
    expect(model.id).toBeDefined()
    expect(model.name).toBe('new-model')
  })
})

describe('updateModel', () => {
  it('updates and returns model', async () => {
    const updated = await updateModel(MODEL_FIXTURE.id, { ...MODEL_FIXTURE, epochs: 10 })
    expect(updated.epochs).toBe(10)
  })
})

describe('deleteModel', () => {
  it('resolves without error', async () => {
    await expect(deleteModel(MODEL_FIXTURE.id)).resolves.toBeUndefined()
  })
})
