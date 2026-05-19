import { describe, it, expect, afterEach } from 'vitest'
import { http, HttpResponse } from 'msw'
import { setTokenGetter, apiClient } from '@/api/client'
import { server } from '../msw/server'

afterEach(() => {
  // Reset to no token getter between tests
  setTokenGetter(null)
})

describe('setTokenGetter', () => {
  it('is exported from api/client', () => {
    expect(typeof setTokenGetter).toBe('function')
  })
})

describe('auth interceptor', () => {
  it('injects Authorization header when token getter is set', async () => {
    let capturedAuth: string | null = null
    server.use(
      http.get('http://localhost:8000/api/models', ({ request }) => {
        capturedAuth = request.headers.get('authorization')
        return HttpResponse.json([])
      })
    )
    setTokenGetter(() => Promise.resolve('my-jwt'))
    await apiClient.get('/api/models')
    expect(capturedAuth).toBe('Bearer my-jwt')
  })

  it('does not inject Authorization header when token getter is null', async () => {
    let capturedAuth: string | null | undefined = undefined
    server.use(
      http.get('http://localhost:8000/api/models', ({ request }) => {
        capturedAuth = request.headers.get('authorization')
        return HttpResponse.json([])
      })
    )
    // token getter is already null (afterEach resets it)
    await apiClient.get('/api/models')
    expect(capturedAuth).toBeNull()
  })

  it('sends request without auth header when token getter throws', async () => {
    let capturedAuth: string | null | undefined = undefined
    server.use(
      http.get('http://localhost:8000/api/models', ({ request }) => {
        capturedAuth = request.headers.get('authorization')
        return HttpResponse.json([])
      })
    )
    setTokenGetter(() => Promise.reject(new Error('token refresh failed')))
    await apiClient.get('/api/models')
    expect(capturedAuth).toBeNull()
  })
})

describe('403 response interceptor', () => {
  it('dispatches auth:access-denied event on 403 response', async () => {
    server.use(
      http.get('http://localhost:8000/api/models', () =>
        new HttpResponse(null, { status: 403 })
      )
    )
    let eventFired = false
    const handler = () => { eventFired = true }
    window.addEventListener('auth:access-denied', handler)
    try {
      await apiClient.get('/api/models').catch(() => {})
    } finally {
      window.removeEventListener('auth:access-denied', handler)
    }
    expect(eventFired).toBe(true)
  })

  it('does not dispatch event on non-403 errors', async () => {
    server.use(
      http.get('http://localhost:8000/api/models', () =>
        new HttpResponse(null, { status: 401 })
      )
    )
    let eventFired = false
    const handler = () => { eventFired = true }
    window.addEventListener('auth:access-denied', handler)
    try {
      await apiClient.get('/api/models').catch(() => {})
    } finally {
      window.removeEventListener('auth:access-denied', handler)
    }
    expect(eventFired).toBe(false)
  })

  it('still rejects the promise on 403', async () => {
    server.use(
      http.get('http://localhost:8000/api/models', () =>
        new HttpResponse(null, { status: 403 })
      )
    )
    await expect(apiClient.get('/api/models')).rejects.toThrow()
  })
})
