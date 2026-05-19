import '@testing-library/jest-dom/vitest'
import { afterAll, afterEach, beforeAll } from 'vitest'
import { server } from './msw/server'
import { resetHandlerState } from './msw/handlers'

window.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} }
Element.prototype.scrollIntoView ??= () => {}
Element.prototype.hasPointerCapture ??= () => false

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => {
  server.resetHandlers()
  resetHandlerState()
})
afterAll(() => server.close())
