import { afterEach, describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { http, HttpResponse } from 'msw'
import { RunDetailPage } from '@/pages/RunDetailPage'
import { RUN_FIXTURE } from '../msw/fixtures'
import { server } from '../msw/server'

function renderPage(runId: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/runs/${runId}`]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/runs" element={<div>runs-list</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('RunDetailPage', () => {
  afterEach(() => vi.restoreAllMocks())

  it('renders workflow_id and status badge', async () => {
    renderPage(RUN_FIXTURE.id)
    await waitFor(() => {
      expect(screen.getByText(RUN_FIXTURE.workflow_id)).toBeInTheDocument()
      expect(screen.getByText('Running')).toBeInTheDocument()
    })
  })

  it('shows not found for unknown run id', async () => {
    renderPage('does-not-exist')
    await waitFor(() => expect(screen.getByText(/not found/i)).toBeInTheDocument())
  })

  it('renders a Delete run button', async () => {
    renderPage(RUN_FIXTURE.id)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /delete run/i })).toBeInTheDocument()
    )
  })

  it('navigates to /runs after confirming delete', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    renderPage(RUN_FIXTURE.id)
    await waitFor(() => screen.getByRole('button', { name: /delete run/i }))
    await userEvent.click(screen.getByRole('button', { name: /delete run/i }))
    await waitFor(() => expect(screen.getByText('runs-list')).toBeInTheDocument())
  })

  it('stays on the page when delete is cancelled', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false)
    renderPage(RUN_FIXTURE.id)
    await waitFor(() => screen.getByRole('button', { name: /delete run/i }))
    await userEvent.click(screen.getByRole('button', { name: /delete run/i }))
    expect(screen.queryByText('runs-list')).not.toBeInTheDocument()
    expect(screen.getByText(RUN_FIXTURE.workflow_id)).toBeInTheDocument()
  })

  it('shows error message when delete fails', async () => {
    server.use(
      http.delete('http://localhost:8000/api/runs/:id', () =>
        HttpResponse.json({ detail: 'Server error' }, { status: 500 })
      )
    )
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    renderPage(RUN_FIXTURE.id)
    await waitFor(() => screen.getByRole('button', { name: /delete run/i }))
    await userEvent.click(screen.getByRole('button', { name: /delete run/i }))
    await waitFor(() =>
      expect(screen.getByText(/failed to delete run/i)).toBeInTheDocument()
    )
  })
})
