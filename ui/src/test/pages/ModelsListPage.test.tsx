import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ModelsListPage } from '@/pages/ModelsListPage'
import { MODEL_FIXTURE } from '../msw/fixtures'

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <ModelsListPage />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('ModelsListPage', () => {
  it('renders model name in table after loading', async () => {
    renderPage()
    await waitFor(() => expect(screen.getByText(MODEL_FIXTURE.name)).toBeInTheDocument())
  })

  it('renders New model link to /models/new', async () => {
    renderPage()
    await waitFor(() => {
      const link = screen.getByRole('link', { name: /new model/i })
      expect(link).toHaveAttribute('href', '/models/new')
    })
  })

  it('renders search input', async () => {
    renderPage()
    await waitFor(() => expect(screen.getByRole('textbox', { name: /search/i })).toBeInTheDocument())
  })

  it('hides rows that do not match the search query', async () => {
    renderPage()
    await waitFor(() => screen.getByText(MODEL_FIXTURE.name))
    await userEvent.type(screen.getByRole('textbox', { name: /search/i }), 'zzznomatch')
    expect(screen.queryByText(MODEL_FIXTURE.name)).not.toBeInTheDocument()
  })

  it('opens RunModal when Run button is clicked', async () => {
    renderPage()
    await waitFor(() => screen.getByText(MODEL_FIXTURE.name))
    await userEvent.click(
      screen.getByRole('button', { name: new RegExp(`trigger run for ${MODEL_FIXTURE.name}`, 'i') })
    )
    expect(screen.getByRole('dialog')).toBeInTheDocument()
  })
})
