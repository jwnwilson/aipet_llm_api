import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { UsersPage } from '@/pages/UsersPage'
import { PENDING_USER_FIXTURE, APPROVED_USER_FIXTURE } from '../msw/fixtures'
import { resetHandlerState } from '../msw/handlers'

beforeEach(() => resetHandlerState())

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <UsersPage />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('UsersPage', () => {
  it('renders pending user email in awaiting approval table', async () => {
    renderPage()
    await waitFor(() =>
      expect(screen.getByText(PENDING_USER_FIXTURE.email!)).toBeInTheDocument()
    )
  })

  it('renders approved user email in approved users table', async () => {
    renderPage()
    await waitFor(() =>
      expect(screen.getByText(APPROVED_USER_FIXTURE.email!)).toBeInTheDocument()
    )
  })

  it('shows Approve button for pending user', async () => {
    renderPage()
    await waitFor(() =>
      expect(
        screen.getByRole('button', {
          name: new RegExp(`approve ${PENDING_USER_FIXTURE.email}`, 'i'),
        })
      ).toBeInTheDocument()
    )
  })

  it('shows Revoke button for approved user', async () => {
    renderPage()
    await waitFor(() =>
      expect(
        screen.getByRole('button', {
          name: new RegExp(`revoke ${APPROVED_USER_FIXTURE.email}`, 'i'),
        })
      ).toBeInTheDocument()
    )
  })

  it('approving a pending user removes them from the pending table', async () => {
    renderPage()
    await waitFor(() => screen.getByText(PENDING_USER_FIXTURE.email!))
    await userEvent.click(
      screen.getByRole('button', {
        name: new RegExp(`approve ${PENDING_USER_FIXTURE.email}`, 'i'),
      })
    )
    await waitFor(() => {
      const section = screen.getByRole('heading', { name: /awaiting approval/i }).closest('section')!
      expect(within(section).queryByText(PENDING_USER_FIXTURE.email!)).not.toBeInTheDocument()
    })
  })
})
