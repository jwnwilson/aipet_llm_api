import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RunModal } from '@/components/RunModal'
import { MODEL_FIXTURE } from '../msw/fixtures'

function renderModal(onClose = vi.fn()) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <RunModal model={MODEL_FIXTURE} onClose={onClose} />
    </QueryClientProvider>
  )
  return { onClose }
}

describe('RunModal', () => {
  it('renders with model name in the heading', () => {
    renderModal()
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(screen.getByText(/trigger run — test-model/i)).toBeInTheDocument()
  })

  it('pre-fills epochs with the model default', () => {
    renderModal()
    const input = screen.getByLabelText(/^epochs$/i)
    expect(input).toHaveValue(MODEL_FIXTURE.epochs)
  })

  it('closes on successful submission', async () => {
    const { onClose } = renderModal()
    await userEvent.click(screen.getByRole('button', { name: /start run/i }))
    await waitFor(() => expect(onClose).toHaveBeenCalled())
  })

  it('calls onClose when Cancel is clicked', async () => {
    const { onClose } = renderModal()
    await userEvent.click(screen.getByRole('button', { name: /cancel/i }))
    expect(onClose).toHaveBeenCalled()
  })

  it('renders num_train_samples and num_eval_samples fields', () => {
    renderModal()
    expect(screen.getByLabelText(/train samples/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/eval samples/i)).toBeInTheDocument()
  })

  it('num_train_samples and num_eval_samples start enabled when skip_generate is false', () => {
    renderModal()
    expect(screen.getByLabelText(/train samples/i)).not.toBeDisabled()
    expect(screen.getByLabelText(/eval samples/i)).not.toBeDisabled()
  })

  it('disables num_train_samples and num_eval_samples when skip_generate is checked', async () => {
    renderModal()
    const skipCheckbox = screen.getByLabelText(/skip dataset generation/i)
    await userEvent.click(skipCheckbox)
    expect(screen.getByLabelText(/train samples/i)).toBeDisabled()
    expect(screen.getByLabelText(/eval samples/i)).toBeDisabled()
  })

  it('re-enables num_train_samples and num_eval_samples when skip_generate is unchecked', async () => {
    renderModal()
    const skipCheckbox = screen.getByLabelText(/skip dataset generation/i)
    await userEvent.click(skipCheckbox) // check
    await userEvent.click(skipCheckbox) // uncheck
    expect(screen.getByLabelText(/train samples/i)).not.toBeDisabled()
    expect(screen.getByLabelText(/eval samples/i)).not.toBeDisabled()
  })
})
