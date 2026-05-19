import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { ModelCard } from '@/components/ModelCard'
import { MODEL_FIXTURE } from '../msw/fixtures'

function renderCard(props?: Partial<Parameters<typeof ModelCard>[0]>) {
  const onTrigger = vi.fn()
  render(
    <MemoryRouter>
      <ModelCard model={MODEL_FIXTURE} onTrigger={onTrigger} {...props} />
    </MemoryRouter>
  )
  return { onTrigger }
}

describe('ModelCard', () => {
  it('renders model name', () => {
    renderCard()
    expect(screen.getByText(MODEL_FIXTURE.name)).toBeInTheDocument()
  })

  it('renders model description', () => {
    renderCard()
    expect(screen.getByText(MODEL_FIXTURE.description)).toBeInTheDocument()
  })

  it('calls onTrigger with model id when Run button clicked', async () => {
    const { onTrigger } = renderCard()
    await userEvent.click(screen.getByRole('button', { name: /trigger training run/i }))
    expect(onTrigger).toHaveBeenCalledWith(MODEL_FIXTURE.id)
  })

  it('disables run button while triggering', () => {
    renderCard({ isTriggering: true })
    expect(screen.getByRole('button', { name: /trigger training run/i })).toBeDisabled()
  })
})
