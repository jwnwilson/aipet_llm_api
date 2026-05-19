import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { AccessPending } from '@/components/AccessPending'

describe('AccessPending', () => {
  it('renders access pending message', () => {
    render(<AccessPending />)
    expect(screen.getByText(/access pending/i)).toBeInTheDocument()
  })

  it('renders contact administrator message', () => {
    render(<AccessPending />)
    expect(screen.getByText(/contact an administrator/i)).toBeInTheDocument()
  })

  it('renders a refresh button', () => {
    render(<AccessPending />)
    expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument()
  })
})
