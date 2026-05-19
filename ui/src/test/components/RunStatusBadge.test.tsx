import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import type { RunStatus } from '@/types'

const cases: Array<[RunStatus, string]> = [
  ['pending', 'Pending'],
  ['generating', 'Generating'],
  ['training', 'Training'],
  ['evaluating', 'Evaluating'],
  ['exporting', 'Exporting'],
  ['running', 'Running'],
  ['completed', 'Completed'],
  ['failed', 'Failed'],
]

describe('RunStatusBadge', () => {
  it.each(cases)('renders label for status %s', (status, label) => {
    render(<RunStatusBadge status={status} />)
    expect(screen.getByText(label)).toBeInTheDocument()
  })

  it('applies green class for completed', () => {
    render(<RunStatusBadge status="completed" />)
    expect(screen.getByTestId('run-status-badge')).toHaveClass('bg-green-100')
  })

  it('applies red class for failed', () => {
    render(<RunStatusBadge status="failed" />)
    expect(screen.getByTestId('run-status-badge')).toHaveClass('bg-red-100')
  })

  it('applies blue class for running', () => {
    render(<RunStatusBadge status="running" />)
    expect(screen.getByTestId('run-status-badge')).toHaveClass('bg-blue-100')
  })
})
