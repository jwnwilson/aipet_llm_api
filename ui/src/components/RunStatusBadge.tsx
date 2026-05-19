import type { RunStatus } from '@/types'
import { cn } from '@/lib/utils'

const STATUS_CONFIG: Record<RunStatus, { label: string; className: string }> = {
  pending:    { label: 'Pending',    className: 'bg-gray-100 text-gray-600' },
  generating: { label: 'Generating', className: 'bg-purple-100 text-purple-800' },
  training:   { label: 'Training',   className: 'bg-blue-100 text-blue-800' },
  evaluating: { label: 'Evaluating', className: 'bg-indigo-100 text-indigo-800' },
  exporting:  { label: 'Exporting',  className: 'bg-teal-100 text-teal-800' },
  running:    { label: 'Running',    className: 'bg-blue-100 text-blue-800' },
  completed:  { label: 'Completed',  className: 'bg-green-100 text-green-800' },
  failed:     { label: 'Failed',     className: 'bg-red-100 text-red-800' },
}

interface RunStatusBadgeProps {
  status: RunStatus
  className?: string
}

export function RunStatusBadge({ status, className }: RunStatusBadgeProps) {
  const config = STATUS_CONFIG[status]
  return (
    <span
      data-testid="run-status-badge"
      className={cn('inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium', config.className, className)}
    >
      {config.label}
    </span>
  )
}
