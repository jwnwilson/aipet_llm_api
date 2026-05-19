import { CheckCircle, Circle, XCircle, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

export type StageStatus = 'pending' | 'active' | 'completed' | 'failed'

export interface PipelineStage {
  name: string
  status: StageStatus
}

interface PipelineStagesProps {
  stages: PipelineStage[]
}

function StageIcon({ status }: { status: StageStatus }) {
  if (status === 'completed') return <CheckCircle className="h-5 w-5 text-green-600" />
  if (status === 'failed') return <XCircle className="h-5 w-5 text-red-600" />
  if (status === 'active') return <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
  return <Circle className="h-5 w-5 text-gray-300" />
}

export function PipelineStages({ stages }: PipelineStagesProps) {
  return (
    <div className="flex items-center gap-2">
      {stages.map((stage, i) => (
        <div key={stage.name} className="flex items-center gap-2">
          <div
            data-testid={`stage-${stage.name.toLowerCase().replace(/\s+/g, '-')}`}
            className={cn('flex flex-col items-center gap-1', {
              'opacity-40': stage.status === 'pending',
            })}
          >
            <StageIcon status={stage.status} />
            <span className="text-xs font-medium text-gray-600">{stage.name}</span>
          </div>
          {i < stages.length - 1 && (
            <div className="h-px w-8 bg-gray-200" />
          )}
        </div>
      ))}
    </div>
  )
}
