import { CheckCircle, XCircle } from 'lucide-react'

interface EvalMetricsProps {
  validPct: number
  passed: boolean
}

export function EvalMetrics({ validPct, passed }: EvalMetricsProps) {
  const pctDisplay = (validPct * 100).toFixed(1)
  return (
    <div className="flex items-center gap-3 rounded-md border p-3">
      {passed
        ? <CheckCircle className="h-5 w-5 text-green-600" />
        : <XCircle className="h-5 w-5 text-red-600" />
      }
      <div>
        <p className="text-sm font-medium">Eval score: {pctDisplay}%</p>
        <p className="text-xs text-gray-500">{passed ? 'Passed (≥95%)' : 'Failed (<95%)'}</p>
      </div>
    </div>
  )
}
