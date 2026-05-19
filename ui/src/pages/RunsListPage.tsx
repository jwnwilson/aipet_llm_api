import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { listRuns } from '@/api/runs'
import { RunStatusBadge } from '@/components/RunStatusBadge'

export function RunsListPage() {
  const { data: runs = [], isLoading } = useQuery({ queryKey: ['runs'], queryFn: listRuns })

  if (isLoading) return <p className="p-8 text-gray-500">Loading…</p>

  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold mb-6">Training Runs</h1>
      {runs.length === 0 ? (
        <p className="text-gray-500">No runs yet. Trigger one from a model.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {runs.map(run => (
            <Link
              key={run.id}
              to={`/runs/${run.id}`}
              className="flex items-center justify-between rounded-md border p-4 text-gray-900 hover:bg-gray-50"
            >
              <div>
                <p className="font-mono text-sm font-medium text-gray-900">{run.workflow_id}</p>
                <p className="text-xs text-gray-500 mt-0.5">
                  {new Date(run.created_at).toLocaleString()}
                </p>
              </div>
              <RunStatusBadge status={run.status} />
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
