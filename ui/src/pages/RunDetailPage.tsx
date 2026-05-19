import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { deleteRun, getRun, isRunActive } from '@/api/runs'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import { PipelineStages } from '@/components/PipelineStages'
import type { PipelineStage, StageStatus } from '@/components/PipelineStages'
import type { RunStatus } from '@/types'

function buildStages(status: RunStatus): PipelineStage[] {
  const stageNames = ['Generate', 'Train', 'Evaluate', 'Export']
  const activeMap: Partial<Record<RunStatus, number>> = {
    generating: 0,
    training:   1,
    evaluating: 2,
    exporting:  3,
  }

  if (status === 'completed') {
    return stageNames.map(name => ({ name, status: 'completed' as StageStatus }))
  }
  if (status === 'failed') {
    return stageNames.map((name, i): PipelineStage => ({
      name,
      status: i === 0 ? 'failed' : 'pending',
    }))
  }

  const activeIdx = activeMap[status] ?? -1
  return stageNames.map((name, i): PipelineStage => ({
    name,
    status: i < activeIdx ? 'completed' : i === activeIdx ? 'active' : 'pending',
  }))
}

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: run, isLoading } = useQuery({
    queryKey: ['runs', runId],
    queryFn: () => getRun(runId!),
    refetchInterval: (query) => {
      const data = query.state.data
      return data && isRunActive(data) ? 5000 : false
    },
  })

  const deleteMutation = useMutation({
    mutationFn: () => deleteRun(runId!),
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: ['runs', runId] })
      navigate('/runs')
    },
  })

  function handleDelete() {
    if (window.confirm('Delete this run? This cannot be undone.')) {
      deleteMutation.mutate()
    }
  }

  if (isLoading) return <p className="p-8 text-gray-500">Loading…</p>
  if (!run) return <p className="p-8 text-red-600">Run not found.</p>

  return (
    <div className="p-8 max-w-2xl">
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-xl font-semibold font-mono truncate">{run.workflow_id}</h1>
        <RunStatusBadge status={run.status} />
        <button
          onClick={handleDelete}
          disabled={deleteMutation.isPending}
          className="ml-auto text-sm text-red-600 border border-red-300 rounded px-3 py-1 hover:bg-red-50 disabled:opacity-50"
        >
          {deleteMutation.isPending ? 'Deleting…' : 'Delete run'}
        </button>
      </div>

      {deleteMutation.isError && (
        <p className="text-sm text-red-600 mb-4">Failed to delete run. Please try again.</p>
      )}

      <div className="mb-8 mt-6">
        <h2 className="text-sm font-medium text-gray-500 mb-3">Pipeline stages</h2>
        <PipelineStages stages={buildStages(run.status)} />
      </div>

      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
        <dt className="text-gray-500">Run ID</dt>
        <dd className="font-mono text-gray-900">{run.id}</dd>
        <dt className="text-gray-500">Started</dt>
        <dd className="text-gray-900">{new Date(run.created_at).toLocaleString()}</dd>
        <dt className="text-gray-500">Updated</dt>
        <dd className="text-gray-900">{new Date(run.updated_at).toLocaleString()}</dd>
        {run.progress != null && (
          <>
            <dt className="text-gray-500">Progress</dt>
            <dd className="text-gray-900">{Math.round(run.progress * 100)}%</dd>
          </>
        )}
        {run.eval_valid_pct != null && (
          <>
            <dt className="text-gray-500">Eval valid</dt>
            <dd className="text-gray-900">{Math.round(run.eval_valid_pct * 100)}%</dd>
          </>
        )}
        {run.progress_detail && (
          <>
            <dt className="text-gray-500">Detail</dt>
            <dd className="text-gray-900">{run.progress_detail}</dd>
          </>
        )}
      </dl>
    </div>
  )
}
