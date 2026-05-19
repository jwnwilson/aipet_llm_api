import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { deleteModel, getModel } from '@/api/models'
import { listRuns, triggerRun } from '@/api/runs'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, Pencil, Trash2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

export function ModelDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: model, isLoading } = useQuery({
    queryKey: ['models', id],
    queryFn: () => getModel(id!),
  })

  const { data: allRuns = [] } = useQuery({ queryKey: ['runs'], queryFn: listRuns })
  const runs = allRuns.filter(r => r.model_id === id)

  const triggerMutation = useMutation({
    mutationFn: () => triggerRun({ model_id: id! }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['runs'] }),
  })

  const deleteMutation = useMutation({
    mutationFn: () => deleteModel(id!),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['models'] }); navigate('/models') },
  })

  if (isLoading || !model) return <p className="p-8 text-gray-500">Loading…</p>

  return (
    <div className="p-8 max-w-3xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold">{model.name}</h1>
          {model.description && <p className="text-gray-500 mt-1">{model.description}</p>}
        </div>
        <div className="flex gap-2">
          <Button onClick={() => triggerMutation.mutate()} disabled={triggerMutation.isPending}>
            <Play className="h-4 w-4 mr-1" />
            {triggerMutation.isPending ? 'Starting…' : 'Run'}
          </Button>
          <Button variant="outline" asChild>
            <Link to={`/models/${id}/edit`}><Pencil className="h-4 w-4 mr-1" />Edit</Link>
          </Button>
          <Button
            variant="destructive"
            onClick={() => deleteMutation.mutate()}
            disabled={deleteMutation.isPending}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Card className="mb-6">
        <CardHeader><CardTitle>Configuration</CardTitle></CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
            {[
              ['Base model', model.base_model],
              ['Training data', model.train_data],
              ['Eval data', model.eval_data],
              ['Epochs', model.epochs],
              ['Patience', model.patience],
              ['Warmup ratio', model.warmup_ratio],
              ['Remote backend', model.remote_backend],
              ['Skip generate', model.skip_generate ? 'Yes' : 'No'],
              ...(model.gguf_path ? [['GGUF path', model.gguf_path]] : []),
            ].map(([key, val]) => (
              <div key={String(key)} className="contents">
                <dt className="text-gray-500">{key}</dt>
                <dd className="font-medium text-gray-900">{String(val)}</dd>
              </div>
            ))}
          </dl>
        </CardContent>
      </Card>

      <h2 className="text-lg font-medium mb-3">Recent runs</h2>
      {runs.length === 0 ? (
        <p className="text-gray-500 text-sm">No runs yet.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {runs.map(run => (
            <Link
              key={run.id}
              to={`/runs/${run.id}`}
              className="flex items-center justify-between rounded-md border p-3 text-gray-900 hover:bg-gray-50"
            >
              <span className="font-mono text-sm truncate">{run.workflow_id}</span>
              <RunStatusBadge status={run.status} />
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
