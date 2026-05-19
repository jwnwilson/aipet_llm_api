import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Pencil, Play, Plus, Trash2 } from 'lucide-react'
import { deleteModel, listModels } from '@/api/models'
import { RunModal } from '@/components/RunModal'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { TrainingModel } from '@/types'

export function ModelsListPage() {
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
  const [runTarget, setRunTarget] = useState<TrainingModel | null>(null)

  const { data: models = [], isLoading } = useQuery({
    queryKey: ['models'],
    queryFn: listModels,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['models'] }),
  })

  const filtered = models.filter(m => {
    const q = search.toLowerCase()
    return (
      m.name.toLowerCase().includes(q) ||
      m.description.toLowerCase().includes(q) ||
      m.base_model.toLowerCase().includes(q)
    )
  })

  if (isLoading) return <p className="p-8 text-gray-500">Loading…</p>

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold">Training Models</h1>
        <Button asChild>
          <Link to="/models/new">
            <Plus className="h-4 w-4 mr-1" />New model
          </Link>
        </Button>
      </div>

      {models.length === 0 ? (
        <div className="text-center py-16 text-gray-500">
          <p className="mb-4">No models yet.</p>
          <Button asChild variant="outline">
            <Link to="/models/new">Create your first model</Link>
          </Button>
        </div>
      ) : (
        <>
          <div className="mb-4">
            <Input
              className="max-w-xs"
              placeholder="Search by name, description, base model…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              aria-label="Search models"
            />
          </div>

          <div className="rounded-md border bg-white overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-gray-50 text-gray-500 text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-3 font-semibold">Name</th>
                  <th className="text-left px-4 py-3 font-semibold">Base model</th>
                  <th className="text-left px-4 py-3 font-semibold">Backend</th>
                  <th className="text-left px-4 py-3 font-semibold">Epochs</th>
                  <th className="text-left px-4 py-3 font-semibold">Active</th>
                  <th className="text-left px-4 py-3 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="text-center py-8 text-gray-400">
                      No models match "{search}"
                    </td>
                  </tr>
                ) : (
                  filtered.map(model => (
                    <tr key={model.id} className="border-b last:border-0 hover:bg-gray-50">
                      <td className="px-4 py-3">
                        <div className="font-medium text-gray-900">{model.name}</div>
                        {model.description && (
                          <div className="text-xs text-gray-400 mt-0.5">{model.description}</div>
                        )}
                      </td>
                      <td className="px-4 py-3 font-mono text-gray-700 text-xs">{model.base_model}</td>
                      <td className="px-4 py-3 text-gray-700">{model.remote_backend}</td>
                      <td className="px-4 py-3 text-gray-700">{model.epochs}</td>
                      <td className="px-4 py-3">
                        {model.is_active ? (
                          <span className="inline-flex items-center rounded-full bg-green-100 text-green-700 px-2 py-0.5 text-xs font-medium">
                            Active
                          </span>
                        ) : (
                          <span className="text-gray-300">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            onClick={() => setRunTarget(model)}
                            aria-label={`Trigger run for ${model.name}`}
                          >
                            <Play className="h-3.5 w-3.5 mr-1" />Run
                          </Button>
                          <Button size="sm" variant="outline" asChild>
                            <Link
                              to={`/models/${model.id}/edit`}
                              aria-label={`Edit ${model.name}`}
                            >
                              <Pencil className="h-3.5 w-3.5 mr-1" />Edit
                            </Link>
                          </Button>
                          <Button
                            size="sm"
                            variant="destructive"
                            onClick={() => deleteMutation.mutate(model.id)}
                            disabled={
                              deleteMutation.isPending &&
                              deleteMutation.variables === model.id
                            }
                            aria-label={`Delete ${model.name}`}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {runTarget && (
        <RunModal model={runTarget} onClose={() => setRunTarget(null)} />
      )}
    </div>
  )
}
