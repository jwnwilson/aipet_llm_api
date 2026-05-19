import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { createModel, getModel, updateModel } from '@/api/models'
import { ModelForm } from '@/components/ModelForm'
import type { TrainingModelConfig } from '@/types'

export function ModelFormPage() {
  const { id } = useParams<{ id: string }>()
  const isEdit = Boolean(id)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: existing, isLoading } = useQuery({
    queryKey: ['models', id],
    queryFn: () => getModel(id!),
    enabled: isEdit,
  })

  const mutation = useMutation({
    mutationFn: (values: TrainingModelConfig) =>
      isEdit ? updateModel(id!, values) : createModel(values),
    onSuccess: (model) => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
      navigate(`/models/${model.id}`)
    },
  })

  if (isEdit && isLoading) return <p className="p-8 text-gray-500">Loading…</p>

  return (
    <div className="p-8 max-w-2xl">
      <h1 className="text-2xl font-semibold mb-6">{isEdit ? 'Edit model' : 'New model'}</h1>
      <ModelForm
        defaultValues={existing}
        onSubmit={mutation.mutate}
        isSubmitting={mutation.isPending}
      />
      {mutation.isError && (
        <p className="mt-4 text-sm text-red-600">Failed to save. Please try again.</p>
      )}
    </div>
  )
}
