import { useForm, Controller } from 'react-hook-form'
import type { Resolver } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { X } from 'lucide-react'
import type { TrainingModel } from '@/types'
import { triggerRun } from '@/api/runs'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Combobox } from './ui/combobox'

const REMOTE_BACKEND_OPTIONS = ['local', 'kaggle', 'ssh', 'colab', 'runpod', 'vastai'] as const

const BASE_MODEL_OPTIONS = [
  'HuggingFaceTB/SmolLM2-360M',
  'HuggingFaceTB/SmolLM2-1.7B',
  'Qwen/Qwen2.5-0.5B',
  'Qwen/Qwen2.5-1.5B',
  'microsoft/phi-2',
  'google/gemma-2-2b',
  'meta-llama/Llama-3.2-1B',
  'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
]

const schema = z.object({
  epochs:             z.coerce.number().int().positive().nullable(),
  patience:           z.coerce.number().int().positive().nullable(),
  warmup_ratio:       z.coerce.number().min(0).max(1).nullable(),
  remote_backend:     z.string().nullable(),
  base_model:         z.string().nullable(),
  skip_generate:      z.boolean(),
  num_train_samples:  z.coerce.number().int().positive().nullable(),
  num_eval_samples:   z.coerce.number().int().positive().nullable(),
})

type FormValues = z.infer<typeof schema>

interface RunModalProps {
  model: TrainingModel
  onClose: () => void
}

export function RunModal({ model, onClose }: RunModalProps) {
  const queryClient = useQueryClient()

  const { register, handleSubmit, control, watch } = useForm<FormValues>({
    resolver: zodResolver(schema) as Resolver<FormValues>,
    defaultValues: {
      epochs:            model.epochs,
      patience:          model.patience,
      warmup_ratio:      model.warmup_ratio,
      remote_backend:    model.remote_backend,
      base_model:        model.base_model,
      skip_generate:     model.skip_generate,
      num_train_samples: null,
      num_eval_samples:  null,
    },
  })

  const skipGenerate = watch('skip_generate')

  const mutation = useMutation({
    mutationFn: triggerRun,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      onClose()
    },
  })

  function onSubmit(values: FormValues) {
    mutation.mutate({
      model_id: model.id,
      ...(values.epochs             != null && { epochs:            values.epochs }),
      ...(values.patience           != null && { patience:          values.patience }),
      ...(values.warmup_ratio       != null && { warmup_ratio:      values.warmup_ratio }),
      ...(values.remote_backend     != null && { remote_backend:    values.remote_backend }),
      ...(values.base_model         != null && { base_model:        values.base_model }),
      ...(!values.skip_generate && values.num_train_samples != null && { num_train_samples: values.num_train_samples }),
      ...(!values.skip_generate && values.num_eval_samples  != null && { num_eval_samples:  values.num_eval_samples }),
      skip_generate: values.skip_generate,
    })
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg p-6 w-full max-w-md shadow-xl"
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="run-modal-title"
      >
        <div className="flex items-center justify-between mb-2">
          <h2 id="run-modal-title" className="text-base font-semibold">
            Trigger run — {model.name}
          </h2>
          <button onClick={onClose} aria-label="Close" className="text-gray-400 hover:text-gray-600">
            <X className="h-4 w-4" />
          </button>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          Override config values for this run only. Leave fields as-is to use model defaults.
        </p>

        <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="epochs">Epochs</Label>
              <Input id="epochs" type="number" {...register('epochs')} />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="patience">Patience</Label>
              <Input id="patience" type="number" {...register('patience')} />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="warmup_ratio">Warmup ratio</Label>
              <Input id="warmup_ratio" type="number" step="0.01" {...register('warmup_ratio')} />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Remote backend</Label>
              <Controller
                name="remote_backend"
                control={control}
                render={({ field }) => (
                  <Select value={field.value ?? ''} onValueChange={field.onChange}>
                    <SelectTrigger onBlur={field.onBlur} ref={field.ref}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {REMOTE_BACKEND_OPTIONS.map(opt => (
                        <SelectItem key={opt} value={opt}>{opt}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="num_train_samples">Train samples</Label>
              <Input
                id="num_train_samples"
                type="number"
                {...register('num_train_samples')}
                disabled={skipGenerate}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="num_eval_samples">Eval samples</Label>
              <Input
                id="num_eval_samples"
                type="number"
                {...register('num_eval_samples')}
                disabled={skipGenerate}
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <Label>Base model</Label>
            <Controller
              name="base_model"
              control={control}
              render={({ field }) => (
                <Combobox
                  value={field.value ?? ''}
                  onChange={field.onChange}
                  options={BASE_MODEL_OPTIONS}
                />
              )}
            />
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="modal_skip_generate"
              {...register('skip_generate')}
              className="h-4 w-4"
            />
            <Label htmlFor="modal_skip_generate">Skip dataset generation</Label>
          </div>

          {mutation.isError && (
            <p className="text-sm text-red-600">Failed to start run. Please try again.</p>
          )}

          <div className="flex justify-end gap-2 mt-2">
            <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? 'Starting…' : '▶ Start run'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
