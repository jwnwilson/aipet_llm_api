import { Play } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { TrainingModel } from '@/types'
import { Button } from './ui/button'
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from './ui/card'

interface ModelCardProps {
  model: TrainingModel
  onTrigger: (id: string) => void
  isTriggering?: boolean
}

export function ModelCard({ model, onTrigger, isTriggering = false }: ModelCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <Link to={`/models/${model.id}`} className="text-gray-900 hover:underline">
            {model.name}
          </Link>
        </CardTitle>
        {model.description && <CardDescription>{model.description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          <dt className="text-gray-500">Base model</dt>
          <dd className="font-medium text-gray-900 truncate">{model.base_model}</dd>
          <dt className="text-gray-500">Epochs</dt>
          <dd className="font-medium text-gray-900">{model.epochs}</dd>
          <dt className="text-gray-500">Backend</dt>
          <dd className="font-medium text-gray-900">{model.remote_backend}</dd>
        </dl>
      </CardContent>
      <CardFooter className="gap-2">
        <Button
          size="sm"
          onClick={() => onTrigger(model.id)}
          disabled={isTriggering}
          aria-label={`Trigger training run for ${model.name}`}
        >
          <Play className="h-3.5 w-3.5 mr-1" />
          {isTriggering ? 'Starting…' : 'Run'}
        </Button>
        <Button size="sm" variant="outline" asChild>
          <Link to={`/models/${model.id}/edit`}>Edit</Link>
        </Button>
      </CardFooter>
    </Card>
  )
}
