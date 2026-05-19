import { useEffect, useRef } from 'react'

interface RunLogViewerProps {
  logs: string
}

export function RunLogViewer({ logs }: RunLogViewerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  return (
    <div className="relative rounded-md border bg-gray-950 p-4 h-64 overflow-y-auto font-mono text-xs text-gray-200">
      <pre className="whitespace-pre-wrap break-words">{logs || 'No output yet.'}</pre>
      <div ref={bottomRef} />
    </div>
  )
}
