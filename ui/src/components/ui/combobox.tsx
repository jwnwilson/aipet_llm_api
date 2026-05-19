import * as React from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ComboboxProps {
  value: string
  onChange: (value: string) => void
  options: string[]
  placeholder?: string
  disabled?: boolean
  className?: string
  id?: string
}

export function Combobox({ value, onChange, options, placeholder, disabled, className, id }: ComboboxProps) {
  const [open, setOpen] = React.useState(false)
  const [activeIndex, setActiveIndex] = React.useState(-1)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const listRef = React.useRef<HTMLUListElement>(null)

  const filtered = React.useMemo(
    () =>
      value.trim() === ''
        ? options
        : options.filter(opt => opt.toLowerCase().includes(value.toLowerCase())),
    [value, options]
  )

  React.useEffect(() => { setActiveIndex(-1) }, [filtered])

  React.useEffect(() => {
    if (activeIndex >= 0 && listRef.current) {
      const el = listRef.current.children[activeIndex] as HTMLElement | undefined
      el?.scrollIntoView({ block: 'nearest' })
    }
  }, [activeIndex])

  React.useEffect(() => {
    if (!open) return
    function handler(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
      setOpen(true)
      return
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex(i => Math.min(i + 1, filtered.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex(i => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && activeIndex >= 0) {
      e.preventDefault()
      onChange(filtered[activeIndex])
      setOpen(false)
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div ref={containerRef} className={cn('relative w-full', className)}>
      <input
        id={id}
        role="combobox"
        aria-expanded={open}
        aria-autocomplete="list"
        aria-controls="combobox-listbox"
        autoComplete="off"
        disabled={disabled}
        value={value}
        placeholder={placeholder}
        onChange={e => {
          onChange(e.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKeyDown}
        className={cn(
          'flex h-9 w-full rounded-md border border-gray-300 bg-white px-3 py-1 pr-8 text-sm text-gray-900 shadow-sm transition-colors',
          'placeholder:text-gray-400',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
          'disabled:cursor-not-allowed disabled:opacity-50'
        )}
      />
      <ChevronDown
        className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500"
        aria-hidden
      />
      {open && filtered.length > 0 && (
        <ul
          id="combobox-listbox"
          role="listbox"
          ref={listRef}
          className="absolute z-50 mt-1 max-h-60 w-full overflow-auto rounded-md border border-gray-200 bg-white py-1 shadow-md text-sm text-gray-900"
        >
          {filtered.map((opt, i) => (
            <li
              key={opt}
              role="option"
              aria-selected={opt === value}
              onMouseDown={e => {
                e.preventDefault()
                onChange(opt)
                setOpen(false)
              }}
              onMouseEnter={() => setActiveIndex(i)}
              className={cn(
                'cursor-pointer select-none px-3 py-1.5',
                i === activeIndex && 'bg-blue-50 text-blue-900',
                opt === value && i !== activeIndex && 'font-medium'
              )}
            >
              {opt}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
