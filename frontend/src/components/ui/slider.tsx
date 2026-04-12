"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface SliderProps {
  className?: string
  min?: number
  max?: number
  step?: number
  value: number[]
  onValueChange: (value: number[]) => void
}

const Slider = React.forwardRef<HTMLDivElement, SliderProps>(
  ({ className, min = 0, max = 100, step = 1, value, onValueChange }, ref) => {
    const current = value[0] ?? min
    const percentage = ((current - min) / (max - min)) * 100

    const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left
      const pct = x / rect.width
      const raw = min + pct * (max - min)
      const snapped = Math.round(raw / step) * step
      onValueChange([Math.min(max, Math.max(min, snapped))])
    }

    return (
      <div
        ref={ref}
        className={cn("relative flex w-full touch-none select-none items-center h-5", className)}
        onClick={handleClick}
        role="slider"
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={current}
        tabIndex={0}
      >
        <div className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary">
          <div
            className="absolute h-full bg-primary rounded-full"
            style={{ width: `${percentage}%` }}
          />
        </div>
        <div
          className="absolute h-5 w-5 rounded-full border-2 border-primary bg-background shadow-md transition-transform hover:scale-110"
          style={{ left: `calc(${percentage}% - 10px)` }}
        />
      </div>
    )
  }
)
Slider.displayName = "Slider"

export { Slider }
