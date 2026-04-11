import { useState, useRef, useCallback } from 'react'

// ── Types ────────────────────────────────────────────────────────────────────

export interface LLMConfig {
  provider: 'openai' | 'anthropic' | 'ollama'
  model: string
  api_key: string
  base_url?: string
  temperature: number
  max_tokens: number
}

export interface SleepConfig {
  duration_hours: number
  sleep_start_clock_time: number
  dt_minutes: number
}

export interface SimulateRequest {
  llm: LLMConfig
  sleep: SleepConfig
  prior_day_events: string[]
  stress_level: number
}

export interface HypnogramPoint {
  time_hours: number
  stage: string
  process_s: number
  process_c: number
}

export interface NeurochemPoint {
  time_hours: number
  ach: number
  serotonin: number
  ne: number
  cortisol: number
}

export interface DreamSegment {
  segment_index: number
  time_hours: number
  stage: string
  narrative: string
  dominant_emotion: string
  bizarreness_score: number
  lucidity_probability: number
  active_memory_ids: string[]
  neurochemistry: { ach: number; serotonin: number; ne: number; cortisol: number }
}

export interface SimulationResult {
  simulation_id: string
  duration_hours: number
  total_segments: number
  hypnogram: HypnogramPoint[]
  neurochemistry_series: NeurochemPoint[]
  dream_segments: DreamSegment[]
  memory_graph: { nodes: any[]; edges: any[] }
  summary_narrative: string
  mean_bizarreness: number
  dominant_emotion: string
}

export interface ProgressEvent {
  simulation_id: string
  progress: number
  stage: string
  message: string
  segment?: DreamSegment
}

export type SimStatus = 'idle' | 'running' | 'complete' | 'error'

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useSimulationData() {
  const [status, setStatus] = useState<SimStatus>('idle')
  const [progress, setProgress] = useState(0)
  const [progressMsg, setProgressMsg] = useState('')
  const [currentStage, setCurrentStage] = useState('')
  const [result, setResult] = useState<SimulationResult | null>(null)
  const [liveSegments, setLiveSegments] = useState<DreamSegment[]>([])
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const runSimulation = useCallback(async (req: SimulateRequest) => {
    setStatus('running')
    setProgress(0)
    setProgressMsg('Starting simulation…')
    setLiveSegments([])
    setResult(null)
    setError(null)

    abortRef.current = new AbortController()

    try {
      const response = await fetch('/api/simulate-night', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
        signal: abortRef.current.signal,
      })

      if (!response.ok) {
        const text = await response.text()
        throw new Error(`API error ${response.status}: ${text}`)
      }

      // Check if server returns SSE or plain JSON
      const contentType = response.headers.get('content-type') || ''

      if (contentType.includes('text/event-stream')) {
        // ── SSE streaming mode ──────────────────────────────────────────
        const reader = response.body!.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const evt: ProgressEvent = JSON.parse(line.slice(6))
                setProgress(evt.progress)
                setProgressMsg(evt.message)
                setCurrentStage(evt.stage)
                if (evt.segment) {
                  setLiveSegments(prev => [...prev, evt.segment!])
                }
                if (evt.progress >= 1.0 && (evt as any).result) {
                  setResult((evt as any).result)
                  setStatus('complete')
                }
              } catch {
                // ignore malformed SSE line
              }
            }
          }
        }
      } else {
        // ── Plain JSON mode (polling fallback) ──────────────────────────
        const data: SimulationResult = await response.json()
        setProgress(1.0)
        setProgressMsg('Simulation complete.')
        setCurrentStage(data.dominant_emotion)
        setLiveSegments(data.dream_segments)
        setResult(data)
        setStatus('complete')
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setStatus('idle')
        setProgressMsg('Simulation cancelled.')
      } else {
        setError(err.message ?? 'Unknown error')
        setStatus('error')
      }
    }
  }, [])

  const cancel = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  return {
    status,
    progress,
    progressMsg,
    currentStage,
    result,
    liveSegments,
    error,
    runSimulation,
    cancel,
  }
}