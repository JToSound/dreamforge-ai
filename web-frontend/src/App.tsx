import { useState } from 'react'
import {
  useSimulationData,
  LLMConfig,
  SleepConfig,
  SimulateRequest,
  DreamSegment,
} from './useSimulationData'

const PROVIDER_MODELS: Record<string, string[]> = {
  openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
  anthropic: ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-3-5'],
  ollama: ['llama3', 'mistral', 'phi3', 'gemma2'],
}

const STAGE_COLOR: Record<string, string> = {

        {/* Download buttons */}
        {result && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <button
              onClick={downloadJSON}
              style={{
                background: 'linear-gradient(90deg, #10b981, #34d399)',
                border: 'none',
                color: '#000',
                borderRadius: 8,
                padding: '8px 12px',
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              Download JSON
            </button>
            <button
              onClick={downloadText}
              style={{
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                border: 'none',
                color: '#fff',
                borderRadius: 8,
                padding: '8px 12px',
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              Download Text
            </button>
          </div>
        )}
  WAKE: '#f59e0b',
  N1: '#60a5fa',
  N2: '#818cf8',
  N3: '#6366f1',
  REM: '#f472b6',
}

function EmotionBadge({ emotion }: { emotion: string }) {
  const colors: Record<string, string> = {
    joy: '#fbbf24',
    fear: '#f87171',
    sadness: '#60a5fa',
    anger: '#ef4444',
    surprise: '#a78bfa',
    disgust: '#34d399',
    neutral: '#9ca3af',
  }
  return (
    <span
      style={{
        background: colors[emotion] ?? '#9ca3af',
        color: '#000',
        borderRadius: 9999,
        padding: '2px 10px',
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: '0.05em',
        textTransform: 'uppercase',
      }}
    >
      {emotion}
    </span>
  )
}

function BizarrenessBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color = value < 0.33 ? '#34d399' : value < 0.66 ? '#fbbf24' : '#f87171'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div
        style={{
          flex: 1,
          height: 6,
          background: '#374151',
          borderRadius: 3,
          overflow: 'hidden',
        }}
      >
        <div
          style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }}
        />
      </div>
      <span style={{ fontSize: 11, color: '#9ca3af', minWidth: 30 }}>{pct}%</span>
    </div>
  )
}

function SegmentCard({ seg }: { seg: DreamSegment }) {
  const [open, setOpen] = useState(false)
  return (
    <div
      onClick={() => setOpen(!open)}
      style={{
        background: '#1f2937',
        borderRadius: 10,
        padding: '14px 16px',
        cursor: 'pointer',
        border: '1px solid #374151',
        transition: 'border-color 0.15s',
      }}
      onMouseEnter={e => ((e.currentTarget as HTMLDivElement).style.borderColor = '#6366f1')}
      onMouseLeave={e => ((e.currentTarget as HTMLDivElement).style.borderColor = '#374151')}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span
            style={{
              background: STAGE_COLOR[seg.stage] ?? '#6b7280',
              color: '#000',
              borderRadius: 6,
              padding: '2px 8px',
              fontSize: 11,
              fontWeight: 700,
            }}
          >
            {seg.stage}
          </span>
          <span style={{ fontSize: 12, color: '#9ca3af' }}>
            {seg.time_hours.toFixed(2)}h
          </span>
          <EmotionBadge emotion={seg.dominant_emotion} />
        </div>
        <span style={{ fontSize: 11, color: '#6b7280' }}>{open ? '▲' : '▼'}</span>
      </div>

      <div style={{ marginTop: 8 }}>
        <BizarrenessBar value={seg.bizarreness_score} />
      </div>

      {open && (
        <p
          style={{
            marginTop: 10,
            fontSize: 13,
            lineHeight: 1.6,
            color: '#d1d5db',
            fontStyle: 'italic',
          }}
        >
          "{seg.narrative}"
        </p>
      )}
    </div>
  )
}

export default function App() {
  // ── LLM config state ─────────────────────────────────────────────────────
  const [provider, setProvider] = useState<LLMConfig['provider']>('openai')
  const [model, setModel] = useState('gpt-4o')
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [temperature, setTemperature] = useState(0.9)
  const [maxTokens, setMaxTokens] = useState(512)

  // ── Sleep config state ───────────────────────────────────────────────────
  const [durationHours, setDurationHours] = useState(8)
  const [stressLevel, setStressLevel] = useState(0.5)
  const [priorEvents, setPriorEvents] = useState(
    'Had a stressful meeting\nWent for an evening run\nWatched a sci-fi film'
  )

  const { status, progress, progressMsg, currentStage, result, liveSegments, error, runSimulation, cancel } =
    useSimulationData()

  const handleRun = () => {
    const req: SimulateRequest = {
      llm: {
        provider,
        model,
        api_key: apiKey,
        base_url: baseUrl || undefined,
        temperature,
        max_tokens: maxTokens,
      },
      sleep: {
        duration_hours: durationHours,
        sleep_start_clock_time: 23,
        dt_minutes: 0.5,
      },
      prior_day_events: priorEvents.split('\n').filter(Boolean),
      stress_level: stressLevel,
    }
    runSimulation(req)
  }

  const handleProviderChange = (p: LLMConfig['provider']) => {
    setProvider(p)
    setModel(PROVIDER_MODELS[p][0])
    if (p === 'ollama') setBaseUrl('http://localhost:11434/v1')
    else setBaseUrl('')
  }

  const running = status === 'running'
  const segments = running ? liveSegments : result?.dream_segments ?? []

  // ── Download helpers ───────────────────────────────────────────────────
  const downloadJSON = () => {
    if (!result) return
    const resAny: any = result as any
    const simId = resAny.simulation_id ?? resAny.id ?? Date.now()
    const filename = `dreamforge-sim-${simId}.json`
    const content = JSON.stringify(resAny, null, 2)
    const blob = new Blob([content], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const downloadText = () => {
    if (!result) return
    const resAny: any = result as any
    const simId = resAny.simulation_id ?? resAny.id ?? ''
    let text = ''
    if (simId) text += `Simulation ID: ${simId}\n`
    const duration = resAny.duration_hours ?? resAny.config?.duration_hours ?? resAny.metadata?.duration_hours ?? ''
    if (duration) text += `Duration: ${duration}h\n`
    text += '\n'
    if (resAny.summary_narrative) text += `${resAny.summary_narrative}\n\n`
    else if (resAny.summary && typeof resAny.summary === 'string') text += `${resAny.summary}\n\n`
    else if (resAny.summary && typeof resAny.summary === 'object') text += `${JSON.stringify(resAny.summary, null, 2)}\n\n`

    const segs = resAny.dream_segments ?? resAny.segments ?? []
    for (let i = 0; i < segs.length; i++) {
      const s: any = segs[i]
      const idx = s.segment_index ?? s.id ?? i
      const time = s.time_hours ?? s.start_time_hours ?? ''
      const stage = s.stage ?? ''
      const narrative = s.narrative ?? s.scene_description ?? s.scene ?? ''
      text += `--- Segment ${idx} (${time}h) [${stage}]\n${narrative}\n\n`
    }

    const filename = `dreamforge-sim-${simId || Date.now()}.txt`
    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#111827',
        color: '#f9fafb',
        fontFamily: "'Inter', sans-serif",
        display: 'grid',
        gridTemplateColumns: '340px 1fr',
        gridTemplateRows: 'auto 1fr',
      }}
    >
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header
        style={{
          gridColumn: '1 / -1',
          borderBottom: '1px solid #1f2937',
          padding: '16px 32px',
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          background: '#0f172a',
        }}
      >
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
          <circle cx="14" cy="14" r="14" fill="#6366f1" opacity="0.15" />
          <path d="M14 6 Q20 10 20 14 Q20 20 14 22 Q8 20 8 14 Q8 10 14 6Z" fill="#6366f1" opacity="0.6" />
          <circle cx="14" cy="14" r="3" fill="#a5b4fc" />
        </svg>
        <div>
          <h1 style={{ margin: 0, fontSize: 18, fontWeight: 700, letterSpacing: '-0.02em' }}>
            DreamForge AI
          </h1>
          <p style={{ margin: 0, fontSize: 11, color: '#6b7280' }}>
            Multi-agent dream simulation
          </p>
        </div>

        {running && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{
                background: '#1f2937',
                borderRadius: 9999,
                height: 8,
                width: 220,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${Math.round(progress * 100)}%`,
                  background: 'linear-gradient(90deg, #6366f1, #f472b6)',
                  transition: 'width 0.3s ease',
                  borderRadius: 9999,
                }}
              />
            </div>
            <span style={{ fontSize: 12, color: '#a5b4fc', minWidth: 36 }}>
              {Math.round(progress * 100)}%
            </span>
            <span
              style={{
                background: STAGE_COLOR[currentStage] ?? '#6b7280',
                color: '#000',
                fontSize: 10,
                fontWeight: 700,
                padding: '2px 8px',
                borderRadius: 4,
              }}
            >
              {currentStage || 'INIT'}
            </span>
            <button
              onClick={cancel}
              style={{
                background: '#374151',
                border: 'none',
                color: '#f9fafb',
                borderRadius: 6,
                padding: '4px 12px',
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
          </div>
        )}
      </header>

      {/* ── Left panel: config ──────────────────────────────────────────────── */}
      <aside
        style={{
          background: '#0f172a',
          borderRight: '1px solid #1f2937',
          padding: '24px 20px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 20,
        }}
      >
        {/* LLM Config */}
        <section>
          <h2
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: '#6b7280',
              marginBottom: 12,
            }}
          >
            LLM Settings
          </h2>

          <label style={labelStyle}>Provider</label>
          <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
            {(['openai', 'anthropic', 'ollama'] as const).map(p => (
              <button
                key={p}
                onClick={() => handleProviderChange(p)}
                style={{
                  flex: 1,
                  padding: '6px 0',
                  borderRadius: 6,
                  border: '1px solid',
                  borderColor: provider === p ? '#6366f1' : '#374151',
                  background: provider === p ? '#312e81' : '#1f2937',
                  color: provider === p ? '#c7d2fe' : '#9ca3af',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: 'pointer',
                  textTransform: 'capitalize',
                }}
              >
                {p}
              </button>
            ))}
          </div>

          <label style={labelStyle}>Model</label>
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            style={inputStyle}
          >
            {PROVIDER_MODELS[provider].map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>

          <label style={labelStyle}>API Key</label>
          <input
            type="password"
            placeholder={provider === 'ollama' ? 'Not required for Ollama' : 'sk-…'}
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            style={inputStyle}
            disabled={provider === 'ollama'}
          />

          {provider === 'ollama' && (
            <>
              <label style={labelStyle}>Ollama Base URL</label>
              <input
                value={baseUrl}
                onChange={e => setBaseUrl(e.target.value)}
                style={inputStyle}
              />
            </>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 4 }}>
            <div>
              <label style={labelStyle}>Temperature</label>
              <input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={temperature}
                onChange={e => setTemperature(Number(e.target.value))}
                style={inputStyle}
              />
            </div>
            <div>
              <label style={labelStyle}>Max Tokens</label>
              <input
                type="number"
                min={64}
                max={4096}
                step={64}
                value={maxTokens}
                onChange={e => setMaxTokens(Number(e.target.value))}
                style={inputStyle}
              />
            </div>
          </div>
        </section>

        {/* Sleep Config */}
        <section>
          <h2 style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#6b7280', marginBottom: 12 }}>
            Simulation
          </h2>

          <label style={labelStyle}>Night Duration (hours)</label>
          <input
            type="range"
            min={4}
            max={12}
            step={0.5}
            value={durationHours}
            onChange={e => setDurationHours(Number(e.target.value))}
            style={{ width: '100%', marginBottom: 4 }}
          />
          <span style={{ fontSize: 12, color: '#9ca3af' }}>{durationHours}h</span>

          <label style={{ ...labelStyle, marginTop: 12 }}>Stress Level</label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={stressLevel}
            onChange={e => setStressLevel(Number(e.target.value))}
            style={{ width: '100%', marginBottom: 4 }}
          />
          <span style={{ fontSize: 12, color: '#9ca3af' }}>
            {stressLevel < 0.33 ? '😌 Low' : stressLevel < 0.66 ? '😐 Medium' : '😰 High'} ({stressLevel.toFixed(2)})
          </span>

          <label style={{ ...labelStyle, marginTop: 12 }}>Prior Day Events</label>
          <textarea
            rows={5}
            value={priorEvents}
            onChange={e => setPriorEvents(e.target.value)}
            placeholder="One event per line…"
            style={{ ...inputStyle, resize: 'vertical', lineHeight: 1.5 }}
          />
        </section>

        {/* Run Button */}
        <button
          onClick={running ? cancel : handleRun}
          disabled={false}
          style={{
            background: running
              ? '#7f1d1d'
              : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            border: 'none',
            borderRadius: 8,
            color: '#fff',
            fontSize: 14,
            fontWeight: 700,
            padding: '12px 0',
            cursor: 'pointer',
            letterSpacing: '0.02em',
            boxShadow: running ? 'none' : '0 0 20px rgba(99,102,241,0.4)',
            transition: 'all 0.2s',
          }}
        >
          {running ? '⏹ Stop Simulation' : '▶ Run Simulation'}
        </button>

        {error && (
          <div
            style={{
              background: '#7f1d1d',
              border: '1px solid #ef4444',
              borderRadius: 8,
              padding: 12,
              fontSize: 12,
              color: '#fca5a5',
            }}
          >
            ⚠ {error}
          </div>
        )}
      </aside>

      {/* ── Main content area ──────────────────────────────────────────────── */}
      <main style={{ overflowY: 'auto', padding: '24px 32px' }}>
        {/* Progress message */}
        {(running || status === 'complete') && (
          <div
            style={{
              background: '#1f2937',
              borderRadius: 8,
              padding: '10px 16px',
              marginBottom: 20,
              fontSize: 13,
              color: '#a5b4fc',
              display: 'flex',
              gap: 10,
              alignItems: 'center',
            }}
          >
            {running && (
              <span
                style={{
                  display: 'inline-block',
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: '#f472b6',
                  animation: 'pulse 1s infinite',
                }}
              />
            )}
            {progressMsg}
          </div>
        )}

        {/* Summary stats */}
        {result && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 12,
              marginBottom: 24,
            }}
          >
            {[
              { label: 'Duration', value: `${result.duration_hours}h` },
              { label: 'Segments', value: result.total_segments },
              { label: 'Mean Bizarreness', value: `${(result.mean_bizarreness * 100).toFixed(0)}%` },
              { label: 'Dominant Emotion', value: result.dominant_emotion },
            ].map(stat => (
              <div
                key={stat.label}
                style={{
                  background: '#1f2937',
                  borderRadius: 10,
                  padding: '14px 16px',
                  border: '1px solid #374151',
                }}
              >
                <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  {stat.label}
                </div>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#f9fafb' }}>
                  {stat.value}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Summary narrative */}
        {result?.summary_narrative && (
          <div
            style={{
              background: '#1f2937',
              borderRadius: 10,
              padding: '16px 20px',
              marginBottom: 24,
              borderLeft: '3px solid #6366f1',
              fontSize: 13,
              color: '#d1d5db',
              lineHeight: 1.6,
            }}
          >
            {result.summary_narrative}
          </div>
        )}

        {/* Dream segments */}
        {segments.length > 0 && (
          <>
            <h3
              style={{
                fontSize: 13,
                fontWeight: 700,
                color: '#9ca3af',
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                marginBottom: 12,
              }}
            >
              Dream Segments ({segments.length})
              {running && <span style={{ color: '#f472b6', marginLeft: 8 }}>● Live</span>}
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {segments.map(seg => (
                <SegmentCard key={seg.segment_index} seg={seg} />
              ))}
            </div>
          </>
        )}

        {/* Empty state */}
        {segments.length === 0 && !running && status === 'idle' && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: 300,
              color: '#374151',
              gap: 16,
            }}
          >
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
              <circle cx="32" cy="32" r="32" fill="#1f2937" />
              <path d="M32 16 Q44 22 44 32 Q44 44 32 48 Q20 44 20 32 Q20 22 32 16Z" fill="#374151" />
              <circle cx="32" cy="32" r="6" fill="#4b5563" />
            </svg>
            <p style={{ fontSize: 14 }}>Configure your LLM and press Run Simulation</p>
          </div>
        )}
      </main>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        select, input, textarea {
          background: #1f2937;
          border: 1px solid #374151;
          border-radius: 6px;
          color: #f9fafb;
          padding: 7px 10px;
          width: 100%;
          font-size: 13px;
          box-sizing: border-box;
          margin-bottom: 10px;
          outline: none;
          transition: border-color 0.15s;
        }
        select:focus, input:focus, textarea:focus {
          border-color: #6366f1;
        }
        select option { background: #1f2937; }
      `}</style>
    </div>
  )
}

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: 11,
  fontWeight: 600,
  color: '#9ca3af',
  marginBottom: 4,
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
}

const inputStyle: React.CSSProperties = {
  background: '#1f2937',
  border: '1px solid #374151',
  borderRadius: 6,
  color: '#f9fafb',
  padding: '7px 10px',
  width: '100%',
  fontSize: 13,
  boxSizing: 'border-box',
  marginBottom: 10,
  outline: 'none',
}