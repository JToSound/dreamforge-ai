import { Suspense, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import { DreamScene } from "./components/DreamScene";
import {
  useSimulationData,
  type SimulationRequestConfig,
  type SleepState,
  type NeuroState,
} from "./useSimulationData";

// ---------------------------------------------------------------------------
// Stage colour map for hypnogram
// ---------------------------------------------------------------------------
const STAGE_Y: Record<string, number> = {
  WAKE: 4,
  REM: 3,
  N1: 2,
  N2: 1,
  N3: 0,
};
const STAGE_COLOR: Record<string, string> = {
  WAKE: "#f59e0b",
  REM: "#22d3ee",
  N1: "#818cf8",
  N2: "#6366f1",
  N3: "#1e40af",
};

// ---------------------------------------------------------------------------
// Mini SVG Hypnogram
// ---------------------------------------------------------------------------
function Hypnogram({ history }: { history: SleepState[] }) {
  if (!history.length) return null;
  const W = 340, H = 100, PAD = 6;
  const maxT = history[history.length - 1].time_hours;
  const xOf = (t: number) => PAD + ((t / maxT) * (W - PAD * 2));
  const yOf = (stage: string) => PAD + ((4 - (STAGE_Y[stage] ?? 2)) / 4) * (H - PAD * 2);

  const points = history
    .map((s) => `${xOf(s.time_hours).toFixed(1)},${yOf(s.stage).toFixed(1)}`)
    .join(" ");

  // Colour segments by stage
  const rects = history.slice(0, -1).map((s, i) => {
    const x1 = xOf(s.time_hours);
    const x2 = xOf(history[i + 1].time_hours);
    return (
      <rect
        key={i}
        x={x1}
        y={yOf(s.stage)}
        width={x2 - x1}
        height={2.5}
        fill={STAGE_COLOR[s.stage] ?? "#6366f1"}
        opacity={0.7}
      />
    );
  });

  const labels = Object.entries(STAGE_Y).map(([label, val]) => (
    <text
      key={label}
      x={1}
      y={yOf(label) + 4}
      fontSize={6}
      fill={STAGE_COLOR[label]}
      fontFamily="monospace"
    >
      {label}
    </text>
  ));

  return (
    <svg width={W} height={H} style={{ display: "block", width: "100%" }}>
      {rects}
      <polyline
        points={points}
        fill="none"
        stroke="rgba(255,255,255,0.35)"
        strokeWidth={0.8}
      />
      {labels}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Mini SVG Neurochemical Flux
// ---------------------------------------------------------------------------
function NeuroFlux({ history }: { history: NeuroState[] }) {
  if (!history.length) return null;
  const W = 340, H = 100, PAD = 6;
  const maxT = history[history.length - 1].time_hours;
  const xOf = (t: number) => PAD + ((t / maxT) * (W - PAD * 2));

  type Key = "ach" | "serotonin" | "ne" | "cortisol";
  const lines: { key: Key; color: string; label: string }[] = [
    { key: "ach", color: "#22d3ee", label: "ACh" },
    { key: "serotonin", color: "#a78bfa", label: "5-HT" },
    { key: "ne", color: "#f472b6", label: "NE" },
    { key: "cortisol", color: "#fbbf24", label: "CORT" },
  ];

  // Normalise each channel to [0,1] over its own range
  const norm = (arr: number[]) => {
    const mn = Math.min(...arr);
    const mx = Math.max(...arr);
    return mx === mn ? arr.map(() => 0.5) : arr.map((v) => (v - mn) / (mx - mn));
  };

  const yOf = (v: number) => PAD + (1 - v) * (H - PAD * 2);

  const polylines = lines.map(({ key, color, label }) => {
    const vals = norm(history.map((n) => n[key] as number));
    const pts = history
      .map((n, i) => `${xOf(n.time_hours).toFixed(1)},${yOf(vals[i]).toFixed(1)}`)
      .join(" ");
    return (
      <g key={key}>
        <polyline points={pts} fill="none" stroke={color} strokeWidth={1.2} opacity={0.85} />
        <text
          x={W - PAD - 1}
          y={yOf(vals[vals.length - 1]) + 3}
          fontSize={5.5}
          fill={color}
          textAnchor="end"
          fontFamily="monospace"
        >
          {label}
        </text>
      </g>
    );
  });

  return (
    <svg width={W} height={H} style={{ display: "block", width: "100%" }}>
      {polylines}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Panel card
// ---------------------------------------------------------------------------
function PanelCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        borderRadius: "0.75rem",
        padding: "0.65rem 0.8rem",
        background: "linear-gradient(135deg,rgba(15,23,42,0.92),rgba(30,64,175,0.55))",
        boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
        border: "1px solid rgba(148,163,184,0.12)",
      }}
    >
      <div
        style={{
          fontSize: "0.7rem",
          fontWeight: 700,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "#94a3b8",
          marginBottom: "0.45rem",
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------
export function App() {
  const [loading, setLoading] = useState(false);
  const { data, error, runSimulation } = useSimulationData(setLoading);

  // Simulation params
  const [durationHours, setDurationHours] = useState(8.0);
  const [dtMinutes, setDtMinutes] = useState(0.5);
  const [ssriStrength, setSsriStrength] = useState(1.0);
  const [stressLevel, setStressLevel] = useState(0.2);

  // LLM settings
  const [llmEnabled, setLlmEnabled] = useState(false);
  const [llmProvider, setLlmProvider] = useState<"" | "openai" | "lmstudio" | "ollama">("openai");
  const [llmModel, setLlmModel] = useState("gpt-4o-mini");
  const [llmImportantOnly, setLlmImportantOnly] = useState(true);
  const [llmApiKey, setLlmApiKey] = useState("");

  // Fake progress bar
  const [progress, setProgress] = useState(0);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleRun = () => {
    const config: SimulationRequestConfig = {
      duration_hours: durationHours,
      dt_minutes: dtMinutes,
      ssri_strength: ssriStrength,
      stress_level: stressLevel,
      llm_enabled: llmEnabled,
      llm_provider: llmEnabled && llmProvider ? llmProvider : null,
      llm_model: llmEnabled && llmModel ? llmModel : null,
      llm_important_only: llmImportantOnly,
      llm_api_key: llmEnabled && llmApiKey ? llmApiKey : null,
    };

    setProgress(0);
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);
    let cur = 0;
    progressTimerRef.current = setInterval(() => {
      cur = Math.min(cur + 4, 88);
      setProgress(cur);
    }, 200);

    runSimulation(config).finally(() => {
      if (progressTimerRef.current) clearInterval(progressTimerRef.current);
      setProgress(100);
      setTimeout(() => setProgress(0), 800);
    });
  };

  // ------------------------------------------------------------------
  // Slider helper
  // ------------------------------------------------------------------
  const SliderRow = ({
    label,
    value,
    min,
    max,
    step,
    onChange,
  }: {
    label: string;
    value: number;
    min: number;
    max: number;
    step: number;
    onChange: (v: number) => void;
  }) => (
    <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
      <span style={{ fontSize: "0.7rem", color: "#94a3b8", width: "70px", flexShrink: 0 }}>
        {label}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ flex: 1, accentColor: "#22d3ee" }}
      />
      <span style={{ fontSize: "0.7rem", color: "#e2e8f0", width: "30px", textAlign: "right" }}>
        {value}
      </span>
    </div>
  );

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "0.25rem 0.4rem",
    borderRadius: "0.4rem",
    border: "1px solid rgba(148,163,184,0.3)",
    background: "rgba(15,23,42,0.9)",
    color: "#e2e8f0",
    fontSize: "0.72rem",
  };

  const selectStyle: React.CSSProperties = { ...inputStyle };

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "#020617",
        color: "#e2e8f0",
        fontFamily: "'Inter',system-ui,sans-serif",
      }}
    >
      {/* ---------------------------------------------------------------- */}
      {/* Header */}
      {/* ---------------------------------------------------------------- */}
      <header
        style={{
          padding: "0.6rem 1.2rem",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "linear-gradient(90deg,#020617,#0f172a)",
          borderBottom: "1px solid rgba(148,163,184,0.18)",
          gap: "1rem",
          flexWrap: "wrap",
          flexShrink: 0,
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: "1rem", fontWeight: 700, letterSpacing: "-0.01em" }}>
            🌙 DreamForge AI
          </h1>
          <p style={{ margin: 0, fontSize: "0.7rem", color: "#94a3b8" }}>
            3D dream space · hypnogram · neurochemistry · memory graph
          </p>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          {/* Progress bar */}
          {progress > 0 && (
            <div
              style={{
                width: "140px",
                height: "5px",
                borderRadius: "999px",
                background: "rgba(30,41,59,0.9)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${progress}%`,
                  height: "100%",
                  borderRadius: "999px",
                  background: "linear-gradient(90deg,#22d3ee,#6366f1)",
                  transition: "width 160ms ease",
                }}
              />
            </div>
          )}

          <button
            onClick={handleRun}
            disabled={loading}
            style={{
              padding: "0.38rem 1rem",
              borderRadius: "999px",
              border: "none",
              background: loading
                ? "rgba(99,102,241,0.4)"
                : "linear-gradient(90deg,#22d3ee,#6366f1)",
              color: "white",
              fontWeight: 700,
              cursor: loading ? "wait" : "pointer",
              fontSize: "0.78rem",
              letterSpacing: "0.02em",
              transition: "opacity 180ms",
            }}
          >
            {loading ? "Simulating…" : "▶ Run Simulation"}
          </button>
        </div>
      </header>

      {/* ---------------------------------------------------------------- */}
      {/* Body: 3D canvas left | side panel right */}
      {/* ---------------------------------------------------------------- */}
      <main
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "row",
          minHeight: 0,
          gap: 0,
        }}
      >
        {/* Left: 3D Dream Space */}
        <section
          style={{
            flex: "1.4 1 0",
            minWidth: 0,
            position: "relative",
            borderRight: "1px solid rgba(148,163,184,0.1)",
          }}
        >
          {error && (
            <div
              style={{
                position: "absolute",
                top: "0.5rem",
                left: "0.75rem",
                zIndex: 10,
                background: "rgba(127,29,29,0.85)",
                padding: "0.35rem 0.75rem",
                borderRadius: "0.5rem",
                fontSize: "0.72rem",
                color: "#fecaca",
                maxWidth: "90%",
              }}
            >
              ⚠ {error}
            </div>
          )}
          <Canvas
            style={{ width: "100%", height: "100%" }}
            camera={{ position: [0, 4, 10], fov: 55 }}
          >
            <color attach="background" args={["#020617"]} />
            <fog attach="fog" args={["#020617", 10, 40]} />
            <ambientLight intensity={0.2} />
            <directionalLight intensity={0.8} position={[5, 10, 5]} />
            <Stars radius={80} depth={50} count={5000} factor={4} fade speed={1} />
            <Suspense fallback={null}>
              {data && <DreamScene simulation={data} />}
            </Suspense>
            <OrbitControls enablePan={false} minDistance={5} maxDistance={40} />
          </Canvas>

          {!data && !loading && (
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                pointerEvents: "none",
              }}
            >
              <div
                style={{
                  textAlign: "center",
                  color: "#475569",
                  fontSize: "0.8rem",
                  lineHeight: 1.6,
                }}
              >
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>💤</div>
                Press <strong style={{ color: "#94a3b8" }}>Run Simulation</strong> to begin
              </div>
            </div>
          )}
        </section>

        {/* Right: Settings + Charts */}
        <aside
          style={{
            flex: "1 1 300px",
            minWidth: "280px",
            maxWidth: "440px",
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: "0.6rem",
            padding: "0.75rem",
            background: "rgba(2,6,23,0.95)",
          }}
        >
          {/* Simulation Params */}
          <PanelCard title="Simulation Parameters">
            <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
              <SliderRow
                label="Duration h"
                value={durationHours}
                min={4}
                max={10}
                step={0.5}
                onChange={setDurationHours}
              />
              <SliderRow
                label="dt min"
                value={dtMinutes}
                min={0.25}
                max={2}
                step={0.25}
                onChange={setDtMinutes}
              />
              <SliderRow
                label="SSRI"
                value={ssriStrength}
                min={0.5}
                max={2.0}
                step={0.1}
                onChange={setSsriStrength}
              />
              <SliderRow
                label="Stress"
                value={stressLevel}
                min={0}
                max={1}
                step={0.05}
                onChange={setStressLevel}
              />
            </div>
          </PanelCard>

          {/* LLM Settings */}
          <PanelCard title="LLM Settings">
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
                fontSize: "0.72rem",
                marginBottom: "0.4rem",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={llmEnabled}
                onChange={(e) => setLlmEnabled(e.target.checked)}
                style={{ accentColor: "#22d3ee" }}
              />
              Enable LLM-backed dream narratives
            </label>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "0.4rem",
                opacity: llmEnabled ? 1 : 0.4,
                pointerEvents: llmEnabled ? "auto" : "none",
              }}
            >
              <div>
                <div style={{ fontSize: "0.65rem", color: "#94a3b8", marginBottom: "0.15rem" }}>
                  Provider
                </div>
                <select
                  value={llmProvider}
                  onChange={(e) => setLlmProvider(e.target.value as any)}
                  style={selectStyle}
                >
                  <option value="">Select…</option>
                  <option value="openai">OpenAI</option>
                  <option value="lmstudio">LM Studio</option>
                  <option value="ollama">Ollama</option>
                </select>
              </div>
              <div>
                <div style={{ fontSize: "0.65rem", color: "#94a3b8", marginBottom: "0.15rem" }}>
                  Model
                </div>
                <input
                  type="text"
                  value={llmModel}
                  onChange={(e) => setLlmModel(e.target.value)}
                  style={inputStyle}
                  placeholder="gpt-4o-mini"
                />
              </div>
              <div style={{ gridColumn: "1 / -1" }}>
                <div style={{ fontSize: "0.65rem", color: "#94a3b8", marginBottom: "0.15rem" }}>
                  API Key
                </div>
                <input
                  type="password"
                  value={llmApiKey}
                  onChange={(e) => setLlmApiKey(e.target.value)}
                  style={inputStyle}
                  placeholder="sk-…  (stored only in-browser)"
                />
              </div>
            </div>

            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.4rem",
                fontSize: "0.68rem",
                marginTop: "0.4rem",
                color: "#94a3b8",
                opacity: llmEnabled ? 1 : 0.4,
                pointerEvents: llmEnabled ? "auto" : "none",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={llmImportantOnly}
                onChange={(e) => setLlmImportantOnly(e.target.checked)}
                style={{ accentColor: "#22d3ee" }}
              />
              Only use LLM for REM / high-replay segments
            </label>
          </PanelCard>

          {/* Hypnogram */}
          {data && data.sleep_history.length > 0 && (
            <PanelCard title="Sleep Hypnogram">
              <Hypnogram history={data.sleep_history} />
              <div
                style={{
                  display: "flex",
                  gap: "0.5rem",
                  flexWrap: "wrap",
                  marginTop: "0.35rem",
                }}
              >
                {Object.entries(STAGE_COLOR).map(([s, c]) => (
                  <span
                    key={s}
                    style={{ fontSize: "0.6rem", color: c, fontFamily: "monospace" }}
                  >
                    ● {s}
                  </span>
                ))}
              </div>
            </PanelCard>
          )}

          {/* Neurochemical Flux */}
          {data && data.neuro_history.length > 0 && (
            <PanelCard title="Neurochemical Flux">
              <NeuroFlux history={data.neuro_history} />
            </PanelCard>
          )}

          {/* Summary stats */}
          {data && (
            <PanelCard title="Night Summary">
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "0.35rem 0.6rem",
                  fontSize: "0.7rem",
                }}
              >
                {Object.entries(data.summary.sleep_stages).map(([s, v]) => (
                  <div key={s}>
                    <span style={{ color: STAGE_COLOR[s] ?? "#94a3b8" }}>{s}</span>
                    <span style={{ float: "right", color: "#e2e8f0" }}>
                      {(v * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
                <div style={{ gridColumn: "1 / -1", borderTop: "1px solid rgba(148,163,184,0.1)", paddingTop: "0.3rem", marginTop: "0.1rem" }}>
                  <span style={{ color: "#94a3b8" }}>Mean bizarreness</span>
                  <span style={{ float: "right" }}>{data.summary.bizarreness.mean.toFixed(2)}</span>
                </div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <span style={{ color: "#94a3b8" }}>Segments</span>
                  <span style={{ float: "right" }}>{data.segments.length}</span>
                </div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <span style={{ color: "#94a3b8" }}>Memory nodes</span>
                  <span style={{ float: "right" }}>{data.memory_graph.nodes.length}</span>
                </div>
              </div>
            </PanelCard>
          )}

          {/* Top memory nodes */}
          {data && data.summary.memory?.top_nodes?.length > 0 && (
            <PanelCard title="Top Memory Nodes">
              <div style={{ display: "flex", flexDirection: "column", gap: "0.2rem" }}>
                {data.summary.memory.top_nodes.slice(0, 6).map((n) => (
                  <div
                    key={n.id}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: "0.68rem",
                      padding: "0.15rem 0",
                      borderBottom: "1px solid rgba(148,163,184,0.06)",
                    }}
                  >
                    <span
                      style={{
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        maxWidth: "180px",
                        color: "#c7d2fe",
                      }}
                    >
                      {n.label}
                    </span>
                    <span style={{ color: "#94a3b8", flexShrink: 0 }}>
                      {n.emotion} · {n.salience.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </PanelCard>
          )}
        </aside>
      </main>
    </div>
  );
}
