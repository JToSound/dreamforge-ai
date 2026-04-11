import { Suspense, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import { DreamScene } from "./components/DreamScene";
import { useSimulationData } from "./useSimulationData";

export function App() {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const [llmEnabled, setLlmEnabled] = useState(false);
  const [llmProvider, setLlmProvider] = useState<string>("");
  const [llmModel, setLlmModel] = useState("qwen-3.5-9b");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [llmImportantOnly, setLlmImportantOnly] = useState(true);

  const { data, error, runSimulation } = useSimulationData(setLoading, setProgress);

  const handleRun = () => {
    setProgress(0);
    runSimulation({
      durationHours: 8.0,
      dtMinutes: 0.5,
      ssriStrength: 1.0,
      stressLevel: 0.2,
      llmEnabled,
      llmProvider: llmEnabled ? llmProvider || null : null,
      llmModel: llmEnabled ? llmModel || null : null,
      llmImportantOnly,
      llmApiKey: llmEnabled && llmApiKey ? llmApiKey : null,
    });
  };

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "#020617",
        color: "#e5e7eb",
        position: "relative",
      }}
    >
      <header
        style={{
          padding: "0.75rem 1.5rem",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "linear-gradient(90deg,#020617,#0f172a)",
          borderBottom: "1px solid rgba(148,163,184,0.5)",
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: "1.1rem" }}>DreamForge 3D Dream Space</h1>
          <p style={{ margin: 0, fontSize: "0.75rem", opacity: 0.8 }}>
            3D visualization of simulated dreams: memory nodes as stars, camera path as timeline.
          </p>
        </div>
        <button
          onClick={handleRun}
          style={{
            padding: "0.4rem 0.9rem",
            borderRadius: "999px",
            border: "none",
            background:
              "radial-gradient(circle at 0 0,#22d3ee 0,#0ea5e9 35%,#4f46e5 70%,#0f172a 100%)",
            color: "white",
            fontWeight: 600,
            cursor: "pointer",
            fontSize: "0.8rem",
          }}
        >
          {loading ? "Simulating..." : "Run Simulation"}
        </button>
      </header>

      {(loading || progress > 0) && (
        <div
          style={{
            height: "3px",
            width: "100%",
            background: "rgba(15,23,42,0.95)",
            boxShadow: "0 1px 0 rgba(15,23,42,0.9)",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${progress}%`,
              background: "linear-gradient(90deg,#22d3ee,#a855f7,#f97316)",
              transition: "width 0.2s ease-out",
            }}
          />
        </div>
      )}

      <main style={{ flex: 1 }}>
        {error && (
          <div style={{ padding: "0.75rem 1.5rem", color: "#fecaca", fontSize: "0.8rem" }}>
            Error: {error}
          </div>
        )}
        <Canvas camera={{ position: [0, 4, 10], fov: 55 }}>
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
      </main>

      {/* LLM settings panel overlay */}
      <div
        style={{
          position: "absolute",
          top: "4.5rem",
          right: "1.5rem",
          width: "min(320px, 90vw)",
          padding: "0.75rem 1rem",
          borderRadius: "0.75rem",
          background: "rgba(15,23,42,0.92)",
          border: "1px solid rgba(148,163,184,0.45)",
          boxShadow: "0 18px 45px rgba(15,23,42,0.95)",
          backdropFilter: "blur(18px)",
          fontSize: "0.8rem",
          zIndex: 10,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "0.35rem",
          }}
        >
          <span style={{ fontWeight: 600 }}>LLM Settings</span>
          <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.75rem" }}>
            <input
              type="checkbox"
              checked={llmEnabled}
              onChange={(e) => setLlmEnabled(e.target.checked)}
              style={{ accentColor: "#22d3ee" }}
            />
            Enable
          </label>
        </div>
        <div style={{ display: "grid", gap: "0.45rem" }}>
          <div>
            <label style={{ display: "block", marginBottom: "0.15rem", opacity: 0.85 }}>Provider</label>
            <select
              value={llmProvider}
              onChange={(e) => setLlmProvider(e.target.value)}
              disabled={!llmEnabled}
              style={{
                width: "100%",
                padding: "0.3rem 0.5rem",
                borderRadius: "0.5rem",
                border: "1px solid rgba(148,163,184,0.6)",
                background: "rgba(15,23,42,0.9)",
                color: "#e5e7eb",
                fontSize: "0.8rem",
              }}
            >
              <option value="">Select provider…</option>
              <option value="lmstudio">LM Studio (local)</option>
              <option value="openai">OpenAI-compatible</option>
              <option value="ollama">Ollama (local)</option>
            </select>
          </div>
          <div>
            <label style={{ display: "block", marginBottom: "0.15rem", opacity: 0.85 }}>Model</label>
            <input
              type="text"
              value={llmModel}
              onChange={(e) => setLlmModel(e.target.value)}
              disabled={!llmEnabled}
              placeholder="e.g. qwen-3.5-9b"
              style={{
                width: "100%",
                padding: "0.3rem 0.5rem",
                borderRadius: "0.5rem",
                border: "1px solid rgba(148,163,184,0.6)",
                background: "rgba(15,23,42,0.9)",
                color: "#e5e7eb",
                fontSize: "0.8rem",
              }}
            />
          </div>
          <div>
            <label style={{ display: "block", marginBottom: "0.15rem", opacity: 0.85 }}>API key (local only)</label>
            <input
              type="password"
              value={llmApiKey}
              onChange={(e) => setLlmApiKey(e.target.value)}
              disabled={!llmEnabled}
              placeholder="Overrides env vars for this run"
              style={{
                width: "100%",
                padding: "0.3rem 0.5rem",
                borderRadius: "0.5rem",
                border: "1px solid rgba(148,163,184,0.6)",
                background: "rgba(15,23,42,0.9)",
                color: "#e5e7eb",
                fontSize: "0.8rem",
              }}
            />
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.75rem", opacity: 0.9 }}>
            <input
              type="checkbox"
              checked={llmImportantOnly}
              onChange={(e) => setLlmImportantOnly(e.target.checked)}
              style={{ accentColor: "#22d3ee" }}
            />
            Use LLM only for vivid segments
          </label>
        </div>
      </div>
    </div>
  );
}
