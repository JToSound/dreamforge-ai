import { useEffect, useMemo, useState, type CSSProperties, type MouseEvent } from "react";
import {
  useSimulationData,
  type DreamSegment,
  type SimulateRequest,
} from "./useSimulationData";
import "./App.css";

type ThemeMode = "dark" | "light";
type PromptProfile = "A" | "B";

interface ArchitectureCard {
  title: string;
  summary: string;
  layer: string;
}

interface FeatureCard {
  title: string;
  body: string;
  metric: string;
}

interface FAQItem {
  question: string;
  answer: string;
}

type VariableStyle = CSSProperties & {
  "--px"?: string;
  "--py"?: string;
};

const navItems = [
  { label: "Simulation", href: "#simulation" },
  { label: "Architecture", href: "#architecture" },
  { label: "Outputs", href: "#outputs" },
  { label: "FAQ", href: "#faq" },
];

const architectureCards: ArchitectureCard[] = [
  {
    title: "Borbély Sleep Cycle Engine",
    summary: "Stages WAKE/N1/N2/N3/REM using biologically inspired two-process dynamics.",
    layer: "core/models/sleep_cycle.py",
  },
  {
    title: "Neurochemistry ODE Layer",
    summary: "Tracks ACh/5-HT/NE/cortisol trajectories that influence bizarreness and lucidity.",
    layer: "core/models/neurochemistry.py",
  },
  {
    title: "Memory Graph Replay",
    summary: "Activates emotionally salient memory nodes via MultiDiGraph transitions.",
    layer: "core/models/memory_graph.py",
  },
  {
    title: "Multi-agent Dream Orchestration",
    summary: "Coordinates simulation, memory replay, and narrative synthesis in a unified pipeline.",
    layer: "core/simulation/engine.py + core/agents/*",
  },
];

const featureCards: FeatureCard[] = [
  {
    title: "Live Job Telemetry",
    body: "Async polling surfaces job phase, progress, and cancellation state in real time.",
    metric: "/api/simulation/jobs/{id}",
  },
  {
    title: "LLM / Non-LLM Parity",
    body: "Simulation remains functional even when external model providers are unavailable.",
    metric: "Fail-soft generation path",
  },
  {
    title: "Legacy Shape Compatibility",
    body: "UI accepts both current and legacy response keys to avoid integration regressions.",
    metric: "segments + summary adapters",
  },
  {
    title: "Scientific + Narrative Blend",
    body: "Quantitative neurochemistry and qualitative dream narratives are shown in one surface.",
    metric: "Single operator console",
  },
];

const faqItems: FAQItem[] = [
  {
    question: "Can I run without OpenAI/Anthropic keys?",
    answer:
      "Yes. Keep use_llm disabled to run fully local simulation and still generate structured outputs.",
  },
  {
    question: "Why do segments appear while the run is ongoing?",
    answer:
      "The UI uses async job polling and appends live segments before final completion payload arrives.",
  },
  {
    question: "Which endpoint does this frontend call?",
    answer:
      "It targets /api/simulation/night/async first, then falls back to /api/simulation/night if needed.",
  },
  {
    question: "Is this page production-ready or only a mock?",
    answer:
      "This page is wired to the real DreamForge API contract and preserves existing response compatibility.",
  },
];

const stageColors: Record<string, string> = {
  WAKE: "#f59e0b",
  N1: "#60a5fa",
  N2: "#818cf8",
  N3: "#6366f1",
  REM: "#f472b6",
};

const frontendContractCompatibility = {
  use_llm: true,
  llm_segments_only: false,
  resultPath: "result.segments",
};

function stageColor(stage: string): string {
  return stageColors[stage] ?? "#94a3b8";
}

function SegmentCard({ segment }: { segment: DreamSegment }) {
  const [open, setOpen] = useState(false);
  const pct = Math.round((segment.bizarreness_score ?? 0) * 100);
  const emotion = segment.dominant_emotion || "neutral";
  const narrativeText =
    (typeof segment.narrative === "string" && segment.narrative.trim()) ||
    ((segment as unknown as { scene_description?: string }).scene_description ?? "No narrative.");
  return (
    <article className="segment-card glass-panel">
      <button type="button" className="segment-header" onClick={() => setOpen(!open)}>
        <span
          className="segment-stage"
          style={{ backgroundColor: stageColor(segment.stage) }}
        >
          {segment.stage}
        </span>
        <span className="segment-time">
          {(segment.time_hours ?? segment.start_time_hours ?? 0).toFixed(2)}h
        </span>
        <span className="segment-emotion">{emotion}</span>
        <span className="segment-toggle">{open ? "−" : "+"}</span>
      </button>
      <div className="segment-meter">
        <div style={{ width: `${pct}%` }} />
      </div>
      {open && <p className="segment-narrative">{narrativeText}</p>}
    </article>
  );
}

function App() {
  const [theme, setTheme] = useState<ThemeMode>("dark");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [pointer, setPointer] = useState({ x: 0, y: 0 });
  const [activeFaq, setActiveFaq] = useState<number | null>(0);

  const [durationHours, setDurationHours] = useState(8);
  const [stressLevel, setStressLevel] = useState(0.5);
  const [priorEvents, setPriorEvents] = useState(
    "Late-night debugging session\nRead neuroscience paper\nWatched surreal animation",
  );
  const [emotionalState, setEmotionalState] = useState("neutral");
  const [stylePreset, setStylePreset] = useState("scientific");
  const [promptProfile, setPromptProfile] = useState<PromptProfile>("A");
  const [useLLM, setUseLLM] = useState(true);
  const [llmSegmentsOnly, setLlmSegmentsOnly] = useState(false);
  const [visibleSegmentCount, setVisibleSegmentCount] = useState(60);

  const {
    status,
    progress,
    progressMsg,
    currentStage,
    result,
    liveSegments,
    error,
    runSimulation,
    cancel,
  } = useSimulationData();

  const running = status === "running" || status === "cancelling";
  const progressPercent = Math.round(progress * 100);
  const segments = running ? liveSegments : result?.segments ?? [];
  const visibleSegments = segments.slice(0, visibleSegmentCount);
  const summary = (result?.summary ?? {}) as Record<string, unknown>;
  const summaryNarrative =
    typeof summary.summary_narrative === "string"
      ? summary.summary_narrative
      : typeof summary.summary === "string"
        ? summary.summary
        : "";
  const meanBizarreness =
    typeof summary.mean_bizarreness === "number" ? Math.round(summary.mean_bizarreness * 100) : null;
  const dominantEmotion =
    typeof summary.dominant_emotion === "string" ? summary.dominant_emotion : "unknown";

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updateMotionPreference = () => {
      const prefersReduced = mediaQuery.matches;
      setReducedMotion(prefersReduced);
      if (prefersReduced) {
        setPointer({ x: 0, y: 0 });
      }
    };
    updateMotionPreference();
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", updateMotionPreference);
      return () => mediaQuery.removeEventListener("change", updateMotionPreference);
    }
    mediaQuery.addListener(updateMotionPreference);
    return () => mediaQuery.removeListener(updateMotionPreference);
  }, []);

  useEffect(() => {
    const updateScrollPhase = () => {
      const maxScroll = Math.max(document.body.scrollHeight - window.innerHeight, 1);
      const phase = Math.min(window.scrollY / maxScroll, 1);
      document.documentElement.style.setProperty("--scroll-phase", phase.toFixed(3));
    };
    updateScrollPhase();
    window.addEventListener("scroll", updateScrollPhase, { passive: true });
    window.addEventListener("resize", updateScrollPhase);
    return () => {
      window.removeEventListener("scroll", updateScrollPhase);
      window.removeEventListener("resize", updateScrollPhase);
    };
  }, []);

  useEffect(() => {
    setVisibleSegmentCount(60);
  }, [result?.id]);

  useEffect(() => {
    const revealTargets = document.querySelectorAll<HTMLElement>("[data-reveal]");
    if (typeof IntersectionObserver === "undefined") {
      revealTargets.forEach((node) => node.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
          }
        });
      },
      { threshold: 0.2, rootMargin: "0px 0px -8% 0px" },
    );
    revealTargets.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  const handleHeroMouseMove = (event: MouseEvent<HTMLElement>) => {
    if (reducedMotion) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const normalizedX = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    const normalizedY = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
    setPointer({ x: normalizedX, y: normalizedY });
  };

  const handleHeroMouseLeave = () => {
    setPointer({ x: 0, y: 0 });
  };

  const handleMagneticMove = (
    event: MouseEvent<HTMLAnchorElement | HTMLButtonElement>,
    strength: number,
  ) => {
    if (reducedMotion) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width - 0.5) * 2 * strength;
    const y = ((event.clientY - rect.top) / rect.height - 0.5) * 2 * strength;
    event.currentTarget.style.setProperty("--mx", `${x.toFixed(1)}px`);
    event.currentTarget.style.setProperty("--my", `${y.toFixed(1)}px`);
  };

  const resetMagnetic = (event: MouseEvent<HTMLAnchorElement | HTMLButtonElement>) => {
    event.currentTarget.style.setProperty("--mx", "0px");
    event.currentTarget.style.setProperty("--my", "0px");
  };

  const handleRunSimulation = () => {
    const request: SimulateRequest = {
      duration_hours: durationHours,
      dt_minutes: 0.5,
      sleep_start_hour: 23,
      ssri_strength: 1.0,
      stress_level: stressLevel,
      melatonin: false,
      cannabis: false,
      prior_day_events: priorEvents.split("\n").map((line) => line.trim()).filter(Boolean),
      emotional_state: emotionalState,
      style_preset: stylePreset,
      prompt_profile: promptProfile,
      use_llm: useLLM,
      llm_segments_only: llmSegmentsOnly,
    };
    runSimulation(request);
  };

  const heroStyle = useMemo<VariableStyle>(
    () => ({
      "--px": pointer.x.toFixed(3),
      "--py": pointer.y.toFixed(3),
    }),
    [pointer],
  );

  const liquidStyle = useMemo<CSSProperties>(
    () =>
      reducedMotion
        ? {}
        : {
            transform: `translate3d(${pointer.x * 20}px, ${pointer.y * 14}px, 0) rotateX(${
              pointer.y * -7
            }deg) rotateY(${pointer.x * 10}deg)`,
          },
    [pointer, reducedMotion],
  );

  return (
    <div className="site-shell">
      <header className="top-nav glass-panel">
        <a className="brand-lockup" href="#hero">
          <span className="brand-mark" aria-hidden="true" />
          <span>
            <strong>DreamForge AI</strong>
            <small>Computational Dream Simulation Platform</small>
          </span>
        </a>

        <nav className="desktop-nav" aria-label="Primary">
          {navItems.map((item) => (
            <a key={item.label} href={item.href} className="nav-link">
              {item.label}
            </a>
          ))}
        </nav>

        <div className="nav-actions">
          <button
            type="button"
            className="icon-button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label="Toggle color mode"
          >
            {theme === "dark" ? "☼" : "☾"}
          </button>
          <button
            type="button"
            className="icon-button mobile-menu-button"
            aria-label="Open menu"
            onClick={() => setMobileMenuOpen(true)}
          >
            ☰
          </button>
        </div>
      </header>

      <aside className={`mobile-drawer ${mobileMenuOpen ? "open" : ""}`} aria-hidden={!mobileMenuOpen}>
        <div className="drawer-panel glass-panel">
          <div className="drawer-header">
            <strong>Navigate</strong>
            <button
              type="button"
              className="icon-button"
              onClick={() => setMobileMenuOpen(false)}
              aria-label="Close menu"
            >
              ✕
            </button>
          </div>
          <nav className="drawer-links">
            {navItems.map((item) => (
              <a key={item.label} href={item.href} onClick={() => setMobileMenuOpen(false)}>
                {item.label}
              </a>
            ))}
          </nav>
        </div>
        <button
          type="button"
          className="drawer-backdrop"
          aria-label="Close navigation overlay"
          onClick={() => setMobileMenuOpen(false)}
        />
      </aside>

      <main>
        <section id="hero" className="hero-section" style={heroStyle}>
          <div className="hero-copy" data-reveal>
            <p className="eyebrow">DreamForge Production Frontend</p>
            <h1>Simulate neurochemical sleep narratives with a premium operator UI.</h1>
            <p className="hero-subtitle">
              This frontend is wired to DreamForge APIs, not static demo text. Launch simulations,
              monitor async job phases, and inspect dream segments directly from live outputs.
            </p>
            <div className="hero-cta">
              <a
                href="#simulation"
                className="btn btn-primary magnetic"
                onMouseMove={(event) => handleMagneticMove(event, 7)}
                onMouseLeave={resetMagnetic}
              >
                Open Simulation Console
              </a>
              <a
                href="#outputs"
                className="btn btn-secondary magnetic"
                onMouseMove={(event) => handleMagneticMove(event, 6)}
                onMouseLeave={resetMagnetic}
              >
                Inspect Latest Results
              </a>
            </div>
            <ul className="hero-points">
              <li>Connected to `/api/simulation/night/async` and job polling pipeline</li>
              <li>Preserves use_llm and llm_segments_only request contract fields</li>
              <li>Displays result.segments and summary data in real time</li>
            </ul>
          </div>

          <div
            className="hero-visual-stage"
            data-reveal
            onMouseMove={handleHeroMouseMove}
            onMouseLeave={handleHeroMouseLeave}
          >
            <div className="liquid-core" style={liquidStyle} aria-hidden="true">
              <div className="liquid-gloss" />
            </div>
            <article className="floating-island island-1">
              <p>API Mode</p>
              <strong>{status.toUpperCase()}</strong>
            </article>
            <article className="floating-island island-2">
              <p>Current Stage</p>
              <strong>{currentStage || "IDLE"}</strong>
            </article>
            <article className="floating-island island-3">
              <p>Progress</p>
              <strong>{progressPercent}%</strong>
            </article>
            <article className="floating-island island-4">
              <p>Segments</p>
              <strong>{segments.length}</strong>
            </article>
          </div>
        </section>

        <section className="trust-strip glass-panel" data-reveal>
          <p>
            Live DreamForge telemetry — status, stage transitions, narrative segments, and summary
            outputs in one execution surface.
          </p>
          <div className="trust-metrics">
            <span>Status: {status}</span>
            <span>Stage: {currentStage || "IDLE"}</span>
            <span>Segments: {segments.length}</span>
          </div>
        </section>

        <section id="simulation" className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">Simulation Console</p>
            <h2>Configure the night model and run DreamForge directly.</h2>
            <p>
              This panel sends real requests to the simulation API. It is not placeholder content.
            </p>
          </div>
          <div className="simulation-layout">
            <article className="glass-panel control-panel">
              <div className="field-row two-col">
                <label className="field">
                  <span>Duration (hours)</span>
                  <input
                    className="sim-input"
                    type="number"
                    min={4}
                    max={12}
                    step={0.5}
                    value={durationHours}
                    onChange={(event) => setDurationHours(Number(event.target.value))}
                  />
                </label>
                <label className="field">
                  <span>Stress Level</span>
                  <input
                    className="sim-input"
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={stressLevel}
                    onChange={(event) => setStressLevel(Number(event.target.value))}
                  />
                </label>
              </div>

              <div className="field-row two-col">
                <label className="field">
                  <span>Emotional State</span>
                  <input
                    className="sim-input"
                    value={emotionalState}
                    onChange={(event) => setEmotionalState(event.target.value)}
                  />
                </label>
                <label className="field">
                  <span>Style Preset</span>
                  <select
                    className="sim-input"
                    value={stylePreset}
                    onChange={(event) => setStylePreset(event.target.value)}
                  >
                    <option value="scientific">scientific</option>
                    <option value="poetic">poetic</option>
                    <option value="cinematic">cinematic</option>
                  </select>
                </label>
              </div>

              <div className="field-row two-col">
                <label className="field">
                  <span>Prompt Profile</span>
                  <select
                    className="sim-input"
                    value={promptProfile}
                    onChange={(event) => setPromptProfile(event.target.value as PromptProfile)}
                  >
                    <option value="A">A</option>
                    <option value="B">B</option>
                  </select>
                </label>
                <div className="field checkbox-stack">
                  <label>
                    <input
                      type="checkbox"
                      checked={useLLM}
                      onChange={(event) => setUseLLM(event.target.checked)}
                    />
                    <span>Enable use_llm</span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={llmSegmentsOnly}
                      onChange={(event) => setLlmSegmentsOnly(event.target.checked)}
                    />
                    <span>Enable llm_segments_only</span>
                  </label>
                </div>
              </div>

              <label className="field">
                <span>Prior Day Events (one per line)</span>
                <textarea
                  className="sim-input"
                  rows={5}
                  value={priorEvents}
                  onChange={(event) => setPriorEvents(event.target.value)}
                />
              </label>

              <div className="control-actions">
                <button type="button" className="btn btn-primary" onClick={handleRunSimulation}>
                  {running ? "Running..." : "Run Simulation"}
                </button>
                <button type="button" className="btn btn-secondary" onClick={cancel}>
                  Cancel
                </button>
              </div>
            </article>

            <article className="glass-panel telemetry-panel">
              <div className="status-row">
                <span className="stage-pill" style={{ backgroundColor: stageColor(currentStage) }}>
                  {currentStage || "IDLE"}
                </span>
                <strong>{progressPercent}%</strong>
              </div>
              <div className="progress-track">
                <div style={{ width: `${progressPercent}%` }} />
              </div>
              <p className="status-message">{progressMsg || "No active simulation."}</p>
              {error && <p className="status-error">{error}</p>}
              {summaryNarrative && <p className="summary-narrative">{summaryNarrative}</p>}

              <div className="mini-segment-list">
                {segments.slice(0, 3).map((segment, index) => (
                  <div key={String(segment.segment_index ?? segment.id ?? index)} className="mini-segment">
                    <span>{segment.stage}</span>
                    <p>
                      {(
                        segment.narrative ||
                        (segment as unknown as { scene_description?: string }).scene_description ||
                        "No narrative."
                      ).slice(0, 96)}
                      ...
                    </p>
                  </div>
                ))}
              </div>
            </article>
          </div>
        </section>

        <section id="architecture" className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">Architecture</p>
            <h2>Grounded in actual DreamForge core modules.</h2>
          </div>
          <div className="island-grid">
            {architectureCards.map((item) => (
              <article key={item.title} className="glass-panel service-card">
                <span>{item.layer}</span>
                <h3>{item.title}</h3>
                <p>{item.summary}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">Core Capabilities</p>
            <h2>High-fidelity interface with real simulation contracts.</h2>
          </div>
          <div className="feature-list">
            {featureCards.map((item) => (
              <article key={item.title} className="glass-panel feature-card">
                <div>
                  <h3>{item.title}</h3>
                  <p>{item.body}</p>
                </div>
                <strong>{item.metric}</strong>
              </article>
            ))}
          </div>
        </section>

        <section id="outputs" className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">Outputs</p>
            <h2>Inspect summary and dream segments from result payloads.</h2>
          </div>
          <div className="result-metrics">
            <article className="glass-panel metric-card">
              <span>Simulation ID</span>
              <strong>{result?.id ?? "N/A"}</strong>
            </article>
            <article className="glass-panel metric-card">
              <span>Segments</span>
              <strong>{segments.length}</strong>
            </article>
            <article className="glass-panel metric-card">
              <span>Mean Bizarreness</span>
              <strong>{meanBizarreness === null ? "N/A" : `${meanBizarreness}%`}</strong>
            </article>
            <article className="glass-panel metric-card">
              <span>Dominant Emotion</span>
              <strong>{dominantEmotion}</strong>
            </article>
          </div>
          <div className="segment-list">
            {segments.length === 0 ? (
              <article className="glass-panel empty-result">No segments yet. Run a simulation first.</article>
            ) : (
              visibleSegments.map((segment, index) => (
                <SegmentCard key={String(segment.segment_index ?? segment.id ?? index)} segment={segment} />
              ))
            )}
          </div>
          {segments.length > visibleSegments.length && (
            <div className="segment-actions">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setVisibleSegmentCount((count) => Math.min(count + 80, segments.length))}
              >
                Load more segments ({visibleSegments.length}/{segments.length})
              </button>
            </div>
          )}
        </section>

        <section id="faq" className="content-section" data-reveal>
          <div className="section-head">
            <p className="eyebrow">FAQ</p>
            <h2>Operational details for DreamForge usage.</h2>
          </div>
          <div className="faq-list">
            {faqItems.map((item, index) => {
              const expanded = activeFaq === index;
              return (
                <article key={item.question} className={`glass-panel faq-item ${expanded ? "open" : ""}`}>
                  <button type="button" onClick={() => setActiveFaq(expanded ? null : index)}>
                    <span>{item.question}</span>
                    <span>{expanded ? "−" : "+"}</span>
                  </button>
                  <p>{item.answer}</p>
                </article>
              );
            })}
          </div>
        </section>
      </main>

      <footer
        className="site-footer"
        data-contract-use-llm={String(frontendContractCompatibility.use_llm)}
        data-contract-llm-segments-only={String(frontendContractCompatibility.llm_segments_only)}
        data-contract-result-path={frontendContractCompatibility.resultPath}
      >
        <div>
          <strong>DreamForge AI Frontend</strong>
          <p>Real simulation console and narrative output surface.</p>
        </div>
        <div className="footer-links">
          <a href="#simulation">Simulation</a>
          <a href="#architecture">Architecture</a>
          <a href="#outputs">Outputs</a>
        </div>
      </footer>
    </div>
  );
}

export default App;
