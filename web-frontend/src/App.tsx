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

interface CompareResponse {
  baseline_id: string;
  candidate_id: string;
  delta: {
    mean_bizarreness: number;
    rem_fraction: number;
    lucid_event_count: number;
    narrative_quality_mean: number;
    narrative_memory_grounding_mean?: number;
    llm_fallback_rate?: number;
  };
  confidence?: {
    sample_size?: number;
  };
  stage_minutes?: {
    baseline?: Record<string, number>;
    candidate?: Record<string, number>;
  };
  anomaly_flags?: string[];
}

interface LlmRunSummary {
  simulation_id: string;
  source: "llm" | "fallback";
  llm_used: boolean;
  llm_model?: string | null;
  generated_at_unix: number;
  executive_summary: string;
  key_findings: string[];
  risk_signals: string[];
  recommended_actions: string[];
  next_run_profile: {
    duration_hours: number;
    stress_level: number;
    style_preset: string;
    prompt_profile: string;
  };
}

type VariableStyle = CSSProperties & {
  "--px"?: string;
  "--py"?: string;
};

const navItems = [
  { label: "Simulation", href: "#simulation" },
  { label: "Compare", href: "#compare" },
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

const emotionalStateOptions = [
  "neutral",
  "calm",
  "joy",
  "curious",
  "anxious",
  "sadness",
  "fear",
  "anger",
  "surprise",
  "disgust",
];

const promptProfileMap: Record<PromptProfile, { label: string; description: string }> = {
  A: {
    label: "Analytical Core",
    description:
      "Balanced scientific tone with concise narrative output, ideal for metric-first review.",
  },
  B: {
    label: "Narrative Rich",
    description:
      "More expressive storytelling with stronger scene texture while preserving structured signals.",
  },
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
  const [formError, setFormError] = useState<string | null>(null);
  const [compareBaselineId, setCompareBaselineId] = useState("");
  const [compareCandidateId, setCompareCandidateId] = useState("");
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [recentSimulationIds, setRecentSimulationIds] = useState<string[]>([]);
  const [llmRunSummary, setLlmRunSummary] = useState<LlmRunSummary | null>(null);
  const [llmRunSummaryLoading, setLlmRunSummaryLoading] = useState(false);
  const [llmRunSummaryError, setLlmRunSummaryError] = useState<string | null>(null);

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
  const priorDayEvents = priorEvents.split("\n").map((line) => line.trim()).filter(Boolean);
  const durationInRange = Number.isFinite(durationHours) && durationHours >= 4 && durationHours <= 12;
  const stressInRange = Number.isFinite(stressLevel) && stressLevel >= 0 && stressLevel <= 1;
  const compareStageKeys = useMemo(() => {
    if (!compareResult?.stage_minutes) {
      return [];
    }
    const baseline = compareResult.stage_minutes.baseline ?? {};
    const candidate = compareResult.stage_minutes.candidate ?? {};
    return Array.from(new Set([...Object.keys(baseline), ...Object.keys(candidate)])).sort();
  }, [compareResult]);
  const activePromptProfile = promptProfileMap[promptProfile];

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
    if (!result?.id) {
      setLlmRunSummary(null);
      setLlmRunSummaryError(null);
      setLlmRunSummaryLoading(false);
      return;
    }
    setRecentSimulationIds((current) => {
      const next = [result.id, ...current.filter((id) => id !== result.id)];
      return next.slice(0, 8);
    });
    setCompareBaselineId((current) => current || result.id);
    let cancelled = false;
    const loadLlmSummary = async () => {
      setLlmRunSummaryLoading(true);
      setLlmRunSummaryError(null);
      try {
        const response = await fetch(`/api/simulation/${result.id}/llm-summary`);
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || `LLM summary failed (${response.status})`);
        }
        const payload = (await response.json()) as LlmRunSummary;
        if (!cancelled) {
          setLlmRunSummary(payload);
        }
      } catch (caughtError: unknown) {
        const message =
          caughtError instanceof Error ? caughtError.message : "Failed to load LLM post-run summary.";
        if (!cancelled) {
          setLlmRunSummary(null);
          setLlmRunSummaryError(message);
        }
      } finally {
        if (!cancelled) {
          setLlmRunSummaryLoading(false);
        }
      }
    };
    void loadLlmSummary();
    return () => {
      cancelled = true;
    };
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
    if (!durationInRange) {
      setFormError("Duration must be between 4.0 and 12.0 hours.");
      return;
    }
    if (!stressInRange) {
      setFormError("Stress level must be between 0.00 and 1.00.");
      return;
    }
    if (priorDayEvents.length === 0) {
      setFormError("Please add at least one prior-day event.");
      return;
    }
    setFormError(null);
    const request: SimulateRequest = {
      duration_hours: durationHours,
      dt_minutes: 0.5,
      sleep_start_hour: 23,
      ssri_strength: 1.0,
      stress_level: stressLevel,
      melatonin: false,
      cannabis: false,
      prior_day_events: priorDayEvents,
      emotional_state: emotionalState,
      style_preset: stylePreset,
      prompt_profile: promptProfile,
      use_llm: useLLM,
      llm_segments_only: llmSegmentsOnly,
    };
    runSimulation(request);
  };

  const handleCompareSimulations = async () => {
    if (!compareBaselineId.trim() || !compareCandidateId.trim()) {
      setCompareError("Please provide both baseline and candidate simulation IDs.");
      return;
    }
    if (compareBaselineId.trim() === compareCandidateId.trim()) {
      setCompareError("Baseline and candidate IDs must be different.");
      return;
    }
    setCompareLoading(true);
    setCompareError(null);
    try {
      const response = await fetch("/api/simulation/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseline_simulation_id: compareBaselineId.trim(),
          candidate_simulation_id: compareCandidateId.trim(),
        }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Compare failed (${response.status})`);
      }
      const payload = (await response.json()) as CompareResponse;
      setCompareResult(payload);
    } catch (caughtError: unknown) {
      const message = caughtError instanceof Error ? caughtError.message : "Failed to compare simulations.";
      setCompareResult(null);
      setCompareError(message);
    } finally {
      setCompareLoading(false);
    }
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

  const formatDeltaPercent = (value: number): string =>
    `${value >= 0 ? "+" : ""}${Math.round(value * 10000) / 100}%`;

  const formatDeltaInteger = (value: number): string => `${value >= 0 ? "+" : ""}${value}`;

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
                    className={`sim-input${durationInRange ? "" : " input-invalid"}`}
                    type="number"
                    min={4}
                    max={12}
                    step={0.5}
                    value={durationHours}
                    onChange={(event) => setDurationHours(Number(event.target.value))}
                  />
                  <small className={durationInRange ? "field-hint" : "field-error"}>
                    Recommended range: 4.0 to 12.0 hours.
                  </small>
                </label>
                <label className="field">
                  <span>Stress Level</span>
                  <input
                    className={`sim-input${stressInRange ? "" : " input-invalid"}`}
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={stressLevel}
                    onChange={(event) => setStressLevel(Number(event.target.value))}
                  />
                  <small className={stressInRange ? "field-hint" : "field-error"}>
                    Valid range: 0.00 (low) to 1.00 (high).
                  </small>
                </label>
              </div>

              <div className="field-row two-col">
                <label className="field">
                  <span>Emotional State</span>
                  <select
                    className="sim-input"
                    value={emotionalState}
                    onChange={(event) => setEmotionalState(event.target.value)}
                  >
                    {emotionalStateOptions.map((emotion) => (
                      <option key={emotion} value={emotion}>
                        {emotion}
                      </option>
                    ))}
                  </select>
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
                    <option value="A">A — {promptProfileMap.A.label}</option>
                    <option value="B">B — {promptProfileMap.B.label}</option>
                  </select>
                  <small className="field-hint">{activePromptProfile.description}</small>
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
                  className={`sim-input${priorDayEvents.length > 0 ? "" : " input-invalid"}`}
                  rows={5}
                  value={priorEvents}
                  onChange={(event) => setPriorEvents(event.target.value)}
                />
                <small className={priorDayEvents.length > 0 ? "field-hint" : "field-error"}>
                  {priorDayEvents.length} event(s) detected. Add at least one line.
                </small>
              </label>

              {formError && <p className="status-error">{formError}</p>}

              <div className="control-actions">
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={handleRunSimulation}
                  disabled={running}
                >
                  {running ? "Running..." : "Run Simulation"}
                </button>
                <button type="button" className="btn btn-secondary" onClick={cancel} disabled={!running}>
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
          <article id="compare" className="glass-panel compare-panel">
            <div className="compare-head">
              <h3>Compare Simulations</h3>
              <p>Run side-by-side delta analysis using two simulation IDs.</p>
            </div>
            <div className="field-row two-col">
              <label className="field">
                <span>Baseline Simulation ID</span>
                <input
                  className="sim-input"
                  list="recent-simulation-ids"
                  value={compareBaselineId}
                  onChange={(event) => setCompareBaselineId(event.target.value)}
                  placeholder="e.g. sim_abc123"
                />
              </label>
              <label className="field">
                <span>Candidate Simulation ID</span>
                <input
                  className="sim-input"
                  list="recent-simulation-ids"
                  value={compareCandidateId}
                  onChange={(event) => setCompareCandidateId(event.target.value)}
                  placeholder="e.g. sim_xyz789"
                />
              </label>
            </div>
            <datalist id="recent-simulation-ids">
              {recentSimulationIds.map((id) => (
                <option key={id} value={id} />
              ))}
            </datalist>
            <div className="control-actions">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => result?.id && setCompareBaselineId(result.id)}
                disabled={!result?.id}
              >
                Use latest as baseline
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => result?.id && setCompareCandidateId(result.id)}
                disabled={!result?.id}
              >
                Use latest as candidate
              </button>
              <button type="button" className="btn btn-primary" onClick={handleCompareSimulations}>
                {compareLoading ? "Comparing..." : "Compare"}
              </button>
            </div>
            {compareError && <p className="status-error">{compareError}</p>}
            {compareResult && (
              <div className="compare-result">
                <div className="result-metrics compare-metrics">
                  <article className="glass-panel metric-card">
                    <span>Mean Bizarreness Δ</span>
                    <strong>{formatDeltaPercent(compareResult.delta.mean_bizarreness)}</strong>
                  </article>
                  <article className="glass-panel metric-card">
                    <span>REM Fraction Δ</span>
                    <strong>{formatDeltaPercent(compareResult.delta.rem_fraction)}</strong>
                  </article>
                  <article className="glass-panel metric-card">
                    <span>Narrative Quality Δ</span>
                    <strong>{formatDeltaPercent(compareResult.delta.narrative_quality_mean)}</strong>
                  </article>
                  <article className="glass-panel metric-card">
                    <span>Lucid Event Δ</span>
                    <strong>{formatDeltaInteger(compareResult.delta.lucid_event_count)}</strong>
                  </article>
                  {typeof compareResult.delta.narrative_memory_grounding_mean === "number" && (
                    <article className="glass-panel metric-card">
                      <span>Memory Grounding Δ</span>
                      <strong>{formatDeltaPercent(compareResult.delta.narrative_memory_grounding_mean)}</strong>
                    </article>
                  )}
                  {typeof compareResult.delta.llm_fallback_rate === "number" && (
                    <article className="glass-panel metric-card">
                      <span>LLM Fallback Rate Δ</span>
                      <strong>{formatDeltaPercent(compareResult.delta.llm_fallback_rate)}</strong>
                    </article>
                  )}
                </div>
                <div className="compare-detail-grid">
                  <article className="glass-panel compare-subcard">
                    <span>Anomaly Flags</span>
                    <p>
                      {compareResult.anomaly_flags && compareResult.anomaly_flags.length > 0
                        ? compareResult.anomaly_flags.join(", ")
                        : "No anomaly flags."}
                    </p>
                  </article>
                  <article className="glass-panel compare-subcard">
                    <span>Sample Size</span>
                    <p>{compareResult.confidence?.sample_size ?? "N/A"} overlapping segment(s).</p>
                  </article>
                </div>
                {compareStageKeys.length > 0 && (
                  <div className="compare-stage-table">
                    <div className="compare-stage-row compare-stage-head">
                      <span>Stage</span>
                      <span>Baseline (min)</span>
                      <span>Candidate (min)</span>
                    </div>
                    {compareStageKeys.map((stage) => (
                      <div key={stage} className="compare-stage-row">
                        <span>{stage}</span>
                        <span>{Math.round((compareResult.stage_minutes?.baseline?.[stage] ?? 0) * 10) / 10}</span>
                        <span>{Math.round((compareResult.stage_minutes?.candidate?.[stage] ?? 0) * 10) / 10}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </article>
          <article className="glass-panel llm-summary-panel">
            <div className="compare-head">
              <h3>Post-run Intelligence Summary</h3>
              <p>
                Automatically generated insights after each run, with actionable next-run recommendations.
              </p>
            </div>
            {llmRunSummaryLoading && <p className="status-message">Generating LLM summary...</p>}
            {llmRunSummaryError && <p className="status-error">{llmRunSummaryError}</p>}
            {!llmRunSummaryLoading && !llmRunSummaryError && !llmRunSummary && (
              <p className="status-message">Run a simulation to generate the post-run summary.</p>
            )}
            {llmRunSummary && (
              <div className="llm-summary-content">
                <div className="llm-summary-meta">
                  <span className="stage-pill llm-source-pill">
                    {llmRunSummary.source === "llm" ? "LLM Summary" : "Fallback Summary"}
                  </span>
                  <span>
                    Model: {llmRunSummary.llm_model ?? "N/A"} · LLM used in run:{" "}
                    {llmRunSummary.llm_used ? "Yes" : "No"}
                  </span>
                </div>
                <p className="summary-narrative">{llmRunSummary.executive_summary}</p>
                <div className="compare-detail-grid">
                  <article className="glass-panel compare-subcard">
                    <span>Key Findings</span>
                    <ul className="insight-list">
                      {llmRunSummary.key_findings.map((item, index) => (
                        <li key={`${index}-${item}`}>{item}</li>
                      ))}
                    </ul>
                  </article>
                  <article className="glass-panel compare-subcard">
                    <span>Risk Signals</span>
                    <ul className="insight-list">
                      {llmRunSummary.risk_signals.map((item, index) => (
                        <li key={`${index}-${item}`}>{item}</li>
                      ))}
                    </ul>
                  </article>
                </div>
                <article className="glass-panel compare-subcard">
                  <span>Recommended Actions</span>
                  <ul className="insight-list">
                    {llmRunSummary.recommended_actions.map((item, index) => (
                      <li key={`${index}-${item}`}>{item}</li>
                    ))}
                  </ul>
                </article>
                <div className="result-metrics compare-metrics">
                  <article className="glass-panel metric-card">
                    <span>Next Duration (h)</span>
                    <strong>{llmRunSummary.next_run_profile.duration_hours}</strong>
                  </article>
                  <article className="glass-panel metric-card">
                    <span>Next Stress</span>
                    <strong>{llmRunSummary.next_run_profile.stress_level}</strong>
                  </article>
                  <article className="glass-panel metric-card">
                    <span>Next Style Preset</span>
                    <strong>{llmRunSummary.next_run_profile.style_preset}</strong>
                  </article>
                  <article className="glass-panel metric-card">
                    <span>Next Prompt Profile</span>
                    <strong>{llmRunSummary.next_run_profile.prompt_profile}</strong>
                  </article>
                </div>
              </div>
            )}
          </article>
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
