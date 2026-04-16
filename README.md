# DreamForge AI

> "The first open-source AI system that thinks while it sleeps."

DreamForge AI is a hierarchical multi-agent framework that simulates an entire night of sleep – from biophysical dynamics and neuromodulators to narrative dream content and phenomenological reports.

It combines:
- A neuroscience-grounded sleep architecture (Borbély two-process model, REM/NREM structure).
- Stage-dependent neurochemistry (acetylcholine, serotonin, noradrenaline, cortisol).
- Memory consolidation over a weighted, emotionally tagged knowledge graph.
- Dream content generation via LLM-backed agents.
- Metacognitive and phenomenological layers (lucidity, bizarreness, first-person reports).
- A real-time interactive dashboard for watching "dreams" unfold.

If DreamForge sparks your curiosity, a star helps others find it. ⭐

---

## 60-second demo (concept)

> In the live demo, you watch a hypnogram scroll across the screen while colored lines for ACh, 5‑HT, NE, and cortisol pulse underneath. Memory fragments in a force-directed graph light up as hippocampal replay cascades through emotionally salient nodes, and a dream timeline fills with segments annotated by bizarreness and emotion.

(Animated GIF placeholder — to be recorded once the dashboard is feature-complete.)

---

## Why this matters

Modern LLMs are powerful sequence models, but they do not explicitly model sleep, dreaming, or memory consolidation. DreamForge explores what happens when we bring together computational neuroscience and agentic AI to:

- Test concrete hypotheses about sleep homeostasis, neuromodulators, and dream content.
- Provide a hands-on playground for dream bizarreness metrics and lucidity modeling.
- Offer an open-source reference implementation for multi-agent scientific simulators.

This project is **not** a clinical tool or a replacement for empirical sleep research; it is a conceptual and educational framework built with scientific humility and explicit limitations.

---

## Architecture at a glance

DreamForge is organized into a 7-layer agent hierarchy, orchestrated over a typed event bus:

- **Layer 0 – OrchestratorAgent**
  - Owns the simulation clock and coordinates all other agents.
- **Layer 1 – SleepCycleAgent**
  - Implements Borbély's two-process model (Process S + Process C) and discrete wake/N1/N2/N3/REM staging.
- **Layer 2 – NeurochemistryAgent**
  - Simulates ACh, 5-HT, NE, and cortisol trajectories conditioned on sleep stage and pharmacology.
- **Layer 3 – MemoryConsolidationAgent**
  - Manages a weighted, emotionally tagged memory graph and hippocampal replay events.
- **Layer 4 – DreamConstructorAgent**
  - Composes narrative dream segments from sleep stage, neurochemistry, and active memories (with optional LLM integration).
- **Layer 5 – MetacognitiveAgent**
  - Tracks lucidity probability and dream-logic consistency.
- **Layer 6 – PhenomenologyReporter**
  - Produces first-person narratives, affect labels, and structured dream files.

The full architecture and design rationale are documented in `ARCHITECTURE.md` and `RESEARCH.md`.

---

## Quick start (Docker, 3 commands)

```bash
git clone https://github.com/JToSound/dreamforge-ai
cd dreamforge-ai
docker-compose up --build
```

Then open:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

> Note: first build may take a few minutes while Python dependencies are installed.

---

## Core features

- **Neuroscience-grounded sleep model**
  - Two-process model (homeostatic Process S and circadian Process C).
  - Empirical REM/NREM cycle structure (N3 early night, lengthening REM towards morning).

- **Neurochemical dynamics**
  - Stage-dependent dynamics for ACh, serotonin, noradrenaline, cortisol.
  - Pharmacological modulators for SSRIs, stress hormones, and related profiles.

- **Memory graph and replay**
  - NetworkX-based graph of memory fragments with emotional tags and recency.
  - Hippocampal sharp-wave ripple–like replay as biased random walks.

- **Dream construction**
  - Dream segments conditioned on stage, neuromodulators, and replayed memories.
  - Hooks for LLM backends (GPT-4o, Claude, local models via Ollama, etc.) for narrative and scene generation.

- **Metacognition and phenomenology**
  - Lucidity probability estimates and dream-logic enforcement.
  - First-person narrative output and symbolic interpretation layer.

- **Visualization dashboard**
  - Hypnogram, neurochemical flux, memory association graph.
  - Dream content timeline and agent activity heatmap.
  - Comparative dream analysis view for side-by-side runs.

- **Novel features**
  - Pharmacological modulators to explore medication and stress effects.
  - Cross-night continuity tracking of recurring themes.
  - Counterfactual dream engine for parameter sweeps and A/B comparisons.

---

## Scientific grounding

The models in DreamForge are calibrated against:

- **Sleep architecture** – Borbély's two-process model of sleep regulation and its modern refinements.
- **Neurochemistry** – monoaminergic and cholinergic dynamics across REM/NREM cycles.
- **Memory consolidation** – hippocampal replay, emotional tagging, and selective forgetting.
- **Dream bizarreness** – metrics inspired by Revonsuo and subsequent work on dream content analysis.

For detailed equations, parameter choices, and literature references, see `RESEARCH.md`.

---

## API and dashboard

- **FastAPI REST API**
  - `POST /api/simulation/night` – run a single-night simulation with configurable parameters (duration, dt, pharmacology).
  - `POST /api/simulation/night/async` – queue a simulation and poll `/api/simulation/jobs/{job_id}`.
  - `POST /api/simulation/multi-night` – run multiple nights with cross-night continuity tracking.
  - `POST /api/simulation/counterfactual` – compare baseline vs perturbed dream runs.
  - `POST /api/simulation/compare` – compare two stored runs and compute deltas.
  - `GET /api/simulation/{id}/report` – generate a structured run report payload.
  - `GET /api/llm/registry` – prompt/model registry and capability matrix.
  - `GET /api/slo`, `GET /api/error-taxonomy`, `GET /metrics/prometheus` – operational readiness surfaces.
  - `GET /api/version` – API contract metadata (`v1`).
  - `GET /health` – simple health check.

- **Streamlit dashboard**
  - Run `streamlit run visualization/dashboard/app.py` in development, or use the `dashboard` service in Docker Compose.
  - Visualizes hypnogram, neuromodulators, memory graph, dream timeline, and agent activity heatmap in real time.
  - A separate comparative dashboard script provides multi-run visualizations.

---

## Development

Run the dashboard locally:

```bash
# local python dev
pip install -r requirements.dashboard.txt
pip install -e .
streamlit run visualization/dashboard/app.py --server.port=8501 --server.address=0.0.0.0
```

## Configuration

Runtime defaults are centralized in `core/config.py` and can be overridden via
environment variables. The most useful overrides are:

- `API_BASE_URL`
- `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`
- `OLLAMA_BASE_URL`, `LMSTUDIO_BASE_URL`
- `SIM_DURATION_HOURS`, `SIM_DT_MINUTES`, `SIM_STRESS_LEVEL`, `SIM_SLEEP_START_HOUR`

## Docker (development)

We provide a cache-friendly dashboard Dockerfile and compose setup. For local development use the editable install build (default DEV=1 in `docker-compose.yml`). Build with BuildKit enabled for best caching:

```bash
# enable BuildKit (Linux/macOS)
export DOCKER_BUILDKIT=1
docker compose build --progress=plain dashboard
docker compose up dashboard
```

## CI

The GitHub Actions workflow uses Buildx with the GHA cache (`cache-from` / `cache-to`) to persist Docker build cache between runs, speeding up subsequent builds.

## Demo GIF capture

With API + dashboard running locally, generate a short demo GIF:

```bash
pip install playwright imageio
playwright install
python scripts/generate_demo_gif.py
```

By default, frames are captured from `http://localhost:8501` and assembled into
`demo.gif` (override URL with `DEMO_URL`).

## Roadmap

- Phase 1 – Core simulation engine
  - [x] SleepCycleModel and NeurochemistryModel.
  - [x] MemoryGraph with replay and forgetting.
  - [x] SleepCycleAgent, NeurochemistryAgent, MemoryConsolidationAgent.
  - [x] DreamConstructorAgent with LLM integration hooks.
  - [x] MetacognitiveAgent and PhenomenologyReporter wired into the API.

- Phase 2 – Visualization
  - [x] Hypnogram, neurochemical flux, memory association graph.
  - [x] Dream content timeline and agent activity heatmap.
  - [x] Comparative dream analysis dashboard.

- Phase 3 – Novel features
  - [x] Pharmacological modulator.
  - [x] Cross-night continuity tracker.
  - [x] Counterfactual dream engine.

---

## Governance and commercialization docs

- `docs/OSS_ROADMAP.md`
- `docs/RFC_PROCESS.md`
- `docs/EDITIONS_AND_PRICING.md`

## Contributing

DreamForge is at an early research-prototype stage and we welcome contributions from:

- Computational neuroscientists (sleep, dreaming, neuromodulators).
- ML researchers and LLM practitioners.
- Data visualization and UI engineers.

Contribution assets (in progress):
- `CONTRIBUTING.md` with setup instructions and good-first-issue templates.
- GitHub issues tagged `good first issue` and `help wanted`.

If you are interested in collaborating more deeply (e.g., co-authoring an arXiv preprint), please open a discussion or issue.

---

## Citation

If DreamForge is useful in your research, you can cite it as:

```text
@misc{dreamforge_ai,
  title  = {DreamForge AI: A Multi-Agent Framework for Computational Dream Simulation},
  author = {To, J. and contributors},
  year   = {2026},
  url    = {https://github.com/JToSound/dreamforge-ai}
}
```

---

## License

DreamForge AI is released under the MIT License.
