# DreamForge AI

> "The first open-source AI system that thinks while it sleeps."

DreamForge AI is a hierarchical multi-agent framework that simulates human-like dreaming from biophysical sleep dynamics up through narrative experience and phenomenological reporting.

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

- Test hypotheses about sleep homeostasis, neuromodulators, and dream content.
- Provide a concrete playground for dream bizarreness metrics and lucidity modeling.
- Offer an open-source reference implementation for multi-agent scientific simulators.

This project is **not** intended as a clinical tool or a replacement for empirical sleep research; instead, it is a conceptual and educational framework built with scientific humility and explicit limitations.

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
  - Composes narrative dream segments from sleep stage, neurochemistry, and active memories.
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
  - Support for pharmacological perturbations (SSRIs, melatonin, stress hormones).

- **Memory graph and replay**
  - NetworkX-based graph of memory fragments with emotional tags and recency.
  - Hippocampal sharp-wave ripple–like replay as biased random walks.

- **Dream construction**
  - Dream segments conditioned on stage, neuromodulators, and replayed memories.
  - Hooks for LLM backends (GPT-4o, Claude, local models via Ollama, etc.).

- **Metacognition and phenomenology**
  - Lucidity probability estimates and dream-logic enforcement.
  - First-person narrative output and symbolic interpretation layer.

- **Visualization dashboard**
  - Hypnogram, neurochemical flux, memory association graph.
  - Designed for extension to dream timelines, agent activity heatmaps, and comparative runs.

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
  - `POST /simulate-night` – run a single-night simulation with configurable parameters.
  - `GET /dream/{id}` – retrieve a structured dream report.
  - `GET /health` – simple health check.

- **Streamlit dashboard**
  - Run `streamlit run visualization/dashboard/app.py` in development, or use the `dashboard` service in Docker Compose.
  - Visualizes hypnogram, neuromodulators, and the memory graph in real time.

---

## Roadmap

- Phase 1 – Core simulation engine
  - [x] SleepCycleModel and NeurochemistryModel.
  - [x] MemoryGraph with replay and forgetting.
  - [x] SleepCycleAgent, NeurochemistryAgent, MemoryConsolidationAgent.
  - [ ] DreamConstructorAgent with LLM integration hooks.
  - [ ] MetacognitiveAgent and PhenomenologyReporter fully wired into the API.

- Phase 2 – Visualization
  - [x] Hypnogram, neurochemical flux, memory association graph.
  - [ ] Dream content timeline and agent activity heatmap.
  - [ ] Comparative dream analysis dashboard.

- Phase 3 – Novel features
  - [ ] Pharmacological modulator.
  - [ ] Cross-night continuity tracker.
  - [ ] Counterfactual dream engine.

---

## Contributing

DreamForge is at an early research-prototype stage and we welcome contributions from:

- Computational neuroscientists (sleep, dreaming, neuromodulators).
- ML researchers and LLM practitioners.
- Data visualization and UI engineers.

Planned contribution assets:
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
