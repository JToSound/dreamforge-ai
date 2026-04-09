# Show HN: DreamForge – AI agents that simulate the neuroscience of dreaming

Hi HN,

I built DreamForge, an open-source project that asks a simple question: what would it look like if an AI actually *slept* and *dreamed*?

Instead of treating LLMs as stateless text machines, DreamForge wraps them in a neuroscience-inspired sleep and dreaming stack and lets a small society of agents run on top of it.

Under the hood, DreamForge includes:

- A two-process sleep model (homeostatic Process S + circadian Process C) that produces realistic hypnograms over an 8-hour night.
- Stage-dependent neurochemistry for acetylcholine, serotonin, noradrenaline, and cortisol.
- A memory graph where fragments are tagged with emotion, salience, and recency, plus hippocampal-style replay events during NREM and REM.
- A dream constructor that turns brain state + replayed memories into narrative segments.
- Metacognitive and phenomenological layers that estimate lucidity and export a first-person dream log.

There is a real-time dashboard that shows:
- Hypnogram (sleep stages over time).
- Neurochemical flux (ACh / 5-HT / NE / cortisol).
- Memory association graph with replay pulses.
- (Coming soon) dream content timeline and agent activity heatmaps.

Why I built this:

- As a playground for computational neuroscience + agentic AI.
- As a teaching tool for students who know transformers but not sleep architecture.
- As a way to think about "world models" and internal simulations in a more embodied way.

You can run it locally with Docker:

```bash
git clone https://github.com/JToSound/dreamforge-ai
cd dreamforge-ai
docker-compose up --build
```

Then open `http://localhost:8501` for the dashboard and `http://localhost:8000/docs` for the API.

I'm particularly interested in feedback on:
- How to better ground the model in real sleep/dream datasets.
- How to design quantitative dream bizarreness and lucidity evaluations.
- Whether this might be useful as a research or teaching platform.

Repo: https://github.com/JToSound/dreamforge-ai
