# Launch Assets for DreamForge AI

This file collects launch-related copy and templates.

---

## README highlights

- One-sentence pitch at the very top.
- Demo GIF within first scroll.
- Quick start in three commands (Docker Compose).
- Feature gallery with screenshots / GIFs.
- Scientific grounding pointing to RESEARCH.md.
- Roadmap inviting contribution.
- Star CTA at top and bottom.

---

## Show HN draft

**Title**

Show HN: DreamForge – AI agents that simulate the neuroscience of dreaming

**Body (outline)**

Hi HN,

I built DreamForge, an open-source project that tries to answer a simple question: what would it look like if an AI actually *slept* and *dreamed*?

Most LLM demos focus on prompt engineering and tools. DreamForge takes a different angle: it builds a neuroscience-inspired sleep and dreaming stack, then lets a small society of agents run wild on top of it.

Under the hood:
- A two-process sleep model (homeostatic Process S + circadian Process C) that produces realistic hypnograms.
- Stage-dependent neurochemistry for acetylcholine, serotonin, noradrenaline, and cortisol.
- A memory graph where fragments are tagged with emotion, salience, and recency, plus hippocampal-style replay events.
- A dream constructor that turns brain state + memory replay into narrative segments.
- Metacognitive and phenomenological layers that estimate lucidity and export a first-person dream log.

The dashboard shows all of this in real time: hypnogram, neuromodulators, memory graph, and (soon) dream timelines and bizarreness metrics.

Why I built it:
- As an experiment in combining computational neuroscience with agentic AI.
- As a teaching tool for students who know about LLMs but not about sleep architecture.
- As a playground for people who care about lucid dreaming, bizarreness metrics, and "world models" inside neural networks.

You can try it locally with Docker:

- `git clone https://github.com/JToSound/dreamforge-ai`
- `cd dreamforge-ai`
- `docker-compose up --build`

Then open `http://localhost:8501` for the dashboard and `http://localhost:8000/docs` for the API.

I would love feedback on:
- Better ways to validate dream bizarreness and lucidity metrics against real dream reports.
- How to make the agent interactions more interpretable.
- Whether this could be useful as a research or teaching tool.

Repo: https://github.com/JToSound/dreamforge-ai

---

## Reddit post templates

### r/MachineLearning

**Title**

[Project] DreamForge – multi-agent AI that simulates the neuroscience of dreaming

**Body (outline)**

- Start with the problem: LLMs are great at text, but they do not model sleep, dreaming, or memory consolidation.
- Introduce DreamForge as an open-source experiment that:
  - Implements a two-process sleep model and stage-dependent neurochemistry.
  - Builds a memory graph and hippocampal-style replay.
  - Uses agents to construct dream narratives and lucidity estimates.
- Include one GIF showcasing hypnogram + neurochemical flux + memory graph.
- Short "how to run" (Docker Compose) with link to GitHub.
- Ask a concrete technical question, e.g.: "If you were to validate dream bizarreness metrics, how would you design the dataset and evaluation protocol?"

Tone: technical, curiosity-driven, no marketing language.

### r/neuroscience

Focus more on the biological inspiration:
- Emphasize Borbély's two-process model, neuromodulator dynamics, and hippocampal replay.
- Be explicit about simplifications and limitations.
- Invite neuroscientists to critique the modeling assumptions and suggest better parameterizations.

### r/selfhosted

Focus on:
- Local-first design, Docker Compose setup, and optional local LLMs via Ollama or other backends.
- How to run the dashboard and API on a home server.

### r/LocalLLaMA

Focus on:
- How DreamForge can plug into local models.
- Where in the DreamConstructorAgent and PhenomenologyReporter the LLM calls would live.

For each subreddit:
- Lead with the problem.
- Explain the architecture at a high level.
- Embed GIF + GitHub link.
- End with one specific, open question to invite serious discussion.
