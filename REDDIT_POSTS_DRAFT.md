# Reddit Post Drafts for DreamForge AI

## 1. r/MachineLearning — project showcase

**Title**

[Project] DreamForge – multi-agent AI that simulates the neuroscience of dreaming

**Body**

Hi all,

I’ve been experimenting with a different kind of LLM project: instead of just chaining tools, I tried to build an AI that actually *sleeps* and *dreams*.

DreamForge is an open-source framework that combines computational neuroscience and multi-agent AI:

- A two-process sleep model (homeostatic Process S + circadian Process C) that produces realistic overnight hypnograms.
- Stage-dependent neurochemistry for ACh / 5-HT / NE / cortisol.
- A memory graph where fragments are tagged with emotion, salience, and recency, plus hippocampal-style replay events.
- A dream constructor that turns brain state + replayed memories into narrative segments (with optional LLM narratives).
- Metacognitive and phenomenological layers that estimate lucidity and export a first-person dream log.

There’s a real-time dashboard that shows hypnogram, neuromodulator flux, and the memory graph. I’m working on adding a dream timeline, agent activity heatmaps, and a comparative analysis view.

You can run it locally with Docker:

```bash
git clone https://github.com/JToSound/dreamforge-ai
cd dreamforge-ai
docker-compose up --build
```

I’d love feedback on one specific question:

> If you wanted to validate dream bizarreness and lucidity metrics against real data, how would you design the dataset and evaluation protocol?

Repo (code + docs + demo GIF): https://github.com/JToSound/dreamforge-ai

---

## 2. r/neuroscience — modeling discussion

**Title**

[Project] DreamForge – an open-source playground for modeling sleep, neurochemistry, and dreaming

**Body**

I’ve built a small open-source project called DreamForge that tries to turn classic sleep/dreaming ideas into a concrete simulation:

- Borbély-style two-process model (Process S/C) as the backbone for sleep staging.
- Stage-dependent dynamics for acetylcholine, serotonin, noradrenaline, and cortisol.
- A memory graph with emotional tagging and recency, plus hippocampal-like replay.
- A multi-agent stack that constructs dream segments and estimates lucidity.

The goal is *not* to claim biological accuracy, but to create a transparent, hackable scaffold that:
- Makes it easy to play with different sleep parameters and neuromodulator curves.
- Visualizes hypnograms, neurochemical trajectories, replay events, and dream content in real time.
- Can be used as a teaching tool for students who know LLMs but not sleep physiology.

I’ve tried to cite the relevant literature and keep assumptions explicit in the repo’s RESEARCH.md.

If you’re a sleep/dream researcher or student, I’d really appreciate critiques of:
- The modeling choices for Process S/C and neuromodulators.
- Whether the memory/replay abstraction makes sense.
- Ideas for small experiments that could make this more than just a toy.

Repo: https://github.com/JToSound/dreamforge-ai

---

## 3. r/selfhosted — for home lab enthusiasts

**Title**

[Project] DreamForge – self-hostable AI that simulates dreaming (FastAPI + Streamlit + Docker)

**Body**

If you like running quirky services on your homelab, this might be fun.

DreamForge is a self-hostable AI experiment that simulates an agent that "sleeps" and "dreams" overnight:

- FastAPI backend for running simulations.
- Streamlit dashboard for visualizing hypnogram, neurochemistry, memory graph, and dream timeline.
- Docker Compose setup with API + dashboard + Redis.

Setup is straightforward:

```bash
git clone https://github.com/JToSound/dreamforge-ai
cd dreamforge-ai
docker-compose up --build
```

Then:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

It’s early-stage and still evolving, but if you enjoy self-hosting interesting AI toys, I’d love to hear what kind of integrations or metrics you’d like to see.

Repo: https://github.com/JToSound/dreamforge-ai

---

## 4. r/LocalLLaMA — local model integration

**Title**

[Project] DreamForge – plug local LLMs into a multi-agent dream simulator

**Body**

I’m working on DreamForge, an open-source framework that simulates an AI "dreaming" on top of a neuroscience-inspired sleep model.

Right now the dream constructor and phenomenology layers are abstracted so they can call out to any LLM backend (OpenAI, Anthropic, or local models via things like Ollama / text-generation-webui / vLLM).

If you’re into local models, I’d love feedback on:
- How you’d structure the LLM backend interface.
- Which local models you’d try first for dream content generation.
- What kind of prompts would best capture vivid, bizarre dream imagery.

Repo (with placeholders for LLM integration hooks): https://github.com/JToSound/dreamforge-ai
