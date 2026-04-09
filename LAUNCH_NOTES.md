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

## Cross-night continuity feature

DreamForge includes a dedicated cross-night continuity module and dashboard:

- Run multi-night simulations with shared parameters (duration, pharmacology).
- Track which memory fragments recur across nights.
- Visualize recurring fragments as a Sankey/alluvial diagram (Night 1 → Night 2 → Night 3, ...).
- Inspect a table of recurring fragments with per-night occurrence counts.

This is useful for exploring longer-term "themes" in the simulated memory graph and for designing more formal experiments around recurring content.

---

## Show HN draft

**Title**

Show HN: DreamForge – AI agents that simulate the neuroscience of dreaming

**Body (outline)**

- 1–2 short paragraphs on motivation (computational neuroscience + multi-agent AI).
- Bullet list of what is actually implemented (sleep model, neurochemistry, memory graph, agents, dashboards including cross-night continuity).
- Screenshots or link to demo GIF.
- Very short "how to run" (Docker Compose) with link to GitHub.
- 2–3 concrete questions asking for feedback (metrics, evaluation, use cases).

Tone: technical, honest, and curious. Avoid hype; focus on what the system really does.

---

## Reddit post templates

### r/MachineLearning

**Title**

[Project] DreamForge – multi-agent AI that simulates the neuroscience of dreaming

**Body (outline)**

- Start with the problem: LLMs are excellent sequence models, but they usually ignore sleep, dreaming, and memory consolidation.
- Introduce DreamForge as an open-source experiment that:
  - Implements a two-process sleep model and stage-dependent neurochemistry.
  - Builds a memory graph and hippocampal-style replay.
  - Uses agents to construct dream narratives and lucidity estimates.
  - Provides dashboards for single-night dynamics and cross-night continuity.
- Include one GIF showcasing hypnogram + neurochemical flux + memory graph.
- Short "how to run" (Docker Compose) with link to GitHub.
- Ask a concrete technical question, e.g.: "If you were to validate dream bizarreness metrics, how would you design the dataset and evaluation protocol?"

Tone: technical, curiosity-driven, no marketing language.

### r/neuroscience

Focus more on the biological inspiration:
- Emphasize Borbély's two-process model, neuromodulator dynamics, and hippocampal replay.
- Be explicit about simplifications and limitations.
- Mention the cross-night continuity dashboard as a way to explore recurring simulated themes.
- Invite neuroscientists to critique the modeling assumptions and suggest better parameterizations.

### r/selfhosted

Focus on:
- Local-first design, Docker Compose setup, and optional local LLMs via Ollama or other backends.
- How to run the dashboard and API on a home server.
- The fact that you can run both single-night and cross-night dashboards locally.

### r/LocalLLaMA

Focus on:
- How DreamForge can plug into local models.
- Where in the DreamConstructorAgent and PhenomenologyReporter the LLM calls live.

For each subreddit:
- Lead with the problem.
- Explain the architecture at a high level.
- Embed GIF + GitHub link.
- End with one specific, open question to invite serious discussion.
