# Changelog

## Unreleased

### Quality gates
- Ran full backend test suite (`python -m pytest tests/ -v --tb=short`) and resolved lint blockers.
- Resolved Ruff violations across the repository and applied Black formatting so `ruff check .` and `black . --check` pass.

### Task 1.1 — NeurochemistryModel staged ODE integration
- Implemented/retained staged integration via `integrate_staged()` with per-stage `solve_ivp` calls and `max_step=1/120` hours.
- Updated cortisol drive to an asymmetric sigmoid profile with a peak at hour `5.5` by default (`cortisol_rise_time`).

### Task 1.2 — SleepCycleModel FSM
- Added `CycleStateMachine` and shared `CYCLE_TEMPLATES`/`DEFAULT_CYCLE_TEMPLATE` constants for template-driven ultradian scheduling.
- Wired `SleepCycleModel` to use the FSM schedule builder.
- Kept N3 calibration updates in model parameters:
  - `tau_sleep`: `3.5`
  - `n3_s_threshold`: `0.65`
- Validated N3 proportion target (`0.12 <= N3_fraction <= 0.25`) with `tools/validate_n3.py`.

### Task 1.3 — Bizarreness parametric model
- Maintained the parametric bizarreness scoring model in `core/utils/bizarreness_scorer.py` with stage base weights and neurochemical/arousal/cycle factors.
- Preserved structured component outputs (`discontinuity`, `incongruity`, `implausibility`) and confidence interval reporting.

### Task 1.4 — MemoryGraph activation pulse + decay
- Maintained replay pulse propagation in `MemoryGraph.apply_replay_pulse()` with attenuation and replay event logging.
- Maintained exponential activation decay in `MemoryGraph.decay_activations()` with salience-aware lower bound.

### Task 1.5 — Lucidity multi-factor gated model
- Maintained multi-factor lucidity estimation in `MetacognitiveAgent.compute_lucidity_probability()`:
  - REM stage gating
  - Neurochemical gating (ACh/cortisol)
  - Temporal late-night bias
  - Training + reality-check history effects
  - Weak bizarreness coupling
- Validated acceptance target `|r(bizarreness, lucidity)| < 0.55` with `tools/validate_lucidity.py`.

### Prior foundation updates completed before this pass
- Added trigger-based narrative infrastructure:
  - `core/simulation/llm_trigger.py`
  - `core/simulation/narrative_cache.py`
- Refactored dream construction flow to use triggers, offline/template fallback behavior, and LLM call accounting.
- Expanded API compatibility surfaces in `api/main.py` (aliases/resource endpoints/streaming endpoint and health enrichment).
- Repaired structural blockers:
  - fixed merge-conflicted `api/schemas.py`
  - fixed malformed duplicate workflow content in `.github/workflows/ci.yml`

### Phase 4 — centralized runtime config
- Added `core/config.py` with environment-driven shared runtime defaults.
- Wired shared config into:
  - `core/utils/llm_backend.py`
  - `core/llm_client.py`
  - `visualization/dashboard/app.py`
- Documented the new config surface in `README.md`.

### Phase 7 — demo GIF docs
- Updated `scripts/generate_demo_gif.py` header to match in-repo usage.
- Added README instructions for Playwright-based demo GIF capture.

### Phase 6 — full test suite + coverage
- Added `tests/test_phase1_models.py` covering:
  - neurochemistry staged integration transitions
  - cortisol peak-time behavior
  - sleep-cycle template FSM + N3 fraction constraint
  - memory replay pulse + decay dynamics
  - lucidity/bizarreness correlation bound
- Added API and utility coverage suites:
  - `tests/test_api_main_endpoints.py`
  - `tests/test_api_routes_and_schemas.py`
  - `tests/test_core_config_and_llm_client.py`
  - `tests/test_utils_misc.py`
- Coverage now reaches **81%** for `core` + `api` (`python -m pytest tests/ -q --cov=core --cov=api`), with zero test failures.

### Phase 3 — DreamScript offline engine completion
- Rebuilt `core/simulation/dreamscript.py` with a fully featured `DreamScriptEngine`.
- Added 4 expanded template banks (each 30+ templates):
  - `NREM_LIGHT`
  - `NREM_DEEP`
  - `REM_EARLY`
  - `REM_LATE`
- Added explicit modulation hooks:
  - `ACh > 0.7` → bizarre vocabulary injection
  - `NE < 0.1` → reality-failure phrasing
  - high cortisol (`> 0.75`) → anxiety phrasing
- Strengthened continuity by extracting an entity from prior text and carrying it forward into subsequent narrative output.
- Added `DEMO_MODE` gating in `core/utils/llm_backend.py` so offline DreamScript is auto-selected when no live provider is configured.

### Phase 5 — dashboard + static visualization completion
- Added static visualization module: `visualization/charts/static_visualizations.py` with:
  - `plot_rem_episode_trend()`
  - `plot_affect_ratio_timeline()`
  - `plot_bizarreness_cortisol_scatter()`
  - `plot_per_cycle_architecture()`
- Added Okabe-Ito palette usage in dashboard charts and live traces.
- Added chart export plumbing (modebar image export + optional SVG/PNG download buttons via Plotly image export in dashboard).
- Integrated the four static charts into `visualization/dashboard/app.py` under a dedicated analytics section.

### Additional validation expansion
- Added `tests/test_dreamscript_and_charts.py` for DreamScript bank size/modulation/continuity and runner throughput.
- Added `tests/test_llm_backend_adapters.py` for LLM backend/provider detection, retry/stream paths, offline parsing, and adapter callables.
- Ran requested validation command with visualization coverage scope:
  - `python -m pytest tests/ -q --cov=core --cov=api --cov=visualization --cov-report=term`
  - Result: **82% total coverage**, **36 passed**, **0 failed** (1 skipped due missing optional Plotly dependency in test env).
