# Changelog

## Unreleased

- Benchmark, governance, and compatibility hardening:
  - added `scripts/benchmark_pack.py` for fixed-profile, seed-controlled benchmark output with optional baseline delta calculation,
  - added `scripts/generate_baseline_report.py` to generate both JSON and markdown benchmark comparison artifacts,
  - added artifact manifest and health checks (`artifacts/manifest.json`, `/api/artifacts/health`, startup validation log),
  - added async job provenance contract fields (`schema_version`, `progress_source`, `eta_source`, `provenance`) in job status payloads,
  - improved async ETA/progress accuracy with LLM-aware projection (expected LLM invocation count + observed/historical per-invocation latency + concurrency),
  - added durable state-event log fallback (`outputs/state-events.jsonl`) for simulation/job/workspace/audit persistence when Redis is unavailable,
  - added async queue governance limits (`ASYNC_MAX_PENDING_JOBS`, `ASYNC_MAX_RUNNING_JOBS`) and 429 backpressure response when pending queue is full,
  - added PSG connector channel QA endpoint (`/api/psg/connectors/channel-qa`) and plugin evaluator surfaces (`/api/plugins/evaluators`, `/api/plugins/evaluators/run`).
- Collaboration and contribution governance:
  - added GitHub issue templates (`bug-report`, `runtime-regression`, `feature-request`) and template config,
  - added compatibility and deprecation policy docs (`docs/COMPATIBILITY_POLICY.md`, `docs/DEPRECATION_POLICY.md`),
  - updated README with artifact health endpoint, benchmark pack usage, and policy document links.

- Frontend contract convergence and build fixes:
  - fixed Vite plugin mismatch (`@vitejs/plugin-react-swc`) and corrected frontend app entry import wiring,
  - rewired `web-frontend/src/useSimulationData.ts` + `App.tsx` to active API contract and async job submit/poll/cancel flow.
- API surface and operations hardening:
  - added compatibility adapter in `api/routes/simulation.py` to map legacy request shape onto active `SimulationConfig`,
  - release-gate latency check now uses in-process p95 (`simulation_latency_p95_pass`) with new p95/p99 runtime and Prometheus metrics,
  - metrics auth exemption is now configurable via `METRICS_PUBLIC`.
- Data governance and collaboration surfaces:
  - added outputs retention + metadata durability (`outputs/index.json`, `OUTPUT_RETENTION_DAYS`, `OUTPUT_RETENTION_MAX_RUNS`),
  - added collaboration workspace APIs (`/api/workspaces*`) with run attachment endpoints.
- CI/docs updates:
  - CI now includes frontend Node setup and `web-frontend` production build gate,
  - added top-level `CONTRIBUTING.md` and updated README API/configuration/CI documentation.
- Regression coverage expanded for p95 metrics, metrics-access policy, workspace APIs, outputs retention/index behavior, CI frontend gate, and frontend/API source guard expectations.
- Added productization baseline APIs in `api/main.py`:
  - API contract metadata endpoint (`/api/version`, `/api/v1/version`)
  - SLO and error taxonomy endpoints (`/api/slo`, `/api/error-taxonomy`)
  - Prometheus text metrics endpoint (`/metrics/prometheus`)
  - async simulation queue endpoints (`/api/simulation/night/async`, `/api/simulation/jobs/{job_id}`)
  - compare and report endpoints (`/api/simulation/compare`, `/api/simulation/{id}/report`)
  - v1 alias routes for major LLM/simulation endpoints
- Added request telemetry/security middleware baseline:
  - optional API key auth (`API_ACCESS_TOKEN`),
  - in-process per-client rate limiting (`API_RATE_LIMIT_PER_MINUTE`),
  - request ID response header (`X-Request-ID`),
  - structured audit event logging for key actions.
- Added prompt/model capability registry module `core/llm_registry.py` and API exposure via `/api/llm/registry`.
- Added chart design system + export spec in `visualization/charts/static_visualizations.py`, including per-chart provenance annotations.
- Added dashboard i18n baseline and compare/report center in `visualization/dashboard/app.py`:
  - locale selector (`en`, `zh-HK`) with key-based labels,
  - session-history comparison table and downloadable JSON report.
- Added governance/product docs:
  - `docs/OSS_ROADMAP.md`
  - `docs/RFC_PROCESS.md`
  - `docs/EDITIONS_AND_PRICING.md`
- Added/updated regression tests:
  - `tests/test_api_main_endpoints.py`
  - `tests/test_dreamscript_and_charts.py`
- Dashboard/API reliability and UX hardening:
  - widened `sleep_start_hour` validation to `0.0–26.0` (supports naps and early-morning sleep starts),
  - aligned dashboard simulation payload to API contract (`dt_minutes`, `ssri_strength`, `melatonin`, `cannabis`, `emotional_state`, `use_llm`),
  - switched dashboard LLM status + test flow to API-backed config/health checks (no local demo-only detection),
  - updated runtime and compose default simulation request timeout to `3600` seconds.
- Added regression coverage for mixed parameter/pharmacology/event combinations and nap/early-start scenarios.
- Reliability/productization hardening:
  - Redis-backed persistence fallback for simulations/jobs/audit events in `api/main.py`,
  - release gate endpoint (`/api/release-gate`) with SLO check summary,
  - audit query endpoint (`/api/audit/events`) and enterprise metadata endpoint (`/api/enterprise`),
  - token role/scope policy support via `API_TOKEN_ROLE_MAP`.
- Narrative quality controls:
  - style presets (`scientific/cinematic/minimal/therapeutic`) and prompt profile A/B controls wired into narrative generation.
- Dashboard compare/report upgraded to call API compare endpoint, with confidence/anomaly/event marker display.
- Dashboard i18n expanded with additional key coverage and `zh-CN` locale baseline.
- Added operational/product docs:
  - `docs/INCIDENT_RUNBOOKS.md`
  - `docs/ENTERPRISE.md`
- Updated defaults and positioning:
  - `docker-compose.yml` API `LLM_MAX_TOKENS` default aligned to `2048`,
  - README wording and docs references aligned to productization direction.
- Dashboard presentation and export hardening:
  - static analytics export now provides graceful runtime fallback with interactive HTML export when image renderer is unavailable,
  - REM narrative viewer sanitizes pasted-content/code artifacts and uses product-style expandable cards,
  - hypnogram/neuro/lucidity/memory/heatmap visualizations upgraded with clearer annotations, filters, and compare delta charting.
- Simulation stop-control hardening:
  - added async job cancellation API (`/api/simulation/jobs/{job_id}/cancel`, plus `/api/v1` alias),
  - dashboard run flow now uses async submit/poll so `Stop` can terminate in-flight simulation jobs (including active LLM generation),
  - added regression coverage for async cancel lifecycle and dashboard stop wiring.
- Dashboard control + export reliability fixes:
  - fixed stop-state UX so one cancel action can return the UI to runnable state without a second stop click,
  - static chart PNG/SVG export now uses layered fallback (local Plotly export + API export endpoint) to improve export success rate,
  - export fallback now uses JSON-safe figure serialization and includes runtime kaleido bootstrap retry for environments missing local exporter binaries,
  - added chart export API (`/api/charts/export`, plus `/api/v1` alias) and compare-delta validation coverage.
- Dependency hardening:
  - standardized on `kaleido>=1.2.0` and added Chromium to Docker images (`BROWSER_PATH=/usr/bin/chromium`) for stable containerized static exports.
- Async simulation UX:
  - job status payload now includes `progress_percent`, `eta_seconds`, and `estimated_duration_seconds`,
  - dashboard displays a live progress bar with percentage and ETA (`mm:ss`) during `Simulation running...`.
- P0 reliability + progress accuracy hardening:
  - upgraded async job telemetry to phase-aware progress (`phase`, `progress_percent`, `eta_seconds`, `eta_margin_seconds`) with event-driven updates from simulation pipeline stages,
  - dashboard progress UI now renders phase labels and ETA ranges, and uses a `Finalizing report...` state before result fetch completion,
  - narrative generation now includes timeout circuit-breaker behavior and progress callbacks for segment-level completion tracking,
  - LLM client now supports layered network timeouts (`connect/read/write/pool`) and provider-aware timeout retry policy to reduce repeated local-provider timeout stalls,
  - narrative fallback/normalization improved with de-dup + grammar sanitization,
  - narrative quality scoring improved with repetition penalty and token-overlap memory grounding metrics (`matched/unmatched/confidence` signals),
  - memory graph realism improved with edge sparsity capping and anti-saturation activation boosts to avoid near-complete-graph and always-maxed activation artifacts,
  - added regression coverage for the new progress, timeout, and narrative-quality behaviors.
- Compare/report and release-gate productization extension:
  - comparison payload now includes `narrative_memory_grounding_mean` and `llm_fallback_rate` deltas, fallback-rate event markers, explicit methodology formulas, and extra anomaly flags (`llm_fallback_spike`, `memory_grounding_drop`),
  - simulation report methodology now includes metric definitions and release-target thresholds for quality/fallback governance,
  - release-gate now evaluates recent simulation quality window checks (`narrative_quality_pass`, `llm_fallback_sla_pass`, `memory_grounding_pass`),
  - dashboard compare center now renders extended anomaly explanations and collapsible comparison methodology details.
- Product report bundle delivery:
  - added report bundle endpoints (`/api/simulation/{id}/report/bundle`, `/api/v1/simulation/{id}/report/bundle`) returning a ZIP package with `report.json`, `summary.json`, `segments_overview.csv`, and `methodology.txt`,
  - dashboard download center now includes one-click “Download product report bundle (ZIP)” using the new API surface,
  - added regression coverage for report bundle API and dashboard wiring.
- Final UX/docs alignment:
  - localized compare methodology + report bundle labels in dashboard i18n (`en`, `zh-HK`, `zh-CN`),
  - updated README API/dashboard sections to document report bundle endpoint and quality-aware release-gate behavior.
- Next-phase roadmap delivery:
  - added `POST /api/simulation/multi-night` and `/api/v1/simulation/multi-night` with continuity-aware carryover mode, recurring-memory summary, and Sankey-ready continuity links,
  - multi-night responses now include per-night metadata (`multi_night_series_id`, `night_index`, `carryover_event_count`) for downstream analytics,
  - CI workflow hardened by removing permissive `|| true` bypass paths and adding strict mypy checks for critical runtime modules.
- Phase 3 roadmap completion:
  - strict typing gate expanded to the full core runtime (`mypy --strict core/`) after resolving all prior strict-typing errors,
  - dashboard now includes a Multi-night Continuity Center with configurable multi-night runs, per-night continuity table, and Sankey flow visualization,
  - added production observability provisioning bundle (`observability/prometheus/*`, `observability/grafana/*`) and compose profile services for Prometheus + Grafana,
  - added CI non-functional release gates via `tests/performance/test_release_gates.py` (load + soak checks) and wired this gate explicitly into workflow execution.

## [Round 6]

- Added `core/generation/narrative_generator.py` with async batch generation, REM/high-biz gating, continuity context, dedicated scene generation, and timeout/error fallback logging.
- Added calibrated lucidity module `core/simulation/lucidity_model.py` and integrated it into API physics generation with REM-depth signals.
- Added lucid-event detection (3+ consecutive threshold ticks), segment-level `is_lucid`, and top-level `lucid_events` in simulation payloads.
- Reordered API pipeline so memory activations are computed before narrative generation, and active memory labels are injected into prompts.
- Updated memory integration surfaces:
  - `MemoryGraph.apply_replay_pulse()` now returns active IDs above threshold.
  - Added `MemoryGraph.label(node_id)` and `MemoryGraph.active_node_ids()`.
  - Exported `active_memory_ids` as pipe-delimited values in segment CSVs.
- Updated dashboard (`visualization/dashboard/app.py`) with:
  - REM narrative viewer panel including scene text and lucid highlighting,
  - lucidity timeline chart with lucid-event markers,
  - in-app memory activation heatmap (CSV-first with payload fallback).
- Added `settings.yaml` runtime knobs:
  - `lucidity_threshold`
  - `llm_enabled`
  - `narrative_min_words_rem`
- Added Round 6 tests:
  - `tests/generation/test_narrative_generator.py`
  - `tests/simulation/test_lucidity_model.py`
  - `tests/simulation/test_memory_integration.py`

### Round 3 metadata + memory graph completion
- Added segment-level provenance fields across simulation/export surfaces:
  - `llm_trigger_type`, `llm_latency_ms`, `template_bank`
  - expanded `generation_mode` to include `CACHED` in model/agent paths.
- Added activation snapshot persistence to `MemoryGraph` and exported
  `memory_graph.activation_snapshots` alongside existing replay events.
- Enriched dream-constructor prompts with qualitative neurochemical context blocks
  and memory-node hints for improved narrative conditioning.
- Updated CSV exports (runner + dashboard ZIP) to include the new generation and
  neurochemistry audit columns.
- Added regression coverage for activation snapshots and new CSV columns.

### v5 physiology and dashboard calibrations
- Recalibrated sleep architecture in `core/models/sleep_cycle.py`:
  - `tau_sleep` default set to `4.2`
  - cycle REM durations reduced and N3 early-night weighting preserved for physiological REM/N3 ratios.
- Reworked cortisol dynamics in `core/models/neurochemistry.py` to a delayed nadir + steep morning rise profile (nadir ~02:30, onset ~05:30, peak ~07:30).
- Updated API simulation physiology in `api/main.py`:
  - reduced REM bizarreness saturation by lowering REM stage prior and reweighting ACh/NE/arousal/cycle terms,
  - enforced REM-only lucidity output with softer ACh sigmoid gating centered at `0.60`.
- Hardened dashboard rendering in `visualization/dashboard/app.py`:
  - neurochemistry fallback chain now supports `neurochemistry`, `neurochemistry_series`, `neurochemistry_ticks`, and segment-derived reconstruction,
  - dream narrative tab now prefers `narrative` first (with additional fallbacks) and supports stage/mode filtering.
- Added regression coverage in `tests/test_fixes_v5.py` for:
  - sleep parameter/cycle proportion targets,
  - cortisol shape constraints,
  - bizarreness and lucidity calibration checks,
  - API physics output behavior,
  - dashboard fallback key presence.

### Post-run diagnostic fixes (session `ec094636`)
- Added Fix 5 (`generation_mode` provenance):
  - normalized `GenerationMode` enum values to `LLM | TEMPLATE | LLM_FALLBACK`,
  - added `generation_mode` to `core/models/dream_segment.py`,
  - added `export_segments_csv()` in `core/simulation/runner.py`,
  - added `generation_mode` column to dashboard ZIP `segments.csv`.
- Added Fix 6 (LLM token exhaustion hardening):
  - prepended `/no_think` to user-side LLM prompts,
  - raised default token budget to `LLM_MAX_TOKENS=2048` in runtime config and API schema defaults,
  - added `.env.example` with `LLM_MAX_TOKENS=2048`,
  - added robust `parse_narrative_response()` in `core/simulation/llm_client.py`,
  - added prompt helper module `core/simulation/llm_prompts.py` and simulation config constant module `core/simulation/config.py`.
- Added `core/simulation/runner.py::build_neurochemistry_ticks()` and `export_neurochemistry_csv()` to reliably export per-segment neurochemistry (`time_hours, stage, ach, serotonin, ne, cortisol`) for dashboard CSV consumers.
- Wired `tools/run_and_analyze_sim.py` to emit `neurochemistry.csv` immediately after simulation output is produced.
- Added memory activation snapshot support in `core/models/memory_graph.py` via immutable `MemoryActivationSnapshot` and `capture_memory_snapshot()`.
- Updated `core/simulation/engine.py` to:
  - record REM-linked activation snapshots after replay/decay,
  - expose top-level compatibility keys required by downstream consumers:
    - `segments`
    - `neurochemistry_ticks`
    - `memory_activations`
    - `memory_graph`
- Calibrated sleep-cycle templates in `core/models/sleep_cycle.py` with explicit `N3_DURATION_BY_CYCLE` (30→0 min across cycles) while preserving calibrated `tau_sleep=3.5` and `n3_s_threshold=0.65`.
- Added physiology-focused cortisol function `cortisol_profile()` in `core/models/neurochemistry.py` and aligned `_cortisol_drive()` to an asymmetric nadir-to-morning-rise profile (nadir ~02:30, peak ~07:30 by default).
- Updated API simulation output in `api/main.py` to include invariant payload keys (`neurochemistry_ticks`, `memory_activations`, `memory_graph`) and switched stage generation to `SleepCycleModel` so N3/N1 proportions remain physiological in API runs.
- Updated dashboard export/render fallbacks in `visualization/dashboard/app.py` to consume:
  - `neurochemistry_ticks` or segment-derived neurochemistry when top-level series is absent,
  - `memory_activations` as an alias for `memory_activation_series`.
- Added regression suite `tests/test_dreamforge_fixes.py` covering:
  - neurochemistry CSV export row/column invariants,
  - memory key presence and non-empty activation snapshots,
  - N3 (13–23%) and N1 (<=12%) stage proportion constraints,
  - cortisol nadir/peak/monotonic rise checks.

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
