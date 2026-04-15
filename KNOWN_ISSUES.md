# Known Issues After Round 4

## 1. Strict typing debt remains in legacy modules
- **Current state:** `mypy --strict core/` reports 60 errors (Round 4 threshold met, debt still present).
- **Impact:** Gate is now passing, but strict-mode maintainability is still fragile in older modules.
- **Main buckets:** missing third-party stubs (`networkx`, `pandas`, `scipy`), generic annotations (`dict`/`list` type params), legacy untyped helpers in LLM adapters and orchestration modules.
- **Estimated effort:** Medium (1–2 focused sessions).
- **Recommended Round 5 approach:**
  1. Add/lock required stubs and mypy config suppressions only for external libs.
  2. Fix high-volume `dict`/`list` type-arg errors in `sleep_cycle`, `continuity_tracker`, `orchestrator`, `llm_trigger`.
  3. Close remaining untyped function signatures in `llm_adapters` and `core/llm_client`.

## 2. Live LM Studio token telemetry is not asserted in CI/tests
- **Current state:** `/no_think` and JSON-mode are enforced in request construction, but LM Studio log fields (`finish_reason`, `reasoning_tokens`) are not verifiable in automated tests.
- **Impact:** Runtime provider behavior can drift without immediate CI detection.
- **Estimated effort:** Small/Medium.
- **Recommended Round 5 approach:**
  1. Add an integration harness with a mock OpenAI-compatible server that returns explicit usage fields.
  2. Add assertions on response metadata and structured-output success rates over batch runs.

## 3. Template library is improved but not yet externalized to YAML banks
- **Current state:** Template quality floor and de-duplication improved, but full stage YAML bank architecture is not yet implemented.
- **Impact:** Authoring and large-scale narrative tuning remain code-coupled.
- **Estimated effort:** Medium.
- **Recommended Round 5 approach:**
  1. Introduce `core/data/templates/*.yaml` loader + schema validator.
  2. Migrate current inline templates into stage-specific banks with neurochem filter metadata.
  3. Add a deterministic selection strategy test matrix.
