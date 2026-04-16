# DreamForge Incident Runbooks

## Purpose
Operational playbooks for the highest-frequency production incidents.

## 1. LLM timeout burst
**Detection**
- `/api/release-gate` fails latency or completion checks.
- Rising `timeout` / `provider_error` in `/api/error-taxonomy`.

**Immediate actions**
1. Confirm provider health with `GET /api/health/llm`.
2. Raise provider timeout in `POST /api/llm/config` if backend is healthy but slow.
3. Reduce concurrency/load and retry a representative simulation.

**Recovery validation**
- `/api/release-gate` returns `pass=true`.
- New simulations complete without timeout-driven fallback spikes.

## 2. Provider 400 payload rejection
**Detection**
- API logs contain repeated `LLM payload rejected (400)` warnings.
- Segment fallback reason trends to `provider_error`.

**Immediate actions**
1. Verify target model compatibility from `GET /api/llm/registry`.
2. Update provider/model via `POST /api/llm/config`.
3. Re-run one simulation and inspect summary fallback counters.

**Recovery validation**
- Reduced `llm_fallback_segments` in simulation summaries.
- `/api/health/llm` remains healthy and model list stable.

## 3. Export failure
**Detection**
- `export_failures_total` increases in `/metrics` or `/metrics/prometheus`.
- Missing expected CSV artifacts in ZIP export.

**Immediate actions**
1. Reproduce export on latest successful simulation ID.
2. Check disk/path permissions and output directory availability.
3. Confirm required payload keys are present (`segments`, `memory_activations`, `memory_graph`).

**Recovery validation**
- Export succeeds for a new run.
- `export_success_rate` in `/api/release-gate` is back within target.

## 4. Async queue backlog
**Detection**
- `job_queue_pending` grows continuously in `/metrics`.
- `/api/release-gate` fails completion checks.

**Immediate actions**
1. Check worker/API health and recent exceptions.
2. Inspect `/api/simulation/jobs/{job_id}` for stuck job patterns.
3. Restart service only after preserving job diagnostics and audit trail.

**Recovery validation**
- Pending queue drains.
- Newly submitted jobs transition `pending -> running -> completed`.
