import asyncio
import io
import os
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


class DummyConfig:
    def __init__(
        self,
        provider: str = "stub",
        base_url: str = "http://stub.local",
        model: str = "stub-model",
        api_key: str = "stub-key",
        max_tokens: int = 128,
        temperature: float = 0.2,
        timeout: int = 5,
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout


class DummyLLMClient:
    def __init__(self, config=None):
        self.config = config or DummyConfig()

    async def chat(self, system: str, user: str) -> str:
        return '{"narrative":"narrative from llm","scene":"scene from llm"}'

    async def check_health(self) -> dict:
        return {"ok": True, "models": [self.config.model]}

    async def aclose(self):
        return None


@pytest.fixture
def patched_api(monkeypatch, tmp_path):
    monkeypatch.setenv("DREAMFORGE_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(
        api_main, "_STATE_EVENT_LOG_FILE", str(tmp_path / "state-events.jsonl")
    )
    api_main._simulations.clear()
    with api_main._workspaces_lock:
        api_main._workspaces.clear()
    with api_main._jobs_lock:
        api_main._simulation_jobs.clear()
        for task in api_main._simulation_job_tasks.values():
            if not task.done():
                task.cancel()
        api_main._simulation_job_tasks.clear()
    with api_main._rate_limit_lock:
        api_main._request_windows.clear()
    with api_main._metrics_lock:
        for key in list(api_main._runtime_metrics.keys()):
            api_main._runtime_metrics[key] = 0.0
        api_main._api_error_codes.clear()
        api_main._simulation_duration_seconds_history.clear()
        api_main._simulation_duration_seconds_llm_history.clear()
        api_main._simulation_duration_seconds_no_llm_history.clear()
    with api_main._artifact_health_lock:
        api_main._artifact_health_cache.clear()
        api_main._artifact_health_cached_at = 0.0
    fake = DummyLLMClient()
    monkeypatch.setattr(api_main, "get_llm_client", lambda: fake)
    monkeypatch.setattr(api_main, "LLMClient", DummyLLMClient)
    return api_main


def _sim_payload(**overrides):
    payload = {
        "duration_hours": 1.0,
        "dt_minutes": 0.5,
        "ssri_strength": 1.0,
        "stress_level": 0.2,
        "sleep_start_hour": 23.0,
        "melatonin": False,
        "cannabis": False,
        "prior_day_events": ["had a long meeting"],
        "emotional_state": "neutral",
        "style_preset": "scientific",
        "prompt_profile": "A",
        "use_llm": False,
        "llm_segments_only": False,
    }
    payload.update(overrides)
    return payload


def test_system_and_llm_routes(patched_api):
    with TestClient(patched_api.app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["status"] == "running"

        llm_health = client.get("/api/health/llm")
        assert llm_health.status_code == 200
        assert llm_health.json()["ok"] is True

        llm_health_v1 = client.get("/api/v1/health/llm")
        assert llm_health_v1.status_code == 200
        assert llm_health_v1.json()["ok"] is True

        llm_health_alias = client.get("/api/llm/health")
        assert llm_health_alias.status_code == 200
        assert llm_health_alias.json()["ok"] is True

        cfg = client.get("/api/llm/config")
        assert cfg.status_code == 200
        assert cfg.json()["model"] == "stub-model"

        registry = client.get("/api/llm/registry")
        assert registry.status_code == 200
        assert registry.json()["active"]["provider"] == "stub"

        updated = client.post(
            "/api/llm/config",
            json={"provider": "openai", "model": "new-model", "temperature": 0.7},
        )
        assert updated.status_code == 200
        assert updated.json()["provider"] == "openai"
        assert updated.json()["model"] == "new-model"

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["llm_connected"] is True

        version = client.get("/api/version")
        assert version.status_code == 200
        assert version.json()["api_contract"] == "v1"
        assert "artifact_manifest_pass" in version.json()
        assert "async_job_schema_version" in version.json()

        artifact_health = client.get("/api/artifacts/health")
        assert artifact_health.status_code == 200
        assert "manifest_loaded" in artifact_health.json()
        assert "total_artifacts" in artifact_health.json()


def test_chart_export_endpoint_returns_png_and_svg(patched_api):
    figure_payload = {
        "data": [{"type": "scatter", "x": [0, 1, 2], "y": [1, 3, 2]}],
        "layout": {"title": {"text": "Export Smoke"}},
    }
    with TestClient(patched_api.app) as client:
        png_resp = client.post(
            "/api/charts/export",
            json={"figure": figure_payload, "format": "png", "scale": 1.0},
        )
        assert png_resp.status_code == 200
        assert png_resp.headers["content-type"].startswith("image/png")
        assert len(png_resp.content) > 100

        svg_resp = client.post(
            "/api/charts/export",
            json={"figure": figure_payload, "format": "svg", "scale": 1.0},
        )
        assert svg_resp.status_code == 200
        assert svg_resp.headers["content-type"].startswith("image/svg+xml")
        assert b"<svg" in svg_resp.content


def test_build_comparison_payload_uses_candidate_minus_baseline(patched_api):
    baseline = {
        "id": "base-1",
        "summary": {
            "mean_bizarreness": 0.30,
            "rem_fraction": 0.20,
            "lucid_event_count": 2,
            "narrative_quality_mean": 0.50,
            "narrative_memory_grounding_mean": 0.35,
            "llm_fallback_segments": 1,
            "llm_total_invocations": 10,
        },
        "segments": [{"stage": "REM", "start_time_hours": 0.0, "end_time_hours": 0.5}],
    }
    candidate = {
        "id": "cand-1",
        "summary": {
            "mean_bizarreness": 0.45,
            "rem_fraction": 0.30,
            "lucid_event_count": 6,
            "narrative_quality_mean": 0.65,
            "narrative_memory_grounding_mean": 0.18,
            "llm_fallback_segments": 3,
            "llm_total_invocations": 10,
        },
        "segments": [{"stage": "REM", "start_time_hours": 0.0, "end_time_hours": 0.5}],
    }

    payload = patched_api._build_comparison_payload(baseline, candidate)
    assert payload["baseline_id"] == "base-1"
    assert payload["candidate_id"] == "cand-1"
    assert payload["delta"]["mean_bizarreness"] == 0.15
    assert payload["delta"]["rem_fraction"] == 0.1
    assert payload["delta"]["lucid_event_count"] == 4
    assert payload["delta"]["narrative_quality_mean"] == 0.15
    assert payload["delta"]["narrative_memory_grounding_mean"] == -0.17
    assert payload["delta"]["llm_fallback_rate"] == 0.2
    assert "memory_grounding_drop" in payload["anomaly_flags"]
    assert "llm_fallback_spike" in payload["anomaly_flags"]
    assert payload["methodology"]["delta_formula"] == "candidate - baseline"


def test_simulation_crud_and_counterfactual(patched_api):
    with TestClient(patched_api.app) as client:
        created = client.post("/api/simulation/night", json=_sim_payload())
        assert created.status_code == 201
        data = created.json()
        sim_id = data["id"]
        assert len(data["segments"]) > 0

        fetched = client.get(f"/api/simulation/{sim_id}")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == sim_id

        fetched_alias = client.get(f"/simulation/{sim_id}")
        assert fetched_alias.status_code == 200
        assert fetched_alias.json()["id"] == sim_id

        segments = client.get(f"/simulation/{sim_id}/segments?offset=1&limit=2")
        assert segments.status_code == 200
        assert segments.json()["offset"] == 1
        assert segments.json()["limit"] == 2

        neuro = client.get(f"/simulation/{sim_id}/neurochemistry")
        assert neuro.status_code == 200
        assert len(neuro.json()["series"]) > 0

        hyp = client.get(f"/simulation/{sim_id}/hypnogram")
        assert hyp.status_code == 200
        assert len(hyp.json()["hypnogram"]) > 0

        listed = client.get("/api/dreams")
        assert listed.status_code == 200
        assert listed.json()["count"] >= 1

        report = client.get(f"/api/simulation/{sim_id}/report")
        assert report.status_code == 200
        assert report.json()["simulation_id"] == sim_id
        assert "metric_definitions" in report.json()["methodology"]
        assert "release_targets" in report.json()["methodology"]

        bundle = client.get(f"/api/simulation/{sim_id}/report/bundle")
        assert bundle.status_code == 200
        assert bundle.headers["content-type"].startswith("application/zip")
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            names = set(archive.namelist())
            assert "report.json" in names
            assert "summary.json" in names
            assert "segments_overview.csv" in names
            assert "methodology.txt" in names

        bundle_v1 = client.get(f"/api/v1/simulation/{sim_id}/report/bundle")
        assert bundle_v1.status_code == 200
        assert bundle_v1.headers["content-type"].startswith("application/zip")

        counter = client.post(
            "/api/simulation/counterfactual",
            json={
                "base_simulation_id": sim_id,
                "perturbations": {"stress_level": 0.9, "duration_hours": 1.5},
                "use_llm": False,
            },
        )
        assert counter.status_code == 200
        assert counter.json()["config"]["stress_level"] == 0.9

        missing = client.post(
            "/api/simulation/counterfactual",
            json={
                "base_simulation_id": "does-not-exist",
                "perturbations": {"stress_level": 0.1},
                "use_llm": False,
            },
        )
        assert missing.status_code == 404

        compare = client.post(
            "/api/simulation/compare",
            json={
                "baseline_simulation_id": sim_id,
                "candidate_simulation_id": counter.json()["id"],
            },
        )
        assert compare.status_code == 200
        assert "delta" in compare.json()
        assert "confidence" in compare.json()
        assert "anomaly_flags" in compare.json()
        assert "methodology" in compare.json()


def test_multi_night_simulation_endpoint(patched_api):
    with TestClient(patched_api.app) as client:
        response = client.post(
            "/api/simulation/multi-night",
            json={
                "nights": 3,
                "carryover_memory": True,
                "carryover_top_k": 3,
                "max_prior_events": 10,
                "config": _sim_payload(use_llm=False, duration_hours=1.0),
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["series_id"]
        assert len(payload["nights"]) == 3
        assert payload["summary"]["night_count"] == 3
        assert "continuity" in payload
        assert "recurring_memory_count" in payload["continuity"]
        assert "sankey" in payload["continuity"]
        assert len(payload["continuity"]["carryover_events_by_night"]) == 3
        assert (
            payload["continuity"]["carryover_events_by_night"][0]["carryover_events"]
            == []
        )
        for idx, night in enumerate(payload["nights"], start=1):
            assert night["summary"]["multi_night_series_id"] == payload["series_id"]
            assert night["summary"]["night_index"] == idx

        response_v1 = client.post(
            "/api/v1/simulation/multi-night",
            json={
                "nights": 2,
                "config": _sim_payload(use_llm=False, duration_hours=1.0),
            },
        )
        assert response_v1.status_code == 200


def test_simulation_stream_and_missing_id(patched_api):
    with TestClient(patched_api.app) as client:
        stream = client.post("/simulate-night/stream", json=_sim_payload())
        assert stream.status_code == 200
        assert "Simulation complete" in stream.text

        missing = client.get("/api/simulation/not-found")
        assert missing.status_code == 404


def test_metrics_and_narrative_quality_integration(patched_api):
    with TestClient(patched_api.app) as client:
        created = client.post("/api/simulation/night", json=_sim_payload())
        assert created.status_code == 201
        payload = created.json()
        assert payload["segments"]
        assert payload["segments"][0].get("narrative_quality") is not None
        assert "narrative_quality_mean" in payload["summary"]

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        metrics = metrics_resp.json()["metrics"]
        assert metrics["simulation_requests_total"] >= 1
        assert metrics["simulation_completed_total"] >= 1
        assert "simulation_duration_seconds_p95" in metrics
        assert "simulation_duration_seconds_p99" in metrics

        slo = client.get("/api/slo")
        assert slo.status_code == 200
        assert "targets" in slo.json()

        taxonomy = client.get("/api/error-taxonomy")
        assert taxonomy.status_code == 200
        assert "taxonomy" in taxonomy.json()

        release_gate = client.get("/api/release-gate")
        assert release_gate.status_code == 200
        assert "pass" in release_gate.json()
        assert "checks" in release_gate.json()
        assert "quality_window" in release_gate.json()
        assert "narrative_quality_pass" in release_gate.json()["checks"]
        assert "llm_fallback_sla_pass" in release_gate.json()["checks"]
        assert "memory_grounding_pass" in release_gate.json()["checks"]
        assert "simulation_latency_p95_pass" in release_gate.json()["checks"]

        prom = client.get("/metrics/prometheus")
        assert prom.status_code == 200
        assert "dreamforge_simulation_requests_total" in prom.text
        assert "dreamforge_simulation_duration_seconds_p95" in prom.text
        assert "dreamforge_simulation_duration_seconds_p99" in prom.text


def test_estimate_job_duration_uses_llm_specific_history(patched_api):
    with api_main._metrics_lock:
        api_main._simulation_duration_seconds_history.clear()
        api_main._simulation_duration_seconds_llm_history.clear()
        api_main._simulation_duration_seconds_no_llm_history.clear()
        api_main._simulation_duration_seconds_llm_history.extend([220.0, 260.0, 340.0])
        api_main._simulation_duration_seconds_no_llm_history.extend([8.0, 10.0, 12.0])
        api_main._simulation_duration_seconds_history.extend(
            [220.0, 260.0, 340.0, 8.0, 10.0, 12.0]
        )

    llm_cfg = api_main.SimulationConfig(
        duration_hours=8.0, dt_minutes=0.5, use_llm=True
    )
    no_llm_cfg = api_main.SimulationConfig(
        duration_hours=8.0, dt_minutes=0.5, use_llm=False
    )
    llm_est = api_main._estimate_job_duration_seconds(llm_cfg)
    no_llm_est = api_main._estimate_job_duration_seconds(no_llm_cfg)

    assert llm_est > no_llm_est
    assert llm_est >= 240.0
    assert no_llm_est <= 12.0


def test_job_progress_snapshot_recalibrates_stale_eta_for_long_llm_runs(
    patched_api, monkeypatch
):
    monkeypatch.setattr(api_main.time, "time", lambda: 1000.0)
    job = {
        "status": "running",
        "phase": "narrative",
        "progress_percent": 20.0,
        "eta_seconds": 5,
        "estimated_duration_seconds": 30.0,
        "started_at": 700.0,
        "last_progress_event_at": 760.0,
        "use_llm": True,
    }
    progress, eta_seconds = api_main._job_progress_snapshot(job, "running")

    assert progress < 20.0
    assert eta_seconds is not None
    assert eta_seconds > 60


def test_metrics_auth_exempt_is_configurable(patched_api, monkeypatch):
    monkeypatch.setattr(api_main, "_METRICS_PUBLIC", False)
    monkeypatch.setattr(api_main, "_API_ACCESS_TOKEN", "secret-token")
    monkeypatch.setattr(api_main, "_API_TOKEN_ROLE_MAP", {})

    with TestClient(patched_api.app) as client:
        denied = client.get("/api/v1/metrics/prometheus")
        assert denied.status_code == 401

        allowed = client.get(
            "/api/v1/metrics/prometheus", headers={"x-api-key": "secret-token"}
        )
        assert allowed.status_code == 200


def test_workspace_create_attach_and_list_runs(patched_api):
    with TestClient(patched_api.app) as client:
        created = client.post("/api/simulation/night", json=_sim_payload(use_llm=False))
        assert created.status_code == 201
        sim_id = created.json()["id"]

        ws = client.post(
            "/api/workspaces",
            json={
                "name": "Lab Workspace",
                "description": "night-run collection",
                "tags": ["qa", "continuity"],
            },
        )
        assert ws.status_code == 201
        workspace_id = ws.json()["id"]
        assert ws.json()["run_ids"] == []

        attached = client.post(
            f"/api/workspaces/{workspace_id}/runs",
            json={"simulation_id": sim_id, "label": "baseline"},
        )
        assert attached.status_code == 200
        assert attached.json()["attached"] is True
        assert attached.json()["run_count"] == 1

        attached_again = client.post(
            f"/api/workspaces/{workspace_id}/runs",
            json={"simulation_id": sim_id, "label": "baseline"},
        )
        assert attached_again.status_code == 200
        assert attached_again.json()["attached"] is False
        assert attached_again.json()["run_count"] == 1

        got_ws = client.get(f"/api/workspaces/{workspace_id}")
        assert got_ws.status_code == 200
        assert got_ws.json()["name"] == "Lab Workspace"
        assert got_ws.json()["run_ids"] == [sim_id]

        runs = client.get(f"/api/workspaces/{workspace_id}/runs")
        assert runs.status_code == 200
        assert runs.json()["count"] == 1
        assert runs.json()["items"][0]["simulation_id"] == sim_id

        ws_list = client.get("/api/workspaces")
        assert ws_list.status_code == 200
        assert ws_list.json()["count"] >= 1


def test_psg_channel_qa_and_plugin_evaluator_endpoints(patched_api):
    with TestClient(patched_api.app) as client:
        qa = client.post(
            "/api/psg/connectors/channel-qa",
            json={
                "channels": ["EEG C3-M2", "EOG-L", "Chin EMG", "ECG"],
                "expected_roles": {
                    "eeg": ["C3", "C4"],
                    "eog": ["EOG-L", "EOG-R"],
                    "emg": ["EMG", "CHIN"],
                },
            },
        )
        assert qa.status_code == 200
        qa_body = qa.json()
        assert qa_body["pass_check"] is True
        assert "eeg" in qa_body["matched_roles"]
        assert "ecg" not in qa_body["missing_roles"]

        evaluator_list = client.get("/api/plugins/evaluators")
        assert evaluator_list.status_code == 200
        assert evaluator_list.json()["count"] >= 2

        evaluator_run = client.post(
            "/api/plugins/evaluators/run",
            json={
                "evaluator": "quality-v1",
                "summary": {
                    "narrative_quality_mean": 0.8,
                    "narrative_memory_grounding_mean": 0.4,
                    "llm_total_invocations": 10,
                    "llm_fallback_segments": 1,
                },
            },
        )
        assert evaluator_run.status_code == 200
        assert evaluator_run.json()["evaluator"] == "quality-v1"
        assert evaluator_run.json()["scores"]["overall"] > 0.0


def test_output_index_and_retention(patched_api, monkeypatch, tmp_path):
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("DREAMFORGE_OUTPUT_DIR", str(output_root))
    monkeypatch.setattr(api_main, "_OUTPUT_RETENTION_DAYS", 1)
    monkeypatch.setattr(api_main, "_OUTPUT_RETENTION_MAX_RUNS", 2)

    old_dir = output_root / "old-run"
    old_dir.mkdir(parents=True, exist_ok=True)
    old_marker = old_dir / "payload.json"
    old_marker.write_text("{}", encoding="utf-8")
    old_ts = time.time() - (5 * 86400)
    os.utime(old_marker, (old_ts, old_ts))
    os.utime(old_dir, (old_ts, old_ts))

    result_payload = {
        "created_at_unix": time.time(),
        "llm_used": False,
        "summary": {
            "total_segments": 10,
            "rem_fraction": 0.2,
            "narrative_quality_mean": 0.6,
        },
    }

    run_a = output_root / "run-a"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b = output_root / "run-b"
    run_b.mkdir(parents=True, exist_ok=True)
    run_c = output_root / "run-c"
    run_c.mkdir(parents=True, exist_ok=True)

    api_main._upsert_output_metadata("run-a", result_payload, run_a)
    api_main._upsert_output_metadata("run-b", result_payload, run_b)
    api_main._upsert_output_metadata("run-c", result_payload, run_c)
    api_main._enforce_outputs_retention()

    index_payload = api_main._load_outputs_index()
    assert "items" in index_payload
    assert len(index_payload["items"]) <= 2
    assert not old_dir.exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "sleep_start_hour": 5.5,
            "duration_hours": 1.5,
            "melatonin": True,
            "prior_day_events": ["early run", "sunrise nap"],
        },
        {
            "sleep_start_hour": 13.0,
            "duration_hours": 2.0,
            "stress_level": 0.7,
            "ssri_strength": 1.2,
            "cannabis": True,
            "prior_day_events": ["lunch break nap", "heavy coffee crash"],
        },
        {
            "sleep_start_hour": 23.5,
            "duration_hours": 3.0,
            "stress_level": 0.1,
            "ssri_strength": 0.9,
            "melatonin": True,
            "cannabis": False,
            "prior_day_events": ["quiet reading", "deep relaxation"],
        },
    ],
)
def test_simulation_parameter_combinations_are_accepted(patched_api, overrides):
    with TestClient(patched_api.app) as client:
        payload = _sim_payload(**overrides)
        response = client.post("/api/simulation/night", json=payload)
        assert response.status_code == 201
        body = response.json()
        assert body["segments"]
        assert body["config"]["sleep_start_hour"] == payload["sleep_start_hour"]
        assert body["config"]["melatonin"] == payload["melatonin"]
        assert body["config"]["cannabis"] == payload["cannabis"]
        assert (
            body["summary"]["pharmacology_profile"]["ssri_strength"]
            == payload["ssri_strength"]
        )


def test_async_simulation_job_flow(patched_api):
    with TestClient(patched_api.app) as client:
        submit = client.post("/api/simulation/night/async", json=_sim_payload())
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        final_status = None
        last_progress = 0.0
        for _ in range(20):
            resp = client.get(f"/api/simulation/jobs/{job_id}")
            assert resp.status_code == 200
            body = resp.json()
            assert "progress_percent" in body
            assert "phase" in body
            assert "schema_version" in body
            assert body["schema_version"] == "v2"
            assert "progress_source" in body
            assert "eta_source" in body
            assert "provenance" in body
            assert "eta_seconds" in body
            assert "eta_margin_seconds" in body
            assert "estimated_duration_seconds" in body
            assert float(body["progress_percent"]) >= last_progress
            last_progress = float(body["progress_percent"])
            status_val = body["status"]
            if status_val in {"completed", "failed"}:
                final_status = status_val
                break
            time.sleep(0.05)

        assert final_status == "completed"


def test_async_simulation_job_queue_limit_returns_429(patched_api, monkeypatch):
    monkeypatch.setattr(api_main, "_ASYNC_MAX_PENDING_JOBS", 1)
    with api_main._jobs_lock:
        api_main._simulation_jobs["pending-existing"] = {
            "job_id": "pending-existing",
            "status": "pending",
            "created_at": time.time(),
        }

    with TestClient(patched_api.app) as client:
        submit = client.post("/api/simulation/night/async", json=_sim_payload())
        assert submit.status_code == 429
        assert "Async queue is full" in submit.json()["detail"]


def test_async_simulation_job_can_be_cancelled(patched_api, monkeypatch):
    async def _slow_simulation(_config):
        await asyncio.sleep(2.0)
        return {}

    monkeypatch.setattr(api_main, "simulate_night", _slow_simulation)

    with TestClient(patched_api.app) as client:
        submit = client.post("/api/simulation/night/async", json=_sim_payload())
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        cancel = client.post(f"/api/simulation/jobs/{job_id}/cancel")
        assert cancel.status_code == 200
        assert cancel.json()["status"] in {"cancelling", "cancelled"}

        final_status = None
        for _ in range(40):
            resp = client.get(f"/api/simulation/jobs/{job_id}")
            assert resp.status_code == 200
            status_val = resp.json()["status"]
            if status_val == "cancelled":
                final_status = status_val
                break
            time.sleep(0.05)

        assert final_status == "cancelled"


def test_state_log_fallback_persists_when_redis_unavailable(patched_api, monkeypatch):
    monkeypatch.setattr(api_main, "_get_redis_client", lambda: None)

    sim_payload = {"id": "sim-fallback", "summary": {"mean_bizarreness": 0.5}}
    api_main._persist_simulation("sim-fallback", sim_payload)
    loaded_sim = api_main._load_persisted_simulation("sim-fallback")
    assert loaded_sim is not None
    assert loaded_sim["id"] == "sim-fallback"

    job_payload = {"job_id": "job-fallback", "status": "pending", "created_at": 1.0}
    api_main._persist_job("job-fallback", job_payload)
    loaded_job = api_main._load_persisted_job("job-fallback")
    assert loaded_job is not None
    assert loaded_job["job_id"] == "job-fallback"

    workspace_payload = {"id": "ws-fallback", "name": "Fallback WS", "run_ids": []}
    api_main._persist_workspace("ws-fallback", workspace_payload)
    loaded_workspace = api_main._load_persisted_workspace("ws-fallback")
    assert loaded_workspace is not None
    assert loaded_workspace["id"] == "ws-fallback"


def test_enterprise_and_audit_surfaces(patched_api):
    with TestClient(patched_api.app) as client:
        created = client.post("/api/simulation/night", json=_sim_payload())
        assert created.status_code == 201

        enterprise = client.get("/api/enterprise")
        assert enterprise.status_code == 200
        enterprise_payload = enterprise.json()
        assert enterprise_payload["status"] == "active"
        assert "editions" in enterprise_payload

        audit = client.get("/api/audit/events")
        assert audit.status_code == 200
        assert "items" in audit.json()


@pytest.mark.asyncio
async def test_api_main_helpers_and_llm_parser(monkeypatch):
    fake = DummyLLMClient()
    monkeypatch.setattr(api_main, "get_llm_client", lambda: fake)

    await api_main.startup_event()

    assert api_main._strip_thinking_tags("<think>hidden</think>visible") == "visible"
    cfg = api_main.SimulationConfig(duration_hours=1.0, dt_minutes=0.5, use_llm=False)
    segs = api_main._simulate_night_physics(cfg)
    assert len(segs) > 0

    template_narrative, template_scene, template_bank = api_main._template_narrative(
        segs[0], cfg
    )
    assert template_narrative
    assert template_scene
    assert template_bank

    class JSONClient(DummyLLMClient):
        async def chat(self, system: str, user: str) -> str:
            return '```json {"narrative":"A dream","scene":"A room"} ```'

    narrative, scene = await api_main._generate_llm_narrative(
        segs[0], cfg, JSONClient()
    )
    assert narrative == "A dream"
    assert scene == "A room"

    class PlainClient(DummyLLMClient):
        async def chat(self, system: str, user: str) -> str:
            return "not-json output"

    fallback_narrative, fallback_scene = await api_main._generate_llm_narrative(
        segs[0], cfg, PlainClient()
    )
    assert fallback_narrative == "not-json output"
    assert segs[0]["stage"] in fallback_scene
