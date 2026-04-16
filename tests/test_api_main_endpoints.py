import time

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
def patched_api(monkeypatch):
    api_main._simulations.clear()
    with api_main._jobs_lock:
        api_main._simulation_jobs.clear()
    with api_main._rate_limit_lock:
        api_main._request_windows.clear()
    with api_main._metrics_lock:
        for key in list(api_main._runtime_metrics.keys()):
            api_main._runtime_metrics[key] = 0.0
        api_main._api_error_codes.clear()
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

        slo = client.get("/api/slo")
        assert slo.status_code == 200
        assert "targets" in slo.json()

        taxonomy = client.get("/api/error-taxonomy")
        assert taxonomy.status_code == 200
        assert "taxonomy" in taxonomy.json()

        prom = client.get("/metrics/prometheus")
        assert prom.status_code == 200
        assert "dreamforge_simulation_requests_total" in prom.text


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
        for _ in range(20):
            resp = client.get(f"/api/simulation/jobs/{job_id}")
            assert resp.status_code == 200
            status_val = resp.json()["status"]
            if status_val in {"completed", "failed"}:
                final_status = status_val
                break
            time.sleep(0.05)

        assert final_status == "completed"


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
