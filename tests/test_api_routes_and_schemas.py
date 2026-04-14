import importlib

import pytest

from api.schemas import (
    DreamNightSchema,
    DreamSimulationRequest,
    SimulateNightRequest,
    serialize_dream_night,
)


def test_serialize_dream_night_from_dict_and_object():
    payload = {
        "id": "night-1",
        "segments": [
            {
                "id": "seg-1",
                "start_time_hours": 0.0,
                "end_time_hours": 0.5,
                "stage": "N2",
            }
        ],
        "config": {"duration_hours": 1.0},
        "metadata": {"k": "v"},
    }
    out = serialize_dream_night(payload)
    assert isinstance(out, DreamNightSchema)
    assert out.id == "night-1"
    assert out.segments[0].stage == "N2"

    class NightObject:
        def __init__(self):
            self.id = "night-2"
            self.segments = payload["segments"]
            self.config = {"duration_hours": 2.0}
            self.notes = "note"
            self.metadata = {"m": 1}

    out_obj = serialize_dream_night(NightObject())
    assert out_obj.id == "night-2"
    assert out_obj.notes == "note"


def test_simulate_night_request_defaults():
    req = SimulateNightRequest()
    assert req.sleep.duration_hours == 8.0
    assert req.memory.max_nodes >= 10


@pytest.mark.asyncio
async def test_simulation_route_with_fake_engine(monkeypatch):
    import core.agents.orchestrator as orchestrator_module

    if not hasattr(orchestrator_module, "OrchestratorConfig"):

        class _OrchestratorConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        monkeypatch.setattr(
            orchestrator_module,
            "OrchestratorConfig",
            _OrchestratorConfig,
            raising=False,
        )

    sim_route = importlib.import_module("api.routes.simulation")

    class FakeEngine:
        def __init__(self, config):
            self.config = config

        def simulate_night(self):
            return None

        def build_night(self):
            class FakeNight:
                def __init__(self):
                    self.id = "fake-night"
                    self.segments = [
                        {
                            "id": "seg-1",
                            "start_time_hours": 0.0,
                            "end_time_hours": 0.5,
                            "stage": "N2",
                        }
                    ]
                    self.config = {}
                    self.notes = None
                    self.metadata = {"source": "fake"}

            return FakeNight()

    monkeypatch.setattr(sim_route, "SimulationEngine", FakeEngine)
    body = DreamSimulationRequest(duration_hours=1.0, dt_minutes=0.5, llm_enabled=False)
    out = await sim_route.simulate_night(body)
    assert out.id == "fake-night"
    assert out.config["duration_hours"] == 1.0


@pytest.mark.asyncio
async def test_llm_settings_routes(monkeypatch):
    import core.agents.orchestrator as orchestrator_module

    if not hasattr(orchestrator_module, "OrchestratorConfig"):

        class _OrchestratorConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        monkeypatch.setattr(
            orchestrator_module,
            "OrchestratorConfig",
            _OrchestratorConfig,
            raising=False,
        )

    llm_route = importlib.import_module("api.routes.llm_settings")

    class FakeConfig:
        provider = "openai"
        base_url = "http://fake.local/v1"
        model = "fake-model"
        max_tokens = 64
        temperature = 0.4

    class FakeClient:
        def __init__(self, cfg=None):
            self.config = cfg or FakeConfig()

        async def check_health(self):
            return {"ok": True, "models": [self.config.model]}

        async def aclose(self):
            return None

    fake_client = FakeClient()
    monkeypatch.setattr(llm_route.llm_module, "get_llm_client", lambda: fake_client)
    monkeypatch.setattr(llm_route, "LLMClient", FakeClient)
    llm_route.llm_module._default_client = fake_client

    health = await llm_route.llm_health()
    assert health["ok"] is True

    settings = await llm_route.get_settings()
    assert settings["model"] == "fake-model"

    updated = await llm_route.update_settings(
        llm_route.LLMSettingsUpdate(model="m2", max_tokens=256)
    )
    assert updated["updated"] is True
    assert "health" in updated


@pytest.mark.asyncio
async def test_journal_route(monkeypatch):
    import core.agents.orchestrator as orchestrator_module

    if not hasattr(orchestrator_module, "OrchestratorConfig"):

        class _OrchestratorConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        monkeypatch.setattr(
            orchestrator_module,
            "OrchestratorConfig",
            _OrchestratorConfig,
            raising=False,
        )

    journal_route = importlib.import_module("api.routes.journal")

    captured = {}

    def _fake_append_journal_entry(text, emotion, stress_level, tags):
        captured["text"] = text
        captured["emotion"] = emotion
        captured["stress_level"] = stress_level
        captured["tags"] = tags

    monkeypatch.setattr(
        journal_route, "append_journal_entry", _fake_append_journal_entry
    )

    body = journal_route.JournalEntryRequest(
        text="meeting with friend",
        emotion="joy",
        stress_level=0.2,
        tags=["social", "friend"],
    )
    out = await journal_route.encode_journal(body)
    assert out["status"] == "ok"
    assert captured["emotion"] == "joy"
