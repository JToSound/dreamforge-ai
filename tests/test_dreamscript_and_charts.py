from __future__ import annotations

import time
import importlib

import pytest

from core.models.sleep_cycle import SleepStage
from core.simulation.dreamscript import DreamScriptEngine
from core.simulation.runner import SimulationRunner
from core.utils.bizarreness_scorer import BizarrenessScore
from core.utils.llm_backend import LLMBackend, Providers


def _biz(score: float = 0.8) -> BizarrenessScore:
    return BizarrenessScore(
        total_score=score,
        discontinuity_score=min(1.0, score * 0.4),
        incongruity_score=min(1.0, score * 0.35),
        implausibility_score=min(1.0, score * 0.25),
        confidence_interval=(max(0.0, score - 0.05), min(1.0, score + 0.05)),
    )


def _segments() -> list[dict]:
    return [
        {
            "stage": "N1",
            "start_time_hours": 0.0,
            "end_time_hours": 0.5,
            "dominant_emotion": "neutral",
            "bizarreness_score": 0.22,
            "neurochemistry": {"cortisol": 0.35},
        },
        {
            "stage": "N2",
            "start_time_hours": 0.5,
            "end_time_hours": 1.0,
            "dominant_emotion": "joy",
            "bizarreness_score": 0.35,
            "neurochemistry": {"cortisol": 0.42},
        },
        {
            "stage": "REM",
            "start_time_hours": 1.0,
            "end_time_hours": 1.5,
            "dominant_emotion": "fear",
            "bizarreness_score": 0.8,
            "neurochemistry": {"cortisol": 0.68},
        },
        {
            "stage": "REM",
            "start_time_hours": 1.5,
            "end_time_hours": 2.0,
            "dominant_emotion": "joy",
            "bizarreness_score": 0.86,
            "neurochemistry": {"cortisol": 0.78},
        },
    ]


def test_dreamscript_has_full_banks():
    engine = DreamScriptEngine(seed=1)
    assert len(engine.NREM_LIGHT) >= 30
    assert len(engine.NREM_DEEP) >= 30
    assert len(engine.REM_EARLY) >= 30
    assert len(engine.REM_LATE) >= 30


def test_dreamscript_modulation_and_continuity():
    engine = DreamScriptEngine(seed=3)
    neuro = type("N", (), {"ach": 0.9, "ne": 0.05, "cortisol": 0.88})()
    text = engine.generate_narrative(
        stage=SleepStage.REM,
        neurochemistry=neuro,
        active_memories=[],
        bizarreness=_biz(0.92),
        prev_segment_text='I am following "Alice" through mirrored rooms.',
    )
    assert "Alice" in text
    assert "The geometry turns" in text
    assert any(
        token in text
        for token in (
            "Nobody questions the contradiction",
            "I accept the impossible rule",
            "The scene breaks continuity",
            "Causality slips",
            "Time jumps forward and backward",
        )
    )
    assert any(
        token in text
        for token in (
            "background alarm",
            "sharp edge of urgency",
            "chest tightens",
            "tense, watchful",
            "low dread hums",
        )
    )


def test_llm_backend_demo_mode_forces_offline(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    class NoNetworkClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, *_args, **_kwargs):
            raise RuntimeError("network disabled")

    monkeypatch.setattr("core.utils.llm_backend.httpx.Client", NoNetworkClient)

    backend = LLMBackend()
    assert backend.config.provider == Providers.DREAMSCRIPT
    out = backend.generate_offline("stage=REM;ach=0.9;ne=0.05;cortisol=0.9")
    assert isinstance(out, str) and len(out) > 0


def test_static_chart_builders_return_figures():
    go = pytest.importorskip("plotly.graph_objects")
    charts = importlib.import_module("visualization.charts.static_visualizations")

    segs = _segments()
    figs = [
        charts.plot_rem_episode_trend(segs),
        charts.plot_affect_ratio_timeline(segs),
        charts.plot_bizarreness_cortisol_scatter(segs),
        charts.plot_per_cycle_architecture(segs),
    ]
    for fig in figs:
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    cfg = charts.chart_export_config()
    assert cfg["toImageButtonOptions"]["format"] == "png"


def test_simulation_runner_handles_960_ticks_quickly():
    neuro = [
        {
            "time_hours": i / 120.0,
            "ach": 0.5,
            "serotonin": 0.3,
            "ne": 0.2,
            "cortisol": 0.4,
        }
        for i in range(960)
    ]
    segments = [
        {
            "stage": "N2",
            "start_time_hours": i / 120.0,
            "end_time_hours": (i + 1) / 120.0,
            "bizarreness_score": 0.3,
            "active_memory_ids": [],
            "narrative": "",
        }
        for i in range(960)
    ]
    runner = SimulationRunner({"segments": segments, "neurochemistry": neuro})
    t0 = time.perf_counter()
    ticks = list(runner.run())
    elapsed = time.perf_counter() - t0

    assert len(ticks) == 960
    assert elapsed < 3.0
