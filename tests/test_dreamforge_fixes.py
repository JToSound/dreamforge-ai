"""Regression tests for DreamForge session ec094636 bug fixes."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from core.simulation.runner import export_neurochemistry_csv


@pytest.fixture
def mock_result_960() -> dict[str, Any]:
    dt_hours = 1.0 / 120.0
    segments: list[dict[str, Any]] = []
    for i in range(960):
        t = round(i * dt_hours, 6)
        stage = "REM" if i % 20 == 0 else "N2"
        segments.append(
            {
                "id": f"seg-{i}",
                "start_time_hours": t,
                "end_time_hours": round(t + dt_hours, 6),
                "stage": stage,
                "dominant_emotion": "neutral",
                "bizarreness_score": 0.7 if stage == "REM" else 0.3,
                "lucidity_probability": 0.4 if stage == "REM" else 0.05,
                "narrative": "mock narrative",
                "scene_description": "mock scene",
                "active_memory_ids": [f"m{i%3}"],
                "generation_mode": "TEMPLATE",
                "neurochemistry": {
                    "ach": 0.8 if stage == "REM" else 0.4,
                    "serotonin": 0.1 if stage == "REM" else 0.35,
                    "ne": 0.08 if stage == "REM" else 0.3,
                    "cortisol": 0.25 + 0.6 * min(1.0, t / 8.0),
                },
            }
        )
    return {"segments": segments}


@pytest.fixture
def completed_simulation_result(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    random.seed(0)
    np.random.seed(0)

    class DummyClient:
        class _Config:
            provider = "offline"
            base_url = "http://offline.local"
            model = "offline"

        config = _Config()

        async def check_health(self) -> dict[str, Any]:
            return {"ok": True, "models": ["offline"]}

        async def chat(self, system: str, user: str) -> str:
            return '{"narrative":"offline","scene":"offline"}'

        async def aclose(self) -> None:
            return None

    api_main._simulations.clear()
    monkeypatch.setattr(api_main, "get_llm_client", lambda: DummyClient())

    payload = {
        "duration_hours": 8.0,
        "dt_minutes": 0.5,
        "ssri_strength": 1.0,
        "stress_level": 0.2,
        "sleep_start_hour": 23.0,
        "prior_day_events": ["team review", "morning commute"],
        "emotional_state": "neutral",
        "use_llm": True,
        "llm_segments_only": False,
    }
    with TestClient(api_main.app) as client:
        response = client.post("/api/simulation/night", json=payload)
        assert response.status_code == 201
        return response.json()


@pytest.fixture
def completed_hypnogram(completed_simulation_result: dict[str, Any]) -> list[str]:
    return [str(s.get("stage", "")) for s in completed_simulation_result["segments"]]


class TestNeurochemistryCSVExport:
    def test_export_creates_960_rows(
        self, tmp_path: Path, mock_result_960: dict[str, Any]
    ) -> None:
        out = tmp_path / "nchem.csv"
        export_neurochemistry_csv(mock_result_960, out)
        df = pd.read_csv(out)
        assert len(df) == 960
        assert set(df.columns) >= {"time_hours", "ach", "serotonin", "ne", "cortisol"}

    def test_export_raises_on_empty_segments(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            export_neurochemistry_csv({"segments": []}, tmp_path / "nchem.csv")


class TestMemoryActivations:
    def test_result_contains_memory_keys(
        self, completed_simulation_result: dict[str, Any]
    ) -> None:
        assert "memory_activations" in completed_simulation_result
        assert "memory_graph" in completed_simulation_result

    def test_memory_activations_nonempty_for_rem_session(
        self, completed_simulation_result: dict[str, Any]
    ) -> None:
        snaps = completed_simulation_result["memory_activations"]
        assert len(snaps) >= 1
        assert all("time_hours" in s and "activations" in s for s in snaps)

    def test_memory_graph_contains_activation_snapshots(
        self, completed_simulation_result: dict[str, Any]
    ) -> None:
        memory_graph = completed_simulation_result["memory_graph"]
        assert "activation_snapshots" in memory_graph
        assert isinstance(memory_graph["activation_snapshots"], list)
        assert len(memory_graph["activation_snapshots"]) >= 1


class TestSleepStageProportions:
    def test_n3_proportion_within_physiological_range(
        self, completed_hypnogram: list[str]
    ) -> None:
        total = len(completed_hypnogram)
        n3_count = sum(1 for s in completed_hypnogram if s == "N3")
        proportion = n3_count / total
        assert (
            0.13 <= proportion <= 0.23
        ), f"N3 proportion {proportion:.1%} outside physiological range 13–23%"

    def test_n1_proportion_below_ceiling(self, completed_hypnogram: list[str]) -> None:
        total = len(completed_hypnogram)
        n1_count = sum(1 for s in completed_hypnogram if s == "N1")
        assert n1_count / total <= 0.12, "N1 exceeds 12% ceiling"


class TestCortisolProfile:
    def test_nadir_between_2_and_3_hours(self) -> None:
        from core.models.neurochemistry import cortisol_profile

        values = [cortisol_profile(t) for t in [x / 10 for x in range(20, 30)]]
        assert min(values) < 0.20, "Cortisol nadir should be < 0.20 at 02:00–03:00"

    def test_peak_at_7_5_hours(self) -> None:
        from core.models.neurochemistry import cortisol_profile

        peak_val = cortisol_profile(7.5)
        assert (
            peak_val >= 0.85
        ), f"Cortisol at 07:30 should be >= 0.85, got {peak_val:.3f}"

    def test_profile_monotonically_rising_from_nadir(self) -> None:
        from core.models.neurochemistry import cortisol_profile

        rising_values = [cortisol_profile(t) for t in [3.0, 4.5, 6.0, 7.5]]
        assert rising_values == sorted(
            rising_values
        ), "Cortisol must rise monotonically from nadir to peak"


class TestGenerationModeCSV:
    """Verify generation_mode column exists and contains valid values."""

    def test_segments_csv_has_generation_mode_column(
        self, tmp_path: Path, mock_result_960: dict[str, Any]
    ) -> None:
        from core.simulation.runner import export_segments_csv

        out = tmp_path / "segments.csv"
        export_segments_csv(mock_result_960, out)
        df = pd.read_csv(out)
        assert "generation_mode" in df.columns, (
            "generation_mode column missing from segments CSV — "
            "LLM trigger rate cannot be audited"
        )

    def test_generation_mode_values_are_valid_enum(
        self, tmp_path: Path, mock_result_960: dict[str, Any]
    ) -> None:
        from core.simulation.runner import export_segments_csv

        out = tmp_path / "segments.csv"
        export_segments_csv(mock_result_960, out)
        df = pd.read_csv(out)
        valid = {"LLM", "TEMPLATE", "LLM_FALLBACK", "CACHED"}
        unexpected = set(df["generation_mode"].unique()) - valid
        assert not unexpected, f"Unexpected generation_mode values found: {unexpected}"

    def test_segments_csv_has_trigger_latency_and_template_columns(
        self, tmp_path: Path, mock_result_960: dict[str, Any]
    ) -> None:
        from core.simulation.runner import export_segments_csv

        out = tmp_path / "segments.csv"
        export_segments_csv(mock_result_960, out)
        df = pd.read_csv(out)
        assert {"llm_trigger_type", "llm_latency_ms", "template_bank"}.issubset(
            set(df.columns)
        )

    def test_llm_trigger_rate_within_expected_range(
        self, completed_simulation_result: dict[str, Any]
    ) -> None:
        segments = completed_simulation_result["segments"]
        llm_count = sum(1 for s in segments if s.get("generation_mode") == "LLM")
        assert 10 <= llm_count <= 50, (
            f"LLM trigger count {llm_count} is outside expected range 10–50 "
            f"(architecture target: ~15–30 per 8h simulation)"
        )

    def test_generation_mode_not_all_template(
        self, completed_simulation_result: dict[str, Any]
    ) -> None:
        segments = completed_simulation_result["segments"]
        modes = [s.get("generation_mode") for s in segments]
        assert "LLM" in modes, (
            "All segments are TEMPLATE — LLM was never triggered. "
            "Check trigger conditions and LLM connectivity."
        )


class TestParseNarrativeResponse:
    """Verify parse_narrative_response handles all observed failure modes."""

    def test_parses_clean_json(self) -> None:
        from core.simulation.llm_client import parse_narrative_response

        content = (
            '{"narrative": "I stood at the edge of a vast ocean.", '
            '"scene": "coastal cliff"}'
        )
        result = parse_narrative_response(content)
        assert result["narrative"] == "I stood at the edge of a vast ocean."
        assert result["scene_description"] == "coastal cliff"

    def test_strips_thinking_tags_before_parsing(self) -> None:
        from core.simulation.llm_client import parse_narrative_response

        content = (
            "<think>Let me think about this dream scene carefully...</think>\n"
            '{"narrative": "The hallway stretched endlessly.", '
            '"scene": "school corridor"}'
        )
        result = parse_narrative_response(content)
        assert result["narrative"] == "The hallway stretched endlessly."

    def test_partial_json_regex_fallback(self) -> None:
        from core.simulation.llm_client import parse_narrative_response

        content = '{"narrative": "Shadows moved across the ceiling", "scene": "bedroo'
        result = parse_narrative_response(content)
        assert result["narrative"] == "Shadows moved across the ceiling"

    def test_empty_content_returns_empty_strings(self) -> None:
        from core.simulation.llm_client import parse_narrative_response

        result = parse_narrative_response("")
        assert result["narrative"] == ""
        assert result["scene_description"] == ""

    def test_no_think_flag_suppresses_reasoning_tokens(self) -> None:
        from core.simulation.llm_prompts import build_narrative_messages

        messages = build_narrative_messages(
            stage="REM",
            emotion="fearful",
            bizarreness=0.82,
            ach=0.9,
            ne=0.1,
            cortisol=0.3,
        )
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert user_content.startswith("/no_think"), (
            "User message must start with /no_think to suppress "
            "Qwen3.5 reasoning tokens."
        )

    def test_max_tokens_is_at_least_2048(self) -> None:
        from core.simulation import config as sim_config

        max_tokens = getattr(sim_config, "LLM_MAX_TOKENS", None)
        assert max_tokens is not None, "LLM_MAX_TOKENS not defined in sim_config"
        assert max_tokens >= 2048, (
            f"LLM_MAX_TOKENS={max_tokens} is too small for Qwen3.5 thinking model. "
            f"Set to 2048 minimum."
        )
