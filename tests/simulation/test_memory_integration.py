from __future__ import annotations

from pathlib import Path

import pandas as pd

import api.main as api_main
from core.models.memory_graph import (
    EmotionLabel,
    MemoryGraph,
    MemoryNodeModel,
    MemoryType,
)
from core.simulation.runner import export_segments_csv


def _run_segments() -> list[dict]:
    cfg = api_main.SimulationConfig(
        duration_hours=8.0,
        dt_minutes=0.5,
        stress_level=0.7,
        use_llm=False,
    )
    segments = api_main._simulate_night_physics(cfg)
    api_main._build_memory_outputs(segments, ["missed train", "family dinner"])
    return segments


def test_active_memory_ids_propagated_to_segment() -> None:
    segments = _run_segments()
    assert any(bool(seg.get("active_memory_ids")) for seg in segments)
    for seg in segments:
        assert isinstance(seg.get("active_memory_ids"), list)


def test_memory_label_helper_returns_string() -> None:
    graph = MemoryGraph()
    node_id = graph.add_memory(
        MemoryNodeModel(
            label="old classroom",
            memory_type=MemoryType.EPISODIC,
            emotion=EmotionLabel.NEUTRAL,
        )
    )
    assert isinstance(graph.label(node_id), str)
    assert graph.label("missing-id") == "missing-id"


def test_segments_csv_active_memory_ids_nonempty_gte_30pct_rem(tmp_path: Path) -> None:
    segments = _run_segments()
    rem_segments = [s for s in segments if s.get("stage") == "REM"]
    assert rem_segments
    nonempty = [s for s in rem_segments if s.get("active_memory_ids")]
    ratio = len(nonempty) / len(rem_segments)
    assert ratio >= 0.30

    out = tmp_path / "segments.csv"
    export_segments_csv({"segments": segments}, out)
    df = pd.read_csv(out)
    assert "active_memory_ids" in df.columns
