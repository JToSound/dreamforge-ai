from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import api.main as api_main
from core.simulation.exporters import (
    export_memory_activations_csv,
    export_neurochemistry_csv,
)


@pytest.fixture
def minimal_result() -> dict[str, object]:
    config = api_main.SimulationConfig(
        duration_hours=1.0, dt_minutes=0.5, use_llm=False
    )
    segments = api_main._simulate_night_physics(config)
    _, memory_activations = api_main._build_memory_outputs(
        segments, config.prior_day_events
    )
    return {"segments": segments, "memory_activations": memory_activations}


def test_neurochemistry_csv_has_all_columns(
    tmp_path: Path, minimal_result: dict[str, object]
) -> None:
    output_path = tmp_path / "neurochemistry.csv"
    export_neurochemistry_csv(minimal_result, output_path)
    df = pd.read_csv(output_path)
    expected = {
        "time_hours",
        "stage",
        "ach",
        "serotonin",
        "ne",
        "cortisol",
        "dominant_emotion",
        "bizarreness_score",
        "lucidity_probability",
    }
    assert expected.issubset(set(df.columns))
    segments = minimal_result["segments"]
    assert isinstance(segments, list)
    assert len(df) == len(segments)


def test_neurochemistry_csv_non_empty(
    tmp_path: Path, minimal_result: dict[str, object]
) -> None:
    output_path = tmp_path / "neurochemistry.csv"
    export_neurochemistry_csv(minimal_result, output_path)
    df = pd.read_csv(output_path)
    assert len(df) > 0


def test_export_raises_on_empty_segments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export_neurochemistry_csv({"segments": []}, tmp_path / "empty.csv")


def test_memory_activations_csv_structure(tmp_path: Path) -> None:
    snapshots = []
    for snapshot_idx in range(3):
        snapshots.append(
            {
                "time_hours": float(snapshot_idx) * 0.5,
                "activations": [
                    {
                        "id": f"node_{node_idx}",
                        "label": f"Node {node_idx}",
                        "activation": 0.1 * (node_idx + 1),
                    }
                    for node_idx in range(5)
                ],
            }
        )

    output_path = tmp_path / "memory_activations.csv"
    export_memory_activations_csv({"memory_activations": snapshots}, output_path)
    df = pd.read_csv(output_path)

    assert set(df.columns) == {"time_hours", "node_id", "node_label", "activation"}
    assert len(df) == 15


def test_memory_graph_is_sparse_and_avoids_activation_saturation() -> None:
    config = api_main.SimulationConfig(
        duration_hours=4.0, dt_minutes=0.5, use_llm=False
    )
    segments = api_main._simulate_night_physics(config)
    memory_graph, _ = api_main._build_memory_outputs(
        segments, ["late meeting", "family dinner", "city bus ride"]
    )
    nodes = memory_graph["nodes"]
    edges = memory_graph["edges"]

    node_count = len(nodes)
    max_complete_edges = (node_count * (node_count - 1)) // 2
    degree_cap_upper_bound = (node_count * 5) // 2

    assert len(edges) < max_complete_edges
    assert len(edges) <= degree_cap_upper_bound
    assert max(float(n["activation"]) for n in nodes) <= 0.95
