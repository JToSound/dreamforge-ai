from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def export_neurochemistry_csv(
    result: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Export per-segment neurochemistry and context fields to CSV.

    Args:
        result: Simulation result containing a ``segments`` array.
        output_path: Target CSV file path.

    Raises:
        ValueError: If no segments are present.
    """
    segments = result.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ValueError("export_neurochemistry_csv: result['segments'] is empty")

    rows: list[dict[str, Any]] = []
    for seg in segments:
        seg_dict = seg.model_dump() if hasattr(seg, "model_dump") else dict(seg)
        neuro = seg_dict.get("neurochemistry", {})
        rows.append(
            {
                "time_hours": seg_dict.get("start_time_hours"),
                "stage": seg_dict.get("stage"),
                "ach": neuro.get("ach"),
                "serotonin": neuro.get("serotonin"),
                "ne": neuro.get("ne"),
                "cortisol": neuro.get("cortisol"),
                "dominant_emotion": seg_dict.get("dominant_emotion"),
                "bizarreness_score": seg_dict.get("bizarreness_score"),
                "lucidity_probability": seg_dict.get("lucidity_probability"),
            }
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def export_memory_activations_csv(
    result: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Flatten memory activation snapshots into a tabular CSV format.

    Args:
        result: Simulation result containing ``memory_activations`` snapshots.
        output_path: Target CSV file path.

    Raises:
        ValueError: If snapshots are missing or empty.
    """
    snapshots = result.get("memory_activations")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError(
            "export_memory_activations_csv: result['memory_activations'] is empty"
        )

    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        snap_dict = snap.model_dump() if hasattr(snap, "model_dump") else dict(snap)
        time_hours = snap_dict.get("time_hours")
        activations = snap_dict.get("activations", [])
        if not isinstance(activations, list):
            continue
        for node in activations:
            node_dict = node.model_dump() if hasattr(node, "model_dump") else dict(node)
            rows.append(
                {
                    "time_hours": time_hours,
                    "node_id": node_dict.get("id"),
                    "node_label": node_dict.get("label"),
                    "activation": node_dict.get("activation"),
                }
            )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
