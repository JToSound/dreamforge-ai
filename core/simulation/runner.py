from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from core.models.sleep_cycle import SleepStage


@dataclass
class SimulationTick:
    time_hours: float
    segment_index: int
    stage: SleepStage | str
    ach: float
    serotonin: float
    ne: float
    cortisol: float
    biz: float
    active_memory_ids: List[str]
    narrative: str
    replay_events: List[Dict[str, Any]]


def _segment_to_dict(segment: Any) -> Dict[str, Any]:
    """Normalize a segment object into a dictionary."""
    if hasattr(segment, "model_dump"):
        dumped = segment.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(segment, dict):
        return segment
    return {}


def build_neurochemistry_ticks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build per-segment neurochemistry rows for CSV/export consumers.

    Args:
        result: Full simulation result dict containing a `segments` list.

    Returns:
        A list of row dicts with time/stage/neurochemistry columns.

    Raises:
        KeyError: If `segments` key is missing.
        ValueError: If `segments` exists but is empty.
    """
    if "segments" not in result:
        raise KeyError("build_neurochemistry_ticks: result['segments'] is missing")

    segments_obj = result["segments"]
    if not isinstance(segments_obj, list):
        raise TypeError("build_neurochemistry_ticks: result['segments'] must be a list")
    if not segments_obj:
        raise ValueError("build_neurochemistry_ticks: result['segments'] is empty")

    rows: List[Dict[str, Any]] = []
    for seg in segments_obj:
        seg_dict = _segment_to_dict(seg)
        if not seg_dict:
            continue
        neuro = seg_dict.get("neurochemistry") or {}
        rows.append(
            {
                "time_hours": seg_dict.get(
                    "start_time_hours", seg_dict.get("time_hours")
                ),
                "stage": seg_dict.get("stage"),
                "ach": neuro.get("ach", float("nan")),
                "serotonin": neuro.get("serotonin", float("nan")),
                "ne": neuro.get("ne", float("nan")),
                "cortisol": neuro.get("cortisol", float("nan")),
            }
        )

    if not rows:
        raise ValueError("build_neurochemistry_ticks: no valid segment rows produced")
    return rows


def export_neurochemistry_csv(result: Dict[str, Any], output_path: Path) -> None:
    """Export per-tick neurochemistry to CSV for dashboard consumption.

    Args:
        result: Full simulation result dict containing `segments`.
        output_path: Destination CSV path.

    Raises:
        KeyError: If required top-level keys are missing.
        ValueError: If segments are empty or cannot produce rows.
    """
    rows = build_neurochemistry_ticks(result)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def export_segments_csv(result: Dict[str, Any], output_path: Path) -> None:
    """Export dream segments to CSV including generation-mode provenance.

    Args:
        result: Simulation result payload containing a `segments` list.
        output_path: Destination CSV path.

    Raises:
        ValueError: If no segments are available for export.
    """
    segments_obj = result.get("segments", [])
    if not isinstance(segments_obj, list) or not segments_obj:
        raise ValueError("export_segments_csv: result['segments'] is empty")

    rows: List[Dict[str, Any]] = []
    for seg in segments_obj:
        seg_dict = _segment_to_dict(seg)
        if not seg_dict:
            continue
        active_ids = seg_dict.get("active_memory_ids", [])
        if isinstance(active_ids, list):
            active_ids_text = ",".join(str(item) for item in active_ids)
        else:
            active_ids_text = str(active_ids or "")

        rows.append(
            {
                "id": seg_dict.get("id"),
                "start_time_hours": seg_dict.get("start_time_hours"),
                "end_time_hours": seg_dict.get("end_time_hours"),
                "stage": seg_dict.get("stage"),
                "dominant_emotion": seg_dict.get("dominant_emotion"),
                "bizarreness_score": seg_dict.get("bizarreness_score"),
                "lucidity_probability": seg_dict.get("lucidity_probability"),
                "generation_mode": seg_dict.get("generation_mode", "TEMPLATE"),
                "narrative": seg_dict.get("narrative", ""),
                "scene_description": seg_dict.get("scene_description", ""),
                "active_memory_ids": active_ids_text,
            }
        )

    if not rows:
        raise ValueError("export_segments_csv: no valid segment rows produced")
    pd.DataFrame(rows).to_csv(output_path, index=False)


class SimulationRunner:
    """Simple generator that yields SimulationTick objects from a simulation result.

    The runner uses the neurochemistry time-series if present, otherwise falls
    back to segment start times. Replay events from memory_graph.replay_events
    are attached to ticks by matching their time_hours to the nearest tick.
    """

    def __init__(self, sim: Dict[str, Any]):
        self.sim = sim or {}
        self.segments = self.sim.get("segments", [])
        self.neuro = (
            self.sim.get("neurochemistry")
            or self.sim.get("neurochemistry_series")
            or []
        )
        self.mem_events: List[Dict[str, Any]] = []
        mg = self.sim.get("memory_graph") or {}
        # Replay events may be exported under different keys
        self.mem_events = (
            mg.get("replay_events")
            or mg.get("replay_event_log")
            or self.sim.get("replay_event_log")
            or []
        )

    def _find_segment_for_time(self, t: float) -> Optional[int]:
        # segments are expected to have start_time_hours / end_time_hours
        for idx, s in enumerate(self.segments):
            st = s.get("start_time_hours") or s.get("time_hours") or s.get("start_time")
            en = s.get("end_time_hours") or s.get("end_time")
            if st is None:
                continue
            if en is None:
                # assume segment-resolution aligns
                if float(st) <= float(t):
                    return idx
            else:
                if float(st) <= float(t) < float(en):
                    return idx
        return None

    def run(self) -> Iterator[SimulationTick]:
        # Use neuro timeline if available
        if self.neuro and isinstance(self.neuro, list) and len(self.neuro) > 0:
            timeline = self.neuro
            for i, n in enumerate(timeline):
                t = float(n.get("time_hours", i))
                seg_idx = self._find_segment_for_time(t)
                seg = (
                    self.segments[seg_idx]
                    if seg_idx is not None and seg_idx < len(self.segments)
                    else {}
                )
                # gather replay events near this time (within small epsilon)
                events = [
                    e
                    for e in self.mem_events
                    if abs(float(e.get("time_hours", 0.0)) - t) < 1e-3
                ]
                tick = SimulationTick(
                    time_hours=t,
                    segment_index=seg_idx or 0,
                    stage=(seg.get("stage") or "N2"),
                    ach=float(n.get("ach", 0.0)),
                    serotonin=float(
                        n.get(
                            "serotonin",
                            n.get("5ht", 0.0) if n.get("5ht") is not None else 0.0,
                        )
                    ),
                    ne=float(n.get("ne", 0.0)),
                    cortisol=float(n.get("cortisol", 0.0)),
                    biz=float(
                        seg.get("bizarreness") or seg.get("bizarreness_score") or 0.0
                    ),
                    active_memory_ids=seg.get("active_memory_ids") or [],
                    narrative=seg.get("narrative")
                    or seg.get("scene_description")
                    or "",
                    replay_events=events,
                )
                yield tick
        else:
            # Fallback: step through segments
            for i, s in enumerate(self.segments):
                t = float(s.get("start_time_hours") or s.get("time_hours") or i)
                events = [
                    e
                    for e in self.mem_events
                    if abs(float(e.get("time_hours", 0.0)) - t) < 1e-3
                ]
                tick = SimulationTick(
                    time_hours=t,
                    segment_index=i,
                    stage=(s.get("stage") or "N2"),
                    ach=float((s.get("neurochemistry") or {}).get("ach", 0.0)),
                    serotonin=float(
                        (s.get("neurochemistry") or {}).get("serotonin", 0.0)
                    ),
                    ne=float((s.get("neurochemistry") or {}).get("ne", 0.0)),
                    cortisol=float(
                        (s.get("neurochemistry") or {}).get("cortisol", 0.0)
                    ),
                    biz=float(
                        s.get("bizarreness") or s.get("bizarreness_score") or 0.0
                    ),
                    active_memory_ids=s.get("active_memory_ids") or [],
                    narrative=s.get("narrative") or s.get("scene_description") or "",
                    replay_events=events,
                )
                yield tick
