from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from core.models.sleep_cycle import SleepStage


@dataclass
class SimulationTick:
    time_hours: float
    segment_index: int
    stage: SleepStage
    ach: float
    serotonin: float
    ne: float
    cortisol: float
    biz: float
    active_memory_ids: List[str]
    narrative: str
    replay_events: List[Dict[str, Any]]


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
        self.mem_events = []
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
