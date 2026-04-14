from __future__ import annotations

from typing import Dict, List

from core.models.dream_segment import DreamSegment
from core.models.memory_graph import MemoryGraph
from core.models.neurochemistry import NeurochemistryState
from core.models.sleep_cycle import SleepState, SleepStage


def _stage_durations(states: List[SleepState]) -> Dict[str, float]:
    if not states:
        return {}

    durations: Dict[SleepStage, float] = {s: 0.0 for s in SleepStage}
    for prev, curr in zip(states[:-1], states[1:]):
        dt = curr.time_hours - prev.time_hours
        durations[prev.stage] += max(0.0, dt)

    total = sum(durations.values()) or 1.0
    return {stage.value: dur / total for stage, dur in durations.items()}


def _neuro_stats(states: List[NeurochemistryState]) -> Dict[str, Dict[str, float]]:
    if not states:
        return {}

    def stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "max": 0.0}
        n = float(len(values))
        mean = sum(values) / n
        max_v = max(values)
        return {"mean": mean, "max": max_v}

    ach = [s.ach for s in states]
    five_ht = [s.serotonin for s in states]
    ne = [s.ne for s in states]
    cort = [s.cortisol for s in states]

    return {
        "ACh": stats(ach),
        "5-HT": stats(five_ht),
        "NE": stats(ne),
        "Cortisol": stats(cort),
    }


def _bizarreness_stats(segments: List[DreamSegment]) -> Dict[str, object]:
    if not segments:
        return {"mean": 0.0, "std": 0.0, "top_segments": []}

    scores = [s.bizarreness_score for s in segments]
    n = float(len(scores))
    mean = sum(scores) / n
    var = sum((x - mean) ** 2 for x in scores) / n
    std = var**0.5

    top = sorted(segments, key=lambda s: s.bizarreness_score, reverse=True)[:5]
    top_info = [
        {
            "id": s.id,
            "start_time_hours": s.start_time_hours,
            "end_time_hours": s.end_time_hours,
            "bizarreness_score": s.bizarreness_score,
            "stage": s.stage.value,
        }
        for s in top
    ]

    return {"mean": mean, "std": std, "top_segments": top_info}


def _top_memory_nodes(
    segments: List[DreamSegment], memory_graph: MemoryGraph, k: int = 5
) -> List[Dict[str, object]]:
    g = memory_graph.to_networkx()
    if g.number_of_nodes() == 0:
        return []

    freq: Dict[str, int] = {}
    for seg in segments:
        for nid in seg.active_memory_ids:
            freq[nid] = freq.get(nid, 0) + 1

    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:k]
    result: List[Dict[str, object]] = []
    for nid, count in ranked:
        if nid not in g:
            continue
        data = g.nodes[nid]
        result.append(
            {
                "id": nid,
                "label": data.get("label", nid),
                "emotion": data.get("emotion", "neutral"),
                "count": count,
                "salience": float(data.get("salience", 0.0)),
            }
        )
    return result


def compute_night_summary(
    *,
    sleep_history: List[SleepState],
    neuro_history: List[NeurochemistryState],
    segments: List[DreamSegment],
    memory_graph: MemoryGraph,
) -> Dict[str, object]:
    """Compute summary metrics for a simulated night.

    Returns a JSON-serializable dict that can be used by the API and dashboard
    to render high-level insight cards and charts.
    """

    return {
        "sleep_stages": _stage_durations(sleep_history),
        "neurochemistry": _neuro_stats(neuro_history),
        "bizarreness": _bizarreness_stats(segments),
        "memory": {"top_nodes": _top_memory_nodes(segments, memory_graph)},
    }
