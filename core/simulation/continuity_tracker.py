from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from core.models.dream_segment import DreamNight
from core.simulation.event_bus import Event, EventType


class CrossNightContinuityTracker:
    """Utilities for analyzing cross-night continuity.

    This module focuses on two levels:
    1) Event-level activity matrices (for generic agent/event heatmaps).
    2) Memory-level recurrence across multiple DreamNight objects, including
       Sankey-ready data structures for recurring memory fragments.
    """

    # ------------------------------------------------------------------
    # 1. Event-level activity (already used for simple heatmaps)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_activity_matrix(events: List[Event]) -> dict[str, Any]:
        """Return a simple activity matrix (time bins x event types).

        The result is a dict with:
          - times: sorted list of time bins (floats)
          - types: list of event type names
          - matrix: len(times) x len(types) integer counts
        """

        if not events:
            return {"times": [], "types": [], "matrix": []}

        times = sorted({round(e.timestamp_hours, 2) for e in events})
        types = list(EventType)
        index = {t: i for i, t in enumerate(times)}
        matrix = [[0 for _ in types] for _ in times]

        type_index = {et: i for i, et in enumerate(types)}
        for e in events:
            ti = index[round(e.timestamp_hours, 2)]
            ei = type_index[e.type]
            matrix[ti][ei] += 1

        return {"times": times, "types": [t.value for t in types], "matrix": matrix}

    # ------------------------------------------------------------------
    # 2. Memory-level recurrence across nights
    # ------------------------------------------------------------------

    @staticmethod
    def _memory_usage_by_night(nights: List[DreamNight]) -> Dict[str, Set[int]]:
        """Map memory fragment IDs to the set of night indices where they appear."""

        usage: Dict[str, Set[int]] = {}
        for night_idx, night in enumerate(nights):
            for seg in night.segments:
                for mem_id in seg.active_memory_ids:
                    usage.setdefault(mem_id, set()).add(night_idx)
        return usage

    @staticmethod
    def compute_recurring_memory_stats(
        nights: List[DreamNight], min_nights: int = 2
    ) -> Dict[str, dict[str, Any]]:
        """Compute basic statistics for recurring memory fragments.

        Returns a dict mapping memory_id -> {"nights": [...], "count": int} for
        all fragments that appear in at least `min_nights` different nights.
        """

        usage = CrossNightContinuityTracker._memory_usage_by_night(nights)
        stats: Dict[str, dict[str, Any]] = {}
        for mem_id, night_set in usage.items():
            if len(night_set) >= min_nights:
                stats[mem_id] = {
                    "nights": sorted(night_set),
                    "count": sum(
                        1
                        for night_idx in night_set
                        for seg in nights[night_idx].segments
                        if mem_id in seg.active_memory_ids
                    ),
                }
        return stats

    @staticmethod
    def build_recurring_sankey(nights: List[DreamNight]) -> dict[str, Any]:
        """Build a Sankey-ready structure for recurring memory fragments.

        Nodes are `Night 1`, `Night 2`, ... `Night N`.
        Links aggregate how many distinct memory fragments recur between
        pairs of nights (i -> j, i < j), based on active_memory_ids.

        Returns a dict with keys suitable for Plotly Sankey:
          - nodes: {"labels": [...]}  (night labels)
          - links: {"source": [...], "target": [...], "value": [...]}  (flows)
        """

        num_nights = len(nights)
        if num_nights == 0:
            return {
                "nodes": {"labels": []},
                "links": {"source": [], "target": [], "value": []},
            }

        usage = CrossNightContinuityTracker._memory_usage_by_night(nights)

        # Initialize matrix of unique fragment recurrences between night i and j
        flows: Dict[Tuple[int, int], int] = {}
        for mem_id, night_set in usage.items():
            if len(night_set) < 2:
                continue
            sorted_nights = sorted(night_set)
            for i in range(len(sorted_nights) - 1):
                a = sorted_nights[i]
                b = sorted_nights[i + 1]
                flows[(a, b)] = flows.get((a, b), 0) + 1

        labels = [f"Night {i + 1}" for i in range(num_nights)]
        source: List[int] = []
        target: List[int] = []
        value: List[int] = []

        for (a, b), count in sorted(flows.items()):
            source.append(a)
            target.append(b)
            value.append(count)

        return {
            "nodes": {"labels": labels},
            "links": {"source": source, "target": target, "value": value},
        }
