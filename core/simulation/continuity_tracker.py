from __future__ import annotations

from typing import List

from core.simulation.event_bus import Event, EventType


class CrossNightContinuityTracker:
    """Analyzes recurring themes and memory fragments across nights.

    This uses DreamNight objects at a higher level in the API, but the
    implementation here focuses on the event-level traces.
    """

    @staticmethod
    def compute_activity_matrix(events: List[Event]) -> dict:
        """Return a simple activity matrix (time bins x event types)."""

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
