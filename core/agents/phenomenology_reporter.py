from __future__ import annotations

from typing import List, Optional

from core.models.dream_segment import DreamSegment, DreamNight
from core.simulation.event_bus import EventBus, Event, EventType


class PhenomenologyReporter:
    """Collects dream segments and exposes phenomenological summaries."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus or EventBus()
        self._segments: List[DreamSegment] = []

    def record_segment(self, segment: DreamSegment) -> None:
        self._segments.append(segment)
        event = Event(
            type=EventType.SLEEP_STAGE_UPDATED,
            payload={"segment_id": segment.id},
            timestamp_hours=segment.end_time_hours,
        )
        self.event_bus.publish(event)

    def build_night(self) -> DreamNight:
        return DreamNight(segments=list(self._segments))

    @property
    def segments(self) -> List[DreamSegment]:
        return list(self._segments)
