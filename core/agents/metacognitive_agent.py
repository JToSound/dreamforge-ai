from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.models.dream_segment import DreamSegment
from core.models.neurochemistry import NeurochemistryState
from core.models.sleep_cycle import SleepState, SleepStage
from core.simulation.event_bus import EventBus, Event, EventType


@dataclass
class MetacognitiveConfig:
    base_lucidity_rem: float = 0.3
    base_lucidity_nrem: float = 0.05


class MetacognitiveAgent:
    """Estimates lucidity probability and tracks metacognitive signals."""

    def __init__(self, event_bus: Optional[EventBus] = None, config: Optional[MetacognitiveConfig] = None) -> None:
        self.event_bus = event_bus or EventBus()
        self.config = config or MetacognitiveConfig()

    def update_for_segment(
        self,
        segment: DreamSegment,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
    ) -> None:
        lucidity = self._estimate_lucidity(segment, sleep_state, neuro_state)
        segment.lucidity_probability = lucidity

        event = Event(
            type=EventType.NEUROCHEMISTRY_UPDATED,
            payload={"segment_id": segment.id, "lucidity_probability": lucidity},
            timestamp_hours=segment.end_time_hours,
        )
        self.event_bus.publish(event)

    def _estimate_lucidity(
        self,
        segment: DreamSegment,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
    ) -> float:
        # Simple heuristic: higher lucidity in REM, slightly modulated by stage and arbitrary neurochemistry.
        if sleep_state.stage == SleepStage.REM:
            base = self.config.base_lucidity_rem
        else:
            base = self.config.base_lucidity_nrem

        # Placeholder modulation by ACh level.
        ach_factor = min(1.0, max(0.0, neuro_state.ach))
        return max(0.0, min(1.0, base + 0.1 * (ach_factor - 0.5)))
