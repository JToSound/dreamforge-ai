from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.models.sleep_cycle import SleepCycleModel, SleepState
from core.simulation.event_bus import EventBus, Event, EventType


@dataclass
class SleepCycleConfig:
    sleep_start_clock_time: float = 23.0
    dt_minutes: float = 0.5


class SleepCycleAgent:
    """Agent responsible for updating the sleep stage over simulated time."""

    def __init__(
        self,
        model: Optional[SleepCycleModel] = None,
        event_bus: Optional[EventBus] = None,
        config: Optional[SleepCycleConfig] = None,
    ) -> None:
        self.model = model or SleepCycleModel()
        self.event_bus = event_bus or EventBus()
        self.config = config or SleepCycleConfig()
        self.state: SleepState = self.model.initial_state(
            sleep_start_clock_time=self.config.sleep_start_clock_time
        )

    def step(self) -> SleepState:
        dt_hours = self.config.dt_minutes / 60.0
        self.state = self.model.step(
            self.state,
            dt_hours=dt_hours,
            sleep_start_clock_time=self.config.sleep_start_clock_time,
        )
        self._emit_event()
        return self.state

    def _emit_event(self) -> None:
        event = Event(
            type=EventType.SLEEP_STAGE_UPDATED,
            payload={
                "stage": self.state.stage.value,
                "process_s": self.state.process_s,
                "process_c": self.state.process_c,
                "cycle_index": self.state.cycle_index,
            },
            timestamp_hours=self.state.time_hours,
        )
        self.event_bus.publish(event)
