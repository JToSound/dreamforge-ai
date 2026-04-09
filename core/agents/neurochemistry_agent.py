from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from core.models.neurochemistry import NeurochemistryModel, NeurochemistryState
from core.models.sleep_cycle import SleepStage
from core.simulation.event_bus import EventBus, Event, EventType


@dataclass
class NeurochemistryConfig:
    max_step_hours: float = 1.0 / 60.0


class NeurochemistryAgent:
    """Agent that simulates neuromodulator dynamics and publishes updates."""

    def __init__(
        self,
        model: Optional[NeurochemistryModel] = None,
        event_bus: Optional[EventBus] = None,
        config: Optional[NeurochemistryConfig] = None,
    ) -> None:
        self.model = model or NeurochemistryModel()
        self.event_bus = event_bus or EventBus()
        self.config = config or NeurochemistryConfig()
        self.state: NeurochemistryState = self.model.initial_state()

        # Sleep stage as a function of time, to be injected by SleepCycleAgent or orchestrator.
        self.stage_fn: Callable[[float], SleepStage] = lambda t: SleepStage.N2

    def set_stage_fn(self, fn: Callable[[float], SleepStage]) -> None:
        self.stage_fn = fn

    def step_to(self, t_end_hours: float) -> NeurochemistryState:
        trajectory, t_samples = self.model.integrate(
            state=self.state,
            t_end=t_end_hours,
            stage_fn=self.stage_fn,
            max_step=self.config.max_step_hours,
        )
        self.state = trajectory[-1]
        self._emit_event()
        return self.state

    def _emit_event(self) -> None:
        event = Event(
            type=EventType.NEUROCHEMISTRY_UPDATED,
            payload={
                "ach": self.state.ach,
                "serotonin": self.state.serotonin,
                "ne": self.state.ne,
                "cortisol": self.state.cortisol,
            },
            timestamp_hours=self.state.time_hours,
        )
        self.event_bus.publish(event)
