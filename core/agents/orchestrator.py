from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.agents.sleep_cycle_agent import SleepCycleAgent
from core.agents.neurochemistry_agent import NeurochemistryAgent
from core.agents.memory_consolidation_agent import MemoryConsolidationAgent
from core.simulation.event_bus import EventBus, AgentActivityLogger
from core.simulation.time_model import TimeModel
from core.agents.dream_constructor_agent import DreamConstructorAgent
from core.agents.metacognitive_agent import MetacognitiveAgent
from core.agents.phenomenology_reporter import PhenomenologyReporter
from core.models.dream_segment import DreamSegment
from core.utils.pharmacology import PharmacologyProfile, apply_pharmacology
from core.models.neurochemistry import NeurochemistryModel, NeurochemistryParameters


@dataclass
class OrchestratorConfig:
    night_duration_hours: float = 8.0
    dt_minutes: float = 0.5
    pharmacology: Optional[PharmacologyProfile] = None


class OrchestratorAgent:
    """Top-level coordinator for a single-night dream simulation."""

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()
        self.event_bus = EventBus()
        self.activity_logger = AgentActivityLogger(self.event_bus)

        # Apply pharmacology to neurochemistry parameters if configured.
        neuro_params = NeurochemistryParameters()
        if self.config.pharmacology is not None:
            neuro_params = apply_pharmacology(neuro_params, self.config.pharmacology)

        self.sleep_agent = SleepCycleAgent(event_bus=self.event_bus)
        self.neuro_agent = NeurochemistryAgent(
            model=NeurochemistryModel(params=neuro_params),
            event_bus=self.event_bus,
        )
        self.memory_agent = MemoryConsolidationAgent(event_bus=self.event_bus)
        self.dream_agent = DreamConstructorAgent(event_bus=self.event_bus)
        self.meta_agent = MetacognitiveAgent(event_bus=self.event_bus)
        self.phenom_agent = PhenomenologyReporter(event_bus=self.event_bus)

        self.neuro_agent.set_stage_fn(lambda t: self.sleep_agent.state.stage)

        self.time_model = TimeModel(
            start_time_hours=0.0,
            dt_minutes=self.config.dt_minutes,
            duration_hours=self.config.night_duration_hours,
        )

        self.sleep_history: List = []
        self.neuro_history: List = []

    def run_night(self) -> List[DreamSegment]:
        segments: List[DreamSegment] = []

        while self.time_model.has_next_step:
            sleep_state = self.sleep_agent.step()
            self.sleep_history.append(sleep_state)

            neuro_state = self.neuro_agent.step_to(sleep_state.time_hours)
            self.neuro_history.append(neuro_state)

            replay_seq = self.memory_agent.maybe_replay(current_time_hours=sleep_state.time_hours)
            self.memory_agent.decay_and_prune(dt_hours=self.time_model.dt_hours)

            segment = self.dream_agent.step(
                sleep_state=sleep_state,
                neuro_state=neuro_state,
                replay_seq=replay_seq,
            )

            if segment is not None:
                self.meta_agent.update_for_segment(segment, sleep_state, neuro_state)
                self.phenom_agent.record_segment(segment)
                segments.append(segment)

            self.time_model.step()

        return segments
