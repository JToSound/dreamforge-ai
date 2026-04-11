from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from core.agents.sleep_cycle_agent import SleepCycleAgent, SleepCycleConfig
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
from core.utils.llm_adapters import create_llm_callable


@dataclass
class OrchestratorConfig:
    night_duration_hours: float = 8.0
    dt_minutes: float = 0.5
    pharmacology: Optional[PharmacologyProfile] = None

    # LLM configuration
    llm_enabled: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_important_only: bool = True
    llm_api_key: Optional[str] = None


class OrchestratorAgent:
    """Top-level coordinator for a single-night dream simulation."""

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()
        self.event_bus = EventBus()
        self.activity_logger = AgentActivityLogger(self.event_bus)

        neuro_params = NeurochemistryParameters()
        if self.config.pharmacology is not None:
            neuro_params = apply_pharmacology(neuro_params, self.config.pharmacology)

        sleep_cfg = SleepCycleConfig(dt_minutes=self.config.dt_minutes)
        self.sleep_agent = SleepCycleAgent(event_bus=self.event_bus, config=sleep_cfg)
        self.neuro_agent = NeurochemistryAgent(
            model=NeurochemistryModel(params=neuro_params),
            event_bus=self.event_bus,
        )
        self.memory_agent = MemoryConsolidationAgent(event_bus=self.event_bus)

        llm_callable = None
        if self.config.llm_enabled:
            llm_callable = create_llm_callable(
                self.config.llm_provider,
                self.config.llm_model,
                api_key=self.config.llm_api_key,
            )

        self.dream_agent = DreamConstructorAgent(
            event_bus=self.event_bus,
            llm=llm_callable,
            important_only=self.config.llm_important_only,
        )
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

    def run_night(
        self,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> List[DreamSegment]:
        """Run a full-night simulation.

        Args:
            on_progress: Optional callback receiving simulation progress in [0.0, 1.0].
                         Called after every simulation tick so callers can update progress bars.
        """
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

            # Report progress as fraction of simulated night completed
            if on_progress is not None:
                elapsed = self.time_model.current_time_hours - self.time_model.start_time_hours
                duration = self.time_model.duration_hours
                progress = elapsed / duration if duration > 0 else 1.0
                on_progress(max(0.0, min(progress, 1.0)))

            self.time_model.step()

        # Ensure final 100 % callback
        if on_progress is not None:
            on_progress(1.0)

        return segments
