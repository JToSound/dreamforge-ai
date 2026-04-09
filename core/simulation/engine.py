from __future__ import annotations

from typing import List, Optional

from core.agents.orchestrator import OrchestratorAgent, OrchestratorConfig
from core.models.dream_segment import DreamSegment, DreamNight
from core.utils.llm_adapters import populate_memory_from_journal
from core.utils.night_summary import compute_night_summary


class SimulationEngine:
    """High-level entry point for running simulations from the API or notebooks."""

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.orchestrator = OrchestratorAgent(config=config)
        # Seed memory graph from any stored journal entries
        populate_memory_from_journal(self.orchestrator.memory_agent.graph)

    def simulate_night(self) -> List[DreamSegment]:
        return self.orchestrator.run_night()

    def build_night(self) -> DreamNight:
        night = self.orchestrator.phenom_agent.build_night()
        summary = compute_night_summary(
            sleep_history=self.orchestrator.sleep_history,
            neuro_history=self.orchestrator.neuro_history,
            segments=night.segments,
            memory_graph=self.orchestrator.memory_agent.graph,
        )
        night.metadata["summary"] = summary
        return night
