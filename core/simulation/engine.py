from __future__ import annotations

from typing import Callable, List, Optional

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

    def simulate_night(
        self,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> List[DreamSegment]:
        """Run a full-night simulation, optionally reporting progress [0.0, 1.0]."""
        return self.orchestrator.run_night(on_progress=on_progress)

    def build_night(self) -> DreamNight:
        night = self.orchestrator.phenom_agent.build_night()
        summary = compute_night_summary(
            sleep_history=self.orchestrator.sleep_history,
            neuro_history=self.orchestrator.neuro_history,
            segments=night.segments,
            memory_graph=self.orchestrator.memory_agent.graph,
        )
        night.metadata["summary"] = summary

        # Embed raw timeseries + memory graph for API / frontend visualisations
        night.metadata["sleep_history"] = [
            {
                "time_hours": s.time_hours,
                "stage": s.stage.value,
                "process_s": s.process_s,
                "process_c": s.process_c,
            }
            for s in self.orchestrator.sleep_history
        ]

        night.metadata["neuro_history"] = [
            {
                "time_hours": n.time_hours,
                "ach": n.ach,
                "serotonin": n.serotonin,
                "ne": n.ne,
                "cortisol": n.cortisol,
            }
            for n in self.orchestrator.neuro_history
        ]

        night.metadata["memory_graph"] = (
            self.orchestrator.memory_agent.graph.to_json_serializable()
        )

        return night
