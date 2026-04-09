from __future__ import annotations

from typing import List, Optional

from core.agents.orchestrator import OrchestratorAgent, OrchestratorConfig
from core.models.dream_segment import DreamSegment


class SimulationEngine:
    """Thin wrapper around OrchestratorAgent used by the API layer."""

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.orchestrator = OrchestratorAgent(config=config)

    def simulate_night(self) -> List[DreamSegment]:
        return self.orchestrator.run_night()
