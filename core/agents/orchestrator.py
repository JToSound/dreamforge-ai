from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Optional

from core.agents.dream_constructor_agent import DreamConstructorAgent, DreamSegment
from core.models.memory_graph import (
    MemoryGraph,
    MemoryNodeModel,
    MemoryEdgeModel,
    MemoryType,
    EmotionLabel,
)
from core.models.neurochemistry import NeurochemistryModel, NeurochemistryParameters
from core.models.sleep_cycle import SleepCycleModel, TwoProcessParameters
from core.simulation.engine import SimulationEngine, SimulationConfig

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str, str, Optional[DreamSegment]], None]


class OrchestratorAgent:
    """Top-level orchestrator that wires all agents together and drives a full-night simulation.

    Args:
        llm_config: dict with keys provider/model/api_key/base_url/temperature/max_tokens.
        sleep_config: dict of TwoProcessParameters overrides.
        neuro_config: dict of NeurochemistryParameters overrides.
        memory_config: dict with keys max_nodes/decay_rate/prune_threshold/replay_max_length.
        on_progress: optional callback(progress, stage, message, segment).
    """

    def __init__(
        self,
        llm_config: Optional[dict] = None,
        sleep_config: Optional[dict] = None,
        neuro_config: Optional[dict] = None,
        memory_config: Optional[dict] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.simulation_id = str(uuid.uuid4())
        self.on_progress = on_progress

        # Sleep model
        sleep_params = TwoProcessParameters(**(sleep_config or {}))
        self.sleep_model = SleepCycleModel(params=sleep_params)

        # Neurochemistry model
        neuro_params = NeurochemistryParameters(**(neuro_config or {}))
        self.neuro_model = NeurochemistryModel(params=neuro_params)

        # Memory graph
        self.memory_graph = MemoryGraph()
        self._memory_config = memory_config or {}

        # Dream constructor
        self.dream_constructor = DreamConstructorAgent(llm_config=llm_config)

        # Simulation engine
        self.engine = SimulationEngine(
            sleep_model=self.sleep_model,
            neuro_model=self.neuro_model,
            memory_graph=self.memory_graph,
            dream_constructor=self.dream_constructor,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _emit_progress(
        self,
        progress: float,
        stage: str,
        message: str,
        segment: Optional[DreamSegment] = None,
    ) -> None:
        if self.on_progress:
            try:
                self.on_progress(progress, stage, message, segment)
            except Exception as exc:
                logger.warning("Progress callback raised: %s", exc)

    def seed_memory_from_events(
        self,
        prior_day_events: list[str],
        stress_level: float = 0.5,
    ) -> None:
        """Encode prior-day events into the memory graph before simulation."""
        from core.models.memory_graph import EmotionLabel
        import random

        emotion_map = {
            (0.0, 0.3): EmotionLabel.JOY,
            (0.3, 0.6): EmotionLabel.NEUTRAL,
            (0.6, 0.8): EmotionLabel.SADNESS,
            (0.8, 1.0): EmotionLabel.FEAR,
        }

        def pick_emotion(stress: float) -> EmotionLabel:
            for (lo, hi), emo in emotion_map.items():
                if lo <= stress < hi:
                    return emo
            return EmotionLabel.NEUTRAL

        node_ids: list[str] = []
        for i, event in enumerate(prior_day_events):
            arousal = min(1.0, stress_level + random.uniform(-0.1, 0.1))
            node = MemoryNodeModel(
                label=event[:128],
                memory_type=MemoryType.EPISODIC,
                activation=0.8 - i * 0.05,
                salience=0.9 - i * 0.05,
                emotion=pick_emotion(stress_level),
                arousal=arousal,
                recency_hours=float(i),
            )
            nid = self.memory_graph.add_memory(node)
            node_ids.append(nid)

        # Add associations between adjacent events
        for i in range(len(node_ids) - 1):
            edge = MemoryEdgeModel(
                source_id=node_ids[i],
                target_id=node_ids[i + 1],
                weight=0.6,
                emotion_alignment=0.5,
                context_overlap=0.4,
            )
            try:
                self.memory_graph.add_association(edge)
            except ValueError:
                pass

    def run_night(
        self,
        duration_hours: float = 8.0,
        sleep_start_clock_time: float = 23.0,
        dt_minutes: float = 0.5,
        prior_day_events: Optional[list[str]] = None,
        stress_level: float = 0.5,
        llm_every_n_segments: int = 12,
    ) -> dict[str, Any]:
        """Run a full-night simulation and return structured results."""
        self._emit_progress(0.0, "init", "Seeding memory from prior-day events…")
        self.seed_memory_from_events(prior_day_events or [], stress_level=stress_level)

        config = SimulationConfig(
            duration_hours=duration_hours,
            dt_minutes=dt_minutes,
            sleep_start_clock_time=sleep_start_clock_time,
            stress_level=stress_level,
            prior_day_events=prior_day_events or [],
            llm_every_n_segments=llm_every_n_segments,
        )

        def _internal_progress(
            progress: float, stage: str, message: str, segment: Optional[DreamSegment]
        ):
            self._emit_progress(progress, stage, message, segment)

        result = self.engine.run(
            config=config,
            on_progress=_internal_progress,
        )

        self._emit_progress(1.0, "complete", "Simulation complete.")
        return {
            "simulation_id": self.simulation_id,
            **result,
        }
