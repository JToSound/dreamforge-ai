from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from core.models.sleep_cycle import SleepCycleModel, SleepState, SleepStage
from core.models.neurochemistry import NeurochemistryModel, NeurochemistryState
from core.models.memory_graph import MemoryGraph, ReplaySequence

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    duration_hours: float = 8.0
    dt_minutes: float = 0.5
    sleep_start_clock_time: float = 23.0
    stress_level: float = 0.5
    prior_day_events: list[str] = field(default_factory=list)
    llm_every_n_segments: int = 12   # call LLM every N segments to control cost/latency


ProgressCallback = Callable[
    [float, str, str, Optional[Any]],  # (progress, stage, message, segment|None)
    None,
]


class SimulationEngine:
    """Drives the full-night simulation loop.

    Integrates SleepCycleModel, NeurochemistryModel, MemoryGraph, and
    DreamConstructorAgent across simulated time, emitting progress events.
    """

    def __init__(
        self,
        sleep_model: SleepCycleModel,
        neuro_model: NeurochemistryModel,
        memory_graph: MemoryGraph,
        dream_constructor: Any,  # DreamConstructorAgent
    ) -> None:
        self.sleep_model = sleep_model
        self.neuro_model = neuro_model
        self.memory_graph = memory_graph
        self.dream_constructor = dream_constructor

    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        config: SimulationConfig,
        on_progress: Optional[ProgressCallback] = None,
    ) -> dict[str, Any]:
        """Run one full-night simulation.

        Returns a dict compatible with SimulateNightResponse.
        """
        dt_h = config.dt_minutes / 60.0
        total_steps = int(config.duration_hours / dt_h)

        # ── Initialise state ──────────────────────────────────────────────
        sleep_state = self.sleep_model.initial_state(
            sleep_start_circadian_time=config.sleep_start_clock_time
        )
        neuro_state = self.neuro_model.initial_state()

        hypnogram: list[dict] = []
        neuro_series: list[dict] = []
        dream_segments: list[Any] = []
        segment_index = 0

        def _stage_fn(t_hours: float) -> SleepStage:
            # Thin wrapper so NeurochemistryModel can query stage by time.
            # We approximate with the current sleep_state stage.
            return sleep_state.stage

        # ── Main loop ────────────────────────────────────────────────────
        for step in range(total_steps):
            progress = step / total_steps

            # 1. Advance sleep model
            sleep_state = self.sleep_model.step(
                sleep_state,
                dt_hours=dt_h,
                sleep_start_clock_time=config.sleep_start_clock_time,
            )

            # 2. Advance neurochemistry (one ODE step)
            t_now = sleep_state.time_hours
            try:
                traj, _ = self.neuro_model.integrate(
                    state=neuro_state,
                    t_end=t_now,
                    stage_fn=_stage_fn,
                    max_step=dt_h,
                )
                if traj:
                    neuro_state = traj[-1]
            except Exception as exc:
                logger.debug("neuro integrate error at step %d: %s", step, exc)

            # 3. Record hypnogram & neuro snapshot
            hypnogram.append({
                "time_hours": round(t_now, 4),
                "stage": sleep_state.stage.value,
                "process_s": round(sleep_state.process_s, 4),
                "process_c": round(sleep_state.process_c, 4),
            })
            neuro_series.append({
                "time_hours": round(t_now, 4),
                "ach": round(neuro_state.ach, 4),
                "serotonin": round(neuro_state.serotonin, 4),
                "ne": round(neuro_state.ne, 4),
                "cortisol": round(neuro_state.cortisol, 4),
            })

            # 4. Memory decay every ~6 minutes
            if step % 12 == 0:
                self.memory_graph.decay_salience(dt_hours=dt_h * 12)

            # 5. Generate dream segment at LLM cadence (REM / N2 preferred)
            is_dream_stage = sleep_state.stage in (SleepStage.REM, SleepStage.N2)
            if step % config.llm_every_n_segments == 0 and is_dream_stage:
                replay: Optional[ReplaySequence] = self.memory_graph.sample_replay_sequence(
                    max_length=10,
                    start_bias_tags=config.prior_day_events[:3],
                )
                try:
                    seg = self.dream_constructor.generate_segment(
                        segment_index=segment_index,
                        sleep_state=sleep_state,
                        neuro_state=neuro_state,
                        replay=replay,
                        stress_level=config.stress_level,
                        prior_events=config.prior_day_events,
                    )
                    # Attach memory IDs from replay
                    if replay:
                        seg.active_memory_ids = replay.node_ids[:5]
                    dream_segments.append(seg)
                    segment_index += 1

                    if on_progress:
                        on_progress(
                            progress,
                            sleep_state.stage.value,
                            f"[{t_now:.2f}h] Dream segment #{segment_index} generated.",
                            seg,
                        )
                except Exception as exc:
                    logger.error("Dream generation failed at step %d: %s", step, exc)
                    if on_progress:
                        on_progress(progress, sleep_state.stage.value, f"Segment error: {exc}", None)
            elif step % (config.llm_every_n_segments * 4) == 0 and on_progress:
                on_progress(
                    progress,
                    sleep_state.stage.value,
                    f"[{t_now:.2f}h] Simulating {sleep_state.stage.value}…",
                    None,
                )

        # ── Build summary ─────────────────────────────────────────────────
        mean_biz = float(np.mean([s.bizarreness_score for s in dream_segments])) if dream_segments else 0.0
        from collections import Counter
        emo_counts = Counter(s.dominant_emotion for s in dream_segments)
        dominant_emotion = emo_counts.most_common(1)[0][0] if emo_counts else "neutral"

        summary = (
            f"Night simulation complete: {len(dream_segments)} dream segments across "
            f"{config.duration_hours:.1f} hours. Mean bizarreness: {mean_biz:.2f}. "
            f"Dominant emotion: {dominant_emotion}."
        )

        return {
            "duration_hours": config.duration_hours,
            "total_segments": len(dream_segments),
            "hypnogram": hypnogram,
            "neurochemistry_series": neuro_series,
            "dream_segments": [s.model_dump() for s in dream_segments],
            "memory_graph": self.memory_graph.to_json_serializable(),
            "summary_narrative": summary,
            "mean_bizarreness": round(mean_biz, 3),
            "dominant_emotion": dominant_emotion,
        }