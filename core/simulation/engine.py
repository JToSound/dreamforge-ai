from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import random

from core.models.sleep_cycle import SleepCycleModel, SleepStage
from core.models.neurochemistry import NeurochemistryModel
from core.models.memory_graph import MemoryGraph, ReplaySequence
from core.agents.metacognitive_agent import MetacognitiveAgent

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    duration_hours: float = 8.0
    dt_minutes: float = 0.5
    sleep_start_clock_time: float = 23.0
    stress_level: float = 0.5
    prior_day_events: list[str] = field(default_factory=list)
    llm_every_n_segments: int = 12  # call LLM every N segments to control cost/latency
    # Memory activation recording (for diagnostics / heatmaps)
    record_memory_activations: bool = True
    memory_activation_every_n_steps: int = 12
    memory_activation_top_n: int = 50
    # Bizarreness computation parameters (alpha * ACh + beta * replay_strength + noise)
    biz_alpha: float = 0.6
    biz_beta: float = 0.3
    biz_noise_std: float = 0.05


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

        # Precompute the full-night sleep schedule so we can query stage by time
        states, stages = self.sleep_model.simulate_night(
            duration_hours=config.duration_hours,
            dt_minutes=config.dt_minutes,
            sleep_start_clock_time=config.sleep_start_clock_time,
        )

        total_steps = max(1, len(states) - 1)

        # Initialise neuro state
        neuro_state = self.neuro_model.initial_state()

        hypnogram: list[dict] = []
        neuro_series: list[dict] = []
        dream_segments: list[Any] = []
        memory_activation_series: list[dict] = []
        segment_index = 0

        # Stage lookup based on precomputed states
        def _stage_fn(t_hours: float) -> SleepStage:
            idx = int(min(max(0, t_hours / dt_h), len(states) - 1))
            return states[int(idx)].stage

        # ── Main loop iterating precomputed states ─────────────────────────
        # Instantiate metacognitive agent to compute lucidity probabilities
        metacog = MetacognitiveAgent()

        for step in range(1, len(states)):
            progress = (step - 1) / total_steps
            sleep_state = states[step]

            # 2. Advance neurochemistry (integrate up to this time, using stage_fn that maps t->stage)
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
            hypnogram.append(
                {
                    "time_hours": round(t_now, 4),
                    "stage": sleep_state.stage.value,
                    "process_s": round(sleep_state.process_s, 4),
                    "process_c": round(sleep_state.process_c, 4),
                }
            )
            neuro_series.append(
                {
                    "time_hours": round(t_now, 4),
                    "ach": round(neuro_state.ach, 4),
                    "serotonin": round(neuro_state.serotonin, 4),
                    "ne": round(neuro_state.ne, 4),
                    "cortisol": round(neuro_state.cortisol, 4),
                }
            )

            # 4. Memory decay every ~6 minutes (12 ticks of 30s if dt_minutes=0.5)
            if step % 12 == 0:
                self.memory_graph.decay_salience(dt_hours=dt_h * 12)

            # 5. Generate dream segment at LLM cadence (REM / N2 preferred)
            is_dream_stage = sleep_state.stage in (SleepStage.REM, SleepStage.N2)
            if step % config.llm_every_n_segments == 0 and is_dream_stage:
                replay: Optional[ReplaySequence] = (
                    self.memory_graph.sample_replay_sequence(
                        max_length=10,
                        start_bias_tags=config.prior_day_events[:3],
                        current_time_hours=t_now,
                    )
                )

                # Apply immediate activation boost to replayed nodes so activation dynamics are visible
                if replay:
                    try:
                        # memory_graph may implement apply_replay_effect; fall back safely
                        if hasattr(self.memory_graph, "apply_replay_effect"):
                            self.memory_graph.apply_replay_effect(replay)
                        else:
                            # increase activation heuristically
                            for nid in replay.node_ids:
                                if nid in self.memory_graph.to_networkx().nodes:
                                    cur = float(
                                        self.memory_graph.to_networkx()
                                        .nodes[nid]
                                        .get("activation", 0.5)
                                    )
                                    self.memory_graph.to_networkx().nodes[nid][
                                        "activation"
                                    ] = min(1.0, cur + 0.25)
                    except Exception:
                        pass

                try:
                    seg = self.dream_constructor.generate_segment(
                        segment_index=segment_index,
                        sleep_state=sleep_state,
                        neuro_state=neuro_state,
                        replay=replay,
                        stress_level=config.stress_level,
                        prior_events=config.prior_day_events,
                        prev_segments=dream_segments,
                    )
                    # Recompute lucidity using metacognitive multi-factor model
                    try:
                        metacog.update_for_segment(seg, sleep_state, neuro_state)
                    except Exception:
                        pass
                    # Attach memory IDs from replay
                    if replay:
                        seg.active_memory_ids = replay.node_ids[:5]
                    dream_segments.append(seg)
                    segment_index += 1

                    # Recompute/adjust bizarreness using deterministic formula so
                    # it's consistent across LLM vs fallback generators.
                    try:
                        ach_val = float(neuro_state.ach)
                        replay_strength = 0.0
                        if replay and hasattr(replay, "total_weight"):
                            try:
                                nodes_count = max(
                                    1, self.memory_graph.to_networkx().number_of_nodes()
                                )
                                replay_strength = float(replay.total_weight) / float(
                                    nodes_count
                                )
                                replay_strength = min(1.0, replay_strength)
                            except Exception:
                                replay_strength = min(1.0, float(replay.total_weight))

                        biz = (
                            config.biz_alpha * ach_val
                            + config.biz_beta * replay_strength
                            + random.gauss(0.0, config.biz_noise_std)
                        )
                        biz = max(0.0, min(1.0, biz))
                        # override segment value (keep as float with 3-decimal precision)
                        try:
                            seg.bizarreness_score = round(float(biz), 3)
                        except Exception:
                            # If seg is a dict-like object, set key instead
                            try:
                                seg["bizarreness_score"] = round(float(biz), 3)
                            except Exception:
                                pass
                    except Exception:
                        pass

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
                        on_progress(
                            progress,
                            sleep_state.stage.value,
                            f"Segment error: {exc}",
                            None,
                        )
            elif step % (config.llm_every_n_segments * 4) == 0 and on_progress:
                on_progress(
                    progress,
                    sleep_state.stage.value,
                    f"[{t_now:.2f}h] Simulating {sleep_state.stage.value}…",
                    None,
                )

            # Record memory activations on REM ticks after replay pulse + decay.
            if config.record_memory_activations and sleep_state.stage == SleepStage.REM:
                self.memory_graph.decay_activations(dt_hours=dt_h)
                snapshot = self.memory_graph.capture_memory_snapshot(
                    time_hours=t_now,
                    stage=sleep_state.stage.value,
                )
                graph_view = self.memory_graph.to_networkx()
                activations = [
                    {
                        "id": node_id,
                        "label": graph_view.nodes[node_id].get("label", node_id),
                        "activation": value,
                    }
                    for node_id, value in snapshot.activations.items()
                ]
                memory_activation_series.append(
                    {
                        "time_hours": round(snapshot.time_hours, 4),
                        "stage": snapshot.stage,
                        "activations": activations,
                    }
                )

        # ── Build summary ─────────────────────────────────────────────────
        mean_biz = (
            float(np.mean([s.bizarreness_score for s in dream_segments]))
            if dream_segments
            else 0.0
        )
        from collections import Counter

        emo_counts = Counter(s.dominant_emotion for s in dream_segments)
        dominant_emotion = emo_counts.most_common(1)[0][0] if emo_counts else "neutral"

        summary = (
            f"Night simulation complete: {len(dream_segments)} dream segments across "
            f"{config.duration_hours:.1f} hours. Mean bizarreness: {mean_biz:.2f}. "
            f"Dominant emotion: {dominant_emotion}."
        )
        llm_calls_total = int(getattr(self.dream_constructor, "llm_calls_total", 0))

        dream_segments_payload = [s.model_dump() for s in dream_segments]
        return {
            "duration_hours": config.duration_hours,
            "total_segments": len(dream_segments),
            "hypnogram": hypnogram,
            "neurochemistry_series": neuro_series,
            "neurochemistry_ticks": neuro_series,
            "dream_segments": dream_segments_payload,
            "segments": dream_segments_payload,
            "memory_graph": self.memory_graph.to_json_serializable(),
            "memory_activation_series": memory_activation_series,
            "memory_activations": memory_activation_series,
            "summary_narrative": summary,
            "mean_bizarreness": round(mean_biz, 3),
            "dominant_emotion": dominant_emotion,
            "llm_calls_total": llm_calls_total,
        }
