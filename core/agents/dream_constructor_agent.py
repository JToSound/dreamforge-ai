from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel

from core.models.sleep_cycle import SleepStage, SleepState
from core.models.neurochemistry import NeurochemistryState
from core.models.memory_graph import ReplaySequence, EmotionLabel

logger = logging.getLogger(__name__)


class DreamSegment(BaseModel):
    segment_index: int
    time_hours: float
    stage: str
    narrative: str
    dominant_emotion: str
    bizarreness_score: float
    lucidity_probability: float
    active_memory_ids: list[str]
    neurochemistry: dict[str, float]


class DreamConstructorAgent:
    """Generates dream narrative segments using a real LLM backend.

    Supports OpenAI, Anthropic, and Ollama providers via a unified interface.
    Falls back to a deterministic template when no API key is available.
    """

    def __init__(self, llm_config: Optional[dict] = None) -> None:
        self.llm_config = llm_config or {}
        self._client = None
        self._provider = self.llm_config.get("provider", "openai")
        self._model = self.llm_config.get("model", "gpt-4o")
        self._temperature = float(self.llm_config.get("temperature", 0.9))
        self._max_tokens = int(self.llm_config.get("max_tokens", 512))
        self._api_key = self.llm_config.get("api_key")
        self._base_url = self.llm_config.get("base_url")

    # ─────────────────────────────────────────────────────────────────────────
    # LLM client initialisation (lazy)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self._provider in ("openai", "ollama"):
            try:
                from openai import OpenAI
                kwargs: dict = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                elif self._provider == "ollama":
                    kwargs["base_url"] = "http://localhost:11434/v1"
                    kwargs.setdefault("api_key", "ollama")
                self._client = OpenAI(**kwargs)
            except Exception as e:
                logger.warning("openai client init failed; using fallback generator: %s", e)
        elif self._provider == "anthropic":
            try:
                import anthropic
                kwargs = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                self._client = anthropic.Anthropic(**kwargs)
            except Exception as e:
                logger.warning("anthropic client init failed; using fallback generator: %s", e)

        return self._client

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
        replay: Optional[ReplaySequence],
        stress_level: float,
        prior_events: list[str],
        segment_index: int,
        prev_segments: Optional[list] = None,
    ) -> str:
        ach = round(neuro_state.ach, 3)
        five_ht = round(neuro_state.serotonin, 3)
        ne = round(neuro_state.ne, 3)
        cortisol = round(neuro_state.cortisol, 3)
        stage = sleep_state.stage.value
        replay_summary = ""
        if replay and replay.node_ids:
            replay_summary = (
                f"Active memory replay (dominant emotion: {replay.dominant_emotion.value}): "
                f"{len(replay.node_ids)} fragments firing with total salience "
                f"{replay.total_weight:.2f}."
            )
        events_str = "; ".join(prior_events) if prior_events else "none recorded"

        # Include short context from previous segments to improve coherence
        prev_text = ""
        if prev_segments:
            ctx = []
            for s in prev_segments[-3:]:
                try:
                    ctx.append(str(getattr(s, "narrative", s.get("narrative", ""))))
                except Exception:
                    try:
                        ctx.append(str(s.narrative))
                    except Exception:
                        pass
            if ctx:
                prev_text = "Previous segments context:\n" + "\n".join(f"- {c}" for c in ctx)

        system_msg = (
            "You are DreamForge AI's Dream Constructor. Your job is to generate a "
            "short, vivid, first-person dream narrative segment grounded in the "
            "provided neurobiological state. Be surreal, emotionally resonant, and "
            "scientifically consistent with the current neurotransmitter levels.\n"
            "Respond ONLY with a JSON object with keys:\n"
            "  narrative (str), dominant_emotion (str: neutral/joy/fear/sadness/anger/surprise/disgust),\n"
            "  bizarreness_score (float 0-1), lucidity_probability (float 0-1)."
        )

        user_msg = (
            f"Segment #{segment_index} | Sleep stage: {stage} | "
            f"Time into night: {sleep_state.time_hours:.2f}h\n"
            f"Neurochemistry: ACh={ach}, 5-HT={five_ht}, NE={ne}, Cortisol={cortisol}\n"
            f"Stress level: {stress_level:.2f}\n"
            f"Prior day events: {events_str}\n"
            f"{replay_summary}\n"
            f"{prev_text}\n"
            "Generate the dream segment JSON now."
        )
        return system_msg, user_msg

    # ─────────────────────────────────────────────────────────────────────────
    # LLM call
    # ─────────────────────────────────────────────────────────────────────────

    def _call_llm(self, system_msg: str, user_msg: str) -> dict:
        client = self._get_client()
        if client is None:
            return self._fallback_response()

        try:
            if self._provider in ("openai", "ollama"):
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or "{}"
                return json.loads(raw)

            elif self._provider == "anthropic":
                resp = client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system_msg,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text if resp.content else "{}"
                # Anthropic may not guarantee JSON; extract best-effort
                try:
                    start = raw.index("{")
                    end = raw.rindex("}") + 1
                    return json.loads(raw[start:end])
                except (ValueError, json.JSONDecodeError):
                    return self._fallback_response()

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return self._fallback_response()

        return self._fallback_response()

    @staticmethod
    def _fallback_response() -> dict:
        import random
        narratives = [
            "I find myself in a corridor that keeps extending, doors multiplying as I walk.",
            "There is a lake made of light. My hands pass through it, leaving ripples in the air.",
            "A figure I recognise but cannot name hands me something I immediately forget.",
            "The city rearranges itself each time I look away. Streets fold like paper.",
            "I am trying to speak but the words arrive before I think of them.",
        ]
        return {
            "narrative": random.choice(narratives),
            "dominant_emotion": random.choice(["neutral", "surprise", "fear", "joy"]),
            "bizarreness_score": round(random.uniform(0.3, 0.9), 2),
            "lucidity_probability": round(random.uniform(0.0, 0.25), 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def generate_segment(
        self,
        segment_index: int,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
        replay: Optional[ReplaySequence],
        stress_level: float = 0.5,
        prior_events: Optional[list[str]] = None,
        prev_segments: Optional[list] = None,
    ) -> DreamSegment:
        """Generate one dream segment for the given simulation state."""
        system_msg, user_msg = self._build_prompt(
            sleep_state=sleep_state,
            neuro_state=neuro_state,
            replay=replay,
            stress_level=stress_level,
            prior_events=prior_events or [],
            segment_index=segment_index,
            prev_segments=prev_segments,
        )
        data = self._call_llm(system_msg, user_msg)

        return DreamSegment(
            segment_index=segment_index,
            time_hours=sleep_state.time_hours,
            stage=sleep_state.stage.value,
            narrative=str(data.get("narrative", "")),
            dominant_emotion=str(data.get("dominant_emotion", "neutral")),
            bizarreness_score=float(data.get("bizarreness_score", 0.5)),
            lucidity_probability=float(data.get("lucidity_probability", 0.0)),
            active_memory_ids=[],
            neurochemistry={
                "ach": neuro_state.ach,
                "serotonin": neuro_state.serotonin,
                "ne": neuro_state.ne,
                "cortisol": neuro_state.cortisol,
            },
        )