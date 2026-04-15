from __future__ import annotations

import json
import logging
import re
import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from core.config import load_runtime_config
from core.models.memory_graph import ReplaySequence
from core.models.neurochemistry import NeurochemistryState
from core.models.sleep_cycle import SleepState
from core.simulation.llm_trigger import LLMTriggerDetector, LLMTriggerType
from core.simulation.narrative_cache import NarrativeCache

logger = logging.getLogger(__name__)
_RUNTIME_CONFIG = load_runtime_config()


def strip_thinking_tags(response: str) -> str:
    """Remove <think>...</think> blocks used by some reasoning models."""
    if not response:
        return ""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


class GenerationMode(str, Enum):
    LLM = "LLM"
    TEMPLATE = "TEMPLATE"
    LLM_FALLBACK = "LLM_FALLBACK"
    CACHED = "CACHED"


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
    generation_mode: GenerationMode = GenerationMode.TEMPLATE
    llm_trigger_type: Optional[str] = None
    llm_latency_ms: Optional[float] = None
    template_bank: Optional[str] = None
    llm_error: Optional[str] = None


class DreamConstructorAgent:
    """Generates dream narrative segments with trigger-based LLM calls.

    LLM is only invoked when the trigger detector emits an event. Otherwise,
    segment text is derived from cached narrative context and enhanced templates.
    """

    def __init__(self, llm_config: Optional[dict] = None) -> None:
        self.llm_config = llm_config or {}
        self._client = None
        self._provider = self.llm_config.get("provider", "openai")
        self._model = self.llm_config.get("model", "gpt-4o")
        self._temperature = float(self.llm_config.get("temperature", 0.9))
        # Source: Qwen3.5 docs (reasoning-token budget requires >=2048 output tokens)
        self._max_tokens = int(
            self.llm_config.get("max_tokens", _RUNTIME_CONFIG.llm_max_tokens)
        )
        self._api_key = self.llm_config.get("api_key")
        self._base_url = self.llm_config.get("base_url")

        self.trigger_detector = LLMTriggerDetector()
        self.narrative_cache = NarrativeCache()
        self.llm_calls_total = 0

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self._provider in ("openai", "ollama"):
            try:
                from openai import OpenAI

                kwargs: dict[str, Any] = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                elif self._provider == "ollama":
                    kwargs["base_url"] = "http://localhost:11434/v1"
                    kwargs.setdefault("api_key", "ollama")
                self._client = OpenAI(**kwargs)
            except Exception as exc:
                logger.warning(
                    "openai client init failed; template mode enabled: %s", exc
                )
        elif self._provider == "anthropic":
            try:
                import anthropic

                kwargs: dict[str, Any] = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                self._client = anthropic.Anthropic(**kwargs)
            except Exception as exc:
                logger.warning(
                    "anthropic client init failed; template mode enabled: %s", exc
                )

        return self._client

    def _build_prompt(
        self,
        sleep_state: SleepState,
        neuro_state: NeurochemistryState,
        replay: Optional[ReplaySequence],
        stress_level: float,
        prior_events: list[str],
        segment_index: int,
        trigger_type: LLMTriggerType,
        bizarreness_score: float,
        lucidity_probability: float,
        prev_segments: Optional[list] = None,
    ) -> tuple[str, str]:
        ach = round(float(neuro_state.ach), 3)
        five_ht = round(float(neuro_state.serotonin), 3)
        ne = round(float(neuro_state.ne), 3)
        cortisol = round(float(neuro_state.cortisol), 3)
        stage = sleep_state.stage.value
        replay_summary = ""
        if replay and replay.node_ids:
            replay_summary = (
                f"Replay sequence emotion={replay.dominant_emotion.value}, "
                f"nodes={len(replay.node_ids)}, total_weight={replay.total_weight:.2f}."
            )
        events_str = "; ".join(prior_events) if prior_events else "none recorded"

        prev_text = ""
        if prev_segments:
            ctx = []
            for seg in prev_segments[-3:]:
                try:
                    ctx.append(str(getattr(seg, "narrative", seg.get("narrative", ""))))
                except Exception:
                    pass
            if ctx:
                prev_text = "Previous context:\n" + "\n".join(f"- {c}" for c in ctx)

        ach_desc = (
            "hyper-vivid and hallucinatory"
            if ach > 0.80
            else "moderately vivid and coherent" if ach > 0.55 else "hazy and diffuse"
        )
        ne_desc = (
            "high threat salience and urgency"
            if ne > 0.40
            else (
                "reduced reality resistance; absurd events feel normal"
                if ne < 0.15
                else "mild alertness"
            )
        )
        cortisol_desc = (
            "anxious, fragmented tone"
            if cortisol > 0.75
            else (
                "stable but slightly tense tone"
                if cortisol > 0.25
                else "calm, generative tone"
            )
        )
        serotonin_desc = (
            "reality monitoring is strongly reduced"
            if five_ht < 0.10
            else "partial reality monitoring remains"
        )

        system_msg = (
            "You are DreamForge AI's Dream Constructor. "
            "Output ONLY valid JSON with keys: narrative, dominant_emotion, "
            "bizarreness_score, lucidity_probability."
        )
        user_msg = (
            f"Trigger={trigger_type.value}\n"
            f"Segment #{segment_index} at t={sleep_state.time_hours:.2f}h\n"
            f"Stage={stage}\n"
            f"ACh={ach} 5-HT={five_ht} NE={ne} Cortisol={cortisol}\n"
            f"Bizarreness={bizarreness_score:.2f} Lucidity={lucidity_probability:.2f}\n"
            "[NEUROCHEMICAL CONTEXT]\n"
            f"- ACh texture: {ach_desc}\n"
            f"- NE tone: {ne_desc}\n"
            f"- Cortisol affect: {cortisol_desc}\n"
            f"- Serotonin coherence: {serotonin_desc}\n"
            f"Stress={stress_level:.2f}\n"
            f"Events={events_str}\n"
            f"{replay_summary}\n"
            f"{prev_text}\n"
            f"Memory nodes={(replay.node_ids[:6] if replay else [])}\n"
            "Return JSON now."
        )
        user_msg = f"/no_think\n\n{user_msg}"
        return system_msg, user_msg

    @staticmethod
    def _fallback_payload(stage: str, emotion: str, narrative: str) -> dict:
        return {
            "narrative": narrative,
            "dominant_emotion": emotion,
            "bizarreness_score": 0.5 if stage == "REM" else 0.2,
            "lucidity_probability": 0.1 if stage == "REM" else 0.02,
        }

    def _call_llm(
        self, system_msg: str, user_msg: str
    ) -> tuple[dict[str, Any], Optional[str], bool, Optional[float]]:
        client = self._get_client()
        if client is None:
            return {}, "client_unavailable", False, None

        self.llm_calls_total += 1
        t0 = time.perf_counter()
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
                raw = strip_thinking_tags(resp.choices[0].message.content or "{}")
                cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
                parsed = json.loads(cleaned)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return parsed, None, True, latency_ms

            if self._provider == "anthropic":
                resp = client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system_msg,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = strip_thinking_tags(
                    resp.content[0].text if resp.content else "{}"
                )
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    return json.loads(raw[start : end + 1]), None, True, latency_ms
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return {}, "invalid_json_response", False, latency_ms
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {}, str(exc), False, latency_ms

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {}, "provider_not_supported", False, latency_ms

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
        stage = sleep_state.stage
        stage_value = stage.value
        base_emotion = replay.dominant_emotion.value if replay else "neutral"
        est_biz = max(
            0.0,
            min(
                1.0,
                0.45 * float(neuro_state.ach)
                + 0.35 * (1.0 - float(neuro_state.ne))
                + 0.2 * stress_level,
            ),
        )
        est_lucidity = (
            max(0.0, min(1.0, 0.2 * float(neuro_state.ach)))
            if stage_value == "REM"
            else 0.02
        )
        neuro_snapshot = {
            "ach": float(neuro_state.ach),
            "serotonin": float(neuro_state.serotonin),
            "ne": float(neuro_state.ne),
            "cortisol": float(neuro_state.cortisol),
        }

        memory_fragments = [
            {"id": nid, "salience": 1.0} for nid in (replay.node_ids if replay else [])
        ]
        trigger = self.trigger_detector.detect(
            time_hours=float(sleep_state.time_hours),
            stage=stage,
            bizarreness_score=float(est_biz),
            lucidity_probability=float(est_lucidity),
            memory_replay_occurred=bool(replay and replay.node_ids),
            neurochemistry=neuro_snapshot,
            memory_fragments=memory_fragments,
        )

        mode = GenerationMode.TEMPLATE
        llm_error: Optional[str] = None
        llm_trigger_type: Optional[str] = None
        llm_latency_ms: Optional[float] = None
        template_bank: Optional[str] = None
        payload: dict[str, Any]
        if trigger:
            llm_trigger_type = trigger.trigger_type.value
            system_msg, user_msg = self._build_prompt(
                sleep_state=sleep_state,
                neuro_state=neuro_state,
                replay=replay,
                stress_level=stress_level,
                prior_events=prior_events or [],
                segment_index=segment_index,
                trigger_type=trigger.trigger_type,
                bizarreness_score=est_biz,
                lucidity_probability=est_lucidity,
                prev_segments=prev_segments,
            )
            payload, llm_error, llm_used, llm_latency_ms = self._call_llm(
                system_msg, user_msg
            )
            if llm_used:
                mode = GenerationMode.LLM
                self.narrative_cache.update_from_llm(trigger.trigger_type, payload)
            else:
                mode = GenerationMode.LLM_FALLBACK
                fallback_narrative = self.narrative_cache.get_segment_narrative(
                    segment_index=segment_index,
                    emotion=base_emotion,
                    stage=stage,
                )
                payload = self._fallback_payload(
                    stage_value, base_emotion, fallback_narrative
                )
                template_bank = f"TEMPLATE_{stage_value}"
            logger.info(
                "dream-segment trigger=%s t=%.2f mode=%s llm_calls=%d",
                trigger.trigger_type.value,
                sleep_state.time_hours,
                mode.value,
                self.llm_calls_total,
            )
        else:
            narrative = self.narrative_cache.get_segment_narrative(
                segment_index=segment_index,
                emotion=base_emotion,
                stage=stage,
            )
            payload = self._fallback_payload(stage_value, base_emotion, narrative)
            if stage_value == "REM" and self.narrative_cache.active_rem_blueprint:
                mode = GenerationMode.CACHED
                template_bank = "REM_BLUEPRINT_CACHE"
            else:
                mode = GenerationMode.TEMPLATE
                template_bank = f"TEMPLATE_{stage_value}"

        return DreamSegment(
            segment_index=segment_index,
            time_hours=float(sleep_state.time_hours),
            stage=stage_value,
            narrative=str(payload.get("narrative", "")),
            dominant_emotion=str(payload.get("dominant_emotion", base_emotion)),
            bizarreness_score=float(payload.get("bizarreness_score", est_biz)),
            lucidity_probability=float(
                payload.get("lucidity_probability", est_lucidity)
            ),
            active_memory_ids=[],
            neurochemistry=neuro_snapshot,
            generation_mode=mode,
            llm_trigger_type=llm_trigger_type,
            llm_latency_ms=llm_latency_ms,
            template_bank=template_bank,
            llm_error=llm_error,
        )
