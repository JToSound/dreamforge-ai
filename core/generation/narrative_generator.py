from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)
_yaml = importlib.import_module("yaml")

NARRATIVE_PROMPT_TEMPLATE = """
You are a dream narrator. Generate a vivid, psychologically realistic
dream narrative for a single sleep segment with these properties:

  Sleep stage    : {stage}
  Dominant emotion: {emotion}
  Bizarreness index: {bizarreness:.2f}
  Lucidity probability: {lucidity:.2f}
  Active memories : {memory_summary}
  Prior segment summary: {prior_summary}

Rules:
- REM, bizarreness >= 0.8 -> surreal, non-linear, sensory-rich (60-90 words)
- REM, bizarreness < 0.8 -> coherent narrative with dreamlike twists (40-60 words)
- N2 -> fragmented, impressionistic (20-35 words)
- N3 -> hypnagogic micro-imagery, near-wordless (10-20 words)
- N1 -> hypnagogic edge, liminal, sensory fragments (10-15 words)
- Lucidity >= 0.5 -> dreamer gains partial self-awareness mid-narrative
- Incorporate >= 1 active memory node label into the narrative if available
- Output ONLY the narrative text, no labels, no metadata
"""

SCENE_PROMPT_TEMPLATE = """
Generate one concise visual scene description (max 25 words) for this segment.
Stage={stage}, Emotion={emotion}, Bizarreness={bizarreness:.2f}, Lucidity={lucidity:.2f}.
Output ONLY scene text.
"""


def _load_settings() -> dict[str, Any]:
    settings_path = Path("settings.yaml")
    if not settings_path.exists():
        return {}
    with settings_path.open("r", encoding="utf-8") as handle:
        loaded = _yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


@dataclass
class NarrativeGeneratorConfig:
    llm_enabled: bool = True
    nrem_bizarreness_gate: float = 0.55
    rem_min_words: int = 40
    concurrency: int = 5
    timeout_seconds: float = 20.0

    @classmethod
    def from_settings(cls) -> "NarrativeGeneratorConfig":
        settings = _load_settings()
        return cls(
            llm_enabled=bool(settings.get("llm_enabled", True)),
            rem_min_words=int(settings.get("narrative_min_words_rem", 40)),
        )


class NarrativeGenerator:
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        memory_labeler: Callable[[str], str] | None = None,
        config: NarrativeGeneratorConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.memory_labeler = memory_labeler or (lambda node_id: str(node_id))
        self.config = config or NarrativeGeneratorConfig.from_settings()

    async def generate_batch(self, segments: list[dict[str, Any]]) -> None:
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def _work(index: int, seg: dict[str, Any], prior_summary: str) -> None:
            async with semaphore:
                await self._generate_segment(
                    seg, index=index, prior_summary=prior_summary
                )

        for start in range(0, len(segments), self.config.concurrency):
            chunk = segments[start : start + self.config.concurrency]
            prior_summaries: list[str] = []
            for offset, _seg in enumerate(chunk):
                idx = start + offset
                if idx == 0:
                    prior_summaries.append("")
                else:
                    prev = str(segments[idx - 1].get("narrative", ""))
                    prior_summaries.append(self._truncate_words(prev, max_words=30))
            await asyncio.gather(
                *[
                    _work(
                        start + i,
                        seg,
                        prior_summaries[i] if i < len(prior_summaries) else "",
                    )
                    for i, seg in enumerate(chunk)
                ]
            )

    async def _generate_segment(
        self, segment: dict[str, Any], *, index: int, prior_summary: str
    ) -> None:
        segment["_llm_invoked"] = False
        segment["_llm_fallback"] = False
        segment["_llm_latency_ms"] = None
        stage = str(segment.get("stage", "N2"))
        bizarreness = float(segment.get("bizarreness_score", 0.0))
        should_call_llm = stage in {"REM", "N3"} or (
            stage in {"N1", "N2", "N3"}
            and bizarreness > self.config.nrem_bizarreness_gate
        )
        if not (self.config.llm_enabled and should_call_llm):
            return

        memory_summary = self._memory_summary(segment.get("active_memory_ids", []))
        prompt = NARRATIVE_PROMPT_TEMPLATE.format(
            stage=stage,
            emotion=str(segment.get("dominant_emotion", "neutral")),
            bizarreness=bizarreness,
            lucidity=float(segment.get("lucidity_probability", 0.0)),
            memory_summary=memory_summary,
            prior_summary=prior_summary or "none",
        )
        if bool(segment.get("is_lucid", False)):
            prompt = "[LUCID]\n" + prompt

        scene_prompt = SCENE_PROMPT_TEMPLATE.format(
            stage=stage,
            emotion=str(segment.get("dominant_emotion", "neutral")),
            bizarreness=bizarreness,
            lucidity=float(segment.get("lucidity_probability", 0.0)),
        )

        t0 = asyncio.get_running_loop().time()
        try:
            narrative = await asyncio.wait_for(
                self.llm_client.chat(system="Dream narrative generator", user=prompt),
                timeout=self.config.timeout_seconds,
            )
            if str(narrative).startswith("[LLM unavailable:"):
                raise RuntimeError(str(narrative))
            scene = await asyncio.wait_for(
                self.llm_client.chat(
                    system="Scene description generator", user=scene_prompt
                ),
                timeout=self.config.timeout_seconds,
            )
            if str(scene).startswith("[LLM unavailable:"):
                raise RuntimeError(str(scene))
            segment["narrative"] = self._normalize_narrative(
                text=str(narrative),
                segment=segment,
                index=index,
            )
            segment["scene_description"] = self._truncate_words(str(scene), 25)
            segment["_llm_invoked"] = True
            segment["_llm_fallback"] = False
            segment["_llm_latency_ms"] = round(
                (asyncio.get_running_loop().time() - t0) * 1000.0, 1
            )
            if segment.get("generation_mode") == "TEMPLATE":
                segment["metadata"] = dict(segment.get("metadata", {}))
                segment["metadata"]["llm_invoked"] = True
        except (asyncio.TimeoutError, RuntimeError, ValueError) as exc:
            fallback = self._fallback_text(segment=segment, index=index)
            segment["narrative"] = fallback
            segment["scene_description"] = self._truncate_words(
                f"{stage} segment with {segment.get('dominant_emotion', 'neutral')} imagery.",
                25,
            )
            segment["_llm_invoked"] = True
            segment["_llm_fallback"] = True
            segment["_llm_latency_ms"] = round(
                (asyncio.get_running_loop().time() - t0) * 1000.0, 1
            )
            logger.warning("Narrative fallback used for segment %d: %s", index, exc)

    def _normalize_narrative(
        self, text: str, segment: dict[str, Any], index: int
    ) -> str:
        stage = str(segment.get("stage", "N2"))
        biz = float(segment.get("bizarreness_score", 0.0))
        cleaned = " ".join(text.split())
        if not cleaned:
            return self._fallback_text(segment=segment, index=index)

        if stage == "N3":
            return self._force_word_window(cleaned, minimum=10, maximum=20)
        if stage == "N1":
            return self._force_word_window(cleaned, minimum=10, maximum=15)
        if stage == "N2":
            return self._force_word_window(cleaned, minimum=20, maximum=35)
        if stage == "REM":
            minimum = max(self.config.rem_min_words, 60 if biz >= 0.8 else 40)
            enriched = self._force_word_window(cleaned, minimum=minimum, maximum=95)
            marker = f"At {float(segment.get('start_time_hours', index / 120.0)):.3f}h"
            if marker not in enriched:
                enriched = (
                    f"{enriched} {marker}, the dream shifts around remembered details."
                )
            return enriched
        return cleaned

    @staticmethod
    def _force_word_window(text: str, minimum: int, maximum: int) -> str:
        words = text.split()
        if len(words) < minimum:
            pad = (
                "sensory fragments echo through a changing environment while memory traces"
                " keep blending into new symbols and emotional textures"
            ).split()
            missing = minimum - len(words)
            while missing > 0:
                words.extend(pad[:missing])
                missing = minimum - len(words)
        if len(words) > maximum:
            words = words[:maximum]
        return " ".join(words)

    def _fallback_text(self, segment: dict[str, Any], index: int) -> str:
        stage = str(segment.get("stage", "N2"))
        emotion = str(segment.get("dominant_emotion", "neutral"))
        marker = float(segment.get("start_time_hours", index / 120.0))
        base = (
            f"In this {stage} interval, a {emotion} dream image unfolds around shifting"
            " memory fragments and sensory distortions."
        )
        return self._normalize_narrative(
            f"{base} At {marker:.3f}h, the scene reforms with altered perspective.",
            segment=segment,
            index=index,
        )

    def _memory_summary(self, active_ids: Any) -> str:
        if not isinstance(active_ids, list) or not active_ids:
            return "none"
        labels = [self.memory_labeler(str(node_id)) for node_id in active_ids[:5]]
        return ", ".join(labels)

    @staticmethod
    def _truncate_words(text: str, max_words: int) -> str:
        words = str(text).split()
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words])
