from __future__ import annotations

import asyncio
import importlib
import logging
import re
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
  Style preset: {style_preset}
  Prompt profile: {prompt_profile}

Rules:
- REM, bizarreness >= 0.8 -> surreal, non-linear, sensory-rich (60-90 words)
- REM, bizarreness < 0.8 -> coherent narrative with dreamlike twists (40-60 words)
- N2 -> fragmented, impressionistic (20-35 words)
- N3 -> hypnagogic micro-imagery, near-wordless (10-20 words)
- N1 -> hypnagogic edge, liminal, sensory fragments (10-15 words)
- Lucidity >= 0.5 -> dreamer gains partial self-awareness mid-narrative
- Incorporate >= 1 active memory node label into the narrative if available
- Output ONLY the narrative text, no labels, no metadata
- Style directive: {style_directive}
- Profile directive: {profile_directive}
"""

SCENE_PROMPT_TEMPLATE = """
Generate one concise visual scene description (max 25 words) for this segment.
Stage={stage}, Emotion={emotion}, Bizarreness={bizarreness:.2f}, Lucidity={lucidity:.2f}.
Output ONLY scene text.
"""

STYLE_PRESET_DIRECTIVES: dict[str, str] = {
    "scientific": "Use concrete phenomenology, restrained tone, and low metaphor density.",
    "cinematic": "Use vivid sensory detail and dynamic camera-like scene transitions.",
    "minimal": "Use sparse and clean language with short clauses and minimal adjectives.",
    "therapeutic": "Use compassionate tone, emotional safety, and gentle meaning integration.",
}

PROMPT_PROFILE_DIRECTIVES: dict[str, str] = {
    "A": "Favor immediate sensory grounding first, then emotional interpretation.",
    "B": "Favor symbolic continuity and causal links across memory fragments.",
}


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
    timeout_circuit_breaker: int = 3

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
        style_preset: str = "scientific",
        prompt_profile: str = "A",
    ) -> None:
        self.llm_client = llm_client
        self.memory_labeler = memory_labeler or (lambda node_id: str(node_id))
        self.config = config or NarrativeGeneratorConfig.from_settings()
        self.style_preset = str(style_preset or "scientific").strip().lower()
        self.prompt_profile = str(prompt_profile or "A").strip().upper()

    async def generate_batch(
        self,
        segments: list[dict[str, Any]],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        semaphore = asyncio.Semaphore(self.config.concurrency)
        loop = asyncio.get_running_loop()
        batch_started = loop.time()
        total = len(segments)
        eligible_total = sum(1 for seg in segments if self._llm_should_be_invoked(seg))
        completed = 0
        eligible_completed = 0
        llm_elapsed_seconds = 0.0
        timeout_streak = 0
        circuit_open = False
        progress_lock = asyncio.Lock()

        async def _work(index: int, seg: dict[str, Any], prior_summary: str) -> None:
            nonlocal completed, eligible_completed, llm_elapsed_seconds, timeout_streak, circuit_open
            async with semaphore:
                eligible = self._llm_should_be_invoked(seg)
                if circuit_open and eligible:
                    self._apply_preemptive_fallback(segment=seg, index=index)
                else:
                    await self._generate_segment(
                        seg, index=index, prior_summary=prior_summary
                    )

                async with progress_lock:
                    completed += 1
                    if eligible:
                        eligible_completed += 1
                        latency_ms_raw = seg.get("_llm_latency_ms")
                        if latency_ms_raw is not None:
                            try:
                                latency_seconds = max(
                                    0.0, float(latency_ms_raw) / 1000.0
                                )
                            except (TypeError, ValueError):
                                latency_seconds = 0.0
                            llm_elapsed_seconds += latency_seconds
                    if seg.get("_llm_fallback_reason") == "timeout":
                        timeout_streak += 1
                    elif bool(seg.get("_llm_invoked")) and not bool(
                        seg.get("_llm_fallback")
                    ):
                        timeout_streak = 0
                    if not circuit_open and timeout_streak >= int(
                        self.config.timeout_circuit_breaker
                    ):
                        circuit_open = True
                        logger.warning(
                            "LLM timeout circuit opened after %d consecutive timeouts.",
                            timeout_streak,
                        )
                    if progress_callback:
                        llm_avg_seconds = (
                            llm_elapsed_seconds / float(eligible_completed)
                            if eligible_completed > 0
                            else None
                        )
                        progress_callback(
                            {
                                "completed": completed,
                                "total": total,
                                "eligible_completed": eligible_completed,
                                "eligible_total": eligible_total,
                                "llm_completed_invocations": eligible_completed,
                                "llm_total_invocations": eligible_total,
                                "llm_remaining_invocations": max(
                                    0, eligible_total - eligible_completed
                                ),
                                "llm_elapsed_seconds": round(llm_elapsed_seconds, 3),
                                "llm_avg_invocation_seconds": (
                                    round(llm_avg_seconds, 3)
                                    if llm_avg_seconds is not None
                                    else None
                                ),
                                "batch_elapsed_seconds": max(
                                    0.0, loop.time() - batch_started
                                ),
                            }
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
        segment["_llm_fallback_reason"] = None
        segment["_llm_latency_ms"] = None
        stage = str(segment.get("stage", "N2"))
        bizarreness = float(segment.get("bizarreness_score", 0.0))
        should_call_llm = self._llm_should_be_invoked(segment)
        if not should_call_llm:
            return

        memory_summary = self._memory_summary(segment.get("active_memory_ids", []))
        style_directive = STYLE_PRESET_DIRECTIVES.get(
            self.style_preset, STYLE_PRESET_DIRECTIVES["scientific"]
        )
        profile_directive = PROMPT_PROFILE_DIRECTIVES.get(
            self.prompt_profile, PROMPT_PROFILE_DIRECTIVES["A"]
        )
        prompt = NARRATIVE_PROMPT_TEMPLATE.format(
            stage=stage,
            emotion=str(segment.get("dominant_emotion", "neutral")),
            bizarreness=bizarreness,
            lucidity=float(segment.get("lucidity_probability", 0.0)),
            memory_summary=memory_summary,
            prior_summary=prior_summary or "none",
            style_preset=self.style_preset,
            prompt_profile=self.prompt_profile,
            style_directive=style_directive,
            profile_directive=profile_directive,
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
            segment["scene_description"] = self._normalize_scene(str(scene))
            segment["_llm_invoked"] = True
            segment["_llm_fallback"] = False
            segment["_llm_fallback_reason"] = None
            segment["_llm_latency_ms"] = round(
                (asyncio.get_running_loop().time() - t0) * 1000.0, 1
            )
            if segment.get("generation_mode") == "TEMPLATE":
                segment["metadata"] = dict(segment.get("metadata", {}))
                segment["metadata"]["llm_invoked"] = True
        except (asyncio.TimeoutError, RuntimeError, ValueError) as exc:
            fallback = self._fallback_text(segment=segment, index=index)
            if isinstance(exc, asyncio.TimeoutError):
                reason = "timeout"
            elif isinstance(exc, RuntimeError) and str(exc).startswith(
                "[LLM unavailable:"
            ):
                low = str(exc).lower()
                if "400 bad request" in low or "status/400" in low:
                    reason = "provider_error"
                elif (
                    "connection" in low
                    or "connect" in low
                    or "dns" in low
                    or "refused" in low
                ):
                    reason = "health_unavailable"
                else:
                    reason = "provider_error"
            else:
                reason = "provider_error"
            segment["narrative"] = fallback
            segment["scene_description"] = self._normalize_scene(
                f"{stage} segment with {segment.get('dominant_emotion', 'neutral')} imagery."
            )
            segment["_llm_invoked"] = True
            segment["_llm_fallback"] = True
            segment["_llm_fallback_reason"] = reason
            segment["_llm_latency_ms"] = round(
                (asyncio.get_running_loop().time() - t0) * 1000.0, 1
            )
            logger.warning(
                "Narrative fallback used for segment %d (%s): %s",
                index,
                reason,
                exc,
            )

    def _llm_should_be_invoked(self, segment: dict[str, Any]) -> bool:
        if not bool(self.config.llm_enabled):
            return False
        stage = str(segment.get("stage", "N2"))
        bizarreness = float(segment.get("bizarreness_score", 0.0))
        return stage == "REM" or (
            stage in {"N1", "N2", "N3"}
            and bizarreness >= self.config.nrem_bizarreness_gate
        )

    def _apply_preemptive_fallback(self, segment: dict[str, Any], index: int) -> None:
        if not self._llm_should_be_invoked(segment):
            segment["_llm_invoked"] = False
            segment["_llm_fallback"] = False
            segment["_llm_fallback_reason"] = None
            segment["_llm_latency_ms"] = None
            return
        stage = str(segment.get("stage", "N2"))
        segment["narrative"] = self._fallback_text(segment=segment, index=index)
        segment["scene_description"] = self._normalize_scene(
            f"{stage} segment with {segment.get('dominant_emotion', 'neutral')} imagery."
        )
        segment["_llm_invoked"] = True
        segment["_llm_fallback"] = True
        segment["_llm_fallback_reason"] = "timeout"
        segment["_llm_latency_ms"] = 0.0

    def _normalize_narrative(
        self, text: str, segment: dict[str, Any], index: int
    ) -> str:
        stage = str(segment.get("stage", "N2"))
        biz = float(segment.get("bizarreness_score", 0.0))
        cleaned = self._sanitize_narrative_text(text)
        if not cleaned:
            return self._fallback_text(segment=segment, index=index)

        if stage == "N3":
            return self._polish_narrative_text(
                self._ensure_terminal_punctuation(
                    self._force_word_window(cleaned, minimum=10, maximum=20)
                )
            )
        if stage == "N1":
            return self._polish_narrative_text(
                self._force_word_window(cleaned, minimum=10, maximum=15)
            )
        if stage == "N2":
            return self._polish_narrative_text(
                self._force_word_window(cleaned, minimum=20, maximum=35)
            )
        if stage == "REM":
            if biz >= 0.8:
                minimum = max(self.config.rem_min_words, 60)
                maximum = 90
            else:
                minimum = max(self.config.rem_min_words, 40)
                maximum = 60
            marker = f"At {float(segment.get('start_time_hours', index / 120.0)):.3f}h"
            enriched = cleaned
            if marker not in enriched:
                enriched = (
                    f"{enriched} {marker}, the dream shifts around remembered details."
                )
            return self._polish_narrative_text(
                self._force_word_window(enriched, minimum=minimum, maximum=maximum)
            )
        return self._polish_narrative_text(cleaned)

    @staticmethod
    def _force_word_window(text: str, minimum: int, maximum: int) -> str:
        words = text.split()
        if len(words) < minimum:
            pad = (
                "sensory fragments echo through a changing environment while memory traces"
                " keep blending into new symbols and emotional textures distant sounds"
                " pulse through unstable scenes"
            ).split()
            missing = minimum - len(words)
            pad_idx = 0
            while missing > 0:
                take = min(missing, max(1, len(pad) - pad_idx))
                words.extend(pad[pad_idx : pad_idx + take])
                pad_idx = (pad_idx + take) % len(pad)
                missing = minimum - len(words)
        if len(words) > maximum:
            clipped = words[:maximum]
            last_punct_idx = max(
                (
                    i
                    for i, token in enumerate(clipped)
                    if token.endswith((".", "!", "?"))
                ),
                default=-1,
            )
            if last_punct_idx >= minimum - 1:
                clipped = clipped[: last_punct_idx + 1]
            words = clipped
        out = " ".join(words).strip()
        if out and out[-1] not in ".!?":
            out = f"{out}."
        return out

    @staticmethod
    def _ensure_terminal_punctuation(text: str) -> str:
        cleaned = " ".join(str(text).split()).strip()
        if not cleaned:
            return ""
        if cleaned[-1] in ".!?":
            return cleaned
        return f"{cleaned}."

    def _polish_narrative_text(self, text: str) -> str:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return ""
        cleaned = self._dedupe_consecutive_sentences(cleaned)
        cleaned = self._dedupe_repeated_tail(cleaned, min_phrase_words=4)
        cleaned = self._fix_indefinite_articles(cleaned)
        cleaned = cleaned.replace(";.", ".").replace(":.", ".").replace(",.", ".")
        cleaned = re.sub(r"\s+(?=[.,!?;:])", "", cleaned)
        cleaned = re.sub(r"\b(and|or|but)\s*\.$", ".", cleaned, flags=re.IGNORECASE)
        cleaned = " ".join(cleaned.split()).strip()
        return self._ensure_terminal_punctuation(cleaned)

    @staticmethod
    def _dedupe_consecutive_sentences(text: str) -> str:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", str(text)) if p.strip()]
        out: list[str] = []
        for sentence in parts:
            if out and sentence.lower() == out[-1].lower():
                continue
            out.append(sentence)
        return " ".join(out) if out else str(text)

    @staticmethod
    def _dedupe_repeated_tail(text: str, min_phrase_words: int = 5) -> str:
        words = str(text).split()
        if len(words) < min_phrase_words * 2:
            return str(text)
        max_size = min(30, len(words) // 2)
        for size in range(max_size, min_phrase_words - 1, -1):
            if words[-2 * size : -size] == words[-size:]:
                words = words[:-size]
                break
        return " ".join(words).strip()

    @staticmethod
    def _fix_indefinite_articles(text: str) -> str:
        cleaned = re.sub(r"\ba\s+([aeiouAEIOU]\w*)", r"an \1", str(text))
        cleaned = re.sub(r"\ban\s+([^aeiouAEIOU\W]\w*)", r"a \1", cleaned)
        return cleaned

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

    def _normalize_scene(self, text: str) -> str:
        cleaned = self._sanitize_free_text(text)
        cleaned = self._strip_leading_labels(
            cleaned,
            labels=[
                "scene:",
                "scene description:",
                "scene text:",
                "/no_think",
                "no_think",
            ],
        )
        cleaned = self._truncate_words(cleaned, 25)
        cleaned = " ".join(cleaned.split()).strip()
        if not cleaned:
            return "A dream scene with shifting visual details."
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def _sanitize_narrative_text(self, text: str) -> str:
        cleaned = self._sanitize_free_text(text)
        cleaned = self._strip_leading_labels(
            cleaned,
            labels=[
                "narrative:",
                "dream narrative:",
                "/no_think",
                "no_think",
            ],
        )
        cleaned = re.sub(r"\bactive_memory_([a-zA-Z0-9_]+)\b", r"\1", cleaned)
        cleaned = re.sub(r"\bmem::([a-zA-Z0-9_]+)\b", r"\1", cleaned)
        return cleaned

    @staticmethod
    def _sanitize_free_text(text: str) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"(?is)<think>.*?</think>", " ", cleaned)
        cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
        cleaned = " ".join(cleaned.split()).strip()
        return cleaned

    @staticmethod
    def _strip_leading_labels(text: str, labels: list[str]) -> str:
        cleaned = str(text or "").strip()
        for _ in range(6):
            lowered = cleaned.lower()
            removed = False
            for label in labels:
                if lowered.startswith(label):
                    cleaned = cleaned[len(label) :].strip()
                    removed = True
                    break
            if not removed:
                break
        return cleaned

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
