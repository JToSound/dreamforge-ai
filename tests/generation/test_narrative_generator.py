from __future__ import annotations

import asyncio

import pytest

from core.generation.narrative_generator import (
    NarrativeGenerator,
    NarrativeGeneratorConfig,
)


class _StaticLLM:
    async def chat(self, system: str, user: str) -> str:
        if "Scene description generator" in system:
            return "A dim corridor with distant footsteps."
        return "I move through a changing house where familiar voices fold into strange echoes."


class _TimeoutLLM:
    async def chat(self, system: str, user: str) -> str:
        raise asyncio.TimeoutError("timeout")


@pytest.mark.asyncio
async def test_rem_narrative_word_count_gte_40() -> None:
    seg = {
        "stage": "REM",
        "dominant_emotion": "curious",
        "bizarreness_score": 0.65,
        "lucidity_probability": 0.2,
        "active_memory_ids": ["m1"],
        "start_time_hours": 5.1,
        "narrative": "",
    }
    gen = NarrativeGenerator(
        llm_client=_StaticLLM(),
        memory_labeler=lambda _: "school corridor",
        config=NarrativeGeneratorConfig(llm_enabled=True),
    )
    await gen.generate_batch([seg])
    assert len(seg["narrative"].split()) >= 40


@pytest.mark.asyncio
async def test_narratives_are_unique_across_rem_segments() -> None:
    segments = [
        {
            "stage": "REM",
            "dominant_emotion": "neutral",
            "bizarreness_score": 0.7,
            "lucidity_probability": 0.1,
            "active_memory_ids": [],
            "start_time_hours": t,
            "narrative": "",
        }
        for t in (5.0, 5.1, 5.2, 5.3)
    ]
    gen = NarrativeGenerator(
        llm_client=_StaticLLM(),
        config=NarrativeGeneratorConfig(llm_enabled=True),
    )
    await gen.generate_batch(segments)
    texts = [s["narrative"] for s in segments]
    assert len(set(texts)) == len(texts)


@pytest.mark.asyncio
async def test_n3_narrative_word_count_in_range() -> None:
    seg = {
        "stage": "N3",
        "dominant_emotion": "serene",
        "bizarreness_score": 0.7,
        "lucidity_probability": 0.0,
        "active_memory_ids": [],
        "narrative": "",
    }
    gen = NarrativeGenerator(
        llm_client=_StaticLLM(),
        config=NarrativeGeneratorConfig(llm_enabled=True),
    )
    await gen.generate_batch([seg])
    assert 10 <= len(seg["narrative"].split()) <= 25


@pytest.mark.asyncio
async def test_llm_fallback_on_timeout_returns_template() -> None:
    seg = {
        "stage": "REM",
        "dominant_emotion": "anxious",
        "bizarreness_score": 0.8,
        "lucidity_probability": 0.3,
        "active_memory_ids": [],
        "start_time_hours": 4.5,
        "narrative": "",
    }
    gen = NarrativeGenerator(
        llm_client=_TimeoutLLM(),
        config=NarrativeGeneratorConfig(llm_enabled=True, timeout_seconds=0.01),
    )
    await gen.generate_batch([seg])
    assert seg["narrative"]
    assert seg.get("_llm_fallback") is True
