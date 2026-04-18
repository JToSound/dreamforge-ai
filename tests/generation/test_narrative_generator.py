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


class _BadRequestLLM:
    async def chat(self, system: str, user: str) -> str:
        return (
            "[LLM unavailable: Client error '400 Bad Request' for url "
            "'http://fake/chat/completions']"
        )


class _VerboseLLM:
    async def chat(self, system: str, user: str) -> str:
        if "Scene description generator" in system:
            return "An intense shifting space."
        return " ".join(["surreal"] * 180)


class _SlowLLM:
    async def chat(self, system: str, user: str) -> str:
        await asyncio.sleep(1.2)
        if "Scene description generator" in system:
            return "A corridor stretching into soft shadow."
        return "I walk through repeating hallways while distant voices trail behind me."


class _DirtyOutputLLM:
    async def chat(self, system: str, user: str) -> str:
        if "Scene description generator" in system:
            return "Scene: Scene description: /no_think A dark room with mirrored walls"
        return (
            "Narrative: <div>you run through active_memory_school_corridor while "
            "a clock melts overhead and no_think echoes in the distance</div>"
        )


class _PromptCaptureLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def chat(self, system: str, user: str) -> str:
        self.prompts.append(f"{system}\n{user}")
        if "Scene description generator" in system:
            return "A reflective hallway."
        return "The corridor loops while distant voices fade into static."


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
async def test_llm_segments_only_limits_invocation_to_rem() -> None:
    segments = [
        {
            "stage": "N2",
            "dominant_emotion": "curious",
            "bizarreness_score": 0.95,
            "lucidity_probability": 0.1,
            "active_memory_ids": [],
            "start_time_hours": 1.5,
            "narrative": "",
        },
        {
            "stage": "REM",
            "dominant_emotion": "curious",
            "bizarreness_score": 0.7,
            "lucidity_probability": 0.25,
            "active_memory_ids": [],
            "start_time_hours": 2.0,
            "narrative": "",
        },
    ]
    gen = NarrativeGenerator(
        llm_client=_StaticLLM(),
        config=NarrativeGeneratorConfig(llm_enabled=True, llm_segments_only=True),
    )
    await gen.generate_batch(segments)
    assert segments[0].get("_llm_invoked") is False
    assert segments[0].get("narrative", "") == ""
    assert segments[1].get("_llm_invoked") is True
    assert segments[1].get("narrative", "")


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
    assert seg.get("_llm_fallback_reason") == "timeout"


@pytest.mark.asyncio
async def test_n3_fallback_narrative_has_terminal_punctuation() -> None:
    seg = {
        "stage": "N3",
        "dominant_emotion": "anxious",
        "bizarreness_score": 0.9,
        "lucidity_probability": 0.0,
        "active_memory_ids": [],
        "start_time_hours": 1.75,
        "narrative": "",
    }
    gen = NarrativeGenerator(
        llm_client=_TimeoutLLM(),
        config=NarrativeGeneratorConfig(llm_enabled=True, timeout_seconds=0.01),
    )
    await gen.generate_batch([seg])
    assert seg.get("_llm_fallback") is True
    assert seg["narrative"].endswith((".", "!", "?"))
    assert "a anxious" not in seg["narrative"].lower()
    assert ";." not in seg["narrative"]
    assert " and." not in seg["narrative"].lower()


@pytest.mark.asyncio
async def test_400_bad_request_maps_to_provider_error_reason() -> None:
    seg = {
        "stage": "REM",
        "dominant_emotion": "anxious",
        "bizarreness_score": 0.9,
        "lucidity_probability": 0.0,
        "active_memory_ids": [],
        "start_time_hours": 2.25,
        "narrative": "",
    }
    gen = NarrativeGenerator(
        llm_client=_BadRequestLLM(),
        config=NarrativeGeneratorConfig(llm_enabled=True, timeout_seconds=0.01),
    )
    await gen.generate_batch([seg])
    assert seg.get("_llm_fallback") is True
    assert seg.get("_llm_fallback_reason") == "provider_error"


@pytest.mark.asyncio
async def test_high_biz_rem_narrative_ceiling_is_90_words() -> None:
    seg = {
        "stage": "REM",
        "dominant_emotion": "curious",
        "bizarreness_score": 0.95,
        "lucidity_probability": 0.4,
        "active_memory_ids": ["m1"],
        "start_time_hours": 6.25,
        "narrative": "",
    }
    gen = NarrativeGenerator(
        llm_client=_VerboseLLM(),
        config=NarrativeGeneratorConfig(llm_enabled=True),
    )
    await gen.generate_batch([seg])
    assert len(seg["narrative"].split()) <= 90


@pytest.mark.asyncio
async def test_narrative_and_scene_outputs_are_sanitized() -> None:
    seg = {
        "stage": "REM",
        "dominant_emotion": "anxious",
        "bizarreness_score": 0.9,
        "lucidity_probability": 0.2,
        "active_memory_ids": ["m1"],
        "start_time_hours": 3.3,
        "narrative": "",
    }
    gen = NarrativeGenerator(
        llm_client=_DirtyOutputLLM(),
        config=NarrativeGeneratorConfig(llm_enabled=True),
    )
    await gen.generate_batch([seg])
    assert "<div" not in seg["narrative"].lower()
    assert "active_memory_" not in seg["narrative"]
    assert not seg["narrative"].lower().startswith("narrative:")
    assert not seg["scene_description"].lower().startswith("scene:")
    assert "no_think" not in seg["scene_description"].lower()


@pytest.mark.asyncio
async def test_style_preset_and_prompt_profile_are_injected() -> None:
    seg = {
        "stage": "REM",
        "dominant_emotion": "neutral",
        "bizarreness_score": 0.75,
        "lucidity_probability": 0.3,
        "active_memory_ids": ["m1"],
        "start_time_hours": 2.5,
        "narrative": "",
    }
    capture = _PromptCaptureLLM()
    gen = NarrativeGenerator(
        llm_client=capture,
        style_preset="cinematic",
        prompt_profile="B",
        config=NarrativeGeneratorConfig(llm_enabled=True),
    )
    await gen.generate_batch([seg])
    joined = "\n".join(capture.prompts)
    assert "Style preset: cinematic" in joined
    assert "Prompt profile: B" in joined


@pytest.mark.asyncio
async def test_generate_batch_progress_callback_reports_counts() -> None:
    segs = [
        {
            "stage": "REM",
            "dominant_emotion": "neutral",
            "bizarreness_score": 0.7,
            "lucidity_probability": 0.2,
            "active_memory_ids": [],
            "start_time_hours": 1.0 + i * 0.1,
            "narrative": "",
        }
        for i in range(3)
    ]
    events: list[dict[str, int]] = []
    gen = NarrativeGenerator(
        llm_client=_StaticLLM(),
        config=NarrativeGeneratorConfig(llm_enabled=True),
    )
    await gen.generate_batch(
        segs,
        progress_callback=lambda e: events.append(
            {
                "completed": int(e.get("completed", 0)),
                "total": int(e.get("total", 0)),
                "eligible_completed": int(e.get("eligible_completed", 0)),
                "eligible_total": int(e.get("eligible_total", 0)),
            }
        ),
    )
    assert events
    assert events[-1]["completed"] == 3
    assert events[-1]["total"] == 3
    assert events[-1]["eligible_completed"] == 3
    assert events[-1]["eligible_total"] == 3


@pytest.mark.asyncio
async def test_timeout_circuit_shortcuts_subsequent_segments() -> None:
    segs = [
        {
            "stage": "REM",
            "dominant_emotion": "anxious",
            "bizarreness_score": 0.9,
            "lucidity_probability": 0.3,
            "active_memory_ids": [],
            "start_time_hours": 2.0 + i * 0.1,
            "narrative": "",
        }
        for i in range(4)
    ]
    gen = NarrativeGenerator(
        llm_client=_TimeoutLLM(),
        config=NarrativeGeneratorConfig(
            llm_enabled=True,
            timeout_seconds=0.01,
            timeout_circuit_breaker=1,
            concurrency=1,
        ),
    )
    await gen.generate_batch(segs)
    assert all(bool(s.get("_llm_fallback")) for s in segs)
    assert segs[0]["_llm_latency_ms"] is not None
    assert segs[1]["_llm_latency_ms"] == 0.0


@pytest.mark.asyncio
async def test_generate_batch_progress_callback_emits_heartbeat_and_inflight_metrics() -> (
    None
):
    seg = {
        "stage": "REM",
        "dominant_emotion": "neutral",
        "bizarreness_score": 0.7,
        "lucidity_probability": 0.2,
        "active_memory_ids": [],
        "start_time_hours": 1.0,
        "narrative": "",
    }
    events: list[dict[str, object]] = []
    gen = NarrativeGenerator(
        llm_client=_SlowLLM(),
        config=NarrativeGeneratorConfig(
            llm_enabled=True, concurrency=1, timeout_seconds=5.0
        ),
    )
    await gen.generate_batch([seg], progress_callback=events.append)
    assert events
    assert any(bool(e.get("heartbeat")) for e in events)
    assert any(int(e.get("llm_in_flight_invocations", 0)) > 0 for e in events)
