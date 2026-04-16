from core.quality.narrative_quality import (
    score_narrative_segment,
    summarize_narrative_quality,
)


def test_score_narrative_segment_penalizes_artifacts() -> None:
    segment = {
        "stage": "REM",
        "bizarreness_score": 0.9,
        "narrative": "<think>internal</think> Scene: no_think active_memory_deadline",
        "active_memory_ids": ["active_memory_deadline"],
    }
    score = score_narrative_segment(segment)
    assert score["artifact_score"] < 1.0
    assert 0.0 <= score["overall"] <= 1.0


def test_summarize_narrative_quality_returns_means() -> None:
    segments = [
        {
            "stage": "N1",
            "bizarreness_score": 0.4,
            "narrative": "Colors blur softly as footsteps pass nearby.",
            "active_memory_ids": [],
        },
        {
            "stage": "REM",
            "bizarreness_score": 0.85,
            "narrative": "I drift through mirrored streets where neon rain sings and old names glow in the sky.",
            "active_memory_ids": ["active_memory_rain"],
        },
    ]
    scores, summary = summarize_narrative_quality(segments)
    assert len(scores) == 2
    assert "narrative_quality_mean" in summary
    assert 0.0 <= summary["narrative_quality_mean"] <= 1.0
    assert "repetition_penalty" in scores[0]
    assert "memory_grounding_confidence" in scores[1]


def test_memory_grounding_uses_partial_token_overlap() -> None:
    segment = {
        "stage": "REM",
        "bizarreness_score": 0.9,
        "narrative": "A rain-soaked street appears while school bells echo nearby.",
        "active_memory_ids": [
            "active_memory_school_corridor",
            "active_memory_rain_street",
        ],
    }
    score = score_narrative_segment(segment)
    assert score["memory_grounding"] > 0.0
    assert score["memory_grounding_matched_terms"] >= 1.0
