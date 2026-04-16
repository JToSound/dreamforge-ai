from __future__ import annotations

import re
from typing import Any


def _word_count(text: str) -> int:
    return len([w for w in str(text or "").split() if w])


def _stage_word_bounds(stage: str, bizarreness: float) -> tuple[int, int]:
    stage_key = str(stage or "N2")
    if stage_key == "REM":
        return (60, 90) if float(bizarreness) >= 0.8 else (40, 60)
    if stage_key == "N1":
        return (10, 15)
    if stage_key == "N2":
        return (20, 35)
    if stage_key == "N3":
        return (10, 20)
    return (10, 35)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _repetition_ratio(text: str) -> float:
    tokens = _tokenize(text)
    if len(tokens) < 8:
        return 0.0
    unique = len(set(tokens))
    return max(0.0, 1.0 - (float(unique) / float(len(tokens))))


def _memory_grounding_details(
    active_ids: list[str], narrative_text: str
) -> tuple[float, int, int, float]:
    if not active_ids:
        return 1.0, 0, 0, 1.0
    text_tokens = set(_tokenize(narrative_text))
    if not text_tokens:
        unmatched = len(active_ids)
        return 0.0, 0, unmatched, 0.0

    matched = 0
    best_partial = 0.0
    for mem_id in active_ids:
        canonical = str(mem_id).split("::")[-1].replace("active_memory_", "")
        label_tokens = [tok for tok in _tokenize(canonical) if len(tok) >= 2]
        if not label_tokens:
            continue
        overlap = len([tok for tok in label_tokens if tok in text_tokens])
        ratio = float(overlap) / float(len(label_tokens))
        best_partial = max(best_partial, ratio)
        if ratio >= 0.5:
            matched += 1
    total = max(1, len(active_ids))
    unmatched = max(0, total - matched)
    coverage = float(matched) / float(total)
    confidence = min(1.0, (0.7 * coverage) + (0.3 * best_partial))
    return confidence, matched, unmatched, confidence


def score_narrative_segment(segment: dict[str, Any]) -> dict[str, float]:
    stage = str(segment.get("stage") or "N2")
    biz = float(segment.get("bizarreness_score") or 0.0)
    text = str(segment.get("narrative") or "")
    wc = _word_count(text)
    min_words, max_words = _stage_word_bounds(stage, biz)

    if min_words <= wc <= max_words:
        length_compliance = 1.0
    elif wc < min_words and min_words > 0:
        length_compliance = max(0.0, float(wc) / float(min_words))
    elif wc > max_words and wc > 0:
        length_compliance = max(0.0, float(max_words) / float(wc))
    else:
        length_compliance = 0.0

    lower = text.lower()
    artifact_hits = 0
    artifact_hits += 1 if ("<div" in lower or "<think>" in lower) else 0
    artifact_hits += 1 if "no_think" in lower else 0
    artifact_hits += (
        1 if re.search(r"\b(scene:|scene description:|scene text:)\b", lower) else 0
    )
    artifact_hits += 1 if "active_memory_" in lower else 0
    artifact_hits += 1 if " and." in lower or " or." in lower or " but." in lower else 0
    artifact_hits += 1 if ";." in lower or ":." in lower or ",." in lower else 0
    repetition_penalty = _repetition_ratio(text)
    artifact_score = max(
        0.0,
        1.0 - 0.18 * float(artifact_hits) - 0.35 * float(repetition_penalty),
    )

    active_ids = segment.get("active_memory_ids") or []
    if not isinstance(active_ids, list) or not active_ids:
        memory_grounding = 1.0
        matched_terms = 0
        unmatched_terms = 0
        grounding_confidence = 1.0
    else:
        (
            memory_grounding,
            matched_terms,
            unmatched_terms,
            grounding_confidence,
        ) = _memory_grounding_details([str(mem_id) for mem_id in active_ids], text)

    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    coherence_proxy = min(1.0, float(sentence_count) / 3.0)

    total = (
        0.40 * length_compliance
        + 0.30 * artifact_score
        + 0.20 * memory_grounding
        + 0.10 * coherence_proxy
    )
    return {
        "overall": round(float(total), 4),
        "length_compliance": round(float(length_compliance), 4),
        "artifact_score": round(float(artifact_score), 4),
        "memory_grounding": round(float(memory_grounding), 4),
        "coherence_proxy": round(float(coherence_proxy), 4),
        "memory_grounding_matched_terms": float(matched_terms),
        "memory_grounding_unmatched_terms": float(unmatched_terms),
        "memory_grounding_confidence": round(float(grounding_confidence), 4),
        "repetition_penalty": round(float(repetition_penalty), 4),
    }


def summarize_narrative_quality(
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if not segments:
        return [], {
            "narrative_quality_mean": 0.0,
            "narrative_length_compliance_mean": 0.0,
            "narrative_artifact_score_mean": 0.0,
            "narrative_memory_grounding_mean": 0.0,
        }

    scores = [score_narrative_segment(seg) for seg in segments]
    n = float(len(scores))
    summary = {
        "narrative_quality_mean": round(sum(s["overall"] for s in scores) / n, 4),
        "narrative_length_compliance_mean": round(
            sum(s["length_compliance"] for s in scores) / n, 4
        ),
        "narrative_artifact_score_mean": round(
            sum(s["artifact_score"] for s in scores) / n, 4
        ),
        "narrative_memory_grounding_mean": round(
            sum(s["memory_grounding"] for s in scores) / n, 4
        ),
    }
    return scores, summary
