#!/usr/bin/env python3
import json
import math
import sys
from collections import defaultdict, Counter


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(arr):
    arr_clean = [x for x in arr if x is not None]
    if not arr_clean:
        return {"mean": None, "std": None, "count": 0}
    mean = sum(arr_clean) / len(arr_clean)
    var = sum((x - mean) ** 2 for x in arr_clean) / len(arr_clean)
    return {"mean": mean, "std": math.sqrt(var), "count": len(arr_clean)}


def analyze(sim):
    segs = sim.get("segments") or sim.get("dream_segments") or []

    stage_counts = Counter(s.get("stage") for s in segs)
    total = len(segs) or 1
    stage_pct = {k: v / total for k, v in stage_counts.items()}

    vals_by_stage = defaultdict(lambda: defaultdict(list))
    biz_by_stage = defaultdict(list)
    emo_counts = Counter()
    cortisol_times = []

    for s in segs:
        stage = s.get("stage")
        neuro = s.get("neurochemistry", {})
        ach = neuro.get("ach")
        ser = neuro.get("serotonin")
        ne = neuro.get("ne")
        cort = neuro.get("cortisol")
        if ach is not None:
            vals_by_stage[stage]["ach"].append(ach)
        if ser is not None:
            vals_by_stage[stage]["serotonin"].append(ser)
        if ne is not None:
            vals_by_stage[stage]["ne"].append(ne)
        if cort is not None:
            vals_by_stage[stage]["cortisol"].append(cort)
        biz = s.get("bizarreness_score") or s.get("bizarreness")
        if biz is not None:
            biz_by_stage[stage].append(biz)
        emo = s.get("dominant_emotion")
        if emo:
            emo_counts[emo] += 1
        # track cortisol timeline
        t = s.get("start_time_hours")
        if cort is not None and t is not None:
            cortisol_times.append((t, cort))

    neuro_stats = {
        stage: {k: mean_std(v) for k, v in measures.items()}
        for stage, measures in vals_by_stage.items()
    }
    biz_stats = {stage: mean_std(vals) for stage, vals in biz_by_stage.items()}

    # Cortisol shape: sort by time and compute simple trend (first-half vs second-half mean)
    cortisol_times.sort()
    if cortisol_times:
        times, cort_vals = zip(*cortisol_times)
        half = len(cort_vals) // 2
        first_mean = sum(cort_vals[: max(1, half)]) / max(1, half)
        second_mean = sum(cort_vals[max(1, half) :]) / max(
            1, len(cort_vals) - max(1, half)
        )
        cortisol_trend = {
            "first_half_mean": first_mean,
            "second_half_mean": second_mean,
            "delta": second_mean - first_mean,
        }
    else:
        cortisol_trend = {
            "first_half_mean": None,
            "second_half_mean": None,
            "delta": None,
        }

    analysis = {
        "id": sim.get("id"),
        "config": sim.get("config"),
        "segment_count": len(segs),
        "stage_counts": dict(stage_counts),
        "stage_pct": stage_pct,
        "neuro_stats": neuro_stats,
        "biz_stats": biz_stats,
        "emo_counts": dict(emo_counts),
        "cortisol_trend": cortisol_trend,
    }
    return analysis


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: analyze_uploaded_sim.py <path-to-sim-json>")
        sys.exit(2)
    path = sys.argv[1]
    sim = load_json(path)
    analysis = analyze(sim)
    outp = "uploaded_sim_analysis.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print("Wrote", outp)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
