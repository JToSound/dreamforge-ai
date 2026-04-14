#!/usr/bin/env python3
import json
import math
from collections import defaultdict, Counter
from pathlib import Path

from core.models.sleep_cycle import SleepCycleModel
from core.models.neurochemistry import NeurochemistryModel
from core.models.memory_graph import MemoryGraph
from core.agents.dream_constructor_agent import DreamConstructorAgent
from core.simulation.engine import SimulationEngine, SimulationConfig
from core.simulation.runner import export_neurochemistry_csv


def main():
    sleep_model = SleepCycleModel()
    neuro_model = NeurochemistryModel()
    memory_graph = MemoryGraph()
    dream_constructor = DreamConstructorAgent()

    engine = SimulationEngine(
        sleep_model=sleep_model,
        neuro_model=neuro_model,
        memory_graph=memory_graph,
        dream_constructor=dream_constructor,
    )

    config = SimulationConfig(
        duration_hours=8.0, dt_minutes=0.5, llm_every_n_segments=12
    )

    print("Running simulation... this may take a little while")
    out = engine.run(config)

    fname = "out_sim.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, default=str, indent=2)

    try:
        export_neurochemistry_csv(out, Path("neurochemistry.csv"))
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Warning: neurochemistry CSV export skipped: {exc}")

    # Basic diagnostics
    hypnogram = out.get("hypnogram", [])
    neuro = out.get("neurochemistry_series", [])
    segs = out.get("dream_segments", [])

    # Stage counts
    stage_counts = Counter(h.get("stage") for h in hypnogram)
    total = len(hypnogram) or 1
    stage_pct = {k: v / total for k, v in stage_counts.items()}

    # Neurochemistry stats by stage
    vals_by_stage = defaultdict(
        lambda: {"ach": [], "serotonin": [], "ne": [], "cortisol": []}
    )
    for h, n in zip(hypnogram, neuro):
        stage = h.get("stage")
        vals_by_stage[stage]["ach"].append(n.get("ach"))
        vals_by_stage[stage]["serotonin"].append(n.get("serotonin"))
        vals_by_stage[stage]["ne"].append(n.get("ne"))
        vals_by_stage[stage]["cortisol"].append(n.get("cortisol"))

    neuro_stats = {}
    for stage, d in vals_by_stage.items():
        neuro_stats[stage] = {}
        for k, arr in d.items():
            arr_clean = [x for x in arr if x is not None]
            if arr_clean:
                mean = sum(arr_clean) / len(arr_clean)
                var = sum((x - mean) ** 2 for x in arr_clean) / len(arr_clean)
                neuro_stats[stage][k] = {
                    "mean": mean,
                    "std": math.sqrt(var),
                    "count": len(arr_clean),
                }
            else:
                neuro_stats[stage][k] = {"mean": None, "std": None, "count": 0}

    # Bizarreness per stage
    biz_by_stage = defaultdict(list)
    emo_counts = Counter()
    for s in segs:
        stg = s.get("stage")
        biz = s.get("bizarreness_score") or s.get("bizarreness")
        if biz is not None:
            biz_by_stage[stg].append(biz)
        emo_counts[s.get("dominant_emotion", "neutral")] += 1

    biz_stats = {}
    for stg, arr in biz_by_stage.items():
        if arr:
            mean = sum(arr) / len(arr)
            var = sum((x - mean) ** 2 for x in arr) / len(arr)
            biz_stats[stg] = {"mean": mean, "std": math.sqrt(var), "count": len(arr)}
        else:
            biz_stats[stg] = {"mean": None, "std": None, "count": 0}

    analysis = {
        "stage_counts": dict(stage_counts),
        "stage_pct": stage_pct,
        "neuro_stats": neuro_stats,
        "biz_stats": biz_stats,
        "segment_count": len(segs),
        "emo_counts": dict(emo_counts),
    }

    with open("out_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, default=str, indent=2)

    print("Wrote:", fname, "and out_analysis.json")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
