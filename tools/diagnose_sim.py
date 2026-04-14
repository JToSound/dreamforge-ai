#!/usr/bin/env python3
"""Compute correlation diagnostics for a simulation JSON.

Outputs a JSON file under reports/diagnostics_<simid>.json containing
per-stage and overall Pearson correlation between ACh and bizarreness.
"""
import json
import sys
import os
from collections import defaultdict

import numpy as np


def load_sim(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pearsonr(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2:
        return None
    # handle constant arrays
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    r = np.corrcoef(x, y)[0, 1]
    return float(r)


def analyze(sim):
    sim_id = sim.get("id", "sim")
    segments = sim.get("segments") or sim.get("dream_segments") or []

    overall_pairs = []
    by_stage = defaultdict(lambda: {"ach": [], "biz": []})

    for s in segments:
        neuro = s.get("neurochemistry") or {}
        ach = neuro.get("ach")
        biz = s.get("bizarreness_score") or s.get("bizarreness")
        if ach is None or biz is None:
            continue
        overall_pairs.append((ach, biz))
        st = s.get("stage")
        by_stage[st]["ach"].append(ach)
        by_stage[st]["biz"].append(biz)

    overall_r = None
    if overall_pairs:
        xs, ys = zip(*overall_pairs)
        overall_r = pearsonr(xs, ys)

    per_stage_r = {}
    for st, vals in by_stage.items():
        r = pearsonr(vals["ach"], vals["biz"])
        per_stage_r[st] = r

    result = {
        "id": sim_id,
        "overall_ach_biz_corr": overall_r,
        "per_stage_ach_biz_corr": per_stage_r,
        "n_pairs": len(overall_pairs),
    }
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: diagnose_sim.py <simulation-json>")
        sys.exit(2)
    path = sys.argv[1]
    sim = load_sim(path)
    out = analyze(sim)
    outdir = os.path.join("reports", "diagnostics")
    os.makedirs(outdir, exist_ok=True)
    fname = f"diagnostics_{out.get('id')}.json"
    outpath = os.path.join(outdir, fname)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote", outpath)
    print(json.dumps(out, indent=2))
