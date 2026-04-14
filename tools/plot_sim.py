#!/usr/bin/env python3
"""Plot simulation outputs: hypnogram, neurochemistry time series, bizarreness by stage."""
import json
import sys
import os
from collections import defaultdict

import matplotlib.pyplot as plt


STAGE_ORDER = ["WAKE", "N1", "N2", "N3", "REM"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_ORDER)}


def load_sim(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_hypnogram(segments, outdir, sim_id):
    times = [s.get("start_time_hours") for s in segments]
    stages = [s.get("stage") for s in segments]
    vals = [STAGE_TO_INT.get(s, 0) for s in stages]

    plt.figure(figsize=(12, 3))
    plt.step(times, vals, where="post")
    plt.yticks(list(STAGE_TO_INT.values()), list(STAGE_TO_INT.keys()))
    plt.xlabel("Time (hours)")
    plt.title(f"Hypnogram — {sim_id}")
    plt.tight_layout()
    p = os.path.join(outdir, f"{sim_id}_hypnogram.png")
    plt.savefig(p)
    plt.close()
    return p


def plot_neuro_ts(neuro_series, outdir, sim_id):
    times = [n.get("time_hours") for n in neuro_series]
    ach = [n.get("ach") for n in neuro_series]
    ser = [n.get("serotonin") for n in neuro_series]
    ne = [n.get("ne") for n in neuro_series]
    cort = [n.get("cortisol") for n in neuro_series]

    plt.figure(figsize=(12, 4))
    plt.plot(times, ach, label="ACh")
    plt.plot(times, ser, label="5-HT")
    plt.plot(times, ne, label="NE")
    plt.plot(times, cort, label="Cortisol")
    plt.xlabel("Time (hours)")
    plt.legend()
    plt.title(f"Neurochemistry time series — {sim_id}")
    plt.tight_layout()
    p = os.path.join(outdir, f"{sim_id}_neuro_ts.png")
    plt.savefig(p)
    plt.close()
    return p


def plot_biz_by_stage(segments, outdir, sim_id):
    groups = defaultdict(list)
    for s in segments:
        st = s.get("stage")
        b = s.get("bizarreness_score") or s.get("bizarreness")
        if b is None:
            continue
        groups[st].append(b)

    stages = list(groups.keys())
    data = [groups[s] for s in stages]

    plt.figure(figsize=(8, 4))
    plt.boxplot(data)
    plt.xticks(range(1, len(stages) + 1), stages)
    plt.ylabel("Bizarreness")
    plt.title(f"Bizarreness by stage — {sim_id}")
    plt.tight_layout()
    p = os.path.join(outdir, f"{sim_id}_biz_by_stage.png")
    plt.savefig(p)
    plt.close()
    return p


def plot_biz_hist(segments, outdir, sim_id):
    vals = [
        s.get("bizarreness_score") or s.get("bizarreness")
        for s in segments
        if (s.get("bizarreness_score") or s.get("bizarreness")) is not None
    ]
    if not vals:
        return None
    plt.figure(figsize=(8, 4))
    plt.hist(vals, bins=20, color="C0", edgecolor="k")
    plt.xlabel("Bizarreness")
    plt.ylabel("Count")
    plt.title(f"Bizarreness distribution — {sim_id}")
    plt.tight_layout()
    p = os.path.join(outdir, f"{sim_id}_biz_hist.png")
    plt.savefig(p)
    plt.close()
    return p


def plot_cortisol_curve(neuro_series, outdir, sim_id):
    if not neuro_series:
        return None
    times = [n.get("time_hours") for n in neuro_series]
    cort = [n.get("cortisol") for n in neuro_series]
    plt.figure(figsize=(12, 3))
    plt.plot(times, cort, label="Cortisol", color="C3")
    # annotate first/second half means
    half = len(cort) // 2
    if half > 0:
        mean1 = sum(cort[:half]) / half
        mean2 = sum(cort[half:]) / max(1, len(cort) - half)
        plt.axhline(
            mean1, color="gray", linestyle="--", label=f"first-half mean={mean1:.3f}"
        )
        plt.axhline(
            mean2, color="red", linestyle="--", label=f"second-half mean={mean2:.3f}"
        )
    plt.xlabel("Time (hours)")
    plt.legend()
    plt.title(f"Cortisol time series — {sim_id}")
    plt.tight_layout()
    p = os.path.join(outdir, f"{sim_id}_cortisol_ts.png")
    plt.savefig(p)
    plt.close()
    return p


def plot_ach_by_stage(segments, outdir, sim_id):
    groups = defaultdict(list)
    for s in segments:
        st = s.get("stage")
        neuro = s.get("neurochemistry") or {}
        ach = neuro.get("ach")
        if ach is None:
            continue
        groups[st].append(ach)

    stages = list(groups.keys())
    if not stages:
        return None
    data = [groups[s] for s in stages]

    plt.figure(figsize=(8, 4))
    plt.boxplot(data)
    plt.xticks(range(1, len(stages) + 1), stages)
    plt.ylabel("ACh")
    plt.title(f"ACh by stage — {sim_id}")
    plt.tight_layout()
    p = os.path.join(outdir, f"{sim_id}_ach_by_stage.png")
    plt.savefig(p)
    plt.close()
    return p


def plot_ach_vs_biz(segments, outdir, sim_id):
    pts = []
    for s in segments:
        neuro = s.get("neurochemistry") or {}
        ach = neuro.get("ach")
        biz = s.get("bizarreness_score") or s.get("bizarreness")
        if ach is None or biz is None:
            continue
        pts.append((ach, biz, s.get("stage")))

    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c="C2", alpha=0.6, s=10)
    plt.xlabel("ACh")
    plt.ylabel("Bizarreness")
    plt.title(f"ACh vs Bizarreness — {sim_id}")
    plt.tight_layout()
    p = os.path.join(outdir, f"{sim_id}_ach_vs_biz.png")
    plt.savefig(p)
    plt.close()
    return p


def plot_memory_activation_heatmap(memory_series, outdir, sim_id, top_n=50):
    # memory_series: list of {time_hours: float, activations: [{id,label,activation}, ...]}
    if not memory_series:
        return None

    # collect max activation per node to pick top_n
    node_max = {}
    node_labels = {}
    for frame in memory_series:
        for a in frame.get("activations", []):
            nid = a.get("id")
            act = float(a.get("activation", 0.0))
            node_labels[nid] = a.get("label", nid)
            node_max[nid] = max(node_max.get(nid, 0.0), act)

    if not node_max:
        return None

    top_nodes = sorted(node_max.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    node_order = [nid for nid, _ in top_nodes]

    times = [frame.get("time_hours") for frame in memory_series]
    mat = []
    for frame in memory_series:
        amap = {
            a.get("id"): float(a.get("activation", 0.0))
            for a in frame.get("activations", [])
        }
        row = [amap.get(nid, 0.0) for nid in node_order]
        mat.append(row)

    import numpy as _np

    mat = _np.array(mat)  # shape (T, N)

    # plot heatmap (transpose so nodes on y-axis)
    plt.figure(figsize=(12, max(4, len(node_order) * 0.06)))
    plt.imshow(mat.T, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="Activation")
    # x ticks: approximate times
    xticks = list(range(0, len(times), max(1, len(times) // 8)))
    xtick_labels = [f"{times[i]:.2f}" for i in xticks]
    plt.xticks(xticks, xtick_labels)
    # y ticks: node labels (truncate)
    ytick_pos = list(range(len(node_order)))
    ytick_labels = [str(node_labels.get(nid, nid))[:20] for nid in node_order]
    plt.yticks(ytick_pos, ytick_labels)
    plt.xlabel("Time (hours)")
    plt.ylabel("Memory node (top activations)")
    plt.title(f"Memory Activation Heatmap — {sim_id}")
    plt.tight_layout()
    p = os.path.join(outdir, f"{sim_id}_memory_activation_heatmap.png")
    plt.savefig(p)
    plt.close()
    return p


def main(path):
    sim = load_sim(path)
    sim_id = sim.get("id", "sim")
    segments = sim.get("segments") or sim.get("dream_segments") or []
    neuro_series = sim.get("neurochemistry_series") or []
    mem_activation = sim.get("memory_activation_series") or []

    outdir = os.path.join("reports", "plots")
    os.makedirs(outdir, exist_ok=True)

    results = {}
    if segments:
        results["hypnogram"] = plot_hypnogram(segments, outdir, sim_id)
        results["biz_by_stage"] = plot_biz_by_stage(segments, outdir, sim_id)
        results["biz_hist"] = plot_biz_hist(segments, outdir, sim_id)
        results["ach_by_stage"] = plot_ach_by_stage(segments, outdir, sim_id)
        results["ach_vs_biz"] = plot_ach_vs_biz(segments, outdir, sim_id)
    if neuro_series:
        results["neuro_ts"] = plot_neuro_ts(neuro_series, outdir, sim_id)
        results["cortisol_ts"] = plot_cortisol_curve(neuro_series, outdir, sim_id)
    if mem_activation:
        results["mem_activation_heatmap"] = plot_memory_activation_heatmap(
            mem_activation, outdir, sim_id
        )

    print("Wrote plots:", results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_sim.py <simulation-json>")
        sys.exit(2)
    main(sys.argv[1])
