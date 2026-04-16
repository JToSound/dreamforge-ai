from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import plotly.graph_objects as go

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

STAGE_COLORS = {
    "WAKE": OKABE_ITO["black"],
    "N1": OKABE_ITO["sky"],
    "N2": OKABE_ITO["green"],
    "N3": OKABE_ITO["blue"],
    "REM": OKABE_ITO["purple"],
}

CHART_DESIGN_SYSTEM = {
    "template": "plotly_dark",
    "font_family": "Inter, Segoe UI, Arial, sans-serif",
    "paper_bgcolor": "#0d0d14",
    "plot_bgcolor": "#12121e",
    "grid_color": "#2a2a3e",
    "annotation_color": "#aab0d8",
    "default_height": 360,
    "provenance_tag": "DreamForge v1",
}

EXPORT_SPEC = {
    "default_format": "png",
    "scale": 2,
    "width": 1600,
    "height": 900,
}


def chart_export_config() -> dict:
    return {
        "displaylogo": False,
        "modeBarButtonsToAdd": ["toImage"],
        "toImageButtonOptions": {
            "format": EXPORT_SPEC["default_format"],
            "scale": EXPORT_SPEC["scale"],
            "width": EXPORT_SPEC["width"],
            "height": EXPORT_SPEC["height"],
        },
    }


def apply_chart_design(
    fig: go.Figure,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    chart_id: str,
    data_provenance: str,
    height: int | None = None,
) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template=CHART_DESIGN_SYSTEM["template"],
        paper_bgcolor=CHART_DESIGN_SYSTEM["paper_bgcolor"],
        plot_bgcolor=CHART_DESIGN_SYSTEM["plot_bgcolor"],
        font={"family": CHART_DESIGN_SYSTEM["font_family"]},
        height=height or CHART_DESIGN_SYSTEM["default_height"],
        margin={"l": 60, "r": 20, "t": 65, "b": 55},
    )
    fig.update_xaxes(gridcolor=CHART_DESIGN_SYSTEM["grid_color"])
    fig.update_yaxes(gridcolor=CHART_DESIGN_SYSTEM["grid_color"])
    fig.add_annotation(
        x=1.0,
        y=1.12,
        xref="paper",
        yref="paper",
        text=(
            f"{CHART_DESIGN_SYSTEM['provenance_tag']} · chart={chart_id}"
            f" · source={data_provenance}"
        ),
        showarrow=False,
        font={"size": 10, "color": CHART_DESIGN_SYSTEM["annotation_color"]},
        xanchor="right",
        yanchor="top",
    )
    return fig


def _seg_times(segment: dict) -> tuple[float, float]:
    start = float(segment.get("start_time_hours", segment.get("time_hours", 0.0)))
    end = float(segment.get("end_time_hours", start))
    if end < start:
        end = start
    return start, end


def _segment_duration_minutes(segment: dict) -> float:
    start, end = _seg_times(segment)
    return max(0.0, (end - start) * 60.0)


def plot_rem_episode_trend(segments: Sequence[dict]) -> go.Figure:
    episodes = []
    current = 0.0
    in_rem = False
    for segment in segments:
        stage = str(segment.get("stage", ""))
        dur = _segment_duration_minutes(segment)
        if stage == "REM":
            current += dur
            in_rem = True
        elif in_rem:
            episodes.append(current)
            current = 0.0
            in_rem = False
    if in_rem and current > 0:
        episodes.append(current)

    if not episodes:
        episodes = [0.0]

    x = list(range(1, len(episodes) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=episodes,
            mode="lines+markers",
            marker=dict(color=OKABE_ITO["purple"], size=8),
            line=dict(color=OKABE_ITO["purple"], width=2),
            name="REM duration",
            hovertemplate="Episode %{x}<br>Duration %{y:.1f} min<extra></extra>",
        )
    )
    return apply_chart_design(
        fig,
        title="REM Episode Trend",
        xaxis_title="REM episode index",
        yaxis_title="Duration (minutes)",
        chart_id="rem_episode_trend",
        data_provenance="segments.start_time_hours/end_time_hours/stage",
    )


def plot_affect_ratio_timeline(
    segments: Sequence[dict], window_minutes: float = 30.0
) -> go.Figure:
    negatives = {"fear", "sadness", "anger", "disgust", "anxious", "melancholic"}
    positives = {"joy", "serene", "curious", "surprise", "calm", "hopeful"}

    timeline = []
    for segment in segments:
        start, end = _seg_times(segment)
        midpoint = (start + end) / 2.0
        timeline.append(
            (midpoint, str(segment.get("dominant_emotion", "neutral")).lower())
        )
    timeline.sort(key=lambda item: item[0])

    window_h = window_minutes / 60.0
    dq = deque()
    neg_counts = 0
    pos_counts = 0
    xs, ys = [], []

    for t, emo in timeline:
        dq.append((t, emo))
        if emo in negatives:
            neg_counts += 1
        elif emo in positives:
            pos_counts += 1

        while dq and (t - dq[0][0]) > window_h:
            old_t, old_emo = dq.popleft()
            if old_emo in negatives:
                neg_counts -= 1
            elif old_emo in positives:
                pos_counts -= 1

        ratio = float(neg_counts) / float(max(1, pos_counts))
        xs.append(t)
        ys.append(ratio)

    if not xs:
        xs, ys = [0.0], [0.0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=OKABE_ITO["vermillion"], width=2),
            name="Neg/Pos ratio",
            hovertemplate="t=%{x:.2f}h<br>ratio=%{y:.2f}<extra></extra>",
        )
    )
    return apply_chart_design(
        fig,
        title="Affect Ratio Timeline (rolling 30 min)",
        xaxis_title="Time (hours)",
        yaxis_title="Negative / Positive",
        chart_id="affect_ratio_timeline",
        data_provenance="segments.dominant_emotion/time windows",
    )


def plot_bizarreness_cortisol_scatter(segments: Sequence[dict]) -> go.Figure:
    points_by_stage: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for segment in segments:
        stage = str(segment.get("stage", "N2"))
        neuro = segment.get("neurochemistry") or {}
        cortisol = neuro.get("cortisol")
        biz = segment.get("bizarreness_score", segment.get("bizarreness"))
        if cortisol is None or biz is None:
            continue
        points_by_stage[stage].append((float(cortisol), float(biz)))

    fig = go.Figure()
    for stage, points in points_by_stage.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    color=STAGE_COLORS.get(stage, OKABE_ITO["black"]),
                    size=8,
                    opacity=0.78,
                ),
                name=stage,
                hovertemplate=(
                    f"Stage {stage}<br>Cortisol=%{{x:.2f}}<br>Bizarreness=%{{y:.2f}}<extra></extra>"
                ),
            )
        )

    return apply_chart_design(
        fig,
        title="Bizarreness vs Cortisol",
        xaxis_title="Cortisol",
        yaxis_title="Bizarreness",
        chart_id="bizarreness_vs_cortisol",
        data_provenance="segments.bizarreness_score/neurochemistry.cortisol",
    )


def plot_per_cycle_architecture(
    segments: Sequence[dict], cycle_minutes: float = 90.0
) -> go.Figure:
    cycle_data: dict[int, dict[str, float]] = defaultdict(
        lambda: {"WAKE": 0.0, "N1": 0.0, "N2": 0.0, "N3": 0.0, "REM": 0.0}
    )

    for segment in segments:
        start, _ = _seg_times(segment)
        cycle_idx = int((start * 60.0) // cycle_minutes) + 1
        stage = str(segment.get("stage", "N2"))
        cycle_data[cycle_idx][stage] = cycle_data[cycle_idx].get(
            stage, 0.0
        ) + _segment_duration_minutes(segment)

    if not cycle_data:
        cycle_data[1]["N2"] = 0.0

    cycles = sorted(cycle_data.keys())
    fig = go.Figure()
    for stage in ["WAKE", "N1", "N2", "N3", "REM"]:
        fig.add_trace(
            go.Bar(
                x=cycles,
                y=[cycle_data[c][stage] for c in cycles],
                name=stage,
                marker_color=STAGE_COLORS.get(stage, OKABE_ITO["black"]),
                hovertemplate=(
                    f"Cycle %{{x}}<br>{stage}: %{{y:.1f}} min<extra></extra>"
                ),
            )
        )

    fig.update_layout(barmode="stack")
    return apply_chart_design(
        fig,
        title="Per-cycle Sleep Architecture",
        xaxis_title="90-minute cycle index",
        yaxis_title="Minutes",
        chart_id="per_cycle_architecture",
        data_provenance="segments.stage/start_time_hours/end_time_hours",
    )
