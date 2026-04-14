import streamlit as st
import plotly.graph_objects as go

from typing import List

from core.models.dream_segment import DreamSegment


def render_dream_timeline(segments: List[DreamSegment]) -> None:
    """Render a horizontal dream content timeline with richer styling."""

    if not segments:
        st.info("No dream segments generated for this run.")
        return

    x_start = [s.start_time_hours for s in segments]
    x_end = [s.end_time_hours for s in segments]
    durations = [e - s for s, e in zip(x_start, x_end)]
    texts = [
        f"{seg.stage.value} | {seg.dominant_emotion.value} | B={seg.bizarreness_score:.2f}"
        for seg in segments
    ]

    colors = ["#38bdf8" if seg.stage.value == "REM" else "#4ade80" for seg in segments]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=durations,
            y=[seg.stage.value for seg in segments],
            base=x_start,
            orientation="h",
            text=texts,
            marker=dict(color=colors),
            hovertext=[seg.narrative for seg in segments],
            hoverinfo="text",
            name="Dream segments",
        )
    )

    fig.update_layout(
        title="Dream Content Timeline",
        xaxis_title="Time (hours)",
        yaxis_title="Sleep Stage",
        template="plotly_dark",
        paper_bgcolor="rgba(5,10,24,1)",
        plot_bgcolor="rgba(5,10,24,1)",
    )

    st.plotly_chart(fig, use_container_width=True)
