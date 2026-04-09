import streamlit as st
import plotly.graph_objects as go

from typing import List

from core.models.sleep_cycle import SleepStage, SleepState


def render_hypnogram(states: List[SleepState]) -> None:
    """Render a simple hypnogram using Plotly."""

    if not states:
        st.info("No sleep states to display yet.")
        return

    time = [s.time_hours for s in states]

    stage_to_level = {
        SleepStage.WAKE: 4,
        SleepStage.N1: 3,
        SleepStage.N2: 2,
        SleepStage.N3: 1,
        SleepStage.REM: 0,
    }
    levels = [stage_to_level[s.stage] for s in states]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=levels, mode="lines", name="Stage"))
    fig.update_yaxes(
        tickvals=list(stage_to_level.values()),
        ticktext=[s.name for s in SleepStage],
        autorange="reversed",
        title="Sleep Stage",
    )
    fig.update_xaxes(title="Time (hours)")
    fig.update_layout(title="Hypnogram")

    st.plotly_chart(fig, use_container_width=True)
