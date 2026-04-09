import streamlit as st
import plotly.graph_objects as go

from typing import List

from core.models.neurochemistry import NeurochemistryState


def render_neurochemical_flux(states: List[NeurochemistryState]) -> None:
    """Render multi-line time series of neuromodulator levels with a neon theme."""

    if not states:
        st.info("No neurochemistry data to display yet.")
        return

    time = [s.time_hours for s in states]
    ach = [s.ach for s in states]
    serotonin = [s.serotonin for s in states]
    ne = [s.ne for s in states]
    cortisol = [s.cortisol for s in states]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=ach, mode="lines", name="ACh", line=dict(color="#22d3ee")))
    fig.add_trace(go.Scatter(x=time, y=serotonin, mode="lines", name="5-HT", line=dict(color="#a855f7")))
    fig.add_trace(go.Scatter(x=time, y=ne, mode="lines", name="NE", line=dict(color="#f97316")))
    fig.add_trace(go.Scatter(x=time, y=cortisol, mode="lines", name="Cortisol", line=dict(color="#fb7185")))

    fig.update_xaxes(title="Time (hours)")
    fig.update_yaxes(title="Relative level")
    fig.update_layout(
        title="Neurochemical Flux",
        template="plotly_dark",
        paper_bgcolor="rgba(5,10,24,1)",
        plot_bgcolor="rgba(5,10,24,1)",
    )

    st.plotly_chart(fig, use_container_width=True)
