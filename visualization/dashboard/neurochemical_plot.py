import streamlit as st
import plotly.graph_objects as go

from typing import List

from core.models.neurochemistry import NeurochemistryState


def render_neurochemical_flux(states: List[NeurochemistryState]) -> None:
    """Render multi-line time series of neuromodulator levels."""

    time = [s.time_hours for s in states]
    ach = [s.ach for s in states]
    serotonin = [s.serotonin for s in states]
    ne = [s.ne for s in states]
    cortisol = [s.cortisol for s in states]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=ach, mode="lines", name="ACh"))
    fig.add_trace(go.Scatter(x=time, y=serotonin, mode="lines", name="5-HT"))
    fig.add_trace(go.Scatter(x=time, y=ne, mode="lines", name="NE"))
    fig.add_trace(go.Scatter(x=time, y=cortisol, mode="lines", name="Cortisol"))

    fig.update_xaxes(title="Time (hours)")
    fig.update_yaxes(title="Relative level")
    fig.update_layout(title="Neurochemical Flux")

    st.plotly_chart(fig, use_container_width=True)
