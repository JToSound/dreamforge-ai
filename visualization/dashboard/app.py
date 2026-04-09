import streamlit as st

from core.agents.orchestrator import OrchestratorConfig
from core.simulation.engine import SimulationEngine
from core.utils.pharmacology import PharmacologyProfile
from visualization.dashboard.hypnogram import render_hypnogram
from visualization.dashboard.neurochemical_plot import render_neurochemical_flux
from visualization.dashboard.memory_graph_viz import render_memory_graph
from visualization.dashboard.dream_timeline import render_dream_timeline
from visualization.dashboard.agent_activity_heatmap import render_agent_activity_heatmap


def main() -> None:
    st.set_page_config(page_title="DreamForge Dashboard", layout="wide")
    st.title("DreamForge AI — Dream Simulation Dashboard")

    duration_hours = st.sidebar.slider("Simulated night length (hours)", 4.0, 10.0, 8.0, 0.5)
    dt_minutes = st.sidebar.slider("Time step (minutes)", 0.25, 2.0, 0.5, 0.25)
    ssri_strength = st.sidebar.slider("SSRI strength", 0.5, 2.0, 1.0, 0.1)
    stress_level = st.sidebar.slider("Stress level", 0.0, 1.0, 0.0, 0.1)

    pharm = PharmacologyProfile(ssri_strength=ssri_strength, stress_level=stress_level)
    config = OrchestratorConfig(night_duration_hours=duration_hours, dt_minutes=dt_minutes, pharmacology=pharm)

    engine = SimulationEngine(config=config)
    engine.simulate_night()
    night = engine.build_night()

    sleep_states = engine.orchestrator.sleep_history
    neuro_states = engine.orchestrator.neuro_history
    segments = night.segments

    col1, col2 = st.columns(2)
    with col1:
        render_hypnogram(sleep_states)
    with col2:
        render_neurochemical_flux(neuro_states)

    st.subheader("Memory Association Graph")
    render_memory_graph(engine.orchestrator.memory_agent.graph)

    st.subheader("Dream Content Timeline")
    render_dream_timeline(segments)

    st.subheader("Agent Activity Heatmap")
    render_agent_activity_heatmap(segments)


if __name__ == "__main__":
    main()
