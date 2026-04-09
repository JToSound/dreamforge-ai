import streamlit as st

from core.agents.sleep_cycle_agent import SleepCycleAgent
from core.agents.neurochemistry_agent import NeurochemistryAgent
from core.agents.memory_consolidation_agent import MemoryConsolidationAgent
from core.models.sleep_cycle import SleepStage
from core.simulation.event_bus import EventBus
from visualization.dashboard.hypnogram import render_hypnogram
from visualization.dashboard.neurochemical_plot import render_neurochemical_flux
from visualization.dashboard.memory_graph_viz import render_memory_graph


def main() -> None:
    st.set_page_config(page_title="DreamForge Dashboard", layout="wide")
    st.title("DreamForge AI — Dream Simulation Dashboard")

    event_bus = EventBus()

    sleep_agent = SleepCycleAgent(event_bus=event_bus)
    neuro_agent = NeurochemistryAgent(event_bus=event_bus)
    memory_agent = MemoryConsolidationAgent(event_bus=event_bus)

    # Connect neurochemistry to current sleep stage
    neuro_agent.set_stage_fn(lambda t: sleep_agent.state.stage)

    states_sleep = []
    states_neuro = []

    duration_hours = st.sidebar.slider("Simulated night length (hours)", 4.0, 10.0, 8.0, 0.5)
    dt_minutes = sleep_agent.config.dt_minutes
    num_steps = int(duration_hours / (dt_minutes / 60.0))

    for _ in range(num_steps):
        state_sleep = sleep_agent.step()
        states_sleep.append(state_sleep)

        state_neuro = neuro_agent.step_to(state_sleep.time_hours)
        states_neuro.append(state_neuro)

        memory_agent.maybe_replay(current_time_hours=state_sleep.time_hours)
        memory_agent.decay_and_prune(dt_hours=dt_minutes / 60.0)

    col1, col2 = st.columns(2)
    with col1:
        render_hypnogram(states_sleep)
    with col2:
        render_neurochemical_flux(states_neuro)

    st.subheader("Memory Association Graph")
    render_memory_graph(memory_agent.graph)


if __name__ == "__main__":
    main()
