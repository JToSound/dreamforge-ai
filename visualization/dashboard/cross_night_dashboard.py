import streamlit as st
import plotly.graph_objects as go

from typing import List

from core.agents.orchestrator import OrchestratorConfig
from core.simulation.engine import SimulationEngine
from core.utils.pharmacology import PharmacologyProfile
from core.models.dream_segment import DreamNight
from core.simulation.continuity_tracker import CrossNightContinuityTracker


def _simulate_night(duration_hours: float, ssri_strength: float, stress_level: float) -> DreamNight:
    config = OrchestratorConfig(
        night_duration_hours=duration_hours,
        pharmacology=PharmacologyProfile(ssri_strength=ssri_strength, stress_level=stress_level),
    )
    engine = SimulationEngine(config=config)
    engine.simulate_night()
    night = engine.build_night()
    night.config = {
        "duration_hours": duration_hours,
        "ssri_strength": ssri_strength,
        "stress_level": stress_level,
    }
    return night


def _run_multi_night(
    num_nights: int,
    duration_hours: float,
    ssri_strength: float,
    stress_level: float,
) -> List[DreamNight]:
    nights: List[DreamNight] = []
    for _ in range(num_nights):
        nights.append(_simulate_night(duration_hours, ssri_strength, stress_level))
    return nights


def _render_recurring_sankey(nights: List[DreamNight]) -> None:
    if not nights:
        st.info("No nights simulated yet.")
        return

    data = CrossNightContinuityTracker.build_recurring_sankey(nights)
    labels = data["nodes"]["labels"]
    source = data["links"]["source"]
    target = data["links"]["target"]
    value = data["links"]["value"]

    if not value:
        st.info("No recurring memory fragments across nights (increase number of nights or adjust parameters).")
        return

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=20, thickness=20),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )

    fig.update_layout(title_text="Recurring Memory Fragments Across Nights", font_size=12)
    st.plotly_chart(fig, use_container_width=True)


def _render_recurring_table(nights: List[DreamNight]) -> None:
    stats = CrossNightContinuityTracker.compute_recurring_memory_stats(nights, min_nights=2)
    if not stats:
        st.info("No recurring memory fragments found.")
        return

    rows = []
    for mem_id, info in stats.items():
        rows.append(
            {
                "memory_id": mem_id,
                "nights": ", ".join(str(i + 1) for i in info["nights"]),
                "total_occurrences": info["count"],
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="DreamForge Cross-Night Dashboard", layout="wide")
    st.title("DreamForge AI — Cross-Night Continuity Dashboard")

    with st.sidebar:
        st.header("Simulation parameters")
        num_nights = st.slider("Number of nights", 2, 10, 5)
        duration_hours = st.slider("Night length (hours)", 4.0, 10.0, 8.0, 0.5)
        ssri_strength = st.slider("SSRI strength", 0.5, 2.0, 1.0, 0.1)
        stress_level = st.slider("Stress level", 0.0, 1.0, 0.2, 0.1)

        run_button = st.button("Run multi-night simulation")

    if not run_button:
        st.info("Configure parameters in the sidebar and click 'Run multi-night simulation'.")
        return

    with st.spinner("Simulating nights..."):
        nights = _run_multi_night(num_nights, duration_hours, ssri_strength, stress_level)

    st.subheader("Recurring memory fragments (Sankey view)")
    _render_recurring_sankey(nights)

    st.subheader("Recurring memory fragments (table)")
    _render_recurring_table(nights)


if __name__ == "__main__":
    main()
