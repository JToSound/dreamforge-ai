import streamlit as st
import plotly.graph_objects as go

from typing import List

from core.agents.orchestrator import OrchestratorConfig
from core.simulation.engine import SimulationEngine
from core.utils.pharmacology import PharmacologyProfile
from core.models.dream_segment import DreamSegment


def _run_sim(duration_hours: float, ssri_strength: float, stress_level: float) -> List[DreamSegment]:
    config = OrchestratorConfig(
        night_duration_hours=duration_hours,
        pharmacology=PharmacologyProfile(ssri_strength=ssri_strength, stress_level=stress_level),
    )
    engine = SimulationEngine(config=config)
    engine.simulate_night()
    night = engine.build_night()
    return night.segments


def _summarize(segments: List[DreamSegment]) -> dict:
    if not segments:
        return {"avg_biz": 0.0, "avg_luc": 0.0}
    avg_biz = sum(s.bizarreness_score for s in segments) / len(segments)
    avg_luc = sum(s.lucidity_probability for s in segments) / len(segments)
    return {"avg_biz": avg_biz, "avg_luc": avg_luc}


def main() -> None:
    st.set_page_config(page_title="DreamForge Comparative Dashboard", layout="wide")
    st.title("DreamForge AI — Comparative Dream Analysis")

    duration_hours = st.sidebar.slider("Night length (hours)", 4.0, 10.0, 8.0, 0.5)

    st.sidebar.markdown("### Baseline")
    base_ssri = st.sidebar.slider("Baseline SSRI strength", 0.5, 2.0, 1.0, 0.1)
    base_stress = st.sidebar.slider("Baseline stress", 0.0, 1.0, 0.0, 0.1)

    st.sidebar.markdown("### Counterfactual")
    cf_ssri = st.sidebar.slider("Counterfactual SSRI strength", 0.5, 2.0, 1.2, 0.1)
    cf_stress = st.sidebar.slider("Counterfactual stress", 0.0, 1.0, 0.3, 0.1)

    if st.button("Run comparative simulation"):
        base_segments = _run_sim(duration_hours, base_ssri, base_stress)
        cf_segments = _run_sim(duration_hours, cf_ssri, cf_stress)

        base_summary = _summarize(base_segments)
        cf_summary = _summarize(cf_segments)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Baseline", "Counterfactual"],
                y=[base_summary["avg_biz"], cf_summary["avg_biz"]],
                name="Avg bizarreness",
            )
        )
        fig.add_trace(
            go.Bar(
                x=["Baseline", "Counterfactual"],
                y=[base_summary["avg_luc"], cf_summary["avg_luc"]],
                name="Avg lucidity",
            )
        )

        fig.update_layout(barmode="group", title="Bizarreness and lucidity comparison")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
