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

    # ------------------------------------------------------------------
    # Sidebar: Simulation Parameters
    # ------------------------------------------------------------------
    st.sidebar.subheader("Simulation Parameters")
    duration_hours = st.sidebar.slider("Simulated night length (hours)", 4.0, 10.0, 8.0, 0.5)
    dt_minutes = st.sidebar.slider("Time step (minutes)", 0.25, 2.0, 0.5, 0.25)
    ssri_strength = st.sidebar.slider("SSRI strength", 0.5, 2.0, 1.0, 0.1)
    stress_level = st.sidebar.slider("Stress level", 0.0, 1.0, 0.0, 0.1)

    # ------------------------------------------------------------------
    # Sidebar: LLM Settings
    # ------------------------------------------------------------------
    st.sidebar.subheader("LLM Settings")
    llm_enabled = st.sidebar.checkbox("Enable LLM-backed dreams", value=False)

    llm_provider = st.sidebar.selectbox(
        "Provider",
        ["", "openai", "lmstudio", "ollama"],
        index=0,
        disabled=not llm_enabled,
    )
    llm_model = st.sidebar.text_input(
        "Model name",
        value="gpt-4o-mini",
        disabled=not llm_enabled,
    )
    llm_api_key = st.sidebar.text_input(
        "API key (optional, overrides env var)",
        type="password",
        value="",
        disabled=not llm_enabled,
    )
    important_only = st.sidebar.checkbox(
        "Use LLM only for REM / high-replay segments",
        value=True,
        disabled=not llm_enabled,
    )

    if llm_enabled and not llm_provider:
        st.sidebar.warning("Select a provider to use LLM.")

    # ------------------------------------------------------------------
    # Sidebar: Day Journal
    # ------------------------------------------------------------------
    st.sidebar.subheader("Day Journal")
    from core.utils.journal_store import append_journal_entry  # noqa: PLC0415

    with st.sidebar.form("journal_form"):
        journal_text = st.text_area("What happened today?", height=120)
        journal_emotion = st.selectbox(
            "Dominant emotion",
            ["neutral", "joy", "fear", "sadness", "anger", "surprise", "disgust"],
            index=0,
        )
        journal_stress = st.slider("Daytime stress", 0.0, 1.0, 0.2, 0.05)
        journal_tags_raw = st.text_input("Tags (comma-separated)", value="")
        submitted = st.form_submit_button("Encode into memory graph")

    if submitted and journal_text.strip():
        tags = [t.strip() for t in journal_tags_raw.split(",") if t.strip()]
        append_journal_entry(journal_text, journal_emotion, journal_stress, tags)
        st.sidebar.success("Journal entry encoded. It will influence future dreams.")

    # ------------------------------------------------------------------
    # Build config
    # ------------------------------------------------------------------
    pharm = PharmacologyProfile(ssri_strength=ssri_strength, stress_level=stress_level)
    config = OrchestratorConfig(
        night_duration_hours=duration_hours,
        dt_minutes=dt_minutes,
        pharmacology=pharm,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider or None,
        llm_model=llm_model or None,
        llm_important_only=important_only,
        llm_api_key=llm_api_key.strip() or None,
    )

    # ------------------------------------------------------------------
    # Run button + real progress bar
    # ------------------------------------------------------------------
    if st.button("Run simulation", type="primary"):
        engine = SimulationEngine(config=config)

        progress_bar = st.progress(0, text="0% — preparing simulation...")

        def on_progress(p: float) -> None:
            pct = int(p * 100)
            label = "Simulation complete ✅" if pct >= 100 else f"Simulating night... {pct}%"
            progress_bar.progress(pct, text=label)

        with st.spinner("Running DreamForge simulation..."):
            engine.simulate_night(on_progress=on_progress)
            night = engine.build_night()

        progress_bar.progress(100, text="Simulation complete ✅")

        sleep_states = engine.orchestrator.sleep_history
        neuro_states = engine.orchestrator.neuro_history
        segments = night.segments
        summary = night.metadata.get("summary", {})

        # ------------------------------------------------------------------
        # Summary metrics
        # ------------------------------------------------------------------
        st.subheader("Summary Insights")
        cols = st.columns(4)
        sleep_stages = summary.get("sleep_stages", {})
        neuro = summary.get("neurochemistry", {})
        biz = summary.get("bizarreness", {})

        with cols[0]:
            if sleep_stages:
                dominant_stage = max(sleep_stages.items(), key=lambda kv: kv[1])[0]
                st.metric("Dominant stage", dominant_stage)
        with cols[1]:
            if neuro:
                ach_mean = neuro.get("ACh", {}).get("mean", 0.0)
                st.metric("Mean ACh", f"{ach_mean:.2f}")
        with cols[2]:
            if biz:
                st.metric("Mean bizarreness", f"{biz.get('mean', 0.0):.2f}")
        with cols[3]:
            rem_pct = sleep_stages.get("REM", 0.0)
            st.metric("REM fraction", f"{rem_pct:.0%}" if rem_pct else "—")

        # ------------------------------------------------------------------
        # Visualisations
        # ------------------------------------------------------------------
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
