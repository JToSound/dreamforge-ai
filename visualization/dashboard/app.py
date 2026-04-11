"""Streamlit dashboard for DreamForge AI — with real progress bar and LLM config."""

from __future__ import annotations

import json
import time
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="DreamForge AI",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ── Sidebar: config ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌙 DreamForge AI")
    st.caption("Multi-agent dream simulation")
    st.divider()

    st.markdown("### 🤖 LLM Settings")
    provider = st.selectbox("Provider", ["openai", "anthropic", "ollama"])
    model_options = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-3-5"],
        "ollama": ["llama3", "mistral", "phi3", "gemma2"],
    }
    model = st.selectbox("Model", model_options[provider])
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-… (leave blank for env var)",
        disabled=(provider == "ollama"),
    )
    base_url = st.text_input(
        "Base URL (Ollama)",
        value="http://localhost:11434/v1" if provider == "ollama" else "",
        disabled=(provider != "ollama"),
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.9, 0.05)
    max_tokens = st.slider("Max Tokens", 64, 1024, 512, 64)

    st.divider()
    st.markdown("### 🛏 Sleep Settings")
    duration_hours = st.slider("Night Duration (h)", 4.0, 12.0, 8.0, 0.5)
    stress_level = st.slider("Stress Level", 0.0, 1.0, 0.5, 0.05)
    prior_events_raw = st.text_area(
        "Prior Day Events (one per line)",
        "Had a stressful meeting\nWent for an evening run\nWatched a sci-fi film",
        height=120,
    )
    prior_events = [e.strip() for e in prior_events_raw.split("\n") if e.strip()]

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)


# ── Main content ──────────────────────────────────────────────────────────────

st.title("🌙 DreamForge AI Dashboard")
st.caption("Real-time multi-agent dream simulation with full neurobiological grounding")

if run_btn:
    payload = {
        "llm": {
            "provider": provider,
            "model": model,
            "api_key": api_key or None,
            "base_url": base_url or None,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "sleep": {
            "duration_hours": duration_hours,
            "sleep_start_clock_time": 23.0,
            "dt_minutes": 0.5,
        },
        "prior_day_events": prior_events,
        "stress_level": stress_level,
    }

    progress_bar = st.progress(0, text="Starting simulation…")
    status_placeholder = st.empty()
    segment_container = st.container()

    try:
        with requests.post(
            f"{API_BASE}/simulate-night",
            json=payload,
            stream=True,
            timeout=300,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "text/event-stream" in content_type:
                # SSE streaming
                buffer = ""
                segments_shown = 0
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    buffer += chunk
                    while "\n\n" in buffer:
                        block, buffer = buffer.split("\n\n", 1)
                        for line in block.split("\n"):
                            if line.startswith("data: "):
                                try:
                                    evt = json.loads(line[6:])
                                    pct = int(evt.get("progress", 0) * 100)
                                    msg = evt.get("message", "")
                                    stage = evt.get("stage", "")
                                    progress_bar.progress(pct, text=f"[{stage}] {msg}")
                                    status_placeholder.caption(msg)

                                    if evt.get("segment"):
                                        seg = evt["segment"]
                                        with segment_container:
                                            with st.expander(
                                                f"Segment #{seg['segment_index']} · {seg['stage']} · {seg['time_hours']:.2f}h",
                                                expanded=(segments_shown < 3),
                                            ):
                                                st.write(f"_{seg['narrative']}_")
                                                col1, col2, col3 = st.columns(3)
                                                col1.metric("Emotion", seg["dominant_emotion"])
                                                col2.metric("Bizarreness", f"{seg['bizarreness_score']:.0%}")
                                                col3.metric("Lucidity", f"{seg['lucidity_probability']:.0%}")
                                        segments_shown += 1

                                    if evt.get("result"):
                                        result = evt["result"]
                                        st.session_state["last_result"] = result
                                except json.JSONDecodeError:
                                    pass
                progress_bar.progress(100, text="✅ Simulation complete!")

            else:
                # Plain JSON fallback
                for i in range(0, 95, 5):
                    progress_bar.progress(i, text="Running simulation…")
                    time.sleep(0.2)
                result = resp.json()
                st.session_state["last_result"] = result
                progress_bar.progress(100, text="✅ Simulation complete!")

    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        st.stop()


# ── Show last result ──────────────────────────────────────────────────────────

if "last_result" in st.session_state:
    result: dict[str, Any] = st.session_state["last_result"]
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{result.get('duration_hours', 0):.1f}h")
    col2.metric("Segments", result.get("total_segments", 0))
    col3.metric("Mean Bizarreness", f"{result.get('mean_bizarreness', 0):.0%}")
    col4.metric("Dominant Emotion", result.get("dominant_emotion", "—"))

    st.markdown(f"> {result.get('summary_narrative', '')}")

    if result.get("dream_segments"):
        st.subheader("Dream Segments")
        for seg in result["dream_segments"]:
            with st.expander(
                f"#{seg['segment_index']} · {seg['stage']} · {seg['time_hours']:.2f}h",
                expanded=False,
            ):
                st.write(f"_{seg['narrative']}_")
                c1, c2, c3 = st.columns(3)
                c1.metric("Emotion", seg["dominant_emotion"])
                c2.metric("Bizarreness", f"{seg['bizarreness_score']:.0%}")
                c3.metric("Lucidity", f"{seg['lucidity_probability']:.0%}")

    if result.get("hypnogram"):
        import pandas as pd
        import plotly.express as px

        hyp_df = pd.DataFrame(result["hypnogram"])
        stage_order = ["N3", "N2", "N1", "REM", "WAKE"]
        stage_colors = {
            "WAKE": "#f59e0b",
            "N1": "#60a5fa",
            "N2": "#818cf8",
            "N3": "#6366f1",
            "REM": "#f472b6",
        }
        hyp_df["stage_num"] = hyp_df["stage"].map(
            {"WAKE": 4, "N1": 3, "N2": 2, "REM": 1, "N3": 0}
        )
        fig = px.line(
            hyp_df,
            x="time_hours",
            y="stage_num",
            color="stage",
            color_discrete_map=stage_colors,
            labels={"time_hours": "Time (h)", "stage_num": "Stage"},
            title="Sleep Hypnogram",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#1f2937",
        )
        fig.update_yaxes(
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["N3", "REM", "N2", "N1", "WAKE"],
        )
        st.plotly_chart(fig, use_container_width=True)

    if result.get("neurochemistry_series"):
        import pandas as pd
        import plotly.graph_objects as go

        nc_df = pd.DataFrame(result["neurochemistry_series"])
        fig2 = go.Figure()
        nc_colors = {"ach": "#f472b6", "serotonin": "#60a5fa", "ne": "#fbbf24", "cortisol": "#34d399"}
        for col, color in nc_colors.items():
            if col in nc_df.columns:
                fig2.add_trace(go.Scatter(x=nc_df["time_hours"], y=nc_df[col], name=col.upper(), line=dict(color=color)))
        fig2.update_layout(
            title="Neurochemical Flux",
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#1f2937",
            xaxis_title="Time (h)",
            yaxis_title="Level",
        )
        st.plotly_chart(fig2, use_container_width=True)