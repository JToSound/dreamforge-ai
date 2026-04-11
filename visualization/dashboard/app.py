"""
DreamForge AI — Streamlit Dashboard
Real-time dream simulation visualization with LLM configuration.
"""
import json
import time
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DreamForge AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://api:8000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d0d14; }
  [data-testid="stSidebar"] { background: #12121e; border-right: 1px solid #2a2a3e; }
  h1, h2, h3 { color: #c8b8ff; }
  .stButton > button {
    background: linear-gradient(135deg, #6c3fc5, #9b5de5);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 1.4rem; font-weight: 600;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }
  .metric-card {
    background: #1a1a2e; border: 1px solid #2a2a3e;
    border-radius: 12px; padding: 1rem 1.2rem;
  }
  .stSelectbox label, .stTextInput label, .stSlider label,
  .stNumberInput label { color: #a0a0c0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar — LLM Settings ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DreamForge AI")
    st.markdown("---")

    # ── Download helpers ──────────────────────────────────────────────────
    with st.expander("📥 Download Results", expanded=False):
        json_str = json.dumps(result, indent=2)
        sim_id = result.get("id") or result.get("simulation_id") or int(time.time())
        st.download_button(
            "Download JSON",
            data=json_str,
            file_name=f"dreamforge-sim-{sim_id}.json",
            mime="application/json",
            help="Download the full simulation result as JSON",
        )

        # Build a readable text summary
        text_lines = []
        text_lines.append(f"Simulation ID: {sim_id}")
        text_lines.append(f"Duration: {metadata.get('duration_hours', duration_hours)} h")
        if result.get("summary_narrative"):
            text_lines.append("")
            text_lines.append(result.get("summary_narrative"))
        elif result.get("summary"):
            text_lines.append("")
            text_lines.append(json.dumps(result.get("summary"), indent=2))

        text_lines.append("")
        text_lines.append("Segments:")
        for i, s in enumerate(segments):
            narrative = s.get("narrative") or s.get("scene_description") or s.get("scene") or ""
            time_h = s.get("time_hours") or s.get("start_time_hours") or ""
            stage = s.get("stage") or ""
            text_lines.append(f"--- Segment {i} ({time_h}h) [{stage}]")
            text_lines.append(narrative)

        text_blob = "\n".join(text_lines)
        st.download_button(
            "Download Text",
            data=text_blob,
            file_name=f"dreamforge-sim-{sim_id}.txt",
            mime="text/plain",
            help="Download a plaintext narrative summary",
        )

    st.markdown("### ⚙️ LLM Configuration")

    llm_provider = st.selectbox(
        "Provider",
        ["lmstudio", "openai", "anthropic", "ollama"],
        help="Select your LLM backend",
    )

    provider_defaults = {
        "lmstudio":  ("http://host.docker.internal:1234/v1", "local-model"),
        "openai":    ("https://api.openai.com/v1",           "gpt-4o"),
        "anthropic": ("https://api.anthropic.com",           "claude-3-5-sonnet-20241022"),
        "ollama":    ("http://host.docker.internal:11434/v1","llama3.2"),
    }
    default_url, default_model = provider_defaults[llm_provider]

    llm_base_url = st.text_input("Base URL", value=default_url)
    llm_model    = st.text_input("Model", value=default_model)
    llm_api_key  = st.text_input(
        "API Key",
        value="lm-studio" if llm_provider == "lmstudio" else "",
        type="password",
        help="Leave blank for LM Studio / Ollama",
    )

    st.markdown("---")
    st.markdown("### 🌙 Simulation Parameters")

    duration_hours = st.slider("Night Duration (hours)", 4.0, 10.0, 8.0, 0.5)
    stress_level   = st.slider("Stress Level", 0.0, 1.0, 0.3, 0.05)
    sleep_start    = st.slider("Sleep Start (clock hour)", 20.0, 2.0, 23.0, 0.5)

    st.markdown("---")
    st.markdown("### 💊 Pharmacology")
    ssri_factor  = st.slider("SSRI Factor", 1.0, 3.0, 1.0, 0.1,
                              help="1.0 = no medication; >1 = SSRI effect")
    melatonin    = st.checkbox("Melatonin")
    cannabis     = st.checkbox("Cannabis (THC)")

    st.markdown("---")
    st.markdown("### 📝 Prior Day Events")
    events_text = st.text_area(
        "Describe today's events",
        placeholder="e.g. Had an argument with a colleague. Watched a sci-fi film. Felt anxious about the presentation.",
        height=100,
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🌌 DreamForge AI — Live Simulation")
st.markdown("*The first open-source AI system that thinks while it sleeps.*")

col_run, col_status = st.columns([2, 5])

with col_run:
    run_btn = st.button("▶  Run Simulation", use_container_width=True)

status_placeholder = col_status.empty()

# ── Helper: check API health ──────────────────────────────────────────────────
def check_api() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Helper: run simulation ────────────────────────────────────────────────────
def run_simulation() -> Optional[dict]:
    events_list = [e.strip() for e in events_text.splitlines() if e.strip()]
    payload = {
        "duration_hours":    duration_hours,
        "sleep_start_hour":  sleep_start,
        "stress_level":      stress_level,
        "llm_provider":      llm_provider,
        "llm_base_url":      llm_base_url,
        "llm_model":         llm_model,
        "llm_api_key":       llm_api_key,
        "pharmacology": {
            "ssri_factor": ssri_factor,
            "melatonin":   melatonin,
            "cannabis":    cannabis,
        },
        "prior_day_events": events_list,
    }
    try:
        with st.spinner("🧬 Simulating dream cycle…"):
            r = httpx.post(
                f"{API_BASE}/api/simulation/night",
                json=payload,
                timeout=300,        # LLM 可能需要時間
            )
        if r.status_code in (200, 201):
            return r.json()
        else:
            st.error(f"API Error {r.status_code}: {r.text[:400]}")
            return None
    except httpx.ConnectError:
        st.error("❌ Cannot reach API at `http://api:8000`. Is the `api` container running?")
        return None
    except httpx.ReadTimeout:
        st.error("⏱ Request timed out. The LLM may be slow — try a smaller model or increase timeout.")
        return None


# ── Run & display ─────────────────────────────────────────────────────────────
if run_btn:
    if not check_api():
        st.error("❌ API service not reachable. Check `docker-compose logs api`.")
    else:
        result = run_simulation()
        if result:
            st.session_state["last_result"] = result
            status_placeholder.success("✅ Simulation complete!")

result = st.session_state.get("last_result")

if result is None:
    # ── Empty state ──────────────────────────────────────────────────────────
    st.markdown("---")
    empty_col1, empty_col2, empty_col3 = st.columns([1, 2, 1])
    with empty_col2:
        st.markdown("""
        <div style="text-align:center; padding:3rem 0; color:#6060a0;">
          <div style="font-size:4rem; margin-bottom:1rem;">🌙</div>
          <h3 style="color:#8080c0;">No simulation yet</h3>
          <p>Configure your settings in the sidebar, then press <strong>▶ Run Simulation</strong>.</p>
          <p style="font-size:0.85rem; margin-top:1rem;">
            Make sure your LLM backend (LM Studio / Ollama / OpenAI) is running.
          </p>
        </div>
        """, unsafe_allow_html=True)
else:
    # ── Unpack result ─────────────────────────────────────────────────────────
    segments   = result.get("segments", [])
    neuro_data = result.get("neurochemistry", [])
    mem_graph  = result.get("memory_graph", {"nodes": [], "edges": []})
    metadata   = result.get("metadata", {})

    n_seg = len(segments)
    n_rem = sum(1 for s in segments if s.get("stage") == "REM")
    avg_bizarre = np.mean([s.get("bizarreness", 0) for s in segments]) if segments else 0

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Segments",    n_seg)
    k2.metric("REM Segments",      n_rem)
    k3.metric("Avg Bizarreness",   f"{avg_bizarre:.2f}")
    k4.metric("Night Span",        f"{metadata.get('duration_hours', duration_hours):.1f} h")

    st.markdown("---")

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab_hyp, tab_neuro, tab_mem, tab_dream = st.tabs([
        "🌙 Hypnogram", "🧪 Neurochemistry", "🕸 Memory Graph", "📖 Dream Narrative"
    ])

    # ── Hypnogram ─────────────────────────────────────────────────────────────
    with tab_hyp:
        stage_map = {"WAKE": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}
        stage_color = {
            "WAKE": "#f59e0b", "REM": "#a78bfa",
            "N1": "#60a5fa",   "N2": "#34d399", "N3": "#1d4ed8",
        }
        if segments:
            times  = [s.get("time_hours", i * duration_hours / n_seg) for i, s in enumerate(segments)]
            stages = [s.get("stage", "N2") for s in segments]
            y_vals = [stage_map.get(st, 1) for st in stages]
            colors = [stage_color.get(st, "#888") for st in stages]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times, y=y_vals,
                mode="lines",
                line=dict(color="#a78bfa", width=2),
                fill="tozeroy",
                fillcolor="rgba(167,139,250,0.15)",
                name="Sleep Stage",
            ))
            fig.update_layout(
                title="Sleep Architecture Hypnogram",
                xaxis_title="Time (hours into sleep)",
                yaxis=dict(
                    tickvals=list(stage_map.values()),
                    ticktext=list(stage_map.keys()),
                ),
                template="plotly_dark",
                height=320,
                paper_bgcolor="#0d0d14",
                plot_bgcolor="#12121e",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No segment data to display.")

    # ── Neurochemistry ────────────────────────────────────────────────────────
    with tab_neuro:
        if neuro_data:
            df_n = pd.DataFrame(neuro_data)
            fig2 = go.Figure()
            for col, color, name in [
                ("ach",      "#a78bfa", "ACh"),
                ("serotonin","#34d399", "5-HT"),
                ("ne",       "#f87171", "NE"),
                ("cortisol", "#fbbf24", "Cortisol"),
            ]:
                if col in df_n.columns:
                    fig2.add_trace(go.Scatter(
                        x=df_n.get("time_hours", df_n.index),
                        y=df_n[col],
                        name=name,
                        line=dict(color=color, width=2),
                    ))
            fig2.update_layout(
                title="Neurochemical Flux Over Night",
                xaxis_title="Time (hours)",
                yaxis_title="Relative Level",
                template="plotly_dark",
                height=360,
                paper_bgcolor="#0d0d14",
                plot_bgcolor="#12121e",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No neurochemistry data returned.")

    # ── Memory Graph ──────────────────────────────────────────────────────────
    with tab_mem:
        nodes = mem_graph.get("nodes", [])
        edges = mem_graph.get("edges", [])
        emotion_color = {
            "joy": "#fbbf24", "fear": "#f87171", "sadness": "#60a5fa",
            "anger": "#ef4444", "neutral": "#94a3b8", "surprise": "#a78bfa",
        }
        if nodes:
            import math
            n = len(nodes)
            angle_step = 2 * math.pi / max(n, 1)
            pos = {nd["id"]: (math.cos(i * angle_step), math.sin(i * angle_step))
                   for i, nd in enumerate(nodes)}

            edge_x, edge_y = [], []
            for e in edges:
                x0, y0 = pos.get(e.get("source", ""), (0, 0))
                x1, y1 = pos.get(e.get("target", ""), (0, 0))
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            node_x = [pos[nd["id"]][0] for nd in nodes]
            node_y = [pos[nd["id"]][1] for nd in nodes]
            node_colors = [emotion_color.get(nd.get("emotion", "neutral"), "#94a3b8") for nd in nodes]
            node_sizes  = [max(10, nd.get("activation", 0.5) * 40) for nd in nodes]
            node_labels = [nd.get("label", nd["id"])[:30] for nd in nodes]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=1, color="#2a2a3e"), hoverinfo="none",
            ))
            fig3.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color="#fff")),
                text=node_labels, textposition="top center",
                hovertext=[f"{nd.get('label','')} | {nd.get('emotion','')}" for nd in nodes],
            ))
            fig3.update_layout(
                title="Memory Association Graph",
                showlegend=False,
                template="plotly_dark",
                height=450,
                paper_bgcolor="#0d0d14",
                plot_bgcolor="#12121e",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No memory graph data returned.")

    # ── Dream Narrative ───────────────────────────────────────────────────────
    with tab_dream:
        if segments:
            for i, seg in enumerate(segments):
                stage   = seg.get("stage", "?")
                biz     = seg.get("bizarreness", 0)
                emotion = seg.get("dominant_emotion", "neutral")
                text    = seg.get("narrative_text", "")
                badge_color = stage_color.get(stage, "#888")

                st.markdown(f"""
                <div style="background:#1a1a2e; border-left:3px solid {badge_color};
                     border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.8rem;">
                  <div style="font-size:0.75rem; color:#6060a0; margin-bottom:0.3rem;">
                    Segment {i+1} · Stage: <b style="color:{badge_color}">{stage}</b>
                    · Emotion: {emotion} · Bizarreness: {biz:.2f}
                  </div>
                  <div style="color:#c8c8e0; font-size:0.95rem; line-height:1.6;">
                    {text if text else "<em style='color:#404060'>No narrative generated.</em>"}
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No dream segments returned.")