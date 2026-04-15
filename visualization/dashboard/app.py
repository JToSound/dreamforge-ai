"""
DreamForge AI — Streamlit Dashboard
Real-time dream simulation visualization with LLM configuration.
"""

import json
import time
import html
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import httpx
import io
import zipfile
import csv
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from core.config import load_runtime_config
from core.simulation.runner import SimulationRunner
from core.utils.llm_backend import LLMBackend, Providers as LLMProviders
import networkx as nx
import math

try:
    from tools.plot_sim import plot_memory_activation_heatmap
except Exception:
    plot_memory_activation_heatmap = None

try:
    from visualization.charts.static_visualizations import (
        OKABE_ITO,
        chart_export_config,
        plot_affect_ratio_timeline,
        plot_bizarreness_cortisol_scatter,
        plot_per_cycle_architecture,
        plot_rem_episode_trend,
    )
except Exception:
    OKABE_ITO = {
        "orange": "#E69F00",
        "sky": "#56B4E9",
        "green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "purple": "#CC79A7",
        "black": "#000000",
    }

    def chart_export_config() -> dict:
        return {"displaylogo": False}

    plot_rem_episode_trend = None
    plot_affect_ratio_timeline = None
    plot_bizarreness_cortisol_scatter = None
    plot_per_cycle_architecture = None

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DreamForge AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

_RUNTIME_CONFIG = load_runtime_config()
API_BASE = _RUNTIME_CONFIG.api_base_url

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


# ── Sidebar — LLM Settings ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DreamForge AI")
    st.markdown("---")

    st.markdown("### ⚙️ LLM Configuration")

    # instantiate backend once for auto-discovery
    try:
        _llm_backend = st.session_state.get("llm_backend")
        if _llm_backend is None:
            _llm_backend = LLMBackend()
            st.session_state["llm_backend"] = _llm_backend
    except Exception:
        _llm_backend = None

    llm_provider = st.radio(
        "Provider",
        ["Auto-detect", "OpenAI", "Anthropic", "Ollama", "Offline Demo"],
        index=0,
    )

    # populate fields based on detected backend if available
    detected = None
    if _llm_backend is not None:
        detected = _llm_backend.config.provider.value

    llm_base_url = st.text_input(
        "Base URL",
        value=getattr(
            _llm_backend.config,
            "ollama_base_url",
            _RUNTIME_CONFIG.llm_ollama_base_url,
        ),
    )
    llm_model = st.text_input(
        "Model",
        value=getattr(_llm_backend.config, "model_name", _RUNTIME_CONFIG.llm_model),
    )
    llm_api_key = st.text_input(
        "API Key",
        value=(
            ""
            if detected != LLMProviders.OPENAI.value
            else (getattr(_llm_backend.config, "api_key", ""))
        ),
        type="password",
        help="Leave blank to use auto-detected or offline DreamScript",
    )

    # Status indicator
    status_text = "Unknown"
    status_color = "#f59e0b"
    if _llm_backend is None:
        status_text = "Unavailable"
        status_color = "#ef4444"
    else:
        status_text = f"Detected: {_llm_backend.config.provider.value}"
        status_color = (
            "#10b981"
            if _llm_backend.config.provider != LLMProviders.DREAMSCRIPT
            else "#f59e0b"
        )

    st.markdown(
        f"**Status:** <span style='color:{status_color}; font-weight:700'>{status_text}</span>",
        unsafe_allow_html=True,
    )
    if st.button("Test Connection"):
        if _llm_backend is None:
            st.error("LLM backend unavailable.")
        else:
            with st.spinner("Testing LLM..."):
                try:
                    test_out = _llm_backend.generate("Test: say hi")
                    st.success("LLM responded")
                    st.text_area("LLM test response", value=test_out, height=120)
                except Exception as e:
                    st.error(f"LLM test failed: {e}")

    st.markdown("---")
    st.markdown("### 🌙 Simulation Parameters")

    duration_hours = st.slider(
        "Night Duration (hours)",
        4.0,
        10.0,
        _RUNTIME_CONFIG.simulation_duration_hours,
        0.5,
    )
    stress_level = st.slider(
        "Stress Level", 0.0, 1.0, _RUNTIME_CONFIG.simulation_stress_level, 0.05
    )
    sleep_start = st.slider(
        "Sleep Start (clock hour)",
        20.0,
        2.0,
        _RUNTIME_CONFIG.simulation_sleep_start_hour,
        0.5,
    )
    simulation_request_timeout_seconds = st.number_input(
        "Simulation request timeout (seconds)",
        min_value=30,
        max_value=3600,
        value=int(_RUNTIME_CONFIG.simulation_request_timeout_seconds),
        step=30,
        help="How long the dashboard waits for /api/simulation/night before showing a timeout.",
    )

    st.markdown("---")
    st.markdown("### 💊 Pharmacology")
    ssri_factor = st.slider(
        "SSRI Factor", 1.0, 3.0, 1.0, 0.1, help="1.0 = no medication; >1 = SSRI effect"
    )
    melatonin = st.checkbox("Melatonin")
    cannabis = st.checkbox("Cannabis (THC)")

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


def _render_chart_with_exports(title: str, fig: go.Figure, key_prefix: str) -> None:
    st.markdown(f"#### {title}")
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{key_prefix}_chart",
        config=chart_export_config(),
    )

    col_png, col_svg = st.columns(2)
    try:
        png_bytes = pio.to_image(fig, format="png", scale=2)
        col_png.download_button(
            "Export PNG",
            data=png_bytes,
            file_name=f"{key_prefix}.png",
            mime="image/png",
            key=f"{key_prefix}_png",
        )
    except Exception:
        col_png.caption("PNG export requires kaleido.")

    try:
        svg_bytes = pio.to_image(fig, format="svg")
        col_svg.download_button(
            "Export SVG",
            data=svg_bytes,
            file_name=f"{key_prefix}.svg",
            mime="image/svg+xml",
            key=f"{key_prefix}_svg",
        )
    except Exception:
        col_svg.caption("SVG export requires kaleido.")


# ── Helper: run simulation ────────────────────────────────────────────────────
def run_simulation() -> Optional[dict]:
    events_list = [e.strip() for e in events_text.splitlines() if e.strip()]
    payload = {
        "duration_hours": duration_hours,
        "sleep_start_hour": sleep_start,
        "stress_level": stress_level,
        "llm_provider": llm_provider,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
        "pharmacology": {
            "ssri_factor": ssri_factor,
            "melatonin": melatonin,
            "cannabis": cannabis,
        },
        "prior_day_events": events_list,
    }
    try:
        with st.spinner("🧬 Simulating dream cycle…"):
            r = httpx.post(
                f"{API_BASE}/api/simulation/night",
                json=payload,
                timeout=float(simulation_request_timeout_seconds),
            )
        if r.status_code in (200, 201):
            return r.json()
        else:
            st.error(f"API Error {r.status_code}: {r.text[:400]}")
            return None
    except httpx.ConnectError:
        st.error(
            "❌ Cannot reach API at `http://api:8000`. Is the `api` container running?"
        )
        return None
    except httpx.ReadTimeout:
        st.error(
            "⏱ Request timed out. Increase 'Simulation request timeout (seconds)' in the sidebar and try again."
        )
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
        st.markdown(
            """
        <div style="text-align:center; padding:3rem 0; color:#6060a0;">
          <div style="font-size:4rem; margin-bottom:1rem;">🌙</div>
          <h3 style="color:#8080c0;">No simulation yet</h3>
          <p>Configure your settings in the sidebar, then press <strong>▶ Run Simulation</strong>.</p>
          <p style="font-size:0.85rem; margin-top:1rem;">
            Make sure your LLM backend (LM Studio / Ollama / OpenAI) is running.
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
else:
    # ── Unpack result ─────────────────────────────────────────────────────────
    segments = result.get("segments", [])
    neuro_data = (
        result.get("neurochemistry")
        or result.get("neurochemistry_series")
        or result.get("neurochemistry_ticks")
        or [
            {
                "time_hours": s.get("start_time_hours", i / 120.0),
                "stage": s.get("stage", "N2"),
                "ach": float((s.get("neurochemistry") or {}).get("ach", 0.0)),
                "serotonin": float(
                    (s.get("neurochemistry") or {}).get("serotonin", 0.0)
                ),
                "ne": float((s.get("neurochemistry") or {}).get("ne", 0.0)),
                "cortisol": float((s.get("neurochemistry") or {}).get("cortisol", 0.0)),
            }
            for i, s in enumerate(segments)
            if s.get("neurochemistry")
        ]
    )
    mem_graph = result.get("memory_graph", {"nodes": [], "edges": []})
    lucid_events = result.get("lucid_events", [])
    metadata = result.get("metadata", {})

    n_seg = len(segments)
    n_rem = sum(1 for s in segments if s.get("stage") == "REM")
    avg_bizarre = (
        np.mean([s.get("bizarreness", 0) for s in segments]) if segments else 0
    )

    # Unique simulation key to avoid Streamlit element ID collisions across runs
    sim_id_val = result.get("id") or result.get("simulation_id") or int(time.time())
    sim_ts = int(time.time())
    sim_key = f"{sim_id_val}_{sim_ts}"

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Segments", n_seg)
    k2.metric("REM Segments", n_rem)
    k3.metric("Avg Bizarreness", f"{avg_bizarre:.2f}")
    k4.metric("Night Span", f"{metadata.get('duration_hours', duration_hours):.1f} h")

    mode_counts = Counter(
        str(s.get("generation_mode") or "TEMPLATE").upper() for s in segments
    )
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("LLM", mode_counts.get("LLM", 0))
    g2.metric("Template", mode_counts.get("TEMPLATE", 0))
    g3.metric("Fallback", mode_counts.get("LLM_FALLBACK", 0))
    g4.metric("Cached", mode_counts.get("CACHED", 0))

    llm_latencies = [
        float(s.get("llm_latency_ms"))
        for s in segments
        if s.get("generation_mode") == "LLM" and s.get("llm_latency_ms") is not None
    ]
    if llm_latencies:
        st.caption(
            f"LLM latency median={np.median(llm_latencies):.0f}ms | "
            f"p95={np.quantile(llm_latencies, 0.95):.0f}ms | "
            f"max={np.max(llm_latencies):.0f}ms"
        )

    # Real-time animation controls
    anim_col, _, _ = st.columns([1, 0.2, 1])
    animate_btn = anim_col.button("▶ Animate Night", key=f"animate_{sim_key}")

    if animate_btn:
        runner = SimulationRunner(result)
        # placeholders
        kpi_c1 = st.empty()
        kpi_c2 = st.empty()
        kpi_c3 = st.empty()
        kpi_c4 = st.empty()
        left_ph = st.empty()
        right_ph = st.empty()
        mem_ph = st.empty()
        narrative_ph = st.empty()

        # Prepare figures once
        y_map = {"WAKE": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}

        hyp_fig = go.Figure()
        hyp_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color=OKABE_ITO["purple"], width=2),
            )
        )
        hyp_fig.update_layout(
            title="Hypnogram (live)", template="plotly_dark", height=220
        )

        neuro_fig = go.Figure()
        neuro_colors = {
            "ach": OKABE_ITO["purple"],
            "serotonin": OKABE_ITO["green"],
            "ne": OKABE_ITO["vermillion"],
            "cortisol": OKABE_ITO["yellow"],
        }
        neuro_traces = ["ach", "serotonin", "ne", "cortisol"]
        for name in neuro_traces:
            neuro_fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    name=name.upper(),
                    line=dict(color=neuro_colors.get(name, "#888"), width=2),
                )
            )
        neuro_fig.update_layout(
            title="Neurochemistry (live)", template="plotly_dark", height=320
        )

        # Memory graph layout (precompute positions)
        mg_nodes = mem_graph.get("nodes", [])
        mg_edges = mem_graph.get("edges", [])
        node_ids = [nd.get("id") for nd in mg_nodes]
        G = nx.Graph()
        for nd in node_ids:
            G.add_node(nd)
        for e in mg_edges:
            G.add_edge(e.get("source"), e.get("target"))
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42)
        else:
            pos = {
                nid: (
                    math.cos(i * 2 * math.pi / max(1, len(node_ids))),
                    math.sin(i * 2 * math.pi / max(1, len(node_ids))),
                )
                for i, nid in enumerate(node_ids)
            }

        node_x = [pos[nid][0] for nid in node_ids] if node_ids else []
        node_y = [pos[nid][1] for nid in node_ids] if node_ids else []
        node_labels = [nd.get("label", nid)[:30] for nd, nid in zip(mg_nodes, node_ids)]

        mem_fig = go.Figure()
        mem_fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(size=[10 for _ in node_ids], color="#94a3b8"),
                text=node_labels,
                textposition="top center",
            )
        )
        mem_fig.update_layout(
            title="Memory Graph (live)", template="plotly_dark", height=420
        )

        # state for typewriter effect and REM pulse
        prev_stage = None
        typewriter_state = {
            "active": False,
            "text": "",
            "pos": 0,
            "target": "",
            "last_update": 0,
        }
        rem_pulse_until = 0.0

        start_time = time.time()
        # Iterate ticks
        for tick in runner.run():
            now = time.time()
            # KPIs
            kpi_c1.metric("Current Stage", str(tick.stage))
            kpi_c2.metric("ACh Level", f"{tick.ach:.2f}")
            kpi_c3.metric("Bizarreness", f"{tick.biz:.2f}")
            kpi_c4.metric("Active Memories", len(tick.active_memory_ids or []))

            # Update hypnogram incrementally
            hyp_x = (
                list(hyp_fig.data[0].x)
                if hyp_fig.data and hyp_fig.data[0].x is not None
                else []
            )
            hyp_y = (
                list(hyp_fig.data[0].y)
                if hyp_fig.data and hyp_fig.data[0].y is not None
                else []
            )
            hyp_x.append(tick.time_hours)
            hyp_y.append(y_map.get(str(tick.stage), 1))
            hyp_fig.data[0].x = hyp_x
            hyp_fig.data[0].y = hyp_y

            # REM pulse detection
            if (
                prev_stage is not None
                and str(prev_stage) != str(tick.stage)
                and str(tick.stage) == "REM"
            ):
                rem_pulse_until = now + 0.5
            prev_stage = tick.stage

            # apply transient hypnogram flash if in pulse window
            if now < rem_pulse_until:
                hyp_fig.update_traces(line=dict(color=OKABE_ITO["orange"], width=3))
            else:
                hyp_fig.update_traces(line=dict(color=OKABE_ITO["purple"], width=2))

            left_ph.plotly_chart(
                hyp_fig, use_container_width=True, key=f"hypnogram_live_{sim_key}"
            )

            # Update neurochemistry traces
            for idx, name in enumerate(neuro_traces):
                xs = (
                    list(neuro_fig.data[idx].x)
                    if neuro_fig.data[idx].x is not None
                    else []
                )
                ys = (
                    list(neuro_fig.data[idx].y)
                    if neuro_fig.data[idx].y is not None
                    else []
                )
                xs.append(tick.time_hours)
                val = getattr(tick, name, None)
                ys.append(float(val) if val is not None else 0.0)
                neuro_fig.data[idx].x = xs
                neuro_fig.data[idx].y = ys

            # neuro pulse overlay during REM entry: add a temporary marker
            if now < rem_pulse_until:
                # add small marker for ACh spike (first trace)
                neuro_fig.data[0].marker = dict(size=6, color="#ffffff")
            else:
                neuro_fig.data[0].marker = dict(size=0)

            right_ph.plotly_chart(
                neuro_fig,
                use_container_width=True,
                key=f"neurochemistry_live_{sim_key}",
            )

            # Memory graph: update node sizes from mem activation snapshots if provided
            mem_activation_series = (
                result.get("memory_activation_series")
                or result.get("memory_activations")
                or []
            )
            sizes = [10 for _ in node_ids]
            if mem_activation_series:
                # try to find frame for this tick time
                frame = next(
                    (
                        f
                        for f in mem_activation_series
                        if abs(float(f.get("time_hours", 0.0)) - float(tick.time_hours))
                        < 1e-3
                    ),
                    None,
                )
                if frame:
                    # frame['activations'] expected list of {id,label,activation}
                    act_map = {
                        a.get("id"): float(a.get("activation", 0.0))
                        for a in frame.get("activations", [])
                    }
                    for i, nid in enumerate(node_ids):
                        sizes[i] = max(8, 8 + int(act_map.get(nid, 0.0) * 40))

            # apply replay event pulses
            for ev in tick.replay_events or []:
                for i, nid in enumerate(node_ids):
                    if nid in ev.get("node_ids", []):
                        sizes[i] = min(60, sizes[i] + 20)

            # update mem_fig markers
            if mem_fig.data:
                mem_fig.data[0].marker.size = sizes
                mem_ph.plotly_chart(
                    mem_fig,
                    use_container_width=True,
                    key=f"memory_graph_live_{sim_key}",
                )

            # Narrative: chunked typewriter (few chars per tick)
            if typewriter_state["active"]:
                # advance
                step = 6
                pos = typewriter_state["pos"] + step
                typewriter_state["pos"] = min(len(typewriter_state["target"]), pos)
                display = typewriter_state["target"][: typewriter_state["pos"]]
                # emotion color badge
                emotion = None
                if tick.segment_index is not None and tick.segment_index < len(
                    segments
                ):
                    emotion = segments[tick.segment_index].get(
                        "dominant_emotion", "neutral"
                    )
                color = {
                    "joy": "#fbbf24",
                    "fear": "#f87171",
                    "sadness": "#60a5fa",
                    "anger": "#ef4444",
                    "neutral": "#94a3b8",
                    "surprise": "#a78bfa",
                }.get(emotion, "#94a3b8")
                narrative_ph.markdown(
                    f"<div style='background:{color}; padding:0.6rem; border-radius:6px;'>"
                    f"<strong>Segment {tick.segment_index+1}:</strong> {display}</div>",
                    unsafe_allow_html=True,
                )
                if typewriter_state["pos"] >= len(typewriter_state["target"]):
                    typewriter_state["active"] = False
            else:
                # check if new narrative available
                if tick.narrative and (
                    not typewriter_state["target"]
                    or tick.narrative != typewriter_state["target"]
                ):
                    typewriter_state["active"] = True
                    typewriter_state["target"] = tick.narrative
                    typewriter_state["pos"] = 0
                    narrative_ph.markdown(f"**Segment {tick.segment_index+1}:** ")

            # maintain responsive timing (target ~30ms per tick)
            elapsed = time.time() - now
            sleep_for = max(0.0, 0.03 - elapsed)
            time.sleep(sleep_for)

    st.markdown("---")

    # ── Download / export results ──────────────────────────────────────────
    with st.expander("📥 Download Results", expanded=False):
        # Normalize keys
        sim_id = result.get("id") or result.get("simulation_id") or int(time.time())

        # Full JSON download
        try:
            json_str = json.dumps(result, indent=2)
        except Exception:
            json_str = json.dumps(result, default=str, indent=2)

        st.download_button(
            "Download full JSON",
            data=json_str,
            file_name=f"dreamforge-sim-{sim_id}.json",
            mime="application/json",
            key=f"download_json_{sim_key}",
        )

        # Plaintext narrative summary
        text_lines = []
        text_lines.append(f"Simulation ID: {sim_id}")
        text_lines.append(
            f"Duration: {metadata.get('duration_hours', duration_hours)} h"
        )
        if result.get("summary_narrative"):
            text_lines.append("")
            text_lines.append(result.get("summary_narrative"))
        elif result.get("summary"):
            text_lines.append("")
            try:
                text_lines.append(json.dumps(result.get("summary"), indent=2))
            except Exception:
                text_lines.append(str(result.get("summary")))

        text_lines.append("")
        text_lines.append("Segments:")
        segs = result.get("segments") or result.get("dream_segments") or []
        for i, s in enumerate(segs):
            narrative = (
                s.get("narrative")
                or s.get("scene_description")
                or s.get("scene")
                or s.get("narrative_text")
                or ""
            )
            time_h = s.get("time_hours") or s.get("start_time_hours") or ""
            stage = s.get("stage") or ""
            text_lines.append(f"--- Segment {i} ({time_h}h) [{stage}]")
            text_lines.append(narrative)

        text_blob = "\n".join(text_lines)
        st.download_button(
            "Download plaintext narrative",
            data=text_blob,
            file_name=f"dreamforge-sim-{sim_id}.txt",
            mime="text/plain",
            key=f"download_txt_{sim_key}",
        )

        # ZIP with CSVs for analysis
        # Prepare hypnogram
        hyp = result.get("hypnogram") or [
            {
                "time_hours": s.get("time_hours") or s.get("start_time_hours"),
                "stage": s.get("stage"),
            }
            for s in segs
        ]

        # Neurochemistry series
        neuro = (
            result.get("neurochemistry")
            or result.get("neurochemistry_series")
            or result.get("neurochemistry_ticks")
            or [
                {
                    "time_hours": s.get("start_time_hours"),
                    "stage": s.get("stage"),
                    "ach": (s.get("neurochemistry") or {}).get("ach"),
                    "serotonin": (s.get("neurochemistry") or {}).get("serotonin"),
                    "ne": (s.get("neurochemistry") or {}).get("ne"),
                    "cortisol": (s.get("neurochemistry") or {}).get("cortisol"),
                }
                for s in segs
                if s.get("neurochemistry") is not None
            ]
        )

        # Memory graph
        mg = result.get("memory_graph") or result.get(
            "memory_graph", {"nodes": [], "edges": []}
        )
        nodes = mg.get("nodes", [])
        edges = mg.get("edges", [])
        memory_activations = result.get("memory_activation_series") or result.get(
            "memory_activations", []
        )

        # Build ZIP in-memory
        zip_buf = io.BytesIO()
        artifact_prefix = f"dreamforge-sim-{sim_id}"
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{artifact_prefix}.json", json_str)

            # hypnogram.csv
            sio = io.StringIO()
            writer = csv.writer(sio)
            writer.writerow(["time_hours", "stage"])
            for h in hyp:
                writer.writerow([h.get("time_hours"), h.get("stage")])
            zf.writestr("hypnogram.csv", sio.getvalue())

            # neurochemistry.csv
            sio = io.StringIO()
            if neuro and isinstance(neuro, list) and len(neuro) > 0:
                cols = sorted({k for row in neuro for k in row.keys()})
                writer = csv.writer(sio)
                writer.writerow(cols)
                for row in neuro:
                    writer.writerow([row.get(c, "") for c in cols])
            else:
                writer = csv.writer(sio)
                writer.writerow(["time_hours", "ach", "serotonin", "ne", "cortisol"])
            zf.writestr("neurochemistry.csv", sio.getvalue())

            # segments.csv
            sio = io.StringIO()
            seg_cols = [
                "segment_index",
                "time_hours",
                "start_time_hours",
                "end_time_hours",
                "stage",
                "dominant_emotion",
                "bizarreness_score",
                "lucidity_probability",
                "is_lucid",
                "generation_mode",
                "llm_trigger_type",
                "llm_latency_ms",
                "llm_fallback_reason",
                "template_bank",
                "ach",
                "serotonin",
                "ne",
                "cortisol",
                "narrative",
                "scene_description",
                "active_memory_ids",
            ]
            writer = csv.writer(sio)
            writer.writerow(seg_cols)

            def _first_non_none(*values: object) -> object:
                for value in values:
                    if value is not None:
                        return value
                return ""

            for s in segs:
                neuro = s.get("neurochemistry") or {}
                writer.writerow(
                    [
                        _first_non_none(s.get("segment_index"), s.get("id")),
                        _first_non_none(s.get("time_hours"), s.get("start_time_hours")),
                        _first_non_none(s.get("start_time_hours")),
                        _first_non_none(s.get("end_time_hours")),
                        _first_non_none(s.get("stage"), ""),
                        _first_non_none(
                            s.get("dominant_emotion"), s.get("emotion"), ""
                        ),
                        _first_non_none(
                            s.get("bizarreness_score"), s.get("bizarreness"), ""
                        ),
                        _first_non_none(
                            s.get("lucidity_probability"), s.get("lucidity_score"), ""
                        ),
                        bool(s.get("is_lucid", False)),
                        _first_non_none(s.get("generation_mode"), "TEMPLATE"),
                        _first_non_none(s.get("llm_trigger_type"), ""),
                        _first_non_none(s.get("llm_latency_ms"), ""),
                        _first_non_none(s.get("llm_fallback_reason"), ""),
                        _first_non_none(s.get("template_bank"), ""),
                        neuro.get("ach", ""),
                        neuro.get("serotonin", ""),
                        neuro.get("ne", ""),
                        neuro.get("cortisol", ""),
                        (
                            s.get("narrative")
                            or s.get("narrative_text")
                            or s.get("scene_description")
                            or ""
                        ).replace("\n", " "),
                        s.get("scene_description") or "",
                        (
                            "|".join(s.get("active_memory_ids", []))
                            if isinstance(s.get("active_memory_ids"), list)
                            else s.get("active_memory_ids") or ""
                        ),
                    ]
                )
            zf.writestr("segments.csv", sio.getvalue())

            # memory_activations.csv
            sio = io.StringIO()
            writer = csv.writer(sio)
            writer.writerow(["time_hours", "node_id", "node_label", "activation"])
            if isinstance(memory_activations, list):
                for snap in memory_activations:
                    if not isinstance(snap, dict):
                        continue
                    t_val = snap.get("time_hours", "")
                    activations = snap.get("activations", [])
                    if not isinstance(activations, list):
                        continue
                    for node in activations:
                        if not isinstance(node, dict):
                            continue
                        writer.writerow(
                            [
                                t_val,
                                node.get("id", ""),
                                node.get("label", ""),
                                node.get("activation", ""),
                            ]
                        )
            zf.writestr("memory_activations.csv", sio.getvalue())

            # memory nodes/edges
            sio = io.StringIO()
            if nodes:
                node_cols = sorted({k for n in nodes for k in n.keys()})
                writer = csv.writer(sio)
                writer.writerow(node_cols)
                for n in nodes:
                    writer.writerow([n.get(c, "") for c in node_cols])
            zf.writestr("memory_nodes.csv", sio.getvalue())

            sio = io.StringIO()
            if edges:
                edge_cols = sorted({k for e in edges for k in e.keys()})
                writer = csv.writer(sio)
                writer.writerow(edge_cols)
                for e in edges:
                    writer.writerow([e.get(c, "") for c in edge_cols])
            zf.writestr("memory_edges.csv", sio.getvalue())

            # narrative.txt
            zf.writestr(f"{artifact_prefix}.txt", text_blob)

        zip_buf.seek(0)
        st.download_button(
            "Download ZIP (CSV + JSON + text)",
            data=zip_buf.getvalue(),
            file_name=f"dreamforge-sim-{sim_id}.zip",
            mime="application/zip",
            key=f"download_zip_{sim_key}",
        )

    # ── Tab layout ───────────────────────────────────────────────────────────
    tab_hyp, tab_neuro, tab_mem, tab_dream = st.tabs(
        ["🌙 Hypnogram", "🧪 Neurochemistry", "🕸 Memory Graph", "📖 Dream Narrative"]
    )

    # ── Hypnogram ─────────────────────────────────────────────────────────────
    with tab_hyp:
        stage_map = {"WAKE": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}
        stage_color = {
            "WAKE": OKABE_ITO["black"],
            "REM": OKABE_ITO["purple"],
            "N1": OKABE_ITO["sky"],
            "N2": OKABE_ITO["green"],
            "N3": OKABE_ITO["blue"],
        }
        if segments:
            times = [
                s.get("time_hours", i * duration_hours / n_seg)
                for i, s in enumerate(segments)
            ]
            stages = [s.get("stage", "N2") for s in segments]
            y_vals = [stage_map.get(st, 1) for st in stages]
            colors = [stage_color.get(st, "#888") for st in stages]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=y_vals,
                    mode="lines",
                    line=dict(color=OKABE_ITO["purple"], width=2),
                    fill="tozeroy",
                    fillcolor="rgba(167,139,250,0.15)",
                    name="Sleep Stage",
                )
            )
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
            st.plotly_chart(
                fig, use_container_width=True, key=f"hypnogram_tab_{sim_key}"
            )
        else:
            st.info("No segment data to display.")

    # ── Neurochemistry ────────────────────────────────────────────────────────
    with tab_neuro:
        if neuro_data:
            df_n = pd.DataFrame(neuro_data)
            fig2 = go.Figure()
            for col, color, name in [
                ("ach", OKABE_ITO["purple"], "ACh"),
                ("serotonin", OKABE_ITO["green"], "5-HT"),
                ("ne", OKABE_ITO["vermillion"], "NE"),
                ("cortisol", OKABE_ITO["yellow"], "Cortisol"),
            ]:
                if col in df_n.columns:
                    fig2.add_trace(
                        go.Scatter(
                            x=df_n.get("time_hours", df_n.index),
                            y=df_n[col],
                            name=name,
                            line=dict(color=color, width=2),
                        )
                    )
            fig2.update_layout(
                title="Neurochemical Flux Over Night",
                xaxis_title="Time (hours)",
                yaxis_title="Relative Level",
                template="plotly_dark",
                height=360,
                paper_bgcolor="#0d0d14",
                plot_bgcolor="#12121e",
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"neuro_tab_{sim_key}")
        else:
            st.warning(
                "⚠️ Neurochemistry data unavailable. Check result JSON or CSV export."
            )

        if segments:
            lucid_df = pd.DataFrame(
                [
                    {
                        "time_hours": float(
                            s.get("start_time_hours", s.get("time_hours", 0.0)) or 0.0
                        ),
                        "lucidity_probability": float(
                            s.get("lucidity_probability", 0.0)
                        ),
                    }
                    for s in segments
                ]
            )
            fig_lucid = go.Figure()
            fig_lucid.add_trace(
                go.Scatter(
                    x=lucid_df["time_hours"],
                    y=lucid_df["lucidity_probability"],
                    mode="lines",
                    name="Lucidity",
                    line=dict(color=OKABE_ITO["yellow"], width=2),
                )
            )
            for ev in lucid_events:
                t0 = float(ev.get("time_hours", 0.0))
                fig_lucid.add_vline(
                    x=t0,
                    line_width=1,
                    line_dash="dash",
                    line_color=OKABE_ITO["purple"],
                )
            fig_lucid.update_layout(
                title="Lucidity Probability Timeline",
                xaxis_title="Time (hours)",
                yaxis_title="Lucidity probability",
                template="plotly_dark",
                height=300,
                paper_bgcolor="#0d0d14",
                plot_bgcolor="#12121e",
            )
            st.plotly_chart(
                fig_lucid,
                use_container_width=True,
                key=f"lucidity_tab_{sim_key}",
            )

    # ── Memory Graph ──────────────────────────────────────────────────────────
    with tab_mem:
        nodes = mem_graph.get("nodes", [])
        edges = mem_graph.get("edges", [])
        emotion_color = {
            "joy": OKABE_ITO["yellow"],
            "fear": OKABE_ITO["vermillion"],
            "sadness": OKABE_ITO["blue"],
            "anger": OKABE_ITO["orange"],
            "neutral": OKABE_ITO["black"],
            "surprise": OKABE_ITO["purple"],
        }
        if nodes:
            import math

            n = len(nodes)
            angle_step = 2 * math.pi / max(n, 1)
            pos = {
                nd["id"]: (math.cos(i * angle_step), math.sin(i * angle_step))
                for i, nd in enumerate(nodes)
            }

            edge_x, edge_y = [], []
            for e in edges:
                x0, y0 = pos.get(e.get("source", ""), (0, 0))
                x1, y1 = pos.get(e.get("target", ""), (0, 0))
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            node_x = [pos[nd["id"]][0] for nd in nodes]
            node_y = [pos[nd["id"]][1] for nd in nodes]
            node_colors = [
                emotion_color.get(nd.get("emotion", "neutral"), "#94a3b8")
                for nd in nodes
            ]
            node_sizes = [max(10, nd.get("activation", 0.5) * 40) for nd in nodes]
            node_labels = [nd.get("label", nd["id"])[:30] for nd in nodes]

            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(width=1, color="#2a2a3e"),
                    hoverinfo="none",
                )
            )
            fig3.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        line=dict(width=1, color="#fff"),
                    ),
                    text=node_labels,
                    textposition="top center",
                    hovertext=[
                        f"{nd.get('label','')} | {nd.get('emotion','')}" for nd in nodes
                    ],
                )
            )
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
            st.plotly_chart(fig3, use_container_width=True, key=f"memory_tab_{sim_key}")
        else:
            st.info(
                "Memory graph structure unavailable; showing activation heatmap when data exists."
            )

        heatmap_rows: list[dict[str, object]] = []
        csv_path = (
            Path("outputs")
            / str(result.get("id") or result.get("simulation_id") or "")
            / "memory_activations.csv"
        )
        if csv_path.exists():
            try:
                csv_df = pd.read_csv(csv_path)
                if {"time_hours", "node_label", "activation"}.issubset(csv_df.columns):
                    for row in csv_df.to_dict(orient="records"):
                        heatmap_rows.append(
                            {
                                "time_hours": float(row.get("time_hours", 0.0)),
                                "node_label": str(row.get("node_label", "")),
                                "activation": float(row.get("activation", 0.0)),
                            }
                        )
            except Exception as e:
                st.warning(f"Could not read memory_activations.csv: {e}")

        if not heatmap_rows:
            mem_activation = (
                result.get("memory_activation_series")
                or result.get("memory_activations")
                or result.get("memory_activation")
                or []
            )
            for snapshot in mem_activation:
                time_hours = float(snapshot.get("time_hours", 0.0))
                for node in snapshot.get("activations", []) or []:
                    heatmap_rows.append(
                        {
                            "time_hours": time_hours,
                            "node_label": str(node.get("label", node.get("id", ""))),
                            "activation": float(node.get("activation", 0.0)),
                        }
                    )

        if heatmap_rows:
            heat_df = pd.DataFrame(heatmap_rows)
            pivot_df = (
                heat_df.pivot_table(
                    index="node_label",
                    columns="time_hours",
                    values="activation",
                    aggfunc="max",
                )
                .fillna(0.0)
                .sort_index(axis=0)
            )
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=pivot_df.values,
                    x=[round(float(x), 3) for x in pivot_df.columns],
                    y=list(pivot_df.index),
                    colorscale="Viridis",
                    colorbar=dict(title="Activation"),
                )
            )
            fig_heat.update_layout(
                title="Memory Activation Heatmap",
                xaxis_title="snapshot_time_hours",
                yaxis_title="memory_node_label",
                template="plotly_dark",
                height=520,
                paper_bgcolor="#0d0d14",
                plot_bgcolor="#12121e",
            )
            st.plotly_chart(
                fig_heat, use_container_width=True, key=f"mem_heat_{sim_key}"
            )
        else:
            st.info(
                "No memory activation snapshots found in result or memory_activations.csv."
            )

    # ── Dream Narrative ───────────────────────────────────────────────────────
    with tab_dream:
        if segments:
            rem_segments = []
            scene_prefix_pattern = re.compile(
                r"^(scene:|scene description:|scene text:|/no_think|no_think)\s*",
                re.IGNORECASE,
            )

            def _clean_scene(raw: str) -> str:
                cleaned = " ".join(str(raw or "").split()).strip()
                for _ in range(6):
                    updated = scene_prefix_pattern.sub("", cleaned).strip()
                    if updated == cleaned:
                        break
                    cleaned = updated
                return cleaned

            def _clean_narrative(raw: str) -> str:
                cleaned = re.sub(r"(?is)<[^>]+>", " ", str(raw or ""))
                cleaned = " ".join(cleaned.split()).strip()
                if cleaned.lower().startswith("narrative:"):
                    cleaned = cleaned[len("narrative:") :].strip()
                return cleaned

            for i, seg in enumerate(segments):
                if str(seg.get("stage", "")) != "REM":
                    continue
                narrative = _clean_narrative(str(seg.get("narrative") or ""))
                scene_text = _clean_scene(str(seg.get("scene_description") or ""))
                if not narrative:
                    continue
                rem_segments.append(
                    {
                        "idx": i + 1,
                        "time": float(
                            seg.get("start_time_hours", seg.get("time_hours", 0.0))
                            or 0.0
                        ),
                        "emotion": str(seg.get("dominant_emotion", "neutral")),
                        "biz": float(
                            seg.get("bizarreness_score", seg.get("bizarreness", 0.0))
                        ),
                        "scene": scene_text,
                        "is_lucid": bool(seg.get("is_lucid", False)),
                        "mode": str(seg.get("generation_mode", "TEMPLATE")),
                        "narrative": narrative,
                    }
                )

            if rem_segments:
                st.caption(
                    "REM narrative viewer. Lucid segments are highlighted in gold."
                )
                cards: list[str] = []
                for seg in rem_segments:
                    border_color = "#d4af37" if seg["is_lucid"] else "#4c3f8f"
                    lucid_badge = " · [LUCID]" if seg["is_lucid"] else ""
                    emotion_text = html.escape(seg["emotion"])
                    mode_text = html.escape(seg["mode"])
                    narrative_text = html.escape(seg["narrative"])
                    scene_text = html.escape(seg["scene"])
                    cards.append(
                        f"""
                    <div style="background:#1a1a2e; border-left:4px solid {border_color};
                         border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.8rem;">
                      <div style="font-size:0.75rem; color:#9090c0; margin-bottom:0.3rem;">
                        REM Segment {seg["idx"]} · t={seg["time"]:.2f}h · Emotion: {emotion_text}
                        · Bizarreness: {seg["biz"]:.2f} · Mode: {mode_text}{lucid_badge}
                      </div>
                      <div style="color:#d8d8ef; font-size:0.95rem; line-height:1.6; margin-bottom:0.35rem;">
                        {narrative_text}
                      </div>
                      <div style="color:#aab0d8; font-size:0.82rem;">
                        Scene: {scene_text}
                      </div>
                    </div>
                    """
                    )
                st.markdown(
                    '<div style="max-height:540px; overflow-y:auto; padding-right:0.35rem;">'
                    + "".join(cards)
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No narrative segments found in simulation result.")
        else:
            st.info("No dream segments returned.")

    # ── Phase 5 static visualizations ─────────────────────────────────────────
    if (
        segments
        and plot_rem_episode_trend is not None
        and plot_affect_ratio_timeline is not None
        and plot_bizarreness_cortisol_scatter is not None
        and plot_per_cycle_architecture is not None
    ):
        st.markdown("---")
        st.markdown("### 📊 Static Sleep Analytics")
        c1, c2 = st.columns(2)

        with c1:
            _render_chart_with_exports(
                "REM Duration Trend",
                plot_rem_episode_trend(segments),
                key_prefix=f"rem_trend_{sim_key}",
            )
            _render_chart_with_exports(
                "Bizarreness vs Cortisol (stage-colored)",
                plot_bizarreness_cortisol_scatter(segments),
                key_prefix=f"biz_cort_{sim_key}",
            )
        with c2:
            _render_chart_with_exports(
                "Affect Ratio Timeline (rolling 30 min)",
                plot_affect_ratio_timeline(segments),
                key_prefix=f"affect_ratio_{sim_key}",
            )
            _render_chart_with_exports(
                "Per-cycle Architecture (stacked)",
                plot_per_cycle_architecture(segments),
                key_prefix=f"cycle_arch_{sim_key}",
            )
