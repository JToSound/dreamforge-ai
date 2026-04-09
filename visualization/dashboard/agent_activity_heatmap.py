import streamlit as st
import plotly.graph_objects as go

from typing import List

from core.models.dream_segment import DreamSegment


AGENTS = [
    "SleepCycleAgent",
    "NeurochemistryAgent",
    "MemoryConsolidationAgent",
    "DreamConstructorAgent",
    "MetacognitiveAgent",
    "PhenomenologyReporter",
]


def render_agent_activity_heatmap(segments: List[DreamSegment]) -> None:
    """Approximate agent activity heatmap based on dream segments.

    For now we approximate:
    - Sleep & neuro agents active whenever time advances.
    - Memory agent active when a segment references memories.
    - Dream/meta/phenom agents active whenever a segment exists.
    """

    if not segments:
        st.info("No dream segments generated for this run.")
        return

    times = [seg.end_time_hours for seg in segments]
    agents = AGENTS

    matrix = []
    for seg in segments:
        row = []
        has_memory = bool(seg.active_memory_ids)
        for agent in agents:
            if agent in ("SleepCycleAgent", "NeurochemistryAgent"):
                row.append(1)
            elif agent == "MemoryConsolidationAgent":
                row.append(1 if has_memory else 0)
            else:
                row.append(1)
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=agents,
            y=[f"{t:.2f}h" for t in times],
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        title="Agent Activity Heatmap (approximate)",
        xaxis_title="Agent",
        yaxis_title="Time (hours)",
    )

    st.plotly_chart(fig, use_container_width=True)
