import streamlit as st
import networkx as nx
import plotly.graph_objects as go

from core.models.memory_graph import MemoryGraph


def render_memory_graph(graph: MemoryGraph) -> None:
    """Render a force-directed memory association graph with emotional coloring."""

    g = graph.to_networkx()
    if g.number_of_nodes() == 0:
        st.info("No memory fragments to display yet.")
        return

    pos = nx.spring_layout(g, k=0.6, iterations=60)

    x_nodes = [pos[n][0] for n in g.nodes()]
    y_nodes = [pos[n][1] for n in g.nodes()]
    node_text = [g.nodes[n].get("label", n) for n in g.nodes()]
    emotions = [g.nodes[n].get("emotion", "neutral") for n in g.nodes()]

    color_map = {
        "joy": "#facc15",
        "fear": "#f97373",
        "sadness": "#60a5fa",
        "anger": "#fb7185",
        "surprise": "#a855f7",
        "disgust": "#4ade80",
        "neutral": "#e5e7eb",
    }
    node_colors = [color_map.get(e, "#e5e7eb") for e in emotions]

    edge_x = []
    edge_y = []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="rgba(148,163,184,0.6)"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(size=12, color=node_colors, line=dict(width=1, color="#0f172a")),
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        title="Memory Association Graph",
        template="plotly_dark",
        paper_bgcolor="rgba(5,10,24,1)",
        plot_bgcolor="rgba(5,10,24,1)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    st.plotly_chart(fig, use_container_width=True)
