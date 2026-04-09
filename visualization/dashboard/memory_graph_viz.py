import streamlit as st
import networkx as nx
import plotly.graph_objects as go

from core.models.memory_graph import MemoryGraph


def render_memory_graph(graph: MemoryGraph) -> None:
    """Render a force-directed memory association graph using Plotly."""

    g = graph.to_networkx()
    if g.number_of_nodes() == 0:
        st.info("No memory fragments to display yet.")
        return

    pos = nx.spring_layout(g, k=0.5, iterations=50)

    x_nodes = [pos[n][0] for n in g.nodes()]
    y_nodes = [pos[n][1] for n in g.nodes()]
    node_text = [g.nodes[n].get("label", n) for n in g.nodes()]

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
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(size=10, color="#1f77b4"),
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        title="Memory Association Graph",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    st.plotly_chart(fig, use_container_width=True)
