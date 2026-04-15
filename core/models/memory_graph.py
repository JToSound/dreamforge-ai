from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    EMOTIONAL_SCHEMA = "emotional_schema"


class EmotionLabel(str, Enum):
    NEUTRAL = "neutral"
    JOY = "joy"
    FEAR = "fear"
    SADNESS = "sadness"
    ANGER = "anger"
    SURPRISE = "surprise"
    DISGUST = "disgust"


class MemoryNodeModel(BaseModel):
    """Pydantic model describing a memory fragment node in the graph.

    Nodes are fragments of episodic or semantic memory with emotional tagging,
    recency, and activation strength to support hippocampal replay simulation
    and selective forgetting. [cite:18][cite:28]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique node identifier."
    )
    label: str = Field(..., description="Human-readable label or short description.")
    memory_type: MemoryType = Field(
        ..., description="Type of memory (episodic/semantic/schema)."
    )

    activation: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Current activation/availability of the memory fragment.",
    )
    salience: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance weighting; affects replay probability.",
    )

    emotion: EmotionLabel = Field(
        default=EmotionLabel.NEUTRAL, description="Dominant emotion label."
    )
    arousal: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Emotional arousal intensity.",
    )

    recency_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Time since encoding (hours). Used for recency-based replay/forgetting.",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Arbitrary tags (people, places, themes, etc.).",
    )


class MemoryEdgeModel(BaseModel):
    """Pydantic model for an associative edge between memory nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_id: str
    target_id: str

    weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of association.",
    )
    emotion_alignment: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Similarity of emotional tone between connected fragments.",
    )
    context_overlap: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Contextual overlap (e.g., shared location or people).",
    )


@dataclass
class ReplaySequence:
    """Represents one hippocampal replay sequence (SWR-like event)."""

    id: str
    node_ids: list[str]
    total_weight: float
    dominant_emotion: EmotionLabel


@dataclass(frozen=True)
class MemoryActivationSnapshot:
    """Immutable snapshot of graph activations at one simulation timestamp."""

    time_hours: float
    stage: str
    activations: dict[str, float]


class MemoryGraph:
    """High-level wrapper around a NetworkX graph representing memory.

    Supports:
      - Fragment encoding from user input.
      - Emotionally tagged, recency-aware associations.
      - Hippocampal sharp-wave ripple (SWR) replay as biased random walks.
      - Selective forgetting via salience decay and pruning. [cite:18][cite:28]
    """

    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        # Log of replay events for visualization and export
        self.replay_event_log: list[dict[str, object]] = []
        # Chronological activation snapshots captured during simulation.
        self.activation_snapshots: list[MemoryActivationSnapshot] = []

    # ------------------------------------------------------------------
    # Node and edge management
    # ------------------------------------------------------------------

    def add_memory(self, node: MemoryNodeModel) -> str:
        self._g.add_node(
            node.id,
            label=node.label,
            memory_type=node.memory_type.value,
            activation=node.activation,
            salience=node.salience,
            emotion=node.emotion.value,
            arousal=node.arousal,
            recency_hours=node.recency_hours,
            tags=node.tags,
        )
        return node.id

    def add_association(self, edge: MemoryEdgeModel) -> None:
        if edge.source_id not in self._g or edge.target_id not in self._g:
            raise ValueError("Both source_id and target_id must exist in the graph.")
        self._g.add_edge(
            edge.source_id,
            edge.target_id,
            weight=edge.weight,
            emotion_alignment=edge.emotion_alignment,
            context_overlap=edge.context_overlap,
        )

    def encode_from_user_input(
        self,
        text: str,
        emotion: EmotionLabel,
        tags: Optional[list[str]] = None,
        is_episodic: bool = True,
    ) -> str:
        node = MemoryNodeModel(
            label=text[:128],
            memory_type=MemoryType.EPISODIC if is_episodic else MemoryType.SEMANTIC,
            emotion=emotion,
            activation=0.8,
            salience=0.8,
            arousal=0.7,
            tags=tags or [],
        )
        return self.add_memory(node)

    # ------------------------------------------------------------------
    # Salience decay and forgetting
    # ------------------------------------------------------------------

    def decay_salience(
        self,
        dt_hours: float,
        base_decay_rate: float = 0.02,
        emotion_protection: float = 0.5,
    ) -> None:
        for node_id, data in list(self._g.nodes(data=True)):
            arousal = float(data.get("arousal", 0.5))
            salience = float(data.get("salience", 0.5))
            activation = float(data.get("activation", 0.5))
            recency = float(data.get("recency_hours", 0.0))

            effective_decay = base_decay_rate * (1.0 - emotion_protection * arousal)
            salience_new = max(0.0, salience * math.exp(-effective_decay * dt_hours))
            activation_new = max(
                0.0, activation * math.exp(-effective_decay * dt_hours)
            )

            self._g.nodes[node_id]["salience"] = salience_new
            self._g.nodes[node_id]["activation"] = activation_new
            self._g.nodes[node_id]["recency_hours"] = recency + dt_hours

    def prune_low_salience(self, threshold: float = 0.1) -> None:
        to_remove = [
            node_id
            for node_id, data in self._g.nodes(data=True)
            if float(data.get("salience", 0.0)) < threshold
        ]
        self._g.remove_nodes_from(to_remove)

    # ------------------------------------------------------------------
    # Replay (SWR-like events)
    # ------------------------------------------------------------------

    def _node_weight(self, node_id: str) -> float:
        data = self._g.nodes[node_id]
        salience = float(data.get("salience", 0.5))
        activation = float(data.get("activation", 0.5))
        return 0.5 * (salience + activation)

    def sample_replay_sequence(
        self,
        max_length: int = 10,
        start_bias_tags: Optional[list[str]] = None,
        current_time_hours: float = 0.0,
        pulse_height: float = 0.35,
        propagation_delay_s: float = 0.08,
        decay_tau_hours: float = 0.05,
    ) -> Optional[ReplaySequence]:
        if self._g.number_of_nodes() == 0:
            return None

        node_ids = list(self._g.nodes())
        weights = []
        for nid in node_ids:
            w = self._node_weight(nid)
            if start_bias_tags:
                tags = self._g.nodes[nid].get("tags", [])
                overlap = len(set(start_bias_tags) & set(tags))
                w *= 1.0 + 0.5 * overlap
            weights.append(w)

        weights_arr = np.array(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()
        start = str(np.random.choice(node_ids, p=weights_arr))

        path = [start]
        current = start
        total_weight = self._node_weight(start)

        for _ in range(max_length - 1):
            neighbors = list(self._g.successors(current))
            if not neighbors:
                break
            edge_weights = []
            for n in neighbors:
                edge_data = self._g.get_edge_data(current, n)
                if not edge_data:
                    edge_weights.append(1.0)
                else:
                    max_w = max(e.get("weight", 0.5) for e in edge_data.values())
                    edge_weights.append(max_w)

            w_arr = np.array(edge_weights, dtype=float)
            w_arr = w_arr / w_arr.sum()
            current = str(np.random.choice(neighbors, p=w_arr))
            path.append(current)
            total_weight += self._node_weight(current)

        dominant_emotion = self._dominant_emotion(path)
        seq = ReplaySequence(
            id=str(uuid.uuid4()),
            node_ids=path,
            total_weight=total_weight,
            dominant_emotion=dominant_emotion,
        )

        # Automatically apply replay pulse for downstream dynamics/visualization
        try:
            self.apply_replay_pulse(
                seq,
                pulse_height=pulse_height,
                propagation_delay_s=propagation_delay_s,
                decay_tau_hours=decay_tau_hours,
                current_time_hours=current_time_hours,
            )
        except Exception:
            # Do not raise from sampling; ensure sampling remains robust
            pass
        return seq

    def apply_replay_effect(self, replay: ReplaySequence, spike: float = 0.25) -> None:
        """Apply an activation spike to nodes participating in a replay sequence.

        This simulates the transient hippocampal activation caused by SWR events.
        Nodes receive an additive activation boost (clipped to 1.0) and a small
        salience reinforcement proportional to their activation.
        """
        if not replay or not replay.node_ids:
            return
        for nid in replay.node_ids:
            if nid not in self._g:
                continue
            cur_act = float(self._g.nodes[nid].get("activation", 0.5))
            new_act = min(1.0, cur_act + float(spike))
            self._g.nodes[nid]["activation"] = new_act

            # small salience bump to reflect consolidation
            cur_sal = float(self._g.nodes[nid].get("salience", 0.5))
            self._g.nodes[nid]["salience"] = min(1.0, cur_sal + 0.02 * float(spike))

    def apply_replay_pulse(
        self,
        sequence: ReplaySequence,
        pulse_height: float = 0.35,
        propagation_delay_s: float = 0.08,
        decay_tau_hours: float = 0.05,
        current_time_hours: float = 0.0,
    ) -> None:
        """Apply a decaying pulse sequence across nodes and log the event.

        Each subsequent node in the sequence receives a pulse reduced by a
        factor (0.85 ** i) to simulate propagation attenuation.
        """
        if not sequence or not sequence.node_ids:
            return

        increments = []
        for i, nid in enumerate(sequence.node_ids):
            if nid not in self._g:
                increments.append(0.0)
                continue
            cur_act = float(self._g.nodes[nid].get("activation", 0.5))
            inc = float(pulse_height) * (0.85**i)
            new_act = min(1.0, cur_act + inc)
            self._g.nodes[nid]["activation"] = new_act
            # store last pulse time for visualization hooks
            self._g.nodes[nid]["last_pulse_time"] = float(current_time_hours)
            # small salience bump
            cur_sal = float(self._g.nodes[nid].get("salience", 0.5))
            self._g.nodes[nid]["salience"] = min(1.0, cur_sal + 0.01 * inc)
            increments.append(inc)

        # record event for export/visualization
        self.replay_event_log.append(
            {
                "id": sequence.id,
                "time_hours": float(current_time_hours),
                "node_ids": list(sequence.node_ids),
                "increments": increments,
                "pulse_height": float(pulse_height),
                "dominant_emotion": (
                    sequence.dominant_emotion.value
                    if hasattr(sequence.dominant_emotion, "value")
                    else str(sequence.dominant_emotion)
                ),
                "total_weight": float(sequence.total_weight),
            }
        )

    def decay_activations(self, dt_hours: float, decay_tau_hours: float = 0.05) -> None:
        """Exponentially decay activations across all nodes.

        Activation is decayed then clamped to [salience * 0.1, 1.0].
        """
        if dt_hours <= 0.0:
            return
        factor = math.exp(-float(dt_hours) / float(decay_tau_hours))
        for node_id, data in list(self._g.nodes(data=True)):
            act = float(data.get("activation", 0.0))
            sal = float(data.get("salience", 0.0))
            new_act = act * factor
            floor = sal * 0.1
            new_act = max(floor, min(1.0, new_act))
            self._g.nodes[node_id]["activation"] = new_act

    def capture_memory_snapshot(
        self,
        time_hours: float,
        stage: str,
    ) -> MemoryActivationSnapshot:
        """Capture current node activations as an immutable snapshot."""
        activations: dict[str, float] = {
            str(node): float(data.get("activation", 0.0))
            for node, data in self._g.nodes(data=True)
        }
        snapshot = MemoryActivationSnapshot(
            time_hours=float(time_hours),
            stage=str(stage),
            activations=activations,
        )
        self.activation_snapshots.append(snapshot)
        return snapshot

    def _dominant_emotion(self, node_ids: Iterable[str]) -> EmotionLabel:
        counts = {label: 0.0 for label in EmotionLabel}
        for nid in node_ids:
            data = self._g.nodes[nid]
            emo_str = data.get("emotion", EmotionLabel.NEUTRAL.value)
            emo = EmotionLabel(emo_str)
            counts[emo] += float(data.get("salience", 0.5))
        return max(counts.items(), key=lambda kv: kv[1])[0]

    # ------------------------------------------------------------------
    # Query/export helpers
    # ------------------------------------------------------------------

    def to_networkx(self) -> nx.MultiDiGraph:
        return self._g

    def to_json_serializable(self) -> dict[str, Any]:
        nodes = []
        for node_id, data in self._g.nodes(data=True):
            item = {"id": node_id}
            item.update(data)
            nodes.append(item)

        edges = []
        for u, v, key, data in self._g.edges(keys=True, data=True):
            edge_item = {"source": u, "target": v}
            edge_item.update(data)
            edges.append(edge_item)

        return {
            "nodes": nodes,
            "edges": edges,
            "replay_events": list(self.replay_event_log),
            "activation_snapshots": [
                {
                    "time_hours": s.time_hours,
                    "stage": s.stage,
                    "activations": s.activations,
                }
                for s in self.activation_snapshots
            ],
        }
