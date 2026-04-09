from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Field


class EventType(str, Enum):
    SLEEP_STAGE_UPDATED = "sleep_stage_updated"
    NEUROCHEMISTRY_UPDATED = "neurochemistry_updated"
    MEMORY_REPLAY_EVENT = "memory_replay_event"
    DREAM_SEGMENT_GENERATED = "dream_segment_generated"
    LUCIDITY_UPDATED = "lucidity_updated"
    PHENOMENOLOGY_UPDATED = "phenomenology_updated"


class Event(BaseModel):
    """Typed event on the internal message bus."""

    type: EventType
    payload: Dict[str, Any]
    timestamp_hours: float = Field(..., description="Simulation time in hours.")


EventHandler = Callable[[Event], None]


@dataclass
class Subscription:
    event_type: EventType
    handler: EventHandler


class EventBus:
    """In-process typed event bus.

    Agents register handlers for specific event types and publish events
    during the simulation. This abstracts orchestration tools and/or
    external transports like Redis streams.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[EventHandler]] = {}

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        for handler in self._subscribers.get(event.type, []):
            handler(event)


class AgentActivityLogger:
    """Lightweight logger for agent/event activity over time.

    Used by the dashboard to build an agent activity heatmap.
    """

    def __init__(self, bus: EventBus) -> None:
        self._events: List[Event] = []
        self._bus = bus
        for etype in EventType:
            bus.subscribe(etype, self._handle_event)

    def _handle_event(self, event: Event) -> None:
        self._events.append(event)

    @property
    def events(self) -> List[Event]:
        return list(self._events)
