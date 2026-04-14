from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimeModel:
    """Simple time model for mapping discrete steps to simulated hours."""

    start_time_hours: float
    dt_minutes: float
    duration_hours: float

    def __post_init__(self) -> None:
        self.current_time_hours: float = self.start_time_hours
        self.dt_hours: float = self.dt_minutes / 60.0
        self._end_time_hours: float = self.start_time_hours + self.duration_hours

    @property
    def has_next_step(self) -> bool:
        return self.current_time_hours < self._end_time_hours

    def step(self) -> float:
        self.current_time_hours += self.dt_hours
        return self.current_time_hours
