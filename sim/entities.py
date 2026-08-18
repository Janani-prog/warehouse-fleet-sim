"""Robot and Order entity state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RobotState(str, Enum):
    IDLE = "idle"
    MOVING = "moving"
    BLOCKED = "blocked"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    PICKED_UP = "picked_up"
    COMPLETED = "completed"


@dataclass
class Robot:
    id: int
    x: int
    y: int
    state: RobotState = RobotState.IDLE
    order_id: int | None = None
    path: list[tuple[int, int]] = field(default_factory=list)
    stuck_ticks: int = 0

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass
class Order:
    id: int
    origin: tuple[int, int]
    destination: tuple[int, int]
    arrival_tick: int
    status: OrderStatus = OrderStatus.PENDING
    assigned_robot: int | None = None
    assign_tick: int | None = None
    pickup_tick: int | None = None
    dropoff_tick: int | None = None

    @property
    def wait_ticks(self) -> int | None:
        """Ticks from arrival to pickup (queueing delay), once picked up."""
        if self.pickup_tick is None:
            return None
        return self.pickup_tick - self.arrival_tick
