"""Order arrival process: Poisson arrivals with random origin/destination."""

from __future__ import annotations

from typing import Callable

import numpy as np

from sim.entities import Order
from sim.grid import Warehouse


class OrderGenerator:
    def __init__(
        self,
        warehouse: Warehouse,
        rng: np.random.Generator,
        arrival_rate: float | Callable[[int], float],
    ):
        """arrival_rate: expected number of new orders per tick (Poisson
        lambda), either constant or a function of the current tick - the
        latter is what scripted scenarios (e.g. an engineered congestion
        spike) use to vary load over the course of a run."""
        self.warehouse = warehouse
        self.rng = rng
        self.arrival_rate = arrival_rate
        self._next_id = 0

    def rate_at(self, tick: int) -> float:
        return self.arrival_rate(tick) if callable(self.arrival_rate) else self.arrival_rate

    def step(self, tick: int) -> list[Order]:
        n_new = int(self.rng.poisson(self.rate_at(tick)))
        orders = []
        for _ in range(n_new):
            origin = tuple(
                self.warehouse.pickup_points[self.rng.integers(len(self.warehouse.pickup_points))]
            )
            destination = tuple(
                self.warehouse.dropoff_points[
                    self.rng.integers(len(self.warehouse.dropoff_points))
                ]
            )
            orders.append(
                Order(
                    id=self._next_id,
                    origin=origin,
                    destination=destination,
                    arrival_tick=tick,
                )
            )
            self._next_id += 1
        return orders
