"""Order arrival process: Poisson arrivals with random origin/destination."""

from __future__ import annotations

import numpy as np

from sim.entities import Order
from sim.grid import Warehouse


class OrderGenerator:
    def __init__(self, warehouse: Warehouse, rng: np.random.Generator, arrival_rate: float):
        """arrival_rate: expected number of new orders per tick (Poisson lambda)."""
        self.warehouse = warehouse
        self.rng = rng
        self.arrival_rate = arrival_rate
        self._next_id = 0

    def step(self, tick: int) -> list[Order]:
        n_new = int(self.rng.poisson(self.arrival_rate))
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
