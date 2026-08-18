"""Simulator core: tick-based world tying together the warehouse layout,
robots, order generation, movement, and telemetry logging."""

from __future__ import annotations

import math

import numpy as np

from ml.astar import astar
from sim.entities import Order, OrderStatus, Robot, RobotState
from sim.grid import Warehouse, generate_default_layout
from sim.order_generator import OrderGenerator
from sim.policies import PlannerFn
from sim.telemetry import TelemetryLogger

NEAR_MISS_DISTANCE = 1.5  # Euclidean distance threshold for a logged near-miss
STUCK_THRESHOLD = 3  # ticks a robot can be blocked before forcing a side-step

# num_robots/order_rate defaults are tuned so fleet throughput comfortably
# exceeds arrival rate under A*-driven movement + nearest-idle assignment
# (see CLAUDE.md Current Status for the sweep this came from) - the point is
# a demo run that stays caught up, not one with a permanently growing queue.


class World:
    def __init__(
        self,
        seed: int,
        num_robots: int = 8,
        order_rate: float = 0.15,
        warehouse: Warehouse | None = None,
        planner: PlannerFn = astar,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.warehouse = warehouse or generate_default_layout()
        self.order_rate = order_rate
        self.planner = planner

        spawn = self.warehouse.spawn_points
        self.robots: list[Robot] = [
            Robot(id=i, x=spawn[i % len(spawn)][0], y=spawn[i % len(spawn)][1])
            for i in range(num_robots)
        ]
        self.orders: dict[int, Order] = {}
        self.order_gen = OrderGenerator(self.warehouse, self.rng, order_rate)
        self.telemetry = TelemetryLogger()
        self.tick_count = 0

    def _target_for(self, robot: Robot) -> tuple[int, int] | None:
        if robot.order_id is None:
            return None
        order = self.orders[robot.order_id]
        if order.status == OrderStatus.ASSIGNED:
            return order.origin
        if order.status == OrderStatus.PICKED_UP:
            return order.destination
        return None

    def _assign_orders(self, tick: int) -> None:
        idle_robots = [r for r in self.robots if r.order_id is None]
        pending = [o for o in self.orders.values() if o.status == OrderStatus.PENDING]
        for order in pending:
            if not idle_robots:
                break
            distances = [
                abs(r.x - order.origin[0]) + abs(r.y - order.origin[1]) for r in idle_robots
            ]
            nearest = idle_robots[int(np.argmin(distances))]
            order.status = OrderStatus.ASSIGNED
            order.assigned_robot = nearest.id
            order.assign_tick = tick
            nearest.order_id = order.id
            nearest.state = RobotState.MOVING
            idle_robots.remove(nearest)
            self.telemetry.log_event(tick, "order_assigned", order_id=order.id, robot_id=nearest.id)

    def _move_robots(self, tick: int) -> None:
        # Recompute a robot's cached path only when its target changed (new
        # assignment, or pickup->dropoff transition) or it has none yet.
        for robot in self.robots:
            target = self._target_for(robot)
            if target is None or robot.pos == target:
                robot.path = []
                continue
            if not robot.path or robot.path[-1] != target:
                full_path = self.planner(self.warehouse, robot.pos, target)
                robot.path = list(full_path[1:]) if full_path else []

        # Idle robots hold position; robots with a target follow their cached
        # path one cell per tick, yielding to whichever robot already claimed
        # the next cell this tick (full conflict-free avoidance is M3's job).
        occupied = {r.pos for r in self.robots}
        for robot in self.robots:
            target = self._target_for(robot)
            if target is None:
                robot.state = RobotState.IDLE
                robot.stuck_ticks = 0
                continue
            if robot.pos == target:
                robot.state = RobotState.MOVING
                robot.stuck_ticks = 0
                continue

            occupied.discard(robot.pos)
            if robot.path and robot.path[0] not in occupied:
                next_cell = robot.path.pop(0)
                robot.x, robot.y = next_cell
                robot.state = RobotState.MOVING
                robot.stuck_ticks = 0
            else:
                robot.stuck_ticks += 1
                if robot.stuck_ticks >= STUCK_THRESHOLD:
                    # Deadlock/livelock breaker: two (or more) robots whose
                    # cached paths cross will otherwise wait on each other
                    # forever, since neither ever frees its cell. Force a
                    # side-step onto any free, unclaimed neighbor and drop
                    # the cached path so it's replanned fresh next tick. This
                    # is a minimal liveness guarantee, not real avoidance -
                    # M3's ORCA/learned traffic policy replaces it.
                    alternatives = [c for c in self.warehouse.neighbors(*robot.pos) if c not in occupied]
                    if alternatives:
                        robot.x, robot.y = tuple(alternatives[self.rng.integers(len(alternatives))])
                        robot.path = []
                        robot.state = RobotState.MOVING
                        robot.stuck_ticks = 0
                        self.telemetry.log_event(tick, "deadlock_broken", robot_id=robot.id)
                    else:
                        robot.state = RobotState.BLOCKED
                else:
                    robot.state = RobotState.BLOCKED
            occupied.add(robot.pos)

    def _resolve_pickups_dropoffs(self, tick: int) -> None:
        for robot in self.robots:
            if robot.order_id is None:
                robot.state = RobotState.IDLE
                continue
            order = self.orders[robot.order_id]
            if order.status == OrderStatus.ASSIGNED and robot.pos == order.origin:
                order.status = OrderStatus.PICKED_UP
                order.pickup_tick = tick
                self.telemetry.log_event(tick, "order_picked_up", order_id=order.id, robot_id=robot.id)
            elif order.status == OrderStatus.PICKED_UP and robot.pos == order.destination:
                order.status = OrderStatus.COMPLETED
                order.dropoff_tick = tick
                robot.order_id = None
                robot.state = RobotState.IDLE
                self.telemetry.log_event(tick, "order_completed", order_id=order.id, robot_id=robot.id)

    def _log_telemetry(self, tick: int) -> None:
        positions = np.array([r.pos for r in self.robots], dtype=float)
        if len(positions) >= 2:
            diffs = positions[:, None, :] - positions[None, :, :]
            dists = np.sqrt((diffs**2).sum(axis=-1))
            iu = np.triu_indices(len(positions), k=1)
            pair_dists = dists[iu]
            min_dist = float(pair_dists.min())
            near_miss = int((pair_dists <= NEAR_MISS_DISTANCE).sum())
        else:
            min_dist = math.nan
            near_miss = 0

        active_orders = sum(
            1 for o in self.orders.values() if o.status != OrderStatus.COMPLETED
        )
        self.telemetry.log_tick(tick, min_dist, near_miss, active_orders)

        for robot in self.robots:
            self.telemetry.log_robot(tick, robot.id, robot.x, robot.y, robot.state.value, robot.order_id)

        queue_depth: dict[str, int] = {z: 0 for z in self.warehouse.zone_ids()}
        density: dict[str, int] = {z: 0 for z in self.warehouse.zone_ids()}
        for o in self.orders.values():
            if o.status in (OrderStatus.PENDING, OrderStatus.ASSIGNED):
                queue_depth[self.warehouse.zone_id(*o.origin)] += 1
        for r in self.robots:
            density[self.warehouse.zone_id(*r.pos)] += 1
        for zone_id in self.warehouse.zone_ids():
            self.telemetry.log_zone(tick, zone_id, queue_depth[zone_id], density[zone_id])

    def tick(self) -> None:
        tick = self.tick_count
        for order in self.order_gen.step(tick):
            self.orders[order.id] = order
            self.telemetry.log_event(tick, "order_arrived", order_id=order.id)
        self._assign_orders(tick)
        self._move_robots(tick)
        self._resolve_pickups_dropoffs(tick)
        self._log_telemetry(tick)
        self.tick_count += 1

    def run(self, num_ticks: int) -> None:
        for _ in range(num_ticks):
            self.tick()

    def manifest(self) -> dict:
        return {
            "seed": self.seed,
            "num_robots": len(self.robots),
            "order_rate": self.order_rate,
            "num_ticks": self.tick_count,
            "warehouse": {
                "width": self.warehouse.width,
                "height": self.warehouse.height,
                "zone_size": list(self.warehouse.zone_size),
            },
        }
