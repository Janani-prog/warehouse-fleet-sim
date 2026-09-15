"""Builds the small set of concrete action candidates offered to the LLM for
the current tick, from live World state. The agent never invents a zone/
robot/order id itself - it only ever confirms or rejects one pre-computed
candidate per action type, and agent/executor.py re-validates that choice
against current state before calling into the simulator. This keeps an 8B
prompt-only model from having to do exact-id bookkeeping it isn't reliable
at, while still producing a real, executable target.
"""

from __future__ import annotations

from dataclasses import dataclass

from sim.entities import OrderStatus
from sim.world import World


@dataclass(frozen=True)
class ActionCandidates:
    zone_id: str | None
    zone_queue_depth: int
    zone_density: int
    order_id: int | None
    order_robot_id: int | None
    robot_id: int | None
    robot_stuck_ticks: int

    def describe(self) -> str:
        lines = []
        if self.zone_id is not None:
            lines.append(
                f"- For THROTTLE_ZONE_TRAFFIC: candidate zone \"{self.zone_id}\" "
                f"(queue depth {self.zone_queue_depth}, robot density {self.zone_density}, "
                "the most loaded zone this tick)."
            )
        else:
            lines.append("- For THROTTLE_ZONE_TRAFFIC: no zone currently loaded enough to be a candidate.")
        if self.order_id is not None:
            lines.append(
                f"- For REASSIGN_TASK: candidate order {self.order_id} "
                f"(currently assigned to robot {self.order_robot_id}, in the most loaded zone)."
            )
        else:
            lines.append("- For REASSIGN_TASK: no assigned-but-not-yet-picked-up order currently available to reassign.")
        if self.robot_id is not None:
            lines.append(
                f"- For REPLAN_ROUTE: candidate robot {self.robot_id} "
                f"(blocked for {self.robot_stuck_ticks} consecutive ticks)."
            )
        else:
            lines.append("- For REPLAN_ROUTE: no robot currently blocked long enough to be a candidate.")
        return "\n".join(lines)


def build_candidates(world: World) -> ActionCandidates:
    queue_depth, density = world.zone_stats()
    zone_ids = world.warehouse.zone_ids()
    load = {z: queue_depth[z] + density[z] for z in zone_ids}
    busiest_zone = max(zone_ids, key=lambda z: load[z])
    if load[busiest_zone] == 0:
        busiest_zone = None

    order_candidate = None
    order_robot_id = None
    if busiest_zone is not None:
        for order in world.orders.values():
            if order.status != OrderStatus.ASSIGNED:
                continue
            robot = next((r for r in world.robots if r.id == order.assigned_robot), None)
            if robot is not None and world.warehouse.zone_id(*robot.pos) == busiest_zone:
                order_candidate = order.id
                order_robot_id = robot.id
                break
        if order_candidate is None:
            # fall back to any assigned-not-yet-picked-up order, anywhere
            for order in world.orders.values():
                if order.status == OrderStatus.ASSIGNED:
                    order_candidate = order.id
                    order_robot_id = order.assigned_robot
                    break

    stuck_robot = max(world.robots, key=lambda r: r.stuck_ticks, default=None)
    robot_id = stuck_robot.id if stuck_robot and stuck_robot.stuck_ticks > 0 else None
    robot_stuck_ticks = stuck_robot.stuck_ticks if robot_id is not None else 0

    return ActionCandidates(
        zone_id=busiest_zone,
        zone_queue_depth=queue_depth.get(busiest_zone, 0) if busiest_zone else 0,
        zone_density=density.get(busiest_zone, 0) if busiest_zone else 0,
        order_id=order_candidate,
        order_robot_id=order_robot_id,
        robot_id=robot_id,
        robot_stuck_ticks=robot_stuck_ticks,
    )
