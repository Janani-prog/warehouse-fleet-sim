"""Pluggable path-planning policy interface the simulator drives robot
movement with. `World` defaults to classical A* (ml.astar) but accepts any
callable with this signature — e.g. the learned planner's rollout — so a
learned-vs-classical simulator run is a one-line swap, not a rewrite."""

from __future__ import annotations

from typing import Callable

from sim.grid import Warehouse

Cell = tuple[int, int]
PlannerFn = Callable[[Warehouse, Cell, Cell], "list[Cell] | None"]

# Given each robot's current position and desired next cell, arbitrate
# conflicts and return each robot's actual next cell for the tick - the
# traffic/collision-avoidance layer (classical `ml.traffic.resolve` by
# default, or a learned equivalent).
AvoidanceFn = Callable[[Warehouse, dict[int, Cell], dict[int, Cell]], dict[int, Cell]]
