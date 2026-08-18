"""Classical A* path planning on the warehouse grid (4-connectivity, unit cost)."""

from __future__ import annotations

import heapq
import itertools

from sim.grid import Warehouse

Cell = tuple[int, int]


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(warehouse: Warehouse, start: Cell, goal: Cell) -> list[Cell] | None:
    """Shortest free-space path from start to goal, inclusive of both
    endpoints. Returns None if start/goal is blocked or goal is unreachable."""
    if not warehouse.is_free(*start) or not warehouse.is_free(*goal):
        return None
    if start == goal:
        return [start]

    counter = itertools.count()
    open_heap: list[tuple[int, int, Cell]] = [(_manhattan(start, goal), next(counter), start)]
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, int] = {start: 0}
    closed: set[Cell] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)

        for neighbor in warehouse.neighbors(*current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + _manhattan(neighbor, goal)
                heapq.heappush(open_heap, (f, next(counter), neighbor))

    return None
