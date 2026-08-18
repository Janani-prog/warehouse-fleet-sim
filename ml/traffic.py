"""Classical traffic/collision avoidance: a simplified, grid-discrete
velocity-obstacle variant (the backlog explicitly permits this in place of
literal continuous-space ORCA, since the simulator is grid-based).

Each robot proposes a desired next cell (from its cached A* path); this
module arbitrates between robots' desired cells each tick so that no two
robots ever end up in the same cell (a vertex conflict) or swap cells
head-on (an edge conflict) - the discrete analogue of ORCA's reciprocal
velocity obstacles. A robot that loses a conflict simply yields (velocity 0
for the tick, i.e. it stays put) rather than dodging sideways: an earlier
version had losers side-step to *any* free neighbor, chosen goal-blind, and
at the warehouse's single-cell aisle gaps that caused robots to shuffle back
and forth and repeatedly replan instead of just waiting their turn - a
self-inflicted jam, not real congestion. `World`'s STUCK_THRESHOLD +
obstacle-aware replan (see sim/world.py) is what actually gets a robot
unstuck if waiting alone doesn't resolve within a few ticks, and it does so
with a real A* query around the blockage instead of a blind guess.

3+ robot rotations through a cycle (A -> B's cell, B -> C's cell, C -> A's
cell, all simultaneously) are intentionally allowed, matching standard MAPF
convention: at tick boundaries every robot ends up in a distinct cell, so no
physical collision occurs even though a literal continuous-time reading
might look like agents "passing through" each other.
"""

from __future__ import annotations

from collections import defaultdict

from sim.grid import Warehouse

Cell = tuple[int, int]


def arbitrate(positions: dict[int, Cell], desired: dict[int, Cell]) -> dict[int, Cell]:
    """The safety-net conflict resolver: given what every robot wants to do
    this tick, return what they're actually allowed to do, guaranteed free
    of vertex and edge/swap collisions. Pure priority arbitration, no
    warehouse geometry involved - this is deliberately the single choke
    point both the classical and learned avoidance policies route through,
    so a learned model's predictions can never actually cause a collision
    (same "propose, then validate deterministically" pattern as the action
    executor in agent/, applied here instead of to LLM output)."""
    priority = {rid: rid for rid in positions}  # lower id = higher priority
    want = dict(desired)

    changed = True
    while changed:
        changed = False
        held_cells = {positions[r] for r in want if want[r] == positions[r]}

        claimants: dict[Cell, list[int]] = defaultdict(list)
        for r, w in want.items():
            if w != positions[r]:
                claimants[w].append(r)

        for cell, robots in claimants.items():
            if cell in held_cells:
                for r in robots:
                    want[r] = positions[r]
                    changed = True
                continue
            if len(robots) > 1:
                winner = min(robots, key=lambda r: priority[r])
                for r in robots:
                    if r != winner:
                        want[r] = positions[r]
                        changed = True

        # direct A<->B swap conflicts (head-on collision on the shared edge)
        for r in list(want):
            w = want[r]
            if w == positions[r]:
                continue
            for r2 in want:
                if r2 == r:
                    continue
                if positions[r2] == w and want[r2] == positions[r]:
                    loser = r if priority[r] > priority[r2] else r2
                    if want[loser] != positions[loser]:
                        want[loser] = positions[loser]
                        changed = True

    return want


def resolve(
    warehouse: Warehouse,
    positions: dict[int, Cell],
    desired: dict[int, Cell],
) -> dict[int, Cell]:
    """Classical avoidance policy: every robot proposes its A*-path next
    cell as-is, then `arbitrate` resolves conflicts. `warehouse` is unused
    here (kept so this matches `sim.policies.AvoidanceFn`, the same
    signature the learned policy in `ml/learned_traffic.py` needs it for -
    it inspects nearby rack/robot occupancy before proposing)."""
    del warehouse
    return arbitrate(positions, desired)
