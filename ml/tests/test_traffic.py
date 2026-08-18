from sim.grid import Warehouse
from ml.traffic import resolve


def _empty_warehouse(width=7, height=7):
    return Warehouse(
        width=width,
        height=height,
        racks=frozenset(),
        zone_size=(width, height),
        spawn_points=(),
        dropoff_points=(),
        pickup_points=(),
    )


def _assert_no_vertex_or_swap_collisions(positions, final):
    # no two robots end up in the same cell
    cells = list(final.values())
    assert len(cells) == len(set(cells)), f"vertex collision: {final}"
    # no two robots directly swap cells
    for a in final:
        for b in final:
            if a != b and final[a] == positions[b] and final[b] == positions[a] and final[a] != positions[a]:
                raise AssertionError(f"edge/swap collision between {a} and {b}: {final}")


def test_two_robot_head_on_swap_is_resolved_without_collision():
    # A direct swap is never allowed - even for the higher-priority robot -
    # since in continuous space it represents a head-on collision on the
    # shared edge. Both robots yield (stay put) rather than complete it.
    wh = _empty_warehouse()
    positions = {0: (2, 2), 1: (3, 2)}
    desired = {0: (3, 2), 1: (2, 2)}
    final = resolve(wh, positions, desired)
    _assert_no_vertex_or_swap_collisions(positions, final)
    assert final[0] == positions[0]
    assert final[1] == positions[1]


def test_vertex_conflict_two_robots_want_same_free_cell():
    wh = _empty_warehouse()
    positions = {0: (2, 2), 1: (2, 4)}
    desired = {0: (2, 3), 1: (2, 3)}
    final = resolve(wh, positions, desired)
    _assert_no_vertex_or_swap_collisions(positions, final)
    assert final[0] == (2, 3)  # priority winner
    assert final[1] != (2, 3)


def test_stationary_robot_blocks_mover():
    wh = _empty_warehouse()
    # robot 1 is idle/stationary at (3,3); robot 0 wants to move into it
    positions = {0: (2, 3), 1: (3, 3)}
    desired = {0: (3, 3), 1: (3, 3)}  # robot 1's desired == its own position: not moving
    final = resolve(wh, positions, desired)
    assert final[1] == (3, 3)  # stationary robot never displaced
    assert final[0] == positions[0]  # robot 0 could not enter the held cell, so it waits
    _assert_no_vertex_or_swap_collisions(positions, final)


def test_blocked_robot_waits_rather_than_wandering():
    wh = _empty_warehouse()
    # robots 0 and 1 both want the same free cell (2,3); robot 2 sits nearby.
    positions = {0: (2, 2), 1: (2, 4), 2: (3, 3)}
    desired = {0: (2, 3), 1: (2, 3), 2: (3, 3)}
    final = resolve(wh, positions, desired)
    _assert_no_vertex_or_swap_collisions(positions, final)
    assert final[0] == (2, 3)  # priority winner (lower id)
    assert final[1] == positions[1]  # loser waits in place, doesn't wander
    assert final[2] == positions[2]  # stationary robot never displaced


def test_no_conflict_everyone_proceeds():
    wh = _empty_warehouse()
    positions = {0: (1, 1), 1: (5, 5)}
    desired = {0: (2, 1), 1: (5, 6)}
    final = resolve(wh, positions, desired)
    assert final == desired


def test_idle_robots_never_move():
    wh = _empty_warehouse()
    positions = {0: (1, 1), 1: (2, 1)}
    desired = {0: (1, 1), 1: (2, 1)}  # both idle
    final = resolve(wh, positions, desired)
    assert final == positions


def test_three_robot_rotation_is_allowed_not_treated_as_conflict():
    wh = _empty_warehouse()
    # A->B's cell, B->C's cell, C->A's cell: standard MAPF allows this
    positions = {0: (2, 2), 1: (3, 2), 2: (2, 3)}
    desired = {0: (3, 2), 1: (2, 3), 2: (2, 2)}
    final = resolve(wh, positions, desired)
    _assert_no_vertex_or_swap_collisions(positions, final)
    assert final == desired  # no pairwise swap involved, so nothing should be blocked
