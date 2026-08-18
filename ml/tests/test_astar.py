from sim.grid import Warehouse, generate_default_layout
from ml.astar import astar


def _empty_warehouse(width, height, racks=()):
    return Warehouse(
        width=width,
        height=height,
        racks=frozenset(racks),
        zone_size=(width, height),
        spawn_points=(),
        dropoff_points=(),
        pickup_points=(),
    )


def test_start_equals_goal():
    wh = _empty_warehouse(5, 5)
    assert astar(wh, (2, 2), (2, 2)) == [(2, 2)]


def test_open_grid_shortest_path_length_matches_manhattan_distance():
    wh = _empty_warehouse(5, 5)
    path = astar(wh, (0, 0), (4, 4))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    # unit-cost 4-connectivity: optimal path length = manhattan distance + 1 cells
    assert len(path) == 9


def test_path_detours_around_wall_through_single_gap():
    # 5x5 grid, a full wall at y=2 except a gap at x=2
    racks = {(x, 2) for x in range(5) if x != 2}
    wh = _empty_warehouse(5, 5, racks)
    path = astar(wh, (0, 0), (4, 4))
    assert path is not None
    assert (2, 2) in path  # must pass through the only gap
    assert all(cell not in wh.racks for cell in path)
    # optimal: 4 steps to reach (2,2) is not required, but path must be minimal
    # manhattan distance is 8, but the gap forces a detour to x=2 first
    assert len(path) == astar(wh, (0, 0), (2, 2)).__len__() + astar(wh, (2, 2), (4, 4)).__len__() - 1


def test_unreachable_goal_returns_none():
    wh = _empty_warehouse(3, 3, racks={(1, 0), (1, 1), (1, 2)})
    assert astar(wh, (0, 0), (2, 2)) is None


def test_blocked_start_or_goal_returns_none():
    wh = _empty_warehouse(3, 3, racks={(1, 1)})
    assert astar(wh, (1, 1), (2, 2)) is None
    assert astar(wh, (0, 0), (1, 1)) is None


def test_real_layout_pickup_to_dropoff_path_is_free_and_connected():
    wh = generate_default_layout()
    start = wh.pickup_points[0]
    goal = wh.dropoff_points[0]
    path = astar(wh, start, goal)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert all(wh.is_free(*cell) for cell in path)
    # consecutive cells must be 4-connected steps
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
