from sim.world import World


def _run(seed: int, ticks: int = 150):
    world = World(seed=seed, num_robots=6, order_rate=0.4)
    world.run(ticks)
    return world


def test_same_seed_identical_robot_trajectories():
    a = _run(seed=42)
    b = _run(seed=42)
    assert a.telemetry.robot_rows == b.telemetry.robot_rows


def test_same_seed_identical_orders():
    a = _run(seed=42)
    b = _run(seed=42)

    def order_snapshot(world):
        return [
            (o.id, o.origin, o.destination, o.arrival_tick, o.status.value, o.pickup_tick, o.dropoff_tick)
            for o in world.orders.values()
        ]

    assert order_snapshot(a) == order_snapshot(b)


def test_different_seeds_diverge():
    a = _run(seed=1)
    b = _run(seed=2)
    assert a.telemetry.robot_rows != b.telemetry.robot_rows
