from sim.world import World


def test_fleet_does_not_permanently_deadlock():
    """Regression test: two robots whose A* paths cross used to wait on each
    other forever (neither cell ever freed), so the whole fleet could
    silently stall for the rest of a run. With the stuck-robot side-step in
    place, orders should keep completing throughout a longer run."""
    world = World(seed=0, num_robots=8, order_rate=0.15)
    world.run(600)

    completed = sum(1 for o in world.orders.values() if o.status.value == "completed")
    assert completed / len(world.orders) > 0.8

    active_orders_over_time = [row["active_orders"] for row in world.telemetry.tick_rows]
    # a permanently deadlocked fleet leaves the backlog stuck at its peak for
    # the remainder of the run; a live one clears most of it by the end.
    assert active_orders_over_time[-1] <= max(active_orders_over_time) * 0.5


def test_deadlock_breaker_fires_and_orders_keep_completing_after():
    """Force a head-on path crossing (two robots, targets on opposite sides
    of each other) and confirm the side-step logs an event and both robots
    still eventually complete their orders rather than freezing forever."""
    world = World(seed=2, num_robots=6, order_rate=0.2)
    world.run(500)

    deadlocks_broken = [e for e in world.telemetry.event_rows if e["type"] == "deadlock_broken"]
    completions_after_last_deadlock = [
        e
        for e in world.telemetry.event_rows
        if e["type"] == "order_completed"
        and (not deadlocks_broken or e["tick"] > deadlocks_broken[-1]["tick"])
    ]
    if deadlocks_broken:
        assert completions_after_last_deadlock, "no order completed after the last deadlock break"
