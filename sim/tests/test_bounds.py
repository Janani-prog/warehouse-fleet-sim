from sim.world import World


def test_no_entities_spawn_or_move_out_of_bounds():
    world = World(seed=7, num_robots=8, order_rate=0.6)
    world.run(200)

    wh = world.warehouse
    for row in world.telemetry.robot_rows:
        assert wh.is_free(row["x"], row["y"]), f"robot left free space: {row}"

    for order in world.orders.values():
        assert wh.is_free(*order.origin)
        assert wh.is_free(*order.destination)


def test_robots_never_enter_racks_across_run():
    world = World(seed=3, num_robots=5, order_rate=0.5)
    world.run(300)
    rack_hits = [
        row for row in world.telemetry.robot_rows if (row["x"], row["y"]) in world.warehouse.racks
    ]
    assert rack_hits == []
