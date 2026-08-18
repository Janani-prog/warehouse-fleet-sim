from sim.grid import generate_default_layout


def test_zones_partition_full_grid():
    wh = generate_default_layout()
    zw, zh = wh.zone_size
    for x in range(wh.width):
        for y in range(wh.height):
            zid = wh.zone_id(x, y)
            assert zid.startswith("Z_")
    assert len(wh.zone_ids()) == -(-wh.width // zw) * -(-wh.height // zh)


def test_spawn_and_dock_points_are_free():
    wh = generate_default_layout()
    for x, y in wh.spawn_points:
        assert wh.is_free(x, y)
    for x, y in wh.dropoff_points:
        assert wh.is_free(x, y)
    for x, y in wh.pickup_points:
        assert wh.is_free(x, y)


def test_racks_within_bounds():
    wh = generate_default_layout()
    for x, y in wh.racks:
        assert wh.in_bounds(x, y)


def test_neighbors_never_include_racks_or_out_of_bounds():
    wh = generate_default_layout()
    for x, y in list(wh.racks)[:20]:
        for nx, ny in wh.neighbors(x, y):
            assert wh.is_free(nx, ny)
