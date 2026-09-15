from sim.entities import Order, OrderStatus
from sim.world import World
from agent.context import build_candidates


def test_no_candidates_when_world_is_empty():
    world = World(seed=0, num_robots=0, order_rate=0.0)
    candidates = build_candidates(world)
    assert candidates.zone_id is None  # no robots/orders anywhere -> no zone has any load
    assert candidates.order_id is None
    assert candidates.robot_id is None


def test_busiest_zone_is_picked_by_combined_queue_and_density():
    world = World(seed=0, num_robots=3, order_rate=0.0)
    zone = world.warehouse.zone_ids()[0]
    zx, zy = 0, 0
    zw, zh = world.warehouse.zone_size
    for r in world.robots:
        r.x, r.y = zx * zw, zy * zh
    candidates = build_candidates(world)
    assert candidates.zone_id == zone
    assert candidates.zone_density == 3


def test_order_candidate_prefers_robot_in_busiest_zone():
    world = World(seed=0, num_robots=2, order_rate=0.0)
    zw, zh = world.warehouse.zone_size
    world.robots[0].x, world.robots[0].y = 0, 0  # zone Z_0_0
    world.robots[1].x, world.robots[1].y = zw, zh  # a different zone, Z_1_1
    order = Order(id=1, origin=(0, 0), destination=(5, 5), arrival_tick=0,
                   status=OrderStatus.ASSIGNED, assigned_robot=world.robots[0].id, assign_tick=0)
    world.orders[1] = order
    world.robots[0].order_id = 1

    candidates = build_candidates(world)
    assert candidates.order_id == 1
    assert candidates.order_robot_id == world.robots[0].id


def test_picked_up_orders_are_not_offered_as_reassign_candidates():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    world.robots[0].x, world.robots[0].y = 0, 0
    order = Order(id=1, origin=(0, 0), destination=(5, 5), arrival_tick=0,
                   status=OrderStatus.PICKED_UP, assigned_robot=world.robots[0].id, assign_tick=0)
    world.orders[1] = order
    candidates = build_candidates(world)
    assert candidates.order_id is None


def test_most_stuck_robot_is_the_candidate():
    world = World(seed=0, num_robots=3, order_rate=0.0)
    world.robots[0].stuck_ticks = 1
    world.robots[1].stuck_ticks = 5
    world.robots[2].stuck_ticks = 0
    candidates = build_candidates(world)
    assert candidates.robot_id == world.robots[1].id
    assert candidates.robot_stuck_ticks == 5


def test_no_robot_candidate_when_none_stuck():
    world = World(seed=0, num_robots=3, order_rate=0.0)
    candidates = build_candidates(world)
    assert candidates.robot_id is None
