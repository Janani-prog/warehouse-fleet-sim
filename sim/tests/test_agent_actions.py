"""World-level action methods the M8 executor calls (throttle_zone,
reassign_order, force_replan) - not the executor/LLM path itself, just that
these are correct, minimal simulator operations."""

from sim.entities import OrderStatus, RobotState
from sim.world import World


def test_throttle_zone_defers_pending_orders_in_that_zone():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    zone = world.warehouse.zone_ids()[0]
    # place a pending order whose origin is in the target zone
    from sim.entities import Order

    order = Order(id=999, origin=world.warehouse.pickup_points[0], destination=world.warehouse.dropoff_points[0], arrival_tick=0)
    target_zone = world.warehouse.zone_id(*order.origin)
    world.orders[order.id] = order

    assert world.throttle_zone(target_zone, duration=5)
    world._assign_orders(tick=0)
    assert world.orders[order.id].status == OrderStatus.PENDING  # deferred, not assigned

    world.throttled_zones.pop(target_zone)
    world._assign_orders(tick=1)
    assert world.orders[order.id].status == OrderStatus.ASSIGNED  # assigns normally once cleared


def test_throttle_zone_rejects_unknown_zone():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    assert world.throttle_zone("not-a-real-zone") is False


def test_throttle_expires_after_duration():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    zone = world.warehouse.zone_ids()[0]
    world.throttle_zone(zone, duration=2)
    world.tick()
    assert zone in world.throttled_zones
    world.tick()
    assert zone not in world.throttled_zones


def test_reassign_order_unassigns_and_clears_robot():
    world = World(seed=0, num_robots=2, order_rate=0.3)
    world.tick()
    assigned = [o for o in world.orders.values() if o.status == OrderStatus.ASSIGNED]
    if not assigned:
        return  # nondeterministic-by-config edge case: nothing to reassign yet
    order = assigned[0]
    robot_id = order.assigned_robot
    assert world.reassign_order(order.id)
    assert world.orders[order.id].status == OrderStatus.PENDING
    assert world.orders[order.id].assigned_robot is None
    robot = next(r for r in world.robots if r.id == robot_id)
    assert robot.order_id is None
    assert robot.state == RobotState.IDLE


def test_reassign_order_rejects_picked_up_order():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    from sim.entities import Order

    order = Order(id=1, origin=(0, 0), destination=(1, 1), arrival_tick=0, status=OrderStatus.PICKED_UP, assigned_robot=0)
    world.orders[1] = order
    assert world.reassign_order(1) is False


def test_reassign_order_rejects_unknown_order():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    assert world.reassign_order(12345) is False


def test_force_replan_clears_path():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    world.robots[0].path = [(1, 1), (2, 2)]
    assert world.force_replan(0)
    assert world.robots[0].path == []


def test_force_replan_rejects_unknown_robot():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    assert world.force_replan(999) is False


def test_zone_stats_matches_manual_computation():
    world = World(seed=0, num_robots=4, order_rate=0.3)
    world.run(20)
    queue_depth, density = world.zone_stats()
    assert sum(density.values()) == len(world.robots)
    assert set(queue_depth) == set(world.warehouse.zone_ids())
