import json

from agent.context import ActionCandidates
from sim.entities import Order, OrderStatus
from sim.world import World
from agent.executor import ActionExecutor, parse_llm_output

EMPTY_CANDIDATES = ActionCandidates(
    zone_id=None, zone_queue_depth=0, zone_density=0,
    order_id=None, order_robot_id=None,
    robot_id=None, robot_stuck_ticks=0,
)


def test_parse_llm_output_valid_json():
    assert parse_llm_output('{"action": "NO_ACTION", "target_id": null, "reason": "x"}') == {
        "action": "NO_ACTION", "target_id": None, "reason": "x"
    }


def test_parse_llm_output_garbage_returns_none():
    assert parse_llm_output("not json at all {{{") is None
    assert parse_llm_output("") is None
    assert parse_llm_output(None) is None


def test_parse_llm_output_missing_action_key_returns_none():
    assert parse_llm_output('{"target_id": "Z_0_0"}') is None


def test_malformed_output_falls_back_to_no_action_and_logs_error():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    executor = ActionExecutor(world)
    decision = executor.execute("this is not json", EMPTY_CANDIDATES)
    assert decision.action == "NO_ACTION"
    assert decision.executed is False
    assert decision.error is not None


def test_non_whitelisted_action_falls_back_to_no_action():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    executor = ActionExecutor(world)
    raw = json.dumps({"action": "DELETE_ALL_ROBOTS", "target_id": None, "reason": "malicious"})
    decision = executor.execute(raw, EMPTY_CANDIDATES)
    assert decision.action == "NO_ACTION"
    assert decision.executed is False
    assert "not_whitelisted" in decision.error


def test_no_action_passes_through_as_executed():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    executor = ActionExecutor(world)
    raw = json.dumps({"action": "NO_ACTION", "target_id": None, "reason": "nothing to do"})
    decision = executor.execute(raw, EMPTY_CANDIDATES)
    assert decision.action == "NO_ACTION"
    assert decision.executed is True
    assert decision.error is None


def test_throttle_zone_target_matching_candidate_executes():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    zone = world.warehouse.zone_ids()[0]
    candidates = ActionCandidates(
        zone_id=zone, zone_queue_depth=5, zone_density=2,
        order_id=None, order_robot_id=None, robot_id=None, robot_stuck_ticks=0,
    )
    executor = ActionExecutor(world)
    raw = json.dumps({"action": "THROTTLE_ZONE_TRAFFIC", "target_id": zone, "reason": "high queue"})
    decision = executor.execute(raw, candidates)
    assert decision.executed is True
    assert decision.target_id == zone
    assert zone in world.throttled_zones


def test_adversarial_target_id_not_in_candidates_is_rejected():
    """The LLM tries to act on a zone that was never offered as a candidate -
    the executor must reject it, not trust the model's claim, even though
    the zone is otherwise a real, valid zone id in the warehouse."""
    world = World(seed=0, num_robots=1, order_rate=0.0)
    real_but_unoffered_zone = world.warehouse.zone_ids()[-1]
    candidates = ActionCandidates(
        zone_id=world.warehouse.zone_ids()[0], zone_queue_depth=5, zone_density=2,
        order_id=None, order_robot_id=None, robot_id=None, robot_stuck_ticks=0,
    )
    executor = ActionExecutor(world)
    raw = json.dumps({
        "action": "THROTTLE_ZONE_TRAFFIC", "target_id": real_but_unoffered_zone, "reason": "trying to sneak one in",
    })
    decision = executor.execute(raw, candidates)
    assert decision.action == "NO_ACTION"
    assert decision.executed is False
    assert real_but_unoffered_zone not in world.throttled_zones


def test_action_with_no_candidate_offered_falls_back_to_no_action():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    executor = ActionExecutor(world)
    raw = json.dumps({"action": "REPLAN_ROUTE", "target_id": "5", "reason": "x"})
    decision = executor.execute(raw, EMPTY_CANDIDATES)
    assert decision.action == "NO_ACTION"
    assert decision.executed is False
    assert "no_candidate_offered" in decision.error


def test_reassign_task_executes_and_frees_robot():
    world = World(seed=0, num_robots=2, order_rate=0.0)
    order = Order(id=1, origin=(0, 1), destination=(1, 1), arrival_tick=0,
                   status=OrderStatus.ASSIGNED, assigned_robot=world.robots[0].id, assign_tick=0)
    world.orders[1] = order
    world.robots[0].order_id = 1
    candidates = ActionCandidates(
        zone_id=None, zone_queue_depth=0, zone_density=0,
        order_id=1, order_robot_id=world.robots[0].id,
        robot_id=None, robot_stuck_ticks=0,
    )
    executor = ActionExecutor(world)
    raw = json.dumps({"action": "REASSIGN_TASK", "target_id": "1", "reason": "rebalance"})
    decision = executor.execute(raw, candidates)
    assert decision.executed is True
    assert world.orders[1].status == OrderStatus.PENDING


def test_replan_route_executes_and_clears_path():
    world = World(seed=0, num_robots=1, order_rate=0.0)
    world.robots[0].path = [(1, 1)]
    candidates = ActionCandidates(
        zone_id=None, zone_queue_depth=0, zone_density=0,
        order_id=None, order_robot_id=None,
        robot_id=world.robots[0].id, robot_stuck_ticks=3,
    )
    executor = ActionExecutor(world)
    raw = json.dumps({"action": "REPLAN_ROUTE", "target_id": str(world.robots[0].id), "reason": "stuck"})
    decision = executor.execute(raw, candidates)
    assert decision.executed is True
    assert world.robots[0].path == []
