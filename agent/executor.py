"""Action Executor (Locked Design Decision #4): the only code allowed to
turn an LLM's output into a simulator state change. Never trusts raw text -
parses JSON defensively, validates the action is a whitelist member,
validates the target_id matches one of the pre-computed candidates AND
still exists in current world state, and only then calls the corresponding
World method. Any failure at any stage (unparseable JSON, missing/wrong
keys, non-whitelisted action, invalid/stale target) falls back to NO_ACTION,
is always logged, and never raises out of `execute()` - a broken or
adversarial LLM response must never crash the simulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from agent.action_schema import WHITELISTED_ACTIONS, Action
from agent.context import ActionCandidates
from sim.world import World

# action -> (candidate field on ActionCandidates giving the expected target,
# world method name, kwarg name the id is passed as)
_ACTION_DISPATCH = {
    Action.THROTTLE_ZONE_TRAFFIC.value: ("zone_id", "throttle_zone", "zone_id"),
    Action.REASSIGN_TASK.value: ("order_id", "reassign_order", "order_id"),
    Action.REPLAN_ROUTE.value: ("robot_id", "force_replan", "robot_id"),
}


@dataclass(frozen=True)
class AgentDecision:
    action: str
    target_type: str | None
    target_id: object | None
    reason: str | None
    executed: bool
    error: str | None
    raw_llm_output: str | None


def parse_llm_output(raw: str) -> dict | None:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict) or "action" not in parsed:
        return None
    return parsed


class ActionExecutor:
    def __init__(self, world: World):
        self.world = world

    def execute(self, raw_llm_output: str | None, candidates: ActionCandidates) -> AgentDecision:
        parsed = parse_llm_output(raw_llm_output) if raw_llm_output is not None else None
        if parsed is None:
            return AgentDecision(
                action=Action.NO_ACTION.value, target_type=None, target_id=None,
                reason=None, executed=False, error="unparseable_or_malformed_llm_output",
                raw_llm_output=raw_llm_output,
            )

        action = parsed.get("action")
        reason = parsed.get("reason")
        target_id = parsed.get("target_id")

        if action not in WHITELISTED_ACTIONS:
            return AgentDecision(
                action=Action.NO_ACTION.value, target_type=None, target_id=None,
                reason=reason, executed=False, error=f"action_not_whitelisted:{action!r}",
                raw_llm_output=raw_llm_output,
            )

        if action == Action.NO_ACTION.value:
            return AgentDecision(
                action=action, target_type=None, target_id=None, reason=reason,
                executed=True, error=None, raw_llm_output=raw_llm_output,
            )

        candidate_field, method_name, _ = _ACTION_DISPATCH[action]
        expected_target = getattr(candidates, candidate_field)
        target_type = candidate_field.replace("_id", "")

        if expected_target is None:
            return AgentDecision(
                action=Action.NO_ACTION.value, target_type=None, target_id=None,
                reason=reason, executed=False,
                error=f"no_candidate_offered_for:{action}",
                raw_llm_output=raw_llm_output,
            )

        # target_id must match the offered candidate exactly - int candidates
        # (order/robot ids) may arrive from the LLM as strings, since JSON
        # mode doesn't enforce a schema; compare loosely on that one axis only.
        matches = str(target_id) == str(expected_target)
        if not matches:
            return AgentDecision(
                action=Action.NO_ACTION.value, target_type=None, target_id=None,
                reason=reason, executed=False,
                error=f"target_id_mismatch:got={target_id!r},expected={expected_target!r}",
                raw_llm_output=raw_llm_output,
            )

        method = getattr(self.world, method_name)
        ok = method(expected_target)
        return AgentDecision(
            action=action, target_type=target_type, target_id=expected_target,
            reason=reason, executed=bool(ok),
            error=None if ok else "world_method_returned_false_stale_target",
            raw_llm_output=raw_llm_output,
        )
