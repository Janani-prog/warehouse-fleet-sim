"""The whitelisted action enum (Locked Design Decision #4 in CLAUDE.md) and
the structured-output schema the LLM agent must produce. This enum is
extended only with care - both the RAG corpus (rag/corpus.py) and the LLM
agent's output are validated against it, and the action executor
(agent/executor.py) refuses anything else.
"""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    REASSIGN_TASK = "REASSIGN_TASK"
    REPLAN_ROUTE = "REPLAN_ROUTE"
    THROTTLE_ZONE_TRAFFIC = "THROTTLE_ZONE_TRAFFIC"
    NO_ACTION = "NO_ACTION"


WHITELISTED_ACTIONS = frozenset(a.value for a in Action)
