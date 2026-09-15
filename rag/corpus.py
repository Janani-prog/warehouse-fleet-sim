"""Synthetic SOP corpus loader.

18 hand-authored SOP documents covering the two anomaly types the M4
forecaster predicts (congestion, collision_risk) across four severity bands
(low, moderate, high, critical) plus several zone-specific variants and
general/cross-cutting procedures (false-positive handling, escalation,
concurrent triggers, a reference doc). See rag/data/sop_corpus.json for the
authored text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CORPUS_PATH = Path(__file__).parent / "data" / "sop_corpus.json"


@dataclass(frozen=True)
class SopDoc:
    id: str
    anomaly_type: str
    severity: str
    title: str
    recommended_action: str
    text: str

    @property
    def full_text(self) -> str:
        """Title + body concatenated - this is what gets indexed."""
        return f"{self.title}. {self.text}"


@lru_cache(maxsize=1)
def load_corpus(path: Path | None = None) -> tuple[SopDoc, ...]:
    p = path or CORPUS_PATH
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    docs = tuple(SopDoc(**entry) for entry in raw)
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids)), "duplicate SOP doc id in corpus"
    return docs


def get_doc(doc_id: str) -> SopDoc:
    for doc in load_corpus():
        if doc.id == doc_id:
            return doc
    raise KeyError(f"no SOP doc with id {doc_id!r}")
