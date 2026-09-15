"""BM25 (keyword) index over the SOP corpus, via rank_bm25."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from rag.corpus import SopDoc, load_corpus

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Bm25Index:
    def __init__(self, docs: tuple[SopDoc, ...] | None = None):
        self.docs = docs or load_corpus()
        self.doc_ids = [d.id for d in self.docs]
        self._corpus_tokens = [tokenize(d.full_text) for d in self.docs]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        """Returns (doc_id, bm25_score) pairs, best first."""
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        if k is not None:
            order = order[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]
