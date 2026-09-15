"""Hybrid retrieval: BM25 + dense embedding similarity, combined via
Reciprocal Rank Fusion (RRF). RRF is used instead of a weighted score blend
because BM25 scores and cosine similarities live on different, incomparable
scales - RRF sidesteps that by fusing on rank position alone, which is the
standard, parameter-light way to combine heterogeneous retrievers.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.bm25_index import Bm25Index
from rag.corpus import SopDoc, get_doc, load_corpus
from rag.dense_index import DenseIndex

RRF_K = 60  # standard RRF damping constant


@dataclass(frozen=True)
class RetrievalResult:
    doc: SopDoc
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class HybridRetriever:
    def __init__(self, use_cache: bool = True):
        docs = load_corpus()
        self.bm25 = Bm25Index(docs)
        self.dense = DenseIndex(docs, use_cache=use_cache)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:
        n = len(load_corpus())
        bm25_ranked = self.bm25.search(query, k=n)
        dense_ranked = self.dense.search(query, k=n)

        bm25_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_ranked, start=1)}
        dense_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_ranked, start=1)}

        all_ids = set(bm25_rank) | set(dense_rank)
        fused: list[tuple[str, float]] = []
        for doc_id in all_ids:
            rrf_score = 0.0
            if doc_id in bm25_rank:
                rrf_score += 1.0 / (RRF_K + bm25_rank[doc_id])
            if doc_id in dense_rank:
                rrf_score += 1.0 / (RRF_K + dense_rank[doc_id])
            fused.append((doc_id, rrf_score))

        fused.sort(key=lambda pair: -pair[1])
        top = fused[:k]
        return [
            RetrievalResult(
                doc=get_doc(doc_id),
                score=score,
                bm25_rank=bm25_rank.get(doc_id),
                dense_rank=dense_rank.get(doc_id),
            )
            for doc_id, score in top
        ]
