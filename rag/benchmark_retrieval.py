"""Retrieval eval: top-1 / top-3 accuracy on the 18 hand-labeled queries in
rag/data/retrieval_eval_set.json, for BM25-only, dense-only, and the hybrid
RRF fusion - reported side by side so the hybrid's benefit (or lack of one)
over either retriever alone is honest and visible, not assumed.

Run: python -m rag.benchmark_retrieval
Requires a running local Ollama server (`ollama serve`) for the dense/hybrid
rows; BM25-only still runs without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rag.bm25_index import Bm25Index
from rag.corpus import load_corpus
from rag.dense_index import DenseIndex
from rag.hybrid_retrieval import HybridRetriever

EVAL_SET_PATH = Path(__file__).parent / "data" / "retrieval_eval_set.json"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

TARGET_TOP1_ACCURACY = 0.75


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _accuracy(ranked_ids_per_query: list[list[str]], expected: list[str], k: int) -> float:
    hits = sum(exp in ranked[:k] for ranked, exp in zip(ranked_ids_per_query, expected))
    return hits / len(expected)


def evaluate_bm25(eval_set: list[dict], index: Bm25Index) -> dict:
    ranked = [[doc_id for doc_id, _ in index.search(item["query"])] for item in eval_set]
    expected = [item["expected_doc_id"] for item in eval_set]
    return {"top1": _accuracy(ranked, expected, 1), "top3": _accuracy(ranked, expected, 3)}


def evaluate_dense(eval_set: list[dict], index: DenseIndex) -> dict:
    ranked = [[doc_id for doc_id, _ in index.search(item["query"])] for item in eval_set]
    expected = [item["expected_doc_id"] for item in eval_set]
    return {"top1": _accuracy(ranked, expected, 1), "top3": _accuracy(ranked, expected, 3)}


def evaluate_hybrid(eval_set: list[dict], retriever: HybridRetriever) -> dict:
    ranked = [[r.doc.id for r in retriever.retrieve(item["query"], k=len(load_corpus()))] for item in eval_set]
    expected = [item["expected_doc_id"] for item in eval_set]
    return {"top1": _accuracy(ranked, expected, 1), "top3": _accuracy(ranked, expected, 3)}


def main() -> None:
    eval_set = load_eval_set()
    docs = load_corpus()

    bm25_index = Bm25Index(docs)
    dense_index = DenseIndex(docs)
    hybrid = HybridRetriever()

    results = {
        "bm25_only": evaluate_bm25(eval_set, bm25_index),
        "dense_only": evaluate_dense(eval_set, dense_index),
        "hybrid_rrf": evaluate_hybrid(eval_set, hybrid),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).T
    df.index.name = "retriever"
    df.to_csv(RESULTS_DIR / "retrieval_benchmark.csv")

    report = {
        "n_queries": len(eval_set),
        "n_corpus_docs": len(docs),
        "target_top1_accuracy": TARGET_TOP1_ACCURACY,
        "results": results,
        "meets_target": results["hybrid_rrf"]["top1"] >= TARGET_TOP1_ACCURACY,
    }
    with open(RESULTS_DIR / "retrieval_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(df)
    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
