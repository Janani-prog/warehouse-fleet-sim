"""Dense embedding + hybrid retrieval tests. These need a running local
Ollama server (`ollama serve`) with the nomic-embed-text model pulled - they
skip cleanly rather than failing if Ollama isn't reachable, since Ollama is
an external local service, not something pytest should require to exist for
the rest of the suite to run."""

from __future__ import annotations

import pytest

from rag.corpus import load_corpus
from rag.embeddings import OllamaUnavailableError, embed_text
from rag.hybrid_retrieval import HybridRetriever


@pytest.fixture(scope="module")
def ollama_available() -> bool:
    try:
        embed_text("connectivity check")
        return True
    except OllamaUnavailableError:
        return False


@pytest.fixture(scope="module")
def hybrid(ollama_available):
    if not ollama_available:
        pytest.skip("Ollama not reachable at localhost:11434 - skipping dense/hybrid tests")
    return HybridRetriever(use_cache=True)


def test_hybrid_returns_k_results(hybrid):
    results = hybrid.retrieve("congestion critical severity backstop firing", k=3)
    assert len(results) == 3
    assert all(r.doc.id in {d.id for d in load_corpus()} for r in results)


def test_hybrid_top_result_relevant_for_unambiguous_query(hybrid):
    results = hybrid.retrieve(
        "Collision risk, critical severity. Calibrated confidence 0.9. Rule-based backstop is firing this tick.",
        k=1,
    )
    assert results[0].doc.id == "collision-critical"


def test_hybrid_scores_descending(hybrid):
    results = hybrid.retrieve("congestion", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
