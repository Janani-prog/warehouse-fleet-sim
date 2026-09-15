from rag.bm25_index import Bm25Index, tokenize
from rag.corpus import load_corpus


def test_tokenize_lowercases_and_splits():
    assert tokenize("Congestion, CRITICAL Severity!") == ["congestion", "critical", "severity"]


def test_search_returns_all_docs_ranked():
    index = Bm25Index()
    results = index.search("congestion critical severity backstop")
    assert len(results) == len(load_corpus())
    assert all(isinstance(doc_id, str) for doc_id, _ in results)


def test_top_result_is_relevant_for_exact_keyword_match():
    index = Bm25Index()
    results = index.search("dock zone congestion pickup drop-off queue saturation", k=3)
    top_ids = [doc_id for doc_id, _ in results]
    assert "congestion-critical-dock" in top_ids


def test_k_limits_results():
    index = Bm25Index()
    results = index.search("collision risk", k=5)
    assert len(results) == 5
