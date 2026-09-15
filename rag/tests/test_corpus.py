from rag.corpus import get_doc, load_corpus

import pytest


def test_corpus_size_in_backlog_range():
    docs = load_corpus()
    assert 15 <= len(docs) <= 30


def test_corpus_ids_unique():
    docs = load_corpus()
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids))


def test_covers_both_anomaly_types_all_severities():
    docs = load_corpus()
    core = {(d.anomaly_type, d.severity) for d in docs}
    for anomaly_type in ("congestion", "collision_risk"):
        for severity in ("low", "moderate", "high", "critical"):
            assert (anomaly_type, severity) in core, f"missing {anomaly_type}/{severity}"


def test_recommended_action_is_whitelisted():
    from agent.action_schema import WHITELISTED_ACTIONS

    for doc in load_corpus():
        assert doc.recommended_action in WHITELISTED_ACTIONS


def test_get_doc_roundtrip():
    docs = load_corpus()
    doc = get_doc(docs[0].id)
    assert doc == docs[0]


def test_get_doc_missing_raises():
    with pytest.raises(KeyError):
        get_doc("does-not-exist")
