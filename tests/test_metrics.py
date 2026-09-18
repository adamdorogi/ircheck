import pytest

from ircheck.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_binary_scores() -> None:
    hits = ["a", "noise", "b"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(hits, relevant, 3) == pytest.approx(2 / 3)
    assert mrr_at_k(hits, relevant, 3) == 1.0
    assert ndcg_at_k(hits, relevant, 3) == pytest.approx(1.5 / (1.5 + 1 / 1.584962500721156))


def test_empty_results() -> None:
    assert recall_at_k([], {"a"}, 10) == 0.0
    assert mrr_at_k([], {"a"}, 10) == 0.0
    assert ndcg_at_k([], {"a"}, 10) == 0.0
