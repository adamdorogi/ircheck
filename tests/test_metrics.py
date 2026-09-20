import math

import pytest

from ircheck.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    score_at_k,
)


def test_duplicate_hits_preserve_first_seen_order() -> None:
    hits = ["miss", "miss", "a", "a", "b"]
    assert recall_at_k(hits, ["a", "a", "b"], 2) == 0.5
    assert mrr_at_k(hits, ["a", "b"], 2) == 0.5
    assert hit_rate_at_k(hits, ["a", "b"], 1) == 0.0
    assert hit_rate_at_k(hits, ["a", "b"], 2) == 1.0
    assert ndcg_at_k(["a", "a", "a"], ["a"], 10) == 1.0


def test_graded_ndcg_uses_exponential_gain_and_ideal_order() -> None:
    hits = ["low", "zero", "high", "medium"]
    relevant = {"low": 1.0, "high": 3.0, "zero": 0.0, "medium": 2.0}
    observed = 1 + 7 / math.log2(4)
    ideal = 7 + 3 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(hits, relevant, 3) == pytest.approx(observed / ideal)
    assert recall_at_k(hits, relevant, 3) == pytest.approx(2 / 3)
    assert mrr_at_k(["zero", "high"], relevant, 3) == 0.5


def test_ndcg_ties_and_fewer_relevant_docs_than_k() -> None:
    relevant = {"high": 3.0, "first": 1.0, "second": 1.0}
    assert ndcg_at_k(["high", "second", "first", "miss"], relevant, 10) == 1.0
    assert ndcg_at_k(["first", "high", "second"], relevant, 10) < 1.0


@pytest.mark.parametrize(
    ("hits", "relevant", "k"),
    [
        ([], ["doc"], 3),
        (["doc"], [], 3),
        (["doc"], {"doc": 0.0}, 3),
        (["doc"], ["doc"], 0),
    ],
)
def test_empty_inputs_score_zero(
    hits: list[str], relevant: list[str] | dict[str, float], k: int
) -> None:
    assert score_at_k(hits, relevant, k) == {
        f"ndcg@{k}": 0.0,
        f"mrr@{k}": 0.0,
        f"recall@{k}": 0.0,
        f"hitrate@{k}": 0.0,
    }


def test_ndcg_stays_finite_with_large_grades() -> None:
    assert ndcg_at_k(["a", "b"], {"a": 4000.0, "b": 3998.0}, 10) == 1.0


def test_percentiles_interpolate_without_rounding() -> None:
    assert percentile([30, 20, 0, 10], 50) == 15.0
    assert percentile([30, 20, 0, 10], 95) == pytest.approx(28.5)
    assert percentile([8], 95) == 8.0
    assert percentile([], 95) == 0.0
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1], 105)
