import pytest

from ircheck.config import QUALITY_METRICS
from ircheck.diff import compare_snapshots
from ircheck.runner import QueryResult, Snapshot, summarize


def snapshot(scores: list[float]) -> Snapshot:
    queries = [
        QueryResult(
            id=f"q{index}",
            query=f"fixture {index}",
            tags=["sku"] if index == 0 else ["semantic"],
            expected=["right"],
            grades={"right": 1.0},
            hits=["right"] if score else ["wrong"],
            rank=1 if score else None,
            metrics={f"{metric}@10": score for metric in QUALITY_METRICS},
            latency_ms=1.0,
        )
        for index, score in enumerate(scores)
    ]
    metrics, tags = summarize(queries, 10)
    return Snapshot(
        target="fixture",
        k=10,
        concurrency=1,
        warmup=0,
        elapsed_s=0.001,
        queries=queries,
        metrics=metrics,
        tags=tags,
    )


def test_tag_gate_catches_loss_when_overall_scores_hold() -> None:
    baseline = snapshot([1.0, 0.5])
    current = snapshot([0.5, 1.0])
    report = compare_snapshots(baseline, current)
    assert all(not metric.failed for metric in report.overall)
    assert report.failures == len(QUALITY_METRICS)
    assert report.tags[0].tag == "semantic"
    assert report.tags[1].tag == "sku"
    assert [query.id for query in report.worst] == ["q0"]
    assert compare_snapshots(baseline, current, tag_max_drop={"sku": 0.5}).failures == 0


def test_drop_equal_to_tolerance_passes() -> None:
    baseline = snapshot([0.8])
    current = snapshot([0.78])
    assert compare_snapshots(baseline, current, max_drop=0.02).failures == 0
    assert compare_snapshots(baseline, current, max_drop=0.019).failures == 8


def test_worst_regressions_are_sorted_and_limited() -> None:
    baseline = snapshot([1.0] * 7)
    current = snapshot([0.8, 0.7, 0.0, 0.3, 0.5, 0.1, 0.2])
    report = compare_snapshots(baseline, current)
    assert [query.id for query in report.worst] == ["q2", "q5", "q6", "q3", "q4"]
    worst = report.worst[0]
    assert (worst.baseline_rank, worst.current_rank) == (1, None)
    assert worst.expected == ["right"]
    assert worst.top_hits == ["wrong"]


def test_comparison_requires_same_queries_and_relevance() -> None:
    baseline = snapshot([1.0])
    current = snapshot([1.0])
    current.queries[0].grades["right"] = 3
    with pytest.raises(ValueError, match="differs between snapshots"):
        compare_snapshots(baseline, current)
    with pytest.raises(ValueError, match="query IDs differ"):
        compare_snapshots(baseline, snapshot([1.0, 1.0]))
