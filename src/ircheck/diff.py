from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose

from ircheck.config import QUALITY_METRICS
from ircheck.runner import Snapshot


@dataclass(frozen=True, slots=True)
class MetricDelta:
    metric: str
    baseline: float
    current: float
    delta: float
    max_drop: float
    failed: bool


@dataclass(frozen=True, slots=True)
class TagDelta:
    tag: str
    count: int
    metrics: tuple[MetricDelta, ...]


@dataclass(frozen=True, slots=True)
class Regression:
    id: str
    query: str
    tags: list[str]
    expected: list[str]
    top_hits: list[str]
    baseline_rank: int | None
    current_rank: int | None
    baseline_score: float
    current_score: float


@dataclass(frozen=True, slots=True)
class Comparison:
    k: int
    baseline_target: str
    current_target: str
    overall: tuple[MetricDelta, ...]
    tags: tuple[TagDelta, ...]
    worst: tuple[Regression, ...]

    @property
    def failures(self) -> int:
        return sum(metric.failed for metric in self.overall) + sum(
            metric.failed for tag in self.tags for metric in tag.metrics
        )


def _below(value: float, limit: float) -> bool:
    return value < limit and not isclose(value, limit, rel_tol=0, abs_tol=1e-12)


def _metrics(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    k: int,
    max_drop: float,
) -> tuple[MetricDelta, ...]:
    deltas = []
    for name in QUALITY_METRICS:
        metric = f"{name}@{k}"
        if metric not in baseline or metric not in current:
            raise ValueError(f"snapshot is missing {metric}")
        before = baseline[metric]
        after = current[metric]
        delta = after - before
        deltas.append(MetricDelta(metric, before, after, delta, max_drop, _below(delta, -max_drop)))
    return tuple(deltas)


def compare_snapshots(
    baseline: Snapshot,
    current: Snapshot,
    *,
    max_drop: float = 0.02,
    tag_max_drop: Mapping[str, float] | None = None,
) -> Comparison:
    if baseline.k != current.k:
        raise ValueError(
            f"snapshot cutoffs differ (baseline k={baseline.k}, current k={current.k})"
        )
    if baseline.errors or current.errors:
        raise ValueError("cannot compare snapshots with query errors")

    before_by_id = {query.id: query for query in baseline.queries}
    after_by_id = {query.id: query for query in current.queries}
    if before_by_id.keys() != after_by_id.keys():
        raise ValueError("snapshot query IDs differ. Use a baseline for the same query set")
    if baseline.tags.keys() != current.tags.keys():
        raise ValueError("snapshot tags differ. Use a baseline for the same query set")

    metric = f"ndcg@{current.k}"
    regressions = []
    for query_id, before in before_by_id.items():
        after = after_by_id[query_id]
        if (
            before.query != after.query
            or set(before.tags) != set(after.tags)
            or (before.grades or dict.fromkeys(before.expected, 1.0))
            != (after.grades or dict.fromkeys(after.expected, 1.0))
        ):
            raise ValueError(f"query {query_id!r} differs between snapshots")
        if metric not in before.metrics or metric not in after.metrics:
            raise ValueError(f"query {query_id!r} is missing {metric}")
        if _below(after.metrics[metric], before.metrics[metric]):
            regressions.append(
                Regression(
                    query_id,
                    after.query,
                    after.tags,
                    after.expected,
                    after.hits[:3],
                    before.rank,
                    after.rank,
                    before.metrics[metric],
                    after.metrics[metric],
                )
            )
    regressions.sort(key=lambda row: (row.current_score - row.baseline_score, row.id))
    overrides = tag_max_drop or {}
    tags = []
    for tag, after_tag in current.tags.items():
        before_tag = baseline.tags[tag]
        if before_tag.count != after_tag.count:
            raise ValueError(f"tag {tag!r} has different query counts between snapshots")
        metrics = _metrics(
            before_tag.metrics, after_tag.metrics, current.k, overrides.get(tag, max_drop)
        )
        tags.append(TagDelta(tag, after_tag.count, metrics))
    tags.sort(key=lambda row: (-row.metrics[0].delta, row.tag))
    return Comparison(
        current.k,
        baseline.target,
        current.target,
        _metrics(baseline.metrics, current.metrics, current.k, max_drop),
        tuple(tags),
        tuple(regressions[:5]),
    )
