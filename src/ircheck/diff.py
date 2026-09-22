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
class Comparison:
    k: int
    baseline_target: str
    current_target: str
    overall: tuple[MetricDelta, ...]

    @property
    def failures(self) -> int:
        return sum(metric.failed for metric in self.overall)


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
) -> Comparison:
    if baseline.k != current.k:
        raise ValueError(
            f"snapshot cutoffs differ (baseline k={baseline.k}, current k={current.k})"
        )
    if baseline.errors or current.errors:
        raise ValueError("cannot compare snapshots with query errors")

    return Comparison(
        current.k,
        baseline.target,
        current.target,
        _metrics(baseline.metrics, current.metrics, current.k, max_drop),
    )
