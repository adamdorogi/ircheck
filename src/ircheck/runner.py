import asyncio
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import Literal, Protocol, Self

import httpx
from pydantic import Field, model_validator

from .adapters.http import HttpAdapter
from .config import QUALITY_METRICS, ConfigModel, Grade, Query, Suite
from .metrics import first_relevant_rank, percentile, positive_grades, score_at_k, unique_ids


class Adapter(Protocol):
    async def search(self, query: str, k: int) -> list[str]: ...


class QueryResult(ConfigModel):
    id: str
    query: str
    tags: list[str]
    expected: list[str]
    hits: list[str]
    grades: dict[str, Grade] = Field(default_factory=dict)
    rank: int | None = Field(default=None, gt=0)
    metrics: dict[str, float]
    latency_ms: float | None = Field(default=None, ge=0)
    error: str | None = None


class TagResult(ConfigModel):
    count: int = Field(gt=0)
    metrics: dict[str, float]


class Snapshot(ConfigModel):
    schema_version: Literal[1] = 1
    target: str
    k: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    warmup: int = Field(ge=0)
    elapsed_s: float = Field(ge=0)
    metrics: dict[str, float]
    tags: dict[str, TagResult]
    queries: list[QueryResult]

    @property
    def errors(self) -> int:
        return sum(query.error is not None for query in self.queries)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if len({query.id for query in self.queries}) != len(self.queries):
            raise ValueError("snapshot contains duplicate query IDs")
        names = {f"{name}@{self.k}" for name in QUALITY_METRICS}
        entries = [("overall", self.metrics)]
        entries += [(f"tag {tag!r}", group.metrics) for tag, group in self.tags.items()]
        entries += [(f"query {query.id!r}", query.metrics) for query in self.queries]
        for label, metrics in entries:
            missing = names - metrics.keys()
            if missing:
                raise ValueError(f"{label} is missing metrics: {', '.join(sorted(missing))}")
        return self


def _aggregate(queries: Sequence[QueryResult], k: int) -> dict[str, float]:
    names = [f"{metric}@{k}" for metric in QUALITY_METRICS]
    scores = {
        name: statistics.fmean(query.metrics[name] for query in queries) if queries else 0.0
        for name in names
    }
    latencies = [query.latency_ms for query in queries if query.latency_ms is not None]
    scores["p50_ms"] = percentile(latencies, 50)
    scores["p95_ms"] = percentile(latencies, 95)
    return scores


def summarize(
    queries: Sequence[QueryResult], k: int
) -> tuple[dict[str, float], dict[str, TagResult]]:
    by_tag: dict[str, list[QueryResult]] = defaultdict(list)
    for query in queries:
        for tag in dict.fromkeys(query.tags):
            by_tag[tag].append(query)
    return _aggregate(queries, k), {
        tag: TagResult(count=len(group), metrics=_aggregate(group, k))
        for tag, group in sorted(by_tag.items())
    }


async def _retrieve(
    adapter: Adapter, query: str, k: int
) -> tuple[list[str], float | None, str | None]:
    started = time.perf_counter()
    try:
        hits = await adapter.search(query, k)
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        return [], None, f"HTTP {status} {error.response.reason_phrase}"
    except Exception as error:
        return [], None, f"{type(error).__name__}: {error}".rstrip(": ")
    return hits, (time.perf_counter() - started) * 1000, None


async def run_suite(
    suite: Suite,
    queries: Sequence[Query],
    *,
    concurrency: int | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Snapshot:
    concurrency = suite.concurrency if concurrency is None else concurrency
    if not queries:
        raise ValueError("no queries to evaluate")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    semaphore = asyncio.Semaphore(concurrency)
    async with AsyncExitStack() as stack:
        target = suite.target
        adapter: Adapter
        client = await stack.enter_async_context(
            httpx.AsyncClient(
                timeout=suite.timeout,
                transport=transport,
                limits=httpx.Limits(
                    max_connections=concurrency, max_keepalive_connections=concurrency
                ),
            )
        )
        adapter = HttpAdapter(client, target)
        url = httpx.URL(str(adapter.target.url))
        target_label = str(url.copy_with(username=None, password=None, query=None, fragment=None))


        async def retrieve(query: Query) -> tuple[list[str], float | None, str | None]:
            async with semaphore:
                return await _retrieve(adapter, query.query, suite.k)

        started = time.perf_counter()
        outputs = await asyncio.gather(*(retrieve(query) for query in queries))
        elapsed_s = time.perf_counter() - started

    results: list[QueryResult] = []
    for query, (hits, latency, error) in zip(queries, outputs, strict=True):
        grades = positive_grades(query.relevant)
        ranked = unique_ids(hits)[:suite.k]
        results.append(
            QueryResult(
                id=query.id,
                query=query.query,
                tags=query.tags,
                expected=list(grades),
                hits=ranked,
                grades=grades,
                rank=first_relevant_rank(ranked, grades),
                metrics=score_at_k(ranked, grades, suite.k),
                latency_ms=latency,
                error=error,
            )
        )
    scores, tags = summarize(results, suite.k)
    return Snapshot(
        target=target_label,
        k=suite.k,
        concurrency=concurrency,
        warmup=0,
        elapsed_s=elapsed_s,
        metrics=scores,
        tags=tags,
        queries=results,
    )
