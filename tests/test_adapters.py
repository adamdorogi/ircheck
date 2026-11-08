import asyncio
import json

import httpx
import pytest

from ircheck import runner
from ircheck.adapters.http import extract_ids
from ircheck.config import Query, Suite


def query(text: str = "m12 bolt", identifier: str = "q01") -> Query:
    return Query(id=identifier, query=text, tags=["exact_sku"], relevant=["BOLT-1"])


def test_http_json_template_preserves_query_text_and_numeric_cutoff() -> None:
    text = 'quoted "m12"\nbackslash \\ and $k with a null \x00'
    suite = Suite.model_validate(
        {
            "dataset": "queries.jsonl",
            "k": 4,
            "target": {
                "type": "http",
                "url": "http://search.test/api/search",
                "query_template": {"retrieval": [{"q": "$query", "limit": "$k"}]},
                "id_path": "results[].sku",
            },
        }
    )

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert json.loads(request.content) == {"retrieval": [{"q": text, "limit": 4}]}
        return httpx.Response(200, json={"results": [{"sku": "BOLT-1"}, {"sku": 101}]})

    snapshot = asyncio.run(
        runner.run_suite(suite, [query(text)], transport=httpx.MockTransport(handle))
    )
    assert snapshot.errors == 0
    assert snapshot.queries[0].hits == ["BOLT-1", "101"]
    assert snapshot.queries[0].grades == {"BOLT-1": 1}
    assert snapshot.metrics["ndcg@4"] == 1.0
    assert snapshot.tags["exact_sku"].count == 1


@pytest.mark.parametrize(("id_field", "expected"), [("_id", "doc"), ("_source.sku", "BOLT-1")])
def test_opensearch_uses_index_and_configured_id_field(id_field: str, expected: str) -> None:
    suite = Suite.model_validate(
        {
            "dataset": "queries.jsonl",
            "target": {
                "type": "opensearch",
                "url": "https://search.test/proxy",
                "index": "hardware",
                "query_template": {"query": {"match": {"name": "$query"}}, "size": "$k"},
                "id_field": id_field,
            },
        }
    )

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/proxy/hardware/_search"
        assert request.method == "POST"
        assert json.loads(request.content) == {"query": {"match": {"name": "m12 bolt"}}, "size": 10}
        return httpx.Response(
            200, json={"hits": {"hits": [{"_id": "doc", "_source": {"sku": "BOLT-1"}}]}}
        )

    snapshot = asyncio.run(
        runner.run_suite(suite, [query()], transport=httpx.MockTransport(handle))
    )
    assert snapshot.queries[0].hits == [expected]


def test_empty_list_is_a_hitless_response_but_malformed_paths_fail() -> None:
    assert extract_ids({"hits": []}, "hits[].doc_id") == []
    with pytest.raises(ValueError, match="does not match ID path"):
        extract_ids({"hits": [{}]}, "hits[].doc_id")
    with pytest.raises(ValueError, match="document IDs"):
        extract_ids({"hits": [True]}, "hits")


def sync_search(text: str, k: int) -> tuple[int, str]:
    assert (text, k) == ("m12 bolt", 10)
    return 101, "BOLT-1"


async def async_search(text: str, k: int) -> list[str]:
    assert (text, k) == ("m12 bolt", 10)
    return ["BOLT-1"]


@pytest.mark.parametrize(("fn", "rank"), [("sync_search", 2), ("async_search", 1)])
def test_python_target_handles_sync_and_async_callables(fn: str, rank: int) -> None:
    suite = Suite.model_validate(
        {"dataset": "queries.jsonl", "target": {"type": "python", "callable": f"{__name__}:{fn}"}}
    )
    snapshot = asyncio.run(runner.run_suite(suite, [query()]))
    assert snapshot.queries[0].rank == rank
    assert snapshot.metrics["mrr@10"] == 1 / rank


def test_runner_bounds_concurrency_and_excludes_warmup_hits() -> None:
    suite = Suite.model_validate(
        {"dataset": "queries.jsonl", "target": {"type": "http", "url": "http://search.test"}}
    )
    active = 0
    peak = 0
    calls: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        calls.append(json.loads(request.content)["query"])
        warming = len(calls) <= 3
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return httpx.Response(200, json={"results": [{"id": "MISS" if warming else "BOLT-1"}]})

    queries = [query("first", "q1"), query("second", "q2")]
    snapshot = asyncio.run(
        runner.run_suite(
            suite, queries, concurrency=2, warmup=3, transport=httpx.MockTransport(handle)
        )
    )
    assert peak == 2
    assert calls == ["first", "second", "first", "first", "second"]
    assert [result.id for result in snapshot.queries] == ["q1", "q2"]
    assert snapshot.metrics["ndcg@10"] == 1.0
    assert snapshot.warmup == 3


def test_runner_retries_rate_limits_and_service_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = Suite.model_validate(
        {
            "dataset": "queries.jsonl",
            "retries": 2,
            "target": {"type": "http", "url": "http://search.test"},
        }
    )
    statuses = iter([429, 503, 200])
    delays: list[tuple[float, float]] = []

    def sample_delay(low: float, high: float) -> float:
        delays.append((low, high))
        return 0.0

    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), json={"results": [{"id": "BOLT-1"}]})

    monkeypatch.setattr(runner.random, "uniform", sample_delay)
    snapshot = asyncio.run(
        runner.run_suite(suite, [query()], transport=httpx.MockTransport(handle))
    )
    assert delays == [(0.025, 0.05), (0.05, 0.1)]
    assert snapshot.errors == 0
    assert snapshot.metrics["recall@10"] == 1.0


def test_runner_records_permanent_http_errors_without_retrying() -> None:
    suite = Suite.model_validate(
        {
            "dataset": "queries.jsonl",
            "retries": 1,
            "target": {"type": "http", "url": "http://search.test"},
        }
    )
    calls = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400)

    snapshot = asyncio.run(
        runner.run_suite(suite, [query()], transport=httpx.MockTransport(handle))
    )
    assert calls == snapshot.errors == 1
    assert snapshot.queries[0].error == "HTTP 400 Bad Request"
    assert snapshot.queries[0].latency_ms is None
    assert snapshot.metrics["recall@10"] == 0.0
