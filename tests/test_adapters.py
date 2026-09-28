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


