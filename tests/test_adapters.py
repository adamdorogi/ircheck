import asyncio
import json

import httpx
from pydantic import HttpUrl

from ircheck.adapters.http import HttpAdapter
from ircheck.config import HttpTarget


def test_posts_query_and_reads_ranked_ids() -> None:
    async def run() -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content) == {"query": 'm8 "flange"', "k": 5}
            return httpx.Response(200, json={"results": [{"id": "bolt"}, {"id": "washer"}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            target = HttpTarget(type="http", url=HttpUrl("https://search.example.test/search"))
            assert await HttpAdapter(client, target).search('m8 "flange"', 5) == ["bolt", "washer"]

    asyncio.run(run())
