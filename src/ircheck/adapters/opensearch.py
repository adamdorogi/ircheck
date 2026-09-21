from urllib.parse import quote

import httpx
from pydantic import HttpUrl

from ..config import HttpTarget, OpenSearchTarget
from .http import HttpAdapter


class OpenSearchAdapter(HttpAdapter):
    def __init__(self, client: httpx.AsyncClient, target: OpenSearchTarget) -> None:
        index = quote(target.index, safe="*,-_.")
        endpoint = httpx.URL(str(target.url))
        url = endpoint.copy_with(path=f"{endpoint.path.rstrip('/')}/{index}/_search")
        super().__init__(
            client,
            HttpTarget(
                type="http",
                url=HttpUrl(str(url)),
                query_template=target.query_template,
                headers=target.headers,
                id_path=target.id_path or f"hits.hits[].{target.id_field}",
            ),
        )
