import re
from collections.abc import Iterable, Mapping

import httpx
from pydantic import JsonValue

from ..config import HttpTarget

DEFAULT_TEMPLATE: dict[str, JsonValue] = {"query": "$query", "k": "$k"}
PLACEHOLDER = re.compile(r"\$(query|k)\b")
Param = str | int | float | bool | None


def render_template(value: JsonValue, query: str, k: int) -> JsonValue:
    if isinstance(value, dict):
        return {key: render_template(item, query, k) for key, item in value.items()}
    if isinstance(value, list):
        return [render_template(item, query, k) for item in value]
    if value == "$k":
        return k
    if isinstance(value, str):
        return PLACEHOLDER.sub(lambda match: query if match[1] == "query" else str(k), value)
    return value


def _render_params(
    template: Mapping[str, JsonValue], query: str, k: int
) -> list[tuple[str, Param]]:
    params: list[tuple[str, Param]] = []
    for key, value in template.items():
        rendered = render_template(value, query, k)
        for item in rendered if isinstance(rendered, list) else [rendered]:
            if isinstance(item, (dict, list)):
                raise ValueError(f"request parameter {key!r} must contain only scalar values")
            params.append((key, item))
    return params


def _parse_path(path: str) -> list[str | None]:
    steps: list[str | None] = []
    for part in path.split("."):
        key = part.removesuffix("[]")
        if not part or "[" in key or "]" in key:
            raise ValueError(f"invalid dotted ID path {path!r}")
        if key:
            steps.append(key)
        if part.endswith("[]"):
            steps.append(None)
    return steps


def normalize_ids(values: Iterable[object]) -> list[str]:
    ids: list[str] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
            raise ValueError("document IDs must be non-empty strings or integers")
        ids.append(str(value))
    return ids


def _extract_ids(response: object, path: str, steps: list[str | None]) -> list[str]:
    selected = [response]
    for step in steps:
        next_values: list[object] = []
        for value in selected:
            if isinstance(value, list) and step is None:
                next_values.extend(value)
            elif isinstance(value, dict) and step is not None and step in value:
                next_values.append(value[step])
            else:
                raise ValueError(f"response does not match ID path {path!r}")
        selected = next_values
    return normalize_ids(
        item for value in selected for item in (value if isinstance(value, list) else [value])
    )


def extract_ids(response: object, path: str) -> list[str]:
    return _extract_ids(response, path, _parse_path(path))


class HttpAdapter:
    def __init__(self, client: httpx.AsyncClient, target: HttpTarget) -> None:
        self.client = client
        self.target = target
        self.steps = _parse_path(target.id_path)

    async def search(self, query: str, k: int) -> list[str]:
        template = self.target.query_template
        if template is None:
            template = DEFAULT_TEMPLATE
        params = self.target.params
        if self.target.method == "GET" and params is None:
            params = template
        url = httpx.URL(str(self.target.url))
        if params is not None:
            url = url.copy_merge_params(_render_params(params, query, k))
        response = await self.client.request(
            self.target.method,
            url,
            headers=self.target.headers,
            json=render_template(template, query, k) if self.target.method == "POST" else None,
        )
        response.raise_for_status()
        return _extract_ids(response.json(), self.target.id_path, self.steps)
