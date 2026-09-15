import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    relevant: list[str]


class HttpTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "http"
    url: HttpUrl
    id_path: str = "results[].id"


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Path
    target: HttpTarget
    k: int = Field(default=10, ge=1)
    concurrency: int = Field(default=8, ge=1)
    timeout: float = Field(default=10, gt=0)


def load_suite(path: Path) -> Suite:
    suite = Suite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return suite.model_copy(update={"dataset": path.parent / suite.dataset})


def load_queries(path: Path) -> list[Query]:
    queries: list[Query] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as rows:
        for line_no, row in enumerate(rows, 1):
            if not row.strip():
                continue
            try:
                query = Query.model_validate_json(row)
            except (ValidationError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if query.id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate query ID {query.id!r}")
            seen.add(query.id)
            queries.append(query)
    if not queries:
        raise ValueError(f"{path}: no queries")
    return queries
