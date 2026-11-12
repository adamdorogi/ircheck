from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

QUALITY_METRICS = ("ndcg", "mrr", "recall", "hitrate")
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
Grade = Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=0, le=1, strict=True)]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Query(ConfigModel):
    id: NonEmpty
    query: NonEmpty
    tags: list[NonEmpty] = Field(default_factory=list)
    relevant: list[NonEmpty] | dict[NonEmpty, Grade]

    @field_validator("query")
    @classmethod
    def query_has_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain text")
        return value

    @field_validator("tags")
    @classmethod
    def unique_tags(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class HttpTarget(ConfigModel):
    type: Literal["http"]
    url: HttpUrl
    method: Literal["GET", "POST"] = "POST"
    query_template: dict[str, JsonValue] | None = None
    params: dict[str, JsonValue] | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    id_path: NonEmpty = "results[].id"


class OpenSearchTarget(ConfigModel):
    type: Literal["opensearch"]
    url: HttpUrl
    index: NonEmpty
    query_template: dict[str, JsonValue]
    headers: dict[str, str] = Field(default_factory=dict)
    id_field: NonEmpty = "_id"
    id_path: NonEmpty | None = None


class PythonTarget(ConfigModel):
    type: Literal["python"]
    callable: Annotated[str, StringConstraints(pattern=r"^[\w.]+:[\w.]+$")]


Target = Annotated[HttpTarget | OpenSearchTarget | PythonTarget, Field(discriminator="type")]


class Suite(ConfigModel):
    dataset: Path
    target: Target
    k: int = Field(default=10, gt=0, strict=True)
    concurrency: int = Field(default=8, gt=0, strict=True)
    warmup: int = Field(default=0, ge=0, strict=True)
    timeout: float = Field(default=10.0, gt=0, strict=True)
    retries: int = Field(default=0, ge=0, strict=True)
    max_drop: Fraction = 0.02
    fail_under: dict[str, Fraction] = Field(default_factory=dict)
    tag_max_drop: dict[NonEmpty, Fraction] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_thresholds(self) -> Self:
        thresholds: dict[str, float] = {}
        for name, threshold in self.fail_under.items():
            metric = normalize_metric(name, self.k)
            if metric in thresholds:
                raise ValueError(f"duplicate threshold for {metric}")
            thresholds[metric] = threshold
        self.fail_under = thresholds
        return self


def normalize_metric(name: str, k: int) -> str:
    metric, separator, cutoff = name.strip().lower().partition("@")
    if metric not in QUALITY_METRICS:
        raise ValueError(f"unknown quality metric {name!r}")
    if separator and cutoff not in {"k", str(k)}:
        raise ValueError(f"metric {name!r} must use the suite cutoff @{k}")
    return f"{metric}@{k}"


def load_suite(path: Path) -> Suite:
    try:
        suite = Suite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        suite.dataset = (path.parent / suite.dataset).resolve()
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"{path}: {error}") from error
    return suite


def load_queries(path: Path) -> list[Query]:
    queries: list[Query] = []
    seen: set[str] = set()
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    query = Query.model_validate_json(line)
                    if query.id in seen:
                        raise ValueError(f"duplicate query id {query.id!r}")
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
                seen.add(query.id)
                queries.append(query)
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not queries:
        raise ValueError(f"{path}: dataset has no queries")
    return queries
