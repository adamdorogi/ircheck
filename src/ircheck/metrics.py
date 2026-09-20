import math
from collections.abc import Iterable, Mapping, Sequence

Relevance = Iterable[str] | Mapping[str, float]


def unique_ids(hits: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(hits))


def positive_grades(relevant: Relevance) -> dict[str, float]:
    if isinstance(relevant, Mapping):
        return {doc_id: grade for doc_id, grade in relevant.items() if grade > 0}
    return dict.fromkeys(relevant, 1.0)


def first_relevant_rank(hits: Sequence[str], relevant: Mapping[str, float]) -> int | None:
    return next((rank for rank, doc_id in enumerate(hits, 1) if doc_id in relevant), None)


def _ndcg(hits: Sequence[str], grades: Mapping[str, float], k: int) -> float:
    if not grades or not hits:
        return 0.0
    ideal = sorted(grades.values(), reverse=True)[:k]
    max_grade = ideal[0]

    def dcg(ranked_grades: Iterable[float]) -> float:
        # Scale both DCGs by 2**-max_grade to keep large grades finite.
        return math.fsum(
            math.exp2(grade - max_grade) * -math.expm1(-grade * math.log(2)) / math.log2(rank + 1)
            for rank, grade in enumerate(ranked_grades, 1)
        )

    return dcg(grades.get(doc_id, 0.0) for doc_id in hits) / dcg(ideal)


def score_at_k(hits: Iterable[str], relevant: Relevance, k: int) -> dict[str, float]:
    grades = positive_grades(relevant)
    ranked = unique_ids(hits)[: max(0, k)]
    rank = first_relevant_rank(ranked, grades)
    return {
        f"ndcg@{k}": _ndcg(ranked, grades, k),
        f"mrr@{k}": 1 / rank if rank else 0.0,
        f"recall@{k}": sum(doc_id in grades for doc_id in ranked) / len(grades) if grades else 0.0,
        f"hitrate@{k}": float(rank is not None),
    }


def recall_at_k(hits: Iterable[str], relevant: Relevance, k: int) -> float:
    return score_at_k(hits, relevant, k)[f"recall@{k}"]


def mrr_at_k(hits: Iterable[str], relevant: Relevance, k: int) -> float:
    return score_at_k(hits, relevant, k)[f"mrr@{k}"]


def ndcg_at_k(hits: Iterable[str], relevant: Relevance, k: int) -> float:
    return score_at_k(hits, relevant, k)[f"ndcg@{k}"]


def hit_rate_at_k(hits: Iterable[str], relevant: Relevance, k: int) -> float:
    return score_at_k(hits, relevant, k)[f"hitrate@{k}"]


def percentile(values: Iterable[float], percent: float) -> float:
    if not 0 <= percent <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
