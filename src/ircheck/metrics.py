from collections.abc import Collection, Sequence
from math import log2


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        return 0.0
    ranked = unique_ids(ranked)
    return len(set(ranked[:k]).intersection(relevant)) / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    ranked = unique_ids(ranked)
    return next((1 / rank for rank, doc in enumerate(ranked[:k], 1) if doc in relevant), 0.0)


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    ranked = unique_ids(ranked)
    actual = sum(1 / log2(rank + 1) for rank, doc in enumerate(ranked[:k], 1) if doc in relevant)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return actual / ideal if ideal else 0.0


def unique_ids(hits: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(hits))


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    lower, remainder = divmod((len(ordered) - 1) * percent / 100, 1)
    index = int(lower)
    if not remainder:
        return ordered[index]
    return ordered[index] + remainder * (ordered[index + 1] - ordered[index])


def score_at_k(hits: Sequence[str], relevant: Collection[str], k: int) -> dict[str, float]:
    hits = unique_ids(hits)
    return {
        f"ndcg@{k}": ndcg_at_k(hits, relevant, k),
        f"mrr@{k}": mrr_at_k(hits, relevant, k),
        f"recall@{k}": recall_at_k(hits, relevant, k),
        f"hitrate@{k}": float(bool(set(hits[:k]).intersection(relevant))),
    }
