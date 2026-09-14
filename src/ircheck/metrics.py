from collections.abc import Collection, Sequence
from math import log2


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]).intersection(relevant)) / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    return next((1 / rank for rank, doc in enumerate(ranked[:k], 1) if doc in relevant), 0.0)


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    actual = sum(1 / log2(rank + 1) for rank, doc in enumerate(ranked[:k], 1) if doc in relevant)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return actual / ideal if ideal else 0.0
