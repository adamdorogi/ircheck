# ircheck

A small Python tool for checking ranked retrieval results against known
relevant documents. Start with Recall@k, MRR@k, and nDCG@k.

## Development

```sh
uv sync
```

The package lives under `src/ircheck`. Metric functions take a ranked list
of document IDs, a set of relevant IDs, and a positive cutoff `k`.

HTTP targets, query sets, and snapshot comparisons will use the same metric
functions.
