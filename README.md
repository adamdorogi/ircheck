# ircheck

Run a list of judged queries against a search endpoint and compare retrieval
quality with a saved run. Results include Recall@k, MRR@k, nDCG@k, and HitRate@k.

## Local demo

Start the demo server, then run the suite against the saved BM25 results.

```sh
uv run python examples/mock_server.py
uv run ircheck run examples/suite.yaml --baseline examples/baseline.json
```

The hybrid fixture pushes down some part-number matches on purpose. The run exits
1 and shows the affected tags and queries.

## Run your suite

```sh
uv run ircheck init ./benchmark
uv run ircheck run benchmark/suite.yaml --save-baseline baseline.json
uv run ircheck run benchmark/suite.yaml --output current.json
uv run ircheck diff baseline.json current.json --max-drop 0.02
```

Each JSONL row contains `id`, `query`, `tags`, and `relevant`. Relevance can be a
list of document IDs or a mapping of IDs to grades. Set the dataset path relative
to the suite file. See `examples/suite.yaml` for a generic HTTP target. OpenSearch
targets take an index and a JSON query template.

Use `--json` for machine-readable output. Exit 0 means pass, 1 means a quality
gate failed, and 2 means the run could not complete.
