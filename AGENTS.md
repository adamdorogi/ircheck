# ircheck

Keep the diff to the issue at hand. Read the caller and the closest test
before changing a shared helper. Leave unrelated formatting for another patch.

## Things callers rely on

- CLI flags live in the stdlib `argparse` parser. No `rich`, `typer`, or `click`.
- Runtime deps are `httpx`, `pydantic`, and `pyyaml`. Retrieval math belongs
  in `metrics.py` and uses the stdlib. Leave `numpy`, `scipy`, `pandas`, and
  `scikit-learn` out.
- Deduplicate retrieved IDs in first-seen order before applying `k`.
- Substitute `$query` in parsed HTTP and OpenSearch templates. Let the JSON
  encoder escape quotes and control characters.
- Keep JSON stdout parseable and send actionable errors to stderr. Preserve
  exit codes `0` (pass), `1` (quality gate), and `2` (config/runtime error).
- Keep human output in plain columns. Honor `NO_COLOR` and non-TTY stdout.

## Before handing off

Test changes to flags, suite validation, and snapshots. Use `httpx.MockTransport`
for HTTP tests. Use hand-worked rankings for metric tests.

Run both checks from the repo root.

```sh
uv run ruff check
uv run pytest
```
