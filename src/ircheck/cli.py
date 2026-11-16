import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile

from ircheck import __version__
from ircheck.config import QUALITY_METRICS, Query, Suite, load_queries, load_suite, normalize_metric
from ircheck.diff import Comparison, Threshold, check_thresholds, compare_snapshots
from ircheck.runner import Snapshot, run_suite, summarize
from ircheck.term import Terminal, append_summary, printable


def _fraction(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1") from None
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ircheck", description="Check search retrieval quality.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--json", action="store_true", help="write JSON to stdout")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="evaluate one suite")
    run.add_argument("suite", type=Path)
    run.add_argument("--baseline", type=Path, help="compare with a saved snapshot")
    run.add_argument("--save-baseline", type=Path, help="save a complete run for later comparison")
    run.add_argument("--output", type=Path, help="write the run snapshot to a file")
    run.add_argument("--concurrency", type=int, help="override maximum concurrent queries")
    run.add_argument("--warmup", type=int, help="override number of unmeasured warmup queries")
    run.add_argument(
        "--fail-under",
        action="append",
        default=[],
        metavar="METRIC=MINIMUM",
        help="require an overall score, repeatable",
    )

    diff = commands.add_parser("diff", help="compare two saved snapshots")
    diff.add_argument("baseline", type=Path)
    diff.add_argument("current", type=Path)

    sweep = commands.add_parser("sweep", aliases=["compare"], help="compare multiple suites")
    sweep.add_argument("suites", nargs="+", type=Path, metavar="SUITE")
    sweep.add_argument("--output", type=Path, help="write the runs and comparisons to a file")

    init = commands.add_parser("init", help="write a starter suite and query set")
    init.add_argument("directory", nargs="?", type=Path, default=Path.cwd())

    for command in (run, sweep):
        command.add_argument(
            "--tag", action="append", default=[], help="select queries by tag, repeatable"
        )
    for command in (run, diff, sweep):
        command.add_argument(
            "--max-drop",
            type=_fraction,
            help="default allowed loss in a quality score (suite tag overrides apply)",
        )
        command.add_argument(
            "--json", action="store_true", default=argparse.SUPPRESS, help="write JSON to stdout"
        )
        command.add_argument(
            "--gh-summary", action="store_true", help="append a GitHub Actions step summary"
        )
    return parser


def _load_snapshot(path: Path) -> Snapshot:
    try:
        return Snapshot.model_validate_json(path.read_bytes())
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _write(path: Path, data: str) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _select(suite: Suite, tags: Sequence[str]) -> list[Query]:
    queries = load_queries(suite.dataset)
    unknown = set(tags) - {tag for query in queries for tag in query.tags}
    if unknown:
        raise ValueError(f"no queries with tag {', '.join(sorted(unknown))}")
    return [query for query in queries if not tags or set(tags).intersection(query.tags)]


def _slice_snapshot(snapshot: Snapshot, tags: Sequence[str]) -> Snapshot:
    if not tags:
        return snapshot
    queries = [query for query in snapshot.queries if set(tags).intersection(query.tags)]
    metrics, groups = summarize(queries, snapshot.k)
    return snapshot.model_copy(update={"queries": queries, "metrics": metrics, "tags": groups})


def _parse_thresholds(suite: Suite, specifications: Sequence[str]) -> dict[str, float]:
    minimums = dict(suite.fail_under)
    for specification in specifications:
        name, separator, value = specification.partition("=")
        if not separator:
            raise ValueError(f"invalid threshold {specification!r}. Use METRIC=MINIMUM")
        try:
            minimums[normalize_metric(name, suite.k)] = _fraction(value)
        except (ValueError, argparse.ArgumentTypeError) as error:
            raise ValueError(f"invalid threshold {specification!r}: {error}") from error
    return minimums


def _rank(rank: int | None, k: int) -> str:
    return f"#{rank}" if rank is not None else f">{k}"


def _delta_text(delta: float) -> str:
    value = f"{delta:+.4f}"
    return "0.0000" if value in {"+0.0000", "-0.0000"} else value


def _print_comparison(terminal: Terminal, report: Comparison) -> None:
    rows = [
        (
            row.metric,
            f"{row.baseline:.4f}",
            f"{row.current:.4f}",
            _delta_text(row.delta),
            f"FAIL (max drop {row.max_drop:.4f})" if row.failed else "PASS",
        )
        for row in report.overall
    ]
    terminal.table(
        ("METRIC", "BASELINE", "CURRENT", "DELTA", "GATE"),
        rows,
        right=(1, 2, 3),
        gate=True,
        delta=3,
    )
    if report.tags:
        terminal.line()
        terminal.line(terminal.dim(f"BY TAG (ndcg@{report.k})"))
        labels = [f"{printable(tag.tag)} (n={tag.count})" for tag in report.tags]
        width = max(map(len, labels))
        for label, tag in zip(labels, report.tags, strict=True):
            score = tag.metrics[0]
            failed = ", ".join(row.metric for row in tag.metrics if row.failed)
            note = f"  <-- regression ({failed})" if failed else ""
            delta = terminal.delta(_delta_text(score.delta))
            terminal.line(
                f"  {label:<{width}}  {score.baseline:.4f} -> {score.current:.4f}  {delta}{note}"
            )
    if report.worst:
        terminal.line()
        terminal.line(terminal.dim(f"WORST REGRESSIONS (top {len(report.worst)})"))
        for query in report.worst:
            tags = ", ".join(map(printable, query.tags))
            text = printable(json.dumps(query.query, ensure_ascii=False))
            terminal.line(
                f"  [{printable(query.id)}] {text}" + (f" (tags: {tags})" if tags else "")
            )
            terminal.line(
                f"       ndcg@{report.k}: {query.baseline_score:.3f} -> {query.current_score:.3f}"
                f" | rank: {_rank(query.baseline_rank, report.k)} -> {_rank(query.current_rank, report.k)}"
            )
            terminal.line(
                f"       expected: {', '.join(map(printable, query.expected)) or '(none)'}"
            )
            terminal.line(
                f"       top-3:    {', '.join(map(printable, query.top_hits)) or '(none)'}"
            )


def _print_run(terminal: Terminal, snapshot: Snapshot) -> None:
    terminal.line(
        f"ircheck {__version__}  target={printable(snapshot.target)} "
        f"(k={snapshot.k}, concurrency={snapshot.concurrency}, warmup={snapshot.warmup})"
    )
    latency = []
    for label in ("p50", "p95"):
        value = snapshot.metrics.get(f"{label}_ms")
        latency.append(f"{label}: {value:.1f}ms" if value is not None else f"{label}: n/a")
    terminal.line(
        f"evaluated {len(snapshot.queries)} queries in {snapshot.elapsed_s:.2f}s  "
        f"({', '.join(latency)}, errors: {snapshot.errors})"
    )
    terminal.line()


def _print_scores(terminal: Terminal, snapshot: Snapshot) -> None:
    metrics = [f"{metric}@{snapshot.k}" for metric in QUALITY_METRICS]
    terminal.table(
        ("METRIC", "CURRENT"),
        [(metric, f"{snapshot.metrics[metric]:.4f}") for metric in metrics],
        right=(1,),
    )
    for tag, scores in sorted(snapshot.tags.items()):
        metric = f"ndcg@{snapshot.k}"
        terminal.line(
            f"  {printable(tag)} (n={scores.count})  {metric}={scores.metrics[metric]:.4f}"
        )


def _print_thresholds(terminal: Terminal, thresholds: Sequence[Threshold]) -> None:
    if thresholds:
        terminal.line()
        terminal.table(
            ("METRIC", "CURRENT", "MINIMUM", "GATE"),
            [
                (
                    row.metric,
                    f"{row.current:.4f}",
                    f"{row.minimum:.4f}",
                    "FAIL" if row.failed else "PASS",
                )
                for row in thresholds
            ],
            right=(1, 2),
            gate=True,
        )


def _finish(terminal: Terminal, failures: int) -> int:
    terminal.line()
    if failures:
        noun = "gate" if failures == 1 else "gates"
        terminal.line(terminal.status(f"FAIL: {failures} metric {noun} failed (exit 1)"))
        return 1
    terminal.line(terminal.status("PASS"))
    return 0


def _markdown_label(value: str) -> str:
    return (
        escape(printable(value))
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _summary(
    snapshot: Snapshot,
    report: Comparison | None,
    thresholds: Sequence[Threshold] = (),
) -> str:
    lines = [
        f"### ircheck {_markdown_label(snapshot.target)}",
        "",
        "| Scope | Metric | Reference | Current | Delta | Gate |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    if report is None:
        for metric, value in snapshot.metrics.items():
            lines.append(f"| overall | {metric} | | {value:.4f} | | |")
    else:
        scopes = [("overall", report.overall)] + [
            (f"{_markdown_label(tag.tag)} (n={tag.count})", tag.metrics) for tag in report.tags
        ]
        for scope, metrics in scopes:
            for row in metrics:
                gate = "FAIL" if row.failed else "PASS"
                lines.append(
                    f"| {scope} | {row.metric} | {row.baseline:.4f} | {row.current:.4f} | {_delta_text(row.delta)} | {gate} |"
                )
    for threshold in thresholds:
        gate = "FAIL" if threshold.failed else "PASS"
        lines.append(
            f"| minimum | {threshold.metric} | {threshold.minimum:.4f} | {threshold.current:.4f} | | {gate} |"
        )
    if snapshot.errors:
        lines.extend(("", f"Query errors: {snapshot.errors} (exit 2)."))
    return "\n".join(lines)


def _report_errors(snapshot: Snapshot) -> None:
    for query in snapshot.queries:
        if query.error is not None:
            print(f"ircheck: [{printable(query.id)}] {printable(query.error)}", file=sys.stderr)


def _run(args: argparse.Namespace, terminal: Terminal) -> int:
    suite = load_suite(args.suite)
    queries = _select(suite, args.tag)
    minimums = _parse_thresholds(suite, args.fail_under)
    baseline = _slice_snapshot(_load_snapshot(args.baseline), args.tag) if args.baseline else None
    snapshot = asyncio.run(
        run_suite(suite, queries, concurrency=args.concurrency, warmup=args.warmup)
    )
    report = None
    thresholds: tuple[Threshold, ...] = ()
    if not snapshot.errors:
        thresholds = check_thresholds(snapshot, minimums)
        if baseline is not None:
            report = compare_snapshots(
                baseline,
                snapshot,
                max_drop=suite.max_drop if args.max_drop is None else args.max_drop,
                tag_max_drop=suite.tag_max_drop,
            )
    data = snapshot.model_dump_json(indent=2)
    if args.output:
        _write(args.output, data)
    if args.save_baseline and not snapshot.errors:
        _write(args.save_baseline, data)
    if args.gh_summary:
        append_summary(_summary(snapshot, report, thresholds))
    if args.json:
        terminal.line(data)
    else:
        _print_run(terminal, snapshot)
        if report is not None:
            _print_comparison(terminal, report)
        elif not snapshot.errors:
            _print_scores(terminal, snapshot)
        _print_thresholds(terminal, thresholds)
    if snapshot.errors:
        _report_errors(snapshot)
        return 2
    failures = (report.failures if report is not None else 0) + sum(
        row.failed for row in thresholds
    )
    return int(bool(failures)) if args.json else _finish(terminal, failures)


def _report_dict(report: Comparison) -> dict[str, object]:
    return {**asdict(report), "failures": report.failures}


def _diff(args: argparse.Namespace, terminal: Terminal) -> int:
    baseline = _load_snapshot(args.baseline)
    current = _load_snapshot(args.current)
    report = compare_snapshots(
        baseline, current, max_drop=0.02 if args.max_drop is None else args.max_drop
    )
    if args.gh_summary:
        append_summary(_summary(current, report))
    if args.json:
        terminal.line(json.dumps(_report_dict(report), indent=2))
        return int(bool(report.failures))
    terminal.line(
        f"ircheck {__version__}  {printable(baseline.target)} -> {printable(current.target)} (k={report.k})"
    )
    terminal.line()
    _print_comparison(terminal, report)
    return _finish(terminal, report.failures)


async def _run_sweep(suites: Sequence[Suite], tags: Sequence[str]) -> list[Snapshot]:
    return [await run_suite(suite, _select(suite, tags)) for suite in suites]


def _sweep(args: argparse.Namespace, terminal: Terminal) -> int:
    if len(args.suites) < 2:
        raise ValueError("sweep needs at least two suites. The first is the reference")
    suites = [load_suite(path) for path in args.suites]
    snapshots = asyncio.run(_run_sweep(suites, args.tag))
    reference = snapshots[0]
    comparisons = []
    thresholds = []
    for snapshot, suite in zip(snapshots, suites, strict=True):
        report = None
        limits = () if snapshot.errors else check_thresholds(snapshot, suite.fail_under)
        if snapshot is not reference and not snapshot.errors and not reference.errors:
            report = compare_snapshots(
                reference,
                snapshot,
                max_drop=suite.max_drop if args.max_drop is None else args.max_drop,
                tag_max_drop=suite.tag_max_drop,
            )
        comparisons.append(report)
        thresholds.append(limits)
    if args.json or args.output:
        payload = {
            "runs": [snapshot.model_dump(mode="json") for snapshot in snapshots],
            "comparisons": [
                _report_dict(report) if report is not None else None for report in comparisons
            ],
            "thresholds": [[asdict(row) for row in limits] for limits in thresholds],
        }
        data = json.dumps(payload, indent=2)
        if args.output:
            _write(args.output, data)
        if args.json:
            terminal.line(data)

    failures = 0
    runtime_error = False
    for snapshot, report, limits in zip(snapshots, comparisons, thresholds, strict=True):
        if args.gh_summary:
            append_summary(_summary(snapshot, report, limits))
        if not args.json:
            _print_run(terminal, snapshot)
            if report is not None:
                _print_comparison(terminal, report)
            elif not snapshot.errors:
                _print_scores(terminal, snapshot)
            _print_thresholds(terminal, limits)
            terminal.line()
        if snapshot.errors:
            _report_errors(snapshot)
            runtime_error = True
        failures += (report.failures if report is not None else 0) + sum(
            row.failed for row in limits
        )
    if runtime_error:
        return 2
    return int(bool(failures)) if args.json else _finish(terminal, failures)


def _init(directory: Path, terminal: Terminal) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    suite = directory / "suite.yaml"
    queries = directory / "queries.jsonl"
    if suite.exists() or suite.is_symlink() or queries.exists() or queries.is_symlink():
        raise ValueError(f"{directory} already contains suite.yaml or queries.jsonl")
    with suite.open("x", encoding="utf-8") as stream:
        stream.write(
            "dataset: queries.jsonl\nk: 10\n"
            "target:\n"
            "  type: http\n  url: http://127.0.0.1:8089/search\n  method: POST\n"
            "  query_template:\n    query: $query\n    k: $k\n  id_path: results[].id\n"
        )
    try:
        with queries.open("x", encoding="utf-8") as stream:
            stream.write(
                '{"id":"q01","query":"stainless hex bolt","tags":["sku"],"relevant":["SKU-1"]}\n'
            )
    except OSError:
        suite.unlink()
        raise
    terminal.line(f"wrote {printable(str(suite))} and {printable(str(queries))}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    terminal = Terminal(sys.stdout)
    try:
        if args.command == "run":
            return _run(args, terminal)
        if args.command == "diff":
            return _diff(args, terminal)
        if args.command in {"sweep", "compare"}:
            return _sweep(args, terminal)
        return _init(args.directory, terminal)
    except (OSError, ValueError) as error:
        print(f"ircheck: {error}", file=sys.stderr)
        return 2
