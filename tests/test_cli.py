import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from ircheck.term import Terminal


def correct(query: str, k: int) -> list[str]:
    return [query][:k]


async def regressed(query: str, k: int) -> list[str]:
    return (["unrelated"] if query == "SKU-1" else [query])[:k]


def broken(query: str, k: int) -> list[str]:
    raise RuntimeError("retriever unavailable")


def invoke(
    *args: str | Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    tests = Path(__file__).resolve().parent
    source = tests.parent / "src"
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (source, tests)))}
    environment.pop("GITHUB_STEP_SUMMARY", None)
    return subprocess.run(
        [sys.executable, "-m", "ircheck", *map(str, args)],
        check=False,
        capture_output=True,
        text=True,
        env={**environment, **(env or {})},
        timeout=15,
    )


def write_suite(tmp_path: Path, function: str) -> Path:
    (tmp_path / "queries.jsonl").write_text(
        '{"id":"q1","query":"SKU-1","relevant":["SKU-1"],"tags":["sku"]}\n'
        '{"id":"q2","query":"steel bolt","relevant":{"steel bolt":3},"tags":["semantic"]}\n',
        encoding="utf-8",
    )
    path = tmp_path / f"{function}.yaml"
    path.write_text(
        f"dataset: queries.jsonl\nk: 2\ntarget:\n  type: python\n  callable: test_cli:{function}\n",
        encoding="utf-8",
    )
    return path


def test_run_save_and_diff(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    run = invoke("--json", "run", write_suite(tmp_path, "correct"), "--save-baseline", baseline)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == json.loads(baseline.read_text())
    assert json.loads(run.stdout)["metrics"]["ndcg@2"] == 1.0

    run = invoke(
        "run", write_suite(tmp_path, "regressed"), "--output", current, "--baseline", baseline
    )
    assert run.returncode == 1, run.stderr
    assert "WORST REGRESSIONS (top 1)" in run.stdout
    assert "rank: #1 -> >2" in run.stdout
    assert "\x1b" not in run.stdout

    diff = invoke("diff", baseline, current, "--json")
    assert diff.returncode == 1, diff.stderr
    report = json.loads(diff.stdout)
    assert report["failures"] == 8
    assert report["worst"][0]["id"] == "q1"
    assert invoke("diff", baseline, current, "--max-drop", "1").returncode == 0


def test_tag_filter_rescores_saved_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    assert (
        invoke("run", write_suite(tmp_path, "correct"), "--save-baseline", baseline).returncode == 0
    )
    suite = write_suite(tmp_path, "regressed")
    run = invoke("run", suite, "--tag", "semantic", "--baseline", baseline)
    assert run.returncode == 0, run.stderr
    assert "evaluated 1 queries" in run.stdout
    assert "semantic (n=1)" in run.stdout
    assert "sku (n=1)" not in run.stdout
    assert invoke("run", suite, "--tag", "missing").returncode == 2


def test_runtime_error_exits_two_without_saving_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    run = invoke("run", write_suite(tmp_path, "broken"), "--save-baseline", baseline, "--json")
    assert run.returncode == 2
    assert "retriever unavailable" in run.stderr
    assert json.loads(run.stdout)["queries"][0]["error"]
    assert not baseline.exists()


def test_no_color_for_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    class Interactive(StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    assert Terminal(Interactive()).status("FAIL") == "\x1b[31mFAIL\x1b[0m"
    assert Terminal(Interactive()).delta("+0.1000") == "\x1b[32m+0.1000\x1b[0m"
    assert Terminal(StringIO()).status("FAIL") == "FAIL"
    monkeypatch.setenv("NO_COLOR", "1")
    assert Terminal(Interactive()).status("FAIL") == "FAIL"
