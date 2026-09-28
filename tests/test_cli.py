import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from ircheck.config import load_queries, load_suite
from ircheck.term import Terminal


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


def test_diff_exit_codes(tmp_path: Path) -> None:
    from test_diff import snapshot

    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(snapshot([1.0]).model_dump_json(), encoding="utf-8")
    current.write_text(snapshot([0.0]).model_dump_json(), encoding="utf-8")
    diff = invoke("diff", baseline, current, "--json")
    assert diff.returncode == 1, diff.stderr
    assert json.loads(diff.stdout)["worst"][0]["id"] == "q0"
    assert invoke("diff", baseline, current, "--max-drop", "1").returncode == 0
    current.write_text("{", encoding="utf-8")
    assert invoke("diff", baseline, current).returncode == 2


def test_init_writes_valid_suite(tmp_path: Path) -> None:
    new = tmp_path / "new"
    assert invoke("init", new).returncode == 0
    assert load_queries(load_suite(new / "suite.yaml").dataset)[0].id == "q01"
    assert invoke("init", new).returncode == 2
