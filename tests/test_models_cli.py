"""Lightweight CLI startup tests: --help must never train or load data."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mlb_quant.models.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _artifact_snapshot() -> dict[str, float]:
    artifact_root = REPO_ROOT / "models" / "artifacts"
    return {
        str(path): path.stat().st_mtime
        for path in artifact_root.glob("mlb_*_quant_v0_1.joblib")
    }


def test_help_exits_quickly_without_training_or_data_loading():
    before = _artifact_snapshot()
    start = time.monotonic()
    result = _run("--help", timeout=5.0)
    elapsed = time.monotonic() - start
    after = _artifact_snapshot()
    assert result.returncode == 0
    assert "tournament" in result.stdout
    assert elapsed < 5.0
    assert before == after


def test_tournament_subcommand_help_exits_quickly_without_side_effects():
    before = _artifact_snapshot()
    result = _run("tournament", "--help", timeout=5.0)
    after = _artifact_snapshot()
    assert result.returncode == 0
    assert "--input-path" in result.stdout
    assert before == after


def test_missing_subcommand_exits_with_usage_error_and_no_side_effects():
    before = _artifact_snapshot()
    result = _run(timeout=5.0)
    after = _artifact_snapshot()
    assert result.returncode == 2
    assert before == after


def test_importing_cli_module_never_pulls_in_heavy_dependencies():
    check = subprocess.run(
        [
            sys.executable, "-c",
            "import sys, mlb_quant.models.cli; "
            "heavy = [m for m in sys.modules if m.split('.')[0] in "
            "('sklearn', 'pandas', 'numpy', 'joblib')]; "
            "print(sorted(heavy))",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    assert check.returncode == 0
    assert check.stdout.strip() == "[]"
