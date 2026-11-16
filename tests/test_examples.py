"""Smoke run of the two examples with tiny settings (a few training steps): they must run from the repository root and write their figures. Not a check of any result."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script, tmp_path, *args):
    out = tmp_path / "figure.png"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / script), "--out", str(out), *args], cwd=ROOT, capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": str(ROOT), "JAX_PLATFORMS": "cpu"})
    assert r.returncode == 0, r.stderr[-1500:]
    assert out.exists()
    return r.stdout


def test_uq_workflow(tmp_path):
    assert "MoLA mean equals DE mean" in run("01_uq_workflow.py", tmp_path, "--steps", "20")


def test_active_learning(tmp_path):
    out = run("02_active_learning.py", tmp_path, "--steps", "20", "--n-max", "2", "--methods", "DE", "LA", "--random-paths", "1")
    assert "n_D=2" in out and "Random 1" in out
