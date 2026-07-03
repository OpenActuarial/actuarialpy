"""Every example script must run cleanly; matplotlib-only demos skip."""
import os
import pathlib
import subprocess
import sys

import pytest

_EXAMPLES = sorted(
    p for p in (pathlib.Path(__file__).resolve().parents[1] / "examples").glob("*.py")
    if not p.name.startswith("_")
)


@pytest.mark.parametrize("script", _EXAMPLES, ids=[p.name for p in _EXAMPLES])
def test_example_runs(script):
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, timeout=240, env=env)
    if proc.returncode != 0 and "ModuleNotFoundError" in proc.stderr and "matplotlib" in proc.stderr:
        pytest.skip("optional matplotlib not installed")
    assert proc.returncode == 0, proc.stderr[-800:]
