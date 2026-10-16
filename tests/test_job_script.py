"""A failed GPU cell must propagate failure to the grid launcher."""

from pathlib import Path
import os
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("status", [0, 7])
def test_cell_exit_status(tmp_path, status):
    bash = shutil.which("bash")
    if os.name == "nt" and Path("D:/Git/bin/bash.exe").is_file():
        bash = "D:/Git/bin/bash.exe"
    if bash is None:
        pytest.skip("Bash is unavailable")
    source = (Path(__file__).parents[1] / "job.sh").read_text(encoding="utf-8")
    functions = source[source.index("log_line() {"):source.index('case "${1:-grid4}"')]
    script = (
        "set -euo pipefail\nexport PYTHONUNBUFFERED=1\nmkdir -p logs\n"
        + functions
        + f"\npython3() {{ return {status}; }}\nrun_one 0 6 1 0 hard\n"
    )
    path = tmp_path / "cell.sh"
    path.write_text(script, encoding="utf-8", newline="\n")
    result = subprocess.run([bash, "cell.sh"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == status, result.stdout + result.stderr
    if status:
        assert f"FAILED rc={status}" in result.stdout
