import csv
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_visual_strike_kinematic_hits_and_centers_target(tmp_path):
    log = tmp_path / "vs_kin.csv"
    plot = tmp_path / "vs_kin.png"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sim" / "visual_strike_kinematic.py"),
            "--target-north", "50",
            "--duration", "10",
            "--log", str(log),
            "--plot", str(plot),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hit            : True" in result.stdout
    assert "TERMINAL" in result.stdout

    with open(log, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    errors = [
        math.hypot(float(r["u_meas"]) - float(r["u_ref"]),
                   float(r["v_meas"]) - float(r["v_ref"]))
        for r in rows
        if r["u_meas"] and r["u_meas"].lower() != "nan"
    ]
    assert errors
    assert max(errors) < 5.0  # 目标始终保持在参考点附近
