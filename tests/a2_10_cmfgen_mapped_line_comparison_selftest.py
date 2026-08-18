#!/usr/bin/env python3
"""Known-answer and negative controls for the mapped-line comparator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/compare_a210_cmfgen_mapped_line.py"


def line(phase: str, temperature: float, requested: int = 1) -> str:
    return (
        "[A2-10][LINE-NET-CELL-FINITE] "
        f"phase={phase} line=7 shell=0 T_e_K={temperature} Z=27 ion=3 "
        "n_e_cm3=30 n_atom_cm3=15 "
        "ion_slot=2 tau_raw=1 tau_validity=1 n_upper=1 A_ul=2 nu=3 "
        "chi_raw=2 chi_effective=2 srce_chk=0 eta_per_sr=10 Jbar=4 "
        "Jbar_local_bound=1e-12 absorption_per_sr=8 net_per_sr=2 "
        "signed_rate=25.132741228718345 uncertainty=1e-11 "
        "cancellation_condition=9 status=OK_COOLING exact_zero=0 "
        "deck_scale=1 source_function=5 jbar_over_source=0.8 "
        f"requested_cell={requested} clamp=0 floor=0 jitter=0"
    )


def run(directory: Path, log: str) -> tuple[int, dict[str, object]]:
    stderr = directory / "stderr.log"
    report = directory / "report.json"
    stderr.write_text(log + "\n", encoding="utf-8")
    result = subprocess.run(
        (
            sys.executable, str(SCRIPT),
            "--reference", str(directory / "reference.json"),
            "--stderr", str(stderr),
            "--line", "7", "--shell", "0", "--report", str(report),
        ),
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return result.returncode, json.loads(report.read_text(encoding="utf-8"))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a210-cmfgen-line-") as raw:
        directory = Path(raw)
        reference = {
            "mapping": {"lumina_line_id_zero_based": 7},
            "depths": {
                "67": {
                    "eta_per_sr_cgs": 5.0,
                    "signed_rate_cgs": 1.0,
                    "jbar_over_source": 0.999,
                    "cancellation_condition": 1000.0,
                }
            },
            "physical_mutation": 0,
            "floor": 0, "cap": 0, "clamp": 0, "jitter": 0, "repair": 0,
        }
        (directory / "reference.json").write_text(
            json.dumps(reference), encoding="utf-8"
        )
        valid = "\n".join((
            line("LOWER", 3500), line("UPPER", 140000),
            line("INTERIOR", 10020), line("INTERIOR", 22135),
        ))
        rc, report = run(directory, valid)
        assert rc == 0 and report["status"] == "PASS"
        comparisons = report["comparisons"]
        assert [row["lumina"]["phase"] for row in comparisons] == [
            "LOWER", "UPPER", "PUBLIC_SEED", "GEOMETRIC_MID"
        ]
        assert comparisons[0]["lumina"]["jbar_over_source"] == 0.8
        assert comparisons[0]["lumina"]["electron_density_cm3"] == 30.0
        assert comparisons[0]["lumina"]["atom_density_cm3"] == 15.0
        assert comparisons[0]["lumina"]["logged_deck_scale"] == 1.0
        controls = {
            "missing": "\n".join(valid.splitlines()[:-1]),
            "unrequested": valid.replace("requested_cell=1", "requested_cell=0", 1),
            "repair": valid.replace("floor=0", "floor=1", 1),
            "closure": valid.replace("net_per_sr=2", "net_per_sr=3", 1),
        }
        for name, text in controls.items():
            rc, report = run(directory, text)
            assert rc == 4 and report["status"] == "FAIL", name
    print(
        "a2_10_cmfgen_mapped_line_comparison_selftest: PASS "
        "positive=1 negative_controls=4 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
