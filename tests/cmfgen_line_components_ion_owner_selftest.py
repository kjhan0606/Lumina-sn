#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/summarize_cmfgen_line_components_by_ion.py"


def run(root: Path, lineheat: str, netrate: str) -> subprocess.CompletedProcess[str]:
    line_path = root / "LINEHEAT"
    net_path = root / "NETRATE"
    reference = root / "finite.json"
    line_path.write_text(lineheat)
    net_path.write_text(netrate)
    reference.write_text(json.dumps({"depths": {
        "1": {"signed_internal": 1.0, "absolute_internal": 3.0},
        "2": {"signed_internal": 3.0, "absolute_internal": 5.0},
    }}))
    return subprocess.run([
        "python3", str(SCRIPT), "--lineheat", str(line_path),
        "--netrate", str(net_path), "--depth-count", "2",
        "--depth", "1", "--depth", "2", "--scale-threshold", ".5",
        "--finite-reference",
        str(reference), "--json-out", str(root / "report.json"),
    ], cwd=ROOT, text=True, capture_output=True)


def main() -> int:
    lineheat = (
        " 1 CoIV(test) 10.0 1 2 0.5\n 2.0 4.0\n"
        " 2 FeIII(test) 9.0 3 4 -1.0\n -1.0 -1.0\n"
    )
    netrate = (
        " 1 CoIV(test) 10.0 1 2 11 12\n .5 .25\n"
        " 2 FeIII(test) 9.0 3 4 13 14\n -.25 -.5\n"
    )
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        result = run(root, lineheat, netrate)
        if result.returncode != 0:
            raise SystemExit(result.stdout + result.stderr)
        report = json.loads((root / "report.json").read_text())
        depth1 = report["depths"]["1"]
        if (report["paired_line_records"] != 2 or
                depth1["line_order_scaled_emission_internal"] != 8.0 or
                depth1["line_order_scaled_absorption_internal"] != 7.0 or
                depth1["line_order_signed_internal"] != 1.0 or
                not depth1["cellwise_component_closure_verified"] or
                abs(depth1["cellwise_component_closure_internal"]) >
                depth1["cellwise_component_closure_bound_internal"] or
                not depth1["finite_reference_check"]["signed_bit_exact"]):
            raise SystemExit("positive component decomposition mismatch")

        mismatch = netrate.replace("2 FeIII", "3 FeIII")
        if run(root, lineheat, mismatch).returncode == 0:
            raise SystemExit("header mismatch accepted")
        zero_znet = netrate.replace(".5 .25", "0 .25")
        if run(root, lineheat, zero_znet).returncode == 0:
            raise SystemExit("zero ZNET accepted")
        negative_emission = netrate.replace(".5 .25", "-.5 .25")
        if run(root, lineheat, negative_emission).returncode == 0:
            raise SystemExit("negative derived emission accepted")

    print("PASS cmfgen_line_components_ion_owner positive+3_negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
