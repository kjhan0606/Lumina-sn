#!/usr/bin/env python3
"""Offline round-trip + negative-control battery for the LCMFLP01 line dump.

Builds nothing but the fixture writer (`make selftest_cmf_linepop_dump`), runs
it, and asserts that the fail-closed contracts actually fire.  No model, no GPU,
no plasma solve.  Every control is seeded, i.e. the failure is injected on
purpose and the battery FAILS if the checker does not catch it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_linepop_check import check_artifact, LinePopError  # noqa: E402

BIN = ROOT / "selftest_cmf_linepop_dump"


def run(env: dict, out: Path) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update(env)
    return subprocess.run([str(BIN), str(out)], env=e, capture_output=True,
                          text=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if not args.skip_build:
        subprocess.run(["make", "-B", "selftest_cmf_linepop_dump"],
                       cwd=ROOT, check=True, capture_output=True, text=True)
    work = Path(tempfile.mkdtemp(prefix="cmf_linepop_rt_", dir="/tmp"))
    results: dict[str, object] = {}
    try:
        sel = {"LUMINA_CMF_LINEPOP_SHELLS": "0,1"}

        # (1) positive: contract generation, bitwise round trip
        good = work / "lp.bin"
        proc = run(sel, good)
        if proc.returncode != 0:
            raise RuntimeError(f"reference write failed: {proc.stderr}")
        lp = check_artifact(good)
        if not lp.manifest["chi_line_roundtrip_bitwise"]:
            raise RuntimeError("reference write is not bitwise")
        # the line outside the lambda window must still enter chi_line
        if lp.header["selected_lines"] != 2 or lp.header["rows"] != 4:
            raise RuntimeError("selection did not exclude the out-of-window line")
        if float(lp.chi_line[0].sum()) <= float(
                sum(lp.rows["w"][lp.rows["shell_slot"] == 0])):
            raise RuntimeError("chi_line does not exceed the recorded rows; the "
                               "replay dropped the out-of-window contribution")
        results["reference"] = {"sha256": lp.manifest["sha256"],
                                "rows": lp.header["rows"],
                                "bitwise": True}

        # (2) seeded replay drift: one ulp on cs.chi_line
        drift = work / "drift.bin"
        run({**sel, "LP_SEED_CHI_DRIFT": "1"}, drift)
        try:
            check_artifact(drift)
            raise RuntimeError("1-ulp replay drift was NOT refused")
        except LinePopError as exc:
            results["seeded_replay_drift_refused"] = str(exc)

        # (3) row cap: fail closed, no partial artifact
        cap = work / "cap.bin"
        proc = run({**sel, "LUMINA_CMF_LINEPOP_MAXROWS": "3"}, cap)
        if proc.returncode == 0 or cap.exists():
            raise RuntimeError("row cap did not fail closed")
        results["row_cap_refused"] = proc.stderr.strip()

        # (4) generation swap
        gen = work / "gen11.bin"
        run({**sel, "LP_FIXTURE_ITER": "11"}, gen)
        try:
            check_artifact(gen)
            raise RuntimeError("generation 11 was NOT refused by the contract")
        except LinePopError as exc:
            results["generation_swap_refused"] = str(exc)
        alt = check_artifact(gen, expected_iteration=11,
                             non_contract_override=True)
        results["generation_swap_override_status"] = alt.contract_status

        # (5) payload tamper with a stale sidecar
        tamper = work / "tamper.bin"
        shutil.copy(good, tamper)
        shutil.copy(str(good) + ".manifest.json", str(tamper) + ".manifest.json")
        blob = bytearray(tamper.read_bytes())
        blob[-1] ^= 0x01
        tamper.write_bytes(bytes(blob))
        try:
            check_artifact(tamper)
            raise RuntimeError("payload tamper was NOT refused")
        except LinePopError as exc:
            results["payload_tamper_refused"] = str(exc)

        # (6) missing selection / out-of-range shell
        proc = run({}, work / "nosel.bin")
        if proc.returncode == 0:
            raise RuntimeError("missing shell selection was accepted")
        results["missing_selection_refused"] = proc.stderr.strip()
        proc = run({"LUMINA_CMF_LINEPOP_SHELLS": "0,7"}, work / "badsel.bin")
        if proc.returncode == 0:
            raise RuntimeError("out-of-range shell was accepted")
        results["out_of_range_shell_refused"] = proc.stderr.strip()

        # (7) EPAY disposition actually distinguishes discarded eta cells
        epay = work / "epay.bin"
        run({**sel, "LUMINA_CMF_EPAY": "2", "LUMINA_CMF_EPAY_SMIN": "1",
             "LUMINA_CMF_EPAY_TAUBIN": "1e30", "LUMINA_CMF_EPAY_HOTF": "0"},
            epay)
        ep = check_artifact(epay)
        replaced = int((ep.disposition == 2).sum())
        if replaced != ep.header["n_bins"]:
            raise RuntimeError(
                f"EPAY rate-shape disposition not detected ({replaced} cells)")
        results["epay_rate_shape_cells"] = replaced

        # (8) TAUEFF is refused (the shell gate is not reproducible here)
        proc = run({**sel, "LUMINA_CMF_EPAY": "2",
                    "LUMINA_CMF_EPAY_TAUEFF": "1e4"}, work / "taueff.bin")
        if proc.returncode == 0:
            raise RuntimeError("EPAY_TAUEFF>0 was accepted")
        results["epay_taueff_refused"] = proc.stderr.strip()

        results["verdict"] = "PASS"
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    except Exception as exc:                       # noqa: BLE001
        results["verdict"] = f"FAIL: {exc}"
        print(json.dumps(results, indent=2, sort_keys=True))
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
