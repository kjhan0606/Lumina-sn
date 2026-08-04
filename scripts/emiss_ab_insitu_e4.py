#!/usr/bin/env python3
"""E5 offline judge: in-situ A/B/B2 payloads -> stage31 J_det, bands, Gamma.

The runtime capture is intentionally out of scope here.  This consumer accepts
only tagged A-production/B-Aul-nu/B2-controlled-retention lanes with one common
assembly-state digest and bitwise-identical geometry/grid/opacity/J coordinates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_chieta_check import check_artifact  # noqa: E402
import stage31_cmf_field_bench as bench  # noqa: E402
import w3_gamma_triple_compare as gamma  # noqa: E402


class E4Error(RuntimeError):
    pass


def exact_array(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return np.asarray(left, dtype="<f8").tobytes() == np.asarray(
        right, dtype="<f8").tobytes()


def validate_lanes(base: Path, allow_seeded: bool) -> tuple[Any, Any, Any]:
    a = check_artifact(Path(str(base) + ".A"))
    b = check_artifact(Path(str(base) + ".B"))
    b2 = check_artifact(Path(str(base) + ".B2"))
    ma, mb, mb2 = a.manifest, b.manifest, b2.manifest
    if ma.get("emiss_ab_lane") != "A-production":
        raise E4Error("A sidecar lane tag is absent or wrong")
    if mb.get("emiss_ab_lane") != "B-Aul-nu":
        raise E4Error("B sidecar lane tag is absent or wrong")
    if mb2.get("emiss_ab_lane") != "B2-Aul-nu-retain-A-undefined":
        raise E4Error("B2 sidecar lane tag is absent or wrong")
    states = [m.get("common_assembly_state_sha256") for m in (ma, mb, mb2)]
    if not isinstance(states[0], str) or len(states[0]) != 64 or len(set(states)) != 1:
        raise E4Error("common assembly-state hashes disagree")
    # r_edge, nu, dnu, chi_total, chi_es and J_producer are invariant.  eta is
    # the sole allowed payload delta; the runtime additionally checks all
    # internal opacity decompositions before writing either artifact.
    for index, label in ((0, "r_edge"), (1, "nu"), (2, "dnu"),
                         (3, "chi_total"), (4, "chi_es"), (8, "J_producer")):
        if any(not exact_array(a.arrays[index], lane.arrays[index])
               for lane in (b, b2)):
            raise E4Error(f"non-emissivity payload coordinate differs: {label}")
    if any(a.header[3:8] != lane.header[3:8] for lane in (b, b2)):
        raise E4Error("A/B/B2 header epoch/dimension/flags mismatch")
    if a.header[3] != 50 or a.header[4] != 1000:
        raise E4Error(f"stage31 E4 requires 50x1000, got {a.header[3]}x{a.header[4]}")
    for label, manifest in (("B", mb), ("B2", mb2)):
        seed = manifest.get("seeded_defect", {})
        if not allow_seeded and (seed.get("line_id", -1) != -1 or
                                 seed.get("hits", 0) != 0):
            raise E4Error(f"{label} seeded negative-control payload refused without --allow-seeded")
    coverage = mb.get("coverage")
    if not isinstance(coverage, dict):
        raise E4Error("B coverage block is missing")
    coverage2 = mb2.get("coverage")
    if not isinstance(coverage2, dict):
        raise E4Error("B2 coverage block is missing")
    active = coverage.get("active_transition_count", 0)
    defined = coverage.get("defined_transition_count", 0)
    undefined = coverage.get("undefined_transition_count", 0)
    if active <= 0 or defined + undefined != active:
        raise E4Error("transition coverage census does not close")
    active_cells = coverage.get("active_line_shell_count", 0)
    defined_cells = coverage.get("defined_line_shell_count", 0)
    undefined_cells = coverage.get("undefined_line_shell_count", 0)
    fraction = coverage.get("a_reference_contribution_fraction")
    if active_cells <= 0 or defined_cells + undefined_cells != active_cells:
        raise E4Error("line-shell coverage census does not close")
    if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise E4Error("A-reference contribution coverage is invalid")
    census_keys = ("active_transition_count", "defined_transition_count",
                   "undefined_transition_count", "active_line_shell_count",
                   "defined_line_shell_count", "undefined_line_shell_count")
    if any(coverage2.get(key) != coverage.get(key) for key in census_keys):
        raise E4Error("B/B2 coverage census differs")
    power_keys = ("a_reference_line_power", "a_reference_covered_line_power",
                  "a_reference_undefined_line_power",
                  "a_reference_contribution_fraction",
                  "a_reference_undefined_contribution_fraction")
    if any(coverage2.get(key) != coverage.get(key) for key in power_keys):
        raise E4Error("B/B2 A-reference power ledger differs")
    if (mb.get("controlled_retention") is not False or
        mb.get("undefined_transition_policy") != "zero-undefined-fail-closed"):
        raise E4Error("B zero-undefined policy is absent or wrong")
    if (mb2.get("controlled_retention") is not True or
        mb2.get("undefined_transition_policy") !=
            "retain-production-A-explicit-controlled"):
        raise E4Error("B2 controlled-retention policy is absent or wrong")
    if (coverage2.get("retained_transition_count") != undefined or
        coverage2.get("retained_line_shell_count") != undefined_cells):
        raise E4Error("B2 retained census does not equal undefined census")
    undef_fraction = coverage2.get("a_reference_undefined_contribution_fraction")
    retained_fraction = coverage2.get("a_reference_retained_contribution_fraction")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0.0 <= x <= 1.0
               for x in (undef_fraction, retained_fraction)):
        raise E4Error("B2 undefined/retained contribution fraction is invalid")
    if retained_fraction != undef_fraction:
        raise E4Error("B2 retained contribution does not equal undefined A contribution")
    diagnostic = mb2.get("undefined_a_reference_diagnostic")
    if not isinstance(diagnostic, dict) or diagnostic.get("epoch") != "pre-EPAY":
        raise E4Error("B2 pre-EPAY undefined-emissivity diagnostic is missing")
    if mb.get("undefined_a_reference_diagnostic") != diagnostic:
        raise E4Error("B/B2 undefined-emissivity diagnostics differ")
    by_band, by_shell = diagnostic.get("by_band"), diagnostic.get("by_shell")
    if not isinstance(by_band, list) or len(by_band) != a.header[4]:
        raise E4Error("B2 undefined-emissivity by-band array has wrong length")
    if not isinstance(by_shell, list) or len(by_shell) != a.header[3]:
        raise E4Error("B2 undefined-emissivity by-shell array has wrong length")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0.0
           for x in by_band + by_shell):
        raise E4Error("B2 undefined-emissivity diagnostic contains invalid values")
    undef_power = coverage2.get("a_reference_undefined_line_power")
    if not isinstance(undef_power, (int, float)) or not math.isfinite(undef_power):
        raise E4Error("B2 undefined A-reference power is invalid")
    tolerance = 1.0e-12 * max(abs(float(undef_power)), 1.0e-300)
    if abs(math.fsum(by_band) - undef_power) > tolerance or \
       abs(math.fsum(by_shell) - undef_power) > tolerance:
        raise E4Error("B2 undefined-emissivity diagnostic arrays do not close")
    csv_bytes = []
    for suffix in ("B", "B2"):
        undefined_path = Path(str(base) + f".{suffix}.undefined.csv")
        if not undefined_path.is_file():
            raise E4Error(f"undefined-transition list missing: {undefined_path}")
        csv_bytes.append(undefined_path.read_bytes())
        with undefined_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != undefined:
            raise E4Error(f"{suffix} undefined list has {len(rows)} rows, sidecar says {undefined}")
    if csv_bytes[0] != csv_bytes[1]:
        raise E4Error("B/B2 undefined-transition lists differ")
    return a, b, b2


def run_stage31(executable: Path, payload: Path, output: Path,
                shell: int, nmu: int, t_inner: float, bb_scale: float) -> dict[str, Any]:
    output.unlink(missing_ok=True)
    command = [str(executable), str(payload), str(payload) + ".manifest.json",
               str(shell), str(nmu), repr(t_inner), repr(bb_scale), str(output)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        raise E4Error(f"stage31 failed for {payload}: rc={completed.returncode}\n"
                      f"stdout={completed.stdout}\nstderr={completed.stderr}")
    meta, table = bench.parse_driver_table(output)
    for key in ("clamp", "solution_negative_excess", "sign_uncertain", "nonfinite"):
        if int(meta[key]) != 0:
            raise E4Error(f"stage31 guard {key}={meta[key]} for {payload}")
    if float(meta["transport_residual"]) > 1.0e-4:
        raise E4Error(f"transport residual exceeds 1e-4 for {payload}")
    return {"metadata": meta, "table": table, "stdout": completed.stdout,
            "stderr": completed.stderr, "command": command}


def gamma_rows(edges: np.ndarray, fields: dict[str, np.ndarray | None],
               capture_dir: Path, cmf_run: Path,
               ew_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gamma.CMF_RUN = cmf_run.resolve()
    gamma.EW_DIR = ew_dir.resolve()
    if fields["A"] is None:
        raise E4Error("A field is required for Gamma reference")
    context, rates_a = bench.load_gamma_context(capture_dir, edges, fields["A"])
    rates_by_lane = {"A": {row["matrix_index"]: row for row in rates_a}}
    for lane in ("B", "B2"):
        if fields[lane] is not None:
            _, lane_rates = bench.load_gamma_context(capture_dir, edges, fields[lane])
            rates_by_lane[lane] = {row["matrix_index"]: row for row in lane_rates}
    rows = []
    for ra in rates_a:
        ga = ra["Gamma_det_D"]
        gc = ra["Gamma_CMFGEN_C"]
        if not all(math.isfinite(x) and x > 0.0 for x in (ga, gc)):
            raise E4Error(f"non-positive/non-finite Gamma for {ra['target']}")
        lane_gamma: dict[str, float | None] = {"A": ga, "B": None, "B2": None}
        for lane in ("B", "B2"):
            if lane in rates_by_lane:
                value = rates_by_lane[lane][ra["matrix_index"]]["Gamma_det_D"]
                if not math.isfinite(value) or value <= 0.0:
                    raise E4Error(f"non-positive/non-finite Gamma for {ra['target']} {lane}")
                lane_gamma[lane] = value
        rows.append({
            "target": ra["target"], "matrix_index": ra["matrix_index"],
            "Gamma_A": ga, "Gamma_B": lane_gamma["B"],
            "Gamma_B2": lane_gamma["B2"], "Gamma_CMFGEN": gc,
            "A_over_CMFGEN": ga / gc,
            "B_over_CMFGEN": (lane_gamma["B"] / gc if lane_gamma["B"] else None),
            "B2_over_CMFGEN": (lane_gamma["B2"] / gc if lane_gamma["B2"] else None),
            "B_over_A": (lane_gamma["B"] / ga if lane_gamma["B"] else None),
            "B2_over_A": (lane_gamma["B2"] / ga if lane_gamma["B2"] else None),
            "log10_B_over_A": (math.log10(lane_gamma["B"] / ga)
                                  if lane_gamma["B"] else None),
            "log10_B2_over_A": (math.log10(lane_gamma["B2"] / ga)
                                   if lane_gamma["B2"] else None),
            "member_count": ra["member_count"],
            "route_count": ra["route_count"],
            "threshold_eV": ra["threshold_eV"],
        })
    return context, rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path,
                    help="runtime base; consumes BASE.A, BASE.B, and BASE.B2")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "validation/emiss_e5")
    ap.add_argument("--capture-dir", type=Path, default=Path(
        "/gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605"))
    ap.add_argument("--cmf-run", type=Path, default=bench.DEFAULT_CMF)
    ap.add_argument("--ew-dir", type=Path, default=gamma.EW_DIR)
    ap.add_argument("--shell", type=int, default=8)
    ap.add_argument("--nmu", type=int, default=16)
    ap.add_argument("--t-inner", type=float, default=10020.0)
    ap.add_argument("--bb-scale", type=float, default=1.0)
    ap.add_argument("--driver", type=Path, default=Path("/tmp/stage31_cmf_field_driver_e4"))
    ap.add_argument("--allow-seeded", action="store_true")
    args = ap.parse_args()
    try:
        if args.shell != 8:
            raise E4Error("the preregistered Fe III idx201/S II SL4 identity is shell 8")
        a, b, b2 = validate_lanes(args.base.resolve(), args.allow_seeded)
        edges, centers, _ = bench.canonical_grid()
        nu_descending = np.asarray(a.arrays[1])
        grid_error = float(np.max(np.abs(nu_descending[::-1] / centers - 1.0)))
        if grid_error > 1.0e-12:
            raise E4Error(f"capture/canonical frequency grid mismatch {grid_error}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        bench.compile_driver(args.driver)
        artifacts = {"A": a, "B": b, "B2": b2}
        solves: dict[str, dict[str, Any] | None] = {}
        solve_errors: dict[str, str] = {}
        for lane in ("A", "B", "B2"):
            try:
                solves[lane] = run_stage31(
                    args.driver, Path(str(args.base.resolve()) + f".{lane}"),
                    args.out_dir / f"jdet_{lane}.tsv", args.shell, args.nmu,
                    args.t_inner, args.bb_scale)
            except E4Error as exc:
                solves[lane] = None
                solve_errors[lane] = str(exc)
        if solves["A"] is None:
            raise E4Error(f"A reference solve failed: {solve_errors['A']}")
        jprod = np.asarray(a.arrays[8]).reshape(a.header[3], a.header[4])[args.shell][::-1]
        fields: dict[str, np.ndarray | None] = {
            lane: (solves[lane]["table"]["J_det"][::-1]
                   if solves[lane] is not None else None)
            for lane in ("A", "B", "B2")
        }
        context, rates = gamma_rows(edges, fields, args.capture_dir.resolve(),
                                    args.cmf_run, args.ew_dir)
        cmf = context["cmf"]["J"]
        bands_by_lane = {
            lane: (bench.make_band_rows(edges, field, jprod, cmf)
                   if field is not None else None)
            for lane, field in fields.items()
        }
        bands = []
        assert bands_by_lane["A"] is not None
        for index, ra in enumerate(bands_by_lane["A"]):
            lane_rows = {lane: (bands_by_lane[lane][index]
                                if bands_by_lane[lane] is not None else None)
                         for lane in ("A", "B", "B2")}
            present = [row["J_det"] for row in lane_rows.values() if row is not None]
            if not all(math.isfinite(x) and x > 0.0 for x in
                       (present + [ra["J_CMFGEN"]])):
                raise E4Error(f"non-positive/non-finite band mean for {ra['band']}")
            jb = lane_rows["B"]["J_det"] if lane_rows["B"] else None
            jb2 = lane_rows["B2"]["J_det"] if lane_rows["B2"] else None
            bands.append({
                "band": ra["band"], "wavelength_A": ra["wavelength_A"],
                "A_over_CMFGEN": ra["J_det_over_J_CMFGEN"],
                "B_over_CMFGEN": (lane_rows["B"]["J_det_over_J_CMFGEN"]
                                   if lane_rows["B"] else None),
                "B2_over_CMFGEN": (lane_rows["B2"]["J_det_over_J_CMFGEN"]
                                    if lane_rows["B2"] else None),
                "B_over_A": (jb / ra["J_det"] if jb is not None else None),
                "B2_over_A": (jb2 / ra["J_det"] if jb2 is not None else None),
                "A_J_det": ra["J_det"], "B_J_det": jb, "B2_J_det": jb2,
                "J_CMFGEN": ra["J_CMFGEN"],
            })
        result = {
            "schema": "lumina-emiss-ab-e5-verdict-v1",
            "base": str(args.base.resolve()),
            "common_assembly_state_sha256": a.manifest["common_assembly_state_sha256"],
            "coverage": b.manifest["coverage"],
            "controlled_retention": b2.manifest["coverage"],
            "undefined_a_reference_diagnostic":
                b2.manifest["undefined_a_reference_diagnostic"],
            "seeded_defect": {lane: artifacts[lane].manifest["seeded_defect"]
                              for lane in ("A", "B", "B2")},
            "nu_grid_max_relative_error": grid_error,
            "stage31": {
                lane: (solves[lane]["metadata"] if solves[lane] is not None
                       else {"status": "UNRESOLVED", "error": solve_errors[lane]})
                for lane in ("A", "B", "B2")
            },
            "bands": bands, "gamma": rates,
            "no_new_clamp": True,
        }
        (args.out_dir / "verdict.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n")
        write_csv(args.out_dir / "band_table.csv", bands,
                  ["band", "wavelength_A", "A_over_CMFGEN", "B_over_CMFGEN",
                   "B2_over_CMFGEN", "B_over_A", "B2_over_A", "A_J_det",
                   "B_J_det", "B2_J_det", "J_CMFGEN"])
        write_csv(args.out_dir / "gamma_table.csv", rates,
                  ["target", "matrix_index", "Gamma_A", "Gamma_B", "Gamma_B2",
                   "Gamma_CMFGEN", "A_over_CMFGEN", "B_over_CMFGEN",
                   "B2_over_CMFGEN", "B_over_A", "B2_over_A",
                   "log10_B_over_A", "log10_B2_over_A",
                   "member_count", "route_count", "threshold_eV"])
        def show(value: float | None) -> str:
            return "UNRESOLVED" if value is None else f"{value:.8g}"
        for row in bands:
            print(f"{row['band']:4s} A/CMFGEN={show(row['A_over_CMFGEN'])} "
                  f"B/CMFGEN={show(row['B_over_CMFGEN'])} "
                  f"B2/CMFGEN={show(row['B2_over_CMFGEN'])} "
                  f"B/A={show(row['B_over_A'])} B2/A={show(row['B2_over_A'])}")
        for row in rates:
            print(f"Gamma {row['target']}: A/CMFGEN={show(row['A_over_CMFGEN'])} "
                  f"B/CMFGEN={show(row['B_over_CMFGEN'])} "
                  f"B2/CMFGEN={show(row['B2_over_CMFGEN'])} "
                  f"B/A={show(row['B_over_A'])} B2/A={show(row['B2_over_A'])}")
        if solve_errors:
            print("UNRESOLVED lanes: " + ", ".join(sorted(solve_errors)), file=sys.stderr)
            print(f"PARTIAL: {args.out_dir / 'verdict.json'}")
            return 2
        print(f"PASS: {args.out_dir / 'verdict.json'}")
        return 0
    except (E4Error, OSError, ValueError, KeyError, subprocess.CalledProcessError,
            gamma.Unresolved, bench.BenchError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
