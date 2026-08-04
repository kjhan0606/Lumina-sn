#!/usr/bin/env python3
"""Offline T1/T2 discriminator for the frozen parity59 UV payloads.

T1 constructs an exactly energy-conserving uniform redistribution source at
shell 8 and runs the existing CPU stage31 driver.  T2 is fail-closed: it audits
whether the supplied artifacts are sufficient to replace *both* line opacity
and emissivity from the same iteration-10 populations, and never substitutes a
different-epoch population dump or an opacity proxy for the requested test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_chieta_check import check_artifact  # noqa: E402
from emiss_e6_direct_fields import cmfgen_all_shells, weighted_mean  # noqa: E402
import stage31_cmf_field_bench as bench  # noqa: E402
import w3_gamma_triple_compare as gamma  # noqa: E402


HEADER = struct.Struct("<8sIIQQQQIId")
RUN = Path("/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828")
AB2_RUN = Path("/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766")
CMF_RUN = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
E9 = ROOT / "validation/emiss_e12/e9_same_capture/emiss_e9_effective_iter10"
E9_JDET = ROOT / "validation/emiss_e12/jdet_e9_same_capture_s8.tsv"
PREREG = ROOT / "validation/uv_t1t2/preregistration.json"
OUT = ROOT / "validation/uv_t1t2"
DRIVER = Path("/tmp/stage31_cmf_field_driver_uv_t1t2")


class T1T2Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize(header: tuple, arrays: list[np.ndarray]) -> bytes:
    return b"".join(
        [HEADER.pack(*header)]
        + [np.asarray(a, dtype="<f8").reshape(-1).tobytes(order="C") for a in arrays]
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise T1T2Error(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_preregistration() -> tuple[dict[str, Any], str]:
    prereg = json.loads(PREREG.read_text())
    if prereg.get("status") != "FROZEN-BEFORE-T1-T2-CONSTRUCTION-AND-STAGE31-MEASUREMENT":
        raise T1T2Error("preregistration is absent or not frozen")
    actual = {
        "fluormat_chieta_sha256": sha256(RUN / "chieta_iter10"),
        "fluormat_A_sha256": sha256(RUN / "emiss_ab_iter10.A"),
        "fluormat_B2_sha256": sha256(RUN / "emiss_ab_iter10.B2"),
        "fluor_matrix_sha256": sha256(RUN / "fluor_matrix_iter10"),
        "e9_same_capture_sha256": sha256(E9),
    }
    if actual != prereg["inputs"]:
        raise T1T2Error(f"preregistration input hash mismatch: {actual}")
    return prereg, sha256(PREREG)


def build_t1(prereg: dict[str, Any], prereg_hash: str) -> tuple[Path, dict[str, Any]]:
    e9 = check_artifact(E9)
    source = check_artifact(RUN / "emiss_ab_iter10.A")
    if e9.header != source.header:
        raise T1T2Error("E9 and capture-A headers differ")
    nr, nnu = int(e9.header[3]), int(e9.header[4])
    shell = int(prereg["scope"]["shell"])
    if (nr, nnu, shell) != (50, 1000, 8):
        raise T1T2Error("unexpected T1 grid/shell")
    arrays = [np.asarray(a).copy() for a in e9.arrays]
    source_arrays = [np.asarray(a) for a in source.arrays]
    fields = {
        "chi_coherent": arrays[4].reshape(nr, nnu)[:, ::-1],
        "eta_fixed": arrays[5].reshape(nr, nnu)[:, ::-1],
        "eta_coherent": arrays[6].reshape(nr, nnu)[:, ::-1],
        "eta_total": arrays[7].reshape(nr, nnu)[:, ::-1],
        "J": arrays[8].reshape(nr, nnu)[:, ::-1],
    }
    captured_chi = source_arrays[4].reshape(nr, nnu)[:, ::-1]
    chi_e = np.min(captured_chi, axis=1)[:, None]
    chi_line = captured_chi - chi_e
    if np.any(chi_line < 0.0) or not np.isfinite(chi_line).all():
        raise T1T2Error("invalid chi_line proxy; no clamp is permitted")
    eps_mc = float(prereg["T1"]["eps_MC"])
    line_return = (1.0 - eps_mc) * chi_line * fields["J"]
    electron_return = chi_e * fields["J"]
    reconstructed = fields["eta_fixed"] + electron_return + line_return
    reconstruction_max = float(np.max(np.abs(reconstructed - fields["eta_total"])))
    reconstruction_rel = reconstruction_max / float(np.max(np.abs(fields["eta_total"])))
    if reconstruction_rel > 2.0e-15:
        raise T1T2Error(f"E9 component reconstruction failed: {reconstruction_rel}")

    edges, _, widths = bench.canonical_grid()
    removed_by_bin = line_return[shell] * widths
    removed = float(np.sum(removed_by_bin))
    uniform_power_per_bin = removed / nnu
    redistributed = np.full(nnu, uniform_power_per_bin) / widths
    eta_coherent_new = fields["eta_coherent"].copy()
    eta_coherent_new[shell] = electron_return[shell] + redistributed
    eta_total_new = fields["eta_fixed"] + eta_coherent_new
    chi_coherent_new = fields["chi_coherent"].copy()
    chi_coherent_new[shell] = chi_e[shell]
    if (np.any(eta_total_new < 0.0) or np.any(eta_coherent_new < 0.0)
            or np.any(chi_coherent_new < 0.0)
            or not all(np.isfinite(x).all() for x in
                       (eta_total_new, eta_coherent_new, chi_coherent_new))):
        raise T1T2Error("T1 constructed a negative/nonfinite cell; no clamp permitted")
    injected = float(np.sum(redistributed * widths))
    energy_error = injected / removed - 1.0
    if abs(energy_error) > 2.0e-15:
        raise T1T2Error(f"T1 energy conservation failed: {energy_error}")

    arrays[4] = chi_coherent_new[:, ::-1].reshape(-1)
    arrays[6] = eta_coherent_new[:, ::-1].reshape(-1)
    arrays[7] = eta_total_new[:, ::-1].reshape(-1)
    raw = serialize(e9.header, arrays)
    payload = OUT / "t1_uniform_iter10"
    payload.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    manifest = {
        "schema": "LCMFCE01-v1", "sha256": digest,
        "iteration": int(e9.header[5]), "field_generation": int(e9.header[6]),
        "post_damping": True, "coherent_frozen": True,
        "frequency_descending": True, "eta_decomposition_bitwise": True,
        "eta_decomposition_max_abs": 0,
        "diagnostic": "UV-T1-uniform-R-shell8",
        "source_payload_sha256": e9.manifest["sha256"],
        "capture_A_sha256": source.manifest["sha256"],
        "preregistration_sha256": prereg_hash,
        "construction": "shell-8 chi_coherent reduced to chi_e proxy; all same-bin nonthermal line-return energy redistributed with R[j,i]=1/1000",
        "all_other_shells_unchanged": True,
        "chi_total_unchanged": True,
        "clamp_floor_fallback": False,
        "production_code_modified": False,
    }
    Path(str(payload) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    check_artifact(payload)

    wavelength = bench.C_ANGSTROM / np.sqrt(edges[:-1] * edges[1:])
    source_rows = []
    for name, lo, hi in tuple(bench.BANDS) + (("OPTICAL", 3000.0, 10000.0),):
        weights = bench.band_weights(edges, lo, hi)
        base = float(np.sum(fields["eta_total"][shell] * weights))
        new = float(np.sum(eta_total_new[shell] * weights))
        source_rows.append({
            "band": name, "lambda_lo_A": lo, "lambda_hi_A": hi,
            "E9_source_energy": base, "T1_source_energy": new,
            "T1_over_E9_source": new / base,
        })
    write_csv(OUT / "t1_source_bands.csv", source_rows)
    audit = {
        "payload_sha256": digest, "shell": shell, "n_output_bins": nnu,
        "R_probability_per_output_bin": 1.0 / nnu,
        "R_column_sum": math.fsum([1.0 / nnu] * nnu),
        "removed_line_return_energy": removed,
        "injected_uniform_energy": injected,
        "relative_energy_error": energy_error,
        "E9_reconstruction_max_abs": reconstruction_max,
        "E9_reconstruction_relative": reconstruction_rel,
        "negative_cells": 0, "nonfinite_cells": 0, "clamp": 0,
        "fallback": 0, "input_wavelength_bin_count": int(wavelength.size),
        "source_bands": source_rows,
    }
    (OUT / "t1_construction.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload, audit


def run_stage31(payload: Path, prereg: dict[str, Any]) -> tuple[dict[str, str], dict[str, np.ndarray], list[str]]:
    bench.compile_driver(DRIVER)
    hashes = []
    primary_meta: dict[str, str] | None = None
    primary_table: dict[str, np.ndarray] | None = None
    for repeat in range(1, 4):
        output = OUT / ("t1_jdet_s8.tsv" if repeat == 1 else f"t1_jdet_s8_repeat{repeat}.tsv")
        command = [str(DRIVER), str(payload), str(payload) + ".manifest.json",
                   str(prereg["scope"]["shell"]), str(prereg["scope"]["n_mu"]),
                   repr(prereg["scope"]["T_inner_K"]),
                   repr(prereg["scope"]["bb_scale"]), str(output)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if completed.returncode:
            raise T1T2Error(f"stage31 T1 failed rc={completed.returncode}: {completed.stderr}")
        meta, table = bench.parse_driver_table(output)
        for key in ("clamp", "solution_negative_excess", "sign_uncertain", "nonfinite"):
            if int(meta[key]) != 0:
                raise T1T2Error(f"stage31 T1 guard {key}={meta[key]}")
        if float(meta["transport_residual"]) > 1.0e-4:
            raise T1T2Error("stage31 T1 transport residual exceeds 1e-4")
        hashes.append(sha256(output))
        if repeat == 1:
            primary_meta, primary_table = meta, table
    if len(set(hashes)) != 1:
        raise T1T2Error(f"T1 stage31 is not byte deterministic: {hashes}")
    assert primary_meta is not None and primary_table is not None
    return primary_meta, primary_table, hashes


def measure_t1(payload: Path, construction: dict[str, Any], prereg: dict[str, Any],
               meta: dict[str, str], table: dict[str, np.ndarray], hashes: list[str]) -> dict[str, Any]:
    checked = check_artifact(payload)
    _, e9_table = bench.parse_driver_table(E9_JDET)
    j_t1 = table["J_det"][::-1]
    j_e9 = e9_table["J_det"][::-1]
    j_prod = table["J_producer"][::-1]
    if not all(np.isfinite(x).all() for x in (j_t1, j_e9, j_prod)):
        raise T1T2Error("nonfinite stage31 output")
    edges, _, _ = bench.canonical_grid()
    r_edge = np.asarray(checked.arrays[0])
    velocities = 0.5 * (r_edge[:-1] + r_edge[1:]) / checked.header[-1] / 1.0e5
    cmf, cmf_meta = cmfgen_all_shells(edges, velocities, CMF_RUN)
    shell = int(prereg["scope"]["shell"])
    rows = []
    for name, lo, hi in tuple(bench.BANDS) + (("OPTICAL", 3000.0, 10000.0),):
        weights = bench.band_weights(edges, lo, hi)
        t1 = weighted_mean(j_t1, weights)
        e9 = weighted_mean(j_e9, weights)
        producer = weighted_mean(j_prod, weights)
        cmf_band = weighted_mean(cmf[shell], weights)
        rows.append({
            "band": name, "lambda_lo_A": lo, "lambda_hi_A": hi,
            "E9_over_CMFGEN": e9 / cmf_band,
            "T1_over_CMFGEN": t1 / cmf_band,
            "T1_over_E9": t1 / e9,
            "T1_over_J_producer": t1 / producer,
            "abs_log10_E9_over_CMFGEN": abs(math.log10(e9 / cmf_band)),
            "abs_log10_T1_over_CMFGEN": abs(math.log10(t1 / cmf_band)),
            "toward_CMFGEN": abs(math.log10(t1 / cmf_band)) < abs(math.log10(e9 / cmf_band)),
        })
    write_csv(OUT / "t1_band_table.csv", rows)

    gamma.CMF_RUN = CMF_RUN
    context, rates = bench.load_gamma_context(RUN, edges, j_t1)
    _, rates_e9 = bench.load_gamma_context(RUN, edges, j_e9)
    e9_by_index = {row["matrix_index"]: row for row in rates_e9}
    gamma_rows = []
    for row in rates:
        e9_row = e9_by_index[row["matrix_index"]]
        gamma_rows.append({
            "target": row["target"], "matrix_index": row["matrix_index"],
            "Gamma_E9": e9_row["Gamma_det_D"],
            "Gamma_T1": row["Gamma_det_D"], "Gamma_CMFGEN": row["Gamma_CMFGEN_C"],
            "E9_over_CMFGEN": e9_row["Gamma_det_over_CMFGEN"],
            "T1_over_CMFGEN": row["Gamma_det_over_CMFGEN"],
            "T1_over_E9": row["Gamma_det_D"] / e9_row["Gamma_det_D"],
            "member_count": row["member_count"], "route_count": row["route_count"],
            "threshold_eV": row["threshold_eV"],
        })
    write_csv(OUT / "t1_gamma_table.csv", gamma_rows)
    by_band = {row["band"]: row for row in rows}
    gate = bool(by_band["B0"]["toward_CMFGEN"] and by_band["B1"]["toward_CMFGEN"])
    controls = {}
    for name, path in (
        ("E10_MC_prefix_R", ROOT / "validation/emiss_e10/stage31_summary.json"),
        ("E12_MC_full_R", ROOT / "validation/emiss_e12/stage31_summary.json"),
    ):
        data = json.loads(path.read_text())
        controls[name] = {row["band"]: row["E10_J_det_over_CMFGEN"]
                          for row in data["bands"]}
    summary = {
        "schema": "lumina-uv-t1-measurement-v1",
        "status": "RESOLVED",
        "preregistration_sha256": sha256(PREREG),
        "payload_sha256": checked.manifest["sha256"],
        "stage31_jdet_sha256": hashes[0], "repeat_hashes": hashes,
        "repeat_hashes_identical": len(set(hashes)) == 1,
        "construction": construction, "bands": rows, "gamma": gamma_rows,
        "controls_requoted": controls,
        "shape_gate_B0_and_B1_toward_CMFGEN": gate,
        "preregistered_readout": ("R_SHAPE_IS_THE_PROBLEM" if gate else
                                   "SAME_BIN_COHERENCE_ASSUMPTION_IS_THE_PROBLEM"),
        "driver_metadata": meta, "cmfgen": cmf_meta,
        "raw_negative_Jdet_bins": int(np.sum(j_t1 < 0.0)),
        "raw_minimum_Jdet": float(np.min(j_t1)),
        "clamp_floor_fallback": 0, "production_code_modified": False,
        "new_model_or_gpu_run": False,
    }
    (OUT / "t1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def audit_t2(prereg: dict[str, Any]) -> dict[str, Any]:
    a = check_artifact(AB2_RUN / "emiss_ab_iter10.A")
    b2 = check_artifact(AB2_RUN / "emiss_ab_iter10.B2")
    if a.header != b2.header:
        raise T1T2Error("T2 A/B2 headers differ")
    array_names = ("r_edge", "nu", "dnu", "chi_total", "chi_coherent",
                   "eta_fixed", "eta_coherent", "eta_total", "J_producer")
    comparisons = {}
    for index, name in enumerate(array_names):
        left, right = np.asarray(a.arrays[index]), np.asarray(b2.arrays[index])
        comparisons[name] = {
            "bitwise_equal": bool(np.array_equal(left, right)),
            "different_cells": int(np.count_nonzero(left != right)),
            "max_abs_difference": float(np.max(np.abs(left - right))),
        }
    stdout = (AB2_RUN / "stdout.log").read_text(errors="strict")
    capture_pos = stdout.find("[EMISS-AB] wrote A=")
    levelpop_pos = stdout.find("Per-level departure b_k dump -> lumina_levelpop.csv")
    final_iter_pos = stdout.find("final pure-CMFGEN it=11")
    fine_header = next(csv.reader((AB2_RUN / "cmf_fine_linedump_s8.csv").open()))
    payload_arrays = list(array_names)
    blockers = [
        "LCMFCE01 contains chi_total and chi_coherent but does not serialize chi_line, chi_line_th, chi_abs, or per-line opacity; exact removal of the production line assembly is therefore impossible.",
        "B2 serializes population-native eta only: its chi_total and chi_coherent arrays are bitwise identical to A.",
        "The supplied fine-line tau dump covers shell 8 only, while the stage31 formal solve consumes all 50 radial shells.",
        "The available lumina_levelpop.csv is written after the iteration-10 A/B2 capture and immediately before the logged final pure-CMFGEN iteration 11 resolve; it is not an authenticated iteration-10 population payload.",
        "No exact iteration-10 lower-level population array is present, so the requested stimulated-opacity factor cannot be assembled without changing population epoch or inventing a proxy.",
    ]
    exact_available = False
    result = {
        "schema": "lumina-uv-t2-input-availability-v1",
        "status": "UNRESOLVED",
        "preregistered_gate": prereg["T2"]["amplitude_confirmation_gate"],
        "prediction_retained_unmeasured": prereg["T2"]["prediction_before_measurement"],
        "exact_population_native_chi_eta_pair_available": exact_available,
        "input_capture": str(AB2_RUN),
        "input_sha256": {
            "chieta": sha256(AB2_RUN / "chieta_iter10"),
            "A": sha256(AB2_RUN / "emiss_ab_iter10.A"),
            "B2": sha256(AB2_RUN / "emiss_ab_iter10.B2"),
        },
        "A_B2_array_comparison": comparisons,
        "B2_line_emissivity_formula": b2.manifest.get("line_emissivity_formula"),
        "B2_common_assembly_state_sha256": b2.manifest.get("common_assembly_state_sha256"),
        "payload_arrays": payload_arrays,
        "fine_linedump_shells_present": [8, 45, 49],
        "fine_linedump_s8_columns": fine_header,
        "capture_logged_before_levelpop_dump": 0 <= capture_pos < levelpop_pos,
        "levelpop_dump_precedes_logged_final_iter11_resolve": 0 <= levelpop_pos < final_iter_pos,
        "blocking_evidence": blockers,
        "forbidden_substitutions_not_used": [
            "final-iteration level populations for iteration-10 populations",
            "chi_coherent-minus-minimum as a full native chi replacement",
            "B2 eta-only stage31 failure as a T2 result",
            "clamp, floor, tolerance relaxation, or solver-guard bypass",
        ],
        "BALL_over_CMFGEN": None,
        "B0_over_CMFGEN": None,
        "B1_over_CMFGEN": None,
        "amplitude_cause_final_confirmation": "UNRESOLVED",
        "residual_shape_separation": "UNRESOLVED",
        "production_code_modified": False, "new_model_or_gpu_run": False,
        "clamp_floor_fallback": 0,
    }
    (OUT / "t2_availability.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> int:
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        prereg, prereg_hash = validate_preregistration()
        payload, construction = build_t1(prereg, prereg_hash)
        meta, table, hashes = run_stage31(payload, prereg)
        t1 = measure_t1(payload, construction, prereg, meta, table, hashes)
        t2 = audit_t2(prereg)
        combined = {
            "schema": "lumina-uv-t1t2-combined-v1",
            "T1": {"status": t1["status"], "readout": t1["preregistered_readout"]},
            "T2": {"status": t2["status"], "readout": t2["amplitude_cause_final_confirmation"]},
        }
        (OUT / "combined_summary.json").write_text(
            json.dumps(combined, indent=2, sort_keys=True) + "\n")
        print(json.dumps(combined, indent=2, sort_keys=True))
        return 0
    except (T1T2Error, OSError, ValueError, KeyError, subprocess.SubprocessError,
            bench.BenchError, gamma.Unresolved) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
