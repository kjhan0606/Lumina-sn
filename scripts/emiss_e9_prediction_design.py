#!/usr/bin/env python3
"""E9 preregistered arithmetic test and effective-field payload builder.

This is an offline consumer.  It does not run Lumina transport/plasma, alter a
model, or implement the proposed redistribution repair.  The optional payload
is a frozen diagnostic source for the existing stage31 CPU formal solver.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cmf_chieta_check import check_artifact  # noqa: E402
from emiss_e6_direct_fields import (  # noqa: E402
    cmfgen_all_shells, weighted_integral, weighted_mean,
)
import stage31_cmf_field_bench as bench  # noqa: E402


DEFAULT_RUN = Path("/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766")
DEFAULT_CMF = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
SIGMA_T_CODE = 6.6524587e-25
HEADER = struct.Struct("<8sIIQQQQIId")


class E9Error(RuntimeError):
    pass


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise E9Error(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_eps_mc(path: Path, shell: int) -> tuple[int, int, float]:
    with path.open() as stream:
        rows = [r for r in csv.DictReader(stream)
                if int(r["iter"]) == 10 and int(r["shell"]) == shell]
    if len(rows) != 1:
        raise E9Error("iteration-10 MC destruction row is not unique")
    terminals = int(rows[0]["terminals"])
    destroyed = int(rows[0]["destroyed"])
    if terminals <= 0 or destroyed <= 0:
        raise E9Error("MC destruction fraction is undefined")
    return terminals, destroyed, destroyed / terminals


def read_ne(path: Path, nr: int) -> np.ndarray:
    with path.open() as stream:
        rows = {int(r["shell_id"]): float(r["n_e"])
                for r in csv.DictReader(stream)}
    if sorted(rows) != list(range(nr)):
        raise E9Error("final-state electron-density proxy is incomplete")
    values = np.asarray([rows[s] for s in range(nr)])
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise E9Error("invalid final-state electron density")
    return values


def serialize(header: tuple, arrays: list[np.ndarray]) -> bytes:
    pieces = [HEADER.pack(*header)]
    for values in arrays:
        flat = np.asarray(values, dtype="<f8").reshape(-1)
        pieces.append(flat.tobytes(order="C"))
    return b"".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cmf-run", type=Path, default=DEFAULT_CMF)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "validation/emiss_e9")
    parser.add_argument("--shell", type=int, default=8)
    args = parser.parse_args()
    try:
        run = args.run.resolve()
        source_path = run / "emiss_ab_iter10.A"
        checked = check_artifact(source_path)
        header = checked.header
        nr, nnu = int(header[3]), int(header[4])
        if not (0 <= args.shell < nr):
            raise E9Error("shell is outside payload")
        arrays_desc = [np.asarray(x) for x in checked.arrays]
        fields = {
            "chi_total": arrays_desc[3].reshape(nr, nnu)[:, ::-1],
            "chi_coherent": arrays_desc[4].reshape(nr, nnu)[:, ::-1],
            "eta_fixed": arrays_desc[5].reshape(nr, nnu)[:, ::-1],
            "eta_total": arrays_desc[7].reshape(nr, nnu)[:, ::-1],
            "J_old": arrays_desc[8].reshape(nr, nnu)[:, ::-1],
        }
        for name, values in fields.items():
            if not np.isfinite(values).all() or np.any(values <= 0.0):
                raise E9Error(f"{name} has nonpositive/nonfinite cells; no clamp allowed")

        terminals, destroyed, eps_mc = read_eps_mc(
            run / "lumina_ma_line_destruct.csv", args.shell)
        gain_mc = 1.0 / eps_mc
        eps_old_cell = fields["eta_fixed"] / fields["eta_total"]
        # Cellwise extension of E8: J_new/J_old = G_MC/G_old and
        # G_old=1/eps_old_cell.  No value clipping or fallback is used.
        j_effective = fields["J_old"] * gain_mc * eps_old_cell

        ne_final = read_ne(run / "lumina_plasma_state.csv", nr)
        chi_es_final = ne_final[:, None] * SIGMA_T_CODE
        # LCMFCE01 does not serialize chi_e.  The payload's per-shell minimum
        # coherent opacity is the capture-epoch, line-free proxy.  Unlike the
        # final-state n_e proxy it never makes the inferred line remainder
        # negative, and no clipping is needed.
        chi_es_proxy = np.min(fields["chi_coherent"], axis=1)[:, None]
        chi_line_proxy = fields["chi_coherent"] - chi_es_proxy
        chi_effective = chi_es_proxy + (1.0 - eps_mc) * chi_line_proxy
        if np.any(chi_effective <= 0.0) or not np.isfinite(chi_effective).all():
            raise E9Error("reconstructed chi_coherent is invalid; no clamp allowed")
        eta_coherent_effective = chi_effective * j_effective
        eta_total_effective = fields["eta_fixed"] + eta_coherent_effective
        # Sensitivity diagnostic: opacity-only substitution while holding the
        # original J and fixed source.  This is deliberately not the registered
        # gain test and exposes what changing chi alone can accomplish.
        eta_total_opacity_only = (fields["eta_fixed"] +
                                  chi_effective * fields["J_old"])

        edges, _, _ = bench.canonical_grid()
        r_edge = arrays_desc[0]
        velocities = (0.5 * (r_edge[:-1] + r_edge[1:]) /
                      float(header[-1]) / 1.0e5)
        cmf_j, cmf_meta = cmfgen_all_shells(
            edges, velocities, args.cmf_run.resolve())
        s = args.shell
        rows: list[dict[str, Any]] = []
        for band, lo, hi in bench.BANDS:
            weights = bench.band_weights(edges, lo, hi)
            width = math.fsum(weights)
            j_old = weighted_mean(fields["J_old"][s], weights)
            j_cmf = weighted_mean(cmf_j[s], weights)
            sf = weighted_integral(
                fields["eta_fixed"][s] / fields["chi_total"][s], weights) / width
            st_old = weighted_integral(
                fields["eta_total"][s] / fields["chi_total"][s], weights) / width
            eps_old_band = sf / st_old
            gain_old_band = 1.0 / eps_old_band
            prediction = (j_old / j_cmf) * gain_mc / gain_old_band
            j_arithmetic = weighted_mean(j_effective[s], weights) / j_cmf
            source_rebuilt = weighted_mean(
                eta_total_effective[s] / fields["chi_total"][s], weights) / j_cmf
            source_opacity_only = weighted_mean(
                eta_total_opacity_only[s] / fields["chi_total"][s], weights) / j_cmf
            sf_integral = weighted_integral(
                fields["eta_fixed"][s] / fields["chi_total"][s], weights)
            se_integral = weighted_integral(
                eta_total_effective[s] / fields["chi_total"][s], weights)
            eps_rebuilt = sf_integral / se_integral
            rows.append({
                "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
                "J_old_over_CMFGEN": j_old / j_cmf,
                "eps_old_source": eps_old_band,
                "gain_old_source": gain_old_band,
                "eps_MC": eps_mc, "gain_MC": gain_mc,
                "preregistered_J_over_CMFGEN": prediction,
                "preregistered_low_minus10pct": 0.9 * prediction,
                "preregistered_high_plus10pct": 1.1 * prediction,
                "measured_arithmetic_J_effective_over_CMFGEN": j_arithmetic,
                "arithmetic_relative_to_prediction": j_arithmetic / prediction,
                "arithmetic_hit_within_10pct": abs(j_arithmetic / prediction - 1.0) <= 0.1,
                "rebuilt_source_over_CMFGEN": source_rebuilt,
                "rebuilt_source_relative_to_prediction": source_rebuilt / prediction,
                "rebuilt_source_hit_within_10pct": abs(source_rebuilt / prediction - 1.0) <= 0.1,
                "rebuilt_eps_eff_source": eps_rebuilt,
                "rebuilt_gain_source": 1.0 / eps_rebuilt,
                "opacity_only_source_over_CMFGEN": source_opacity_only,
            })

        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_dir / "prediction_measurement.csv", rows)

        # Frozen diagnostic payload: retain chi_total and eta_fixed, replace the
        # coherent opacity, its frozen emissivity, total emissivity, and J field.
        # This is measurement plumbing, not the redistribution repair.
        out_arrays = [x.copy() for x in arrays_desc]
        out_arrays[4] = chi_effective[:, ::-1].reshape(-1)
        out_arrays[6] = eta_coherent_effective[:, ::-1].reshape(-1)
        out_arrays[7] = eta_total_effective[:, ::-1].reshape(-1)
        out_arrays[8] = j_effective[:, ::-1].reshape(-1)
        raw = serialize(header, out_arrays)
        payload_path = args.out_dir / "emiss_e9_effective_iter10"
        payload_path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        manifest = {
            "schema": "LCMFCE01-v1", "sha256": digest,
            "iteration": int(header[5]), "field_generation": int(header[6]),
            "post_damping": True, "coherent_frozen": True,
            "frequency_descending": True,
            "eta_decomposition_bitwise": True,
            "eta_decomposition_max_abs": 0,
            "e9_diagnostic_only": True,
            "source_payload_sha256": checked.manifest["sha256"],
            "construction": "cellwise J_old*(eps_old/eps_MC), then eta_fixed+[chi_es_proxy+(1-eps_MC)chi_line_proxy]*J_effective",
            "chi_es_proxy": "per-shell minimum payload chi_coherent (capture-epoch line-free proxy)",
            "repair_implemented": False,
            "clamp_or_floor_added": False,
        }
        Path(str(payload_path) + ".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Fail closed on the newly written diagnostic contract.
        check_artifact(payload_path)

        target_line_proxy = chi_line_proxy[s]
        summary = {
            "schema": "lumina-emiss-e9-prediction-v1",
            "shell": s, "MC_terminals": terminals, "MC_destroyed": destroyed,
            "eps_MC": eps_mc, "gain_MC": gain_mc,
            "payload_input": str(source_path),
            "payload_input_sha256": checked.manifest["sha256"],
            "effective_payload": str(payload_path.resolve()),
            "effective_payload_sha256": digest,
            "cmfgen": cmf_meta,
            "bands": rows,
            "target_shell_line_proxy": {
                "negative_cells_all_frequencies": int(np.sum(target_line_proxy < 0.0)),
                "minimum": float(np.min(target_line_proxy)),
                "chi_es_payload_min_proxy": float(chi_es_proxy[s, 0]),
                "chi_es_final_ne_proxy": float(chi_es_final[s, 0]),
                "payload_min_over_final_ne_proxy": float(
                    chi_es_proxy[s, 0] / chi_es_final[s, 0]),
                "note": "payload-min proxy is nonnegative by construction but is not an exact serialized component split",
            },
            "all_shell_line_proxy_negative_cells": int(np.sum(chi_line_proxy < 0.0)),
            "final_ne_proxy_would_make_negative_cells": int(np.sum(
                fields["chi_coherent"] - chi_es_final < 0.0)),
            "no_transport_or_plasma_solve_in_this_script": True,
            "no_new_clamp": True,
            "repair_implemented": False,
        }
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({
            "eps_MC": eps_mc, "gain_MC": gain_mc,
            "BALL": next(r for r in rows if r["band"] == "BALL"),
            "effective_payload_sha256": digest,
        }, indent=2, allow_nan=False))
        return 0
    except (E9Error, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
