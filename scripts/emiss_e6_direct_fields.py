#!/usr/bin/env python3
"""E6: compare captured A/B/B2 emissivity and source fields without transport.

This program is deliberately an arithmetic-only consumer.  It does not compile
or call the Stage-3.1 transport solver.  CMFGEN jnu4 is mapped with the existing
RVTJ log-J velocity interpolation and integral-preserving frequency bin average.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from emiss_ab_insitu_e4 import validate_lanes  # noqa: E402
import stage31_cmf_field_bench as bench  # noqa: E402
import w3_gamma_triple_compare as gamma  # noqa: E402


C_ANGSTROM = 2.99792458e18
DEFAULT_BASE = Path(
    "/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/"
    "emiss_ab_iter10"
)
DEFAULT_E5 = ROOT / "validation/emiss_e5/verdict.json"
TRIP_RE = re.compile(
    r"radial=(\d+) frequency=(\d+) ray=(\d+) segment=(\d+) substep=(\d+)"
)


class E6Error(RuntimeError):
    pass


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise E6Error(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    width = math.fsum(float(x) for x in weights)
    if not width > 0.0:
        raise E6Error("empty band weight")
    return math.fsum(float(x) * float(w) for x, w in zip(values, weights)) / width


def weighted_integral(values: np.ndarray, weights: np.ndarray) -> float:
    return math.fsum(float(x) * float(w) for x, w in zip(values, weights))


def ratio(numerator: float, denominator: float) -> float | None:
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or denominator == 0.0:
        return None
    return numerator / denominator


def cmfgen_all_shells(edges: np.ndarray, velocities: np.ndarray,
                      cmf_run: Path) -> tuple[np.ndarray, dict[str, Any]]:
    edd = cmf_run / "EDDFACTOR"
    rvtj = cmf_run / "RVTJ"
    if not edd.is_file() or not rvtj.is_file():
        raise E6Error(f"CMFGEN jnu4 inputs missing under {cmf_run}")
    j_native, nu_native, nd, finish, meta = gamma.read_eddfactor(edd)
    text = rvtj.read_text()
    velocity_native = gamma.rvtj_block(text, "Velocity (km/s)", nd)
    order = np.argsort(velocity_native)
    velocity_sorted = velocity_native[order]
    fields = []
    brackets = []
    quadrature_ratios = []
    unresolved_shells = []
    for shell, velocity in enumerate(velocities):
        if velocity <= velocity_sorted[0] or velocity >= velocity_sorted[-1]:
            fields.append(np.full(edges.size - 1, np.nan))
            brackets.append({"velocity_kms": float(velocity),
                             "status": "UNRESOLVED-outside-RVTJ-grid"})
            unresolved_shells.append(shell)
            continue
        upper = int(np.searchsorted(velocity_sorted, velocity))
        v0, v1 = velocity_sorted[upper - 1], velocity_sorted[upper]
        weight = float((velocity - v0) / (v1 - v0))
        j0 = j_native[:, order[upper - 1]]
        j1 = j_native[:, order[upper]]
        if np.any(j0 <= 0.0) or np.any(j1 <= 0.0):
            raise E6Error(f"nonpositive CMFGEN J in velocity bracket for {velocity}")
        j_velocity = np.exp((1.0 - weight) * np.log(j0) + weight * np.log(j1))
        j_bar = gamma.bin_average(nu_native, j_velocity, edges)
        cumulative = gamma.cum_from_below(nu_native, j_velocity)
        endpoints = gamma.eval_cum(nu_native, j_velocity, cumulative,
                                   np.array([edges[0], edges[-1]]))
        native_integral = float(endpoints[1] - endpoints[0])
        binned_integral = float(np.sum(j_bar * np.diff(edges)))
        quadrature_ratios.append(binned_integral / native_integral)
        fields.append(j_bar)
        brackets.append({"velocity_kms": float(velocity), "status": "resolved",
                         "v0": float(v0), "v1": float(v1), "weight": weight})
    return np.asarray(fields), {
        "run": str(cmf_run.resolve()), "EDDFACTOR_bytes": edd.stat().st_size,
        "RVTJ_bytes": rvtj.stat().st_size, "finish": finish, "reader": meta,
        "shell_brackets": brackets,
        "unresolved_shells_outside_RVTJ_grid": unresolved_shells,
        "bin_integral_ratio_min": min(quadrature_ratios),
        "bin_integral_ratio_max": max(quadrature_ratios),
        "bin_integral_max_abs_error": max(abs(x - 1.0) for x in quadrature_ratios),
    }


def trip_location(e5_path: Path) -> dict[str, int]:
    try:
        e5 = json.loads(e5_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise E6Error(f"cannot read E5 verdict {e5_path}: {exc}") from exc
    found = []
    for lane in ("B", "B2"):
        entry = e5.get("stage31", {}).get(lane, {})
        match = TRIP_RE.search(str(entry.get("error", "")))
        if match is None:
            raise E6Error(f"E5 {lane} trip coordinates absent")
        found.append(tuple(int(x) for x in match.groups()))
    if found[0] != found[1]:
        raise E6Error(f"B/B2 trip coordinates differ: {found}")
    return dict(zip(("radial", "frequency", "ray", "segment", "substep"),
                    found[0], strict=True))


def gauss_nodes_unit(n: int) -> np.ndarray:
    # numpy's ascending roots transformed to [0,1] have the same ordering as
    # lumina_cmf_ray_cache_build_at_radius.
    nodes, _ = np.polynomial.legendre.leggauss(n)
    return 0.5 * (nodes + 1.0)


def reconstruct_segment(r_edge: np.ndarray, trip: dict[str, int],
                        nmu: int) -> dict[str, Any]:
    radial = trip["radial"]
    flattened_ray = trip["ray"]
    if flattened_ray // nmu != radial:
        raise E6Error("E5 flattened ray is inconsistent with radial index/nmu")
    m = flattened_ray % nmu
    centers = 0.5 * (r_edge[:-1] + r_edge[1:])
    target = centers[radial]
    mu = float(gauss_nodes_unit(nmu)[m])
    impact = target * math.sqrt(1.0 - mu * mu)
    inner, outer = r_edge[0], r_edge[-1]
    z = [-math.sqrt(outer * outer - impact * impact),
         math.sqrt(outer * outer - impact * impact)]
    for radius in centers:
        if radius > impact:
            root = math.sqrt(radius * radius - impact * impact)
            z.extend((-root, root))
    if impact < inner:
        root = math.sqrt(inner * inner - impact * impact)
        z.extend((-root, root))
    else:
        z.append(0.0)
    target_z = math.sqrt(target * target - impact * impact)
    z.extend((-target_z, target_z))
    z = sorted(set(z))
    segment = trip["segment"]
    if segment + 1 >= len(z):
        raise E6Error("E5 segment index outside reconstructed path")
    midpoint_z = 0.5 * (z[segment] + z[segment + 1])
    midpoint_r = math.sqrt(impact * impact + midpoint_z * midpoint_z)
    upper = int(np.searchsorted(centers, midpoint_r))
    if upper <= 0:
        bracket = [0, 1]
    elif upper >= len(centers):
        bracket = [len(centers) - 2, len(centers) - 1]
    else:
        bracket = [upper - 1, upper]
    return {
        "nmu": nmu, "mu_index": m, "mu": mu, "impact_cm": impact,
        "path_node_count": len(z), "z_up_cm": z[segment],
        "z_down_cm": z[segment + 1], "midpoint_radius_cm": midpoint_r,
        "radial_interpolation_shells": bracket,
    }


def field_counts(field: np.ndarray, band_mask: np.ndarray) -> dict[str, int]:
    selected = field[:, band_mask]
    return {"negative": int(np.sum(selected < 0.0)),
            "zero": int(np.sum(selected == 0.0)),
            "nonfinite": int(np.sum(~np.isfinite(selected)))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", nargs="?", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--cmf-run", type=Path, default=bench.DEFAULT_CMF)
    parser.add_argument("--e5-verdict", type=Path, default=DEFAULT_E5)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "validation/emiss_e6")
    parser.add_argument("--nmu", type=int, default=16)
    args = parser.parse_args()
    try:
        a, b, b2 = validate_lanes(args.base.resolve(), allow_seeded=False)
        artifacts = {"A": a, "B": b, "B2": b2}
        nr, nnu = a.header[3], a.header[4]
        arrays = {
            lane: {
                "chi": np.asarray(item.arrays[3]).reshape(nr, nnu)[:, ::-1],
                "eta_fixed": np.asarray(item.arrays[5]).reshape(nr, nnu)[:, ::-1],
                "eta_coherent": np.asarray(item.arrays[6]).reshape(nr, nnu)[:, ::-1],
                "eta_total": np.asarray(item.arrays[7]).reshape(nr, nnu)[:, ::-1],
            } for lane, item in artifacts.items()
        }
        edges, centers, widths = bench.canonical_grid()
        payload_nu = np.asarray(a.arrays[1])[::-1]
        payload_dnu = np.asarray(a.arrays[2])[::-1]
        grid_nu_error = float(np.max(np.abs(payload_nu / centers - 1.0)))
        grid_dnu_error = float(np.max(np.abs(payload_dnu / widths - 1.0)))
        # The payload dnu is the producer's center-times-dlog width, whereas
        # the established CMFGEN comparator integrates over exact log-bin
        # edges.  Frequency centers are the identity contract; retain the dnu
        # discrepancy as an audit value instead of silently substituting it.
        if grid_nu_error > 1.0e-12 or grid_dnu_error > 2.0e-6:
            raise E6Error(f"payload/canonical grid mismatch nu={grid_nu_error} dnu={grid_dnu_error}")
        r_edge = np.asarray(a.arrays[0])
        shell_width = np.diff(r_edge)
        shell_velocity = 0.5 * (r_edge[:-1] + r_edge[1:]) / a.header[-1] / 1.0e5
        cmf_j, cmf_meta = cmfgen_all_shells(edges, shell_velocity,
                                             args.cmf_run.resolve())
        resolved_cmf = np.isfinite(cmf_j).all(axis=1)
        # Deep-Wien bins outside the requested 600--3000 A interval may
        # underflow to exact zero after averaging.  Negative/nonfinite resolved
        # values remain forbidden; requested band means are checked below.
        if (np.any(~np.isfinite(cmf_j[resolved_cmf])) or
                np.any(cmf_j[resolved_cmf] < 0.0)):
            raise E6Error("resolved CMFGEN mapped field is negative/nonfinite")

        chi = arrays["A"]["chi"]
        if any(not np.array_equal(chi, arrays[lane]["chi"])
               for lane in ("B", "B2")):
            raise E6Error("chi_total differs across lanes")
        tau_cell = chi * shell_width[:, None]
        tau_out = np.cumsum(tau_cell[::-1], axis=0)[::-1]
        source = {lane: arrays[lane]["eta_total"] / chi
                  for lane in ("A", "B", "B2")}
        wavelength = C_ANGSTROM / centers
        uv_mask = (wavelength >= 600.0) & (wavelength <= 3000.0)

        band_rows: list[dict[str, Any]] = []
        for band, lo, hi in bench.BANDS:
            weights = bench.band_weights(edges, lo, hi)
            width = float(np.sum(weights))
            for shell in range(nr):
                eta_mean = {lane: weighted_mean(arrays[lane]["eta_total"][shell], weights)
                            for lane in ("A", "B", "B2")}
                eta_integral = {
                    lane: weighted_integral(arrays[lane]["eta_total"][shell], weights)
                    for lane in ("A", "B", "B2")
                }
                s_mean = {lane: weighted_mean(source[lane][shell], weights)
                          for lane in ("A", "B", "B2")}
                j_mean = (weighted_mean(cmf_j[shell], weights)
                          if resolved_cmf[shell] else None)
                if j_mean is not None and not j_mean > 0.0:
                    raise E6Error(f"nonpositive CMFGEN mean at shell {shell} band {band}")
                a_dex = (math.log10(s_mean["A"] / j_mean)
                         if j_mean is not None else None)
                b2_dex = (math.log10(s_mean["B2"] / j_mean)
                          if j_mean is not None else None)
                abs_reduction = (abs(a_dex) - abs(b2_dex)
                                 if a_dex is not None and b2_dex is not None else None)
                closure = (abs_reduction / abs(a_dex)
                           if a_dex not in (None, 0.0) else None)
                ge1_fraction = weighted_mean((tau_out[shell] >= 1.0).astype(float), weights)
                ge10_fraction = weighted_mean((tau_out[shell] >= 10.0).astype(float), weights)
                covered_delta = arrays["B2"]["eta_fixed"][shell] - arrays["A"]["eta_fixed"][shell]
                retained_undefined = (arrays["B2"]["eta_fixed"][shell]
                                      - arrays["B"]["eta_fixed"][shell])
                row = {
                    "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
                    "shell": shell, "velocity_kms": shell_velocity[shell],
                    "bandwidth_Hz": width,
                    "eta_A_mean_per_Hz": eta_mean["A"],
                    "eta_B_mean_per_Hz": eta_mean["B"],
                    "eta_B2_mean_per_Hz": eta_mean["B2"],
                    "eta_A_integral": eta_integral["A"],
                    "eta_B_integral": eta_integral["B"],
                    "eta_B2_integral": eta_integral["B2"],
                    "eta_B_over_A": ratio(eta_integral["B"], eta_integral["A"]),
                    "eta_B2_over_A": ratio(eta_integral["B2"], eta_integral["A"]),
                    "eta_B2_over_B": ratio(eta_integral["B2"], eta_integral["B"]),
                    "covered_delta_B2_minus_A_integral": weighted_integral(covered_delta, weights),
                    "undefined_retained_B2_minus_B_integral": weighted_integral(retained_undefined, weights),
                    "S_A_mean": s_mean["A"], "S_B_mean": s_mean["B"],
                    "S_B2_mean": s_mean["B2"],
                    "S_B_over_A": ratio(s_mean["B"], s_mean["A"]),
                    "S_B2_over_A": ratio(s_mean["B2"], s_mean["A"]),
                    "J_CMFGEN_mean": j_mean,
                    "S_A_over_CMFGEN": (s_mean["A"] / j_mean if j_mean is not None else None),
                    "S_B_over_CMFGEN": (s_mean["B"] / j_mean if j_mean is not None else None),
                    "S_B2_over_CMFGEN": (s_mean["B2"] / j_mean if j_mean is not None else None),
                    "A_log10_over_CMFGEN": a_dex,
                    "B2_log10_over_CMFGEN": b2_dex,
                    "B2_toward_CMFGEN": (abs(b2_dex) < abs(a_dex)
                                           if a_dex is not None and b2_dex is not None
                                           else None),
                    "abs_dex_reduction_A_to_B2": abs_reduction,
                    "fractional_abs_dex_closure": closure,
                    "tau_cell_mean": weighted_mean(tau_cell[shell], weights),
                    "tau_out_mean": weighted_mean(tau_out[shell], weights),
                    "tau_out_ge1_fraction": ge1_fraction,
                    "tau_out_ge10_fraction": ge10_fraction,
                    "thick90": ge1_fraction >= 0.9,
                }
                band_rows.append(row)

        trip = trip_location(args.e5_verdict.resolve())
        trip_geometry = reconstruct_segment(r_edge, trip, args.nmu)
        k_desc = trip["frequency"]
        k_asc = nnu - 1 - k_desc
        if k_asc <= 0 or k_asc >= nnu - 1:
            raise E6Error("trip bin lacks two neighbors")
        trip_rows: list[dict[str, Any]] = []
        for shell in range(nr):
            base_row: dict[str, Any] = {
                "shell": shell, "velocity_kms": shell_velocity[shell],
                "payload_frequency_index": k_desc,
                "frequency_Hz": centers[k_asc],
                "wavelength_A": C_ANGSTROM / centers[k_asc],
                "chi_total": chi[shell, k_asc],
            }
            for lane in ("A", "B", "B2"):
                eta = arrays[lane]["eta_total"][shell]
                sfield = source[lane][shell]
                base_row[f"eta_{lane}"] = eta[k_asc]
                base_row[f"S_{lane}"] = sfield[k_asc]
                base_row[f"eta_{lane}_over_blue_neighbor"] = eta[k_asc] / eta[k_asc + 1]
                base_row[f"eta_{lane}_over_red_neighbor"] = eta[k_asc] / eta[k_asc - 1]
                base_row[f"eta_{lane}_over_neighbor_mean"] = (
                    eta[k_asc] / (0.5 * (eta[k_asc - 1] + eta[k_asc + 1])))
                base_row[f"S_{lane}_over_blue_neighbor"] = sfield[k_asc] / sfield[k_asc + 1]
                base_row[f"S_{lane}_over_red_neighbor"] = sfield[k_asc] / sfield[k_asc - 1]
                base_row[f"S_{lane}_over_neighbor_mean"] = (
                    sfield[k_asc] / (0.5 * (sfield[k_asc - 1] + sfield[k_asc + 1])))
            base_row["eta_B_over_A"] = base_row["eta_B"] / base_row["eta_A"]
            base_row["eta_B2_over_A"] = base_row["eta_B2"] / base_row["eta_A"]
            base_row["eta_B2_over_B"] = base_row["eta_B2"] / base_row["eta_B"]
            base_row["S_B_over_A"] = base_row["S_B"] / base_row["S_A"]
            base_row["S_B2_over_A"] = base_row["S_B2"] / base_row["S_A"]
            base_row["S_B2_over_B"] = base_row["S_B2"] / base_row["S_B"]
            trip_rows.append(base_row)

        band_summary = []
        for band, lo, hi in bench.BANDS:
            rows = [r for r in band_rows if r["band"] == band]
            thick = [r for r in rows if r["thick90"]]
            population = thick if thick else rows
            def stats(key: str) -> dict[str, float]:
                values = np.asarray([float(r[key]) for r in population
                                     if r[key] is not None])
                if not values.size:
                    raise E6Error(f"no resolved {key} values for band {band}")
                return {"min": float(values.min()), "median": float(np.median(values)),
                        "max": float(values.max())}
            band_summary.append({
                "band": band, "wavelength_A": [lo, hi],
                "thick90_shell_count": len(thick),
                "thick90_shell_range": ([thick[0]["shell"], thick[-1]["shell"]]
                                          if thick else None),
                "summary_population": "thick90" if thick else "all-shell-fallback",
                "eta_B2_over_A": stats("eta_B2_over_A"),
                "S_A_over_CMFGEN": stats("S_A_over_CMFGEN"),
                "S_B2_over_CMFGEN": stats("S_B2_over_CMFGEN"),
                "S_B2_over_A": stats("S_B2_over_A"),
                "fractional_abs_dex_closure": stats("fractional_abs_dex_closure"),
                "toward_CMFGEN_count": sum(r["B2_toward_CMFGEN"] is True
                                            for r in population),
                "resolved_CMFGEN_population_count": sum(
                    r["B2_toward_CMFGEN"] is not None for r in population),
                "population_count": len(population),
            })

        trip_target = trip_rows[trip["radial"]]
        bracket_rows = [trip_rows[x] for x in trip_geometry["radial_interpolation_shells"]]
        summary = {
            "schema": "lumina-emiss-e6-direct-fields-v1",
            "arithmetic_only_no_transport_solve": True,
            "base": str(args.base.resolve()),
            "payload_sha256": {lane: artifacts[lane].manifest["sha256"]
                                for lane in ("A", "B", "B2")},
            "common_assembly_state_sha256": a.manifest["common_assembly_state_sha256"],
            "grid": {"nu_max_relative_error": grid_nu_error,
                     "dnu_max_relative_error": grid_dnu_error},
            "cmfgen": cmf_meta,
            "field_anomalies_600_3000": {
                lane: {quantity: field_counts(arrays[lane][quantity], uv_mask)
                       for quantity in ("chi", "eta_fixed", "eta_coherent", "eta_total")}
                for lane in ("A", "B", "B2")
            },
            "line_separation": {
                "total_line_absolute": "UNRESOLVED",
                "reason": "LCMFCE01 has no continuum/line split",
                "identifiable": ["B2-A covered-formulation delta",
                                 "B2-B retained-undefined contribution"],
                "undefined_manifest_epoch": "pre-EPAY",
                "payload_lane_differences_epoch": "post-EPAY",
            },
            "band_summary": band_summary,
            "trip": {**trip, **trip_geometry,
                     "frequency_Hz": float(centers[k_asc]),
                     "wavelength_A": float(C_ANGSTROM / centers[k_asc]),
                     "target_shell_row": trip_target,
                     "segment_bracket_rows": bracket_rows},
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_dir / "band_shell.csv", band_rows)
        write_csv(args.out_dir / "trip_1208_shells.csv", trip_rows)
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(f"PASS arithmetic-only: {len(band_rows)} band-shell rows")
        print(f"CMFGEN bin integral max abs error: {cmf_meta['bin_integral_max_abs_error']:.3e}")
        print(f"trip: shell={trip['radial']} bin={trip['frequency']} "
              f"lambda={summary['trip']['wavelength_A']:.12f} A")
        print(f"outputs: {args.out_dir.resolve()}")
        return 0
    except (E6Error, OSError, ValueError, KeyError, gamma.Unresolved) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
