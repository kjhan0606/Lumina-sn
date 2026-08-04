#!/usr/bin/env python3
"""E7 arithmetic-only audit of fixed source, opacity, and upper populations.

This consumer reads existing LCMFCE01 payloads, CMFGEN jnu4/MEANOPAC, and
archived Lumina plasma/population CSVs.  It does not compile or invoke a
transport/model solver.  Quantities that cannot be recovered from the frozen
epoch (pure electron scattering, bound-free split, per-line epsilon) are
reported as bounds/proxies rather than silently identified with mixed fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from emiss_ab_insitu_e4 import validate_lanes  # noqa: E402
import stage31_cmf_field_bench as bench  # noqa: E402
from emiss_e6_direct_fields import cmfgen_all_shells, weighted_mean  # noqa: E402


C_A_S = 2.99792458e18
C_CGS = 2.99792458e10
H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16
KB_EV = 8.617333262e-5
SIGMA_T_CODE = 6.6524587e-25
SOBOLEV_COEFF = 2.6540281e-2
EPS_FLOOR_CONFIG = 1.0e-5
SHELLS = (0, 8, 20)

DEFAULT_RUN = Path(
    "/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766"
)
DEFAULT_CMF = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
DEFAULT_MODEL = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv"


class E7Error(RuntimeError):
    pass


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise E7Error(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_ratio(a: float, b: float) -> float | None:
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0.0:
        return None
    return a / b


def read_plasma(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            shell = int(row["shell_id"])
            rows[shell] = {key: float(row[key]) for key in ("W", "T_rad", "n_e", "T_e")}
    if sorted(rows) != list(range(50)):
        raise E7Error("plasma CSV is not a complete 50-shell state")
    return rows


def read_ion_pops(path: Path) -> dict[tuple[int, int, int], float]:
    result: dict[tuple[int, int, int], float] = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            shell = int(row["shell_id"])
            if shell in SHELLS:
                result[(shell, int(row["Z"]), int(row["stage"]))] = float(row["n_ion"])
    return result


def read_meanopac(path: Path) -> dict[str, np.ndarray]:
    values = []
    with path.open() as stream:
        for line in stream:
            parts = line.split()
            if len(parts) != 15:
                continue
            try:
                values.append([float(x) for x in parts])
            except ValueError:
                continue
    table = np.asarray(values)
    if table.shape != (90, 15):
        raise E7Error(f"unexpected MEANOPAC table shape {table.shape}")
    order = np.argsort(table[:, 14])
    return {
        "velocity_kms": table[order, 14],
        "tau_ross": table[order, 2],
        "tau_es": table[order, 10],
        "chi_ross": table[order, 5],
        "chi_es": table[order, 8],
    }


def planck(nu: float, temperature: float) -> float:
    x = H_CGS * nu / (K_CGS * temperature)
    if x >= 700.0:
        return 0.0
    return 2.0 * H_CGS * nu**3 / C_CGS**2 / math.expm1(x)


def line_candidates(line_path: Path, target_lo: float, target_hi: float
                    ) -> list[dict[str, Any]]:
    result = []
    with line_path.open() as stream:
        for row in csv.DictReader(stream):
            wavelength = float(row["wavelength"])
            z = int(row["atomic_number"])
            ion = int(row["ion_number"])
            in_target = target_lo <= wavelength <= target_hi
            iron_peak_uv = (z, ion) in ((26, 2), (27, 2)) and 1000.0 <= wavelength <= 3000.0
            if not (in_target or iron_peak_uv):
                continue
            result.append({
                "line_id": int(row["line_id"]), "Z": z, "ion": ion,
                "lower": int(row["level_number_lower"]),
                "upper": int(row["level_number_upper"]),
                "wavelength_A": wavelength, "nu": float(row["nu"]),
                "f_lu": float(row["f_lu"]), "A_ul": float(row["A_ul"]),
                "B_lu": float(row["B_lu"]), "B_ul": float(row["B_ul"]),
                "in_target_bin": in_target,
            })
    if not result:
        raise E7Error("no target/Fe III/Co III UV lines found")
    return result


def read_level_pops(path: Path, needed: set[tuple[int, int, int]]) -> dict[tuple[int, int, int, int], dict[str, float]]:
    result: dict[tuple[int, int, int, int], dict[str, float]] = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            shell = int(row["shell"])
            if shell not in SHELLS:
                continue
            key3 = (int(row["Z"]), int(row["ion"]), int(row["level_num"]))
            if key3 not in needed:
                continue
            result[(shell, *key3)] = {
                "E_eV": float(row["E_eV"]), "g": float(row["g"]),
                "n_k": float(row["n_k"]), "n_ground": float(row["n_ground"]),
                "b_ground": float(row["b_k"]),
            }
    return result


def read_level_partitions(path: Path, ions: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, float, float]]]:
    result: dict[tuple[int, int], list[tuple[int, float, float]]] = {key: [] for key in ions}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = (int(row["atomic_number"]), int(row["ion_number"]))
            if key in result:
                result[key].append((int(row["level_number"]), float(row["energy_eV"]), float(row["g"])))
    return result


def make_line_rows(candidates: list[dict[str, Any]], level_pops: dict,
                   partitions: dict, ion_pops: dict, plasma: dict,
                   t_exp: float, target_width_hz: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored: dict[tuple[int, int], list[tuple[float, int]]] = {}
    metrics: dict[tuple[int, int], dict[str, Any]] = {}
    for ci, line in enumerate(candidates):
        for shell in SHELLS:
            upper = level_pops.get((shell, line["Z"], line["ion"], line["upper"]))
            lower = level_pops.get((shell, line["Z"], line["ion"], line["lower"]))
            if upper is None:
                continue
            direct = H_CGS * line["nu"] / (4.0 * math.pi) * line["A_ul"] * upper["n_k"]
            tau = None
            eps1 = None
            ratio_lo = None
            ratio_hi = None
            if lower is not None and lower["n_k"] > 0.0 and upper["n_k"] >= 0.0:
                stim = 1.0 - lower["g"] * upper["n_k"] / (upper["g"] * lower["n_k"])
                if stim > 0.0:
                    tau = SOBOLEV_COEFF * line["f_lu"] * (line["wavelength_A"] * 1e-8) * t_exp * lower["n_k"] * stim
                    frac = -math.expm1(-tau) if tau < 700.0 else 1.0
                    eps1 = frac * line["nu"] / (C_CGS * t_exp) * planck(line["nu"], plasma[shell]["T_e"])
                    ratio_lo = finite_ratio(direct, eps1)
                    ratio_hi = ratio_lo / EPS_FLOOR_CONFIG if ratio_lo is not None else None
            metrics[(ci, shell)] = {
                "upper": upper, "lower": lower, "direct": direct, "tau": tau,
                "eps1": eps1, "ratio_lo": ratio_lo, "ratio_hi": ratio_hi,
            }
            if line["in_target_bin"]:
                scored.setdefault((shell, 0), []).append((direct, ci))
            if shell == 8 and (line["Z"], line["ion"]) == (26, 2):
                scored.setdefault((shell, 26), []).append((direct, ci))
            if shell == 8 and (line["Z"], line["ion"]) == (27, 2):
                scored.setdefault((shell, 27), []).append((direct, ci))

    chosen: dict[int, set[str]] = {}
    for (shell, group), values in scored.items():
        label = f"target1208_top_s{shell}" if group == 0 else f"{'FeIII' if group == 26 else 'CoIII'}_UV_top_s8"
        for _, ci in sorted(values, reverse=True)[:5]:
            chosen.setdefault(ci, set()).add(label)

    rows = []
    for ci in sorted(chosen, key=lambda x: candidates[x]["line_id"]):
        line = candidates[ci]
        levels = partitions[(line["Z"], line["ion"])]
        for shell in SHELLS:
            data = metrics.get((ci, shell))
            upper = data["upper"] if data else None
            nion = ion_pops.get((shell, line["Z"], line["ion"]), 0.0)
            temperature = plasma[shell]["T_e"]
            zpart = math.fsum(g * math.exp(-energy / (KB_EV * temperature)) for _, energy, g in levels)
            n_lte = None
            b_abs = None
            if upper is not None and nion > 0.0 and zpart > 0.0:
                n_lte = nion * upper["g"] * math.exp(-upper["E_eV"] / (KB_EV * temperature)) / zpart
                b_abs = finite_ratio(upper["n_k"], n_lte)
            rows.append({
                "categories": ";".join(sorted(chosen[ci])), "shell": shell,
                "line_id": line["line_id"], "Z": line["Z"], "ion": line["ion"],
                "lower_level": line["lower"], "upper_level": line["upper"],
                "wavelength_A": line["wavelength_A"], "A_ul_s-1": line["A_ul"],
                "upper_E_eV": upper["E_eV"] if upper else None,
                "upper_g": upper["g"] if upper else None,
                "n_upper": upper["n_k"] if upper else None,
                "n_upper_LTE_absolute_within_ion": n_lte,
                "b_upper_absolute_within_ion": b_abs,
                "b_upper_relative_to_own_ground_dump": upper["b_ground"] if upper else None,
                "tau_sob_final_state_proxy": data["tau"] if data else None,
                "Aul_nu_line_power_proxy": data["direct"] if data else None,
                "epsB_line_power_at_eps_1_proxy": data["eps1"] if data else None,
                "Aulnu_over_epsB_lower_eps1": data["ratio_lo"] if data else None,
                "Aulnu_over_epsB_upper_epsfloor_1e-5": data["ratio_hi"] if data else None,
                "exact_capture_epoch_ratio_status": "UNRESOLVED-missing-iter9-pop-tau-epsilon",
            })

    correlation = {}
    for shell in SHELLS:
        vals = [r["b_upper_absolute_within_ion"] for r in rows
                if r["shell"] == shell and r["Z"] == 26 and r["ion"] == 2
                and r["b_upper_absolute_within_ion"] is not None]
        correlation[str(shell)] = ({"FeIII_selected_b_min": min(vals),
                                    "FeIII_selected_b_median": float(np.median(vals)),
                                    "FeIII_selected_b_max": max(vals)} if vals else None)
    correlation["known_overionization_dex"] = {
        "s0_FeIII_to_IV": 0.97, "s8_FeII_to_III": 1.12,
        "s8_FeIII_to_IV": 2.37,
        "source": "docs/FABLE_WAVE3_INTERPRETATION.md section 1.2",
    }
    correlation["statistical_test"] = "UNRESOLVED-only-two-Fe-shell-sign-pairs-and-no-Co-link-table"
    return rows, correlation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cmf-run", type=Path, default=DEFAULT_CMF)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/emiss_e7")
    args = parser.parse_args()
    try:
        base = args.run.resolve() / "emiss_ab_iter10"
        a, b, b2 = validate_lanes(base, allow_seeded=False)
        artifacts = {"A": a, "B": b, "B2": b2}
        nr, nnu = a.header[3], a.header[4]
        fields = {}
        for lane, item in artifacts.items():
            fields[lane] = {
                "chi_total": np.asarray(item.arrays[3]).reshape(nr, nnu)[:, ::-1],
                "chi_coherent": np.asarray(item.arrays[4]).reshape(nr, nnu)[:, ::-1],
                "eta_fixed": np.asarray(item.arrays[5]).reshape(nr, nnu)[:, ::-1],
                "eta_coherent": np.asarray(item.arrays[6]).reshape(nr, nnu)[:, ::-1],
                "eta_total": np.asarray(item.arrays[7]).reshape(nr, nnu)[:, ::-1],
                "J_producer": np.asarray(item.arrays[8]).reshape(nr, nnu)[:, ::-1],
            }
        if any(not np.array_equal(fields["A"][key], fields[lane][key])
               for lane in ("B", "B2") for key in ("chi_total", "chi_coherent", "J_producer")):
            raise E7Error("A/B/B2 common opacity/J fields are not bitwise equal")

        edges, centers, _ = bench.canonical_grid()
        r_edge = np.asarray(a.arrays[0])
        shell_width = np.diff(r_edge)
        velocities = 0.5 * (r_edge[:-1] + r_edge[1:]) / a.header[-1] / 1e5
        cmf_j, cmf_meta = cmfgen_all_shells(edges, velocities, args.cmf_run.resolve())
        plasma = read_plasma(args.run.resolve() / "lumina_plasma_state.csv")
        meanopac = read_meanopac(args.cmf_run.resolve() / "MEANOPAC")
        chi = fields["A"]["chi_total"]
        coherent = fields["A"]["chi_coherent"]
        tau_out = np.cumsum((chi * shell_width[:, None])[::-1], axis=0)[::-1]

        band_rows = []
        for band, lo, hi in bench.BANDS:
            weights = bench.band_weights(edges, lo, hi)
            for shell in range(nr):
                jcmf = weighted_mean(cmf_j[shell], weights) if np.isfinite(cmf_j[shell]).all() else None
                sf = {lane: weighted_mean(fields[lane]["eta_fixed"][shell] / chi[shell], weights)
                      for lane in fields}
                st = {lane: weighted_mean(fields[lane]["eta_total"][shell] / chi[shell], weights)
                      for lane in fields}
                scoh = weighted_mean(fields["A"]["eta_coherent"][shell] / chi[shell], weights)
                chi_total = weighted_mean(chi[shell], weights)
                chi_mixed = weighted_mean(coherent[shell], weights)
                chi_e_final = plasma[shell]["n_e"] * SIGMA_T_CODE
                line_coherent_proxy = chi_mixed - chi_e_final
                noncoherent = chi_total - chi_mixed
                tau_ge1_fraction = weighted_mean((tau_out[shell] >= 1.0).astype(float), weights)
                band_rows.append({
                    "band": band, "lambda_lo_A": lo, "lambda_hi_A": hi,
                    "shell": shell, "velocity_kms": velocities[shell],
                    "S_fixed_A": sf["A"], "S_total_A_E6": st["A"],
                    "S_coherent_A": scoh,
                    "S_fixed_A_over_CMFGEN": finite_ratio(sf["A"], jcmf) if jcmf else None,
                    "S_total_A_over_CMFGEN_E6": finite_ratio(st["A"], jcmf) if jcmf else None,
                    "S_total_over_S_fixed": finite_ratio(st["A"], sf["A"]),
                    "coherent_fraction_of_S_total": finite_ratio(scoh, st["A"]),
                    "S_fixed_B_over_A": finite_ratio(sf["B"], sf["A"]),
                    "S_fixed_B2_over_A": finite_ratio(sf["B2"], sf["A"]),
                    "J_producer_mean": weighted_mean(fields["A"]["J_producer"][shell], weights),
                    "J_CMFGEN_mean": jcmf,
                    "chi_total_mean": chi_total,
                    "chi_payload_coherent_mixed_mean": chi_mixed,
                    "chi_e_final_ne_sigmaT": chi_e_final,
                    "chi_mixed_over_electron": finite_ratio(chi_mixed, chi_e_final),
                    "chi_coherent_line_proxy_mixed_minus_electron": line_coherent_proxy,
                    "chi_noncoherent_bf_ff_thermal_line": noncoherent,
                    "line_fraction_lower_coherent_only": finite_ratio(line_coherent_proxy, chi_total),
                    "line_fraction_upper_if_bf_ff_zero": finite_ratio(chi_total - chi_e_final, chi_total),
                    "tau_out_mean": weighted_mean(tau_out[shell], weights),
                    "tau_out_ge1_fraction": tau_ge1_fraction,
                    "thick90": tau_ge1_fraction >= 0.9,
                    "CMFGEN_tau_Ross_MEANOPAC": float(np.interp(velocities[shell], meanopac["velocity_kms"], meanopac["tau_ross"])),
                    "CMFGEN_tau_es_MEANOPAC": float(np.interp(velocities[shell], meanopac["velocity_kms"], meanopac["tau_es"])),
                })

        attenuation = []
        for band, _, _ in bench.BANDS:
            by_shell = {r["shell"]: r for r in band_rows if r["band"] == band}
            for inner, outer in ((0, 8), (8, 20)):
                ri, ro = by_shell[inner], by_shell[outer]
                attenuation.append({
                    "band": band, "inner_shell": inner, "outer_shell": outer,
                    "delta_ln_J_producer": math.log(ro["J_producer_mean"] / ri["J_producer_mean"]),
                    "delta_ln_J_CMFGEN": math.log(ro["J_CMFGEN_mean"] / ri["J_CMFGEN_mean"]),
                    "caveat": "effective depth indicator only; geometry and distributed source are entangled",
                })

        k_target = nnu - 1 - 470
        target_center = C_A_S / centers[k_target]
        target_lo = min(C_A_S / edges[k_target], C_A_S / edges[k_target + 1])
        target_hi = max(C_A_S / edges[k_target], C_A_S / edges[k_target + 1])
        candidates = line_candidates(args.model_dir.resolve() / "line_list.csv", target_lo, target_hi)
        needed = {(line["Z"], line["ion"], line["lower"]) for line in candidates}
        needed |= {(line["Z"], line["ion"], line["upper"]) for line in candidates}
        level_pops = read_level_pops(args.run.resolve() / "lumina_levelpop.csv", needed)
        ions = {(line["Z"], line["ion"]) for line in candidates}
        partitions = read_level_partitions(args.model_dir.resolve() / "levels.csv", ions)
        ion_pops = read_ion_pops(args.run.resolve() / "lumina_ion_pops.csv")
        line_rows, correlation = make_line_rows(
            candidates, level_pops, partitions, ion_pops, plasma,
            float(a.header[-1]), float(edges[k_target + 1] - edges[k_target]))

        target_payload = {}
        for shell in SHELLS:
            target_payload[str(shell)] = {}
            for quantity in ("eta_fixed", "eta_total"):
                av = fields["A"][quantity][shell, k_target]
                target_payload[str(shell)][quantity] = {
                    "A": float(av), "B": float(fields["B"][quantity][shell, k_target]),
                    "B2": float(fields["B2"][quantity][shell, k_target]),
                    "B2_over_A": float(fields["B2"][quantity][shell, k_target] / av),
                }

        category_stats = {}
        categories = sorted({cat for row in line_rows for cat in row["categories"].split(";")})
        for category in categories:
            category_stats[category] = {}
            for shell in SHELLS:
                selected = [row for row in line_rows
                            if row["shell"] == shell and category in row["categories"].split(";")]
                bvals = [row["b_upper_absolute_within_ion"] for row in selected
                         if row["b_upper_absolute_within_ion"] is not None]
                ratios = [row["Aulnu_over_epsB_lower_eps1"] for row in selected
                          if row["Aulnu_over_epsB_lower_eps1"] is not None]
                category_stats[category][str(shell)] = {
                    "row_count": len(selected),
                    "b_absolute_min_median_max": ([min(bvals), float(np.median(bvals)), max(bvals)]
                                                   if bvals else None),
                    "Aulnu_over_epsB_eps1_min_median_max": (
                        [min(ratios), float(np.median(ratios)), max(ratios)] if ratios else None),
                }

        def row_for(band: str, shell: int) -> dict[str, Any]:
            return next(r for r in band_rows if r["band"] == band and r["shell"] == shell)

        summary = {
            "schema": "lumina-emiss-e7-arithmetic-v1",
            "arithmetic_only_no_transport_solve": True,
            "payload_sha256": {lane: artifacts[lane].manifest["sha256"] for lane in artifacts},
            "cmfgen": cmf_meta,
            "fixed_source_canonical_s8": {band: {
                key: row_for(band, 8)[key] for key in (
                    "S_fixed_A_over_CMFGEN", "S_total_A_over_CMFGEN_E6",
                    "S_total_over_S_fixed", "coherent_fraction_of_S_total")}
                for band, _, _ in bench.BANDS},
            "fixed_source_thick90": {band: {
                "shell_count": len(selected),
                "shell_range": ([selected[0]["shell"], selected[-1]["shell"]] if selected else None),
                "S_fixed_A_over_CMFGEN_min_median_max": (
                    [min(vals), float(np.median(vals)), max(vals)] if vals else None),
                "S_total_A_over_CMFGEN_E6_min_median_max": (
                    [min(tvals), float(np.median(tvals)), max(tvals)] if tvals else None),
                "coherent_fraction_min_median_max": (
                    [min(cvals), float(np.median(cvals)), max(cvals)] if cvals else None),
            } for band, _, _ in bench.BANDS
              for selected in [[r for r in band_rows if r["band"] == band and r["thick90"]
                                and r["J_CMFGEN_mean"] is not None]]
              for vals in [[r["S_fixed_A_over_CMFGEN"] for r in selected]]
              for tvals in [[r["S_total_A_over_CMFGEN_E6"] for r in selected]]
              for cvals in [[r["coherent_fraction_of_S_total"] for r in selected]]},
            "opacity_canonical_shells": {str(shell): {band: {
                key: row_for(band, shell)[key] for key in (
                    "chi_total_mean", "chi_payload_coherent_mixed_mean",
                    "chi_e_final_ne_sigmaT", "chi_mixed_over_electron",
                    "line_fraction_lower_coherent_only", "line_fraction_upper_if_bf_ff_zero",
                    "tau_out_mean", "CMFGEN_tau_Ross_MEANOPAC", "CMFGEN_tau_es_MEANOPAC")}
                for band, _, _ in bench.BANDS} for shell in SHELLS},
            "opacity_identity": {
                "code_electron_term": "chi_e=n_e*6.6524587e-25 cm^-1",
                "payload_chi_coherent": "chi_e + (chi_line-chi_line_th); not pure electron scattering",
                "exact_capture_chi_bf_split": "UNRESOLVED-LCMFCE01-does-not-serialize-chi_abs-or-chi_line_th",
                "direct_CMFGEN_UV_chi_comparison": "UNRESOLVED-MEANOPAC-is-grey-Rosseland/flux-not-UV-bin-opacity",
            },
            "attenuation_indicators": attenuation,
            "target_1208": {"payload_index_descending": 470, "index_ascending": k_target,
                            "center_A": target_center, "bin_A": [target_lo, target_hi],
                            "payload_ratios": target_payload},
            "line_departure_correlation": correlation,
            "selected_line_category_stats": category_stats,
            "line_ratio_limit": "exact iter-9 n_u/tau/epsilon absent; CSV rows are final-state proxies and epsilon-range bounds",
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_dir / "band_shell_fixed_opacity.csv", band_rows)
        write_csv(args.out_dir / "j_depth_indicators.csv", attenuation)
        write_csv(args.out_dir / "line_departure_proxies.csv", line_rows)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(f"PASS arithmetic-only: {len(band_rows)} band-shell rows, {len(line_rows)} selected line-shell rows")
        print(f"target bin: {target_lo:.6f}--{target_hi:.6f} A, center={target_center:.12f} A")
        print(f"outputs: {args.out_dir.resolve()}")
        return 0
    except (E7Error, OSError, ValueError, KeyError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
