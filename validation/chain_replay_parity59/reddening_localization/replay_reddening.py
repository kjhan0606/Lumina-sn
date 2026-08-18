#!/usr/bin/env python3
"""Parameterized CSV/EDDFACTOR replay of reddening-localization Task A.

This script never reads lumina_events.bin.  Event-derived quantities remain
UNRESOLVED and are listed by the driver/report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import C_A, cmfgen_field, cmfgen_j_at_velocity, integrate_j, load_field, write_csv

SHELLS = [(0, 4264, "s0"), (2, 5720, "s2"), (4, 7176, "s4")]
EDGES = [100, 300, 450, 918, 1290, 2000, 3000, 4500, 7000, 10000, 19933]
LABELS = ["soft_100_300", "EUV_300_450", "xuv_450_918", "FUV_918_1290",
          "NUV_1290_2000", "UV_2000_3000", "blue_3000_4500", "opt_4500_7000",
          "red_7000_10000", "NIR_10000_19933"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cmfgen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    field_path = args.input_dir / "lumina_coevolve_field.csv"
    if not field_path.is_file():
        raise FileNotFoundError(field_path)
    field = load_field(field_path)
    lam_c, j_c_all, v_c = cmfgen_field(args.cmfgen_dir)

    band_rows: list[list[object]] = []
    overlay_rows: list[list[object]] = []
    concentration_rows: list[list[object]] = []
    ratio_rows: list[list[object]] = []
    log_edges = np.logspace(np.log10(100.0), np.log10(19933.0), 41)

    for shell, target_v, label in SHELLS:
        data = field[shell]
        lam_l = data["wavelength_A"]
        mc = data["mc_J"]
        cs = data["cs_J"]
        jc = cmfgen_j_at_velocity(lam_c, j_c_all, v_c, target_v)
        total_l = integrate_j(lam_l, mc, 100, 19933)
        total_c = integrate_j(lam_c, jc, 100, 19933)
        for name, lo, hi in zip(LABELS, EDGES[:-1], EDGES[1:]):
            ul = integrate_j(lam_l, mc, lo, hi)
            ucs = integrate_j(lam_l, cs, lo, hi)
            uc = integrate_j(lam_c, jc, lo, hi)
            band_rows.append([label, shell, target_v, name, lo, hi, uc, ul, ucs, ul / uc, ul / ucs,
                              np.log10(ul / uc), uc / total_c, ul / total_l,
                               str(args.cmfgen_dir / "EDDFACTOR"), "J_nu", str(field_path), "mc_J; cs_J"])

        order_c = np.argsort(lam_c)
        if np.any(jc[order_c] <= 0):
            raise ValueError(f"CMFGEN field has non-positive values at {label}")
        jc_interp = 10.0 ** np.interp(lam_l, lam_c[order_c], np.log10(jc[order_c]))
        for wavelength, mc_value, cs_value, cmf_value in zip(lam_l, mc, cs, jc_interp):
            overlay_rows.append([label, shell, target_v, wavelength, mc_value, cs_value, cmf_value,
                                 mc_value / cmf_value, mc_value / cs_value, str(field_path),
                                 "wavelength_A; mc_J; cs_J"])

        coarse_bins: list[tuple[float, float, float]] = []
        for lo, hi in zip(log_edges[:-1], log_edges[1:]):
            ul = integrate_j(lam_l, mc, lo, hi)
            uc = integrate_j(lam_c, jc, lo, hi)
            coarse_bins.append((np.sqrt(lo * hi), ul, uc))
        coarse_l_total = sum(item[1] for item in coarse_bins)
        coarse_c_total = sum(item[2] for item in coarse_bins)
        fractions = [(mid, ul / coarse_l_total, uc / coarse_c_total) for mid, ul, uc in coarse_bins]
        peak = max(fractions, key=lambda item: item[1])
        concentration_rows.append([label, shell, target_v, peak[0], peak[1], peak[2],
                                   "40 equal log-lambda bins from 100 to 19933 A; fraction denominator is sum of the 40 independently integrated bins (historical taskA_sed_shape.py definition)",
                                   str(field_path), "wavelength_A; mc_J", str(args.cmfgen_dir / "EDDFACTOR"), "J_nu"])

        nearest = int(np.argmin(np.abs(lam_l - 1526.17)))
        ratio_rows.append([label, shell, target_v, 1526.17, lam_l[nearest], mc[nearest], cs[nearest],
                           mc[nearest] / cs[nearest], str(field_path), "wavelength_A; mc_J; cs_J"])

    write_csv(args.output_dir / "taskA_band_table.csv",
              ["shell_label", "shell", "v_kms", "band", "lo_A", "hi_A", "u_cmfgen", "u_lumina_mc",
               "u_lumina_cs", "mc_over_cmfgen", "mc_over_cs", "dex", "frac_cmfgen", "frac_lumina",
               "cmfgen_source_file", "cmfgen_field", "lumina_source_file", "lumina_field"], band_rows)
    write_csv(args.output_dir / "taskA_overlay_spectrum.csv",
              ["shell_label", "shell", "v_kms", "wavelength_A", "mc_J", "cs_J", "cmfgen_J_interp",
               "mc_over_cmfgen", "mc_over_cs", "source_file", "source_fields"], overlay_rows)
    write_csv(args.output_dir / "taskA_logbin_concentration.csv",
              ["shell_label", "shell", "v_kms", "peak_logbin_mid_A", "lumina_u_fraction",
               "cmfgen_u_fraction_same_bin", "definition", "lumina_source_file", "lumina_fields",
               "cmfgen_source_file", "cmfgen_field"], concentration_rows)
    write_csv(args.output_dir / "taskA_mc_cs_1526.csv",
              ["shell_label", "shell", "v_kms", "target_A", "native_bin_A", "mc_J", "cs_J", "mc_over_cs",
               "source_file", "source_fields"], ratio_rows)


if __name__ == "__main__":
    main()
