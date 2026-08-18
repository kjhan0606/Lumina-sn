#!/usr/bin/env python3
"""Same-snapshot closure for the 56-bin SH-GRID upper band and Si V edge."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import compare_sh_grid_low_band as low


FIRST_HIGH_BIN = 1178
EXPECTED_NBIN = 1234
EXPECTED_GLOBAL = 333
EXPECTED_Z = 14
EXPECTED_STAGE = 5
EXPECTED_ION = 4
EXPECTED_LEVEL = 0
RATE_LIMIT = 2.0e-2
ETA_LIMIT = 2.0e-2
EPOCH_SECONDS = 19.48 * 86400.0
RATE_EFFECT_LIMIT = 1.0e-12


def local_bin_average(x: np.ndarray, y: np.ndarray,
                      edges: np.ndarray) -> np.ndarray:
    """Bin averages without subtracting a huge below-band cumulative sum."""
    out = np.empty(edges.size - 1)
    for index, (lo_edge, hi_edge) in enumerate(zip(edges[:-1], edges[1:])):
        inside = (x > lo_edge) & (x < hi_edge)
        xb = np.concatenate(([lo_edge], x[inside], [hi_edge]))
        yb = np.interp(xb, x, y)
        out[index] = np.trapezoid(yb, xb) / (hi_edge - lo_edge)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=low.DEFAULT_DECK)
    parser.add_argument("--run", type=Path, default=low.DEFAULT_RUN)
    parser.add_argument(
        "--out", type=Path,
        default=Path("validation/sh_grid_upper_closure_2026-08-08/upper_band")
    )
    args = parser.parse_args()
    deck = args.deck.resolve()
    run = args.run.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    provenance = json.loads((deck / "DECK_PROVENANCE.json").read_text())
    active = low.read_rows(deck / "active_ions.csv")
    levels = low.read_rows(deck / "levels.csv")
    caps = {(int(row["atomic_number"]), int(row["ion_stage"])):
            int(row["n_full"]) for row in active}
    links = Path(provenance["cmfgen_links"])
    if low.sha256(links) != provenance["cmfgen_links_sha256"]:
        raise SystemExit("CMFGEN links hash changed")

    sigma_path = deck / "cmfgen_sigma_bf.bin"
    binary = low.read_sigma(sigma_path)
    if (binary["nlev"] != len(levels) or binary["nbin"] != EXPECTED_NBIN or
            binary["flags"][EXPECTED_GLOBAL] != 1):
        raise SystemExit("upper-band sigma contract mismatch")
    numin, numax = float(binary["numin"]), float(binary["numax"])
    dlog = math.log(numax / numin) / int(binary["nbin"])
    edges = numin * np.exp(np.arange(EXPECTED_NBIN + 1) * dlog)
    centers = numin * np.exp((np.arange(EXPECTED_NBIN) + 0.5) * dlog)
    band_edges = edges[FIRST_HIGH_BIN:]
    band_centers = centers[FIRST_HIGH_BIN:]
    band_widths = np.diff(band_edges)
    stored = np.asarray(binary["sigma"][EXPECTED_GLOBAL, FIRST_HIGH_BIN:])

    expand = low.load_expand(links, caps)
    ion_data = expand.parse_all_ions()
    for key, data in ion_data.items():
        data["n_kept"] = caps[key]
        data["levels"] = data["osc"].levels[:caps[key]]
    rebuilt, lookup, _ = expand.build_global_levels(ion_data)
    if lookup.get((EXPECTED_Z, EXPECTED_STAGE, EXPECTED_LEVEL + 1)) != EXPECTED_GLOBAL:
        raise SystemExit("Si V ground identity changed")
    row = rebuilt[EXPECTED_GLOBAL]
    if row[:3] != (EXPECTED_Z, EXPECTED_ION, EXPECTED_LEVEL):
        raise SystemExit(f"unexpected global 333 identity: {row[:3]}")

    data = ion_data[(EXPECTED_Z, EXPECTED_STAGE)]
    levs = data["levels"]
    cfg0 = expand._norm_cfg(levs["config"][EXPECTED_LEVEL])
    term0 = expand._term_cfg(levs["config"][EXPECTED_LEVEL])
    zion = float(data["osc"].z_screen)
    threshold_eV = (data["osc"].ionization_eV -
                    float(levs["E_cm"][EXPECTED_LEVEL]) * 1.239841984e-4)
    nu_threshold = threshold_eV * low.EV / low.H
    model = None
    phot_path = Path(data["provenance"]["phot_path"])
    for entry in data["phot"].entries:
        if (expand._norm_cfg(entry.config) != cfg0 and
                expand._term_cfg(entry.config) != term0):
            continue
        nef = expand._cmfgen_nef(EXPECTED_Z, zion, nu_threshold)
        bake = expand._sigma_model(entry.cs_type, entry.energy, entry.sigma_Mb,
                                   nu_threshold, zion=zion, nef=nef)
        if bake is not None:
            model = (entry, bake, low.native_model(entry, nu_threshold, bake))
    if model is None:
        raise SystemExit("linked CMFGEN Si V ground photo model missing")
    entry, bake, native = model

    reconstructed = expand._bin_average_sigma(
        bake[0], bake[1], band_edges, band_widths, bake[2])
    sigma_abs = np.abs(stored - reconstructed)
    common = np.maximum(np.abs(reconstructed), 1.0e-300)
    sigma_rel = np.where((stored == 0.0) & (reconstructed == 0.0), 0.0,
                         sigma_abs / common)

    _, read_edd, rvtj_block = low.load_snapshot_parsers()
    J, nu_cmf, nd, finish = read_edd(str(run / "EDDFACTOR"))
    temperature = rvtj_block((run / "RVTJ").read_text(),
                             "Temperature (10^4K)", nd) * 1.0e4
    if (not np.isfinite(finish) or finish == 0.0 or
            float(nu_cmf[0]) > band_edges[0] or
            float(nu_cmf[-1]) < band_edges[-1]):
        raise SystemExit("CMFGEN snapshot does not cover the upper band")

    Jbar = np.empty((band_centers.size, nd))
    for depth in range(nd):
        Jbar[:, depth] = local_bin_average(
            nu_cmf, J[:, depth], band_edges)
    mask = (nu_cmf > band_edges[0]) & (nu_cmf < band_edges[-1])
    nu_native = np.concatenate(([band_edges[0]], nu_cmf[mask],
                                [band_edges[-1]]))
    J_native = np.empty((nu_native.size, nd))
    J_native[1:-1] = J[mask]
    for depth in range(nd):
        J_native[0, depth] = np.interp(nu_native[0], nu_cmf, J[:, depth])
        J_native[-1, depth] = np.interp(nu_native[-1], nu_cmf, J[:, depth])
    weights = low.trap_weights(nu_native)
    sigma_native = native(nu_native)

    gamma_native = low.FOUR_PI * np.sum(
        weights[:, None] * sigma_native[:, None] * J_native /
        (low.H * nu_native[:, None]), axis=0)
    # Production canonical rate: K=2 J bins, with the stored full-bin-average
    # sigma mass relocated onto the physical threshold sub-interval.
    canonical_edges = np.empty(2 * band_centers.size + 1)
    canonical_edges[0::2] = band_edges
    canonical_edges[1::2] = np.sqrt(band_edges[:-1] * band_edges[1:])
    Jcanonical = np.empty((canonical_edges.size - 1, nd))
    for depth in range(nd):
        Jcanonical[:, depth] = local_bin_average(
            nu_cmf, J[:, depth], canonical_edges)
    gamma_binned = np.zeros(nd)
    for q in range(canonical_edges.size - 1):
        bf_bin = q // 2
        sigma_step = stored[bf_bin]
        blo, bhi = band_edges[bf_bin], band_edges[bf_bin + 1]
        if sigma_step > 0.0 and blo < nu_threshold < bhi:
            sigma_step *= (bhi - blo) / (bhi - nu_threshold)
        lo = max(canonical_edges[q], nu_threshold)
        hi = canonical_edges[q + 1]
        if sigma_step > 0.0 and hi > lo:
            gamma_binned += (low.FOUR_PI * Jcanonical[q] * sigma_step /
                             low.H * math.log(hi / lo))

    x_native = low.H * nu_native[:, None] / (low.KB * temperature[None, :])
    eta_native = np.sum(
        weights[:, None] * sigma_native[:, None] *
        (2.0 * low.H * nu_native[:, None] ** 3 / low.C ** 2) *
        np.exp(-x_native), axis=0)
    partial = ((band_centers < nu_threshold) &
               (band_edges[1:] > nu_threshold))
    nu_eta = np.where(partial,
                      0.5 * (nu_threshold + band_edges[1:]), band_centers)
    x_bin = low.H * nu_eta[:, None] / (low.KB * temperature[None, :])
    eta_binned = np.sum(
        (stored * band_widths)[:, None] *
        (2.0 * low.H * nu_eta[:, None] ** 3 / low.C ** 2) *
        np.exp(-x_bin), axis=0)

    rate_rel = low.relative_error(gamma_binned, gamma_native)
    eta_rel = low.relative_error(eta_binned, eta_native)
    rate_abs_error = np.abs(gamma_binned - gamma_native)
    metrics = {
        "sigma_reconstruction_max_abs_cm2": low.max_finite(sigma_abs),
        "sigma_reconstruction_max_rel": low.max_finite(sigma_rel),
        "photo_rate_depth_max_rel": low.max_finite(rate_rel),
        "photo_rate_depth_max_abs_s1": low.max_finite(rate_abs_error),
        "photo_rate_error_per_epoch":
            low.max_finite(rate_abs_error) * EPOCH_SECONDS,
        "milne_integral_depth_max_rel": low.max_finite(eta_rel),
        "photo_rate_native_min_s1": float(np.min(gamma_native)),
        "photo_rate_native_max_s1": float(np.max(gamma_native)),
        "photo_rate_binned_min_s1": float(np.min(gamma_binned)),
        "photo_rate_binned_max_s1": float(np.max(gamma_binned)),
        "photo_rate_ratio_min": float(np.min(np.where(
            gamma_native > 0.0, gamma_binned / gamma_native, np.nan))),
        "photo_rate_ratio_max": float(np.max(np.where(
            gamma_native > 0.0, gamma_binned / gamma_native, np.nan))),
        "high_band_nonzero_bins": int(np.count_nonzero(stored > 0.0)),
    }
    accuracy_checks = {
        "identity": True,
        "threshold_contained": bool(band_edges[0] < nu_threshold < band_edges[-1]),
        "sigma_reconstruction": metrics["sigma_reconstruction_max_rel"] <= 1.0e-10,
        "photo_rate_accuracy": metrics["photo_rate_depth_max_rel"] <= RATE_LIMIT,
        "milne_integral_closure": metrics["milne_integral_depth_max_rel"] <= ETA_LIMIT,
        "finite_nonnegative": bool(
            np.all(np.isfinite(gamma_native)) and np.all(gamma_native >= 0.0) and
            np.all(np.isfinite(gamma_binned)) and np.all(gamma_binned >= 0.0) and
            np.all(np.isfinite(eta_native)) and np.all(eta_native >= 0.0) and
            np.all(np.isfinite(eta_binned)) and np.all(eta_binned >= 0.0)),
    }
    impact_checks = {
        "photo_rate_effect_negligible":
            metrics["photo_rate_error_per_epoch"] <= RATE_EFFECT_LIMIT,
    }
    accuracy_verdict = "PASS" if all(accuracy_checks.values()) else "FAIL"
    impact_verdict = ("NEGLIGIBLE" if all(impact_checks.values()) else
                      "NOT_NEGLIGIBLE")
    verdict = ("PASS" if accuracy_verdict == "PASS" else
               f"ACCURACY_FAIL_EFFECT_{impact_verdict}")
    manifest = {
        "schema": "lumina-sh-grid-upper-band-closure-v1",
        "verdict": verdict,
        "scope": "same-snapshot discretisation accuracy plus separately reported physical impact; not solver convergence",
        "grid": {"first_bin": FIRST_HIGH_BIN, "n_bins": band_centers.size,
                 "nu_lo_hz": float(band_edges[0]),
                 "nu_hi_hz": float(band_edges[-1]), "dlog_nu": dlog},
        "witness": {"global_level": EXPECTED_GLOBAL, "atomic_number": EXPECTED_Z,
                    "ion_number": EXPECTED_ION, "level_number": EXPECTED_LEVEL,
                    "threshold_eV": threshold_eV,
                    "nu_threshold_hz": nu_threshold,
                    "cs_type": int(entry.cs_type), "sigma_path": str(bake[3])},
        "snapshot": {"run": str(run), "depths": nd,
                     "native_frequency_points": int(nu_native.size),
                     "edd_finish_record": float(finish)},
        "gates": {"photo_rate_depth_max_rel": RATE_LIMIT,
                  "photo_rate_error_per_epoch": RATE_EFFECT_LIMIT,
                  "photo_rate_accuracy_rule": "relative error only",
                  "photo_rate_impact_rule": "separate classification; never substitutes for accuracy",
                  "milne_integral_depth_max_rel": ETA_LIMIT},
        "accuracy_verdict": accuracy_verdict,
        "impact_verdict": impact_verdict,
        "metrics": metrics, "accuracy_checks": accuracy_checks,
        "impact_checks": impact_checks,
        "input_sha256": {str(sigma_path): low.sha256(sigma_path),
                         str(run / "EDDFACTOR"): low.sha256(run / "EDDFACTOR"),
                         str(run / "RVTJ"): low.sha256(run / "RVTJ"),
                         str(phot_path): low.sha256(phot_path),
                         str(links): low.sha256(links)},
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "VERDICT.md").write_text(
        "# SH-GRID 신규 상한 band 동일-snapshot 폐합\n\n"
        f"정확도 판정: **{accuracy_verdict}**\n\n"
        f"현재 snapshot 물리영향: **{impact_verdict}**\n\n"
        f"- Si V ground threshold: `{nu_threshold:.17g} Hz`\n"
        f"- sigma 재구성 max rel: `{metrics['sigma_reconstruction_max_rel']:.6e}`\n"
        f"- depth photo-rate max rel: `{metrics['photo_rate_depth_max_rel']:.6e}`\n"
        f"- photo-rate error per epoch: `{metrics['photo_rate_error_per_epoch']:.6e}`\n"
        f"- depth Milne integral max rel: `{metrics['milne_integral_depth_max_rel']:.6e}`\n"
        "\nphoto-rate 정확도는 상대오차 2%만으로 판정한다. 절대오차의 "
        "epoch 영향은 현재 snapshot의 영향도 분류일 뿐 정확도 PASS를 대신하지 않는다. "
        "이 판정은 고정 EDDFACTOR/RVTJ "
        "snapshot의 격자 적분 폐합이며 "
        "수렴 solution 주장이 아니다.\n"
    )
    print(f"[SH-GRID][UPPER-BAND] global={EXPECTED_GLOBAL} depths={nd} "
          f"native_nu={nu_native.size} cs_type={entry.cs_type}")
    for key, value in metrics.items():
        print(f"[SH-GRID][UPPER-BAND][METRIC] {key}={value}")
    for key, passed in accuracy_checks.items():
        print(f"[SH-GRID][UPPER-BAND][ACCURACY] {key}={'PASS' if passed else 'FAIL'}")
    for key, passed in impact_checks.items():
        print(f"[SH-GRID][UPPER-BAND][IMPACT] {key}={'PASS' if passed else 'FAIL'}")
    print(f"[SH-GRID][UPPER-BAND][{verdict}] manifest={manifest_path}")
    return 0 if accuracy_verdict == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
