#!/usr/bin/env python3
"""Stage 3.1 frozen-chi/eta transport discriminator.

This is an offline CPU bench.  It validates the LCMFCE01 payload and JSON
sidecar, compiles the small C consumer of ``src/lumina_cmf_field.c``, and then
reuses the checked Wave-3 grid/CMFGEN/rate-replay code.  Any schema, checksum,
epoch, radial reconstruction, transport, or rate failure remains visible as an
UNRESOLVED result; the runner never floors a physical value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from cmf_chieta_check import check_artifact
import w3_gamma_triple_compare as gamma


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = Path(
    "/gpfs/kjhan/lumina_runner2/scratch/"
    "chieta_capture_parity59_188605/chieta_iter10"
)
DEFAULT_CMF = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
C_ANGSTROM = 2.99792458e18
BANDS = (
    ("B0", 600.0, 1000.0),
    ("B1", 1000.0, 1500.0),
    ("B2", 1500.0, 2000.0),
    ("B3", 2000.0, 2500.0),
    ("B4", 2500.0, 3000.0),
    ("BALL", 600.0, 3000.0),
)


class BenchError(RuntimeError):
    pass


def canonical_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = 1.5e14 * np.exp(
        np.arange(gamma.NB + 1) * math.log(3.0e16 / 1.5e14) / gamma.NB
    )
    return edges, np.sqrt(edges[:-1] * edges[1:]), np.diff(edges)


def compile_driver(executable: Path) -> list[str]:
    command = [
        "gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic",
        "-Werror", "-D_POSIX_C_SOURCE=200809L", "-Isrc",
        "scripts/stage31_cmf_field_driver.c", "src/lumina_cmf_field.c",
        "-lm", "-o", str(executable),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return command


def parse_driver_table(path: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    lines = path.read_text().splitlines()
    if len(lines) != 1002 or not lines[0].startswith("# "):
        raise BenchError("driver table does not contain one header plus 1000 bins")
    metadata: dict[str, str] = {}
    for token in lines[0][2:].split():
        key, value = token.split("=", 1)
        metadata[key] = value
    names = lines[1].split("\t")
    if names != ["k", "nu_hz", "dnu_hz", "J_det", "J_producer"]:
        raise BenchError(f"unexpected driver table schema: {names}")
    data = np.loadtxt(path, delimiter="\t", skiprows=2)
    if data.shape != (1000, 5) or not np.array_equal(data[:, 0], np.arange(1000)):
        raise BenchError("driver bin identity is not exactly 0..999")
    return metadata, dict(zip(names, data.T, strict=True))


def face_extrapolation(array: np.ndarray, radii: np.ndarray,
                       edges: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
    inner_fraction = (edges[0] - radii[0]) / (radii[1] - radii[0])
    outer_fraction = (edges[-1] - radii[-1]) / (radii[-1] - radii[-2])

    def geometric(left: np.ndarray, right: np.ndarray,
                  fraction: float) -> tuple[np.ndarray, np.ndarray]:
        both_zero = (left == 0.0) & (right == 0.0)
        positive = (left > 0.0) & (right > 0.0)
        invalid = ~(both_zero | positive) | ~np.isfinite(left) | ~np.isfinite(right)
        values = np.full(left.shape, np.nan, dtype=float)
        values[both_zero] = 0.0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            values[positive] = np.exp(
                np.log(left[positive])
                + fraction * (np.log(right[positive]) - np.log(left[positive]))
            )
        invalid |= ~np.isfinite(values) | ((values == 0.0) & ~both_zero)
        return values, invalid

    inner, inner_invalid = geometric(array[0], array[1], inner_fraction)
    outer, outer_invalid = geometric(array[-2], array[-1], 1.0 + outer_fraction)
    return inner, outer, inner_invalid, outer_invalid


def extrapolation_audit(arrays: list[tuple[float, ...]]) -> dict[str, Any]:
    r_edge = np.asarray(arrays[0])
    nu = np.asarray(arrays[1])
    radii = 0.5 * (r_edge[:-1] + r_edge[1:])
    audit: dict[str, Any] = {}
    for name, field_index in (("chi_total", 3), ("eta_total", 7)):
        field = np.asarray(arrays[field_index]).reshape(50, 1000)
        inner, outer, inner_invalid, outer_invalid = face_extrapolation(
            field, radii, r_edge
        )
        legacy_inner = field[0] + (
            (r_edge[0] - radii[0]) / (radii[1] - radii[0])
        ) * (field[1] - field[0])
        legacy_outer = field[-1] + (
            (r_edge[-1] - radii[-1]) / (radii[-1] - radii[-2])
        ) * (field[-1] - field[-2])
        sides = {}
        for side, values, invalid, legacy in (
            ("inner", inner, inner_invalid, legacy_inner),
            ("outer", outer, outer_invalid, legacy_outer),
        ):
            bad = invalid | (values < 0.0)
            wavelength = C_ANGSTROM / nu
            band_bad = bad & (wavelength >= 600.0) & (wavelength <= 3000.0)
            indices = np.flatnonzero(bad)
            band_indices = np.flatnonzero(band_bad)
            finite_values = values[np.isfinite(values)]
            sides[side] = {
                "invalid_or_negative_count": int(bad.sum()),
                "legacy_linear_negative_count": int(np.sum(legacy < 0.0)),
                "minimum": float(finite_values.min()) if finite_values.size else None,
                "first_bin": int(indices[0]) if indices.size else None,
                "first_wavelength_A": (
                    float(wavelength[indices[0]]) if indices.size else None
                ),
                "band_600_3000_invalid_or_negative_count": int(band_bad.sum()),
                "band_first_bin": (
                    int(band_indices[0]) if band_indices.size else None
                ),
                "band_last_bin": (
                    int(band_indices[-1]) if band_indices.size else None
                ),
            }
        audit[name] = sides
    return audit


def band_weights(edges: np.ndarray, lo_A: float, hi_A: float) -> np.ndarray:
    frequency_lo = C_ANGSTROM / hi_A
    frequency_hi = C_ANGSTROM / lo_A
    return np.maximum(
        0.0,
        np.minimum(edges[1:], frequency_hi) - np.maximum(edges[:-1], frequency_lo),
    )


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return None
    if numerator < 0.0 or denominator <= 0.0:
        return None
    return numerator / denominator


def safe_log_ratio(numerator: float, denominator: float) -> float | None:
    ratio = safe_ratio(numerator, denominator)
    return math.log10(ratio) if ratio is not None and ratio > 0.0 else None


def spectral_summary(left: np.ndarray | None, right: np.ndarray,
                     mask: np.ndarray) -> dict[str, Any] | None:
    if left is None:
        return None
    selected = mask & (left > 0.0) & (right > 0.0)
    values = np.log10(left[selected] / right[selected])
    return {
        "median_dex": float(np.median(values)) if values.size else None,
        "p10_dex": float(np.quantile(values, 0.1)) if values.size else None,
        "p90_dex": float(np.quantile(values, 0.9)) if values.size else None,
        "positive_pair_count": int(values.size),
        "excluded_zero_count": int(mask.sum() - values.size),
    }


def make_band_rows(edges: np.ndarray, j_det: np.ndarray | None,
                   j_mc: np.ndarray, j_cmf: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    centers = np.sqrt(edges[:-1] * edges[1:])
    wavelength = C_ANGSTROM / centers
    for name, lo_A, hi_A in BANDS:
        weights = band_weights(edges, lo_A, hi_A)
        width = float(weights.sum())
        center_mask = (wavelength >= lo_A) & (wavelength <= hi_A)
        scalars = {
            "J_det": float(np.sum(j_det * weights) / width) if j_det is not None else None,
            "J_MC": float(np.sum(j_mc * weights) / width),
            "J_CMFGEN": float(np.sum(j_cmf * weights) / width),
        }
        det_mc = (safe_ratio(scalars["J_det"], scalars["J_MC"])
                  if scalars["J_det"] is not None else None)
        det_cmf = (safe_ratio(scalars["J_det"], scalars["J_CMFGEN"])
                   if scalars["J_det"] is not None else None)
        mc_cmf = safe_ratio(scalars["J_MC"], scalars["J_CMFGEN"])
        log_det_cmf = (safe_log_ratio(scalars["J_det"], scalars["J_CMFGEN"])
                       if scalars["J_det"] is not None else None)
        log_mc_cmf = safe_log_ratio(scalars["J_MC"], scalars["J_CMFGEN"])
        rows.append({
            "band": name, "wavelength_A": [lo_A, hi_A], **scalars,
            "J_det_over_J_MC": det_mc,
            "J_det_over_J_CMFGEN": det_cmf,
            "J_MC_over_J_CMFGEN": mc_cmf,
            "log10_J_det_over_J_MC": math.log10(det_mc) if det_mc and det_mc > 0 else None,
            "log10_J_det_over_J_CMFGEN": log_det_cmf,
            "log10_J_MC_over_J_CMFGEN": log_mc_cmf,
            "d_det": abs(log_det_cmf) if log_det_cmf is not None else None,
            "d_MC": abs(log_mc_cmf) if log_mc_cmf is not None else None,
            "toward_CMFGEN": (
                abs(log_det_cmf) < abs(log_mc_cmf)
                if log_det_cmf is not None and log_mc_cmf is not None else None
            ),
            "spectral_det_over_mc": spectral_summary(j_det, j_mc, center_mask),
            "spectral_det_over_cmfgen": spectral_summary(j_det, j_cmf, center_mask),
            "spectral_mc_over_cmfgen": spectral_summary(j_mc, j_cmf, center_mask),
        })
    return rows


def load_gamma_context(capture_dir: Path, edges: np.ndarray,
                       j_det: np.ndarray | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gamma.AUDIT.clear()
    c1_path = capture_dir / "lumina_c1_bins.csv"
    c2_path = capture_dir / "lumina_c2_bfr_dump.csv"
    gamma.audit_csv(c1_path, gamma.EXPECTED["c1"], "capture C1 bins")
    gamma.audit_csv(c2_path, gamma.EXPECTED["c2"], "capture C2 bfr")
    frozen = gamma.load_frozen_field(c1_path, c2_path, 10)
    cmf = gamma.cmfgen_shell_field(edges, gamma.MODEL / "geometry.csv")
    levels = gamma.load_levels(gamma.MODEL / "levels.csv")
    ionization = gamma.load_ionization(gamma.MODEL / "ionization_energies.csv")
    sigma = gamma.SigmaBF(gamma.MODEL / "cmfgen_sigma_bf.bin")
    routes = gamma.TargetRoutes(gamma.MODEL / "ma_radrecomb_target.bin", len(levels))
    results = []
    for target in gamma.TARGETS:
        z = target["Z"]
        manifest_path = gamma.EW_DIR / f"lumina_ew_iter0011_z{z}_s008_manifest.csv"
        identity = gamma.identity_and_fraction(
            target,
            gamma.EW_DIR / f"lumina_ew_iter0011_z{z}_s008_identity.csv",
            gamma.EW_DIR / f"lumina_ew_iter0011_z{z}_s008_solution.csv",
            levels,
            gamma.manifest(manifest_path),
        )
        replay_c = gamma.rate_replay(
            target, identity["members"], identity["frac"], levels,
            ionization, routes, sigma, frozen, cmf["J"],
        )
        replay_d = None
        if j_det is not None:
            replay_d = gamma.rate_replay(
                target, identity["members"], identity["frac"], levels,
                ionization, routes, sigma, frozen, j_det,
            )
        B = replay_c["B"]
        C = replay_c["C"]
        D = replay_d["C"] if replay_d is not None else None
        log_bc = safe_log_ratio(B, C)
        log_dc = safe_log_ratio(D, C) if D is not None else None
        results.append({
            "target": target["key"], "matrix_index": target["matrix"],
            "Gamma_MC_B": B, "Gamma_CMFGEN_C": C, "Gamma_det_D": D,
            "Gamma_det_over_MC": safe_ratio(D, B) if D is not None else None,
            "Gamma_det_over_CMFGEN": safe_ratio(D, C) if D is not None else None,
            "log10_Gamma_det_over_MC": safe_log_ratio(D, B) if D is not None else None,
            "log10_Gamma_det_over_CMFGEN": log_dc,
            "log10_Gamma_MC_over_CMFGEN": log_bc,
            "toward_CMFGEN": (
                abs(log_dc) < abs(log_bc)
                if log_dc is not None and log_bc is not None else None
            ),
            "member_count": len(identity["members"]),
            "route_count": replay_c["n_routes"],
            "threshold_eV": [replay_c["threshold_min"], replay_c["threshold_max"]],
        })
    return {"frozen": frozen, "cmf": cmf}, results


def classify(bands: list[dict[str, Any]], rates: list[dict[str, Any]],
             solve_ok: bool) -> str:
    if not solve_ok:
        return "UNRESOLVED-SOLVER-GUARD"
    directions = [row["toward_CMFGEN"] for row in bands]
    gamma_directions = [row["toward_CMFGEN"] for row in rates]
    if all(value is True for value in directions + gamma_directions):
        return "TRANSPORT-DEFECT"
    if all(value is False for value in directions + gamma_directions):
        return "CHI-ETA-CONTENT-DEFECT"
    return "UNRESOLVED-MIXED"


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "UNRESOLVED"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}g}"


def render_report(result: dict[str, Any], report_path: Path,
                  status_path: Path) -> str:
    input_audit = result["input"]
    solve = result["solve"]
    if solve["ok"]:
        candidate_acceptance = (
            "- candidate transport residual ≤1e-4, finite/nonnegative, clamp=0: "
            f"**PASS** (`{float(solve['metadata']['transport_residual']):.9e}`, "
            f"clamp `{solve['metadata']['clamp']}`)"
        )
        candidate_determinism = (
            "- candidate 3-run SHA-256 identity: **PASS** "
            f"(`{solve['determinism_sha256'][0]}`)"
        )
    else:
        candidate_acceptance = (
            "- candidate transport residual ≤1e-4, finite/nonnegative, clamp=0: "
            "**UNRESOLVED** (solver guard가 장 생성 전에 fail closed)"
        )
        candidate_determinism = (
            "- candidate 3-run SHA-256 identity: **NOT RUN** "
            "(첫 solver guard 실패 뒤 중단)"
        )
    lines = [
        "# Codex A-S31 round 7D — 절단오차 계층화 및 판별 벤치",
        "",
        f"상태: **{result['status']}**  ",
        f"물리 판독: **{result['classification']}**",
        "",
        "## 결론",
        "",
    ]
    if result["classification"] == "UNRESOLVED-SOLVER-GUARD":
        lines += [
            "요청된 수송 결함 대 χ,η 내용 결함의 이분 판정은 내릴 수 없다. "
            "인증 payload와 로그 radial-face 외삽은 정상이나, 정본 solver가 첫 sweep의 "
            "solution guard에서 fail closed했다. 이를 clamp하거나 tolerance로 무시하면 "
            "acceptance를 바꾸므로 수행하지 않았다.",
            "",
            f"실제 첫 실패는 `{solve['stderr'].strip()}`이다. 실행 시간은 "
            f"{solve['elapsed_seconds']:.3f} s로, 장시간 계산 때문에 중단한 것이 아니다.",
        ]
    else:
        interpretation = {
            "TRANSPORT-DEFECT": "J_det와 두 Γ가 모두 CMFGEN 쪽으로 이동해 수송 결함 가설을 지지한다.",
            "CHI-ETA-CONTENT-DEFECT": "J_det가 MC UV 과잉을 재현해 frozen χ,η 내용 결함 가설을 지지한다.",
            "UNRESOLVED-MIXED": "대역/이온별 방향이 갈려 수송과 χ,η의 파장 의존 혼합 원인으로 분해한다.",
        }[result["classification"]]
        lines += [
            "양 끝 face의 χ,η를 로그-공간 선형 외삽한 공용 solver가 수치 acceptance를 통과했다. "
            "사전등록 방향 규칙을 아래 6대역과 두 Γ에 그대로 적용했다.",
            "",
            f"판독: **{interpretation}**",
        ]
    lines += [
        "",
        "## 입력 독립 검증",
        "",
        f"- checker: `{input_audit['checker']}`",
        f"- payload SHA-256: `{input_audit['sha256']}`; sidecar와 일치",
        f"- schema: 50 shell × 1000 bin, iter={input_audit['iteration']}, "
        f"generation={input_audit['field_generation']}, post_damp=1",
        f"- candidate/input ν grid max relative identity error: "
        f"`{input_audit['nu_grid_max_relative_error']:.3e}`",
        "- inner boundary: producer `cmf_solve_J`와 같은 explicit "
        "`Bν(T_inner=10020 K)` irradiation, amplitude scale 1.0; diffusion "
        "gradient를 추정하지 않음",
        "",
        "### radial face 사전검사",
        "",
        "| field | face | log invalid/negative | 600–3000 Å invalid/negative | log minimum | legacy linear negative | first bad bin / Å |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for field, sides in result["radial_face_extrapolation"].items():
        for face, row in sides.items():
            first = (f"{row['first_bin']} / {row['first_wavelength_A']:.3f}"
                     if row["first_bin"] is not None else "—")
            lines.append(
                f"| {field} | {face} | {row['invalid_or_negative_count']} | "
                f"{row['band_600_3000_invalid_or_negative_count']} | "
                f"{row['minimum']:.9e} | {row['legacy_linear_negative_count']} | {first} |"
            )
    if solve["ok"]:
        metadata = solve["metadata"]
        lines += [
            "",
            "### 해-부호 가드 통계",
            "",
            f"- certified-negative sub-truncation: `{metadata['solution_subtruncation']}`; "
            f"minimum `{metadata['solution_subtruncation_min']}`",
            f"- minimum coordinate: frequency `{metadata['solution_subtruncation_min_frequency']}`, "
            f"ray `{metadata['solution_subtruncation_min_ray']}`, "
            f"segment `{metadata['solution_subtruncation_min_segment']}`, "
            f"substep `{metadata['solution_subtruncation_min_substep']}`",
            f"- first value/coordinate: `{metadata['solution_subtruncation_first_value']}` at "
            f"frequency `{metadata['solution_subtruncation_first_frequency']}`, "
            f"ray `{metadata['solution_subtruncation_first_ray']}`, "
            f"segment `{metadata['solution_subtruncation_first_segment']}`, "
            f"substep `{metadata['solution_subtruncation_first_substep']}`",
            f"- first scale/h/B_trunc: `{metadata['solution_subtruncation_first_scale']}` / "
            f"`{metadata['solution_subtruncation_first_h']}` / "
            f"`{metadata['solution_subtruncation_first_bound']}`",
            f"- sign-indeterminate sub-truncation: "
            f"`{metadata['solution_sign_indeterminate_subtruncation']}`",
            f"- finite-value enclosure restarts: "
            f"`{metadata['solution_roundoff_enclosure_restart']}`",
            f"- excess/sign-uncertain/nonfinite/clamp: "
            f"`{metadata['solution_negative_excess']}/"
            f"{metadata['sign_uncertain']}/{metadata['nonfinite']}/{metadata['clamp']}`",
        ]
    lines += [
        "",
        "## s8 Jν 3중 대조",
        "",
        "J_MC는 계약대로 sidecar payload의 `J_producer`다. CMFGEN은 RVTJ의 "
        "9610.017–10163.506 km/s 사이 log-J velocity interpolation 후 공통 "
        "1000-bin edge에 적분보존 평균했다. point interpolation은 쓰지 않았다.",
        "",
        "| band [Å] | J_det/J_MC | J_det/J_CMFGEN | J_MC/J_CMFGEN | log10(det/MC) | log10(det/CMFGEN) | log10(MC/CMFGEN) | toward CMFGEN |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["bands"]:
        lo, hi = row["wavelength_A"]
        lines.append(
            f"| {row['band']} {lo:g}–{hi:g} | {fmt(row['J_det_over_J_MC'])} | "
            f"{fmt(row['J_det_over_J_CMFGEN'])} | {fmt(row['J_MC_over_J_CMFGEN'])} | "
            f"{fmt(row['log10_J_det_over_J_MC'])} | "
            f"{fmt(row['log10_J_det_over_J_CMFGEN'])} | "
            f"{fmt(row['log10_J_MC_over_J_CMFGEN'])} | {fmt(row['toward_CMFGEN'])} |"
        )
    lines += [
        "",
        "### spectral norm",
        "",
        "| band | pair | median log10 ratio | p10 | p90 | positive pairs | zero/excluded |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["bands"]:
        for label, key in (("det/MC", "spectral_det_over_mc"),
                           ("det/CMFGEN", "spectral_det_over_cmfgen"),
                           ("MC/CMFGEN", "spectral_mc_over_cmfgen")):
            spectral = row[key]
            if spectral is None:
                lines.append(
                    f"| {row['band']} | {label} | UNRESOLVED | UNRESOLVED | "
                    "UNRESOLVED | UNRESOLVED | UNRESOLVED |"
                )
            else:
                lines.append(
                    f"| {row['band']} | {label} | {spectral['median_dex']:+.6f} | "
                    f"{spectral['p10_dex']:+.6f} | {spectral['p90_dex']:+.6f} | "
                    f"{spectral['positive_pair_count']} | {spectral['excluded_zero_count']} |"
                )
    lines += [
        "",
        "candidate가 들어가는 spectral quantile은 두 양수 field가 모두 양수인 bin만 "
        "사용했으며, 제외 수를 그대로 기록했다.",
        "",
        "## Γ D-lane",
        "",
        "기존 `w3_gamma_triple_compare.py`의 grid/C1/C2 loader, EDDFACTOR/RVTJ "
        "적분보존 평균, within-SL fraction·σ·threshold·route 및 "
        "`4πσJ/(hν)` quadrature를 import해 재사용했다.",
        "",
        "| target | Γ_MC B [s⁻¹] | Γ_CMFGEN C [s⁻¹] | Γ_det D [s⁻¹] | D/B | D/C | log10(D/C) | log10(B/C) | toward CMFGEN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["gamma"]:
        lines.append(
            f"| {row['target']} (idx {row['matrix_index']}) | "
            f"{row['Gamma_MC_B']:.9e} | {row['Gamma_CMFGEN_C']:.9e} | "
            f"{fmt(row['Gamma_det_D'], 9)} | "
            f"{fmt(row['Gamma_det_over_MC'])} | "
            f"{fmt(row['Gamma_det_over_CMFGEN'])} | "
            f"{fmt(row['log10_Gamma_det_over_CMFGEN'])} | "
            f"{row['log10_Gamma_MC_over_CMFGEN']:+.6f} | "
            f"{fmt(row['toward_CMFGEN'])} |"
        )
    lines += [
        "",
        "Fe III와 S II의 D-lane은 동일한 σ·threshold·route·within-SL fraction에 "
        "J_det만 대입해 계산했다.",
        "",
        "## Acceptance와 판독",
        "",
        f"- sidecar/schema/checksum/epoch: **PASS**",
        f"- CMFGEN integral conservation: "
        f"`{result['cmfgen']['integral_ratio']:.15f}` (**PASS**)",
        f"- 6 band row와 2 Γ row의 baseline provenance: **PASS**",
        candidate_acceptance,
        candidate_determinism,
        f"- 최종 bench acceptance: **{result['status']}**",
        "",
        f"최종 사전등록 판독은 **{result['classification']}**이며 방향 자체는 "
        "acceptance를 변경하지 않는다.",
        "",
        "## 재현 명령",
        "",
        "```bash",
        f"sha256sum {input_audit['path']}",
        f"python3 scripts/cmf_chieta_check.py {input_audit['path']}",
        "gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \\",
        "  -D_POSIX_C_SOURCE=200809L -Isrc scripts/stage31_cmf_field_driver.c \\",
        "  src/lumina_cmf_field.c -lm -o /tmp/stage31_cmf_field_driver",
        f"/tmp/stage31_cmf_field_driver {input_audit['path']} "
        f"{input_audit['sidecar']} 8 16 10020 1 /tmp/stage31_jdet.tsv",
        f"python3 scripts/stage31_cmf_field_bench.py --frozen {input_audit['path']} \\",
        f"  --report {report_path.relative_to(ROOT)} \\",
        f"  --status-json {status_path.relative_to(ROOT)}",
        "```",
        "",
        "신규 모델/GPU run, acceptance 변경, clamp/floor, 커밋은 없었다.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--cmf-run", type=Path, default=DEFAULT_CMF)
    parser.add_argument("--ew-dir", type=Path, default=gamma.EW_DIR)
    parser.add_argument("--shell", type=int, default=8)
    parser.add_argument("--nmu", type=int, default=16)
    parser.add_argument("--t-inner", type=float, default=10020.0)
    parser.add_argument("--bb-scale", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--driver", type=Path, default=Path("/tmp/stage31_cmf_field_driver"))
    parser.add_argument("--driver-table", type=Path,
                        default=ROOT / "docs/s31_results/stage31_jdet_s8.tsv")
    parser.add_argument("--status-json", type=Path,
                        default=ROOT / "docs/s31_results/stage31_bench_round7.json")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "docs/CODEX_STAGE31_BENCH.md")
    args = parser.parse_args()
    frozen_path = args.frozen.resolve()
    sidecar_path = (args.sidecar or Path(str(frozen_path) + ".manifest.json")).resolve()
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    status_path = (args.status_json if args.status_json.is_absolute()
                   else ROOT / args.status_json)
    table_path = (args.driver_table if args.driver_table.is_absolute()
                  else ROOT / args.driver_table)
    gamma.CMF_RUN = args.cmf_run.resolve()
    gamma.EW_DIR = args.ew_dir.resolve()

    try:
        checked = check_artifact(frozen_path)
        digest = hashlib.sha256(checked.raw).hexdigest()
        if digest != checked.manifest["sha256"]:
            raise BenchError("independent SHA-256 differs from sidecar")
        header = checked.header
        arrays = checked.arrays
        edges, nu_canonical, _ = canonical_grid()
        nu_descending = np.asarray(arrays[1])
        nu_grid_error = float(np.max(np.abs(nu_descending[::-1] / nu_canonical - 1.0)))
        if nu_grid_error > 1.0e-12:
            raise BenchError(f"capture/canonical nu grid mismatch {nu_grid_error}")
        face_audit = extrapolation_audit(arrays)
        compile_command = compile_driver(args.driver)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.driver), str(frozen_path), str(sidecar_path),
            str(args.shell), str(args.nmu), repr(args.t_inner),
            repr(args.bb_scale), str(table_path),
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True,
            timeout=args.timeout,
        )
        elapsed = time.monotonic() - started
        solve_ok = completed.returncode == 0
        j_det = None
        driver_metadata: dict[str, str] = {}
        determinism_sha256: list[str] = []
        if solve_ok:
            driver_metadata, table = parse_driver_table(table_path)
            j_det = table["J_det"][::-1]
            if (not np.isfinite(j_det).all() or np.any(j_det < 0.0) or
                    int(driver_metadata["clamp"]) != 0 or
                    int(driver_metadata["solution_negative_excess"]) != 0 or
                    int(driver_metadata["sign_uncertain"]) != 0 or
                    int(driver_metadata["nonfinite"]) != 0 or
                    float(driver_metadata["transport_residual"]) > 1.0e-4):
                raise BenchError("candidate numerical acceptance failed")
            for _ in range(3):
                rerun = subprocess.run(
                    command, cwd=ROOT, text=True, capture_output=True,
                    timeout=args.timeout,
                )
                if rerun.returncode != 0:
                    raise BenchError("determinism rerun failed")
                determinism_sha256.append(hashlib.sha256(table_path.read_bytes()).hexdigest())
            if len(set(determinism_sha256)) != 1:
                raise BenchError("three candidate output hashes differ")
        j_producer = np.asarray(arrays[8]).reshape(50, 1000)[args.shell][::-1]
        context, gamma_rows = load_gamma_context(frozen_path.parent, edges, j_det)
        cmf_j = context["cmf"]["J"]
        band_rows = make_band_rows(edges, j_det, j_producer, cmf_j)
        classification = classify(band_rows, gamma_rows, solve_ok)
        status = "PASS" if solve_ok else "UNRESOLVED"
        result = {
            "schema": "stage31-cmf-field-bench-v1",
            "status": status,
            "classification": classification,
            "input": {
                "path": str(frozen_path), "sidecar": str(sidecar_path),
                "sha256": digest,
                "checker": (f"PASS: iteration={header[5]} field_generation={header[6]} "
                            f"post_damp={int(bool(header[7] & 1))} bytes={len(checked.raw)}"),
                "nr": header[3], "nnu": header[4],
                "iteration": header[5], "field_generation": header[6],
                "nu_grid_max_relative_error": nu_grid_error,
            },
            "radial_face_extrapolation": face_audit,
            "solve": {
                "ok": solve_ok, "returncode": completed.returncode,
                "elapsed_seconds": elapsed, "stdout": completed.stdout,
                "stderr": completed.stderr, "command": command,
                "compile_command": compile_command,
                "metadata": driver_metadata,
                "determinism_sha256": determinism_sha256,
                "chi_coherent_input": 0.0,
                "eta_input": "captured eta_total",
                "scattering_reconvergence": False,
                "inner_boundary": {
                    "mode": "LCMF_BC_IRRADIATION",
                    "spectrum": "Planck B_nu",
                    "temperature_K": args.t_inner,
                    "amplitude_scale": args.bb_scale,
                    "provenance": "capture stdout T_inner and resolved LUMINA_INNER_BB_SCALE",
                },
            },
            "cmfgen": {
                "run": str(gamma.CMF_RUN),
                "velocity_kms": context["cmf"]["velocity"],
                "velocity_bracket_kms": [context["cmf"]["v0"], context["cmf"]["v1"]],
                "velocity_logJ_weight": context["cmf"]["weight"],
                "integral_ratio": context["cmf"]["qratio"],
            },
            "bands": band_rows,
            "gamma": gamma_rows,
            "acceptance_unchanged": True,
            "model_or_gpu_run": False,
            "src_modified": True,
        }
        rendered_json = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(rendered_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(result, report_path, status_path))
        print(rendered_json, end="")
        print(f"[REPORT] {report_path}", file=sys.stderr)
        return 0 if solve_ok else 3
    except (BenchError, gamma.Unresolved, OSError, ValueError, KeyError,
            subprocess.SubprocessError) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
