#!/usr/bin/env python3
"""Fail-closed DET-SPRIM ledger reconstruction and Stage-1 L6 verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


H_PLANCK = 6.62607015e-27
C_LIGHT = 2.99792458e10
K_BOLTZMANN = 1.380649e-16
FOUR_PI = 12.56637061435917295385057353311801153679
REQUESTED_TEMPERATURE_K = 10020.0
RECONSTRUCTION_TOLERANCE = 1.0e-12
ROW_PREFIX = "[A2-10][LINE-SATURATION-ROW]"
SUMMARY_PREFIX = "[A2-10][LINE-SATURATION-SUMMARY]"
R7_PREFIX = "[R7][PHASE] event=R7_MATERIAL_PHASE_COMMITTED"
KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


class AnalysisError(RuntimeError):
    pass


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fields(line: str) -> dict[str, str]:
    return dict(KEY_VALUE.findall(line))


def integer(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError) as exc:
        raise AnalysisError(f"INVALID_INTEGER_{key}") from exc


def finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as exc:
        raise AnalysisError(f"UNAVAILABLE_{key}") from exc
    if not math.isfinite(value):
        raise AnalysisError(f"NONFINITE_{key}")
    return value


def exponx(tau: float) -> tuple[float, float]:
    """Independent transcription of line_net_rate.c:137-167."""
    if not math.isfinite(tau):
        raise AnalysisError("EXPONX_NONFINITE_TAU")
    if abs(tau) < 1.0e-3:
        companion = 0.5 - tau / 6.0 * (1.0 - tau / 4.0)
        beta = 1.0 - tau * companion
    elif tau < 40.0:
        beta = (1.0 - math.exp(-tau)) / tau
        companion = (1.0 - beta) / tau
    else:
        beta = 1.0 / tau
        companion = (1.0 - beta) / tau
    if (not math.isfinite(beta) or beta <= 0.0 or
            not math.isfinite(companion) or companion <= 0.0):
        raise AnalysisError("EXPONX_DOMAIN")
    return beta, companion


def planck_nu(nu: float, temperature: float) -> float:
    if (not math.isfinite(nu) or nu <= 0.0 or
            not math.isfinite(temperature) or temperature <= 0.0):
        raise AnalysisError("PLANCK_DOMAIN")
    exponent = H_PLANCK * nu / (K_BOLTZMANN * temperature)
    try:
        denominator = math.expm1(exponent)
    except OverflowError as exc:
        raise AnalysisError("PLANCK_EXPONENT_OVERFLOW") from exc
    value = 2.0 * H_PLANCK * nu ** 3 / (C_LIGHT ** 2 * denominator)
    if not math.isfinite(value) or value <= 0.0:
        raise AnalysisError("PLANCK_NONFINITE")
    return value


def relative_deviation(observed: float, reconstructed: float) -> float:
    if observed == reconstructed:
        return 0.0
    denominator = max(abs(observed), abs(reconstructed))
    if denominator == 0.0:
        return math.inf
    return abs(observed - reconstructed) / denominator


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise AnalysisError("EMPTY_DISTRIBUTION")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    probabilities = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50,
                     0.75, 0.90, 0.95, 0.99, 1.0)
    quantiles = {
        f"q{int(probability * 100):02d}": percentile(ordered, probability)
        for probability in probabilities
    }
    return {
        "count": len(ordered),
        "quantiles": quantiles,
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def bracket_rows(lines: list[str]) -> dict[int, list[dict[str, str]]]:
    pending: list[dict[str, str]] = []
    summaries = 0
    iterations: dict[int, list[dict[str, str]]] = {}
    total_rows = 0
    for line in lines:
        if line.startswith(ROW_PREFIX):
            pending.append(fields(line))
            total_rows += 1
        elif line.startswith(SUMMARY_PREFIX):
            summaries += 1
        elif line.startswith(R7_PREFIX):
            marker = fields(line)
            iteration = integer(marker, "iter")
            if marker.get("lane") != "DET" or marker.get("phase") != "A2-10":
                raise AnalysisError("INVALID_R7_MARKER")
            expected_transition = f"{iteration + 1}->{iteration + 2}"
            if marker.get("te_generation") != expected_transition:
                raise AnalysisError("INVALID_R7_GENERATION")
            if pending:
                if summaries != 1 or iteration in iterations:
                    raise AnalysisError("ITER_ATTRIBUTION_BLOCKED")
                iterations[iteration] = pending
                pending = []
                summaries = 0
    if total_rows == 0:
        raise AnalysisError("NO_ROWS")
    if pending or summaries:
        raise AnalysisError("ITER1_ATTRIBUTION_BLOCKED")
    if 0 not in iterations:
        raise AnalysisError("ITER0_ATTRIBUTION_BLOCKED")
    if 1 not in iterations:
        raise AnalysisError("ITER1_ATTRIBUTION_BLOCKED")
    if set(iterations) != {0, 1}:
        raise AnalysisError("UNEXPECTED_ITERATION_BLOCK")
    return iterations


def complete_bracket_rows(lines: list[str]) -> dict[int, list[dict[str, str]]]:
    """Return only marker-closed blocks for named D1/D2 termination reports."""
    pending: list[dict[str, str]] = []
    summaries = 0
    iterations: dict[int, list[dict[str, str]]] = {}
    for line in lines:
        if line.startswith(ROW_PREFIX):
            pending.append(fields(line))
        elif line.startswith(SUMMARY_PREFIX):
            summaries += 1
        elif line.startswith(R7_PREFIX):
            marker = fields(line)
            iteration = integer(marker, "iter")
            if pending and summaries == 1 and iteration not in iterations:
                iterations[iteration] = pending
            pending = []
            summaries = 0
    return iterations


def analyze_iteration(
    rows: list[dict[str, str]], time_explosion_s: float,
    temperature_K: float,
) -> dict[str, Any]:
    census = {
        "TOTAL": len(rows),
        "FINITE_POSITIVE_CHI": 0,
        "NEGATIVE_CHI": 0,
        "INVERSION_BOUNDARY": 0,
        "EXACT_ZERO": 0,
        "UNAVAILABLE": 0,
        "JBAR_ZERO": 0,
        "SUPERTHERMAL_GT10": 0,
    }
    ratios: list[float] = []
    maximum_errors = {"continuum": 0.0, "local": 0.0, "jbar": 0.0}
    for row in rows:
        required_flags = (
            integer(row, "producer_terms_defined"),
            integer(row, "producer_raw_defined"),
            integer(row, "independent_fields_defined"),
        )
        if required_flags != (1, 1, 1):
            census["UNAVAILABLE"] += 1
            continue
        eta = finite(row, "producer_eta")
        tau_eff = finite(row, "producer_tau_eff")
        nu = finite(row, "nu")
        j_cont = finite(row, "J_cont")
        continuum_observed = finite(row, "producer_continuum_term")
        local_observed = finite(row, "producer_local_emission_term")
        jbar_observed = finite(row, "Jbar")
        srce_chk = integer(row, "producer_srce_chk")
        exact_zero = integer(row, "producer_exact_zero")
        if eta < 0.0 or tau_eff < -0.5 or srce_chk not in (0, 1) or exact_zero not in (0, 1):
            raise AnalysisError("INVALID_RAW_PRODUCER_FIELD")
        beta, companion = exponx(tau_eff)
        continuum_reconstructed = beta * j_cont
        local_reconstructed = (
            eta * (C_LIGHT * time_explosion_s / nu) * companion
        )
        jbar_reconstructed = continuum_reconstructed + local_reconstructed
        errors = {
            "continuum": relative_deviation(
                continuum_observed, continuum_reconstructed),
            "local": relative_deviation(local_observed, local_reconstructed),
            "jbar": relative_deviation(jbar_observed, jbar_reconstructed),
        }
        for name, error in errors.items():
            maximum_errors[name] = max(maximum_errors[name], error)
            if not math.isfinite(error) or error > RECONSTRUCTION_TOLERANCE:
                raise AnalysisError(
                    f"G4B_RECONSTRUCTION_FAIL_{name.upper()}_LINE_{row.get('line', 'UNKNOWN')}"
                )
        if jbar_observed == 0.0:
            census["JBAR_ZERO"] += 1

        chi_effective = tau_eff * nu / (C_LIGHT * time_explosion_s)
        if chi_effective == 0.0:
            if eta > 0.0:
                census["INVERSION_BOUNDARY"] += 1
            else:
                census["EXACT_ZERO"] += 1
            continue
        source = eta / chi_effective
        ratio = source / planck_nu(nu, temperature_K)
        if not math.isfinite(ratio):
            raise AnalysisError("NONFINITE_SPROD_OVER_B")
        ratios.append(ratio)
        if chi_effective < 0.0:
            census["NEGATIVE_CHI"] += 1
        else:
            census["FINITE_POSITIVE_CHI"] += 1
        if ratio > 10.0:
            census["SUPERTHERMAL_GT10"] += 1

    if census["UNAVAILABLE"]:
        raise AnalysisError("UNAVAILABLE_ROWS")
    if not ratios:
        raise AnalysisError("NO_NONZERO_CHI_ROWS")
    if sum(census[name] for name in (
        "FINITE_POSITIVE_CHI", "NEGATIVE_CHI", "INVERSION_BOUNDARY",
        "EXACT_ZERO", "UNAVAILABLE")) != census["TOTAL"]:
        raise AnalysisError("CENSUS_NOT_EXHAUSTIVE")
    return {
        "rows": len(rows),
        "census": census,
        "sprod_over_b": distribution(ratios),
        "ratios": ratios,
        "g4b_max_relative_deviation": maximum_errors,
    }


def verdict_for(iteration_one: dict[str, Any]) -> tuple[str, dict[str, float]]:
    ratios = iteration_one["ratios"]
    count = len(ratios)
    f_super = sum(value > 10.0 for value in ratios) / count
    f_depart_1pct = sum(abs(value - 1.0) > 0.01 for value in ratios) / count
    f_lte_1e3 = sum(abs(value - 1.0) <= 1.0e-3 for value in ratios) / count
    q50 = iteration_one["sprod_over_b"]["quantiles"]["q50"]
    if f_super >= 0.10:
        verdict = "A_PRIME"
    elif f_depart_1pct >= 0.10 and 0.5 <= q50 < 1.0:
        verdict = "A"
    elif f_lte_1e3 >= 0.99:
        verdict = "B"
    else:
        verdict = "C"
    return verdict, {
        "f_super": f_super,
        "f_depart_1pct": f_depart_1pct,
        "f_lte_1e3": f_lte_1e3,
    }


def analyze_text(
    text: str, time_explosion_s: float,
    temperature_K: float = REQUESTED_TEMPERATURE_K,
) -> dict[str, Any]:
    if not math.isfinite(time_explosion_s) or time_explosion_s <= 0.0:
        raise AnalysisError("INVALID_TIME_EXPLOSION")
    lines = text.splitlines()
    d2 = "INDEPENDENT_SPROBE_UNDEFINED" in text
    named_block = next((
        fields(line).get("reason", "NAMED_FATAL")
        for line in lines
        if ("[BLOCKED]" in line or "-BLOCKED]" in line or "[FATAL]" in line)
    ), None)
    if d2 or named_block is not None:
        complete = complete_bracket_rows(lines)
        public_iterations: dict[str, Any] = {}
        for iteration, rows in sorted(complete.items()):
            result = analyze_iteration(rows, time_explosion_s, temperature_K)
            public_iterations[str(iteration)] = {
                key: value for key, value in result.items() if key != "ratios"
            }
        return {
            "schema": "DET_STAGE12_L6_VERDICT_V1",
            "status": "PARTIAL",
            "verdict": "D2" if d2 else "D1",
            "blocking_reason": (
                "INDEPENDENT_SPROBE_UNDEFINED" if d2 else named_block
            ),
            "f_super": None,
            "fractions": None,
            "iter1_distribution": None,
            "temperature_K": temperature_K,
            "time_explosion_s": time_explosion_s,
            "iterations": public_iterations,
            "physical_values_modified": False,
        }
    iterations = bracket_rows(lines)
    analyzed = {
        iteration: analyze_iteration(rows, time_explosion_s, temperature_K)
        for iteration, rows in sorted(iterations.items())
    }
    iter0_median = analyzed[0]["sprod_over_b"]["quantiles"]["q50"]
    if not 0.999 <= iter0_median <= 1.001:
        raise AnalysisError("G4_ITER0_ANCHOR_FAIL")
    verdict, fractions = verdict_for(analyzed[1])
    public_iterations: dict[str, Any] = {}
    for iteration, result in analyzed.items():
        public_iterations[str(iteration)] = {
            key: value for key, value in result.items() if key != "ratios"
        }
    return {
        "schema": "DET_STAGE12_L6_VERDICT_V1",
        "status": "PASS",
        "verdict": verdict,
        "f_super": fractions["f_super"],
        "fractions": fractions,
        "iter0_anchor_median": iter0_median,
        "temperature_K": temperature_K,
        "time_explosion_s": time_explosion_s,
        "constants": {
            "h_planck": H_PLANCK,
            "c_light": C_LIGHT,
            "k_boltzmann": K_BOLTZMANN,
            "four_pi": FOUR_PI,
            "exponx_source": "src/line_net_rate.c:137-167 independent transcription",
        },
        "iterations": public_iterations,
        "physical_values_modified": False,
    }


def synthetic_row(
    *, line: int, ratio: float = 1.0, tau: float = 1.0,
    time_explosion_s: float = 1.0e6, perturb_tau: float = 0.0,
    inversion: bool = False,
) -> str:
    nu = 5.0e14
    b_nu = planck_nu(nu, REQUESTED_TEMPERATURE_K)
    if inversion:
        tau = 0.0
        eta = b_nu * nu / (C_LIGHT * time_explosion_s)
    else:
        chi = tau * nu / (C_LIGHT * time_explosion_s)
        eta = ratio * b_nu * chi
    beta, companion = exponx(tau)
    j_cont = b_nu
    continuum = beta * j_cont
    local = eta * (C_LIGHT * time_explosion_s / nu) * companion
    jbar = continuum + local
    return (
        f"{ROW_PREFIX} phase=REQUESTED_TE shell=0 rank=1 line={line} "
        f"nu={nu:.17g} Jbar={jbar:.17g} J_cont={j_cont:.17g} "
        "independent_fields_defined=1 "
        f"producer_continuum_term={continuum:.17g} "
        f"producer_local_emission_term={local:.17g} producer_terms_defined=1 "
        f"producer_eta={eta:.17g} producer_tau_eff={tau + perturb_tau:.17g} "
        "producer_srce_chk=0 producer_exact_zero=0 producer_raw_defined=1"
    )


def synthetic_log(
    iter1_ratio: float = 0.8, *, perturb_tau: bool = False,
    delete_iter1_marker: bool = False, add_inversion: bool = False,
) -> str:
    lines: list[str] = []
    next_line = 1
    for iteration, ratio in ((0, 1.0), (1, iter1_ratio)):
        for _ in range(20):
            lines.append(synthetic_row(
                line=next_line, ratio=ratio,
                perturb_tau=1.0e-9 if perturb_tau and iteration == 1 and next_line == 21 else 0.0,
            ))
            next_line += 1
        if add_inversion and iteration == 1:
            lines.append(synthetic_row(line=next_line, inversion=True))
            next_line += 1
        lines.append(
            f"{SUMMARY_PREFIX} phase=REQUESTED_TE shell=0 complete=1"
        )
        if not (delete_iter1_marker and iteration == 1):
            lines.append(
                f"{R7_PREFIX} lane=DET iter={iteration} phase=A2-10 "
                f"te_generation={iteration + 1}->{iteration + 2}"
            )
    return "\n".join(lines) + "\n"


def selftest() -> int:
    forged = analyze_text(synthetic_log(iter1_ratio=1.0), 1.0e6)
    if forged["verdict"] != "B":
        raise AnalysisError("NC-A1_DID_NOT_SELECT_B")
    print("NC-A1 inject=FORGED_ITER1_LTE status=FAIL reason=BRANCH_B verdict=B")
    restored = analyze_text(synthetic_log(iter1_ratio=0.8), 1.0e6)
    if restored["verdict"] != "A":
        raise AnalysisError("NC-A1_REMOVAL_DID_NOT_SELECT_A")
    print("NC-A1 remove=NLTE_ITER1 status=PASS verdict=A")

    try:
        analyze_text(synthetic_log(perturb_tau=True), 1.0e6)
    except AnalysisError as exc:
        if not str(exc).startswith("G4B_RECONSTRUCTION_FAIL"):
            raise
        print(f"NC-A2 inject=TAU_EFF_PLUS_1E-9 status=FAIL reason={exc}")
    else:
        raise AnalysisError("NC-A2_PERTURBATION_ACCEPTED")
    analyze_text(synthetic_log(), 1.0e6)
    print("NC-A2 remove=TAU_EFF_EXACT status=PASS")

    try:
        analyze_text(synthetic_log(delete_iter1_marker=True), 1.0e6)
    except AnalysisError as exc:
        if str(exc) != "ITER1_ATTRIBUTION_BLOCKED":
            raise
        print(f"NC-A3 inject=DELETE_ITER1_R7 status=FAIL reason={exc}")
    else:
        raise AnalysisError("NC-A3_MISSING_MARKER_ACCEPTED")
    analyze_text(synthetic_log(), 1.0e6)
    print("NC-A3 remove=RESTORE_ITER1_R7 status=PASS")

    inversion = analyze_text(synthetic_log(add_inversion=True), 1.0e6)
    count = inversion["iterations"]["1"]["census"]["INVERSION_BOUNDARY"]
    if count != 1 or inversion["verdict"] != "A":
        raise AnalysisError("NC-A4_INVERSION_DID_NOT_CONTINUE")
    print("NC-A4 inject=CHI_ZERO_ETA_POSITIVE status=FAIL "
          "reason=INVERSION_BOUNDARY census=1 continued=1 verdict=A")
    clean = analyze_text(synthetic_log(), 1.0e6)
    if clean["iterations"]["1"]["census"]["INVERSION_BOUNDARY"] != 0:
        raise AnalysisError("NC-A4_REMOVAL_CENSUS_NONZERO")
    print("NC-A4 remove=INVERSION_ROW status=PASS")

    no_rows = (
        f"{R7_PREFIX} lane=DET iter=0 phase=A2-10 te_generation=1->2\n"
    )
    try:
        analyze_text(no_rows, 1.0e6)
    except AnalysisError as exc:
        if str(exc) != "NO_ROWS":
            raise
        print(f"NC-A5 inject=IDSEAL_ZERO_ROWS status=FAIL reason={exc}")
    else:
        raise AnalysisError("NC-A5_ZERO_ROWS_ACCEPTED")
    analyze_text(synthetic_log(), 1.0e6)
    print("NC-A5 remove=RESTORE_ROWS status=PASS")
    print("DET_STAGE12_L6_SELFTEST_PASS NC-A1..NC-A5=PASS")
    return 0


def time_explosion_from_run(run_root: Path) -> float:
    config = run_root / "input" / "model" / "config.json"
    if not config.is_file() or config.is_symlink():
        raise AnalysisError("MISSING_SAFE_MODEL_CONFIG")
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        value = float(payload["time_explosion_s"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError,
            TypeError, ValueError) as exc:
        raise AnalysisError("INVALID_MODEL_TIME_EXPLOSION") from exc
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--time-explosion-s", type=float)
    parser.add_argument("--temperature-k", type=float,
                        default=REQUESTED_TEMPERATURE_K)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        try:
            return selftest()
        except AnalysisError as exc:
            print(f"DET_STAGE12_L6_SELFTEST_FAIL reason={exc}", file=sys.stderr)
            return 4
    try:
        if args.run_root is None and args.stderr is None:
            raise AnalysisError("RUN_ROOT_OR_STDERR_REQUIRED")
        stderr_path = args.stderr or args.run_root / "stderr.log"
        if args.report is not None:
            report_path = args.report
        elif args.run_root is not None:
            report_path = args.run_root / "det_stage12_l6_verdict.json"
        else:
            raise AnalysisError("REPORT_REQUIRED_WITH_STDERR")
        if not stderr_path.is_file() or stderr_path.is_symlink():
            raise AnalysisError("MISSING_SAFE_STDERR")
        if args.time_explosion_s is not None:
            time_explosion_s = args.time_explosion_s
        elif args.run_root is not None:
            time_explosion_s = time_explosion_from_run(args.run_root)
        else:
            raise AnalysisError("TIME_EXPLOSION_REQUIRED")
        text = stderr_path.read_text(encoding="utf-8", errors="strict")
        report = analyze_text(text, time_explosion_s, args.temperature_k)
        report["stderr_sha256"] = sha256(stderr_path)
        atomic_write_json(report_path, report)
    except (AnalysisError, OSError, UnicodeError) as exc:
        failure_path = args.report
        if failure_path is None and args.run_root is not None:
            failure_path = args.run_root / "det_stage12_l6_verdict.json"
        if failure_path is not None:
            atomic_write_json(failure_path, {
                "schema": "DET_STAGE12_L6_VERDICT_V1",
                "status": "FAIL",
                "error": str(exc),
            })
        print(f"DET_STAGE12_L6_FAIL reason={exc}", file=sys.stderr)
        return 4
    f_super = report["f_super"]
    f_super_text = "UNAVAILABLE" if f_super is None else f"{f_super:.17g}"
    print(
        "DET_STAGE12_L6_PASS "
        f"verdict={report['verdict']} f_super={f_super_text} "
        f"report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
