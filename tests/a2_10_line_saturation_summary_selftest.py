#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/summarize_a210_line_saturation.py"
FOUR_PI = 12.56637061435917295385057353311801153679


def exponx(tau: float) -> tuple[float, float]:
    if abs(tau) < 1.0e-3:
        companion = 0.5 - tau / 6.0 * (1.0 - tau / 4.0)
        return 1.0 - tau * companion, companion
    if tau < 40.0:
        beta = (1.0 - math.exp(-tau)) / tau
        return beta, (1.0 - beta) / tau
    beta = 1.0 / tau
    return beta, (1.0 - beta) / tau


def row(rank: int, line: int, z: int, eta: float,
        cumulative: float, total: float, repair: int = 0) -> str:
    tau = 2.0
    chi = 1.0
    jbar = 0.8 * eta
    absorption = chi * jbar
    net = eta - absorption
    deck = 1.0
    factor = FOUR_PI * deck
    beta, companion = exponx(tau)
    source = eta / chi
    return (
        "[A2-10][LINE-SATURATION-ROW] phase=REQUESTED_TE shell=0 "
        f"rank={rank} line={line} Z={z} ion=3 ion_label=4 ion_slot={rank} "
        f"lower_global={100 + rank} upper_global={200 + rank} "
        f"lower_level={rank - 1} upper_level={rank + 9} "
        f"nu={5.0e15 + rank:.17g} tau_raw={tau:.17g} "
        f"tau_effective={tau:.17g} tau_validity=1 chi_raw={chi:.17g} "
        f"chi_effective={chi:.17g} srce_chk=0 n_upper=2 A_ul=3 "
        f"eta_per_sr={eta:.17g} Jbar={jbar:.17g} Jbar_absolute_bound=1e-9 "
        f"beta={beta:.17g} one_minus_beta_over_tau={companion:.17g} "
        f"one_minus_beta={tau * companion:.17g} "
        f"source_function_defined=1 source_function={source:.17g} "
        f"jbar_over_source={absorption / eta:.17g} deck_scale={deck:.17g} "
        f"absorption_per_sr={absorption:.17g} net_per_sr={net:.17g} "
        f"signed_rate={net * factor:.17g} uncertainty=1e-10 "
        f"cancellation_condition={(eta + abs(absorption)) / abs(net):.17g} "
        f"scaled_emission={eta * factor:.21g} "
        f"scaled_absorption={absorption * factor:.21g} "
        f"cumulative_scaled_emission={cumulative:.21g} "
        f"cumulative_fraction={cumulative / total:.21g} "
        "selection_target_fraction=0.9 scan_complete=1 "
        "interpretation=DIAGNOSTIC_ONLY physical_values_modified=0 "
        f"clamp=0 floor=0 cap=0 jitter=0 repair={repair}"
    )


def summary(total: float, selected: float) -> str:
    return (
        "[A2-10][LINE-SATURATION-SUMMARY] phase=REQUESTED_TE shell=0 "
        "target_Z=26,27,28 target_ion=3 candidate_rows=3 selected_rows=2 "
        f"total_scaled_emission={total:.21g} "
        f"selected_scaled_emission={selected:.21g} "
        f"selected_fraction={selected / total:.21g} "
        "selection_target_fraction=0.9 selected_reaches_target=1 complete=1 "
        "interpretation=DIAGNOSTIC_ONLY physical_values_modified=0 "
        "clamp=0 floor=0 cap=0 jitter=0 repair=0"
    )


def union_meta(line: int, z: int, global_rank: int, ion_rank: int,
               candidates: int, total: float, cumulative: float) -> str:
    return (
        "[A2-10][LINE-SATURATION-UNION-META] phase=REQUESTED_TE shell=0 "
        f"line={line} Z={z} ion=3 global_rank={global_rank} "
        f"ion_rank={ion_rank} ion_candidate_rows={candidates} "
        f"ion_total_scaled_emission={total:.21g} "
        f"ion_cumulative_scaled_emission={cumulative:.21g} "
        f"ion_cumulative_fraction={cumulative / total:.21g} "
        "selection_target_fraction=0.9 selection_mode=PER_ION_UNION "
        "scan_complete=1 interpretation=DIAGNOSTIC_ONLY "
        "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0"
    )


def union_ion_summary(z: int, candidates: int, selected_rows: int,
                      total: float, selected: float) -> str:
    return (
        "[A2-10][LINE-SATURATION-UNION-ION-SUMMARY] "
        f"phase=REQUESTED_TE shell=0 Z={z} ion=3 "
        f"candidate_rows={candidates} selected_rows={selected_rows} "
        f"total_scaled_emission={total:.21g} "
        f"selected_scaled_emission={selected:.21g} "
        f"selected_fraction={selected / total:.21g} "
        "selection_target_fraction=0.9 selected_reaches_target=1 "
        "prefix_minimal=1 selection_mode=PER_ION_UNION complete=1 "
        "interpretation=DIAGNOSTIC_ONLY physical_values_modified=0 "
        "clamp=0 floor=0 cap=0 jitter=0 repair=0"
    )


def union_summary(total: float, selected: float) -> str:
    return (
        "[A2-10][LINE-SATURATION-SUMMARY] phase=REQUESTED_TE shell=0 "
        "target_Z=26,27,28 target_ion=3 candidate_rows=14 selected_rows=11 "
        f"total_scaled_emission={total:.21g} "
        f"selected_scaled_emission={selected:.21g} "
        f"selected_fraction={selected / total:.21g} "
        "selection_target_fraction=0.9 selected_reaches_target=1 "
        "selection_mode=PER_ION_UNION complete=1 "
        "interpretation=DIAGNOSTIC_ONLY physical_values_modified=0 "
        "clamp=0 floor=0 cap=0 jitter=0 repair=0"
    )


def run(log: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCRIPT), "--log", str(log), "--report", str(report)],
        cwd=ROOT, text=True, capture_output=True,
    )


def main() -> int:
    total = 10.0 * FOUR_PI
    first = 6.0 * FOUR_PI
    selected = 9.5 * FOUR_PI
    positive = [
        row(1, 101, 26, 6.0, first, total),
        row(2, 202, 27, 3.5, selected, total),
        summary(total, selected),
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "positive.log"
        report = root / "positive.json"
        path.write_text("\n".join(positive) + "\n", encoding="utf-8")
        result = run(path, report)
        if result.returncode != 0:
            raise SystemExit(f"positive failed: {result.stdout} {result.stderr}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload["status"] != "PASS" or len(payload["rows"]) != 2:
            raise SystemExit("positive report mismatch")

        malformed = list(positive)
        malformed[0] = malformed[0].replace(" beta=", " missing_beta=")
        path = root / "malformed.log"
        path.write_text("\n".join(malformed) + "\n", encoding="utf-8")
        if run(path, root / "malformed.json").returncode != 4:
            raise SystemExit("malformed row accepted")

        under = list(positive)
        under[1] = under[1].replace(
            f"cumulative_fraction={selected / total:.21g}",
            "cumulative_fraction=0.89",
        )
        path = root / "under90.log"
        path.write_text("\n".join(under) + "\n", encoding="utf-8")
        if run(path, root / "under90.json").returncode != 4:
            raise SystemExit("under-90 selection accepted")

        repaired = list(positive)
        repaired[0] = repaired[0].replace("repair=0", "repair=1")
        path = root / "repair.log"
        path.write_text("\n".join(repaired) + "\n", encoding="utf-8")
        if run(path, root / "repair.json").returncode != 4:
            raise SystemExit("repair marker accepted")

        blocked = list(positive)
        blocked.append(
            "[A2-10][LINE-SATURATION-BLOCKED] reason=SEEDED phase=REQUESTED_TE "
            "shell=0 candidate_rows=2 target_Z=26,27,28 target_ion=3 "
            "complete=0 interpretation=DIAGNOSTIC_ONLY "
            "physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0"
        )
        path = root / "blocked.log"
        path.write_text("\n".join(blocked) + "\n", encoding="utf-8")
        if run(path, root / "blocked.json").returncode != 4:
            raise SystemExit("blocked record accepted")

        total = 60.0 * FOUR_PI
        selected_total = 54.0 * FOUR_PI
        union: list[str] = []
        union.append(row(1, 501, 28, 27.0, 27.0 * FOUR_PI, total))
        union.append(union_meta(
            501, 28, 1, 1, 2, 30.0 * FOUR_PI, 27.0 * FOUR_PI
        ))
        union.append(row(2, 401, 27, 18.0, 45.0 * FOUR_PI, total))
        union.append(union_meta(
            401, 27, 2, 1, 2, 20.0 * FOUR_PI, 18.0 * FOUR_PI
        ))
        for offset in range(9):
            rank = 5 + offset
            line = 100 + offset
            ion_cumulative = (offset + 1.0) * FOUR_PI
            candidate_cumulative = (51.0 + offset) * FOUR_PI
            union.append(row(
                rank, line, 26, 1.0, candidate_cumulative, total
            ))
            union.append(union_meta(
                line, 26, rank, offset + 1, 10,
                10.0 * FOUR_PI, ion_cumulative
            ))
        union.extend([
            union_ion_summary(
                26, 10, 9, 10.0 * FOUR_PI, 9.0 * FOUR_PI
            ),
            union_ion_summary(
                27, 2, 1, 20.0 * FOUR_PI, 18.0 * FOUR_PI
            ),
            union_ion_summary(
                28, 2, 1, 30.0 * FOUR_PI, 27.0 * FOUR_PI
            ),
            union_summary(total, selected_total),
        ])
        path = root / "union.log"
        report = root / "union.json"
        path.write_text("\n".join(union) + "\n", encoding="utf-8")
        result = run(path, report)
        if result.returncode != 0:
            raise SystemExit(f"union positive failed: {result.stdout} {result.stderr}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload["summary"].get("selection_mode") != "PER_ION_UNION" or \
           len(payload.get("union_metadata", [])) != 11:
            raise SystemExit("union report mismatch")

        deleted = list(union)
        del deleted[0]
        path = root / "union_deleted_row.log"
        path.write_text("\n".join(deleted) + "\n", encoding="utf-8")
        if run(path, root / "union_deleted_row.json").returncode != 4:
            raise SystemExit("union row deletion accepted")

        perturbed = list(union)
        perturbed[0] = perturbed[0].replace(
            f"scaled_emission={27.0 * FOUR_PI:.21g}",
            f"scaled_emission={26.0 * FOUR_PI:.21g}",
        )
        path = root / "union_perturbed_emission.log"
        path.write_text("\n".join(perturbed) + "\n", encoding="utf-8")
        if run(path, root / "union_perturbed_emission.json").returncode != 4:
            raise SystemExit("union scaled-emission perturbation accepted")

    print(
        "PASS a2_10_line_saturation_summary "
        "combined_positive+4_negative+union_positive+2_negative"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
