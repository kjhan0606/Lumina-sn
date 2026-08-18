#!/usr/bin/env python3
"""Compare production exact J_nu points with an independent CMFGEN EDDFACTOR.

This is a same-quantity/same-coordinate finite comparison, not a parity claim:
the historical external CMFGEN run and the sealed Lumina deck do not yet carry
a certified common-state identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


C_A_PER_S = 2.99792458e18


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_info(path: Path) -> tuple[int, int, int, bool]:
    lines = path.read_text(errors="strict").splitlines()
    if len(lines) < 3:
        raise ValueError("EDDFACTOR_INFO is truncated")
    fields = lines[2].split()
    if len(fields) < 6:
        raise ValueError("EDDFACTOR_INFO layout is invalid")
    nd, recl, word = map(int, fields[:3])
    little = fields[5] == "T"
    if nd <= 1 or word not in (4, 8) or recl != (nd + 1) * word:
        raise ValueError("EDDFACTOR_INFO dimensions are inconsistent")
    return nd, recl, word, little


def parse_rvtj_block(text: str, label: str, count: int) -> np.ndarray:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != label:
            continue
        values: list[float] = []
        for following in lines[index + 1 :]:
            try:
                values.extend(float(token) for token in following.split())
            except ValueError:
                break
            if len(values) >= count:
                break
        if len(values) < count:
            raise ValueError(f"RVTJ block {label!r} is truncated")
        result = np.asarray(values[:count], dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError(f"RVTJ block {label!r} is nonfinite")
        return result
    raise ValueError(f"RVTJ block {label!r} is missing")


def load_cmfgen(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    edd = root / "EDDFACTOR"
    info = root / "EDDFACTOR_INFO"
    rvtj = root / "RVTJ"
    nd, recl, word, little = read_info(info)
    dtype = ("<" if little else ">") + ("f8" if word == 8 else "f4")
    words_per_record = recl // word
    raw = np.fromfile(edd, dtype=dtype)
    if raw.size % words_per_record != 0:
        raise ValueError("EDDFACTOR byte count is not a whole record count")
    records = raw.reshape(-1, words_per_record)
    if records.shape[0] <= 14:
        raise ValueError("EDDFACTOR has no radiation records")
    finish = float(records[4, 0])
    if not math.isfinite(finish) or finish == 0.0:
        raise ValueError("EDDFACTOR FINISH_REC does not certify completion")
    data = records[14:]
    good = np.isfinite(data[:, :nd]).all(axis=1) & (data[:, nd] > 0.0)
    if not good.any():
        raise ValueError("EDDFACTOR has no finite positive-frequency records")
    frequency = np.asarray(data[good, nd], dtype=np.float64) * 1.0e15
    field = np.asarray(data[good, :nd], dtype=np.float64)
    velocity = parse_rvtj_block(
        rvtj.read_text(errors="strict"), "Velocity (km/s)", nd
    )
    frequency_order = np.argsort(frequency)
    velocity_order = np.argsort(velocity)
    frequency = frequency[frequency_order]
    field = field[frequency_order][:, velocity_order]
    velocity = velocity[velocity_order]
    if not np.all(np.diff(frequency) > 0.0):
        raise ValueError("CMFGEN frequency coordinate is not strictly increasing")
    if not np.all(np.diff(velocity) > 0.0):
        raise ValueError("CMFGEN velocity coordinate is not strictly increasing")
    return frequency, velocity, field, finish


def positive_log_bilinear(
    frequency: np.ndarray,
    velocity: np.ndarray,
    field: np.ndarray,
    target_frequency: float,
    target_velocity: float,
) -> float | None:
    if not (
        frequency[0] <= target_frequency <= frequency[-1]
        and velocity[0] <= target_velocity <= velocity[-1]
    ):
        return None
    fi = int(np.searchsorted(frequency, target_frequency))
    vi = int(np.searchsorted(velocity, target_velocity))
    fi = min(max(fi, 1), len(frequency) - 1)
    vi = min(max(vi, 1), len(velocity) - 1)
    f0, f1 = float(frequency[fi - 1]), float(frequency[fi])
    v0, v1 = float(velocity[vi - 1]), float(velocity[vi])
    values = np.asarray(
        [
            field[fi - 1, vi - 1],
            field[fi - 1, vi],
            field[fi, vi - 1],
            field[fi, vi],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or not np.all(values > 0.0):
        return None
    velocity_weight = (target_velocity - v0) / (v1 - v0)
    log_frequency_weight = (
        (math.log(target_frequency) - math.log(f0))
        / (math.log(f1) - math.log(f0))
    )
    logs = np.log(values)
    lower = (1.0 - velocity_weight) * logs[0] + velocity_weight * logs[1]
    upper = (1.0 - velocity_weight) * logs[2] + velocity_weight * logs[3]
    value = math.exp(
        (1.0 - log_frequency_weight) * lower
        + log_frequency_weight * upper
    )
    return value if value > 0.0 and math.isfinite(value) else None


def load_lumina(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {
            "shell",
            "v_mid_km_s",
            "target_lambda_A",
            "actual_lambda_A",
            "nu_hz",
            "J_nu",
        }
        if set(reader.fieldnames or ()) != expected:
            raise ValueError("Lumina fixture columns do not match the contract")
        for raw in reader:
            row: dict[str, float | int] = {
                "shell": int(raw["shell"]),
                "v_mid_km_s": float(raw["v_mid_km_s"]),
                "target_lambda_A": float(raw["target_lambda_A"]),
                "actual_lambda_A": float(raw["actual_lambda_A"]),
                "nu_hz": float(raw["nu_hz"]),
                "J_nu": float(raw["J_nu"]),
            }
            numeric = [float(value) for key, value in row.items() if key != "shell"]
            if (not all(math.isfinite(value) for value in numeric)
                    or float(row["nu_hz"]) <= 0.0
                    or float(row["actual_lambda_A"]) <= 0.0
                    or float(row["J_nu"]) <= 0.0):
                raise ValueError(f"invalid Lumina fixture row: {row}")
            reconstructed = C_A_PER_S / float(row["actual_lambda_A"])
            if reconstructed != float(row["nu_hz"]):
                relative = abs(reconstructed - float(row["nu_hz"])) / float(
                    row["nu_hz"]
                )
                if relative > 2.0e-15:
                    raise ValueError("Lumina wavelength/frequency identity mismatch")
            rows.append(row)
    if len(rows) != 400 or len({int(row["shell"]) for row in rows}) != 50:
        raise ValueError("Lumina fixture is not 50 shells x 8 points")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumina-fixture", type=Path, required=True)
    parser.add_argument("--cmfgen-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    lumina_rows = load_lumina(args.lumina_fixture)
    frequency, velocity, field, finish = load_cmfgen(args.cmfgen_root)
    comparisons: list[dict[str, float | int]] = []
    outside = 0
    unavailable = 0
    for row in lumina_rows:
        external = positive_log_bilinear(
            frequency,
            velocity,
            field,
            float(row["nu_hz"]),
            float(row["v_mid_km_s"]),
        )
        if external is None:
            if not (
                frequency[0] <= float(row["nu_hz"]) <= frequency[-1]
                and velocity[0] <= float(row["v_mid_km_s"]) <= velocity[-1]
            ):
                outside += 1
            else:
                unavailable += 1
            continue
        lumina = float(row["J_nu"])
        ratio = lumina / external
        log_ratio = math.log10(ratio)
        if not (ratio > 0.0 and math.isfinite(ratio) and math.isfinite(log_ratio)):
            raise ValueError("nonfinite external comparison ratio")
        comparisons.append(
            {
                **row,
                "cmfgen_J_nu": external,
                "lumina_over_cmfgen": ratio,
                "log10_lumina_over_cmfgen": log_ratio,
            }
        )

    if len(comparisons) < 300 or unavailable != 0:
        raise ValueError(
            f"insufficient finite overlap: compared={len(comparisons)} "
            f"outside={outside} unavailable={unavailable}"
        )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    logs = np.asarray(
        [float(row["log10_lumina_over_cmfgen"]) for row in comparisons]
    )
    by_target: dict[str, dict[str, float | int]] = {}
    for target in sorted({float(row["target_lambda_A"]) for row in comparisons}):
        subset = np.asarray(
            [
                float(row["log10_lumina_over_cmfgen"])
                for row in comparisons
                if float(row["target_lambda_A"]) == target
            ]
        )
        by_target[f"{target:.0f}"] = {
            "count": int(subset.size),
            "median_log10_ratio": float(np.median(subset)),
            "min_log10_ratio": float(subset.min()),
            "max_log10_ratio": float(subset.max()),
        }
    report = {
        "schema": "lumina-external-cmfgen-jnu-finite-comparison-v1",
        "verdict": "PASS_COMPARISON_NOT_PARITY",
        "quantity": "J_nu",
        "units": "erg s^-1 cm^-2 Hz^-1 sr^-1",
        "coordinate_rule": "actual Lumina fine-bin centre; log-bilinear CMFGEN interpolation in frequency and velocity",
        "lumina_rows": len(lumina_rows),
        "compared_rows": len(comparisons),
        "outside_cmfgen_domain": outside,
        "unavailable_inside_domain": unavailable,
        "median_log10_lumina_over_cmfgen": float(np.median(logs)),
        "min_log10_lumina_over_cmfgen": float(logs.min()),
        "max_log10_lumina_over_cmfgen": float(logs.max()),
        "by_target_lambda_A": by_target,
        "cmfgen_finish_record": finish,
        "cmfgen_frequency_points": int(frequency.size),
        "cmfgen_depth_points": int(velocity.size),
        "provenance": {
            "lumina_fixture_sha256": sha256(args.lumina_fixture),
            "cmfgen_EDDFACTOR_sha256": sha256(args.cmfgen_root / "EDDFACTOR"),
            "cmfgen_EDDFACTOR_INFO_sha256": sha256(
                args.cmfgen_root / "EDDFACTOR_INFO"
            ),
            "cmfgen_RVTJ_sha256": sha256(args.cmfgen_root / "RVTJ"),
        },
        "scope": "Independent executable output and same scalar J_nu definition; common physical-state identity is not certified, so no closeness threshold or CMFGEN parity claim is made.",
        "numerical_repair": {
            "floor": 0,
            "cap": 0,
            "clamp": 0,
            "jitter": 0,
        },
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
