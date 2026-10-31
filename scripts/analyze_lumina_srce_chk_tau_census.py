#!/usr/bin/env python3
"""같은 generation의 저장 C-full 인구로 native SRCE_CHK line-tau trigger를 센다.

읽기 전용 분석이다. chi/eta 계수, 복사장, 인구를 변경하거나 수송을 실행하지 않는다.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

import cmfgen_model_sources as S


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_hash(records: dict, name: str) -> str:
    matches = [value for path, value in records.items()
               if Path(path).name == name]
    if len(matches) != 1:
        raise ValueError(f"생산 기록의 {name} 입력 해시가 하나로 특정되지 않음")
    return matches[0]


def analyze(sources: Path, material: Path, c_full: Path,
            preparation: Path, output: Path) -> dict:
    sources, material, c_full, preparation, output = map(
        Path, (sources, material, c_full, preparation, output))
    if output.exists():
        raise FileExistsError("기존 진단 결과 보존")
    source_report_path = sources / "sources.json"
    source_report = json.loads(source_report_path.read_text())
    source_hash = sha256(source_report_path)
    lines_path = sources / "line_catalog.npz"
    levels_path = material / "atomic" / "levels.csv"
    population_path = material / "native_material_nodes.bin"

    line_hash = sha256(lines_path)
    if line_hash != source_report["output_sha256"]["line_catalog.npz"]:
        raise ValueError("선 목록이 계수 생산 기록과 다름")
    checked_inputs = {}
    for path in (levels_path, population_path, c_full):
        expected = recorded_hash(source_report["input_sha256"], path.name)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{path.name}이 계수 생산 입력과 다름")
        checked_inputs[path.name] = actual

    prep = json.loads(preparation.read_text())
    epoch = float(prep["epoch_seconds"])
    if not math.isfinite(epoch) or epoch <= 0:
        raise ValueError("팽창시간이 유효하지 않음")

    with np.load(lines_path, allow_pickle=False) as archive:
        required = ("lower", "upper", "nu0", "f", "g_ratio")
        if any(name not in archive.files for name in required):
            raise ValueError("선 목록 필드 누락")
        lower = np.asarray(archive["lower"], dtype=np.int64)
        upper = np.asarray(archive["upper"], dtype=np.int64)
        nu0 = np.asarray(archive["nu0"], dtype=np.float64)
        oscillator = np.asarray(archive["f"], dtype=np.float64)
        g_ratio = np.asarray(archive["g_ratio"], dtype=np.float64)
    nline = len(lower)
    if (source_report.get("cells") < 1 or nline != source_report.get("lines") or
            any(a.shape != (nline,) for a in (upper, nu0, oscillator, g_ratio)) or
            np.any(lower < 0) or np.any(upper < 0) or
            np.any(~np.isfinite(nu0)) or np.any(nu0 <= 0) or
            np.any(~np.isfinite(oscillator)) or np.any(oscillator < 0) or
            np.any(~np.isfinite(g_ratio)) or np.any(g_ratio <= 0)):
        raise ValueError("선 목록 형상 또는 값 오류")

    levels = list(csv.DictReader(levels_path.open(newline="")))
    with c_full.open("rb") as stream:
        magic, nlevel, nshell = struct.unpack("<8sII", stream.read(16))
        if (magic != b"LMATFL01" or nlevel != len(levels) or
                nshell != source_report["cells"] or
                np.any(lower >= nlevel) or np.any(upper >= nlevel)):
            raise ValueError("C-full 인구 형상/주소 오류")
        population = np.fromfile(stream, dtype="<f8").reshape(nlevel, nshell)
        if stream.read(1):
            raise ValueError("C-full 인구 뒤에 예상치 못한 데이터")
    if not np.isfinite(population).all() or np.any(population < 0):
        raise ValueError("C-full 인구가 유한·비음수가 아님")

    atomic_number = np.asarray([int(row["atomic_number"]) for row in levels])
    ion_number = np.asarray([int(row["ion_number"]) for row in levels])
    level_number = np.asarray([int(row["level_number"]) for row in levels])
    line_z, line_ion = atomic_number[lower], ion_number[lower]
    # Same integrated-opacity convention as the Gaussian source producer:
    # chi_int = pi*e^2/(m_e*c) * f * (n_l - g_l/g_u*n_u), tau=chi_int*c*t/nu.
    opacity_constant = math.pi * S.B.atomic.E_CGS**2 / (S.B.ME * S.B.C)
    line_factor = opacity_constant * oscillator * (S.B.C / nu0) * epoch
    if not np.isfinite(line_factor).all():
        raise ValueError("Sobolev tau 계수에 비유한 값")

    groups = {(14, 1): "Si_II", (14, 2): "Si_III",
              (16, 1): "S_II", (16, 2): "S_III"}
    group_mask = {(z, ion): (line_z == z) & (line_ion == ion)
                  for z, ion in groups}
    ion_counts = {name: {"line_shell_tau_negative": 0,
                         "line_shell_tau_below_minus_0_5": 0}
                  for name in groups.values()}
    negative_pairs = 0
    replacement_pairs = 0
    triggered_lines = np.zeros(nline, dtype=bool)
    minimum_tau = math.inf
    minimum_line = minimum_shell = -1
    for shell in range(nshell):
        tau = line_factor * (population[lower, shell] -
                             g_ratio * population[upper, shell])
        negative = tau < 0.0
        replacement = tau < -0.5
        negative_pairs += int(np.count_nonzero(negative))
        replacement_pairs += int(np.count_nonzero(replacement))
        triggered_lines |= replacement
        for key, name in groups.items():
            selected = group_mask[key]
            ion_counts[name]["line_shell_tau_negative"] += int(
                np.count_nonzero(negative[selected]))
            ion_counts[name]["line_shell_tau_below_minus_0_5"] += int(
                np.count_nonzero(replacement[selected]))
        line = int(np.argmin(tau))
        if tau[line] < minimum_tau:
            minimum_tau, minimum_line, minimum_shell = (
                float(tau[line]), line, shell)

    target = np.flatnonzero(
        (line_z == 14) & (line_ion == 1) &
        (level_number[lower] == 123) & (level_number[upper] == 124))
    if len(target) != 1:
        raise ValueError("Si II FL124→125가 유일하게 식별되지 않음")
    target_line = int(target[0])
    target_tau = line_factor[target_line] * (
        population[lower[target_line], :] -
        g_ratio[target_line] * population[upper[target_line], :])

    report = dict(
        schema="lumina-srce-chk-tau-census-v1",
        state="COMPLETE_READ_ONLY_TAU_RECONSTRUCTION",
        source_report_sha256=source_hash,
        line_catalog_sha256=line_hash,
        checked_input_sha256=checked_inputs,
        cells=nshell,
        lines=nline,
        epoch_seconds=epoch,
        tau_definition="(pi*e^2/(m_e*c))*f*(c/nu0)*t_exp*(n_lower-g_ratio*n_upper)",
        srce_chk_rule="tau < -0.5",
        line_shell_pairs_tau_negative=negative_pairs,
        line_shell_pairs_tau_below_minus_0_5=replacement_pairs,
        unique_lines_with_any_tau_below_minus_0_5=int(
            np.count_nonzero(triggered_lines)),
        minimum=dict(tau=minimum_tau, shell_0based=minimum_shell,
                     line_catalog_row=minimum_line,
                     Z=int(line_z[minimum_line]),
                     ion_number=int(line_ion[minimum_line]),
                     lower_FL_1based=int(level_number[lower[minimum_line]]) + 1,
                     upper_FL_1based=int(level_number[upper[minimum_line]]) + 1,
                     wavelength_A=float(S.B.C / nu0[minimum_line] * 1.0e8)),
        Si_II_FL124_to_125=dict(
            shell54_tau=float(target_tau[54]),
            minimum_shell_tau=float(np.min(target_tau)),
            maximum_shell_tau=float(np.max(target_tau)),
            wavelength_A=float(S.B.C / nu0[target_line] * 1.0e8)),
        ion_counts=ion_counts,
        interpretation=("SRCE_CHK tau replacement is inactive for this saved "
                        "line/material generation if the reconstructed Sobolev "
                        "tau is the native trigger quantity; this does not test "
                        "the separate continuum-opacity floor or overall CMFGEN parity."),
        physical_RT_executions=0,
        modified_opacity=False,
        final_goal="NOT_MET")
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--c-full", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.sources, args.material, args.c_full,
                     args.preparation, args.out)
    print(json.dumps({key: result[key] for key in (
        "line_shell_pairs_tau_negative",
        "line_shell_pairs_tau_below_minus_0_5",
        "unique_lines_with_any_tau_below_minus_0_5", "minimum",
        "Si_II_FL124_to_125", "ion_counts", "final_goal")},
        ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
