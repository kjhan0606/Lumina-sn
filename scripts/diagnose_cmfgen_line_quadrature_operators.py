#!/usr/bin/env python3
"""고정 J에서 line-profile 격자·적분 연산자를 독립적으로 대조한다.

새 물리해/SE/온도응답/수송은 수행하지 않는다. source-sampled C 적분을
순수 Python 사다리꼴과, piecewise-linear C 적분을 별도 Gauss-Legendre
참조 적분과 비교하고 fine/native 주파수 격자 차이를 분리한다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import cmfgen_model_continuum as B
import cmfgen_model_sources as S
from compare_lumina_native_grid_heat import parse_native_cont_freq_text
from lumina_model_linear_lines import LinearModelLines, SourceModelLines
from reconstruct_cmfgen_lineheat_fine_j import read_eddfactor_depths


PROFILE_PHYSICS_KEYS = (
    "kind", "TDOP_1e4K", "AMASS_DOP_amu", "VTURB_kms", "velocity_kms",
    "width_fraction", "support_fraction", "support_native_NUM_DOP",
    "profile_normalization", "MAX_DOP",
)
# Shared native/source profile kernel literal; using math.sqrt(pi) instead
# introduces a 2.8e-11 normalization offset and falsely fails the sampled check.
PROFILE_DENOM = 1.77245385095516
DEPTHS_1BASED = (71, 81, 85, 90)
TOP_LINES_PER_DEPTH = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_error(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.maximum(np.abs(right), np.finfo(np.float64).tiny)
    return np.abs(left - right) / denominator


def sampled_reference(grid, field, center, profile):
    """독립 Python 구현의 nodal Gaussian·사다리꼴 내적."""
    low = center * (1.0 - profile["support_fraction"])
    high = center * (1.0 + profile["support_fraction"])
    if low < grid[0] or high > grid[-1]:
        raise ValueError("Python 표본 적분의 선 지지범위가 격자 밖임")
    first = int(np.searchsorted(grid, low, side="right"))
    if first and grid[first - 1] == low:
        first -= 1
    stop = int(np.searchsorted(grid, high, side="right"))
    indices = np.arange(first, stop, dtype=np.int64)
    if not len(indices):
        raise ValueError("Python 표본 적분에서 선이 미표본됨")
    dnu = np.zeros(len(indices), dtype=np.float64)
    valid_left = indices > 0
    valid_right = indices + 1 < len(grid)
    dnu[valid_left] += 0.5 * (grid[indices[valid_left]] - grid[indices[valid_left] - 1])
    dnu[valid_right] += 0.5 * (grid[indices[valid_right] + 1] - grid[indices[valid_right]])
    width = center * profile["width_fraction"]
    coordinate = (grid[indices] - center) / width
    phi = np.exp(-coordinate * coordinate) / (PROFILE_DENOM * width)
    weights = phi * dnu
    mass = float(np.sum(weights.astype(np.longdouble), dtype=np.longdouble))
    values = np.asarray(
        np.sum(weights.astype(np.longdouble)[:, None] *
               field[indices].astype(np.longdouble), axis=0, dtype=np.longdouble),
        dtype=np.float64,
    )
    return values, mass


def piecewise_linear_gl_reference(grid, field, center, profile, order=16):
    """Gauss-Legendre로 piecewise-linear J와 Gaussian의 곱을 직접 적분한다."""
    low = center * (1.0 - profile["support_fraction"])
    high = center * (1.0 + profile["support_fraction"])
    if low < grid[0] or high > grid[-1]:
        raise ValueError("Gauss 참조 적분의 선 지지범위가 격자 밖임")
    first = max(0, int(np.searchsorted(grid, low, side="right")) - 1)
    last = min(len(grid) - 2, int(np.searchsorted(grid, high, side="left")))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    width = center * profile["width_fraction"]
    total = np.zeros(field.shape[1], dtype=np.longdouble)
    mass = np.longdouble(0.0)
    for segment in range(first, last + 1):
        left = max(low, float(grid[segment]))
        right = min(high, float(grid[segment + 1]))
        if right <= left:
            continue
        half = 0.5 * (right - left)
        midpoint = 0.5 * (right + left)
        frequency = midpoint + half * nodes
        fraction = (frequency - grid[segment]) / (grid[segment + 1] - grid[segment])
        interpolated = ((1.0 - fraction)[:, None] * field[segment] +
                        fraction[:, None] * field[segment + 1])
        coordinate = (frequency - center) / width
        phi = np.exp(-coordinate * coordinate) / (PROFILE_DENOM * width)
        quadrature = (half * weights * phi).astype(np.longdouble)
        total += np.sum(quadrature[:, None] * interpolated.astype(np.longdouble), axis=0,
                        dtype=np.longdouble)
        mass += np.sum(quadrature, dtype=np.longdouble)
    if not np.isfinite(float(mass)) or mass <= 0:
        raise ValueError("Gauss 참조 적분의 profile mass 오류")
    return np.asarray(total, dtype=np.float64), float(mass)


def summarize_errors(values):
    array = np.asarray(values, dtype=np.float64)
    return dict(max=float(np.max(array)), p95=float(np.quantile(array, 0.95)),
                median=float(np.median(array)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--response-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--binding-report", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.sources, args.response_root, args.native_root,
                 args.binding_report.parent, args.replay_report.parent,
                 args.output.parent):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"출력 파일 보존 규칙: 이미 존재함: {args.output}")

    response_record_path = args.response_root / "result.json"
    response_record = json.loads(response_record_path.read_text())
    source_record = json.loads((args.sources / "sources.json").read_text())
    binding = json.loads(args.binding_report.read_text())
    replay = json.loads(args.replay_report.read_text())
    selected = response_record["selected_response"]
    evaluation = args.response_root / Path(selected["root"]).name
    response_path = evaluation / "response.npz"
    balance_path = evaluation / "balance.npz"
    jbar_path = args.response_root / "jbar.npy"
    mass_path = args.response_root / "line-profile-mass.npy"
    catalog_path = args.sources / "line_catalog.npz"
    source_bin = args.sources / "sources.bin"
    model_paths = [Path(name) for name in response_record["input_sha256"]
                   if Path(name).name == "MODEL"]
    levels_paths = [Path(name) for name in response_record["input_sha256"]
                    if Path(name).name == "levels.csv"]
    if len(model_paths) != 1 or len(levels_paths) != 1:
        raise ValueError("선택 응답의 원본 MODEL/levels.csv 신원이 모호함")
    model_path, levels_path = model_paths[0], levels_paths[0]
    native_model = args.native_root / "MODEL"
    native_grid_path = args.native_root / "CONT_FREQ"
    edd_path = args.native_root / "EDDFACTOR"
    edd_info_path = args.native_root / "EDDFACTOR_INFO"
    runtime_path = args.native_root / "seq_logs/fixedj-heat-runtime.json"
    runtime = json.loads(runtime_path.read_text())

    if (binding.get("same_state_verified") is not True or
            binding.get("selected_response_sha256") != selected["response_sha256"] or
            binding.get("selected_balance_sha256") != selected["balance_sha256"] or
            binding.get("producer_root_converged") is not False):
        raise ValueError("고정-state binding 또는 미수렴 상태 표기가 예상과 다름")
    if (runtime.get("status") != "CAPTURED" or
            runtime.get("captured_state_exact") is not True or
            runtime.get("material_update") is not False or
            runtime.get("new_transport") is not False or
            runtime.get("fixed_j_sha256") != binding["fixed_J_sha256"] or
            runtime.get("accepted_response_sha256") != selected["response_sha256"]):
        raise ValueError("native 자료가 승인된 고정-state/동일 J 포획이 아님")
    if response_record["source_identity"]["source_sha256"] != source_record["output_sha256"]["sources.bin"]:
        raise ValueError("source roster가 선택 응답과 다름")
    if binding["input_sha256"]["fixed_J_field_factor"]["sha256"] != sha256(edd_path):
        raise ValueError("고정 J EDDFACTOR hash 불일치")
    if binding["input_sha256"]["fixed_J_field_info"]["sha256"] != sha256(edd_info_path):
        raise ValueError("고정 J EDDFACTOR_INFO hash 불일치")
    if binding["input_sha256"]["native_runtime"]["sha256"] != sha256(runtime_path):
        raise ValueError("native capture runtime hash 불일치")

    expected_small = {
        catalog_path: source_record["output_sha256"]["line_catalog.npz"],
        source_bin: source_record["output_sha256"]["sources.bin"],
        jbar_path: response_record["output_sha256"]["jbar.npy"],
        mass_path: response_record["output_sha256"]["line-profile-mass.npy"],
        response_path: selected["response_sha256"],
        balance_path: selected["balance_sha256"],
        model_path: response_record["input_sha256"][str(model_path)],
        levels_path: response_record["input_sha256"][str(levels_path)],
    }
    for path, expected in expected_small.items():
        if sha256(path) != expected:
            raise ValueError(f"봉인 입력 hash 불일치: {path.name}")

    source_profile = source_record["profile"]
    original_profile = S.fixed_profile(model_path)
    coarse_profile = S.fixed_profile(native_model)
    for key in PROFILE_PHYSICS_KEYS:
        if source_profile.get(key) != original_profile.get(key):
            raise ValueError(f"source/original MODEL 물리 프로파일 불일치: {key}")
        if source_profile.get(key) != coarse_profile.get(key):
            raise ValueError(f"fine/native 물리 프로파일 불일치: {key}")
    profile_grid_setting = dict(source=source_profile.get("FRAC_DOP"),
                                native=coarse_profile.get("FRAC_DOP"))

    with catalog_path.open("rb") as stream:
        catalog = np.load(stream)
        lines = {name: catalog[name] for name in catalog.files}
    if set(lines) != {"lower", "upper", "nu0", "f", "A", "g_ratio"}:
        raise ValueError("line catalogue 필드가 예상 계약과 다름")
    rows = list(csv.DictReader(levels_path.open(newline="")))
    lookup = {}
    for index, (lower, upper) in enumerate(zip(lines["lower"], lines["upper"])):
        lo, up = rows[int(lower)], rows[int(upper)]
        key = (int(lo["atomic_number"]), int(lo["ion_number"]) + 1,
               int(lo["level_number"]) + 1, int(up["level_number"]) + 1)
        if key in lookup:
            raise ValueError(f"중복 원자 전이 주소: {key}")
        lookup[key] = index

    top = replay["native_BB_heat_ranking"]["top_lines"]
    selected_lines = {}
    top_evidence = []
    for depth in DEPTHS_1BASED:
        entries = top.get(str(depth))
        if not entries:
            raise ValueError(f"열차 순위 근거 깊이 누락: {depth}")
        for item in entries[:TOP_LINES_PER_DEPTH]:
            key = (int(item["Z"]), int(item["stage"]),
                   int(item["lower_full"]), int(item["upper_full"]))
            index = lookup.get(key)
            if index is None:
                raise ValueError(f"순위 전이를 source catalogue에서 찾지 못함: {key}")
            reported_nu = B.C / (float(item["wavelength_A"]) * 1.0e-8)
            if abs(float(lines["nu0"][index]) - reported_nu) > 5.01e8:
                raise ValueError("순위 전이/source catalogue 주파수 불일치")
            selected_lines.setdefault(index, item)
            top_evidence.append(dict(depth=depth, line_id=int(item["line_id"]),
                                     catalog_index=index, Z=key[0], stage=key[1],
                                     native_minus_fine_heat=float(item["delta"])))
    indices = np.asarray(sorted(selected_lines), dtype=np.int64)
    if not len(indices):
        raise ValueError("대표 선 선택 결과가 비어 있음")
    subset = {name: np.ascontiguousarray(value[indices], dtype=np.float64)
              for name, value in lines.items()}
    depth_indices = tuple(depth - 1 for depth in DEPTHS_1BASED)

    fine_nu, fine_j = read_eddfactor_depths(edd_path, edd_info_path, depth_indices)
    native_nu, _, native_grid_metadata = parse_native_cont_freq_text(
        native_grid_path.read_text(encoding="ascii", errors="strict"))
    if (native_nu[0] < fine_nu[0] or native_nu[-1] > fine_nu[-1] or
            not np.isfinite(fine_j).all() or np.any(fine_j < 0)):
        raise ValueError("fine/native 공통 격자 외삽 또는 J 값 오류")
    native_j = np.column_stack([
        np.interp(native_nu, fine_nu, fine_j[:, column])
        for column in range(len(depth_indices))
    ])

    library = args.response_root.parent / "build/lines.so"
    if not library.is_file():
        raise FileNotFoundError(library)
    source_fine = SourceModelLines(subset, source_profile, library, fine_nu, fine_j,
                                   radiation_generation=1)
    exact_fine = LinearModelLines(subset, source_profile, library, fine_nu, fine_j,
                                  radiation_generation=1)
    source_native = SourceModelLines(subset, source_profile, library, native_nu, native_j,
                                     radiation_generation=1)
    fine_source_values = source_fine.contract(0, len(indices), radiation_generation=1)
    fine_exact_values = exact_fine.contract(0, len(indices), radiation_generation=1)
    native_source_values = source_native.contract(0, len(indices), radiation_generation=1)
    for name, values in (("fine_source", fine_source_values),
                         ("fine_linear", fine_exact_values),
                         ("native_source", native_source_values)):
        if not np.isin(values["status"], (1, 2)).all():
            raise ValueError(f"선 지지/표본이 유효하지 않음: {name}")

    python_fine_sample = np.empty_like(fine_source_values["jbar"])
    python_native_sample = np.empty_like(native_source_values["jbar"])
    python_linear = np.empty_like(fine_exact_values["jbar"])
    python_linear_mass = np.empty_like(fine_exact_values["profile_mass"])
    for line, center in enumerate(subset["nu0"]):
        python_fine_sample[line], _ = sampled_reference(fine_nu, fine_j, center, source_profile)
        python_native_sample[line], _ = sampled_reference(native_nu, native_j, center, source_profile)
        python_linear[line], python_linear_mass[line] = piecewise_linear_gl_reference(
            fine_nu, fine_j, center, source_profile, order=16)

    sampled_fine_vs_python = relative_error(fine_source_values["jbar"], python_fine_sample)
    sampled_native_vs_python = relative_error(native_source_values["jbar"], python_native_sample)
    exact_vs_gl = relative_error(fine_exact_values["jbar"], python_linear)
    exact_mass_vs_gl = relative_error(fine_exact_values["profile_mass"], python_linear_mass)
    saved_jbar = np.load(jbar_path, mmap_mode="r")
    saved_selected = np.asarray(saved_jbar[np.ix_(indices, depth_indices)])
    production_vs_replayed_fine = relative_error(fine_source_values["jbar"], saved_selected)
    native_vs_fine_sampled = relative_error(native_source_values["jbar"],
                                            fine_source_values["jbar"])
    native_vs_fine_exact = relative_error(native_source_values["jbar"],
                                          fine_exact_values["jbar"])
    mass_native_vs_fine = relative_error(native_source_values["profile_mass"],
                                         fine_source_values["profile_mass"])

    limits = dict(source_sample_vs_python_max_relative=1.0e-11,
                  linear_vs_Gauss16_max_relative=1.0e-9,
                  profile_mass_vs_Gauss16_max_relative=1.0e-9,
                  saved_production_jbar_vs_fine_replay_max_relative=1.0e-8)
    checks = dict(
        source_sample_vs_python_trapezoid_pass=bool(np.max(sampled_fine_vs_python) <= limits["source_sample_vs_python_max_relative"] and
                                                   np.max(sampled_native_vs_python) <= limits["source_sample_vs_python_max_relative"]),
        linear_vs_independent_Gauss16_pass=bool(np.max(exact_vs_gl) <= limits["linear_vs_Gauss16_max_relative"] and
                                                np.max(exact_mass_vs_gl) <= limits["profile_mass_vs_Gauss16_max_relative"]),
        saved_production_jbar_replay_pass=bool(np.max(production_vs_replayed_fine) <= limits["saved_production_jbar_vs_fine_replay_max_relative"]),
    )
    all_pass = all(checks.values())

    line_rows = []
    for row_index, catalog_index in enumerate(indices):
        exemplar = selected_lines[int(catalog_index)]
        for column, depth in enumerate(DEPTHS_1BASED):
            line_rows.append(dict(
                line_id=int(exemplar["line_id"]), transition=exemplar["transition"],
                Z=int(exemplar["Z"]), stage=int(exemplar["stage"]),
                catalog_index=int(catalog_index), depth=depth,
                wavelength_A=float(B.C / subset["nu0"][row_index] * 1.0e8),
                fine_sampled_Jbar=float(fine_source_values["jbar"][row_index, column]),
                fine_linear_Jbar=float(fine_exact_values["jbar"][row_index, column]),
                native_sampled_Jbar=float(native_source_values["jbar"][row_index, column]),
                saved_production_Jbar=float(saved_selected[row_index, column]),
                native_profile_mass=float(native_source_values["profile_mass"][row_index]),
                fine_profile_mass=float(fine_source_values["profile_mass"][row_index]),
                relative_native_vs_fine_sampled=float(native_vs_fine_sampled[row_index, column]),
                relative_native_vs_fine_linear=float(native_vs_fine_exact[row_index, column]),
                relative_fine_sampled_vs_linear=float(relative_error(
                    fine_source_values["jbar"][row_index, column],
                    fine_exact_values["jbar"][row_index, column])),
            ))

    hashed_inputs = {
        "sources.json": sha256(args.sources / "sources.json"),
        "source.bin": sha256(source_bin),
        "line_catalog.npz": sha256(catalog_path),
        "response.result.json": sha256(response_record_path),
        "selected.response.npz": sha256(response_path),
        "selected.balance.npz": sha256(balance_path),
        "saved.jbar.npy": sha256(jbar_path),
        "line.profile.mass.npy": sha256(mass_path),
        "original.MODEL": sha256(model_path),
        "native.MODEL": sha256(native_model),
        "native.CONT_FREQ": sha256(native_grid_path),
        "native.EDDFACTOR": sha256(edd_path),
        "native.EDDFACTOR_INFO": sha256(edd_info_path),
        "native.runtime": sha256(runtime_path),
        "line.integrator.library": sha256(library),
        "same.state.binding": sha256(args.binding_report),
        "all.depth.replay": sha256(args.replay_report),
    }
    output = dict(
        schema="LUMINA_CMFGEN_LINE_QUADRATURE_OPERATOR_DIAGNOSTIC_V1",
        status="PASS" if all_pass else "FAIL",
        scope="고정 동일 J에서 상위 열 기여 Co/기타 선의 line-profile 적분 연산자 대조",
        final_goal="NOT_MET",
        new_physical_evaluations=0,
        new_SE=0,
        new_temperature_response=0,
        new_transport=0,
        material_update=0,
        producer_root_converged=False,
        selected_response_sha256=selected["response_sha256"],
        fixed_J_sha256=binding["fixed_J_sha256"],
        depths_1based=list(DEPTHS_1BASED),
        selected_line_count=int(len(indices)),
        unique_species_stages=sorted({f"Z{item['Z']}_stage{item['stage']}" for item in selected_lines.values()}),
        line_ids=sorted({int(item["line_id"]) for item in selected_lines.values()}),
        frequency_points=dict(fine=int(len(fine_nu)), native=int(len(native_nu)),
                              native_metadata=native_grid_metadata),
        profile_contract=dict(physical_keys=PROFILE_PHYSICS_KEYS,
                              source_vs_original_exact=True,
                              source_vs_native_physics_exact=True,
                              grid_setting_FRAC_DOP=profile_grid_setting,
                              interpretation="FRAC_DOP는 native 주파수격자 설정이며 Gaussian width/support의 물리 프로파일과 분리"),
        checks=checks,
        limits=limits,
        errors=dict(fine_source_vs_python_trapezoid=summarize_errors(sampled_fine_vs_python),
                    native_source_vs_python_trapezoid=summarize_errors(sampled_native_vs_python),
                    fine_linear_vs_independent_Gauss16=summarize_errors(exact_vs_gl),
                    fine_profile_mass_vs_independent_Gauss16=summarize_errors(exact_mass_vs_gl),
                    saved_production_jbar_vs_fine_source_replay=summarize_errors(production_vs_replayed_fine),
                    native_vs_fine_source_sample=summarize_errors(native_vs_fine_sampled),
                    native_vs_fine_linear_reference=summarize_errors(native_vs_fine_exact),
                    native_vs_fine_profile_mass=summarize_errors(mass_native_vs_fine)),
        selected_line_depth_values=line_rows,
        ranking_evidence=top_evidence,
        input_sha256=hashed_inputs,
        method="C source-sampled fine/native versus pure-Python nodal trapezoid; C piecewise-linear Gaussian versus independent 16-point Gauss-Legendre integration of the same piecewise-linear J. Native J is linearly interpolated from the sealed fine J; no native transport or SE.",
        interpretation="이 시험은 적분 구현의 재현성과 격자 표본화 차이만 판정한다. 미수렴 물질 상태·복사장 왕복·CMFGEN 이온분포/스펙트럼 동질성은 판정하지 않는다.",
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": output["status"], "output": str(args.output),
                      "selected_lines": output["selected_line_count"],
                      "checks": checks, "errors": output["errors"]},
                     ensure_ascii=False, sort_keys=True))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
