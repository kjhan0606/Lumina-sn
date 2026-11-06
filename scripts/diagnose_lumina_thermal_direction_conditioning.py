#!/usr/bin/env python3
"""저장된 90점 T/ne Jacobian의 Newton 방향과 조건 민감도를 독립 재현한다.

선형화 입력만 소비하며 새 물리 응답, 온도 갱신, 수송은 수행하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reduced_direction(jacobian, scaled_residual):
    """Match the solver's charge elimination and row-normalized thermal solve."""
    n = len(scaled_residual) // 2
    a, b = jacobian[:n, :n], jacobian[:n, n:]
    d, e = jacobian[n:, :n], jacobian[n:, n:]
    charge_temperature = np.linalg.solve(b, a)
    charge_residual = np.linalg.solve(b, scaled_residual[:n])
    schur = d - e @ charge_temperature
    thermal_rhs = scaled_residual[n:] - e @ charge_residual
    row_scale = np.max(np.abs(schur), axis=1)
    if not np.isfinite(row_scale).all() or np.any(row_scale <= 0):
        raise ValueError("전하 제거 열행의 유한 row scale이 없음")
    normalized_schur = schur / row_scale[:, None]
    normalized_rhs = thermal_rhs / row_scale
    temperature_direction = np.linalg.solve(normalized_schur, -normalized_rhs)
    ne_direction = -charge_residual - charge_temperature @ temperature_direction
    direction = np.r_[temperature_direction, ne_direction]
    singular = np.linalg.svd(normalized_schur, compute_uv=False)
    return dict(direction=direction, schur=normalized_schur,
                normalized_rhs=normalized_rhs, row_scale=row_scale,
                charge_block_condition=float(np.linalg.cond(b)),
                normalized_schur_singular_values=singular,
                normalized_schur_condition=float(singular[0] / singular[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linearization", type=Path, required=True)
    parser.add_argument("--compare-linearization", type=Path)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않음: {args.output}")
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)

    run = json.loads(args.run_result.read_text())
    block = run["temperature_block"]
    if run.get("final_goal") != "NOT_MET" or block.get("fixed_field_root_converged") is not False:
        raise ValueError("검사 대상이 미수렴 고정장 결과가 아님")
    def load_linearization(path):
        with np.load(path) as data:
            jacobian = np.asarray(data["jacobian"], dtype=np.float64)
            residual = np.asarray(data["scaled_residual"], dtype=np.float64)
            saved = np.asarray(data["log_direction"], dtype=np.float64)
            temperature = np.asarray(data["temperature"], dtype=np.float64)
            ne = np.asarray(data["ne"], dtype=np.float64)
        if (jacobian.shape != (180, 180) or residual.shape != (180,) or
                saved.shape != (180,) or temperature.shape != (90,) or
                ne.shape != (90,) or not all(np.isfinite(x).all() for x in
                                             (jacobian, residual, saved, temperature, ne))):
            raise ValueError("저장 선형화 형상/유한성 계약 오류")
        reduced = reduced_direction(jacobian, residual)
        return jacobian, residual, saved, temperature, ne, reduced

    jacobian, residual, saved_direction, temperature, ne, reduced = load_linearization(args.linearization)
    direct = reduced["direction"]
    reproduction_error = float(np.linalg.norm(direct - saved_direction) /
                                max(np.linalg.norm(saved_direction), np.finfo(float).tiny))
    relative_linear_residual = float(np.linalg.norm(jacobian @ direct + residual) /
                                     max(np.linalg.norm(residual), np.finfo(float).tiny))
    max_direction = float(np.max(np.abs(direct)))
    max_temperature = int(np.argmax(np.abs(direct[:90]))) + 1
    max_ne = int(np.argmax(np.abs(direct[90:]))) + 1
    if (abs(max_direction - float(block["root_fresh_direction_max"])) > 1.0e-8 or
            reproduction_error > 1.0e-7 or relative_linear_residual > 1.0e-7):
        raise ValueError("독립 선형해가 저장 방향/실행 결과를 재현하지 못함")

    comparison = None
    if args.compare_linearization is not None:
        other_jacobian, other_residual, other_saved, _, _, other = load_linearization(
            args.compare_linearization)
        other_direction = other["direction"]
        other_error = float(np.linalg.norm(other_direction - other_saved) /
                            max(np.linalg.norm(other_saved), np.finfo(float).tiny))
        direction_t = direct[:90]
        other_t = other_direction[:90]
        comparison = dict(
            other_linearization_sha256=sha256(args.compare_linearization),
            other_saved_direction_relative_error=other_error,
            temperature_direction_cosine=float(direction_t @ other_t /
                max(np.linalg.norm(direction_t) * np.linalg.norm(other_t), np.finfo(float).tiny)),
            temperature_direction_relative_change=float(np.linalg.norm(direction_t - other_t) /
                max(np.linalg.norm(other_t), np.finfo(float).tiny)),
            normalized_schur_relative_change=float(np.linalg.norm(reduced["schur"] - other["schur"]) /
                max(np.linalg.norm(other["schur"]), np.finfo(float).tiny)),
            normalized_thermal_rhs_relative_change=float(np.linalg.norm(
                reduced["normalized_rhs"] - other["normalized_rhs"]) /
                max(np.linalg.norm(other["normalized_rhs"]), np.finfo(float).tiny)),
            row_scale_relative_max=float(np.max(np.abs(reduced["row_scale"] - other["row_scale"]) /
                np.maximum(np.abs(other["row_scale"]), np.finfo(float).tiny))),
            other_max_abs_logT=float(np.max(np.abs(other_direction[:90]))),
            other_charge_block_condition=other["charge_block_condition"],
            other_normalized_schur_condition=other["normalized_schur_condition"],
        )
        if other_error > 1.0e-7:
            raise ValueError("비교 선형화의 독립 Schur 해가 저장 방향을 재현하지 못함")

    output = dict(
        schema="LUMINA_THERMAL_JACOBIAN_INDEPENDENT_CONDITIONING_V1",
        status="PASS_SAVED_NEWTON_DIRECTION_REPRODUCED",
        scientific_goal="NOT_MET",
        scope="저장된 90점 T/ne Jacobian의 독립 SVD 및 직접 선형해 검증",
        new_physical_evaluations=0,
        new_temperature_response=0,
        new_SE=0,
        new_transport=0,
        material_update=0,
        root_converged=False,
        source_root_state=run["state"],
        selected_generation=run["selected_response"]["generation"],
        selected_response_sha256=run["selected_response"]["response_sha256"],
        saved_root_fresh_direction_max=float(block["root_fresh_direction_max"]),
        root_log_step_tolerance=float(block["root_log_step_tolerance"]),
        thermal_physical_scaled_max=float(block["thermal_physical_scaled_max"]),
        thermal_tolerance=float(block["thermal_tolerance"]),
        matrix=dict(shape=list(jacobian.shape),
                    raw_full_matrix_condition_2_descriptive_only=float(np.linalg.cond(jacobian)),
                    warning="raw full-matrix condition mixes differently scaled charge/thermal rows; do not interpret as physical ill-conditioning",
                    charge_ne_block_condition_2=reduced["charge_block_condition"],
                    row_normalized_schur_condition_2=reduced["normalized_schur_condition"],
                    row_normalized_schur_singular_value_min=float(
                        reduced["normalized_schur_singular_values"][-1]),
                    row_normalized_schur_singular_value_max=float(
                        reduced["normalized_schur_singular_values"][0])),
        independent_solve=dict(
            sign_convention="J * delta = -scaled_residual",
            saved_direction_relative_error=reproduction_error,
            relative_linear_residual=relative_linear_residual,
            max_abs_logT=float(np.max(np.abs(direct[:90]))),
            depth_max_abs_logT=max_temperature,
            max_abs_logne=float(np.max(np.abs(direct[90:]))),
            depth_max_abs_logne=max_ne,
        ),
        block_elimination=dict(method="B^{-1} charge elimination; S=D-E B^{-1}A; row_scale=max(abs(S_row)); solve row-normalized S",
                               temperature_direction_max=int(max_temperature),
                               row_normalized_schur_condition_2=reduced["normalized_schur_condition"]),
        comparison_between_centered_linearizations=comparison,
        interpretation="저장된 Newton 방향은 실제 해법의 전하 제거·행 정규화 Schur 계를 독립 재구성해 재현했다. raw full-matrix 조건수는 서로 다른 방정식 규모를 섞으므로 특이성 판정에 쓰지 않는다. 두 centered 상태에서 큰 방향이 안정적으로 남으면 수치 선형해 오류보다 미해결 열근 방향으로 분류하되, 이를 수렴/물리 근으로 보지 않는다.",
        input_sha256={"linearization.npz": sha256(args.linearization),
                      "comparison.linearization.npz": (sha256(args.compare_linearization)
                          if args.compare_linearization is not None else None),
                      "run.result.json": sha256(args.run_result)},
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": output["status"], "output": str(args.output),
                      "row_normalized_schur_condition_2": output["matrix"]["row_normalized_schur_condition_2"],
                      "saved_direction_relative_error": reproduction_error,
                      "relative_linear_residual": relative_linear_residual,
                      "max_abs_logT": output["independent_solve"]["max_abs_logT"],
                      "depth_max_abs_logT": max_temperature,
                      "comparison_between_centered_linearizations": comparison},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
