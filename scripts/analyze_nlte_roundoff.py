#!/usr/bin/env python3
"""Classify a dumped NLTE solve as solve roundoff, data sensitivity, or neither.

The dump contains five int32 values (N, n_lo, Z, ion, shell), followed by a
column-major float64 A and float64 RHS.  The production row/column balancing
is replayed exactly in float64.  An optional mpmath solve then solves those
already-dumped coefficients at higher precision; it cannot repair roundoff
which entered while A was assembled, so that distinction remains explicit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np


def read_dump(path: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    expected_header = 5 * np.dtype(np.int32).itemsize
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=np.int32, count=5)
        if header.size != 5:
            raise ValueError(f"{path}: short header ({header.size}/5)")
        n, n_lo, z, ion, shell = (int(value) for value in header)
        matrix_flat = np.fromfile(stream, dtype=np.float64, count=n * n)
        rhs = np.fromfile(stream, dtype=np.float64, count=n)
        trailing = stream.read(1)
    expected_size = expected_header + (n * n + n) * np.dtype(np.float64).itemsize
    if matrix_flat.size != n * n or rhs.size != n or trailing:
        raise ValueError(
            f"{path}: malformed payload size={path.stat().st_size}, "
            f"expected={expected_size}"
        )
    matrix = matrix_flat.reshape((n, n), order="F")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError(f"{path}: matrix or RHS contains a non-finite value")
    return {
        "N": n,
        "n_lo": n_lo,
        "Z": z,
        "ion": ion,
        "shell": shell,
    }, matrix, rhs


def production_equilibrate(
    matrix: np.ndarray, rhs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Replay pop_dense_equilibrate() sequentially, including its stop rule."""
    balanced = matrix.copy()
    scaled_rhs = rhs.copy()
    n = matrix.shape[0]
    row_scale = np.ones(n)
    column_scale = np.ones(n)
    iterations = 0
    for iteration in range(10):
        changed = False
        for index in range(n):
            row_norm = float(np.linalg.norm(balanced[index, :]))
            column_norm = float(np.linalg.norm(balanced[:, index]))
            if not (row_norm > 0.0 and column_norm > 0.0):
                raise ValueError(f"zero row/column during equilibration at {index}")
            factor = math.exp(0.5 * (math.log(column_norm) - math.log(row_norm)))
            if abs(factor - 1.0) <= 1.0e-3:
                continue
            changed = True
            balanced[index, :] *= factor
            scaled_rhs[index] *= factor
            row_scale[index] *= factor
            balanced[:, index] /= factor
            column_scale[index] /= factor
        iterations = iteration + 1
        if not changed:
            break
    return balanced, scaled_rhs, row_scale, column_scale, iterations


def componentwise_backward_error(
    matrix: np.ndarray, rhs: np.ndarray, solution: np.ndarray
) -> float:
    residual = rhs.astype(np.longdouble) - (
        matrix.astype(np.longdouble) @ solution.astype(np.longdouble)
    )
    denominator = np.abs(rhs.astype(np.longdouble)) + (
        np.abs(matrix.astype(np.longdouble)) @ np.abs(solution.astype(np.longdouble))
    )
    relative = np.divide(
        np.abs(residual), denominator, out=np.abs(residual), where=denominator > 0
    )
    return float(np.max(relative))


def solution_summary(solution: np.ndarray) -> dict[str, float | int]:
    minimum_index = int(np.argmin(solution))
    maximum_abs = float(np.max(np.abs(solution)))
    return {
        "negative_count": int(np.count_nonzero(solution < 0.0)),
        "minimum": float(solution[minimum_index]),
        "minimum_index": minimum_index,
        "maximum_abs": maximum_abs,
        "negative_relative_scale": (
            float(abs(solution[minimum_index]) / maximum_abs)
            if solution[minimum_index] < 0.0 and maximum_abs > 0.0
            else 0.0
        ),
    }


def high_precision_solve(
    balanced: np.ndarray,
    scaled_rhs: np.ndarray,
    column_scale: np.ndarray,
    dps: int,
) -> np.ndarray:
    mp.mp.dps = dps
    # mp.mpf(float) imports the binary64 value; the solve therefore tests the
    # dumped system, not an imagined higher-precision reassembly of its rates.
    matrix_mp = mp.matrix(
        [[mp.mpf(float(value)) for value in row] for row in balanced]
    )
    rhs_mp = mp.matrix([mp.mpf(float(value)) for value in scaled_rhs])
    scaled_solution = mp.lu_solve(matrix_mp, rhs_mp)
    return np.array(
        [float(scaled_solution[i] * mp.mpf(float(column_scale[i])))
         for i in range(len(column_scale))]
    )


def high_precision_generator_with_constraints(
    rate_matrix: np.ndarray,
    constraint_matrix: np.ndarray,
    rhs: np.ndarray,
    constraint_rows: np.ndarray,
    dps: int,
) -> np.ndarray:
    """Solve constraints on the exact generator implied by dumped off-diagonals.

    Unlike a high-precision solve of a binary64 repaired diagonal, this rebuilds
    each diagonal *in multiprecision* as the exact negative sum of the imported
    off-diagonal rates.  It therefore separates the generator identity from the
    final rounding of a binary64 diagonal.
    """
    mp.mp.dps = dps
    n = rate_matrix.shape[0]
    matrix_mp = mp.matrix(n, n)
    for col in range(n):
        off_diagonal = []
        for row in range(n):
            if row == col:
                continue
            value = mp.mpf(float(rate_matrix[row, col]))
            matrix_mp[row, col] = value
            off_diagonal.append(value)
        matrix_mp[col, col] = -mp.fsum(off_diagonal)
    for row in constraint_rows:
        for col in range(n):
            matrix_mp[int(row), col] = mp.mpf(
                float(constraint_matrix[int(row), col])
            )
    rhs_mp = mp.matrix([mp.mpf(float(value)) for value in rhs])
    solution_mp = mp.lu_solve(matrix_mp, rhs_mp)
    return np.array([float(solution_mp[i]) for i in range(n)])


def stage_partition_summary(
    solution: np.ndarray,
    n_lo: int,
    target_lo: float,
    target_hi: float,
) -> dict[str, float | int | None]:
    lo = float(np.sum(solution[:n_lo], dtype=np.longdouble))
    hi = float(np.sum(solution[n_lo:], dtype=np.longdouble))
    total = lo + hi
    target_total = target_lo + target_hi
    stationary_ratio = lo / hi if hi != 0.0 else None
    target_ratio = target_lo / target_hi if target_hi != 0.0 else None
    return {
        **solution_summary(solution),
        "lower_total": lo,
        "upper_total": hi,
        "total": total,
        "lower_fraction": lo / total if total != 0.0 else None,
        "target_lower_total": target_lo,
        "target_upper_total": target_hi,
        "target_lower_fraction": (
            target_lo / target_total if target_total != 0.0 else None
        ),
        "stationary_lower_to_upper_ratio": stationary_ratio,
        "target_lower_to_upper_ratio": target_ratio,
        "lower_total_over_target": lo / target_lo if target_lo != 0.0 else None,
        "ratio_over_target": (
            stationary_ratio / target_ratio
            if stationary_ratio is not None and target_ratio not in (None, 0.0)
            else None
        ),
    }


def single_total_partition_summary(
    solution: np.ndarray,
    n_lo: int,
    target_total: float,
) -> dict[str, float | int | None]:
    """Summarize a generator state constrained by one combined total."""
    lo = float(np.sum(solution[:n_lo], dtype=np.longdouble))
    hi = float(np.sum(solution[n_lo:], dtype=np.longdouble))
    total = lo + hi
    return {
        **solution_summary(solution),
        "lower_total": lo,
        "upper_total": hi,
        "total": total,
        "lower_fraction": lo / total if total != 0.0 else None,
        "target_total": target_total,
        "total_over_target": total / target_total if target_total != 0.0 else None,
        "lower_to_upper_ratio": lo / hi if hi != 0.0 else None,
    }


def generator_graph_summary(rate_matrix: np.ndarray) -> dict[str, object]:
    """Return SCC/closed-class topology for column-oriented transition rates.

    A positive off-diagonal A[row, col] is the directed transition col -> row.
    Multiple closed communicating classes make a one-total stationary state
    non-unique even when a rounded/anchored solve happens to return a vector.
    """
    n = rate_matrix.shape[0]
    adjacency = [
        [row for row in range(n)
         if row != col and rate_matrix[row, col] > 0.0]
        for col in range(n)
    ]
    reverse = [[] for _ in range(n)]
    for source, targets in enumerate(adjacency):
        for target in targets:
            reverse[target].append(source)

    index = 0
    indices = [-1] * n
    lowlink = [0] * n
    stack: list[int] = []
    on_stack = [False] * n
    components: list[list[int]] = []

    def visit(vertex: int) -> None:
        nonlocal index
        indices[vertex] = index
        lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack[vertex] = True
        for target in adjacency[vertex]:
            if indices[target] < 0:
                visit(target)
                lowlink[vertex] = min(lowlink[vertex], lowlink[target])
            elif on_stack[target]:
                lowlink[vertex] = min(lowlink[vertex], indices[target])
        if lowlink[vertex] == indices[vertex]:
            component = []
            while True:
                member = stack.pop()
                on_stack[member] = False
                component.append(member)
                if member == vertex:
                    break
            components.append(sorted(component))

    for vertex in range(n):
        if indices[vertex] < 0:
            visit(vertex)

    component_of = [-1] * n
    for component_index, component in enumerate(components):
        for member in component:
            component_of[member] = component_index
    closed = []
    for component_index, component in enumerate(components):
        if not any(
            component_of[target] != component_index
            for member in component for target in adjacency[member]
        ):
            closed.append(component_index)

    return {
        "directed_edge_count": int(sum(len(targets) for targets in adjacency)),
        "strong_component_count": len(components),
        "strong_component_sizes_desc": sorted(
            (len(component) for component in components), reverse=True
        ),
        "closed_class_count": len(closed),
        "closed_classes": [components[index] for index in closed],
        "zero_outgoing_nodes": [
            node for node, targets in enumerate(adjacency) if not targets
        ],
        "zero_incoming_nodes": [
            node for node, sources in enumerate(reverse) if not sources
        ],
    }


def locked_generator_flow_summary(
    rate_matrix: np.ndarray,
    solution: np.ndarray,
    n_lo: int,
    lower_lock_row: int,
    upper_lock_row: int,
    target_lo: float,
    target_hi: float,
) -> dict[str, float | None]:
    """Evaluate the locked population against an exact-sum generator.

    The two lock rows replaced two SE equations.  Their residuals and the sum
    over each ion block measure the artificial stage flow required to hold the
    prescribed partition away from Q's own stationary partition.
    """
    n = rate_matrix.shape[0]
    generator = rate_matrix.astype(np.longdouble)
    for col in range(n):
        generator[col, col] = -np.sum(
            np.concatenate((generator[:col, col], generator[col + 1:, col])),
            dtype=np.longdouble,
        )
    residual = generator @ solution.astype(np.longdouble)
    lower_net = np.sum(residual[:n_lo], dtype=np.longdouble)
    upper_net = np.sum(residual[n_lo:], dtype=np.longdouble)
    return {
        "lower_lock_row_residual": float(residual[lower_lock_row]),
        "upper_lock_row_residual": float(residual[upper_lock_row]),
        "lower_stage_net_flow": float(lower_net),
        "upper_stage_net_flow": float(upper_net),
        "total_net_flow": float(lower_net + upper_net),
        "lower_net_over_target_per_s": (
            float(lower_net / np.longdouble(target_lo))
            if target_lo != 0.0 else None
        ),
        "upper_net_over_target_per_s": (
            float(upper_net / np.longdouble(target_hi))
            if target_hi != 0.0 else None
        ),
        "max_abs_level_residual": float(np.max(np.abs(residual))),
    }


def analyze(path: Path, dps: int, skip_mp: bool) -> dict[str, object]:
    header, matrix, rhs = read_dump(path)
    balanced, scaled_rhs, row_scale, column_scale, iterations = (
        production_equilibrate(matrix, rhs)
    )
    singular = np.linalg.svd(matrix, compute_uv=False)
    singular_balanced = np.linalg.svd(balanced, compute_uv=False)
    condition_2 = float(singular[0] / singular[-1])
    condition_2_balanced = float(singular_balanced[0] / singular_balanced[-1])
    condition_inf = float(np.linalg.cond(matrix, np.inf))
    condition_inf_balanced = float(np.linalg.cond(balanced, np.inf))

    scaled_solution = np.linalg.solve(balanced, scaled_rhs)
    solution = column_scale * scaled_solution
    double_summary = solution_summary(solution)
    backward_error = componentwise_backward_error(matrix, rhs, solution)

    n = matrix.shape[0]
    gamma_n = (n * np.finfo(np.float64).eps) / (
        1.0 - n * np.finfo(np.float64).eps
    )
    historical_bound_raw = condition_inf * gamma_n * double_summary["maximum_abs"]
    historical_bound_balanced_y = (
        condition_inf_balanced * gamma_n * float(np.max(np.abs(scaled_solution)))
    )

    terms = np.abs(matrix) @ np.abs(solution)
    lhs = matrix.astype(np.longdouble) @ solution.astype(np.longdouble)
    lhs_abs = np.abs(lhs.astype(np.float64))
    cancellation = np.divide(
        terms, lhs_abs, out=np.full(n, np.inf), where=lhs_abs > 0.0
    )
    zero_rhs = rhs == 0.0
    finite_zero_rhs = cancellation[zero_rhs & np.isfinite(cancellation)]

    result: dict[str, object] = {
        "path": str(path.resolve()),
        "header": header,
        "equilibration_iterations": iterations,
        "condition_2_raw": condition_2,
        "condition_2_balanced": condition_2_balanced,
        "condition_inf_raw": condition_inf,
        "condition_inf_balanced": condition_inf_balanced,
        "gamma_N": float(gamma_n),
        "historical_negative_bound_raw": float(historical_bound_raw),
        "historical_negative_bound_balanced_y": float(
            historical_bound_balanced_y
        ),
        "double": {
            **double_summary,
            "componentwise_backward_error": backward_error,
        },
        "row_cancellation": {
            "zero_rhs_rows": int(np.count_nonzero(zero_rhs)),
            "max_finite_zero_rhs": (
                float(np.max(finite_zero_rhs)) if finite_zero_rhs.size else None
            ),
            "median_finite_zero_rhs": (
                float(np.median(finite_zero_rhs)) if finite_zero_rhs.size else None
            ),
            "infinite_zero_rhs_rows": int(
                np.count_nonzero(zero_rhs & ~np.isfinite(cancellation))
            ),
        },
    }

    if not skip_mp:
        mp_solution = high_precision_solve(
            balanced, scaled_rhs, column_scale, dps
        )
        mp_summary = solution_summary(mp_solution)
        difference = np.abs(solution - mp_solution)
        result["high_precision"] = {
            "decimal_digits": dps,
            **mp_summary,
            "max_abs_difference_from_double": float(np.max(difference)),
            "max_relative_difference_from_double": float(
                np.max(difference) / max(float(np.max(np.abs(mp_solution))), 1e-300)
            ),
            "componentwise_backward_error_after_float64_projection": (
                componentwise_backward_error(matrix, rhs, mp_solution)
            ),
        }

    return result


def analyze_prelock_pair(
    prelock_path: Path,
    postlock_path: Path,
    dps: int,
    skip_mp: bool,
) -> dict[str, object]:
    """Restore exact rate-generator column sums, then reapply the lock rows."""
    pre_header, rate_matrix, rate_rhs = read_dump(prelock_path)
    post_header, post_matrix, post_rhs = read_dump(postlock_path)
    if pre_header != post_header:
        raise ValueError("prelock/postlock headers differ")
    n = rate_matrix.shape[0]
    changed = np.any(rate_matrix != post_matrix, axis=1) | (rate_rhs != post_rhs)
    changed_rows = np.flatnonzero(changed)
    off_diagonal = ~np.eye(n, dtype=bool)
    negative_off_diagonal = int(np.count_nonzero(
        rate_matrix[off_diagonal] < 0.0
    ))
    prelock_rhs_nonzero = int(np.count_nonzero(rate_rhs))
    if prelock_rhs_nonzero or negative_off_diagonal:
        raise ValueError(
            "column-sum repair is valid only for a homogeneous rate generator; "
            f"prelock_rhs_nonzero={prelock_rhs_nonzero}, "
            f"negative_off_diagonal={negative_off_diagonal}"
        )

    total_lock = np.ones(n)
    total_rows = [
        int(row) for row in changed_rows
        if np.array_equal(post_matrix[row], total_lock)
    ]
    if changed_rows.size == 1 and len(total_rows) == 1:
        target_total = float(post_rhs[total_rows[0]])
        if not math.isfinite(target_total) or target_total < 0.0:
            raise ValueError(f"invalid combined target total={target_total}")

        column_sum = np.array([
            math.fsum(float(value) for value in rate_matrix[:, col])
            for col in range(n)
        ])
        column_l1 = np.sum(np.abs(rate_matrix), axis=0)
        column_relative = np.divide(
            np.abs(column_sum), column_l1,
            out=np.zeros(n), where=column_l1 > 0.0,
        )
        repaired_rate = rate_matrix.copy()
        diagonal_delta = np.empty(n)
        for col in range(n):
            repaired_diagonal = -math.fsum(
                float(rate_matrix[row, col])
                for row in range(n) if row != col
            )
            diagonal_delta[col] = repaired_diagonal - rate_matrix[col, col]
            repaired_rate[col, col] = repaired_diagonal

        normalization_row = total_rows[0]
        if target_total == 0.0:
            # Under the nonnegative-population contract, sum_i n_i=0 uniquely
            # implies n_i=0.  This is an exact boundary, not a singular
            # stationary-distribution problem; do not invent a tolerance or
            # ask LU/GTH to recover an intentionally zero normalization.
            return {
                "constraint_mode": "single_combined_total_exact_zero",
                "prelock_path": str(prelock_path.resolve()),
                "postlock_path": str(postlock_path.resolve()),
                "header": pre_header,
                "changed_rows": [normalization_row],
                "changed_row_count": 1,
                "normalization_row": normalization_row,
                "target_total": target_total,
                "prelock_rhs_nonzero": prelock_rhs_nonzero,
                "prelock_negative_off_diagonal": negative_off_diagonal,
                "generator_graph": generator_graph_summary(rate_matrix),
                "raw_rate_column_sum": {
                    "max_abs": float(np.max(np.abs(column_sum))),
                    "max_relative_to_column_l1": float(np.max(column_relative)),
                    "median_relative_to_column_l1": float(np.median(column_relative)),
                    "nonzero_columns": int(np.count_nonzero(column_sum)),
                },
                "diagonal_repair": {
                    "max_abs": float(np.max(np.abs(diagonal_delta))),
                    "max_relative_to_column_l1": float(np.max(np.divide(
                        np.abs(diagonal_delta), column_l1,
                        out=np.zeros(n), where=column_l1 > 0.0,
                    ))),
                    "nonzero_columns": int(np.count_nonzero(diagonal_delta)),
                },
                "exact_zero_solution": {
                    "reason": (
                        "nonnegative populations plus exact combined total zero"
                    ),
                    "solve_attempted": False,
                    "negative_count": 0,
                    "minimum": 0.0,
                    "maximum_abs": 0.0,
                    "lower_total": 0.0,
                    "upper_total": 0.0,
                    "total": 0.0,
                },
            }
        repaired_post = repaired_rate.copy()
        repaired_post[normalization_row, :] = post_matrix[normalization_row, :]
        repaired_rhs = rate_rhs.copy()
        repaired_rhs[normalization_row] = target_total
        balanced, scaled_rhs, _row_scale, column_scale, iterations = (
            production_equilibrate(repaired_post, repaired_rhs)
        )
        scaled_solution = np.linalg.solve(balanced, scaled_rhs)
        solution = column_scale * scaled_solution

        uniformization_lambda = float(np.max(-np.diag(repaired_rate)))
        if not uniformization_lambda > 0.0:
            raise ValueError(
                f"generator has no positive exit rate: lambda={uniformization_lambda}"
            )
        uniformization = np.eye(n) + repaired_rate / uniformization_lambda
        uniformization_fixed_point = uniformization @ solution - solution

        result: dict[str, object] = {
            "constraint_mode": "single_combined_total",
            "prelock_path": str(prelock_path.resolve()),
            "postlock_path": str(postlock_path.resolve()),
            "header": pre_header,
            "changed_rows": [normalization_row],
            "changed_row_count": 1,
            "normalization_row": normalization_row,
            "target_total": target_total,
            "prelock_rhs_nonzero": prelock_rhs_nonzero,
            "prelock_negative_off_diagonal": negative_off_diagonal,
            "generator_graph": generator_graph_summary(rate_matrix),
            "raw_rate_column_sum": {
                "max_abs": float(np.max(np.abs(column_sum))),
                "max_relative_to_column_l1": float(np.max(column_relative)),
                "median_relative_to_column_l1": float(np.median(column_relative)),
                "nonzero_columns": int(np.count_nonzero(column_sum)),
            },
            "diagonal_repair": {
                "max_abs": float(np.max(np.abs(diagonal_delta))),
                "max_relative_to_column_l1": float(np.max(np.divide(
                    np.abs(diagonal_delta), column_l1,
                    out=np.zeros(n), where=column_l1 > 0.0,
                ))),
                "nonzero_columns": int(np.count_nonzero(diagonal_delta)),
            },
            "repaired_post_equilibration_iterations": iterations,
            "repaired_double": {
                **single_total_partition_summary(
                    solution, pre_header["n_lo"], target_total
                ),
                "componentwise_backward_error": componentwise_backward_error(
                    repaired_post, repaired_rhs, solution
                ),
            },
            "uniformization": {
                "lambda": uniformization_lambda,
                "negative_entry_count": int(np.count_nonzero(uniformization < 0.0)),
                "minimum_entry": float(np.min(uniformization)),
                "max_abs_column_sum_error": float(np.max(np.abs(
                    np.sum(uniformization, axis=0) - 1.0
                ))),
                "max_abs_fixed_point_error": float(np.max(
                    np.abs(uniformization_fixed_point)
                )),
            },
        }
        if not skip_mp:
            mp_solution = high_precision_solve(
                balanced, scaled_rhs, column_scale, dps
            )
            result["repaired_high_precision"] = {
                "decimal_digits": dps,
                **single_total_partition_summary(
                    mp_solution, pre_header["n_lo"], target_total
                ),
                "max_abs_difference_from_double": float(
                    np.max(np.abs(solution - mp_solution))
                ),
                "componentwise_backward_error_after_float64_projection": (
                    componentwise_backward_error(
                        repaired_post, repaired_rhs, mp_solution
                    )
                ),
            }
            exact_solution = high_precision_generator_with_constraints(
                rate_matrix, post_matrix, repaired_rhs, changed_rows, dps
            )
            result["repaired_exact_generator_high_precision"] = {
                "decimal_digits": dps,
                **single_total_partition_summary(
                    exact_solution, pre_header["n_lo"], target_total
                ),
                "max_abs_difference_from_repaired_double": float(
                    np.max(np.abs(solution - exact_solution))
                ),
            }
        return result

    lower_lock = np.zeros(n)
    lower_lock[:pre_header["n_lo"]] = 1.0
    upper_lock = np.zeros(n)
    upper_lock[pre_header["n_lo"]:] = 1.0
    lower_rows = [
        int(row) for row in changed_rows
        if np.array_equal(post_matrix[row], lower_lock)
    ]
    upper_rows = [
        int(row) for row in changed_rows
        if np.array_equal(post_matrix[row], upper_lock)
    ]
    if len(lower_rows) != 1 or len(upper_rows) != 1 or changed_rows.size != 2:
        raise ValueError(
            "expected exactly one indicator lock row per ion stage; "
            f"changed_rows={changed_rows.tolist()}, lower_rows={lower_rows}, "
            f"upper_rows={upper_rows}"
        )
    target_lo = float(post_rhs[lower_rows[0]])
    target_hi = float(post_rhs[upper_rows[0]])
    target_total = target_lo + target_hi
    if not (target_lo >= 0.0 and target_hi >= 0.0 and target_total > 0.0):
        raise ValueError(
            f"invalid stage targets lower={target_lo}, upper={target_hi}"
        )

    column_sum = np.array(
        [math.fsum(float(value) for value in rate_matrix[:, col])
         for col in range(n)]
    )
    column_l1 = np.sum(np.abs(rate_matrix), axis=0)
    column_relative = np.divide(
        np.abs(column_sum), column_l1,
        out=np.zeros(n), where=column_l1 > 0.0,
    )

    repaired_rate = rate_matrix.copy()
    diagonal_delta = np.empty(n)
    for col in range(n):
        repaired_diagonal = -math.fsum(
            float(rate_matrix[row, col]) for row in range(n) if row != col
        )
        diagonal_delta[col] = repaired_diagonal - rate_matrix[col, col]
        repaired_rate[col, col] = repaired_diagonal

    repaired_post = repaired_rate.copy()
    repaired_post[changed_rows, :] = post_matrix[changed_rows, :]
    repaired_rhs = rate_rhs.copy()
    repaired_rhs[changed_rows] = post_rhs[changed_rows]
    balanced, scaled_rhs, _row_scale, column_scale, iterations = (
        production_equilibrate(repaired_post, repaired_rhs)
    )
    scaled_solution = np.linalg.solve(balanced, scaled_rhs)
    solution = column_scale * scaled_solution

    # One total-element normalization leaves Q free to choose its own ion-stage
    # partition.  This is the stationary state shared by exp(tQ) and
    # uniformization; it is a diagnostic and cannot honor two independent locks.
    normalization_row = int(changed_rows[-1])
    stationary_matrix = repaired_rate.copy()
    stationary_matrix[normalization_row, :] = 1.0
    stationary_rhs = np.zeros(n)
    stationary_rhs[normalization_row] = target_total
    (
        stationary_balanced,
        stationary_scaled_rhs,
        _stationary_row_scale,
        stationary_column_scale,
        stationary_iterations,
    ) = production_equilibrate(stationary_matrix, stationary_rhs)
    stationary_scaled_solution = np.linalg.solve(
        stationary_balanced, stationary_scaled_rhs
    )
    stationary_solution = stationary_column_scale * stationary_scaled_solution

    uniformization_lambda = float(np.max(-np.diag(repaired_rate)))
    if not uniformization_lambda > 0.0:
        raise ValueError(
            f"generator has no positive exit rate: lambda={uniformization_lambda}"
        )
    uniformization = np.eye(n) + repaired_rate / uniformization_lambda
    uniformization_column_sum_error = np.sum(uniformization, axis=0) - 1.0
    uniformization_fixed_point = (
        uniformization @ stationary_solution - stationary_solution
    )

    result: dict[str, object] = {
        "prelock_path": str(prelock_path.resolve()),
        "postlock_path": str(postlock_path.resolve()),
        "header": pre_header,
        "changed_rows": [int(row) for row in changed_rows],
        "changed_row_count": int(changed_rows.size),
        "stage_lock_rows": {
            "lower": lower_rows[0],
            "upper": upper_rows[0],
        },
        "prelock_rhs_nonzero": prelock_rhs_nonzero,
        "prelock_negative_off_diagonal": negative_off_diagonal,
        "raw_rate_column_sum": {
            "max_abs": float(np.max(np.abs(column_sum))),
            "max_relative_to_column_l1": float(np.max(column_relative)),
            "median_relative_to_column_l1": float(np.median(column_relative)),
            "nonzero_columns": int(np.count_nonzero(column_sum)),
        },
        "diagonal_repair": {
            "max_abs": float(np.max(np.abs(diagonal_delta))),
            "max_relative_to_column_l1": float(np.max(np.divide(
                np.abs(diagonal_delta), column_l1,
                out=np.zeros(n), where=column_l1 > 0.0,
            ))),
            "nonzero_columns": int(np.count_nonzero(diagonal_delta)),
        },
        "repaired_post_equilibration_iterations": iterations,
        "repaired_double": {
            **solution_summary(solution),
            "componentwise_backward_error": componentwise_backward_error(
                repaired_post, repaired_rhs, solution
            ),
        },
        "locked_generator_flow_double": locked_generator_flow_summary(
            rate_matrix,
            solution,
            pre_header["n_lo"],
            lower_rows[0],
            upper_rows[0],
            target_lo,
            target_hi,
        ),
        "unconstrained_generator_stationary": {
            "meaning": (
                "single-total stationary state shared by exp(tQ) and "
                "uniformization; diagnostic only"
            ),
            "normalization_row": normalization_row,
            "equilibration_iterations": stationary_iterations,
            "double": stage_partition_summary(
                stationary_solution,
                pre_header["n_lo"],
                target_lo,
                target_hi,
            ),
            "uniformization": {
                "lambda": uniformization_lambda,
                "negative_entry_count": int(np.count_nonzero(uniformization < 0.0)),
                "minimum_entry": float(np.min(uniformization)),
                "max_abs_column_sum_error": float(
                    np.max(np.abs(uniformization_column_sum_error))
                ),
                "max_abs_fixed_point_error": float(
                    np.max(np.abs(uniformization_fixed_point))
                ),
            },
        },
    }
    if not skip_mp:
        mp_solution = high_precision_solve(
            balanced, scaled_rhs, column_scale, dps
        )
        result["repaired_high_precision"] = {
            "decimal_digits": dps,
            **solution_summary(mp_solution),
            "max_abs_difference_from_double": float(
                np.max(np.abs(solution - mp_solution))
            ),
            "componentwise_backward_error_after_float64_projection": (
                componentwise_backward_error(
                    repaired_post, repaired_rhs, mp_solution
                )
            ),
        }
        exact_generator_solution = high_precision_generator_with_constraints(
            rate_matrix,
            post_matrix,
            repaired_rhs,
            changed_rows,
            dps,
        )
        result["repaired_exact_generator_high_precision"] = {
            "decimal_digits": dps,
            **solution_summary(exact_generator_solution),
            "max_abs_difference_from_repaired_double": float(
                np.max(np.abs(solution - exact_generator_solution))
            ),
        }
        result["locked_generator_flow_exact_generator_high_precision"] = (
            locked_generator_flow_summary(
                rate_matrix,
                exact_generator_solution,
                pre_header["n_lo"],
                lower_rows[0],
                upper_rows[0],
                target_lo,
                target_hi,
            )
        )
        stationary_exact_solution = high_precision_generator_with_constraints(
            rate_matrix,
            stationary_matrix,
            stationary_rhs,
            np.array([normalization_row]),
            dps,
        )
        result["unconstrained_generator_stationary"][
            "exact_generator_high_precision"
        ] = {
            "decimal_digits": dps,
            **stage_partition_summary(
                stationary_exact_solution,
                pre_header["n_lo"],
                target_lo,
                target_hi,
            ),
            "max_abs_difference_from_double": float(
                np.max(np.abs(
                    stationary_solution - stationary_exact_solution
                ))
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path, nargs="+")
    parser.add_argument("--mp-dps", type=int, default=80)
    parser.add_argument("--skip-mp", action="store_true")
    parser.add_argument(
        "--prelock", type=Path,
        help="raw rate-matrix dump paired with exactly one post-lock dump",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.mp_dps < 32:
        parser.error("--mp-dps must be at least 32")

    if args.prelock and len(args.dump) != 1:
        parser.error("--prelock requires exactly one post-lock dump")
    results = [analyze(path, args.mp_dps, args.skip_mp) for path in args.dump]
    if args.prelock:
        results[0]["prelock_repair"] = analyze_prelock_pair(
            args.prelock, args.dump[0], args.mp_dps, args.skip_mp
        )
    rendered = json.dumps(results, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
