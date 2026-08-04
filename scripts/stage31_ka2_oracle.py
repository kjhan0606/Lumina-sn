"""Independent singularity-subtracted Nyström oracle for Stage 3.1 KA2."""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
from scipy.special import exp1, roots_legendre
from scipy.interpolate import CubicSpline


def _q_singular_mp(radius: float) -> float:
    r = mp.mpf(str(radius))

    def primitive(value: mp.mpf) -> mp.mpf:
        if value == 0:
            return mp.mpf("0")
        return value * mp.e1(value) - mp.exp(-value) + 1

    return float(primitive(r) + primitive(1 - r))


def _operator_matrix(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    abscissa, weight = roots_legendre(order)
    radius = 0.5 * (abscissa + 1.0)
    weight = 0.5 * weight
    difference = np.abs(radius[:, None] - radius[None, :])
    radial_sum = radius[:, None] + radius[None, :]
    kernel_difference = exp1(difference)
    np.fill_diagonal(kernel_difference, 0.0)
    kernel_sum = exp1(radial_sum)
    weighted_radius = weight * radius
    operator = (kernel_difference - kernel_sum) * weighted_radius[None, :]
    q_value = np.fromiter((_q_singular_mp(float(r)) for r in radius),
                          dtype=np.float64, count=order)
    diagonal_correction = radius * (q_value - kernel_difference @ weight)
    operator[np.diag_indices(order)] += diagonal_correction
    operator *= (0.5 / radius)[:, None]
    return radius, weight, operator


def _evaluate(radius: np.ndarray, weight: np.ndarray, source: np.ndarray,
              targets: np.ndarray) -> np.ndarray:
    f_value = radius * source
    values = np.empty_like(targets)
    for i, target in enumerate(targets):
        source_target = float(np.interp(target, radius, source))
        f_target = target * source_target
        difference = np.abs(target - radius)
        first = exp1(difference)
        if np.any(difference == 0.0):
            first[difference == 0.0] = 0.0
        integrand = (f_value - f_target) * first - f_value * exp1(target + radius)
        q_value = _q_singular_mp(float(target))
        values[i] = (np.dot(weight, integrand) + f_target * q_value) / (2.0 * target)
    return values


def solve(order: int, targets: list[float], tolerance: float = 2.0e-14) -> dict:
    mp.mp.dps = 80
    radius, weight, operator = _operator_matrix(order)
    epsilon = 0.2
    j_value = np.zeros(order, dtype=np.float64)
    source_residual = math.inf
    iterations = 0
    for iterations in range(1, 501):
        next_j = operator @ (epsilon + (1.0 - epsilon) * j_value)
        source_residual = float(np.max(np.abs(next_j - j_value)) /
                                (np.max(np.abs(next_j)) + np.finfo(float).tiny))
        j_value = next_j
        if source_residual <= tolerance:
            break
    else:
        raise RuntimeError(f"Nyström oracle did not converge at Nref={order}")
    target_array = np.asarray(targets, dtype=np.float64)
    evaluated = CubicSpline(radius, j_value, bc_type="natural")(target_array)
    del operator
    return {
        "order": order,
        "mpmath_dps": mp.mp.dps,
        "singularity_subtraction": "analytic r=r' logarithmic subtraction",
        "matrix_storage": "binary64 with 80-digit singular primitives",
        "iterations": iterations,
        "source_residual": source_residual,
        "targets": target_array.tolist(),
        "J": evaluated.tolist(),
    }


def relative_difference(left: list[float], right: list[float]) -> float:
    a = np.asarray(left)
    b = np.asarray(right)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
