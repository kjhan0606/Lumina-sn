#!/usr/bin/env python3
"""Stage 31 KA2 full-80-digit Nyström oracle.

This is a preparation/cluster script.  The preregistered production invocation
uses ``--nref 2048 4096``.  A single small order (for example ``--nref 64``)
is useful only as an arithmetic-path smoke test and can never report an
acceptance PASS.

Every numerical stage before JSON serialization uses mpmath ``mpf`` at 80
decimal digits: Gauss--Legendre construction, E1 kernel assembly, logarithmic
singularity subtraction, dense-operator GMRES, interpolation, and comparison
norms.  NumPy and SciPy are intentionally not imported.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import multiprocessing as multiprocessing
import os
import pathlib
import pickle
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Sequence

import mpmath as mp


DPS = 80
EPSILON_TEXT = "0.2"
SCATTERING_TEXT = "0.8"
SOLVE_TOLERANCE_TEXT = "1e-60"
MAX_GMRES_ITERATIONS = 80
PRODUCTION_NREF = (2048, 4096)
SCHEMA_VERSION = "s31-ka2-oracle-hp-v1"

mp.mp.dps = DPS
EPSILON = mp.mpf(EPSILON_TEXT)
SCATTERING = mp.mpf(SCATTERING_TEXT)
SOLVE_TOLERANCE = mp.mpf(SOLVE_TOLERANCE_TEXT)


def decimal(value: mp.mpf, digits: int = DPS) -> str:
    """Serialize an mpf without first lowering it to binary64."""
    return mp.nstr(value, digits, strip_zeros=False)


def timestamped(message: str) -> str:
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    return f"[{stamp}] {message}"


def atomic_pickle(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_pickle(path: pathlib.Path) -> object:
    with path.open("rb") as stream:
        return pickle.load(stream)


def atomic_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def target_union() -> tuple[mp.mpf, ...]:
    """All production radii plus the 512 preregistered shell centres."""
    values = {mp.mpf(index) / 512 for index in range(513)}
    values.update(mp.mpf(2 * index + 1) / 1024 for index in range(512))
    return tuple(sorted(values))


def canonical_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def legendre_value_derivative(order: int, x_value: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    previous = mp.mpf(1)
    current = x_value
    if order == 0:
        return previous, mp.mpf(0)
    if order == 1:
        return current, mp.mpf(1)
    for degree in range(2, order + 1):
        following = ((2 * degree - 1) * x_value * current
                     - (degree - 1) * previous) / degree
        previous, current = current, following
    derivative = order * (x_value * current - previous) / (x_value * x_value - 1)
    return current, derivative


def positive_legendre_root(task: tuple[int, int]) -> tuple[int, str, str, str]:
    order, index = task
    mp.mp.dps = DPS
    x_value = mp.cos(mp.pi * (mp.mpf(index) - mp.mpf("0.25"))
                     / (mp.mpf(order) + mp.mpf("0.5")))
    for _ in range(20):
        polynomial, derivative = legendre_value_derivative(order, x_value)
        correction = polynomial / derivative
        x_value -= correction
        if abs(correction) <= mp.mpf("1e-75"):
            break
    else:
        raise RuntimeError(f"Legendre Newton failure: N={order}, root={index}")
    polynomial, derivative = legendre_value_derivative(order, x_value)
    weight = 2 / ((1 - x_value * x_value) * derivative * derivative)
    return index, decimal(x_value), decimal(weight), decimal(abs(polynomial))


@dataclass(frozen=True)
class Quadrature:
    radius: tuple[mp.mpf, ...]
    weight: tuple[mp.mpf, ...]
    max_root_residual: mp.mpf
    weight_sum_error: mp.mpf
    symmetry_error: mp.mpf


def gauss_legendre_unit(order: int, workers: int,
                        log: Callable[[str], None]) -> Quadrature:
    if order <= 0 or order % 2:
        raise ValueError("--nref entries must be positive even integers")
    mp.mp.dps = DPS
    started = time.monotonic()
    log(f"Nref={order}: Gauss-Legendre nodes/weights at {DPS} dps")
    tasks = [(order, index) for index in range(1, order // 2 + 1)]
    context = multiprocessing.get_context("fork")
    with context.Pool(processes=min(workers, len(tasks))) as pool:
        positive = pool.map(positive_legendre_root, tasks, chunksize=1)
    positive.sort(key=lambda item: mp.mpf(item[1]))
    positive_x = [mp.mpf(item[1]) for item in positive]
    positive_w = [mp.mpf(item[2]) for item in positive]
    abscissa = [-value for value in reversed(positive_x)] + positive_x
    weights = list(reversed(positive_w)) + positive_w
    radius = tuple((value + 1) / 2 for value in abscissa)
    unit_weight = tuple(value / 2 for value in weights)
    root_residual = max(mp.mpf(item[3]) for item in positive)
    weight_error = abs(mp.fsum(unit_weight) - 1)
    symmetry_error = max(abs(radius[i] + radius[-1-i] - 1)
                         for i in range(order))
    log(f"Nref={order}: quadrature ready in {time.monotonic()-started:.3f}s; "
        f"max|P_N|={decimal(root_residual, 12)}, "
        f"|sum(w)-1|={decimal(weight_error, 12)}")
    return Quadrature(radius, unit_weight, root_residual,
                      weight_error, symmetry_error)


def singular_integral(radius: mp.mpf) -> mp.mpf:
    """Integral of E1(|r-r'|) over r' in [0,1]."""
    def primitive(value: mp.mpf) -> mp.mpf:
        if value == 0:
            return mp.mpf(0)
        return value * mp.e1(value) - mp.exp(-value) + 1
    return primitive(radius) + primitive(1 - radius)


def matrix_worker(connection: object, worker_index: int, start: int, stop: int,
                  radius: Sequence[mp.mpf], weight: Sequence[mp.mpf]) -> None:
    """Retain one block of the dense, singularity-subtracted Lambda matrix."""
    try:
        mp.mp.dps = DPS
        order = len(radius)
        half = mp.mpf("0.5")
        rows: list[tuple[mp.mpf, ...]] = []
        started = time.monotonic()
        for row_index in range(start, stop):
            row_radius = radius[row_index]
            q_value = singular_integral(row_radius)
            difference_weight_sum = mp.mpf(0)
            row: list[mp.mpf] = []
            scale = half / row_radius
            for column_index in range(order):
                column_radius = radius[column_index]
                sum_kernel = mp.e1(row_radius + column_radius)
                if row_index == column_index:
                    difference_kernel = mp.mpf(0)
                else:
                    difference_kernel = mp.e1(abs(row_radius - column_radius))
                    difference_weight_sum += weight[column_index] * difference_kernel
                row.append((difference_kernel - sum_kernel)
                           * weight[column_index] * column_radius * scale)
            # Analytic logarithmic subtraction adds f(r_i) * integral E1.
            row[row_index] += half * (q_value - difference_weight_sum)
            rows.append(tuple(row))
        connection.send(("ready", worker_index, start, stop,
                         time.monotonic() - started))
        while True:
            command = connection.recv()
            if command[0] == "stop":
                connection.send(("stopped", worker_index))
                return
            if command[0] != "apply":
                raise RuntimeError(f"unknown worker command: {command[0]!r}")
            vector = command[1]
            connection.send(("result", worker_index, start,
                             [mp.fdot(row, vector) for row in rows]))
    except BaseException:
        connection.send(("error", worker_index, traceback.format_exc()))


class DenseOperator:
    """Block-distributed storage for the complete 80-digit dense operator."""

    def __init__(self, quadrature: Quadrature, workers: int,
                 log: Callable[[str], None]) -> None:
        self.order = len(quadrature.radius)
        self.apply_count = 0
        self._processes: list[multiprocessing.Process] = []
        self._connections: list[object] = []
        worker_count = min(workers, self.order)
        block_size = math.ceil(self.order / worker_count)
        context = multiprocessing.get_context("fork")
        log(f"Nref={self.order}: assembling full dense mpf operator in "
            f"{worker_count} retained blocks")
        for worker_index, start in enumerate(range(0, self.order, block_size)):
            stop = min(start + block_size, self.order)
            parent, child = context.Pipe()
            process = context.Process(
                target=matrix_worker,
                args=(child, worker_index, start, stop,
                      quadrature.radius, quadrature.weight),
            )
            process.start()
            child.close()
            self._processes.append(process)
            self._connections.append(parent)
        pending = set(range(len(self._connections)))
        while pending:
            ready = multiprocessing.connection.wait(
                [self._connections[index] for index in pending], timeout=30.0)
            if not ready:
                log(f"Nref={self.order}: assembly checkpoint "
                    f"{len(self._connections)-len(pending)}/{len(self._connections)} blocks")
                continue
            for connection in ready:
                message = connection.recv()
                if message[0] == "error":
                    raise RuntimeError(message[2])
                _, worker_index, start, stop, elapsed = message
                pending.remove(worker_index)
                log(f"Nref={self.order}: block {worker_index+1}/"
                    f"{len(self._connections)} rows [{start},{stop}) ready "
                    f"({elapsed:.3f}s)")

    def apply(self, vector: Sequence[mp.mpf]) -> list[mp.mpf]:
        if len(vector) != self.order:
            raise ValueError("dense operator/vector dimension mismatch")
        payload = tuple(vector)
        for connection in self._connections:
            connection.send(("apply", payload))
        result = [mp.mpf(0)] * self.order
        for connection in self._connections:
            message = connection.recv()
            if message[0] == "error":
                raise RuntimeError(message[2])
            _, _, start, block = message
            result[start:start+len(block)] = block
        self.apply_count += 1
        return result

    def close(self) -> None:
        for connection in self._connections:
            connection.send(("stop",))
        for connection in self._connections:
            try:
                connection.recv()
            finally:
                connection.close()
        for process in self._processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"matrix worker exit status {process.exitcode}")

    def __enter__(self) -> "DenseOperator":
        return self

    def __exit__(self, exc_type: object, exc_value: object, exc_tb: object) -> None:
        self.close()


def norm(vector: Sequence[mp.mpf]) -> mp.mpf:
    return mp.sqrt(mp.fdot(vector, vector))


def axpy(left: Sequence[mp.mpf], scale: mp.mpf,
         right: Sequence[mp.mpf]) -> list[mp.mpf]:
    return [a + scale * b for a, b in zip(left, right)]


@dataclass(frozen=True)
class SolveResult:
    j_value: tuple[mp.mpf, ...]
    gmres_iterations: int
    gmres_reported_residual: mp.mpf
    linear_residual: mp.mpf
    source_residual: mp.mpf
    operator_matvecs: int


def gmres(operator: DenseOperator, rhs: Sequence[mp.mpf],
          log: Callable[[str], None]) -> tuple[list[mp.mpf], int, mp.mpf]:
    """Unrestarted, twice-reorthogonalized MGS GMRES in mp arithmetic."""
    beta = norm(rhs)
    basis = [[value / beta for value in rhs]]
    hessenberg = [[mp.mpf(0) for _ in range(MAX_GMRES_ITERATIONS)]
                  for _ in range(MAX_GMRES_ITERATIONS + 1)]
    cosines = [mp.mpf(0)] * MAX_GMRES_ITERATIONS
    sines = [mp.mpf(0)] * MAX_GMRES_ITERATIONS
    transformed_rhs = [mp.mpf(0)] * (MAX_GMRES_ITERATIONS + 1)
    transformed_rhs[0] = beta
    relative_residual = mp.mpf(1)
    used = 0
    for column in range(MAX_GMRES_ITERATIONS):
        applied = operator.apply(basis[column])
        work = [basis[column][i] - SCATTERING * applied[i]
                for i in range(operator.order)]
        for _ in range(2):
            for row in range(column + 1):
                projection = mp.fdot(basis[row], work)
                hessenberg[row][column] += projection
                work = axpy(work, -projection, basis[row])
        subdiagonal = norm(work)
        hessenberg[column+1][column] = subdiagonal
        if subdiagonal != 0:
            basis.append([value / subdiagonal for value in work])
        for row in range(column):
            top = (cosines[row] * hessenberg[row][column]
                   + sines[row] * hessenberg[row+1][column])
            bottom = (-sines[row] * hessenberg[row][column]
                      + cosines[row] * hessenberg[row+1][column])
            hessenberg[row][column], hessenberg[row+1][column] = top, bottom
        diagonal = hessenberg[column][column]
        subdiagonal = hessenberg[column+1][column]
        magnitude = mp.sqrt(diagonal * diagonal + subdiagonal * subdiagonal)
        cosines[column] = diagonal / magnitude if magnitude else mp.mpf(1)
        sines[column] = subdiagonal / magnitude if magnitude else mp.mpf(0)
        hessenberg[column][column] = (cosines[column] * diagonal
                                      + sines[column] * subdiagonal)
        hessenberg[column+1][column] = mp.mpf(0)
        transformed_rhs[column+1] = -sines[column] * transformed_rhs[column]
        transformed_rhs[column] *= cosines[column]
        relative_residual = abs(transformed_rhs[column+1]) / beta
        used = column + 1
        log(f"Nref={operator.order}: GMRES {used}, relres="
            f"{decimal(relative_residual, 14)}, matvecs={operator.apply_count}")
        if relative_residual <= SOLVE_TOLERANCE:
            break
    else:
        raise RuntimeError("GMRES iteration limit reached")
    if relative_residual > SOLVE_TOLERANCE:
        raise RuntimeError(f"GMRES stopped above {SOLVE_TOLERANCE_TEXT}")
    coefficients = [mp.mpf(0)] * used
    for row in range(used - 1, -1, -1):
        tail = mp.fsum(hessenberg[row][column] * coefficients[column]
                       for column in range(row + 1, used))
        coefficients[row] = ((transformed_rhs[row] - tail)
                             / hessenberg[row][row])
    solution = [mp.fsum(coefficients[column] * basis[column][row]
                        for column in range(used))
                for row in range(operator.order)]
    return solution, used, relative_residual


def solve_nodes(quadrature: Quadrature, workers: int,
                log: Callable[[str], None]) -> SolveResult:
    with DenseOperator(quadrature, workers, log) as operator:
        lambda_one = operator.apply([mp.mpf(1)] * operator.order)
        rhs = [EPSILON * value for value in lambda_one]
        solution, iterations, reported = gmres(operator, rhs, log)
        lambda_solution = operator.apply(solution)
        linear_vector = [solution[i] - SCATTERING * lambda_solution[i] - rhs[i]
                         for i in range(operator.order)]
        linear_residual = norm(linear_vector) / norm(rhs)
        source = [EPSILON + SCATTERING * value for value in solution]
        fixed_point = operator.apply(source)
        source_residual = (max(abs(fixed_point[i] - solution[i])
                               for i in range(operator.order))
                           / max(abs(value) for value in solution))
        matvecs = operator.apply_count
    return SolveResult(tuple(solution), iterations, reported,
                       linear_residual, source_residual, matvecs)


def natural_cubic_coefficients(x_value: Sequence[mp.mpf],
                               y_value: Sequence[mp.mpf]
                               ) -> tuple[list[mp.mpf], list[mp.mpf], list[mp.mpf]]:
    count = len(x_value)
    interval = [x_value[i+1] - x_value[i] for i in range(count - 1)]
    alpha = [mp.mpf(0)] * count
    for index in range(1, count - 1):
        alpha[index] = (3 * (y_value[index+1] - y_value[index]) / interval[index]
                        - 3 * (y_value[index] - y_value[index-1]) / interval[index-1])
    diagonal = [mp.mpf(1)] + [mp.mpf(0)] * (count - 1)
    upper = [mp.mpf(0)] * count
    rhs = [mp.mpf(0)] * count
    for index in range(1, count - 1):
        diagonal[index] = (2 * (x_value[index+1] - x_value[index-1])
                           - interval[index-1] * upper[index-1])
        upper[index] = interval[index] / diagonal[index]
        rhs[index] = (alpha[index] - interval[index-1] * rhs[index-1]) / diagonal[index]
    diagonal[-1] = mp.mpf(1)
    second = [mp.mpf(0)] * count
    linear = [mp.mpf(0)] * (count - 1)
    cubic = [mp.mpf(0)] * (count - 1)
    for index in range(count - 2, -1, -1):
        second[index] = rhs[index] - upper[index] * second[index+1]
        linear[index] = ((y_value[index+1] - y_value[index]) / interval[index]
                         - interval[index] * (second[index+1] + 2 * second[index]) / 3)
        cubic[index] = (second[index+1] - second[index]) / (3 * interval[index])
    return linear, second[:-1], cubic


def evaluate_targets(quadrature: Quadrature, solution: SolveResult,
                     targets: Sequence[mp.mpf], log: Callable[[str], None]
                     ) -> tuple[mp.mpf, ...]:
    started = time.monotonic()
    log(f"Nref={len(quadrature.radius)}: evaluating {len(targets)} targets "
        "with 80-digit natural cubic interpolation")
    linear, second, cubic = natural_cubic_coefficients(
        quadrature.radius, solution.j_value)
    values: list[mp.mpf] = []
    for target in targets:
        index = bisect.bisect_right(quadrature.radius, target) - 1
        index = max(0, min(index, len(quadrature.radius) - 2))
        offset = target - quadrature.radius[index]
        values.append(solution.j_value[index]
                      + linear[index] * offset
                      + second[index] * offset ** 2
                      + cubic[index] * offset ** 3)
    log(f"Nref={len(quadrature.radius)}: target evaluation ready in "
        f"{time.monotonic()-started:.3f}s")
    return tuple(values)


def relative_l2(left: Sequence[mp.mpf], right: Sequence[mp.mpf]) -> mp.mpf:
    return mp.sqrt(mp.fsum((a - b) ** 2 for a, b in zip(left, right))
                   / mp.fsum(value * value for value in right))


def solve_order(order: int, targets: Sequence[mp.mpf], workers: int,
                log: Callable[[str], None]) -> dict[str, object]:
    mp.mp.dps = DPS
    started = time.monotonic()
    quadrature = gauss_legendre_unit(order, workers, log)
    solution = solve_nodes(quadrature, workers, log)
    evaluated = evaluate_targets(quadrature, solution, targets, log)
    elapsed = time.monotonic() - started
    log(f"Nref={order}: complete in {elapsed:.3f}s; linear relres="
        f"{decimal(solution.linear_residual, 14)}")
    values = [decimal(value) for value in evaluated]
    return {
        "order": order,
        "mpmath_dps": DPS,
        "arithmetic": "mpmath mpf, 80 decimal digits end-to-end",
        "quadrature": "80-digit Legendre-Newton Gauss-Legendre",
        "singularity_subtraction": "analytic r=r' logarithmic subtraction",
        "operator_storage": "complete block-distributed dense mpf rows",
        "solver": "80-digit unrestarted MGS-GMRES over dense operator",
        "solve_tolerance": SOLVE_TOLERANCE_TEXT,
        "gmres_iterations": solution.gmres_iterations,
        "gmres_reported_residual": decimal(solution.gmres_reported_residual),
        "linear_residual": decimal(solution.linear_residual),
        "source_residual": decimal(solution.source_residual),
        "operator_matvecs": solution.operator_matvecs,
        "target_evaluation": "80-digit natural cubic interpolation",
        "quadrature_max_root_residual": decimal(quadrature.max_root_residual),
        "quadrature_weight_sum_error": decimal(quadrature.weight_sum_error),
        "quadrature_symmetry_error": decimal(quadrature.symmetry_error),
        "elapsed_seconds": elapsed,
        "J": values,
        "J_sha256": canonical_hash(values),
    }


def checkpoint_fingerprint(order: int, targets: Sequence[str]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "order": order,
        "mpmath_dps": DPS,
        "epsilon": EPSILON_TEXT,
        "scattering": SCATTERING_TEXT,
        "solve_tolerance": SOLVE_TOLERANCE_TEXT,
        "targets_sha256": canonical_hash(targets),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nref", type=int, nargs="+", default=list(PRODUCTION_NREF),
                        help="even Nyström orders; production contract: 2048 4096")
    parser.add_argument("--out", required=True, type=pathlib.Path,
                        help="result JSON path")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK",
                                                   min(32, os.cpu_count() or 1))))
    parser.add_argument("--checkpoint-dir", type=pathlib.Path,
                        help="default: OUT with suffix .checkpoints")
    parser.add_argument("--log", type=pathlib.Path,
                        help="default: OUT with suffix .progress.log")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore compatible completed-order checkpoints")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if len(set(args.nref)) != len(args.nref):
        parser.error("--nref entries must be unique")
    if any(order <= 0 or order % 2 for order in args.nref):
        parser.error("--nref entries must be positive even integers")
    return args


def main() -> int:
    args = parse_arguments()
    mp.mp.dps = DPS
    args.out = args.out.resolve()
    checkpoint_dir = (args.checkpoint_dir.resolve() if args.checkpoint_dir
                      else args.out.with_suffix(".checkpoints"))
    log_path = (args.log.resolve() if args.log
                else args.out.with_suffix(".progress.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(message: str) -> None:
        rendered = timestamped(message)
        print(rendered, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(rendered + "\n")

    started_wall = dt.datetime.now().astimezone()
    progress(f"run start; nref={args.nref}, workers={args.workers}, out={args.out}")
    targets = target_union()
    target_text = [decimal(value) for value in targets]
    references: dict[int, dict[str, object]] = {}
    resumed: list[int] = []
    for order in args.nref:
        fingerprint = checkpoint_fingerprint(order, target_text)
        checkpoint = checkpoint_dir / f"N{order}.pickle"
        reference: object | None = None
        if not args.no_resume and checkpoint.exists():
            candidate = load_pickle(checkpoint)
            if (isinstance(candidate, dict)
                    and candidate.get("fingerprint") == fingerprint
                    and isinstance(candidate.get("reference"), dict)):
                reference = candidate["reference"]
                resumed.append(order)
                progress(f"Nref={order}: resumed completed checkpoint {checkpoint}")
            else:
                progress(f"Nref={order}: incompatible checkpoint ignored: {checkpoint}")
        if reference is None:
            reference = solve_order(order, targets, args.workers, progress)
            atomic_pickle(checkpoint, {"fingerprint": fingerprint,
                                       "reference": reference})
            progress(f"Nref={order}: wrote completed checkpoint {checkpoint}")
        assert isinstance(reference, dict)
        references[order] = reference

    arithmetic_audit = {
        "nodes_and_weights_mpmath_80d": True,
        "E1_kernel_assembly_mpmath_80d": True,
        "log_singularity_subtraction_mpmath_80d": True,
        "dense_operator_storage_mpmath_80d": True,
        "dense_operator_solve_mpmath_80d": True,
        "target_evaluation_mpmath_80d": True,
        "comparison_norm_mpmath_80d": True,
    }
    production_pair_present = all(order in references for order in PRODUCTION_NREF)
    comparison: dict[str, object]
    if production_pair_present:
        centre_indices = [2 * index + 1 for index in range(512)]
        left = [mp.mpf(references[2048]["J"][index]) for index in centre_indices]
        right = [mp.mpf(references[4096]["J"][index]) for index in centre_indices]
        difference = relative_l2(left, right)
        comparison = {
            "evaluated": True,
            "orders": [2048, 4096],
            "targets": "512 shell centres (i+0.5)/512",
            "relative_l2": decimal(difference),
            "threshold": "1e-9",
            "pass": difference < mp.mpf("1e-9"),
        }
    else:
        comparison = {
            "evaluated": False,
            "required_orders": [2048, 4096],
            "reason": "production Nref pair not both present; smoke/non-acceptance run",
            "threshold": "1e-9",
            "pass": False,
        }
    linear_checks = {
        str(order): mp.mpf(references[order]["linear_residual"]) < mp.mpf("1e-50")
        for order in args.nref
    }
    # The original section 6.2 gate is the full-80-digit arithmetic contract
    # plus the Nref pair agreement.  The <1e-50 residual checks are deliberately
    # reported as solver diagnostics, not introduced as a new acceptance gate.
    qualified = (production_pair_present and bool(comparison["pass"])
                 and all(arithmetic_audit.values()))
    finished_wall = dt.datetime.now().astimezone()
    report = {
        "schema_version": SCHEMA_VERSION,
        "rung": 10,
        "status": "PASS" if qualified else ("FAIL" if production_pair_present else "SMOKE"),
        "acceptance_unchanged": True,
        "contract": {
            "equation": "original design section 6.2, Equation (15)",
            "method": "Gauss-Legendre Nystrom with analytic logarithmic singularity subtraction",
            "mpmath_dps": DPS,
            "required_nref": [2048, 4096],
            "nref_relative_l2_threshold": "1e-9",
            "parameters": {"chi0_R": "1", "epsilon": EPSILON_TEXT, "B0": "1"},
        },
        "requested_nref": args.nref,
        "arithmetic_audit": arithmetic_audit,
        "self_check": comparison,
        "solver_diagnostics": {"linear_residual_lt_1e-50": linear_checks},
        "oracle_qualified": qualified,
        "serialization": "80-significant-digit decimal strings; decisions precede JSON",
        "targets": {
            "description": "union of 513 production radii i/512 and 512 shell centres",
            "values": target_text,
            "sha256": canonical_hash(target_text),
        },
        "references": {str(order): references[order] for order in args.nref},
        "runtime": {
            "started": started_wall.isoformat(timespec="seconds"),
            "finished": finished_wall.isoformat(timespec="seconds"),
            "elapsed_seconds": (finished_wall - started_wall).total_seconds(),
            "workers": args.workers,
            "python": platform.python_version(),
            "mpmath": mp.__version__,
            "hostname": platform.node(),
            "resumed_nref": resumed,
            "progress_log": str(log_path),
            "checkpoint_dir": str(checkpoint_dir),
            "model_or_gpu_run": False,
        },
    }
    atomic_json(args.out, report)
    progress(f"wrote {args.out}; status={report['status']}")
    print(json.dumps({
        "out": str(args.out),
        "status": report["status"],
        "oracle_qualified": qualified,
        "self_check": comparison,
    }, indent=2))
    return 0 if report["status"] in {"PASS", "SMOKE"} else 1


if __name__ == "__main__":
    sys.exit(main())
