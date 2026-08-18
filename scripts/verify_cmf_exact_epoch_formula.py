#!/usr/bin/env python3
"""Structural proof harness for the exact two-stack epoch decomposition."""

from __future__ import annotations

import math
import random
import sys


ROUND_LOWER = -1
ROUND_NEAREST = 0
ROUND_UPPER = 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CMF_EXACT_EPOCH_FORMULA_FAIL {message}")


def add_bound(a: float, b: float, rounding: int) -> float:
    require(a >= 0.0 and b >= 0.0 and math.isfinite(a) and math.isfinite(b),
            "invalid-add-input")
    result = a + b
    require(math.isfinite(result), "nonfinite-add")
    if rounding != ROUND_NEAREST and a != 0.0 and b != 0.0:
        b_virtual = result - a
        error = (a - (result - b_virtual)) + (b - b_virtual)
        if rounding == ROUND_UPPER and error > 0.0:
            result = math.nextafter(result, math.inf)
        elif rounding == ROUND_LOWER and error < 0.0:
            result = math.nextafter(result, 0.0)
    require(result >= 0.0 and math.isfinite(result), "invalid-add-output")
    return result


def multiply_bound(a: float, b: float, rounding: int) -> float:
    require(a >= 0.0 and b >= 0.0 and math.isfinite(a) and math.isfinite(b),
            "invalid-multiply-input")
    if a == 0.0 or b == 0.0:
        return 0.0
    result = a * b
    require(math.isfinite(result), "nonfinite-multiply")
    if rounding == ROUND_UPPER:
        result = math.nextafter(result, math.inf)
    elif rounding == ROUND_LOWER and result != 0.0:
        result = math.nextafter(result, 0.0)
    require(result >= 0.0 and math.isfinite(result),
            "invalid-multiply-output")
    return result


def reverse_compose(
        first: tuple[float, float], second: tuple[float, float],
        rounding: int) -> tuple[float, float]:
    transmission = multiply_bound(second[0], first[0], rounding)
    attenuated = multiply_bound(second[0], first[1], rounding)
    emission = add_bound(second[1], attenuated, rounding)
    return transmission, emission


def push_back(
        back: list[tuple[tuple[float, float], tuple[float, float]]],
        value: tuple[float, float], rounding: int) -> None:
    aggregate = value if not back else reverse_compose(
        back[-1][1], value, rounding)
    back.append((value, aggregate))


def transfer(
        front: list[tuple[tuple[float, float], tuple[float, float]]],
        back: list[tuple[tuple[float, float], tuple[float, float]]],
        rounding: int) -> None:
    require(not front, "transfer-with-nonempty-front")
    while back:
        value = back.pop()[0]
        aggregate = value if not front else reverse_compose(
            value, front[-1][1], rounding)
        front.append((value, aggregate))


def aggregate_window(
        front: list[tuple[tuple[float, float], tuple[float, float]]],
        back: list[tuple[tuple[float, float], tuple[float, float]]],
        rounding: int) -> tuple[float, float]:
    if not front and not back:
        return 1.0, 0.0
    if not front:
        return back[-1][1]
    if not back:
        return front[-1][1]
    return reverse_compose(front[-1][1], back[-1][1], rounding)


def bounded_bin_index(index: int, bins: int) -> int:
    if index < 0:
        return 0
    if index >= bins:
        return bins - 1
    return index


def serial_aggregates(
        values: list[tuple[float, float]], window: int,
        rounding: int) -> list[tuple[float, float]]:
    bins = len(values)
    highest = bins - 1
    front: list[tuple[tuple[float, float], tuple[float, float]]] = []
    back: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for index in range(highest + window, highest, -1):
        push_back(back, values[bounded_bin_index(index, bins)], rounding)
    result = []
    for output_bin in range(highest, -1, -1):
        result.append(aggregate_window(front, back, rounding))
        if output_bin == 0 or window == 0:
            continue
        if not front:
            transfer(front, back, rounding)
        front.pop()
        push_back(back, values[output_bin], rounding)
    return result


def epoch_aggregates(
        values: list[tuple[float, float]], window: int,
        rounding: int) -> list[tuple[float, float]]:
    bins = len(values)
    if window == 0:
        return [(1.0, 0.0)] * bins

    def value(index: int) -> tuple[float, float]:
        return values[bounded_bin_index(index, bins)]

    highest = bins - 1
    result = []
    epoch = 0
    while True:
        boundary_bin = highest - epoch * window
        if boundary_bin < 0:
            break
        epoch_outputs = window
        if boundary_bin + 1 < epoch_outputs:
            epoch_outputs = boundary_bin + 1

        boundary = value(boundary_bin + window)
        for index in range(boundary_bin + window - 1,
                           boundary_bin, -1):
            boundary = reverse_compose(boundary, value(index), rounding)

        front = [value(boundary_bin + 1)]
        for index in range(boundary_bin + 2,
                           boundary_bin + window + 1):
            front.append(reverse_compose(value(index), front[-1], rounding))

        new_back = []
        for offset in range(epoch_outputs - 1):
            incoming = value(boundary_bin - offset)
            new_back.append(incoming if not new_back else reverse_compose(
                new_back[-1], incoming, rounding))

        result.append(boundary)
        for offset in range(1, epoch_outputs):
            result.append(reverse_compose(
                front[window - offset - 1], new_back[offset - 1], rounding))
        epoch += 1
    return result


NONASSOCIATIVE_CASES = {
    ROUND_NEAREST: (
        (float.fromhex("0x1.7ef1e3b8709dcp-7"),
         float.fromhex("0x1.14368367ce85ap-30")),
        (float.fromhex("0x1.f47d6ab739746p-3"),
         float.fromhex("0x1.2916788bc7ab5p-12")),
        (float.fromhex("0x1.f858297efd535p-2"),
         float.fromhex("0x1.94c196e83936ep+1")),
    ),
    ROUND_LOWER: (
        (float.fromhex("0x1.fda66fd3b058fp-4"),
         float.fromhex("0x1.42f617f05525ap-3")),
        (float.fromhex("0x1.4f9fee1fe330ep-7"),
         float.fromhex("0x1.fb0e76b3ec976p-14")),
        (float.fromhex("0x1.b9d3be38204abp-4"),
         float.fromhex("0x1.7e82febc2e09cp-24")),
    ),
    ROUND_UPPER: (
        (float.fromhex("0x1.c0a2e8eeb73d8p-4"),
         float.fromhex("0x1.742d6735617fep+4")),
        (float.fromhex("0x1.05e72455d5868p-4"),
         float.fromhex("0x1.f7875c0da4e2ap-1")),
        (float.fromhex("0x1.64b1e4f1221f8p-7"),
         float.fromhex("0x1.05e51e0c95836p+7")),
    ),
}


def main() -> int:
    require(sys.float_info.radix == 2 and sys.float_info.mant_dig == 53,
            "host-is-not-binary64")
    for rounding, (first, second, third) in NONASSOCIATIVE_CASES.items():
        left = reverse_compose(
            reverse_compose(first, second, rounding), third, rounding)
        right = reverse_compose(
            first, reverse_compose(second, third, rounding), rounding)
        require(left != right,
                f"nonassociative-witness-collapsed rounding={rounding}")
    print("CMF_EXACT_AFFINE_NONASSOCIATIVE PASS modes=lower/nearest/upper")

    generator = random.Random(20260810)
    cases = 0
    bin_counts = list(range(1, 18)) + [31, 32, 33, 63, 64, 65, 96]
    for bins in bin_counts:
        windows = {0, 1, 2, 3, bins - 1, bins, bins + 1, 2 * bins + 3}
        for window in sorted(value for value in windows if value >= 0):
            for trial in range(12):
                values = []
                for _ in range(bins):
                    if trial == 0:
                        transform = (1.0, 0.0)
                    elif trial == 1:
                        transform = (
                            math.nextafter(1.0, 0.0),
                            math.ldexp(1.0, -1074),
                        )
                    else:
                        transform = (
                            generator.random(),
                            math.ldexp(generator.random(),
                                       generator.randint(-1020, 20)),
                        )
                    values.append(transform)
                for rounding in (ROUND_LOWER, ROUND_NEAREST, ROUND_UPPER):
                    serial = serial_aggregates(values, window, rounding)
                    epoch = epoch_aggregates(values, window, rounding)
                    require(serial == epoch,
                            f"aggregate-mismatch bins={bins} window={window} "
                            f"trial={trial} rounding={rounding}")
                    cases += 1
    print(
        "CMF_EXACT_EPOCH_FORMULA PASS "
        f"cases={cases} aggregate_pairs_bit_identical=all "
        "numerical_repairs=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
