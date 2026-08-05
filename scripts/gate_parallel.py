#!/usr/bin/env python3
"""Small shared executor for case-level gate parallelism."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar


Task = TypeVar("Task")
Result = TypeVar("Result")


def worker_count(serial: bool) -> int:
    """Return the contract worker count, retaining a one-worker fallback."""
    return 1 if serial else (os.cpu_count() or 1)


def run_cases(
    battery: str,
    function: Callable[[Task], Result],
    tasks: Iterable[Task],
    *,
    serial: bool,
    case_name: Callable[[Task], str],
) -> list[Result]:
    """Run cases and emit exactly one completion-progress line per case.

    Results are returned in input order so the stable PASS/FAIL table is
    independent of completion order.
    """
    ordered = list(tasks)
    total = len(ordered)
    completed = 0
    results: dict[int, Result] = {}

    if serial:
        for index, task in enumerate(ordered):
            try:
                results[index] = function(task)
            finally:
                completed += 1
                print(
                    f"PROGRESS battery={battery} completed={completed}/{total} "
                    f"case={case_name(task)}",
                    flush=True,
                )
    else:
        first_error: BaseException | None = None
        with ProcessPoolExecutor(max_workers=worker_count(False)) as pool:
            futures = {
                pool.submit(function, task): (index, task)
                for index, task in enumerate(ordered)
            }
            for future in as_completed(futures):
                index, task = futures[future]
                try:
                    results[index] = future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                finally:
                    completed += 1
                    print(
                        f"PROGRESS battery={battery} completed={completed}/{total} "
                        f"case={case_name(task)}",
                        flush=True,
                    )
        if first_error is not None:
            raise first_error

    return [results[index] for index in range(total)]
