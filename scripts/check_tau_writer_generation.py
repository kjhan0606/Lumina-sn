#!/usr/bin/env python3
"""Census every production write to OpacityState.tau_sobolev.

The raw line slab is intentionally not copied into A2-08.  Its replacement
contract therefore requires a closed writer set: every writer advances the
required generation before its first cell write and marks that generation
computed after its last write.  The CUDA solve lane must call the shared host
writer and must not carry a second implementation.

Negative controls mutate source text in memory only.
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent.parent
PLASMA = ROOT / "src" / "lumina_plasma.c"
CUDA = ROOT / "src" / "lumina_cuda.cu"
WRITERS = (
    "compute_tau_sobolev",
    "nlte_update_tau_sobolev",
    "apply_overlap_corrections",
)
ASSIGN = re.compile(
    r"\bopacity\s*->\s*tau_sobolev\s*\[[^\]]+\]\s*=(?!=)", re.S
)
CUDA_ASSIGN = re.compile(
    r"\bopacity(?:\s*->|\s*\.)\s*tau_sobolev\s*\[[^\]]+\]\s*=(?!=)",
    re.S,
)


def function_span(text: str, name: str) -> tuple[int, int]:
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^;]*?\)\s*\{", text, re.S)
    if not match:
        raise RuntimeError(f"function not found: {name}")
    start = match.end() - 1
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return start, pos + 1
    raise RuntimeError(f"unterminated function: {name}")


def inspect(plasma: str, cuda: str) -> tuple[list[str], dict[str, int]]:
    failures: list[str] = []
    spans = {name: function_span(plasma, name) for name in WRITERS}
    counts = {name: 0 for name in WRITERS}

    for match in ASSIGN.finditer(plasma):
        owners = [name for name, (lo, hi) in spans.items() if lo <= match.start() < hi]
        if len(owners) != 1:
            line = plasma.count("\n", 0, match.start()) + 1
            failures.append(f"unregistered raw tau writer at lumina_plasma.c:{line}")
        else:
            counts[owners[0]] += 1

    for name, (lo, hi) in spans.items():
        body = plasma[lo:hi]
        writes = list(ASSIGN.finditer(body))
        if not writes:
            failures.append(f"registered writer has no tau writes: {name}")
            continue
        require = body.find("tau_sobolev_require_refresh")
        mark = body.rfind("tau_sobolev_mark_computed")
        if require < 0 or require > writes[0].start():
            failures.append(f"{name}: generation is not advanced before first write")
        if mark < 0 or mark < writes[-1].end():
            failures.append(f"{name}: generation is not marked after last write")

    cuda_writes = list(CUDA_ASSIGN.finditer(cuda))
    if cuda_writes:
        for match in cuda_writes:
            line = cuda.count("\n", 0, match.start()) + 1
            failures.append(f"duplicate/unregistered CUDA raw tau writer at lumina_cuda.cu:{line}")
    if "nlte_update_tau_sobolev(nlte, atom, opacity" not in cuda:
        failures.append("CUDA NLTE solve does not route through shared host tau writer")
    if "EMISS_TAU_MUTATED_DURING_CONSUME" not in plasma:
        failures.append("A2-09 end-of-consumption generation abort is absent")
    return failures, counts


def negative_controls(plasma: str, cuda: str) -> list[str]:
    mutations = (
        (plasma.replace("tau_sobolev_require_refresh(opacity, \"compute_tau_sobolev\");", "", 1), cuda),
        (plasma.replace("tau_sobolev_mark_computed(opacity, \"nlte_update_tau_sobolev\");", "", 1), cuda),
        (plasma + "\nvoid rogue_tau_writer(OpacityState *opacity){opacity->tau_sobolev[0]=0.0;}\n", cuda),
        (plasma, cuda + "\nvoid rogue_cuda(OpacityState *opacity){opacity->tau_sobolev[0]=0.0;}\n"),
    )
    missed: list[str] = []
    for index, (mutated_plasma, mutated_cuda) in enumerate(mutations, start=1):
        if mutated_plasma == plasma and mutated_cuda == cuda:
            missed.append(f"injection-{index}-not-applied")
        elif not inspect(mutated_plasma, mutated_cuda)[0]:
            missed.append(f"injection-{index}-not-detected")
    return missed


def main() -> int:
    plasma = PLASMA.read_text()
    cuda = CUDA.read_text()
    failures, counts = inspect(plasma, cuda)
    if failures:
        for failure in failures:
            print(f"[TAU-WRITER-CENSUS][FAIL] {failure}", file=sys.stderr)
        return 1
    detail = " ".join(f"{name}={counts[name]}" for name in WRITERS)
    print(f"[TAU-WRITER-CENSUS][PASS] writers={len(WRITERS)} {detail} cuda_writers=0")

    missed = negative_controls(plasma, cuda)
    if missed:
        print("[TAU-WRITER-CENSUS][NEGATIVE-CONTROL][FAIL] " + ",".join(missed),
              file=sys.stderr)
        return 1
    print("[TAU-WRITER-CENSUS][NEGATIVE-CONTROL][PASS] injections=4 detected=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
