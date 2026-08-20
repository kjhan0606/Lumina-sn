#!/usr/bin/env python3
"""Census the statically visible writes to OpacityState.tau_sobolev.

The raw line slab is intentionally not copied into A2-08.  Its replacement
contract has three disjoint source-level classes: bracketed element writers,
preflight-proven generation transplants, and diagnostic save/restores.  One
bulk-call census is shared by the latter two registries.  The CUDA solve lane
must call the shared host writer and must not carry a second element writer.

This is a lexical/static gate.  Its PASS output explicitly retains the runtime,
semantic, alias/macro/function-pointer, and device-side surfaces that it cannot
certify; see VERDICT_GR2_transplant.md section 3 and VERDICT_GR2b_cuda.md
section 3.

Negative controls mutate source text in memory only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import sys
from typing import Callable

from gate_source_lib import (
    body_raw,
    find_anchored_block,
    find_definition,
    inject_at_head,
    lexical_view,
)


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PLASMA = "lumina_plasma.c"
CUDA = "lumina_cuda.cu"
WRITERS = (
    "compute_tau_sobolev",
    "nlte_update_tau_sobolev_with_authority",
    "apply_overlap_corrections",
)
TRANSPLANTS = (
    "nlte_population_candidate_commit_bundle",
    "nlte_population_candidate_commit_seed_material",
)
ASSIGN = re.compile(
    r"\b(?:opacity|public_opacity)(?:\s*->|\s*\.)\s*"
    r"tau_sobolev\s*\[[^\]]+\]\s*=(?!=)",
    re.S,
)
BULK_TAU = re.compile(
    r"\b(?P<kind>memcpy|memmove|memset)\s*\(\s*"
    r"(?P<dest>[^,;]*\btau_sobolev\b[^,;]*)\s*,[^;]*?;",
    re.S,
)

TAU_WRAPPER = """{
    nlte_update_tau_sobolev_with_authority(
        nlte, atom, opacity, time_explosion, n_shells,
        g_ew_tau_authority, g_ew_tau_authority_nshells);
}"""
TRANSPLANT_BULK = """memcpy(public_opacity->tau_sobolev,candidate->tau_sobolev,
           candidate->n_line_values*sizeof(double));"""
TRANSPLANT_GENERATIONS = """public_opacity->tau_required_generation=
        candidate->opacity.tau_required_generation;
    public_opacity->tau_computed_generation=
        candidate->opacity.tau_computed_generation;
    public_opacity->tau_first_consumer_generation=
        candidate->opacity.tau_computed_generation;"""
TRANSPLANT_FIRST_CONSUMER = """public_opacity->tau_first_consumer_generation=
        candidate->opacity.tau_computed_generation;"""
TRANSPLANT_GUARDS = {
    "nlte_population_candidate_commit_bundle": """if(!candidate_bundle_commit_preflight(
           candidate,te_candidate,public_nlte,public_atom,public_plasma,
           public_opacity,public_bf)){
        if(candidate)candidate->status=NLTE_CANDIDATE_COMMIT_FAILED;
        return NLTE_CANDIDATE_COMMIT_FAILED;
    }""",
    "nlte_population_candidate_commit_seed_material": """if(!candidate_seed_commit_preflight(
           candidate,public_nlte,public_atom,public_plasma,public_opacity,
           public_bf)){
        if(candidate)candidate->status=NLTE_CANDIDATE_COMMIT_FAILED;
        return NLTE_CANDIDATE_COMMIT_FAILED;
    }""",
}
PREFLIGHT_TAU_ALIAS = "candidate->tau_sobolev==opacity->tau_sobolev||"
PREFLIGHT_TAU_CLOSURE = """candidate->opacity.tau_required_generation==0||
       candidate->opacity.tau_computed_generation!=
           candidate->opacity.tau_required_generation)return 0;"""

SAVERESTORE_ENV = """static int fr_on = -1;
                if (fr_on < 0) { const char *e = getenv("LUMINA_NLTE_FINAL_RESOLVE");
                                 fr_on = (e && atoi(e)) ? 1 : 0; }
                if (fr_on && nlte.enabled && enable_nlte) {"""
SAVERESTORE_DERIVATION = """static int fr_on = -1;
                if (fr_on < 0) { const char *e = getenv("LUMINA_NLTE_FINAL_RESOLVE");
                                 fr_on = (e && atoi(e)) ? 1 : 0; }
                """
SAVERESTORE_GUARD = "if (fr_on && nlte.enabled && enable_nlte) {"
SAVERESTORE_TAU = (
    "memcpy(opacity.tau_sobolev,         tau_save, nline * sizeof(double));"
)


@dataclass(frozen=True)
class BulkWrite:
    filename: str
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class Census:
    writer_counts: dict[str, int]
    cuda_assign_writers: int
    bulk_total: int
    bulk_registered: int
    transplant_bulk: int
    saverestore_bulk: int


def source_texts() -> dict[str, str]:
    paths = sorted(SRC.glob("*.c")) + sorted(SRC.glob("*.cu"))
    return {path.name: path.read_text(encoding="utf-8") for path in paths}


def line_at(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


@lru_cache(maxsize=512)
def structural_view(text: str) -> str:
    return lexical_view(text, blank_literals=True)


@lru_cache(maxsize=512)
def code_view(text: str) -> str:
    return lexical_view(text)


@lru_cache(maxsize=512)
def bulk_writes(filename: str, text: str) -> tuple[BulkWrite, ...]:
    structural = structural_view(text)
    return tuple(
        BulkWrite(filename, match.start(), match.end(), match.group("kind"))
        for match in BULK_TAU.finditer(structural)
    )


def exact_offsets(text: str, anchor: str, lo: int, hi: int) -> list[int]:
    return [lo + match.start() for match in re.finditer(re.escape(anchor), text[lo:hi])]


def append_failure(failures: list[str], reason: str) -> None:
    if reason not in failures:
        failures.append(reason)


def inspect(sources: dict[str, str]) -> tuple[list[str], Census]:
    failures: list[str] = []
    plasma = sources[PLASMA]
    cuda = sources[CUDA]
    plasma_code = structural_view(plasma)
    bulk_by_file = {
        filename: bulk_writes(filename, text) for filename, text in sources.items()
    }

    writer_spans: dict[str, tuple[int, int]] = {}
    transplant_spans: dict[str, tuple[int, int]] = {}
    for registry, destination in (
        (WRITERS, writer_spans),
        (TRANSPLANTS, transplant_spans),
    ):
        for name in registry:
            try:
                destination[name] = find_definition(plasma, name)
            except RuntimeError as error:
                append_failure(failures, f"source anchor failure:{error}")

    if set(WRITERS) & set(TRANSPLANTS):
        append_failure(failures, "writer/transplant registries overlap")

    counts = {name: 0 for name in WRITERS}
    cuda_assign_writers = 0
    for filename, text in sources.items():
        structural = structural_view(text)
        for match in ASSIGN.finditer(structural):
            line = line_at(text, match.start())
            if filename == PLASMA:
                owners = [
                    name for name, (lo, hi) in writer_spans.items()
                    if lo <= match.start() < hi
                ]
                if len(owners) != 1:
                    append_failure(
                        failures,
                        f"unregistered raw tau writer at {filename}:{line}",
                    )
                else:
                    counts[owners[0]] += 1
            elif filename == CUDA:
                cuda_assign_writers += 1
                append_failure(
                    failures,
                    f"duplicate/unregistered CUDA raw tau writer at {filename}:{line}",
                )
            else:
                append_failure(
                    failures,
                    f"unregistered raw tau writer at {filename}:{line}",
                )

    for name, (lo, hi) in writer_spans.items():
        body = plasma_code[lo:hi]
        writes = list(ASSIGN.finditer(body))
        if not writes:
            append_failure(failures, f"registered writer has no tau writes: {name}")
            continue
        require = body.find("tau_sobolev_require_refresh")
        mark = body.rfind("tau_sobolev_mark_computed")
        if require < 0 or require > writes[0].start():
            append_failure(
                failures, f"{name}: generation is not advanced before first write"
            )
        if mark < 0 or mark < writes[-1].end():
            append_failure(
                failures, f"{name}: generation is not marked after last write"
            )

    if body_raw(plasma, "nlte_update_tau_sobolev") != TAU_WRAPPER:
        append_failure(
            failures, "nlte_update_tau_sobolev wrapper is not exact pure delegation"
        )

    registered_bulk: dict[tuple[str, int], str] = {}
    for name, span in transplant_spans.items():
        lo, hi = span
        body = plasma[lo:hi]
        guard = TRANSPLANT_GUARDS[name]
        bulk_offsets = exact_offsets(plasma, TRANSPLANT_BULK, lo, hi)
        bulk_in_span = [
            write for write in bulk_by_file[PLASMA]
            if lo <= write.start < hi
        ]
        guard_at = body.find(guard)
        first_bulk_at = bulk_offsets[0] - lo if bulk_offsets else -1
        if body.count(guard) != 1 or first_bulk_at < 0 or guard_at > first_bulk_at:
            append_failure(failures, f"transplant not preflight-guarded:{name}")
        if len(bulk_offsets) != 1 or len(bulk_in_span) != 1:
            append_failure(failures, f"transplant tau memcpy pin mismatch:{name}")
        else:
            registered_bulk[(PLASMA, bulk_offsets[0])] = "transplant"
        generation_at = body.find(TRANSPLANT_GENERATIONS)
        if body.count(TRANSPLANT_GENERATIONS) != 1 or (
            first_bulk_at >= 0 and generation_at <= first_bulk_at
        ):
            append_failure(
                failures, f"transplant does not carry the tau generation:{name}"
            )

    try:
        preflight_lo, preflight_hi = find_definition(
            plasma, "candidate_material_commit_preflight"
        )
        preflight = plasma[preflight_lo:preflight_hi]
        if preflight.count(PREFLIGHT_TAU_ALIAS) != 1:
            append_failure(failures, "preflight tau alias prohibition pin missing")
        if preflight.count(PREFLIGHT_TAU_CLOSURE) != 1:
            append_failure(failures, "preflight tau generation closure pin missing")
    except RuntimeError as error:
        append_failure(failures, f"source anchor failure:{error}")

    saverestore_span: tuple[int, int] | None = None
    fr_on_count = len(re.findall(r"\bfr_on\b", code_view(cuda)))
    if cuda.count(SAVERESTORE_ENV) != 1 or fr_on_count != 4:
        append_failure(failures, "saverestore guard anchor missing")
    if cuda.count(SAVERESTORE_GUARD) == 1:
        try:
            saverestore_span = find_anchored_block(cuda, SAVERESTORE_GUARD)
        except RuntimeError:
            append_failure(failures, "saverestore guard anchor missing")
    else:
        append_failure(failures, "saverestore guard anchor missing")

    if saverestore_span is not None:
        lo, hi = saverestore_span
        save_offsets = exact_offsets(cuda, SAVERESTORE_TAU, lo, hi)
        bulk_in_span = [
            write for write in bulk_by_file[CUDA]
            if lo <= write.start < hi
        ]
        if len(save_offsets) != 2 or len(bulk_in_span) != 2:
            append_failure(failures, "saverestore tau memcpy pin mismatch")
        else:
            for offset in save_offsets:
                registered_bulk[(CUDA, offset)] = "saverestore"
    else:
        append_failure(failures, "saverestore tau memcpy pin mismatch")

    all_bulk = [
        write
        for filename in sources
        for write in bulk_by_file[filename]
    ]
    registered_seen: list[BulkWrite] = []
    for write in all_bulk:
        if (write.filename, write.start) in registered_bulk:
            registered_seen.append(write)
            continue
        append_failure(
            failures,
            "unregistered bulk tau writer "
            f"({write.kind}) at {write.filename}:{line_at(sources[write.filename], write.start)}",
        )

    if "nlte_update_tau_sobolev(nlte, atom, opacity" not in cuda:
        append_failure(failures, "CUDA NLTE solve does not route through shared host tau writer")
    if "EMISS_TAU_MUTATED_DURING_CONSUME" not in plasma:
        append_failure(failures, "A2-09 end-of-consumption generation abort is absent")

    transplant_bulk = sum(
        registered_bulk[(write.filename, write.start)] == "transplant"
        for write in registered_seen
    )
    saverestore_bulk = sum(
        registered_bulk[(write.filename, write.start)] == "saverestore"
        for write in registered_seen
    )
    census = Census(
        writer_counts=counts,
        cuda_assign_writers=cuda_assign_writers,
        bulk_total=len(all_bulk),
        bulk_registered=len(registered_seen),
        transplant_bulk=transplant_bulk,
        saverestore_bulk=saverestore_bulk,
    )
    return failures, census


def replace_in_definition(
    text: str, name: str, old: str, new: str
) -> str:
    try:
        lo, hi = find_definition(text, name)
    except RuntimeError:
        return text
    definition = text[lo:hi]
    if definition.count(old) != 1:
        return text
    return text[:lo] + definition.replace(old, new, 1) + text[hi:]


def replace_nth(text: str, old: str, new: str, occurrence: int) -> str:
    offsets = [match.start() for match in re.finditer(re.escape(old), text)]
    if occurrence < 1 or occurrence > len(offsets):
        return text
    at = offsets[occurrence - 1]
    return text[:at] + new + text[at + len(old):]


def changed(
    sources: dict[str, str], filename: str, transform: Callable[[str], str]
) -> dict[str, str]:
    mutated = dict(sources)
    mutated[filename] = transform(sources[filename])
    return mutated


def reason_at(
    sources: dict[str, str], filename: str, token: str, prefix: str
) -> str:
    offset = sources[filename].find(token)
    return f"{prefix} at {filename}:{line_at(sources[filename], offset)}"


def negative_controls(sources: dict[str, str]) -> list[str]:
    rogue_raw = "void rogue_tau_writer(OpacityState *opacity){opacity->tau_sobolev[0]=0.0;}"
    rogue_cuda = "void rogue_cuda(OpacityState *opacity){opacity->tau_sobolev[0]=0.0;}"
    rogue_bulk = (
        "void rogue_bulk_tau(OpacityState *public_opacity,const void *x,size_t n)"
        "{memcpy(public_opacity->tau_sobolev, x, n);}"
    )
    rogue_cuda_bulk = (
        "void rogue_cuda_restore(OpacityState *public_opacity,const void *x,size_t n)"
        "{memcpy(public_opacity->tau_sobolev, x, n);}"
    )

    def move_seed_bulk(text: str) -> str:
        without = replace_in_definition(
            text,
            "nlte_population_candidate_commit_seed_material",
            TRANSPLANT_BULK,
            "",
        )
        if without == text:
            return text
        return inject_at_head(
            without,
            "nlte_population_candidate_commit_seed_material",
            TRANSPLANT_BULK,
        )

    controls: list[
        tuple[str, Callable[[dict[str, str]], dict[str, str]], Callable[[dict[str, str]], str]]
    ] = [
        (
            "writer-require",
            lambda original: changed(
                original,
                PLASMA,
                lambda text: text.replace(
                    'tau_sobolev_require_refresh(opacity, "compute_tau_sobolev");',
                    "",
                    1,
                ),
            ),
            lambda _: "compute_tau_sobolev: generation is not advanced before first write",
        ),
        (
            "writer-mark",
            lambda original: changed(
                original,
                PLASMA,
                lambda text: text.replace(
                    'tau_sobolev_mark_computed(opacity, "nlte_update_tau_sobolev");',
                    "",
                    1,
                ),
            ),
            lambda _: "nlte_update_tau_sobolev_with_authority: generation is not marked after last write",
        ),
        (
            "rogue-plasma-assign",
            lambda original: changed(
                original, PLASMA, lambda text: text + "\n" + rogue_raw + "\n"
            ),
            lambda mutated: reason_at(
                mutated,
                PLASMA,
                "opacity->tau_sobolev[0]=0.0",
                "unregistered raw tau writer",
            ),
        ),
        (
            "rogue-cuda-assign",
            lambda original: changed(
                original, CUDA, lambda text: text + "\n" + rogue_cuda + "\n"
            ),
            lambda mutated: reason_at(
                mutated,
                CUDA,
                "opacity->tau_sobolev[0]=0.0",
                "duplicate/unregistered CUDA raw tau writer",
            ),
        ),
        (
            "transplant-guard-removed",
            lambda original: changed(
                original,
                PLASMA,
                lambda text: replace_in_definition(
                    text,
                    "nlte_population_candidate_commit_bundle",
                    TRANSPLANT_GUARDS["nlte_population_candidate_commit_bundle"],
                    "",
                ),
            ),
            lambda _: "transplant not preflight-guarded:nlte_population_candidate_commit_bundle",
        ),
        (
            "transplant-bulk-before-guard",
            lambda original: changed(original, PLASMA, move_seed_bulk),
            lambda _: "transplant not preflight-guarded:nlte_population_candidate_commit_seed_material",
        ),
        (
            "transplant-generation-removed",
            lambda original: changed(
                original,
                PLASMA,
                lambda text: replace_in_definition(
                    text,
                    "nlte_population_candidate_commit_bundle",
                    TRANSPLANT_FIRST_CONSUMER,
                    "",
                ),
            ),
            lambda _: "transplant does not carry the tau generation:nlte_population_candidate_commit_bundle",
        ),
        (
            "rogue-transplant-bulk",
            lambda original: changed(
                original, PLASMA, lambda text: text + "\n" + rogue_bulk + "\n"
            ),
            lambda mutated: reason_at(
                mutated,
                PLASMA,
                "memcpy(public_opacity->tau_sobolev, x, n)",
                "unregistered bulk tau writer (memcpy)",
            ),
        ),
        (
            "rogue-assign-inside-transplant",
            lambda original: changed(
                original,
                PLASMA,
                lambda text: inject_at_head(
                    text,
                    "nlte_population_candidate_commit_bundle",
                    "public_opacity->tau_sobolev[0]=0.0;",
                ),
            ),
            lambda mutated: reason_at(
                mutated,
                PLASMA,
                "public_opacity->tau_sobolev[0]=0.0",
                "unregistered raw tau writer",
            ),
        ),
        (
            "rogue-saverestore-outside-guard",
            lambda original: changed(
                original, CUDA, lambda text: text + "\n" + rogue_cuda_bulk + "\n"
            ),
            lambda mutated: reason_at(
                mutated,
                CUDA,
                "memcpy(public_opacity->tau_sobolev, x, n)",
                "unregistered bulk tau writer (memcpy)",
            ),
        ),
        (
            "saverestore-guard-weakened",
            lambda original: changed(
                original,
                CUDA,
                lambda text: text.replace(
                    SAVERESTORE_GUARD,
                    "if (nlte.enabled && enable_nlte) {",
                    1,
                ),
            ),
            lambda _: "saverestore guard anchor missing",
        ),
        (
            "saverestore-env-derivation-removed",
            lambda original: changed(
                original,
                CUDA,
                lambda text: text.replace(SAVERESTORE_DERIVATION, "", 1),
            ),
            lambda _: "saverestore guard anchor missing",
        ),
        (
            "rogue-assign-inside-saverestore",
            lambda original: changed(
                original,
                CUDA,
                lambda text: text.replace(
                    SAVERESTORE_GUARD,
                    SAVERESTORE_GUARD + "\n                    opacity.tau_sobolev[0]=0.0;",
                    1,
                ),
            ),
            lambda mutated: reason_at(
                mutated,
                CUDA,
                "opacity.tau_sobolev[0]=0.0",
                "duplicate/unregistered CUDA raw tau writer",
            ),
        ),
        (
            "saverestore-memcpy-pin-mutated",
            lambda original: changed(
                original,
                CUDA,
                lambda text: replace_nth(
                    text,
                    SAVERESTORE_TAU,
                    SAVERESTORE_TAU.replace("tau_save", "sl_save"),
                    2,
                ),
            ),
            lambda _: "saverestore tau memcpy pin mismatch",
        ),
    ]

    missed: list[str] = []
    for index, (name, mutate, expected_reason) in enumerate(controls, start=1):
        mutated = mutate(sources)
        if mutated == sources:
            missed.append(f"injection-{index}-not-applied")
            continue
        expected = expected_reason(mutated)
        failures, _ = inspect(mutated)
        if expected not in failures:
            missed.append(f"injection-{index}-wrong-reason")
            continue
        print(
            "[TAU-WRITER-CENSUS][NEGATIVE-CONTROL][DETECTED] "
            f"injection={index} name={name} reason={expected}"
        )
    return missed


def main() -> int:
    sources = source_texts()
    failures, census = inspect(sources)
    if failures:
        for failure in failures:
            print(f"[TAU-WRITER-CENSUS][FAIL] {failure}", file=sys.stderr)
        return 1
    detail = " ".join(
        f"{name}={census.writer_counts[name]}" for name in WRITERS
    )
    print(
        f"[TAU-WRITER-CENSUS][PASS] element_writers={len(WRITERS)} {detail} "
        f"cuda_assign_writers={census.cuda_assign_writers} "
        f"bulk_tau={census.bulk_total}/{census.bulk_registered} registered: "
        f"transplant={census.transplant_bulk}, "
        f"diag_saverestore={census.saverestore_bulk}"
        "(gate=LUMINA_NLTE_FINAL_RESOLVE)"
    )
    print(
        "[TAU-WRITER-CENSUS][PASS] noncertified_residuals=present "
        "refs=VERDICT_GR2_transplant.md§3(6),VERDICT_GR2b_cuda.md§3(5); "
        "runtime-preflight/env,semantic-completeness,public-generation-monotonicity,"
        "armed-saverestore-ledger-gaps,local-alias/macro/function-pointer-bypass,"
        "device-tau"
    )

    missed = negative_controls(sources)
    if missed:
        print("[TAU-WRITER-CENSUS][NEGATIVE-CONTROL][FAIL] " + ",".join(missed),
              file=sys.stderr)
        return 1
    print("[TAU-WRITER-CENSUS][NEGATIVE-CONTROL][PASS] injections=14 detected=14")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
