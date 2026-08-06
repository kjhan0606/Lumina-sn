#!/usr/bin/env python3
"""A2-13~15 five-CUDA-TU read trace and contract wiring census."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_FILES = tuple(
    ROOT / "src" / name
    for name in (
        "lumina_cuda.cu", "lumina_bf_gemm.cu", "lumina_nlte_gemm.cu",
        "lumina_nlte_assemble.cu", "lumina_cmf_solve.cu",
    )
)
ORACLE_CUDA_FILES = tuple(
    ROOT / "src" / name
    for name in (
        "gpu_physics_kernels.cu", "gpu_opacity_kernels.cu",
        "gpu_emissivity_kernels.cu",
    )
)
TRACE_FILES = CUDA_FILES + ORACLE_CUDA_FILES

# One entry per normative ledger occurrence.  Duplicate same-line occurrences
# remain distinct IDs by design.
LEDGER = (
    ("G13-01", "lumina_cuda.cu", r"plasma->T_e\[s\].*plasma->T_rad\[s\]"),
    ("G13-02", "lumina_cuda.cu", r"T_rad"),
    ("G13-03", "lumina_cuda.cu", r"T_rad"),
    ("G13-04", "lumina_cuda.cu", r"T_rad"),
    ("G13-05", "lumina_cuda.cu", r"T_e"),
    ("G13-06", "lumina_cuda.cu", r"T_rad"),
    ("G13-07", "lumina_cuda.cu", r"\bW\b"),
    ("G13-08", "lumina_cuda.cu", r"T_rad"),
    ("G14-01", "lumina_bf_gemm.cu", r"T_rad"),
    ("G14-02", "lumina_bf_gemm.cu", r"\bW\b"),
    ("G14-03", "lumina_bf_gemm.cu", r"d_T_rad"),
    ("G14-04", "lumina_bf_gemm.cu", r"d_W"),
    ("G14-05", "lumina_bf_gemm.cu", r"d_T_rad"),
    ("G14-06", "lumina_bf_gemm.cu", r"d_W"),
    ("G14-07", "lumina_bf_gemm.cu", r"T_rad"),
    ("G14-08", "lumina_bf_gemm.cu", r"\bW\b"),
    ("G14-09", "lumina_bf_gemm.cu", r"d_T_rad"),
    ("G14-10", "lumina_bf_gemm.cu", r"d_W"),
    ("G14-11", "lumina_nlte_assemble.cu", r"a_planck_bnu|dilute"),
    ("G14-12", "lumina_nlte_assemble.cu", r"plasma->W|d_W"),
    ("G14-13", "lumina_nlte_assemble.cu", r"T_rad\[0\]"),
    ("G15-01", "lumina_cuda.cu", r"d_bf_absorption_event"),
    ("G15-02", "lumina_cuda.cu", r"d_T_rad"),
    ("G15-03", "lumina_cuda.cu", r"d_T_rad"),
    ("G15-04", "lumina_cuda.cu", r"d_bf_absorption_event"),
)
TRACE_RE = re.compile(
    r"\b(?:T_rad|d_T_rad|W|d_W|J_nu|d_J_nu|j_blue|d_jbar_line|"
    r"planck|Planck|chi_|eta_|transition_probabilities|kpacket_cdf)\b"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "report"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    for path in TRACE_FILES:
        if not path.is_file():
            errors.append(f"missing required CUDA TU: {path.name}")

    ledger_rows = []
    for stage_id, filename, pattern in LEDGER:
        text = (ROOT / "src" / filename).read_text(errors="replace")
        matches = list(re.finditer(pattern, text))
        terminal = ("CLOSED_A2_13_GPU_RATE" if stage_id.startswith("G13-") else
                    "CLOSED_A2_14_GPU_SIGNED_OPACITY" if stage_id.startswith("G14-") else
                    "CLOSED_A2_15_GPU_EMISSIVITY")
        ledger_rows.append({
            "id": stage_id,
            "file": f"src/{filename}",
            "pattern": pattern,
            "match_count": len(matches),
            "disposition": terminal,
        })
        if not matches:
            # Removal is a valid final disposition; absence is not an error.
            pass

    trace = []
    for path in TRACE_FILES:
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for match in TRACE_RE.finditer(line):
                trace.append({"file": f"src/{path.name}", "line": lineno,
                              "token": match.group(0), "text": line.strip()[:240]})

    wiring = (ROOT / "scripts/run_gate_battery.py").read_text()
    z_direct_links = wiring.count('"src/gpu_physics_contract.c"')
    if z_direct_links < 5:  # four hard-coded Z builds plus its own fixture
        errors.append(f"gpu_physics_contract Z direct links={z_direct_links}, expected>=5")
    runner = (ROOT / "scripts/run_zinert_selftest.py").read_text()
    if "--a2-13-15-contract" not in runner or \
       "a2-13-15-gpu-physics-contract" not in runner:
        errors.append("run_zinert_selftest A2-13~15 row not wired")

    contract = (ROOT / "src/gpu_physics_contract.c").read_text()
    if "bf_cpu_oracle_pass && i->bb_cpu_oracle_pass" not in contract:
        errors.append("BF/BB conjunction missing")
    production = {
        p.name: p.read_text(errors="replace") for p in CUDA_FILES
    }
    if "gpu_radiation_field_production_bind(" not in production["lumina_cuda.cu"]:
        errors.append("production A2-12 mirror bind missing")
    if "canonical_jnu_to_float" not in production["lumina_nlte_gemm.cu"] or \
       "gpu_radiation_field_production_view(" not in production["lumina_nlte_gemm.cu"]:
        errors.append("BF canonical global-grid device view missing")
    if "radiation_view.line_jbar" not in production["lumina_nlte_assemble.cu"] or \
       "radiation_view.line_id" not in production["lumina_nlte_assemble.cu"]:
        errors.append("BB LineJbarCache device lookup missing")
    if "gpu_opacity_production_bind(" not in production["lumina_cuda.cu"] or \
       "opacity_view_ce.bf_event_measure" not in production["lumina_cuda.cu"]:
        errors.append("A2-14 checked signed/event-measure production view missing")
    emissivity_header = (ROOT / "src/gpu_emissivity_kernels.h").read_text()
    if ("gpu_emissivity_production_bind(" not in production["lumina_cuda.cu"] or
            "gpu_emissivity_sample_device" not in emissivity_header):
        errors.append("A2-15 checked emissivity/CDF transport view missing")
    if "GPU_OPACITY_NOT_MIGRATED" in production["lumina_bf_gemm.cu"]:
        errors.append("A2-14 production guard remains")
    if "GPU_EMISSIVITY_NOT_MIGRATED" in production["lumina_cuda.cu"]:
        errors.append("A2-15 production guard remains")
    bf_code = re.sub(r"/\*.*?\*/|//.*?$", "", production["lumina_bf_gemm.cu"],
                     flags=re.MULTILINE | re.DOTALL)
    for forbidden in ("d_T_rad", "d_W", "plasma->T_rad", "plasma->W"):
        if forbidden in bf_code:
            errors.append(f"A2-14 BF scalar read remains: {forbidden}")
    if "fine_correct_R_bf();" in production["lumina_nlte_gemm.cu"] or \
       "nlte_rates_gpu_set_fine(" in production["lumina_cuda.cu"]:
        errors.append("forbidden production fine-grid BF call remains")
    asm = production["lumina_nlte_assemble.cu"]
    asm_code = re.sub(r"/\*.*?\*/|//.*?$", "", asm,
                      flags=re.MULTILINE | re.DOTALL)
    for forbidden in ("plasma->W", "plasma->T_rad", "d_W", "d_T_rad"):
        if forbidden in asm_code:
            errors.append(f"BB production scalar read remains: {forbidden}")
    report = {
        "schema": "A2_13_15_STATIC_CENSUS_V1",
        "status": "PASS" if not errors else "FAIL",
        "required_cuda_files": [f"src/{p.name}" for p in CUDA_FILES],
        "oracle_cuda_files": [f"src/{p.name}" for p in ORACLE_CUDA_FILES],
        "source_sha256": {f"src/{p.name}": sha(p) for p in TRACE_FILES},
        "ledger_rows_expected": 25,
        "ledger_rows": ledger_rows,
        "outside_ledger_grep_command":
            "rg -n 'T_rad|d_T_rad|W|d_W|J_nu|d_J_nu|j_blue|d_jbar_line|planck|chi_|eta_|transition_probabilities|kpacket_cdf' src/*.cu",
        "outside_ledger_raw_hit_count": len(trace),
        "outside_ledger_raw_hits": trace,
        "z_direct_link_occurrences": z_direct_links,
        "production_wiring": {
            "a2_12_single_mirror_bind": "gpu_radiation_field_production_bind(" in production["lumina_cuda.cu"],
            "bf_global_grid_device_view": "canonical_jnu_to_float" in production["lumina_nlte_gemm.cu"],
            "bb_line_cache_device_lookup": "radiation_view.line_jbar" in asm,
            "fine_grid_production_calls": production["lumina_nlte_gemm.cu"].count("fine_correct_R_bf();") + production["lumina_cuda.cu"].count("nlte_rates_gpu_set_fine("),
            "bb_scalar_reads": sum(asm_code.count(x) for x in ("plasma->W", "plasma->T_rad", "d_W", "d_T_rad")),
            "bb_scalar_comment_tombstones": sum(asm.count(x) - asm_code.count(x) for x in ("plasma->W", "plasma->T_rad", "d_W", "d_T_rad")),
            "a2_14_production_guard": "GPU_OPACITY_NOT_MIGRATED" in production["lumina_bf_gemm.cu"],
            "a2_15_production_guard": "GPU_EMISSIVITY_NOT_MIGRATED" in production["lumina_cuda.cu"],
            "a2_14_checked_production_view": "gpu_opacity_production_bind(" in production["lumina_cuda.cu"],
            "a2_15_checked_production_view": "gpu_emissivity_production_bind(" in production["lumina_cuda.cu"],
            "a2_15_cdf_sampler": "gpu_emissivity_sample_device" in production["lumina_cuda.cu"],
        },
        "errors": errors,
    }
    output = args.output or ROOT / "validation/a2_13_15/static_census.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{report['status']} A2_13_15_CENSUS ledger=25/25 "
          f"cuda_files={len(TRACE_FILES)}/8 raw_hits={len(trace)} Z={z_direct_links}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
