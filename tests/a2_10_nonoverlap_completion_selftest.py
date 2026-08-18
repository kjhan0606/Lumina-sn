#!/usr/bin/env python3
"""Positive fixture and isolated negative controls for the A2-10 final audit."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/finalize_a210_nonoverlap_gate.py"
ZERO_ENV = (
    "LUMINA_NLTE_LTE_FLOOR",
    "LUMINA_NLTE_FLOOR_MODE",
    "LUMINA_NLTE_FLOOR_REG",
    "LUMINA_NLTE_BK_CEIL",
    "LUMINA_NLTE_INV_CEIL",
    "LUMINA_NLTE_COLL_FLOOR",
    "LUMINA_DR_FLOOR_CMS",
    "LUMINA_STAGE4_BK_CAP",
    "LUMINA_HRESP_CLAMP",
    "LUMINA_TE_STEP_CLAMP",
    "LUMINA_J_CAP_FACTOR",
    "LUMINA_J_FLOOR_FACTOR",
    "LUMINA_RADEQ_LINE_CULL",
    "LUMINA_NLTE_GREY_TAU",
    "LUMINA_NLTE_ASSEMBLE_GPU",
    "LUMINA_NLTE_FALLBACK_TE",
)
FOUR_PI = 12.56637061435917295385057353311801153679


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_witness_line(
    phase: str,
    line: int,
    shell: int,
    eta: float,
    chi: float,
    jbar: float,
    bound: float,
    deck_scale: float,
) -> tuple[str, float, float]:
    factor = FOUR_PI * deck_scale
    signed_rate = math.fma(-chi, jbar, eta) * factor
    uncertainty = math.fma(abs(chi), bound, 0.0) * factor
    status = "OK_COOLING" if signed_rate > 0.0 else "OK_HEATING"
    text = (
        "[A2-10][LINE-NET-CELL-FINITE] "
        f"phase={phase} line={line} shell={shell} "
        f"eta_per_sr={eta:.17g} chi_effective={chi:.17g} "
        f"Jbar={jbar:.17g} Jbar_local_bound={bound:.17g} "
        f"signed_rate={signed_rate:.17g} uncertainty={uncertainty:.17g} "
        f"deck_scale={deck_scale:.17g} status={status} requested_cell=0 "
        "floor=0 cap=0 clamp=0 jitter=0 repair=0\n"
    )
    return text, signed_rate, uncertainty


def build_fixture(root: Path, refinements: int = 24) -> tuple[Path, Path, Path]:
    run = root / "run"
    input_dir = run / "input"
    model = input_dir / "model"
    atomic = input_dir / "global_atomic"
    manual = run / "manual_control"
    for path in (model, atomic, manual):
        path.mkdir(parents=True)

    (model / "deck.dat").write_bytes(b"sealed-deck\n")
    (atomic / "levels.csv").write_bytes(b"sealed-atomic\n")
    (input_dir / "deck.sha256").write_text(
        f"{digest(model / 'deck.dat')}  deck.dat\n", encoding="utf-8"
    )
    (input_dir / "topion.sha256").write_text(
        f"{digest(atomic / 'levels.csv')}  levels.csv\n", encoding="utf-8"
    )
    (input_dir / "lumina_cuda").write_bytes(b"sealed-binary\n")
    (input_dir / "binary.sha256").write_text(
        f"{digest(input_dir / 'lumina_cuda')}\n", encoding="utf-8"
    )
    (input_dir / "precore_tau_refresh.txt").write_text("0\n", encoding="utf-8")
    exports = [f'declare -x {name}="0"' for name in ZERO_ENV]
    exports.append('declare -x LUMINA_CMF_FINE_MGPU_DEVICES="2"')
    exports.append(
        f'declare -x LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS="{refinements}"'
    )
    (input_dir / "resolved_lumina.exports").write_text(
        "\n".join(exports) + "\n", encoding="utf-8"
    )

    stdout = run / "stdout.log"
    stderr = run / "stderr.log"
    stdout.write_text("gate floor=0 cap=0 clamp=0 jitter=0 repair=0\n", encoding="utf-8")
    lower = {
        "phase": "LOWER", "line": 1_154_618, "shell": 5,
        "eta": 8.6353957108970182e-17, "chi": 2.8522910357512541e-12,
        "jbar": 3.02753056515078e-05, "deck": 0.9999999227244345,
        "k24_bound": 1.1274339633283025e-11, "k30_bound": 5.0e-12,
        "required": 9.04727269265495e-12,
    }
    upper = {
        "phase": "UPPER", "line": 894_169, "shell": 27,
        "eta": 1.9388923091568966e-35, "chi": 7.0455165542937056e-29,
        "jbar": 2.7519483330291897e-07, "deck": 0.999999853229645,
        "k24_bound": 6.4701861976580376e-13, "k30_bound": 2.0e-13,
        "required": 3.627079809671681e-13,
    }
    finite_lines: list[str] = []
    for item in (lower, upper):
        line_text, _, _ = finite_witness_line(
            str(item["phase"]), int(item["line"]), int(item["shell"]),
            float(item["eta"]), float(item["chi"]), float(item["jbar"]),
            float(item["k30_bound"]), float(item["deck"]),
        )
        finite_lines.append(line_text)
    stderr.write_text(
        "[cmf_fine][EXACT-MULTIGPU-EPOCH] status=OK cap=64 "
        "floor=0 clamp=0 jitter=0\n"
        "physics floor=0 cap=0 clamp=0 jitter=0 repair=0\n" +
        "".join(finite_lines),
        encoding="utf-8",
    )
    (run / "model.rc").write_text("0\n", encoding="utf-8")
    (run / "TARGETED_GATE_VERDICT.txt").write_text(
        "A210_TARGETED_GATE_ACCEPT status=PASS job_id=fake run_root=fake\n",
        encoding="utf-8",
    )
    (run / "RUN_FOOTER.txt").write_text(
        "outer_iterations=1\n"
        "diagnostic_mode=A210_TARGETED_GATE\n"
        f"envelope_refinements={refinements}\n"
        "LUMINA_CMF_FINE_MGPU_DEVICES=2\n",
        encoding="utf-8",
    )
    targeted = {
        "schema": "LUMINA_A210_TARGETED_GATE_V3",
        "status": "PASS",
        "expected_devices": 2,
        "expected_refinements": refinements,
        "exact_publications": 2,
        "r6_radiation_generations": [1, 2],
        "seed_material_predictor_commit": True,
        "line_operator": "CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0",
        "sobolev_jbar_cells": 109_014_300,
        "r7_material_commit": True,
        "physics_comparison_commit": True,
        "cancellation_census_present": False,
        "physical_values_modified_by_numerical_repair": False,
        "r2_signed_material": {
            "signed_cells": 22_866_166,
            "exact_zero_tau": 86_148_134,
            "raw_negative": 4_246_581,
            "mild_negative": 4_246_577,
            "srce_chk": 4,
        },
        "repair": 0,
    }
    write_json(run / "a210_targeted_gate_report.json", targeted)
    write_json(run / "a210_targeted_snapshot_report.json", {
        "schema": "LUMINA_DET_CONVERGENCE_V1",
        "status": "CONVERGED",
        "expected_iterations": 1,
        "tail_transitions": 0,
        "expected_bins": 1234,
        "transitions": [],
    })

    reference_log = root / "reference.log"
    reference_log.write_text("sealed R1 reference\n", encoding="utf-8")
    proof_mode = refinements != 24
    write_json(run / "r1_k24_reference_comparison.json", {
        "schema": "LUMINA_A210_TARGETED_REFERENCE_COMPARISON_V1",
        "status": "PASS",
        "reason": (
            "PHYSICAL_AND_SOLVER_FIELDS_BIT_EXACT_PROOF_BOUNDS_CONTRACTED"
            if proof_mode else "EXACT_AND_R6_FIELDS_BIT_EXACT"
        ),
        "comparison_mode": "PROOF_REFINEMENT_ONLY" if proof_mode else "STRICT_BIT_EXACT",
        "reference_refinements": 24 if proof_mode else None,
        "candidate_refinements": refinements if proof_mode else None,
        "proof_bounds_nonincreasing": True if proof_mode else None,
        "proof_field_changes": ([
            {
                "record": "[cmf_fine][EXACT-MULTIGPU-EPOCH]",
                "field": "component_error", "reference": "[1e-9,2e-9]",
                "candidate": "[1e-9,1e-9]", "upper_bound_ratio": 0.5,
            },
            {
                "record": "[cmf_fine][EXACT-MULTIGPU-EPOCH]",
                "field": "refinements", "reference": "24",
                "candidate": str(refinements),
            },
            {
                "record": "[R6][LINE-IDENTITY]",
                "field": "component_error", "reference": "[1e-9,2e-9]",
                "candidate": "[1e-9,1e-9]", "upper_bound_ratio": 0.5,
            },
            {
                "record": "[R6][LINE-IDENTITY]",
                "field": "profile_error", "reference": "[1e-9,4e-9]",
                "candidate": "[1e-9,2e-9]", "upper_bound_ratio": 0.5,
            },
            {
                "record": "[R6][LINE-IDENTITY]",
                "field": "refinements", "reference": "24",
                "candidate": str(refinements),
            },
        ] if proof_mode else []),
        "differences": [],
        "reference_occurrence": 0,
        "candidate_occurrence": 0,
        "reference_stderr": str(reference_log),
        "reference_sha256": digest(reference_log),
        "candidate_stderr": str(stderr),
        "candidate_sha256": digest(stderr),
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "repair": 0,
    })

    (manual / "COMPLETED").touch()
    (manual / "child.rc").write_text("0\n", encoding="utf-8")
    (manual / "supervisor.log").write_text(
        "MANUAL_DET_TRIPWIRE status=START gpu_indices=5,6\n"
        "MANUAL_DET_TRIPWIRE gpu_preflight=5, NVIDIA A100-SXM4-80GB\n"
        "MANUAL_DET_TRIPWIRE gpu_preflight=6, NVIDIA A100-SXM4-80GB\n"
        "MANUAL_DET_TRIPWIRE status=CHILD_STARTED child_pid=100\n"
        "MANUAL_DET_TRIPWIRE status=COMPLETED child_rc=0\n",
        encoding="utf-8",
    )

    baseline = root / "k12.csv"
    candidate = root / "k18.csv"
    header = "phase,line,shell,status\n"
    baseline.write_text(
        header + "".join(f"LOWER,{index},0,UNRESOLVED\n" for index in range(19)),
        encoding="utf-8",
    )
    candidate.write_text(header, encoding="utf-8")
    refinement = root / "refinement.json"
    write_json(refinement, {
        "schema": "lumina-a2-10-refinement-only-comparison-v1",
        "status": "PASS",
        "baseline": {
            "csv": str(baseline), "sha256": digest(baseline),
            "refinements": 12, "unresolved": 19,
        },
        "candidate": {
            "csv": str(candidate), "sha256": digest(candidate),
            "refinements": 18, "unresolved": 0,
        },
        "surviving": [],
        "surviving_count": 0,
        "physical_values_modified": False,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
    })
    k24_log = root / "k24.stderr.log"
    k24_log.write_text("sealed K24 witness source\n", encoding="utf-8")
    proof_baseline = root / "proof_baseline.json"
    proof_witnesses = []
    for item in (lower, upper):
        k24_line, k24_rate, k24_uncertainty = finite_witness_line(
            str(item["phase"]), int(item["line"]), int(item["shell"]),
            float(item["eta"]), float(item["chi"]), float(item["jbar"]),
            float(item["k24_bound"]), float(item["deck"]),
        )
        del k24_line
        proof_witnesses.append({
            "phase": item["phase"],
            "line": item["line"],
            "shell": item["shell"],
            "status": "UNRESOLVED_CANCELLATION",
            "inputs": {
                "eta_per_sr": str(item["eta"]),
                "chi_effective": str(item["chi"]),
                "jbar": str(item["jbar"]),
                "jbar_bound": str(item["k24_bound"]),
                "deck_scale": str(item["deck"]),
            },
            "reconstructed": {
                "signed_rate": str(k24_rate),
                "absolute_uncertainty": str(k24_uncertainty),
            },
            "proof_requirement": {
                "required_symmetric_jbar_bound_strictly_below":
                    str(item["required"]),
            },
            "repair_counters": {"floor": 0, "clamp": 0, "jitter": 0},
        })
    write_json(proof_baseline, {
        "schema": "lumina-a2-10-cancellation-witness-audit-v1",
        "status": "PASS",
        "witness_count": 2,
        "observed_line_shell": [[894_169, 27], [1_154_618, 5]],
        "source_log": str(k24_log),
        "source_sha256": digest(k24_log),
        "physical_values_modified": False,
        "floor": 0,
        "cap": 0,
        "clamp": 0,
        "jitter": 0,
        "witnesses": proof_witnesses,
    })
    return run, refinement, proof_baseline


def invoke(
    root: Path,
    mutate: str | None = None,
    refinements: int = 24,
    expected_refinements: int | None = None,
) -> tuple[int, dict[str, object]]:
    run, refinement, proof_baseline = build_fixture(root, refinements)
    if mutate == "repair_env":
        path = run / "input/resolved_lumina.exports"
        path.write_text(
            path.read_text().replace('LUMINA_J_CAP_FACTOR="0"', 'LUMINA_J_CAP_FACTOR="1"'),
            encoding="utf-8",
        )
    elif mutate == "k18_row":
        path = root / "k18.csv"
        path.write_text(path.read_text() + "UPPER,1,0,UNRESOLVED\n", encoding="utf-8")
    elif mutate == "tripwire_yield":
        (run / "manual_control/YIELDED").touch()
    elif mutate == "candidate_changed":
        (run / "stderr.log").write_text("changed repair=0\n", encoding="utf-8")
    elif mutate == "operator":
        path = run / "a210_targeted_gate_report.json"
        value = json.loads(path.read_text())
        value["line_operator"] = "SHARED_GAUSSIAN"
        write_json(path, value)
    elif mutate == "proof_required":
        value = json.loads(proof_baseline.read_text())
        value["witnesses"][0]["proof_requirement"][
            "required_symmetric_jbar_bound_strictly_below"
        ] = "1e-30"
        write_json(proof_baseline, value)
    elif mutate == "proof_change_set":
        path = run / "r1_k24_reference_comparison.json"
        value = json.loads(path.read_text())
        value["proof_field_changes"][0]["record"] = "[R6][LINE-COVERAGE]"
        write_json(path, value)
    report = root / "completion.json"
    if expected_refinements is None:
        expected_refinements = refinements
    result = subprocess.run(
        (
            sys.executable, str(AUDITOR),
            "--run-root", str(run),
            "--refinement-comparison", str(refinement),
            "--proof-witness-baseline", str(proof_baseline),
            "--expected-refinements", str(expected_refinements),
            "--report", str(report),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode, json.loads(report.read_text(encoding="utf-8"))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a210-completion-") as raw:
        rc, report = invoke(Path(raw) / "positive")
        if rc != 0 or report.get("status") != "PASS":
            print("FAIL A2_10_NONOVERLAP_COMPLETION_SELFTEST positive")
            return 4
        rc, report = invoke(Path(raw) / "positive_k30", refinements=30)
        if rc != 0 or report.get("status") != "PASS":
            print("FAIL A2_10_NONOVERLAP_COMPLETION_SELFTEST positive_k30")
            return 4
        controls = (
            "repair_env", "k18_row", "tripwire_yield", "candidate_changed",
            "operator", "proof_required",
        )
        for name in controls:
            rc, report = invoke(Path(raw) / name, name)
            if rc != 4 or report.get("status") != "FAIL":
                print(f"FAIL A2_10_NONOVERLAP_COMPLETION_SELFTEST negative={name}")
                return 4
        rc, report = invoke(
            Path(raw) / "proof_change_set", "proof_change_set", refinements=30
        )
        if rc != 4 or report.get("status") != "FAIL":
            print("FAIL A2_10_NONOVERLAP_COMPLETION_SELFTEST negative=proof_change_set")
            return 4
    print(
        "PASS A2_10_NONOVERLAP_COMPLETION_SELFTEST "
        "positive=2 refinements=24,30 negative_controls=7 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
