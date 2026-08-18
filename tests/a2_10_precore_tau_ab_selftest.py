#!/usr/bin/env python3
"""Positive and negative controls for the pre-core tau A/B judge."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/compare_a210_precore_tau_ab.py"


def common() -> str:
    return "\n".join((
        "[cmf_fine][SIGNED-MATERIAL-CENSUS] line_shells=10 raw_negative=0 repair=0",
        "[cmf_fine][EXACT-MULTIGPU-EPOCH] status=OK devices=2/2 iterations=45 refinements=24 residual=9e-9 repair=0",
        "[R6][LINE-IDENTITY] q_lines=3 e_lines=4 refinements=24 residual=9e-9",
        "[R6][LINE-COVERAGE] valid_cells=200 partial_lines=0 unsampled_lines=0",
    ))


def baseline() -> str:
    lines = [common(), "[A2-10][VECTOR-NOBRACKET] count=4 same_negative=4"]
    for shell in range(4):
        lines.append(
            "[A2-10][VECTOR-INTERIOR-SCAN] phase=GEOMETRIC_MID "
            f"shell={shell} T_mid=22000 res_lo=-{shell + 1}.0 "
            f"res_mid=-{shell + 2}.0 res_hi=-{shell + 3}.0"
        )
    return "\n".join(lines) + "\n"


def candidate(no_bracket: bool = True, *, floor: int = 0) -> str:
    seed = (
        "[A2-10][PRECORE-TAU-SEED] status=DIAGNOSTIC_AB_ONLY "
        "source=TRIAL_LTE_IONIZATION rate_consumer=MODE3_BETA_JINC "
        "population_tau_fixed_point=0 public_mutation=0 "
        f"floor={floor} cap=0 clamp=0 jitter=0 repair=0"
    )
    lines = [common(), seed, seed]
    if no_bracket:
        lines.append("[A2-10][VECTOR-NOBRACKET] count=4 same_negative=4")
        for shell in range(4):
            lines.append(
                "[A2-10][VECTOR-INTERIOR-SCAN] phase=GEOMETRIC_MID "
                f"shell={shell} T_mid=22000 res_lo=-{shell + 0.5} "
                f"res_mid=-{shell + 1.0} res_hi=-{shell + 1.5}"
            )
    return "\n".join(lines) + "\n"


def seal_root(root: Path, *, seed: int, binary: bytes = b"same-binary") -> None:
    (root / "input").mkdir(parents=True)
    (root / "input/model").mkdir()
    (root / "input/global_atomic").mkdir()
    (root / "READY").write_text("READY\n", encoding="ascii")
    (root / "input/lumina_cuda").write_bytes(binary)
    binary_sha = hashlib.sha256(binary).hexdigest()
    (root / "input/binary.sha256").write_text(
        f"{binary_sha}\n", encoding="ascii"
    )
    model_files = {
        "DECK_PROVENANCE.json": b"{}\n",
        "cmfgen_sigma_bf.bin": b"same-sigma",
    }
    atomic_files = {
        "topion_ground_levels.csv": b"ground\n",
        "topion_levels.csv": b"levels\n",
        "ionization_reference.csv": b"ionization\n",
    }
    for name, payload in model_files.items():
        (root / "input/model" / name).write_bytes(payload)
    for name, payload in atomic_files.items():
        (root / "input/global_atomic" / name).write_bytes(payload)
    deck_manifest = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(model_files.items())
    )
    topion_manifest = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(atomic_files.items())
    )
    (root / "input/deck.sha256").write_text(deck_manifest, encoding="ascii")
    (root / "input/topion.sha256").write_text(topion_manifest, encoding="ascii")
    (root / "input/sigma.sha256").write_text(
        hashlib.sha256(model_files["cmfgen_sigma_bf.bin"]).hexdigest() + "\n",
        encoding="ascii",
    )
    for name, value in {
        "outer_iterations.txt": "1\n",
        "single_total.txt": "0\n",
        "stage4.txt": "0\n",
        "envelope_refinements.txt": "24\n",
        "diagnostic_mode.txt": "A210_TARGETED_GATE\n",
        "precore_tau_refresh.txt": f"{seed}\n",
    }.items():
        (root / "input" / name).write_text(value, encoding="ascii")
    env = [
        'declare -x LUMINA_NLTE_ION_LOCK="1"',
        'declare -x LUMINA_NLTE_PER_ION_RESCALE="1"',
        f'declare -x LUMINA_MODEL_DIR="{root}/input/model"',
    ]
    if seed:
        env.append('declare -x LUMINA_A210_PRECORE_TAU_REFRESH="1"')
    (root / "input/resolved_lumina.exports").write_text(
        "\n".join(sorted(env)) + "\n", encoding="utf-8"
    )


def run(directory: Path, case: str, text: str, rc: int, *,
        confound_env: bool = False,
        mismatch_binary: bool = False) -> tuple[int, dict[str, object]]:
    case_dir = directory / case
    base_root = case_dir / "baseline"
    candidate_root = case_dir / "candidate"
    seal_root(base_root, seed=0)
    seal_root(candidate_root, seed=1,
              binary=b"different-binary" if mismatch_binary else b"same-binary")
    if confound_env:
        path = candidate_root / "input/resolved_lumina.exports"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                'LUMINA_NLTE_ION_LOCK="1"', 'LUMINA_NLTE_ION_LOCK="0"'
            ),
            encoding="utf-8",
        )
    base = base_root / "stderr.log"
    trial = candidate_root / "stderr.log"
    baseline_model_rc = base_root / "model.rc"
    model_rc = candidate_root / "model.rc"
    report = case_dir / "report.json"
    base.write_text(baseline(), encoding="utf-8")
    trial.write_text(text, encoding="utf-8")
    baseline_model_rc.write_text("1\n", encoding="ascii")
    model_rc.write_text(f"{rc}\n", encoding="ascii")
    result = subprocess.run(
        (sys.executable, str(CHECKER),
         "--baseline-root", str(base_root),
         "--candidate-root", str(candidate_root),
         "--baseline-stderr", str(base),
         "--candidate-stderr", str(trial), "--candidate-model-rc",
         str(model_rc), "--baseline-model-rc", str(baseline_model_rc),
         "--report", str(report)),
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return result.returncode, json.loads(report.read_text(encoding="utf-8"))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="a210-precore-ab-") as raw:
        directory = Path(raw)
        proc_rc, report = run(directory, "persists", candidate(), 1)
        if (proc_rc != 0 or report.get("status") != "PASS" or
                report.get("outcome") != "NO_BRACKET_PERSISTS"):
            print("FAIL A2_10_PRECORE_TAU_AB_SELFTEST persists")
            return 4
        proc_rc, report = run(directory, "restored", candidate(False), 0)
        if (proc_rc != 0 or report.get("status") != "PASS" or
                report.get("outcome") != "BRACKET_RESTORED_GATE_PASS"):
            print("FAIL A2_10_PRECORE_TAU_AB_SELFTEST restored")
            return 4
        controls = {
            "missing_seed": (common() + "\n", 1, False, False),
            "seed_repair": (candidate(floor=1), 1, False, False),
            "identity": (candidate().replace("residual=9e-9", "residual=8e-9", 1), 1, False, False),
            "rc_without_reason": (candidate(False), 1, False, False),
            "confounded_env": (candidate(), 1, True, False),
            "binary_mismatch": (candidate(), 1, False, True),
        }
        for name, (text, model_rc, confound_env, mismatch_binary) in controls.items():
            proc_rc, report = run(
                directory, name, text, model_rc,
                confound_env=confound_env,
                mismatch_binary=mismatch_binary,
            )
            if proc_rc != 4 or report.get("status") != "FAIL":
                print(f"FAIL A2_10_PRECORE_TAU_AB_SELFTEST negative={name}")
                return 4
    print(
        "PASS A2_10_PRECORE_TAU_AB_SELFTEST outcomes=2 "
        "negative_controls=6 repair=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
