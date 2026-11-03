#!/usr/bin/env python3
"""Run the approved, isolated fixed-J CMFGEN line-heat capture once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from prepare_cmfgen_fixedj_heat import read_heat_row
from compare_cmfgen_fixedj_full import read_state


ACTIVATION = "LUMINA_FIXEDJ_HEAT_ONLY"
EXPECTED_STOP = 96
SUCCESS_MARKER = "FIXED_J_INTEGRAL_RE_CAPTURED_BEFORE_POP_UPDATE_NOT_EHB_OR_CONVERGENCE"
OUTPUT_NAMES = ("LINEHEAT", "NETRATE", "TOTRATE", "NATIVE_HEAT_ROW.bin")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def control(vadat: str, key: str) -> str:
    values = re.findall(r"^(?!\s*[!#])\s*(\S+)\s+\[" + re.escape(key) + r"\]", vadat, re.M)
    if len(values) != 1:
        raise ValueError(f"VADAT control missing/duplicated: {key}")
    return values[0]


def validate_inputs(root: Path) -> dict:
    build = root / "build"
    inputs = root / "inputs"
    run = root / "native"
    exe = build / "cmfgen_fixedj_heat.exe"
    build_manifest_path = build / "fixedj-heat-build.json"
    state_manifest_path = inputs / "heat-state-input.json"
    field_manifest_path = inputs / "fixedj-field-input.json"
    required = (exe, build_manifest_path, state_manifest_path, field_manifest_path,
                run / "NATIVE_HEAT_INPUT.bin", run / "EDDFACTOR", run / "EDDFACTOR_INFO",
                run / "VADAT")
    if not root.is_dir() or any(not path.is_file() for path in required):
        raise ValueError("attempt root is missing a sealed build/input/run artifact")
    if not os.access(exe, os.X_OK):
        raise ValueError("fixed-J executable is not executable")

    build_manifest = json.loads(build_manifest_path.read_text())
    state_manifest = json.loads(state_manifest_path.read_text())
    field_manifest = json.loads(field_manifest_path.read_text())
    if (build_manifest.get("schema") != "lumina-native-fixedj-integral-re-build-v1"
            or state_manifest.get("schema") != "lumina-native-heat-state-input-v1"
            or field_manifest.get("schema") != "lumina-native-current-fixedj-input-v1"):
        raise ValueError("unexpected sealed manifest schema")
    expected_exe = build_manifest.get("output_sha256", {}).get(exe.name)
    if not expected_exe or sha(exe) != expected_exe:
        raise ValueError("fixed-J executable differs from its build manifest")

    for name, manifest in (("NATIVE_HEAT_INPUT.bin", state_manifest),
                           ("EDDFACTOR", field_manifest),
                           ("EDDFACTOR_INFO", field_manifest)):
        expected = manifest.get("output_sha256", {}).get(name)
        if not expected or sha(run / name) != expected:
            raise ValueError(f"sealed input mismatch: {name}")
    j_hashes = [digest for name, digest in field_manifest.get("input_sha256", {}).items()
                if Path(name).name == "J.npy"]
    if (state_manifest.get("points") != 90 or state_manifest.get("variables") != 1800
            or field_manifest.get("layout", {}).get("depths") != 90
            or field_manifest.get("layout", {}).get("frequencies") != 1_115_020
            or len(j_hashes) != 1
            or j_hashes[0] !=
            "40a48d9aedaafa4d8f6f8adb0fdb6ea191bd152ed3f55de38e8862fe445232a6"):
        raise ValueError("not the approved generation14/evaluation4 fixed-J state")
    if (state_manifest.get("accepted_result_sha256") !=
            "be3bf940802800c7df94396e1fbd6a8828aa444a9c444082582179349f1eaf47"
            or state_manifest.get("accepted_response_sha256") !=
            "252993d3da03dd43ede0d02908c5134896aea8df902b9a54f947027b9556436e"):
        raise ValueError("accepted material/response identity differs from approval")

    vadat = (run / "VADAT").read_text()
    expected_controls = {"COH_ES": "T", "USE_FIXED_J": "T", "FLUX_CAL_ONLY": "F",
                         "WRITE_RATES": "T"}
    actual_controls = {key: control(vadat, key) for key in expected_controls}
    if actual_controls != expected_controls:
        raise ValueError(f"fixed-J capture VADAT controls differ: {actual_controls}")
    run_logs = run / "seq_logs"
    run_logs.mkdir(exist_ok=True)
    expected_outputs = ("LINEHEAT", "NETRATE", "TOTRATE", "NATIVE_HEAT_ROW.bin")
    runtime_manifest = run_logs / "fixedj-heat-runtime.json"
    stdout = run_logs / "fixedj-heat-capture.stdout"
    stderr = run_logs / "fixedj-heat-capture.stderr"
    if runtime_manifest.exists() or stdout.exists() or stderr.exists() or any((run / n).exists() for n in expected_outputs):
        raise FileExistsError("capture outputs already exist; refusing to overwrite")
    return {"exe": exe, "run": run, "state_manifest": state_manifest,
            "field_manifest": field_manifest, "build_manifest": build_manifest,
            "runtime_manifest": runtime_manifest, "stdout": stdout, "stderr": stderr,
            "controls": actual_controls, "fixed_j_sha256": j_hashes[0]}


def verify_outputs(artifacts: dict, job_id: str, returncode: int, elapsed: float) -> dict:
    run = artifacts["run"]
    missing = [name for name in OUTPUT_NAMES if not (run / name).is_file() or (run / name).stat().st_size == 0]
    stdout = artifacts["stdout"].read_text(errors="replace") if artifacts["stdout"].exists() else ""
    if returncode != EXPECTED_STOP:
        raise RuntimeError(f"CMFGEN stop code was {returncode}, expected {EXPECTED_STOP}")
    if SUCCESS_MARKER not in stdout:
        raise RuntimeError("expected pre-update capture marker is missing")
    if missing:
        raise RuntimeError("required native line/heat output missing: " + ", ".join(missing))

    row = read_heat_row(run / "NATIVE_HEAT_ROW.bin")
    state = read_state(run / "NATIVE_HEAT_INPUT.bin")
    if (not np.array_equal(row["radius_native"], state["radius"])
            or not np.array_equal(row["pops"], state["pops"])):
        raise ValueError("captured native material state differs from the sealed input")
    outputs = {name: {"bytes": (run / name).stat().st_size, "sha256": sha(run / name)}
               for name in OUTPUT_NAMES}
    return {"schema": "lumina-fixedj-heat-capture-runtime-v1", "status": "CAPTURED",
            "slurm_job_id": job_id, "srun_exit_code": returncode,
            "expected_stop": EXPECTED_STOP, "elapsed_seconds": elapsed,
            "activation_env": f"{ACTIVATION}=1 passed explicitly to srun",
            "controls": artifacts["controls"], "build_sha256": sha(artifacts["exe"]),
            "accepted_result_sha256": artifacts["state_manifest"]["accepted_result_sha256"],
            "accepted_response_sha256": artifacts["state_manifest"]["accepted_response_sha256"],
            "fixed_j_sha256": artifacts["fixed_j_sha256"],
            "captured_state_exact": True, "outputs": outputs,
            "new_transport": False, "material_update": False, "final_goal": "NOT_MET"}


def run(root: Path) -> int:
    root = root.resolve(strict=True)
    if (os.environ.get("SLURM_CPUS_PER_TASK") != "16"
            or os.environ.get("SLURM_MEM_PER_NODE") != "32768"
            or not os.environ.get("SLURM_JOB_ID")):
        raise ValueError("run only in the approved Grammar CPU16/32GiB Slurm allocation")
    artifacts = validate_inputs(root)
    env = os.environ.copy()
    env[ACTIVATION] = "1"
    env.update(OMP_NUM_THREADS="16", OMP_DYNAMIC="FALSE", OMP_STACKSIZE="512M",
               OMP_PROC_BIND="close", OMP_PLACES="cores", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    command = ["srun", "--cpu-bind=cores", "--export=ALL,LUMINA_FIXEDJ_HEAT_ONLY=1",
               str(artifacts["exe"])]
    started = time.monotonic()
    with artifacts["stdout"].open("xb") as stdout, artifacts["stderr"].open("xb") as stderr:
        completed = subprocess.run(command, cwd=artifacts["run"], env=env,
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   check=False)
    elapsed = time.monotonic() - started
    try:
        result = verify_outputs(artifacts, os.environ["SLURM_JOB_ID"], completed.returncode, elapsed)
        status = 0
    except Exception as error:
        result = {"schema": "lumina-fixedj-heat-capture-runtime-v1", "status": "FAILED",
                  "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "srun_exit_code": completed.returncode, "elapsed_seconds": elapsed,
                  "activation_env": f"{ACTIVATION}=1 passed explicitly to srun",
                  "failure": f"{type(error).__name__}: {error}", "final_goal": "NOT_MET"}
        status = 1
    with artifacts["runtime_manifest"].open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args.attempt_root)
    except Exception as error:
        print(f"fixed-J heat capture preflight failed: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
