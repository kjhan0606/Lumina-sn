import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_cmfgen_fixedj_heat_capture as runner


RESULT_SHA = "be3bf940802800c7df94396e1fbd6a8828aa444a9c444082582179349f1eaf47"
RESPONSE_SHA = "252993d3da03dd43ede0d02908c5134896aea8df902b9a54f947027b9556436e"
J_SHA = "40a48d9aedaafa4d8f6f8adb0fdb6ea191bd152ed3f55de38e8862fe445232a6"
ND, NT = 90, 1800


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


def make_attempt(root):
    build, inputs, run = (root / name for name in ("build", "inputs", "native"))
    for path in (build, inputs, run):
        path.mkdir(parents=True)
    (run / "OUTGEN").write_text("baseline model output\n")
    exe = build / "cmfgen_fixedj_heat.exe"
    exe.write_bytes(b"sealed synthetic executable")
    exe.chmod(0o755)
    write_json(build / "fixedj-heat-build.json", {
        "schema": "lumina-native-fixedj-integral-re-build-v1",
        "output_sha256": {exe.name: digest(exe)},
    })

    radius = np.linspace(90.0, 1.0, ND, dtype="<f8")
    pops = np.ones((NT, ND), dtype="<f8", order="F")
    pops[-2] = 2.0e5
    pops[-1] = 1.4
    state = run / "NATIVE_HEAT_INPUT.bin"
    state.write_bytes(b"NATIVEHEATINPV1 " + struct.pack("<4i", 1, 64, NT, ND)
                      + radius.tobytes() + pops.tobytes(order="F"))
    write_json(inputs / "heat-state-input.json", {
        "schema": "lumina-native-heat-state-input-v1", "points": ND, "variables": NT,
        "accepted_result_sha256": RESULT_SHA, "accepted_response_sha256": RESPONSE_SHA,
        "output_sha256": {state.name: digest(state)},
    })

    field = {}
    for name, payload in (("EDDFACTOR", b"current J factor"),
                          ("EDDFACTOR_INFO", b"current J metadata")):
        (run / name).write_bytes(payload)
        field[name] = digest(run / name)
    write_json(inputs / "fixedj-field-input.json", {
        "schema": "lumina-native-current-fixedj-input-v1",
        "layout": {"depths": ND, "frequencies": 1_115_020},
        "input_sha256": {"/sealed/fixed-field/J.npy": J_SHA}, "output_sha256": field,
    })
    (run / "VADAT").write_text(
        "T [COH_ES]\nT [USE_FIXED_J]\nF [FLUX_CAL_ONLY]\nT [WRITE_RATES]\n")
    return root


def fake_srun(bin_dir):
    executable = bin_dir / "srun"
    executable.write_text('''#!/usr/bin/env python3
import os, struct, sys
from pathlib import Path
import numpy as np

assert os.environ.get("LUMINA_FIXEDJ_HEAT_ONLY") == "1"
assert sys.argv[1:3] == ["--cpu-bind=cores", "--export=ALL,LUMINA_FIXEDJ_HEAT_ONLY=1"]
raw = Path("NATIVE_HEAT_INPUT.bin").read_bytes()
_, _, nt, nd = struct.unpack("<4i", raw[16:32])
values = np.frombuffer(raw[32:], dtype="<f8")
radius = values[:nd].copy()
pops = values[nd:].reshape((nt, nd), order="F").copy()
vectors = [radius, np.ones(nd), np.zeros(nd), pops[-1], pops[-2],
           np.ones(nd), np.ones(nd), np.zeros(nd), np.zeros(nd)]
header = b"NATIVEHEATROWV1 " + struct.pack("<8i", 1, 64, nt, nd, 1115020, 65, 9, 0)
body = np.concatenate([*(v.astype("<f8", copy=False) for v in vectors),
                       pops.ravel(order="F")]).astype("<f8", copy=False)
Path("NATIVE_HEAT_ROW.bin").write_bytes(header + body.tobytes())
for name in ("LINEHEAT", "NETRATE", "TOTRATE"):
    Path(name).write_bytes(b"synthetic output\\n")
with Path("OUTGEN").open("a") as stream:
    stream.write("FIXED_J_INTEGRAL_RE_CAPTURED_BEFORE_POP_UPDATE_NOT_EHB_OR_CONVERGENCE\\n")
raise SystemExit(96)
''')
    executable.chmod(0o755)


def test_slurm_wrapper_survives_spooled_script_path_and_exports_guard(tmp_path, monkeypatch):
    root = make_attempt(tmp_path / "attempt")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_srun(bin_dir)
    module = bin_dir / "module"
    module.write_text("#!/bin/sh\nexit 0\n")
    module.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "32768")
    monkeypatch.setenv("SLURM_JOB_ID", "987654")
    monkeypatch.delenv("LUMINA_FIXEDJ_HEAT_ONLY", raising=False)

    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", str(SCRIPTS / "submit_cmfgen_fixedj_heat_capture.slurm"),
         str(root), str(SCRIPTS.parent)],
        text=True, capture_output=True, check=False, timeout=30)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((root / "native/seq_logs/fixedj-heat-runtime.json").read_text())
    assert manifest["status"] == "CAPTURED"
    assert manifest["captured_state_exact"] is True
    assert manifest["completion_marker_source"] == "OUTGEN"
    assert set(manifest["outputs"]) == set(runner.OUTPUT_NAMES)
    assert manifest["srun_exit_code"] == 96


def test_capture_runner_refuses_preexisting_native_outputs(tmp_path):
    root = make_attempt(tmp_path / "attempt")
    (root / "native/LINEHEAT").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        runner.validate_inputs(root)


def test_capture_runner_refuses_stale_marker_in_outgen(tmp_path):
    root = make_attempt(tmp_path / "attempt")
    (root / "native/OUTGEN").write_text(runner.SUCCESS_MARKER + "\n")
    with pytest.raises(FileExistsError, match="stale completion evidence"):
        runner.validate_inputs(root)


def test_capture_runner_requires_exact_slurm_allocation(tmp_path, monkeypatch):
    root = make_attempt(tmp_path / "attempt")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "32768")
    monkeypatch.setenv("SLURM_JOB_ID", "987654")
    with pytest.raises(ValueError, match="CPU16/32GiB"):
        runner.run(root)
