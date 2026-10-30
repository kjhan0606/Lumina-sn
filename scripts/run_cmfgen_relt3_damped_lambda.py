#!/usr/bin/env python3
"""Run five fixed-T LAMBDA steps from the sealed relT3 iteration-55 state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from configure_cmfgen_stageA import set_control
from ptf_gated_metrics import BatteryError, ScrtempReader, read_dimensions, validate_rvtj


SEED_INPUTS = (
    "VADAT", "IN_ITS", "MODEL_SPEC", "SN_HYDRO_DATA", "MODEL",
    "SPECIES_MASSES", "LEVEL_SL_STEQ_LINKS", "CONT_FREQ",
)
SEED_RESTART = (
    "SCRTEMP", "POINT1", "POINT2", "EDDFACTOR", "EDDFACTOR_INFO",
    "CUR_MODEL_DATA", "STEQ_VALS", "RVTJ", "BAMATPNT",
    "JH_AT_CURRENT_TIME", "JH_AT_CURRENT_TIME_INFO",
)
STATIC_MATCH = (
    "MODEL_SPEC", "SN_HYDRO_DATA", "SPECIES_MASSES",
    "LEVEL_SL_STEQ_LINKS",
)
MODEL_RUN_CONTROL_KEYS = frozenset({
    "NUM_ITS", "DO_LAM_IT", "DO_LAM_AUTO", "DO_T_AUTO", "MAX_LIN",
    "MAX_LAM", "COMP_BA", "N_FIX_BA", "DO_NG",
})
MODEL_SEED_CONTROLS = {
    "NUM_ITS": "5", "DO_LAM_IT": "T", "DO_LAM_AUTO": "F",
    "DO_T_AUTO": "F", "MAX_LIN": "1.05", "MAX_LAM": "1.10",
    "COMP_BA": "T", "N_FIX_BA": "0", "DO_NG": "F",
}
MODEL_RECORD = re.compile(r"^\s*\S+\s+\[([A-Z0-9_]+)\]")
MODEL_STAMP = re.compile(r"^\s*Model Started on:", re.IGNORECASE)
LOCAL_ATOMIC = (
    "atomic_local/SUL/III/19apr23/phot_data_A",
    "atomic_local/COB/II/19apr23/phot_data_A",
)
OPTIONAL_DANGLING = {"HI_IS_LINE_LIST", "H2_IS_LINE_LIST"}
MAX_SECONDS = 10_500
BAD_LOG = re.compile(
    r"Error in MOM_J_REL|MOM_JREL[^\n]*(?:error|fail)|"
    r"(?:error|fail)[^\n]*moment|grey[^\n]*(?:error|fail)|"
    r"(?:error|fail)[^\n]*grey|\b(?:NaN|[+-]?Infinity)\b", re.I,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def control(text: str, key: str) -> str:
    pattern = re.compile(
        rf"^(?![ \t]*[!#])[ \t]*(\S+)[ \t]+\[{re.escape(key)}\]",
        re.MULTILINE,
    )
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(f"제어 키 누락/중복: {key}")
    return matches[0]


def configure(vadat: str, in_its: str) -> str:
    required_vadat = {
        "FIX_T": "T", "SCALE_OPT": "MAJOR", "MAX_LIN": "1.05D0",
        "MAX_LAM": "1.10D0",
    }
    for key, expected in required_vadat.items():
        actual = control(vadat, key)
        if key.startswith("MAX_"):
            if float(actual.replace("D", "E")) != float(expected.replace("D", "E")):
                raise ValueError(f"봉인 CMFGEN 제어 불일치: {key}={actual}")
        elif actual != expected:
            raise ValueError(f"봉인 CMFGEN 제어 불일치: {key}={actual}")
    required_its = {
        "NUM_ITS": "5", "DO_LAM_IT": "T", "DO_LAM_AUTO": "F",
        "DO_T_AUTO": "F",
    }
    for key, expected in required_its.items():
        if control(in_its, key) != expected:
            raise ValueError(f"봉인 CMFGEN IN_ITS 불일치: {key}")
    return set_control(vadat, "MAX_LAM", "1.01D0")


def validate_seed_model_controls(text: str) -> dict[str, str]:
    """Require the exact iteration-55 model controls expected by this probe."""
    actual = {}
    for key, expected in MODEL_SEED_CONTROLS.items():
        value = control(text, key)
        if key in {"MAX_LIN", "MAX_LAM"}:
            matches = float(value.replace("D", "E")) == float(expected)
        elif key in {"NUM_ITS", "N_FIX_BA"}:
            matches = int(value) == int(expected)
        else:
            matches = value == expected
        if not matches:
            raise ValueError(f"checkpoint MODEL 실행 제어 불일치: {key}={value}")
        actual[key] = value
    return actual


def model_physics_signature(text: str) -> bytes:
    """Preserve byte identity of MODEL physics while excluding known run controls."""
    kept = []
    for line in text.splitlines(keepends=True):
        if MODEL_STAMP.match(line):
            continue
        match = MODEL_RECORD.match(line)
        if match and match.group(1) in MODEL_RUN_CONTROL_KEYS:
            continue
        kept.append(line)
    return "".join(kept).encode("utf-8")


def point_iteration(path: Path) -> int:
    lines = path.read_text(errors="strict").splitlines()
    if len(lines) < 2:
        raise ValueError(f"CMFGEN restart pointer truncated: {path.name}")
    fields = lines[1].split()
    if len(fields) < 2 or fields[0] != fields[1]:
        raise ValueError(f"CMFGEN restart pointer inconsistent: {path.name}")
    return int(fields[0])


def stage(seed: Path, setup_source: Path, run: Path, executable: Path) -> dict:
    seed = seed.resolve()
    setup_source = setup_source.resolve()
    run = run.resolve()
    executable = executable.resolve()
    if not seed.is_dir() or not setup_source.is_dir() or not executable.is_file():
        raise ValueError("CMFGEN 입력 루트 또는 실행 파일을 찾을 수 없음")
    if not os.access(executable, os.X_OK):
        raise ValueError("CMFGEN 실행 파일에 실행 권한이 없음")
    if run.exists() or not run.parent.is_dir():
        raise FileExistsError("새 실행 root만 허용; 기존 경로는 보존")
    if run == seed or run == setup_source or seed in run.parents or setup_source in run.parents:
        raise ValueError("실행 root가 원본 입력 내부임")

    for name in SEED_INPUTS + SEED_RESTART:
        source = seed / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"필수 원본 파일 누락 또는 외부 링크: {name}")
    for name in ("atomic_links.txt", "setup_links.sh", "model_spec_isf.txt"):
        source = setup_source / name
        if not source.is_file():
            raise ValueError(f"원자자료 연결 입력 누락: {name}")
    if not (setup_source / "MODEL").is_file() or (setup_source / "MODEL").is_symlink():
        raise ValueError("기준 물리 MODEL 누락 또는 외부 링크")
    for name in LOCAL_ATOMIC:
        source = setup_source / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"run-local 원자자료 누락 또는 외부 링크: {name}")

    for name in STATIC_MATCH:
        if sha(seed / name) != sha(setup_source / name):
            raise ValueError(f"CMFGEN 고정 입력 계보 불일치: {name}")
    seed_model = (seed / "MODEL").read_text(errors="strict")
    setup_model = (setup_source / "MODEL").read_text(errors="strict")
    seed_model_controls = validate_seed_model_controls(seed_model)
    model_signature = model_physics_signature(seed_model)
    if model_signature != model_physics_signature(setup_model):
        raise ValueError("CMFGEN 물리 MODEL 계보 불일치")
    source_commit = os.environ.get("LUMINA_CMFGEN_PROBE_SOURCE_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("고정 probe 소스 commit 누락 또는 형식 오류")
    for name in ("POINT1", "POINT2"):
        if point_iteration(seed / name) != 55:
            raise ValueError(f"예상한 iteration-55 checkpoint가 아님: {name}")
    nd, nt = read_dimensions(seed)
    reader = ScrtempReader(seed / "SCRTEMP", nd, nt)
    if reader.last_it != 55 or not validate_rvtj(reader, seed / "RVTJ")["complete"]:
        raise ValueError("iteration-55 SCRTEMP/RVTJ 연결 불일치")

    original_vadat = (seed / "VADAT").read_text()
    original_its = (seed / "IN_ITS").read_text()
    damped_vadat = configure(original_vadat, original_its)

    run.mkdir()
    (run / "seq_logs").mkdir()
    for name in SEED_INPUTS + SEED_RESTART:
        shutil.copy2(seed / name, run / name)
    for name in ("atomic_links.txt", "setup_links.sh", "model_spec_isf.txt"):
        shutil.copy2(setup_source / name, run / name)
    for name in LOCAL_ATOMIC:
        target = run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(setup_source / name, target)

    (run / "VADAT").write_text(damped_vadat)
    link_script = (run / "setup_links.sh").read_text()
    old_atomic = str(setup_source / "atomic_local")
    new_atomic = str(run / "atomic_local")
    old_cd = f"cd {setup_source}"
    new_cd = f"cd {run}"
    if old_atomic not in link_script or old_cd not in link_script:
        raise ValueError("setup_links.sh의 기준 경로가 예상과 다름")
    link_script = link_script.replace(old_atomic, new_atomic).replace(old_cd, new_cd)
    (run / "setup_links.sh").write_text(link_script)
    setup_result = subprocess.run(["bash", str(run / "setup_links.sh")], cwd=run,
                                  check=False, timeout=120, capture_output=True, text=True)
    (run / "seq_logs" / "setup_links.stdout").write_text(setup_result.stdout)
    (run / "seq_logs" / "setup_links.stderr").write_text(setup_result.stderr)
    if setup_result.returncode:
        raise RuntimeError(f"setup_links.sh 실패: exit={setup_result.returncode}")

    broken = [entry.name for entry in run.iterdir()
              if entry.is_symlink() and not entry.exists()
              and entry.name not in OPTIONAL_DANGLING]
    if broken:
        raise ValueError(f"필수 원자자료 링크 손상: {broken[:8]}")
    if (run / "XzV_IN").exists():
        raise ValueError("외부 population vector는 이 실험에서 허용하지 않음")
    if (run / "IN_ITS").read_bytes() != (seed / "IN_ITS").read_bytes():
        raise ValueError("IN_ITS 변경 감지")
    if set_control((run / "VADAT").read_text(), "MAX_LAM", "1.10D0") != original_vadat:
        raise ValueError("VADAT에서 MAX_LAM 이외 변경 감지")

    linked = {}
    for entry in run.iterdir():
        if entry.is_symlink() and entry.exists():
            linked[entry.name] = {"target": str(entry.resolve()), "sha256": sha(entry)}
    record = {
        "schema": "cmfgen-relt3-damped-lambda-v1",
        "scope": "fixed-T, five forced LAMBDA iterations from iteration 55; MAX_LAM 1.10 to 1.01 only",
        "source_commit": source_commit,
        "source_files_sha256": {
            name: sha(Path(__file__).resolve().parent / name)
            for name in ("run_cmfgen_relt3_damped_lambda.py",
                         "configure_cmfgen_stageA.py", "ptf_gated_metrics.py")
        },
        "seed_pointer": 55,
        "expected_last_iteration": 60,
        "seed_inputs_sha256": {name: sha(seed / name) for name in SEED_INPUTS + SEED_RESTART},
        "setup_inputs_sha256": {name: sha(setup_source / name)
                                for name in ("MODEL", "atomic_links.txt", "setup_links.sh",
                                             "model_spec_isf.txt")},
        "model_comparison": {
            "method": "byte-identical physics records after excluding only the CMFGEN run stamp and declared iteration controls",
            "physics_signature_sha256": hashlib.sha256(model_signature).hexdigest(),
            "excluded_run_control_keys": sorted(MODEL_RUN_CONTROL_KEYS),
            "seed_run_controls": seed_model_controls,
        },
        "binary": {"path": str(executable), "sha256": sha(executable)},
        "run": str(run),
        "controls": {"FIX_T": "T", "SCALE_OPT": "MAJOR", "MAX_LIN": "1.05D0",
                     "MAX_LAM": "1.01D0", "NUM_ITS": "5", "DO_LAM_IT": "T",
                     "DO_LAM_AUTO": "F", "DO_T_AUTO": "F"},
        "resources": {"CPUs": 16, "GiB": 256, "wall_hours": 3, "GPU": 0,
                      "automatic_retries": 0},
        "linked_inputs": linked,
        "final_goal": "NOT_MET",
    }
    (run / "seq_logs" / "input-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def scan_logs(run: Path, cursors: dict) -> str | None:
    reason = None
    for name in ("OUTGEN", "seq_logs/cmfgen.stdout", "seq_logs/cmfgen.stderr"):
        path = run / name
        if not path.is_file():
            continue
        offset, tail = cursors.get(name, (0, ""))
        with path.open("r", errors="replace") as stream:
            stream.seek(offset)
            new = stream.read()
            offset = stream.tell()
        text = tail + new
        match = BAD_LOG.search(text)
        if match:
            reason = f"solver_error:{name}:{match.group(0)}"
        cursors[name] = (offset, text[-512:])
    return reason


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run_probe(run: Path, executable: Path) -> dict:
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="16", OMP_DYNAMIC="FALSE", OMP_STACKSIZE="512M",
               OMP_PROC_BIND="close", OMP_PLACES="cores")
    stdout = run / "seq_logs" / "cmfgen.stdout"
    stderr = run / "seq_logs" / "cmfgen.stderr"
    cursors: dict = {}
    reason = None
    start = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(["srun", "--cpu-bind=cores", str(executable)],
                                   cwd=run, env=env, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=err, start_new_session=True)
        while process.poll() is None:
            reason = scan_logs(run, cursors)
            if reason or time.monotonic() - start >= MAX_SECONDS:
                reason = reason or f"runtime_cap_{MAX_SECONDS}s"
                stop_child(process)
                break
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        reason = reason or scan_logs(run, cursors)

    elapsed = time.monotonic() - start
    last_it = None
    pointer_valid = False
    try:
        p1, p2 = point_iteration(run / "POINT1"), point_iteration(run / "POINT2")
        last_it = p1
        pointer_valid = p1 == p2
    except (OSError, ValueError):
        p1 = p2 = None
    rvtj_validation = None
    if last_it is not None:
        try:
            nd, nt = read_dimensions(run)
            rvtj_validation = validate_rvtj(
                ScrtempReader(run / "SCRTEMP", nd, nt), run / "RVTJ")
        except (BatteryError, OSError, ValueError) as error:
            rvtj_validation = {"complete": False, "error": str(error)}
    complete = (process.returncode == 0 and reason is None and pointer_valid
                and last_it == 60 and bool(rvtj_validation and rvtj_validation.get("complete")))

    metrics_status = None
    metrics_path = run / "seq_logs" / "physical-metrics-55-60.json"
    if last_it is not None and last_it >= 55:
        metric_tool = Path(__file__).with_name("ptf_gated_metrics.py")
        completed = subprocess.run(
            [sys.executable, str(metric_tool), "--format", "json", "--from-it", "55",
             "--to-it", str(last_it), str(run)],
            capture_output=True, text=True, timeout=240,
        )
        metrics_status = {"returncode": completed.returncode,
                          "stderr": completed.stderr[-4000:]}
        if completed.returncode == 0:
            metrics_path.write_text(completed.stdout)
            metrics_status["path"] = str(metrics_path)

    result = {
        "schema": "cmfgen-relt3-damped-lambda-result-v1",
        "state": "COMPLETE_FIVE_FIXED_T_LAMBDA_STEPS" if complete else "INCOMPLETE_OR_FAILED",
        "returncode": process.returncode,
        "stop_reason": reason,
        "elapsed_seconds": elapsed,
        "POINT1_iteration": p1,
        "POINT2_iteration": p2,
        "expected_last_iteration": 60,
        "RVTJ_validation": rvtj_validation,
        "input_manifest_sha256": sha(run / "seq_logs" / "input-manifest.json"),
        "physical_metrics": metrics_status,
        "final_goal": "NOT_MET",
    }
    (run / "seq_logs" / "probe-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--setup-source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()
    record = stage(args.seed, args.setup_source, args.run, args.executable)
    print(json.dumps({"state": "STAGED", "run": record["run"],
                      "seed_pointer": record["seed_pointer"], "MAX_LAM": "1.01D0"}))
    result = run_probe(args.run.resolve(), args.executable.resolve())
    return 0 if result["state"] == "COMPLETE_FIVE_FIXED_T_LAMBDA_STEPS" else 30


if __name__ == "__main__":
    raise SystemExit(main())
