#!/usr/bin/env python3
"""Measure the fixed-field Si/S response to CMFGEN's public electron density.

This is a companion diagnostic to run_common_cmfgen_field_response.py.  It
keeps the original runner immutable and changes only ne from the validated
CMFGEN PUBLIC_NE rows; Te, parent element totals, atomic data, and the A/B/A'
field construction remain unchanged.
"""
import argparse
import csv
import importlib.util
import json
import resource
import signal
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_ROOT / "scripts/run_common_cmfgen_field_response.py"
ROOT = Path("/gpfs/kjhan/lumina/cmfgen_map/si_s_cmfgen_parity_20260915")
OUTPUT_ROOT = ROOT / "operator_common_cmfgen_ne_response_20260916"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("common_field_runner_base", BASE_PATH)


def cmfgen_ne_data():
    values = {}
    temperatures = {}
    with base.REFERENCE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["element"] != "electron":
                continue
            shell = int(row["shell_id"])
            if not base.SHELL_LO <= shell < base.SHELL_HI:
                continue
            if row["authority"] != "PUBLIC_NE" or row["status"] != "VALID":
                continue
            base.require(shell not in values, "PUBLIC_NE 중복 행")
            values[shell] = base.finite_float(row["reference_ne_raw"])
            temperatures[shell] = base.finite_float(row["temperature_public_K"])
    base.require(len(values) == base.SHELL_HI - base.SHELL_LO, "PUBLIC_NE 셸 수")
    base.require(all(values[s] > 0 for s in values), "PUBLIC_NE 양수")
    return values, temperatures


def execute(args, out):
    base.require(base.sha(base.V4_RUN) == base.V4_RUN_SHA, "v4 source hash")
    base.require(base.sha(base.ROW_VALIDATOR) == base.ROW_VALIDATOR_SHA, "row validator hash")
    base.require(base.sha(base.SE_PATH) == base.SE_PATH_SHA, "Si/S SE source hash")
    base.require(args.parent.resolve() == base.PARENT_STATE.resolve(), "봉인 부모 경로")
    base.require(args.field.resolve() == base.CURRENT_FIELD.resolve(), "봉인 현재 장 경로")
    base.require(args.start == base.SHELL_LO and args.stop == base.SHELL_HI, "사전등록 셸 범위")

    reader = base.load_module("cmfgen_jtable_reader_ne", base.CMFGEN_READER)
    cmf = base.cmfgen_data(reader)
    vmids = base.geometry_velocities(reader)
    reference, reference_velocity, reference_center = base.reference_data()
    cmfgen_ne, cmfgen_temperature = cmfgen_ne_data()
    for shell in range(base.SHELL_LO, base.SHELL_HI):
        base.require(abs(vmids[shell] / reference_center[shell] - 1.0) <= 1e-12,
                     "CSV/geometry center velocity mismatch")

    se = base.load_module("fixed_field_element_se_ne", base.SE_PATH)
    validator = base.load_module("fixed_field_row_validator_ne", base.ROW_VALIDATOR)
    driver, all50, core, native_identity = se.libraries()
    context = se.adapter.context(args.parent.resolve(), args.field.resolve(), driver)
    grid, fine_nu = driver.grid()
    base.require(context["grid"] == grid, "현재 J/grid 계약")
    arrays = context["arrays"]
    keys = context["keys"]
    current_j = context["J"]
    base.require(current_j.shape == (50, grid["nf"]) and np.isfinite(current_j).all(), "현재 J array")
    disk_te = np.fromfile(args.parent / "te.f64", dtype="<f8")
    disk_ne = np.fromfile(args.parent / "ne.f64", dtype="<f8")
    base.require(np.array_equal(disk_te, arrays["te.f64"]) and np.array_equal(disk_ne, arrays["ne.f64"]),
                 "부모 디스크 Te/ne와 소비 배열 불일치")
    for shell in range(base.SHELL_LO, base.SHELL_HI):
        base.require(abs(cmfgen_temperature[shell] - arrays["te.f64"][shell]) <=
                     1e-12 * max(cmfgen_temperature[shell], arrays["te.f64"][shell]),
                     "CMFGEN/부모 Te 불일치")
    fixed = json.loads((se.adapter.FIXED / "state.json").read_text())
    base.require(fixed["keys"] == context["sm"]["keys"], "고정 기하/현재 위상")
    natom, heat, _, _, _, fixed_ids = all50.read_extra_inputs(
        dict(sm=fixed, arrays=arrays), core
    )
    policy = all50.policy(core)
    atoms = {z: se.prepare(z, driver, core) for z in base.ELEMENTS}
    natives = {z: se.Native(z, core) for z in base.ELEMENTS}
    base.require(driver.BF_LO >= cmf["nu"][0] and driver.BF_HI <= cmf["nu"][-1],
                 "CMFGEN 대역 밖 bound-free 문턱")
    low_frequency_lines = {
        z: int(sum(np.count_nonzero(stage["nu"] < cmf["nu"][0]) for stage in atoms[z]["stages"]))
        for z in base.ELEMENTS
    }
    base.require(all(low_frequency_lines[z] > 0 for z in base.ELEMENTS), "저주파 Si/S 선 목록")

    current_field_result = json.loads((args.field / "result.json").read_text())
    provenance = dict(
        runner=base.ident(Path(__file__)),
        base_runner=base.ident(BASE_PATH),
        v4_source=base.ident(base.V4_RUN),
        row_validator=base.ident(base.ROW_VALIDATOR),
        se_source=base.ident(base.SE_PATH),
        adapter_source=base.ident(base.ADAPTER_PATH),
        current_field={"root": str(args.field.resolve()), "J_nu": base.ident(args.field / "J_nu.npy"),
                       "result": base.ident(args.field / "result.json")},
        parent_state={"root": str(args.parent.resolve()), "state": base.ident(args.parent / "state.json"),
                      "ne": base.ident(args.parent / "ne.f64"), "te": base.ident(args.parent / "te.f64")},
        cmfgen_edd=base.ident(base.CMFGEN_EDD),
        cmfgen_info=base.ident(Path(str(base.CMFGEN_EDD) + "_INFO")),
        cmfgen_rvtj=base.ident(base.CMFGEN_RVTJ),
        reference=base.ident(base.REFERENCE),
        geometry=base.ident(base.GEOMETRY),
        fixed_inputs=fixed_ids,
        cmfgen=dict(ND=cmf["ND"], frequency_min_Hz=float(cmf["nu"][0]),
                    frequency_max_Hz=float(cmf["nu"][-1]), velocity_min_km_s=float(cmf["V"][0]),
                    velocity_max_km_s=float(cmf["V"][-1]), finish=cmf["finish"],
                    nonpositive_records=cmf["nonpositive_records"], total_records=cmf["total_records"]),
        grid=grid,
        policy=policy,
        native=native_identity,
        ne_source="CMFGEN_PUBLIC_NE.reference_ne_raw",
        cmfgen_ne_by_shell={str(s): cmfgen_ne[s] for s in sorted(cmfgen_ne)},
        formal_calls=0,
        ca_solves=0,
        ne_updates=0,
        Te_fixed=True,
        ne_fixed=True,
        branch_contract="A=current; B=CMFGEN overlap+common Lumina tail; A'=A roundtrip",
        bound_free_outside_cmfgen_zero=True,
        low_frequency_bound_bound_lines=low_frequency_lines,
        current_field_transport_converged=bool(current_field_result["transport_converged"]),
        current_field_status=int(current_field_result["status"]),
        current_field_relative_change=base.finite_float(current_field_result["history"][0]["relative_change"]),
    )
    base.write_json(out / "provenance.json", provenance)

    rows = []
    failed_rows = []
    interpolation = []
    roundtrip_changed = False
    with (out / "rows.jsonl").open("x") as row_log:
        for shell in range(args.start, args.stop):
            geometry_velocity = float(vmids[shell])
            velocity = float(reference_velocity[shell])
            base.require(cmf["V"][0] <= velocity <= cmf["V"][-1], "CMFGEN 속도 외삽")
            current_row = np.array(current_j[shell], dtype=float, copy=True)
            b_row, overlap = base.cmfgen_row(cmf, velocity, fine_nu, current_row)
            a_prime, roundtrip_mask = base.roundtrip_row(cmf, fine_nu, current_row)
            base.require(np.array_equal(b_row[~overlap], current_row[~overlap]), "B 공통 꼬리 변경")
            if not np.array_equal(a_prime[roundtrip_mask], current_row[roundtrip_mask]):
                roundtrip_changed = True
            cmf_native_at_v = np.exp(base.velocity_row(cmf, velocity))
            for wavelength_lo, wavelength_hi in base.BANDS_A:
                native_value = base.loglog_band_integral(cmf["nu"], cmf_native_at_v,
                                                         wavelength_lo, wavelength_hi)
                fine_value = base.band_integral(fine_nu[overlap], b_row[overlap],
                                               wavelength_lo, wavelength_hi)
                relative = abs(fine_value / native_value - 1.0) if native_value else 0.0
                base.require(relative <= base.INTERP_REL_TOL, "CMFGEN 보간 적분 보존")
                interpolation.append(dict(shell=shell, band_A=[wavelength_lo, wavelength_hi],
                                          native_integral=native_value, fine_integral=fine_value,
                                          relative_error=relative))
            for z in base.ELEMENTS:
                try:
                    T = float(arrays["te.f64"][shell])
                    ne = float(cmfgen_ne[shell])
                    p = atoms[z]
                    lib = natives[z]
                    total, parent = se.parent_population(
                        arrays["ion.f64"], arrays["level.f64"], keys, z, p["gl"], shell
                    )
                    base.require(np.isfinite(parent).all() and np.all(parent >= 0), "부모 population")
                    ca = base.ca_rates(p, arrays, keys, z, shell, p["levels"])
                    extra = se.extra_rates(z, policy, T, float(heat[shell]), float(natom[shell]), *ca)
                    a_result = base.stage_fractions(se, validator, lib, p, grid, current_row.copy(),
                                                    T, ne, extra, total, parent)
                    b_result = base.stage_fractions(se, validator, lib, p, grid, b_row.copy(),
                                                    T, ne, extra, total, parent)
                    ap_result = base.stage_fractions(se, validator, lib, p, grid, a_prime.copy(),
                                                     T, ne, extra, total, parent)
                    row = dict(shell=shell, Z=z, velocity_km_s=velocity,
                               geometry_velocity_km_s=geometry_velocity,
                               sample_distance_km_s=abs(geometry_velocity - velocity),
                               A=a_result, B=b_result, Aprime=ap_result,
                               Te_parent=T, ne_parent=ne, ne_source="CMFGEN_PUBLIC_NE",
                               cmfgen_overlap_bins=int(np.count_nonzero(overlap)),
                               roundtrip_bins=int(np.count_nonzero(roundtrip_mask)),
                               common_tail_bins=int(np.count_nonzero(fine_nu < cmf["nu"][0])),
                               ca_II=ca[0], ca_III=ca[1],
                               reference=[reference[(shell, z, stage)] for stage in base.STAGES])
                except Exception as exc:
                    row = dict(shell=shell, Z=z, status="FAILED", error=repr(exc))
                    failed_rows.append(row)
                rows.append(row)
                row_log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                row_log.flush()
    base.require(not failed_rows, "Si/S branch 실패: rows.jsonl 보존")
    for shell in range(args.start, args.stop):
        for z in base.ELEMENTS:
            branch = [r for r in rows if r["shell"] == shell and r["Z"] == z]
            base.require(len(branch) == 1, "결과 행 중복/누락")
            row = branch[0]
            for key in ("A", "B", "Aprime"):
                base.require(row[key]["injected_fine_j_sha256"] == row[key]["consumed_fine_j_sha256"],
                             "fine J hash 소비 불일치")
                base.require(row[key]["Te"] == row["Te_parent"] and row[key]["ne"] == row["ne_parent"],
                             "Te/ne 고정 위반")
    result = dict(
        complete=True,
        scope="fixed-field local Si/S response with CMFGEN PUBLIC_NE; not a converged Lumina-vs-CMFGEN comparison",
        physical_parity=False,
        ne_source="CMFGEN_PUBLIC_NE.reference_ne_raw",
        formal_calls=0,
        ca_solves=0,
        ne_updates=0,
        shells=[args.start, args.stop],
        rows=rows,
        interpolation=interpolation,
        classification=base.classify(reference, rows),
        current_field_transport_converged=bool(current_field_result["transport_converged"]),
        current_field_status=int(current_field_result["status"]),
        current_field_relative_change=base.finite_float(current_field_result["history"][0]["relative_change"]),
        roundtrip_changed=roundtrip_changed,
        cmfgen_nonpositive_records=cmf["nonpositive_records"],
        reproducibility=dict(
            input_j_consumed_identity=True,
            Te_bitwise_contract="values copied from material_root and not updated",
            ne_source_contract="CMFGEN PUBLIC_NE reference_ne_raw; no electron-density update",
            formal_sweep_skipped=True,
        ),
    )
    base.require(roundtrip_changed, "A/A' 왕복이 완전히 동일하여 주입 경로 음성대조 무효")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=base.PARENT_STATE)
    parser.add_argument("--field", type=Path, default=base.CURRENT_FIELD)
    parser.add_argument("--start", type=int, default=base.SHELL_LO)
    parser.add_argument("--stop", type=int, default=base.SHELL_HI)
    parser.add_argument("--timeout", type=int, default=3300)
    args = parser.parse_args()
    out = args.output.resolve()
    base.require(out.is_relative_to(OUTPUT_ROOT.resolve()) and out != OUTPUT_ROOT.resolve(), "격리 출력 경로")
    base.require(not out.exists(), "기존 출력 덮어쓰기 금지")
    start = time.monotonic()
    cpu = time.process_time()
    out.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS, (16 * 2**30, 16 * 2**30))

    def stop(signum, _frame):
        raise RuntimeError("시간/중단 신호: " + str(signum))

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, stop)
    signal.setitimer(signal.ITIMER_REAL, args.timeout)
    try:
        result = execute(args, out)
        result.update(
            runner_sha256=base.sha(Path(__file__)),
            wall_seconds=time.monotonic() - start,
            cpu_seconds=time.process_time() - cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        base.write_json(out / "result.json", result)
    except BaseException as exc:
        if not (out / "failure.json").exists():
            base.write_json(out / "failure.json", dict(error=repr(exc), complete=False,
                                                        wall_seconds=time.monotonic() - start))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
