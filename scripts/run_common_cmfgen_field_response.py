#!/usr/bin/env python3
"""Run the fixed-field Si/S CMFGEN response diagnostic.

This entry point is intentionally separate from the v4 production runner.  It
does not call v4.actual(), does not run a formal sweep, does not solve Ca, and
does not update electron density.  It solves the same local Si/S SE problem
three times per selected shell:

  A   current Lumina J,
  B   CMFGEN EDDFACTOR J on the common frequency range plus A's common tail,
  A'  A down/up sampled through the CMFGEN frequency grid.

The script only becomes executable after the bundle design gate is closed.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import time
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np

_trapz = getattr(np, "trapezoid", np.trapz)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("/gpfs/kjhan/lumina/cmfgen_map/si_s_cmfgen_parity_20260915")
V4 = ROOT / "operator_si_s_full_coupling_v4"
SE_PATH = ROOT / "operator_si_s_element_se_v2/run.py"
ADAPTER_PATH = ROOT / "sol/ca_current_state_v1/run.py"
CURRENT_FIELD = ROOT / "operator_full_field_ca_001"
PARENT_STATE = ROOT / "operator_ca_material_v1/operator_actual_001/state"
REFERENCE = Path(
    "/gpfs/kjhan/lumina/cmfgen_map/new_radiation_ionization_20260914/"
    "physical_evidence/result/divergence_map.csv"
)
CMFGEN_EDD = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR")
CMFGEN_RVTJ = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ")
GEOMETRY = PROJECT_ROOT / "data/tardis_reference_toy06_19p48d/geometry.csv"
CMFGEN_READER = PROJECT_ROOT / "scripts/build_cmfgen_jtable.py"
OUTPUT_ROOT = ROOT / "operator_common_cmfgen_field_response_20260916"

V4_RUN = V4 / "run.py"
V4_RUN_SHA = "8e4cd4b21b6cea76af3737420f5e5c17600d85e14f1a3ee3d5f6700b360ce5cd"
ROW_VALIDATOR = ROOT / "sol/si_s_local_coupling_v2/run.py"
ROW_VALIDATOR_SHA = "f54838735e38b9a7767c8411ff4411f9023c3c41ad209cc3abda59556dee89e9"
SE_PATH_SHA = "20a6ea9c012b38e97214bb7e3f963d78810a5205991274e2f08ff9386c6bdded"

BANDS_A = ((228.0, 918.0), (918.0, 1290.0), (1290.0, 2000.0), (2000.0, 10000.0))
CLIGHT_A_HZ = 2.99792458e18
ELEMENTS = (14, 16)
STAGES = (1, 2, 3, 4)
DOMINANT_FRACTION = 0.05
FULL_MAE = 0.02
PART_IMPROVEMENT = 0.02
INTERP_REL_TOL = 1e-3
SHELL_LO = 4
SHELL_HI = 44


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ident(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_float(value):
    value = float(value)
    require(math.isfinite(value), "비유한 수치")
    return value


def cmfgen_data(reader):
    J, FL, nd, finish, nbad, nrecords = reader.read_eddfactor(str(CMFGEN_EDD))
    nu = np.ascontiguousarray(FL * 1e15, dtype=float)
    require(J.shape == (len(nu), nd) and nd == 90, "EDDFACTOR shape/ND")
    require(np.isfinite(J).all() and np.all(J > 0), "유효 CMFGEN J 양수/유한")
    order = np.argsort(nu)
    nu = nu[order]
    J = np.ascontiguousarray(J[order], dtype=float)
    V = reader.parse_rvtj_block(CMFGEN_RVTJ.read_text(), "Velocity (km/s)", nd)
    require(V.shape == (nd,) and np.isfinite(V).all() and np.all(V > 0), "RVTJ velocity")
    velocity_order = np.argsort(V)
    V = np.ascontiguousarray(V[velocity_order], dtype=float)
    J = np.ascontiguousarray(J[:, velocity_order], dtype=float)
    require(np.all(np.diff(nu) > 0) and np.all(np.diff(V) > 0), "CMFGEN axes ascending")
    logj = np.log(J)
    return dict(
        J=J,
        logJ=logj,
        nu=nu,
        V=V,
        ND=int(nd),
        finish=finite_float(finish),
        nonpositive_records=int(nbad),
        total_records=int(nrecords),
    )


def velocity_row(cmf, velocity):
    V = cmf["V"]
    logj = cmf["logJ"]
    if velocity <= V[0]:
        return logj[:, 0].copy()
    if velocity >= V[-1]:
        return logj[:, -1].copy()
    j = int(np.searchsorted(V, velocity) - 1)
    weight = (velocity - V[j]) / (V[j + 1] - V[j])
    return (1.0 - weight) * logj[:, j] + weight * logj[:, j + 1]


def cmfgen_row(cmf, velocity, fine_nu, current_row):
    result = np.array(current_row, dtype=float, copy=True)
    require(np.isfinite(result).all() and np.all(result >= 0), "현재 J row")
    lo = float(cmf["nu"][0])
    hi = float(cmf["nu"][-1])
    in_range = (fine_nu >= lo) & (fine_nu <= hi)
    require(np.any(in_range), "CMFGEN/Lumina 주파수 교집합 없음")
    logj = velocity_row(cmf, velocity)
    result[in_range] = np.exp(np.interp(
        np.log(fine_nu[in_range]), np.log(cmf["nu"]), logj
    ))
    require(np.isfinite(result).all() and np.all(result >= 0), "CMFGEN 변환 J")
    return result, in_range


def roundtrip_row(cmf, fine_nu, current_row):
    result = np.array(current_row, dtype=float, copy=True)
    positive = current_row > 0
    require(np.isfinite(current_row).all() and np.all(current_row >= 0), "A J row")
    native_mask = (cmf["nu"] >= fine_nu[0]) & (cmf["nu"] <= fine_nu[-1])
    fine_mask = (fine_nu >= cmf["nu"][0]) & (fine_nu <= cmf["nu"][-1])
    require(np.count_nonzero(native_mask) >= 2 and np.any(fine_mask), "A/A' 교집합")
    require(np.all(current_row[fine_mask] > 0), "A/A' 교집합의 0 bin 처리 미정")
    native_nu = cmf["nu"][native_mask]
    native_positive = np.interp(
        np.log(native_nu), np.log(fine_nu[positive]), np.log(current_row[positive])
    )
    result[fine_mask] = np.exp(np.interp(
        np.log(fine_nu[fine_mask]), np.log(native_nu), native_positive
    ))
    require(np.isfinite(result).all() and np.all(result >= 0), "A' J row")
    return result, fine_mask


def band_integral(x, y, wavelength_lo, wavelength_hi):
    frequency_lo = CLIGHT_A_HZ / wavelength_hi
    frequency_hi = CLIGHT_A_HZ / wavelength_lo
    require(x[0] <= frequency_lo <= x[-1] and x[0] <= frequency_hi <= x[-1], "적분 대역 범위")
    inside = (x > frequency_lo) & (x < frequency_hi)
    points = np.concatenate(([frequency_lo], x[inside], [frequency_hi]))
    values = np.interp(points, x, y)
    return float(_trapz(values, points))


def loglog_band_integral(x, y, wavelength_lo, wavelength_hi):
    """Integrate the log-log CMFGEN reconstruction on one wavelength band."""
    frequency_lo = CLIGHT_A_HZ / wavelength_hi
    frequency_hi = CLIGHT_A_HZ / wavelength_lo
    require(x[0] <= frequency_lo <= x[-1] and x[0] <= frequency_hi <= x[-1], "적분 대역 범위")
    require(np.isfinite(x).all() and np.all(x > 0) and np.all(np.diff(x) > 0), "적분 주파수 축")
    require(np.isfinite(y).all() and np.all(y > 0), "log-log 적분 J")
    segments = np.nonzero((x[:-1] < frequency_hi) & (x[1:] > frequency_lo))[0]
    require(len(segments) > 0, "적분 대역 세그먼트")
    log_x = np.log(x)
    log_y = np.log(y)
    left = np.maximum(x[segments], frequency_lo)
    right = np.minimum(x[segments + 1], frequency_hi)
    log_left = np.log(left)
    log_right = np.log(right)
    slope = (log_y[segments + 1] - log_y[segments]) / (
        log_x[segments + 1] - log_x[segments]
    )
    exponent = slope + 1.0
    delta = log_right - log_left
    prefactor = log_y[segments] - slope * log_x[segments]
    zero = np.abs(exponent) < 1e-12
    terms = np.empty_like(delta)
    terms[zero] = np.exp(prefactor[zero]) * delta[zero]
    terms[~zero] = (
        np.exp(prefactor[~zero] + exponent[~zero] * log_left[~zero])
        * np.expm1(exponent[~zero] * delta[~zero])
        / exponent[~zero]
    )
    return float(np.sum(terms, dtype=np.longdouble))


def reference_data():
    values = {}
    velocities = {}
    centers = {}
    with REFERENCE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["element"] not in ("si", "s"):
                continue
            shell = int(row["shell_id"])
            if not SHELL_LO <= shell < SHELL_HI:
                continue
            if row["authority"] != "SAHA_STAGE" or row["status"] != "VALID":
                continue
            stage = int(row["stage"])
            if stage not in STAGES:
                continue
            z = int(row["Z"])
            key = (shell, z, stage)
            require(key not in values, "SAHA_STAGE/VALID 중복 행")
            values[key] = finite_float(row["cmfgen_raw"])
            velocities[shell] = finite_float(row["vel_mid_km_s"])
            centers[shell] = finite_float(row["v_center_km_s"])
    for z in ELEMENTS:
        for shell in range(SHELL_LO, SHELL_HI):
            row = [values.get((shell, z, stage)) for stage in STAGES]
            require(all(v is not None for v in row), "SAHA_STAGE/VALID 기준 누락")
            require(abs(sum(row) - 1.0) <= 2e-5, "CMFGEN stage1-4 합")
    require(len(velocities) == SHELL_HI - SHELL_LO and len(centers) == len(velocities),
            "기준 속도 셸 수")
    return values, velocities, centers


def geometry_velocities(reader):
    vmids = reader.lumina_shell_velocities(str(GEOMETRY))
    require(vmids.shape == (50,) and np.isfinite(vmids).all(), "Lumina geometry")
    return np.ascontiguousarray(vmids, dtype=float)


def stage_fractions(se, validator, lib, p, grid, jrow, T, ne, extra, total, parent):
    before = np.array(jrow, dtype=float, copy=True)
    matrix, context = se.assemble(lib, p, grid, jrow, T, ne, extra)
    candidate, report = lib.bb.solve(matrix, total)
    require(candidate is not None, "GTH 해 실패")
    require(np.isfinite(candidate).all() and np.all(candidate > 0), "GTH 양수 해")
    require(abs(float(report["total_relative_error"])) < 1e-12, "GTH 총량")
    row_view = validator.row_flux(matrix, candidate)
    relative = row_view["relative"]
    require(np.isfinite(relative).all() and float(relative.max()) < 1e-12, "SE 상대 행 잔차")
    require(np.array_equal(jrow, before), "SE가 입력 J를 변경")
    U = context["U"]
    bf_rates, _, _ = lib.bf(p, context["J"], T, ne, U, candidate)
    require(np.array_equal(bf_rates, context["BF_rates"]), "최종 BF율/J 불일치")
    canonical_bf_sha256 = array_sha(context["J"])
    bb_jbar_sha256 = [array_sha(item["Jbar"]) for item in context["BB"]]
    stage = [
        float(np.sum(candidate[p["off"][q]:p["off"][q + 1]], dtype=np.longdouble))
        for q in range(4)
    ] + [float(candidate[-1])]
    fractions = np.asarray(stage, dtype=float) / total
    require(np.isfinite(fractions).all() and np.all(fractions > 0), "분율 양수")
    require(abs(float(fractions.sum()) - 1.0) < 2e-15, "분율 합")
    return dict(
        fractions=fractions.tolist(),
        total=finite_float(total),
        row_relative_max=finite_float(relative.max()),
        solver_total_relative_error=finite_float(report["total_relative_error"]),
        injected_fine_j_sha256=array_sha(before),
        consumed_fine_j_sha256=array_sha(jrow),
        consumed_canonical_bf_j_sha256=canonical_bf_sha256,
        consumed_bb_jbar_sha256=bb_jbar_sha256,
        Te=finite_float(T),
        ne=finite_float(ne),
    )


def ca_rates(p, arrays, keys, z, shell, levels):
    if z != 14:
        return [0.0, 0.0]
    result = []
    for q in (1, 2):
        lo, hi = p["levels"]["offsets"][(20, q)]
        tracked = float(np.sum(arrays["level.f64"][lo:hi, shell], dtype=np.longdouble))
        saved = float(arrays["ion.f64"][keys.index((20, q)), shell])
        require(abs(tracked - saved) <= 1e-12 * max(tracked, saved), "Ca tracked/ion 불일치")
        result.append(tracked)
    return result


def classify(reference, rows):
    summary = {}
    for z in ELEMENTS:
        for offset, stage in enumerate(STAGES):
            ref = np.asarray([reference[(s, z, stage)] for s in range(SHELL_LO, SHELL_HI)])
            ordered = sorted((r for r in rows if r["Z"] == z), key=lambda r: r["shell"])
            A = np.asarray([r["A"]["fractions"][offset] for r in ordered])
            B = np.asarray([r["B"]["fractions"][offset] for r in ordered])
            require(A.shape == ref.shape and B.shape == ref.shape, "분류 셸 정렬")
            base_mae = float(np.mean(np.abs(A - ref)))
            cmf_mae = float(np.mean(np.abs(B - ref)))
            improvement = base_mae - cmf_mae
            terminal_A = np.asarray([r["A"]["fractions"][4] for r in ordered])
            terminal_B = np.asarray([r["B"]["fractions"][4] for r in ordered])
            dominant = bool(np.any(np.maximum.reduce([ref, A, B]) >= DOMINANT_FRACTION))
            denominator_mismatch = bool(max(terminal_A.max(), terminal_B.max()) > FULL_MAE)
            if cmf_mae <= FULL_MAE:
                label = "F-FULL"
            elif improvement >= PART_IMPROVEMENT:
                label = "F-PART"
            else:
                label = "F-NULL"
            summary[f"Z{z}_stage{stage}"] = dict(
                element=z, stage=stage, baseline_A_mae=base_mae,
                cmfgen_B_mae=cmf_mae, improvement=improvement,
                label=label, dominant=dominant,
                decision_role="primary" if dominant else "trace_only",
                terminal_VI_A_max=float(terminal_A.max()),
                terminal_VI_B_max=float(terminal_B.max()),
                denominator_mismatch=denominator_mismatch,
                qualification="denominator_mismatch" if denominator_mismatch else "none",
                signed_shell_improvement=(np.abs(A - ref) - np.abs(B - ref)).tolist(),
            )
    return summary


def execute(args, out):
    require(sha(V4_RUN) == V4_RUN_SHA, "v4 source hash")
    require(sha(ROW_VALIDATOR) == ROW_VALIDATOR_SHA, "row validator hash")
    require(sha(SE_PATH) == SE_PATH_SHA, "Si/S SE source hash")
    require(args.parent.resolve() == PARENT_STATE.resolve(), "봉인 부모 경로")
    require(args.field.resolve() == CURRENT_FIELD.resolve(), "봉인 현재 장 경로")
    require(args.start == SHELL_LO and args.stop == SHELL_HI, "사전등록 셸 범위")

    reader = load_module("cmfgen_jtable_reader", CMFGEN_READER)
    cmf = cmfgen_data(reader)
    vmids = geometry_velocities(reader)
    reference, reference_velocity, reference_center = reference_data()
    for shell in range(SHELL_LO, SHELL_HI):
        require(abs(vmids[shell] / reference_center[shell] - 1.0) <= 1e-12,
                "CSV/geometry center velocity mismatch")

    se = load_module("fixed_field_element_se", SE_PATH)
    validator = load_module("fixed_field_row_validator", ROW_VALIDATOR)
    driver, all50, core, native_identity = se.libraries()
    context = se.adapter.context(args.parent.resolve(), args.field.resolve(), driver)
    grid, fine_nu = driver.grid()
    require(context["grid"] == grid, "현재 J/grid 계약")
    arrays = context["arrays"]
    keys = context["keys"]
    current_j = context["J"]
    require(current_j.shape == (50, grid["nf"]) and np.isfinite(current_j).all(), "현재 J array")
    disk_te = np.fromfile(args.parent / "te.f64", dtype="<f8")
    disk_ne = np.fromfile(args.parent / "ne.f64", dtype="<f8")
    require(np.array_equal(disk_te, arrays["te.f64"]) and np.array_equal(disk_ne, arrays["ne.f64"]),
            "부모 디스크 Te/ne와 소비 배열 불일치")
    fixed = json.loads((se.adapter.FIXED / "state.json").read_text())
    require(fixed["keys"] == context["sm"]["keys"], "고정 기하/현재 위상")
    natom, heat, _, _, _, fixed_ids = all50.read_extra_inputs(
        dict(sm=fixed, arrays=arrays), core
    )
    policy = all50.policy(core)
    atoms = {z: se.prepare(z, driver, core) for z in ELEMENTS}
    natives = {z: se.Native(z, core) for z in ELEMENTS}
    require(driver.BF_LO >= cmf["nu"][0] and driver.BF_HI <= cmf["nu"][-1],
            "CMFGEN 대역 밖 bound-free 문턱")
    low_frequency_lines = {
        z: int(sum(np.count_nonzero(stage["nu"] < cmf["nu"][0]) for stage in atoms[z]["stages"]))
        for z in ELEMENTS
    }
    require(all(low_frequency_lines[z] > 0 for z in ELEMENTS), "저주파 Si/S 선 목록")

    current_field_result = json.loads((args.field / "result.json").read_text())
    provenance = dict(
        runner=ident(Path(__file__)),
        v4_source=ident(V4_RUN),
        row_validator=ident(ROW_VALIDATOR),
        se_source=ident(SE_PATH),
        adapter_source=ident(ADAPTER_PATH),
        current_field={"root": str(args.field.resolve()), "J_nu": ident(args.field / "J_nu.npy"),
                       "result": ident(args.field / "result.json")},
        parent_state={"root": str(args.parent.resolve()),
                      "state": ident(args.parent / "state.json"),
                      "ne": ident(args.parent / "ne.f64"),
                      "te": ident(args.parent / "te.f64")},
        cmfgen_edd=ident(CMFGEN_EDD),
        cmfgen_info=ident(Path(str(CMFGEN_EDD) + "_INFO")),
        cmfgen_rvtj=ident(CMFGEN_RVTJ),
        reference=ident(REFERENCE),
        geometry=ident(GEOMETRY),
        fixed_inputs=fixed_ids,
        cmfgen=dict(ND=cmf["ND"], frequency_min_Hz=float(cmf["nu"][0]),
                    frequency_max_Hz=float(cmf["nu"][-1]), velocity_min_km_s=float(cmf["V"][0]),
                    velocity_max_km_s=float(cmf["V"][-1]), finish=cmf["finish"],
                    nonpositive_records=cmf["nonpositive_records"], total_records=cmf["total_records"]),
        grid=grid,
        policy=policy,
        native=native_identity,
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
        current_field_relative_change=finite_float(current_field_result["history"][0]["relative_change"]),
    )
    write_json(out / "provenance.json", provenance)

    rows = []
    failed_rows = []
    interpolation = []
    roundtrip_changed = False
    with (out / "rows.jsonl").open("x") as row_log:
        for shell in range(args.start, args.stop):
            geometry_velocity = float(vmids[shell])
            velocity = float(reference_velocity[shell])
            require(cmf["V"][0] <= velocity <= cmf["V"][-1], "CMFGEN 속도 외삽")
            current_row = np.array(current_j[shell], dtype=float, copy=True)
            b_row, overlap = cmfgen_row(cmf, velocity, fine_nu, current_row)
            a_prime, roundtrip_mask = roundtrip_row(cmf, fine_nu, current_row)
            require(np.array_equal(b_row[~overlap], current_row[~overlap]), "B 공통 꼬리 변경")
            if not np.array_equal(a_prime[roundtrip_mask], current_row[roundtrip_mask]):
                roundtrip_changed = True
            cmf_native_at_v = np.exp(velocity_row(cmf, velocity))
            for wavelength_lo, wavelength_hi in BANDS_A:
                native_value = loglog_band_integral(cmf["nu"], cmf_native_at_v, wavelength_lo, wavelength_hi)
                fine_value = band_integral(fine_nu[overlap], b_row[overlap], wavelength_lo, wavelength_hi)
                relative = abs(fine_value / native_value - 1.0) if native_value else 0.0
                require(relative <= INTERP_REL_TOL, "CMFGEN 보간 적분 보존")
                interpolation.append(dict(shell=shell, band_A=[wavelength_lo, wavelength_hi],
                                          native_integral=native_value, fine_integral=fine_value,
                                          relative_error=relative))
            for z in ELEMENTS:
                try:
                    T = float(arrays["te.f64"][shell])
                    ne = float(arrays["ne.f64"][shell])
                    p = atoms[z]
                    lib = natives[z]
                    total, parent = se.parent_population(
                        arrays["ion.f64"], arrays["level.f64"], keys, z, p["gl"], shell
                    )
                    require(np.isfinite(parent).all() and np.all(parent >= 0), "부모 population")
                    ca = ca_rates(p, arrays, keys, z, shell, p["levels"])
                    extra = se.extra_rates(z, policy, T, float(heat[shell]), float(natom[shell]), *ca)
                    a_result = stage_fractions(se, validator, lib, p, grid, current_row.copy(), T, ne, extra, total, parent)
                    b_result = stage_fractions(se, validator, lib, p, grid, b_row.copy(), T, ne, extra, total, parent)
                    ap_result = stage_fractions(se, validator, lib, p, grid, a_prime.copy(), T, ne, extra, total, parent)
                    row = dict(shell=shell, Z=z, velocity_km_s=velocity,
                               geometry_velocity_km_s=geometry_velocity,
                               sample_distance_km_s=abs(geometry_velocity - velocity),
                               A=a_result, B=b_result, Aprime=ap_result,
                               Te_parent=T, ne_parent=ne, cmfgen_overlap_bins=int(np.count_nonzero(overlap)),
                               roundtrip_bins=int(np.count_nonzero(roundtrip_mask)),
                               common_tail_bins=int(np.count_nonzero(fine_nu < cmf["nu"][0])),
                               ca_II=ca[0], ca_III=ca[1],
                               reference=[reference[(shell, z, stage)] for stage in STAGES])
                except Exception as exc:
                    row = dict(shell=shell, Z=z, status="FAILED", error=repr(exc))
                    failed_rows.append(row)
                rows.append(row)
                row_log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                row_log.flush()
    require(not failed_rows, "Si/S branch 실패: rows.jsonl 보존")

    for shell in range(args.start, args.stop):
        for z in ELEMENTS:
            branch = [r for r in rows if r["shell"] == shell and r["Z"] == z]
            require(len(branch) == 1, "결과 행 중복/누락")
            row = branch[0]
            for key in ("A", "B", "Aprime"):
                require(row[key]["injected_fine_j_sha256"] == row[key]["consumed_fine_j_sha256"],
                        "fine J hash 소비 불일치")
                require(row[key]["Te"] == row["Te_parent"] and row[key]["ne"] == row["ne_parent"],
                        "Te/ne 고정 위반")
    result = dict(
        complete=True,
        scope="fixed-field local Si/S response; not a converged Lumina-vs-CMFGEN comparison",
        physical_parity=False,
        formal_calls=0,
        ca_solves=0,
        ne_updates=0,
        shells=[args.start, args.stop],
        rows=rows,
        interpolation=interpolation,
        classification=classify(reference, rows),
        current_field_transport_converged=bool(current_field_result["transport_converged"]),
        current_field_status=int(current_field_result["status"]),
        current_field_relative_change=finite_float(current_field_result["history"][0]["relative_change"]),
        roundtrip_changed=roundtrip_changed,
        cmfgen_nonpositive_records=cmf["nonpositive_records"],
        reproducibility=dict(
            input_j_consumed_identity=True,
            Te_ne_bitwise_contract="values copied from material_root and not updated",
            formal_sweep_skipped=True,
        ),
    )
    require(roundtrip_changed, "A/A' 왕복이 완전히 동일하여 주입 경로 음성대조 무효")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=PARENT_STATE)
    parser.add_argument("--field", type=Path, default=CURRENT_FIELD)
    parser.add_argument("--start", type=int, default=SHELL_LO)
    parser.add_argument("--stop", type=int, default=SHELL_HI)
    parser.add_argument("--timeout", type=int, default=3300)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(OUTPUT_ROOT.resolve()) and out != OUTPUT_ROOT.resolve(), "격리 출력 경로")
    require(not out.exists(), "기존 출력 덮어쓰기 금지")
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
            runner_sha256=sha(Path(__file__)),
            wall_seconds=time.monotonic() - start,
            cpu_seconds=time.process_time() - cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        write_json(out / "result.json", result)
    except BaseException as exc:
        if not (out / "failure.json").exists():
            write_json(out / "failure.json", dict(error=repr(exc), complete=False,
                                                   wall_seconds=time.monotonic() - start))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
