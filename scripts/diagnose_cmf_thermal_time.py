#!/usr/bin/env python3
"""공개 앞뒤 시점으로 기체 열운동 에너지의 시간항만 비교한다. 생산 입력은 바꾸지 않는다."""
import csv
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from diagnose_cmf_reference_inputs import DAY, MODEL, PUBLIC, physical_profile_at_epoch

KB = 1.380649e-16


class AdInput(C.Structure):
    _fields_ = [("n", C.c_size_t), ("epoch", C.c_double)] + [
        (name, C.POINTER(C.c_double)) for name in ("r", "v", "T", "natom", "ne", "u")]


class AdCell(C.Structure):
    _fields_ = [(name, C.c_double) for name in (
        "temperature_gradient", "velocity_divergence", "electron_fraction_gradient",
        "internal_energy_gradient", "signed_total", "cooling", "heating")]


def derivative_at_middle(values, seconds):
    """불등간격 세 시점의 중심 미분."""
    # 전진/후진 값도 남기되, 성긴 시간 표본의 차이를 오차 증명으로 삼지 않는다.
    left, right = np.diff(seconds)
    backward = (values[1] - values[0]) / left
    forward = (values[2] - values[1]) / right
    return (right * backward + left * forward) / (left + right), backward, forward


def main():
    assert os.environ.get("SLURM_JOB_ID"), "Slurm 안에서 계산한다"
    library, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    assert outdir.is_dir()
    profile_path = PUBLIC / "phys_toy06_cmfgen.txt"
    with profile_path.open() as handle:
        times = next(line for line in handle if line.startswith("#TIMES[d]:"))
    times = np.array([float(t) for t in times.split()[1:]])
    middle = np.flatnonzero(np.abs(times - DAY) < 1e-8)
    assert len(middle) == 1 and 0 < middle[0] < len(times) - 1
    epochs = times[middle[0]-1:middle[0]+2]
    with (MODEL / "geometry.csv").open() as handle:
        geometry = list(csv.DictReader(handle))
    with (MODEL / "deposition_cmfgen.csv").open() as handle:
        heat_by_id = {int(row["shell_id"]): float(row["heating_rate"])
                      for row in csv.DictReader(handle)}
    config = json.loads((MODEL / "config.json").read_text())
    assert abs(config["time_explosion_s"] - DAY * 86400) < 1e-6
    v = np.array([(float(row["v_inner"]) + float(row["v_outer"])) / 2
                  for row in geometry])
    radius = v * DAY * 86400
    volume = np.array([4*np.pi/3 * (float(row["r_outer"])**3 - float(row["r_inner"])**3)
                       for row in geometry])
    heat = np.array([heat_by_id[int(row["shell_id"])] for row in geometry])
    samples = []
    for epoch in epochs:
        profile = physical_profile_at_epoch(profile_path, float(epoch))
        assert profile[0, 0] <= v.min()/1e5 and v.max()/1e5 <= profile[-1, 0]
        samples.append(np.array([np.interp(v/1e5, profile[:, 0], profile[:, j])
                                 for j in range(1, 5)]))
    samples = np.array(samples)  # 시점 × (T,rho,ne,natom) × 셸
    te, rho, ne, natom = samples[1]
    specific = 1.5 * KB * samples[:, 0] * (samples[:, 2] + samples[:, 3]) / samples[:, 1]
    derivative, backward, forward = derivative_at_middle(specific, epochs * 86400)
    pressure = KB * te * (natom + ne)
    expansion = 3 * pressure / (DAY * 86400)
    time_terms = [rho * d + expansion for d in (derivative, backward, forward)]

    # 기존 C 생산 함수에 동일한 공개 열운동 상태를 넣는다. 원자 내부에너지는
    # 격리 시험에서만 0으로 하며, 실제 전체 에너지 방정식을 평가했다고 하지 않는다.
    buffers = [np.ascontiguousarray(x, dtype=np.float64)
               for x in (radius, v, te, natom, ne, np.zeros_like(te))]
    args = AdInput(len(v), DAY*86400, *[x.ctypes.data_as(C.POINTER(C.c_double)) for x in buffers])
    cells = (AdCell * len(v))()
    lib = C.CDLL(str(library.resolve()))
    evaluate = lib.cmfgen_adiabatic_v3_homologous_evaluate
    evaluate.argtypes = [C.POINTER(AdInput), C.POINTER(AdCell)]
    evaluate.restype = C.c_int
    assert evaluate(C.byref(args), cells) == 0
    stationary = np.array([cell.signed_total for cell in cells])
    assert np.allclose([cell.velocity_divergence for cell in cells], expansion, rtol=1e-14, atol=0)
    assert all(cell.internal_energy_gradient == 0 for cell in cells)
    # 미분의 차원·불등간격 공식을 독립 해석 함수로 확인한다.
    t = epochs * 86400
    for power in (0, 1, 2):
        got = derivative_at_middle((t/t[1])[:, None]**power, t)[0][0]
        expected = 0 if power == 0 else power/t[1]
        assert abs(got - expected) < 1e-12/max(t)
    assert np.isfinite(stationary).all() and all(np.isfinite(x).all() for x in time_terms)

    names = ("central", "backward", "forward")
    total_heat = float(heat @ volume)
    comparisons = {}
    for name, thermal in zip(names, time_terms):
        delta = stationary - thermal
        comparisons[name] = dict(
            time_thermal_signed_integral_erg_s=float(thermal @ volume),
            stationary_minus_time_integral_erg_s=float(delta @ volume),
            absolute_difference_integral_erg_s=float(np.abs(delta) @ volume),
            absolute_difference_over_deposition=float(np.abs(delta) @ volume / total_heat),
            shells_difference_over_1pct_deposition=int(np.count_nonzero(np.abs(delta) > .01*heat)),
            shells_difference_over_10pct_deposition=int(np.count_nonzero(np.abs(delta) > .1*heat)))
    assert np.all(heat > 0)
    rows = zip(range(len(v)), v/1e5, heat, stationary, *time_terms,
               stationary - time_terms[0], (stationary-time_terms[0])/heat)
    with (outdir / "thermal_time.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("shell", "velocity_km_s", "deposition_erg_s_cm3", "stationary_thermal",
                         "time_thermal_central", "time_thermal_backward", "time_thermal_forward",
                         "stationary_minus_time", "difference_over_deposition"))
        writer.writerows(rows)
    identity = [Path(__file__), Path(__file__).with_name("diagnose_cmf_reference_inputs.py"),
                profile_path, library, MODEL/"geometry.csv", MODEL/"deposition_cmfgen.csv", MODEL/"config.json"]
    summary = dict(
        scope="public-reference thermal-motion term only; not new radiation/populations or full energy",
        epochs_days=epochs.tolist(), velocity_remap="linear at fixed homologous v; no extrapolation",
        stationary_C_return=0, input_atomic_internal_energy="zero only to isolate thermal motion",
        stationary_thermal_signed_integral_erg_s=float(stationary @ volume),
        volume_integrated_deposition_erg_s=total_heat, comparison=comparisons,
        warning="Three sparse epochs do not reproduce CMFGEN's internal time discretization. "
                "Radiation storage, atomic excitation/ionization storage, core flux and spectrum response are not measured. "
                "No production input is modified; no end-to-end physical agreement verdict.",
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in identity})
    with (outdir / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
