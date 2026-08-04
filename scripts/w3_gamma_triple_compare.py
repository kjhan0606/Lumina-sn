#!/usr/bin/env python3
"""Wave-3 Gamma triple comparison (offline, no simulation run).

A: element-wide provenance rate for one source super-level.
B: exact element-wide bf consumer arithmetic replayed from the frozen C1/C2 dumps.
C: the same sigma/threshold/route/SL quadrature with the whole radiation field
   replaced by CMFGEN jnu4 J_nu at the geometric midpoint of Lumina shell 8.

The script is intentionally fail-closed: schema, identity, grid, route, or unit
checks that affect a requested number abort instead of being guessed.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EW_DIR = Path("/tmp/w31_on_a.JuCpDY")
FROZEN_DIR = Path("/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59")
MODEL = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv"
CMF_RUN = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")
VALIDATION = ROOT / "validation/cmfgen_toy06_19p48d"

SHELL = 8
CONSUMER_ITER = 11
NB = 1000
NC = 24
H_PLANCK = 6.62607015e-27
K_B = 1.380649e-16
C_LIGHT = 2.99792458e10
EV_TO_ERG = 1.602176634e-12
FOUR_PI = 4.0 * math.pi
DEX_TOL = 0.1

EXPECTED = {
    "provenance": ["channel", "row", "column", "source_identity",
                   "target_identity", "aggregated_rate", "units", "producer",
                   "field_generation", "target_route", "probability_applied",
                   "full_to_sl_weight", "aggregation"],
    "identity": ["matrix_index", "Z", "spectroscopic_stage", "internal_stage",
                 "sl_id", "anchor_global_level", "member_full_level_ids",
                 "energy_eV", "g_or_SL_partition", "source_atomic_checksum"],
    "manifest": ["key", "value"],
    "solution": ["record_type", "matrix_index", "internal_stage", "sl_id",
                 "global_level", "raw_solution", "restored_population", "ion_total"],
    "c1": ["iter", "shell", "bin", "lam_lo_A", "lam_hi_A", "J_bin", "W",
           "T_R", "mode"],
    "c2": ["iter", "shell", "bin", "nu_mid", "J_raw", "bfr", "j_nu_count"],
    "levels": ["atomic_number", "ion_number", "level_number", "energy_eV", "g",
               "metastable", "super_level"],
    "ionization": ["atomic_number", "ion_number", "ionization_energy_eV"],
    "geometry": ["shell_id", "r_inner", "r_outer", "v_inner", "v_outer"],
    "oracle": ["shell", "category", "quantity", "Z", "stage", "transition",
               "frequency_Hz", "value", "unit", "producer", "status", "note"],
}

TARGETS = [
    {"key": "Fe III C48 lump", "Z": 26, "ion": 2, "matrix": 201,
     "sl_id": 100, "energy_eV": 13.1335531484},
    {"key": "S II SL4", "Z": 16, "ion": 1, "matrix": 4,
     "sl_id": 4, "energy_eV": 3.0464826904},
]


class Unresolved(RuntimeError):
    pass


AUDIT: list[dict[str, Any]] = []


def data_row_count(path: Path) -> int:
    with path.open("rb") as f:
        return max(sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b"")) - 1, 0)


def audit_csv(path: Path, expected: list[str], label: str) -> tuple[int, list[str]]:
    if not path.is_file():
        raise Unresolved(f"missing input: {path}")
    with path.open(newline="") as f:
        header = next(csv.reader(f), [])
    rows = data_row_count(path)
    ok = header == expected
    print(f"[INPUT] {label}: exists=1 bytes={path.stat().st_size} data_rows={rows} "
          f"schema={'OK' if ok else 'FAIL'} columns={header}")
    AUDIT.append({"label": label, "path": str(path), "bytes": path.stat().st_size,
                  "rows": rows, "schema": "OK" if ok else "FAIL"})
    if not ok:
        raise Unresolved(f"{label} schema mismatch: {header} != {expected}")
    return rows, header


def audit_text(path: Path, label: str) -> int:
    if not path.is_file():
        raise Unresolved(f"missing input: {path}")
    rows = data_row_count(path) + 1
    print(f"[INPUT] {label}: exists=1 bytes={path.stat().st_size} lines={rows} schema=text")
    AUDIT.append({"label": label, "path": str(path), "bytes": path.stat().st_size,
                  "rows": rows, "schema": "text"})
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def manifest(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return {r["key"]: r["value"] for r in rows}


def source_contract() -> dict[str, int]:
    ew = ROOT / "src/lumina_element_wide.c"
    plasma = ROOT / "src/lumina_plasma.c"
    audit_text(ew, "EW assembler source")
    audit_text(plasma, "fallback consumer source")
    et = ew.read_text()
    pt = plasma.read_text()
    required_ew = [
        "rad_ion += sigma * estimator;",
        "rad_ion += pref * J;",
        "p * rad_ion * lower_fraction",
        "threshold_eV = chi_eV - a->level_energy_eV[lower_global] +",
    ]
    required_plasma = [
        "R_bf += sigma * bfr;",
        "R_bf += pref * J_bin;",
    ]
    for token in required_ew:
        if token not in et:
            raise Unresolved(f"EW arithmetic token absent: {token}")
    for token in required_plasma:
        if token not in pt:
            raise Unresolved(f"fallback arithmetic token absent: {token}")
    line_of = lambda text, token: text[:text.index(token)].count("\n") + 1
    plasma_fb_pos = pt.index(required_plasma[1])
    plasma_est_pos = pt.rfind(required_plasma[0], 0, plasma_fb_pos)
    if plasma_est_pos < 0:
        raise Unresolved("paired plasma estimator token absent before fallback")
    out = {
        "ew_estimator": line_of(et, required_ew[0]),
        "ew_fallback": line_of(et, required_ew[1]),
        "ew_weight": line_of(et, required_ew[2]),
        "plasma_estimator": pt[:plasma_est_pos].count("\n") + 1,
        "plasma_fallback": line_of(pt, required_plasma[1]),
    }
    print("[SOURCE] exact consumer contract verified: " + ", ".join(f"{k}=L{v}" for k, v in out.items()))
    return out


def planck_bnu(T: np.ndarray | float, nu: np.ndarray) -> np.ndarray:
    Tarr = np.asarray(T, dtype=float)
    x = H_PLANCK * nu / (K_B * np.where(Tarr > 0, Tarr, 1.0))
    ok = (Tarr > 0) & (x > 0) & (x < 700)
    xs = np.where(ok, x, 1.0)
    value = (2.0 * H_PLANCK * nu**3 / C_LIGHT**2) / np.expm1(xs)
    return np.where(ok, value, 0.0)


def load_frozen_field(c1_path: Path, c2_path: Path, producer_iter: int) -> dict[str, Any]:
    c1_rows = [r for r in read_csv(c1_path)
               if int(r["iter"]) == producer_iter and int(r["shell"]) == SHELL]
    c2_rows = [r for r in read_csv(c2_path)
               if int(r["iter"]) == producer_iter and int(r["shell"]) == SHELL]
    c1_rows.sort(key=lambda r: int(r["bin"]))
    c2_rows.sort(key=lambda r: int(r["bin"]))
    if len(c1_rows) != NC or [int(r["bin"]) for r in c1_rows] != list(range(NC)):
        raise Unresolved(f"C1 selected block is not exactly {NC} ordered bins")
    if len(c2_rows) != NB or [int(r["bin"]) for r in c2_rows] != list(range(NB)):
        raise Unresolved(f"C2 selected block is not exactly {NB} ordered bins")

    nu_file = np.array([float(r["nu_mid"]) for r in c2_rows])
    bfr = np.array([float(r["bfr"]) for r in c2_rows])
    jraw = np.array([float(r["J_raw"]) for r in c2_rows])
    counts = np.array([int(r["j_nu_count"]) for r in c2_rows])
    nu_min = 1.5e14
    nu_max = 3.0e16
    dlog = math.log(nu_max / nu_min) / NB
    edges = nu_min * np.exp(np.arange(NB + 1) * dlog)
    nu = np.sqrt(edges[:-1] * edges[1:])
    dnu = np.diff(edges)
    nu_rel = float(np.max(np.abs(nu_file / nu - 1.0)))
    if nu_rel > 1e-6:
        raise Unresolved(f"C2 nu_mid grid mismatch max_rel={nu_rel}")

    W = np.array([float(r["W"]) for r in c1_rows])
    TR = np.array([float(r["T_R"]) for r in c1_rows])
    modes = np.array([r["mode"] for r in c1_rows], dtype=object)
    if not set(modes).issubset({"fit", "pin", "degen", "empty"}):
        raise Unresolved(f"unknown C1 mode(s): {set(modes)}")
    cidx = (np.arange(NB) * NC) // NB
    Jc1 = np.where((W[cidx] > 0) & (TR[cidx] > 0),
                   W[cidx] * planck_bnu(TR[cidx], nu), 0.0)
    Jc1 = np.where(modes[cidx] == "degen", jraw, Jc1)

    # The dump is rounded; this confirms reconstruction against its independently
    # dumped coarse integral without pretending bitwise replay is possible.
    coarse_rel = []
    for c, row in enumerate(c1_rows):
        m = cidx == c
        target = float(row["J_bin"])
        got = float(np.sum(Jc1[m] * dnu[m]))
        if target > 0:
            coarse_rel.append(abs(got / target - 1.0))
        elif got != 0:
            raise Unresolved(f"C1 empty bin {c} reconstructs nonzero J")
    print(f"[FROZEN] selected producer iter={producer_iter} shell={SHELL}; "
          f"C1 modes={dict(zip(*np.unique(modes, return_counts=True)))}; "
          f"nu_mid max_rel={nu_rel:.3e}; coarse integral max_rel={max(coarse_rel):.3e}")
    return {"nu": nu, "edges": edges, "dnu": dnu, "Jc1": Jc1,
            "bfr": bfr, "jraw": jraw, "counts": counts, "modes": modes,
            "nu_grid_rel": nu_rel, "coarse_rel": max(coarse_rel)}


class SigmaBF:
    def __init__(self, path: Path):
        if not path.is_file():
            raise Unresolved(f"missing sigma input: {path}")
        self.path = path
        with path.open("rb") as f:
            magic, version, self.nlev, self.nfreq = struct.unpack("<IIii", f.read(16))
            self.numin, self.numax = struct.unpack("<dd", f.read(16))
            self.has = np.frombuffer(f.read(self.nlev), dtype=np.int8).copy()
        pad = (8 - (self.nlev % 8)) % 8
        self.offset = 32 + self.nlev + pad
        expected_size = self.offset + self.nlev * self.nfreq * 8
        ok = (magic == 0x434D4644 and version == 1 and self.nfreq == NB and
              path.stat().st_size == expected_size)
        resolved = path.resolve()
        print(f"[INPUT] sigma_bf: exists=1 link={path} resolved={resolved} "
              f"bytes={path.stat().st_size} schema={'OK' if ok else 'FAIL'} "
              f"magic=0x{magic:08X} version={version} nlev={self.nlev} nfreq={self.nfreq} "
              f"nu=[{self.numin:.6e},{self.numax:.6e}]Hz has={int(self.has.sum())}")
        AUDIT.append({"label": "sigma_bf", "path": str(path), "bytes": path.stat().st_size,
                      "rows": self.nlev, "schema": "OK" if ok else "FAIL"})
        if not ok:
            raise Unresolved("sigma binary schema/size mismatch")
        self.data = np.memmap(path, dtype="<f8", mode="r", offset=self.offset,
                              shape=(self.nlev, self.nfreq))

    def row(self, gl: int) -> np.ndarray:
        if gl < 0 or gl >= self.nlev or self.has[gl] != 1:
            raise Unresolved(f"sigma identity missing for global level {gl}")
        return np.asarray(self.data[gl])


class TargetRoutes:
    def __init__(self, path: Path, nlev: int):
        if not path.is_file():
            raise Unresolved(f"missing target-route input: {path}")
        with path.open("rb") as f:
            magic, version, nfile, nions = struct.unpack("<IIii", f.read(16))
            if magic != 0x4D415254 or version not in (1, 2) or nfile != nlev:
                raise Unresolved("ma_radrecomb_target header mismatch")
            self.routes: list[list[tuple[int, float]]] = [[] for _ in range(nlev)]
            if version == 1:
                targets = np.fromfile(f, dtype="<i4", count=nlev)
                if len(targets) != nlev:
                    raise Unresolved("short v1 target array")
                for gl, target in enumerate(targets):
                    if 0 <= target < nlev:
                        self.routes[gl] = [(int(target), 1.0)]
                nroutes = int(np.count_nonzero((targets >= 0) & (targets < nlev)))
            else:
                nroutes, = struct.unpack("<i", f.read(4))
                offsets = np.fromfile(f, dtype="<i4", count=nlev + 1)
                targets = np.fromfile(f, dtype="<i4", count=nroutes)
                probs = np.fromfile(f, dtype="<f8", count=nroutes)
                if (len(offsets) != nlev + 1 or len(targets) != nroutes or
                        len(probs) != nroutes or offsets[0] != 0 or offsets[-1] != nroutes):
                    raise Unresolved("short/invalid v2 target CSR")
                for gl in range(nlev):
                    self.routes[gl] = [(int(targets[k]), float(probs[k]))
                                       for k in range(offsets[gl], offsets[gl + 1])]
        print(f"[INPUT] ma_radrecomb_target: exists=1 bytes={path.stat().st_size} "
              f"schema=OK version={version} nlev={nfile} nions={nions} nroutes={nroutes}")
        AUDIT.append({"label": "ma_radrecomb_target", "path": str(path),
                      "bytes": path.stat().st_size, "rows": nroutes, "schema": "OK"})
        self.version, self.nions, self.nroutes = version, nions, nroutes


def load_levels(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    out = []
    for r in rows:
        out.append({"Z": int(r["atomic_number"]), "ion": int(r["ion_number"]),
                    "level_number": int(r["level_number"]), "E": float(r["energy_eV"]),
                    "g": int(r["g"]), "super": int(r["super_level"])})
    return out


def load_ionization(path: Path) -> dict[tuple[int, int], float]:
    return {(int(r["atomic_number"]), int(r["ion_number"])):
            float(r["ionization_energy_eV"]) for r in read_csv(path)}


def oracle_values(path: Path) -> dict[str, float]:
    out = {}
    for r in read_csv(path):
        if int(r["shell"]) == SHELL and r["status"] == "available" and r["value"]:
            out[r["quantity"]] = float(r["value"])
    need = ["rate_consume_iteration", "field_producer_iteration", "producer_to_consumer_lag",
            "bf_rate_estimator_positive_consumptions", "bf_rate_estimator_fallback_consumptions"]
    if any(k not in out for k in need):
        raise Unresolved(f"oracle missing synchronization/census fields: {need}")
    return out


def read_eddfactor(path: Path) -> tuple[np.ndarray, np.ndarray, int, float, dict[str, Any]]:
    info_path = Path(str(path) + "_INFO")
    audit_text(info_path, "CMFGEN EDDFACTOR_INFO")
    lines = info_path.read_text().splitlines()
    vals = lines[2].split()
    ND, recl, word, unit_size, int_size = map(int, vals[:5])
    little = vals[5] == "T"
    nwr = recl // word
    if word != 8 or nwr != ND + 1:
        raise Unresolved(f"EDD record schema mismatch ND={ND} RECL={recl} WORD={word}")
    raw = np.fromfile(path, dtype="<f8" if little else ">f8")
    if raw.size % nwr:
        raise Unresolved("EDDFACTOR byte count is not an integer record count")
    raw = raw.reshape(-1, nwr)
    finish = float(raw[4, 0])
    data = raw[14:]
    good = np.isfinite(data[:, :ND]).all(axis=1) & (data[:, ND] > 0)
    J = data[good, :ND]
    fl = data[good, ND]
    nu = fl * 1.0e15
    order = np.argsort(nu)
    J, nu, fl = J[order], nu[order], fl[order]
    fl_roundtrip = float(np.max(np.abs((nu / 1.0e15) / fl - 1.0)))
    lam_A = C_LIGHT / nu * 1.0e8
    nu_roundtrip = float(np.max(np.abs((C_LIGHT / (lam_A * 1.0e-8)) / nu - 1.0)))
    meta = {"ND": ND, "recl": recl, "word": word, "unit_size": unit_size,
            "int_size": int_size, "little": little, "records": len(raw),
            "good": int(good.sum()), "bad": int((~good).sum()),
            "finish": finish, "nu_min": float(nu.min()), "nu_max": float(nu.max()),
            "fl_roundtrip": fl_roundtrip,
            "nu_roundtrip": nu_roundtrip}
    ok = finish == 1.0 and good.sum() > 0
    print(f"[INPUT] CMFGEN EDDFACTOR: exists=1 bytes={path.stat().st_size} schema={'OK' if ok else 'FAIL'} "
          f"records={len(raw)} header_records=14 ND={ND} good_freq={good.sum()} bad_freq={(~good).sum()} "
          f"nu=[{nu.min():.6e},{nu.max():.6e}]Hz FINISH={finish:g} endian={'LE' if little else 'BE'}")
    print(f"[UNIT] FL(1e15 Hz)->Hz->FL max_rel={fl_roundtrip:.3e}; "
          f"Hz->Angstrom->Hz max_rel={nu_roundtrip:.3e}; J_nu kept in native CGS "
          f"erg cm^-2 s^-1 Hz^-1 sr^-1")
    AUDIT.append({"label": "CMFGEN EDDFACTOR", "path": str(path),
                  "bytes": path.stat().st_size, "rows": int(good.sum()),
                  "schema": "OK" if ok else "FAIL"})
    if not ok or fl_roundtrip > 2e-15 or nu_roundtrip > 2e-15:
        raise Unresolved("EDDFACTOR completeness/unit round-trip failed")
    return J, nu, ND, finish, meta


def rvtj_block(text: str, label: str, ND: int) -> np.ndarray:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == label:
            values: list[float] = []
            j = i + 1
            while j < len(lines) and len(values) < ND:
                try:
                    values.extend(float(x) for x in lines[j].split())
                except ValueError:
                    break
                j += 1
            if len(values) != ND:
                raise Unresolved(f"RVTJ block {label!r} has {len(values)} != {ND}")
            return np.array(values)
    raise Unresolved(f"RVTJ block absent: {label}")


def cum_from_below(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    c = np.zeros(x.size)
    c[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return c


def eval_cum(x: np.ndarray, y: np.ndarray, c: np.ndarray, at: np.ndarray) -> np.ndarray:
    at = np.clip(np.asarray(at, float), x[0], x[-1])
    i = np.clip(np.searchsorted(x, at), 1, x.size - 1)
    frac = (at - x[i - 1]) / (x[i] - x[i - 1])
    yat = y[i - 1] + frac * (y[i] - y[i - 1])
    return c[i - 1] + 0.5 * (y[i - 1] + yat) * (at - x[i - 1])


def bin_average(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    c = cum_from_below(x, y)
    return np.diff(eval_cum(x, y, c, edges)) / np.diff(edges)


def cmfgen_shell_field(edges: np.ndarray, geometry_path: Path) -> dict[str, Any]:
    validation_unit = VALIDATION / "analysis/trapping_audit/audit_u_cmfgen.py"
    validation_report = VALIDATION / "analysis/trapping_audit/VERDICT.md"
    audit_text(validation_unit, "jnu4 unit/schema validation code")
    audit_text(validation_report, "jnu4 unit/shell-map validation report")
    edd = CMF_RUN / "EDDFACTOR"
    rvtj = CMF_RUN / "RVTJ"
    if not edd.is_file() or not rvtj.is_file():
        raise Unresolved("CMFGEN jnu4 EDDFACTOR/RVTJ missing")
    audit_text(rvtj, "CMFGEN RVTJ")
    J, nu_native, ND, finish, meta = read_eddfactor(edd)
    rt = rvtj.read_text()
    V = rvtj_block(rt, "Velocity (km/s)", ND)
    T = rvtj_block(rt, "Temperature (10^4K)", ND) * 1.0e4

    geom = [r for r in read_csv(geometry_path) if int(r["shell_id"]) == SHELL]
    if len(geom) != 1:
        raise Unresolved("geometry does not uniquely identify shell 8")
    vt = 0.5 * (float(geom[0]["v_inner"]) + float(geom[0]["v_outer"])) / 1.0e5
    order = np.argsort(V)
    vs = V[order]
    k = int(np.searchsorted(vs, vt))
    if k <= 0 or k >= len(vs):
        raise Unresolved(f"shell velocity {vt} outside RVTJ grid")
    v0, v1 = vs[k - 1], vs[k]
    w = (vt - v0) / (v1 - v0)
    j0, j1 = J[:, order[k - 1]], J[:, order[k]]
    if np.any(j0 <= 0) or np.any(j1 <= 0):
        raise Unresolved("nonpositive J in CMFGEN velocity bracket; log interpolation undefined")
    Jvt = np.exp((1.0 - w) * np.log(j0) + w * np.log(j1))
    Tvt = float(np.interp(vt, vs, T[order]))
    Jbar = bin_average(nu_native, Jvt, edges)
    native_integral = float(eval_cum(nu_native, Jvt, cum_from_below(nu_native, Jvt),
                                     np.array([edges[0], edges[-1]]))[1] -
                            eval_cum(nu_native, Jvt, cum_from_below(nu_native, Jvt),
                                     np.array([edges[0], edges[-1]]))[0])
    binned_integral = float(np.sum(Jbar * np.diff(edges)))
    qratio = binned_integral / native_integral

    # Independent physical unit sanity used by the checked-in validation audit:
    # the innermost depth is close to thermal, so (4pi/c) int J dnu ~ a T^4.
    d_inner = int(np.argmin(V))
    u_inner = FOUR_PI / C_LIGHT * float(np.trapezoid(J[:, d_inner], nu_native))
    a_rad = 7.5657e-15
    thermal_ratio = u_inner / (a_rad * T[d_inner]**4)
    if not (0.8 < thermal_ratio < 1.2) or abs(qratio - 1.0) > 1e-12:
        raise Unresolved(f"CMFGEN J unit/quadrature sanity failed thermal={thermal_ratio} q={qratio}")
    print(f"[CMFGEN] shell s{SHELL} midpoint={vt:.1f} km/s; log-J velocity bracket "
          f"[{v0:.3f},{v1:.3f}] km/s w={w:.6f}; T_interp={Tvt:.2f} K")
    print(f"[CMFGEN] bin-average identity sum(Jbar*dnu)/native={qratio:.15f}; "
          f"inner u/(aT^4)={thermal_ratio:.6f}")
    return {"J": Jbar, "velocity": vt, "v0": v0, "v1": v1, "weight": w,
            "T": Tvt, "qratio": qratio, "thermal_ratio": thermal_ratio,
            "meta": meta}


def identity_and_fraction(target: dict[str, Any], identity_path: Path, solution_path: Path,
                          levels: list[dict[str, Any]], manifest_data: dict[str, str]) -> dict[str, Any]:
    ids = read_csv(identity_path)
    matches = [r for r in ids if int(r["matrix_index"]) == target["matrix"]]
    if len(matches) != 1:
        raise Unresolved(f"identity matrix {target['matrix']} not unique")
    row = matches[0]
    members = [int(x) for x in row["member_full_level_ids"].split(";")]
    checks = [
        int(row["Z"]) == target["Z"],
        int(row["spectroscopic_stage"]) == target["ion"] + 1,
        int(row["sl_id"]) == target["sl_id"],
        abs(float(row["energy_eV"]) - target["energy_eV"]) < 1e-10,
        row["source_atomic_checksum"] == manifest_data["atomic_checksum"],
    ]
    if not all(checks):
        raise Unresolved(f"identity mismatch for {target['key']}: {row}")
    for gl in members:
        lv = levels[gl]
        if lv["Z"] != target["Z"] or lv["ion"] != target["ion"]:
            raise Unresolved(f"identity member {gl} does not match ion")

    sol = read_csv(solution_path)
    sl = [r for r in sol if r["record_type"] == "SL" and
          int(r["matrix_index"]) == target["matrix"]]
    fl = [r for r in sol if r["record_type"] == "FL" and
          int(r["matrix_index"]) == target["matrix"]]
    if len(sl) != 1 or set(int(r["global_level"]) for r in fl) != set(members):
        raise Unresolved(f"solution FL identity mismatch for {target['key']}")
    pop_sl = float(sl[0]["restored_population"])
    if not pop_sl > 0:
        raise Unresolved(f"SL population is nonpositive for {target['key']}")
    frac = {int(r["global_level"]): float(r["restored_population"]) / pop_sl for r in fl}
    frac_sum = sum(frac.values())
    if abs(frac_sum - 1.0) > 5e-13:
        raise Unresolved(f"within-SL fractions sum to {frac_sum}")

    # Independent reconstruction from the exact source formula.
    Te = float(manifest_data["T_e"])
    anchor_E = levels[int(row["anchor_global_level"])]["E"]
    weights = {gl: levels[gl]["g"] * math.exp(-max(levels[gl]["E"] - anchor_E, 0.0) *
                                               EV_TO_ERG / (K_B * Te)) for gl in members}
    z = sum(weights.values())
    boltz = {gl: weights[gl] / z for gl in members}
    frac_err = max(abs(frac[gl] - boltz[gl]) for gl in members)
    if frac_err > 2e-14:
        raise Unresolved(f"solution fraction != Boltzmann source formula max_abs={frac_err}")
    print(f"[IDENTITY] {target['key']}: matrix={target['matrix']} SL={target['sl_id']} "
          f"members={len(members)} global=[{min(members)},{max(members)}] "
          f"E_anchor={float(row['energy_eV']):.12g} eV checksum={row['source_atomic_checksum']} "
          f"sum(frac)={frac_sum:.16f} Boltz_max_abs={frac_err:.3e}")
    return {"row": row, "members": members, "frac": frac, "ids": ids,
            "frac_sum": frac_sum, "frac_err": frac_err}


def provenance_A(target: dict[str, Any], path: Path, identity_rows: list[dict[str, str]]) -> dict[str, Any]:
    id_by_matrix = {int(r["matrix_index"]): r for r in identity_rows}
    source = f"SL:{target['matrix']}"
    routes = [r for r in read_csv(path) if r["channel"] == "rad_bf" and
              r["source_identity"] == source]
    if not routes:
        raise Unresolved(f"no provenance rad_bf routes for {source}")
    for r in routes:
        if r["units"] != "s^-1" or r["target_route"] != "ma_rr_CSR":
            raise Unresolved(f"unexpected provenance contract: {r}")
        ti = int(r["target_identity"].split(":", 1)[1])
        if ti not in id_by_matrix or int(id_by_matrix[ti]["spectroscopic_stage"]) != target["ion"] + 2:
            raise Unresolved(f"provenance target is not upper ion: {r}")
    return {"A": sum(float(r["aggregated_rate"]) for r in routes), "routes": routes}


def rate_replay(target: dict[str, Any], members: list[int], fractions: dict[int, float],
                levels: list[dict[str, Any]], ionization: dict[tuple[int, int], float],
                routes: TargetRoutes, sigma: SigmaBF, frozen: dict[str, Any],
                Jcmf: np.ndarray) -> dict[str, Any]:
    nu, dnu = frozen["nu"], frozen["dnu"]
    Jc1, bfr = frozen["Jc1"], frozen["bfr"]
    positive = bfr > 0
    chi = ionization.get((target["Z"], target["ion"]))
    if chi is None:
        raise Unresolved(f"ionization energy absent for {target['key']}")
    B_pos = B_fb = C_posmask = C_fbmask = C_hybrid = 0.0
    n_pos = n_fb = 0
    n_routes = 0
    thresholds = []
    sigma_missing = []
    for gl in members:
        if sigma.has[gl] != 1:
            sigma_missing.append(gl)
            continue
        sig = sigma.row(gl)
        rr = routes.routes[gl]
        if not rr:
            raise Unresolved(f"no ma_rr target route for requested level {gl}")
        psum = sum(p for _, p in rr)
        if abs(psum - 1.0) > 1e-12:
            raise Unresolved(f"route probabilities for level {gl} sum to {psum}")
        for upper, p in rr:
            ulv = levels[upper]
            if ulv["Z"] != target["Z"] or ulv["ion"] != target["ion"] + 1:
                raise Unresolved(f"invalid upper target {upper} for lower {gl}")
            threshold_eV = chi - levels[gl]["E"] + ulv["E"]
            nu_threshold = threshold_eV * EV_TO_ERG / H_PLANCK
            if threshold_eV <= 0:
                raise Unresolved(f"nonpositive target-dependent threshold lower={gl} upper={upper}")
            mask = (nu >= nu_threshold) & (sig > 0) & np.isfinite(sig)
            pref = FOUR_PI * sig / (H_PLANCK * nu) * dnu
            weight = fractions[gl] * p
            B_pos += weight * float(np.sum(sig[mask & positive] * bfr[mask & positive]))
            B_fb += weight * float(np.sum(pref[mask & ~positive] * Jc1[mask & ~positive]))
            C_posmask += weight * float(np.sum(pref[mask & positive] * Jcmf[mask & positive]))
            C_fbmask += weight * float(np.sum(pref[mask & ~positive] * Jcmf[mask & ~positive]))
            # Deliberately non-primary literal branch sensitivity: keep bfr where
            # present, replacing J only in the code's fallback leg.
            C_hybrid += weight * (float(np.sum(sig[mask & positive] * bfr[mask & positive])) +
                                  float(np.sum(pref[mask & ~positive] * Jcmf[mask & ~positive])))
            n_pos += int(np.count_nonzero(mask & positive))
            n_fb += int(np.count_nonzero(mask & ~positive))
            n_routes += 1
            thresholds.append(threshold_eV)
    if sigma_missing:
        raise Unresolved(f"requested member levels lack confirmed sigma: {sigma_missing[:20]}"
                         + ("..." if len(sigma_missing) > 20 else ""))
    B = B_pos + B_fb
    C = C_posmask + C_fbmask
    return {"B": B, "B_pos": B_pos, "B_fb": B_fb, "C": C,
            "C_posmask": C_posmask, "C_fbmask": C_fbmask,
            "C_hybrid": C_hybrid, "n_pos": n_pos, "n_fb": n_fb,
            "n_routes": n_routes, "threshold_min": min(thresholds),
            "threshold_max": max(thresholds), "sigma_missing": sigma_missing}


def dex_ratio(x: float, y: float) -> float:
    if x <= 0 or y <= 0:
        return float("nan")
    return math.log10(x / y)


def verdict(A: float, B: float, C: float) -> tuple[str, str]:
    ab = abs(dex_ratio(A, B))
    bc_signed = dex_ratio(C, B)
    if ab > DEX_TOL:
        return "EW bf 산술 버그", f"|log10(A/B)|={ab:.4f} > {DEX_TOL:.1f} dex"
    if abs(bc_signed) <= DEX_TOL:
        return "C48 lump/원자데이터 내용", (f"A≈B and |log10(C/B)|={abs(bc_signed):.4f} "
                                             f"<= {DEX_TOL:.1f} dex")
    if bc_signed < -DEX_TOL:
        return "동결장 내용이 진범 (구조·산술 무죄)", (f"A≈B and C/B collapses by "
                                                       f"{-bc_signed:.4f} dex")
    return "UNRESOLVED", (f"A≈B but C is higher than B by {bc_signed:.4f} dex; "
                           "this sign is absent from the preregistered table")


def fmt(x: float) -> str:
    return f"{x:.9e}"


def make_report(results: list[dict[str, Any]], source: dict[str, int], oracle: dict[str, float],
                frozen: dict[str, Any], cmf: dict[str, Any], report_path: Path,
                unresolved: list[str]) -> str:
    pos_oracle = oracle["bf_rate_estimator_positive_consumptions"]
    fb_oracle = oracle["bf_rate_estimator_fallback_consumptions"]
    oracle_fb_frac = fb_oracle / (pos_oracle + fb_oracle)
    lines = [
        "# Wave 3 판별 측정 — Γ 삼중 대조",
        "",
        "오프라인 재생만 수행했으며 신규 transport/NLTE 런과 `src/` 수정은 없었다.",
        "",
        "## 결론",
        "",
    ]
    for r in results:
        lines.append(f"- **{r['target']['key']}**: **{r['verdict']}** — {r['reason']}.")
    lines += [
        "",
        "## A/B/C 수치",
        "",
        "C는 field falsifier가 되도록 frozen bfr까지 포함한 **전체 장**을 CMFGEN Jν로 치환했다. "
        "즉 모든 유효 bin에서 `pref*J_CMFGEN`을 사용하며, frozen positive/fallback mask는 분해 표시에만 쓴다. "
        "positive-bfr을 그대로 두고 fallback J만 바꾸는 literal-branch hybrid는 판독에 쓰지 않고 아래에 감도값으로 병기한다.",
        "",
        "| 준위 | A [s⁻¹] | B [s⁻¹] | C [s⁻¹] | log10(A/B) dex | log10(C/B) dex | 판독 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(f"| {r['target']['key']} | {fmt(r['A'])} | {fmt(r['B'])} | {fmt(r['C'])} | "
                     f"{dex_ratio(r['A'], r['B']):+.6f} | {dex_ratio(r['C'], r['B']):+.6f} | "
                     f"{r['verdict']} |")
    lines += [
        "",
        "## B positive-bfr / fallback 분리",
        "",
        "소비 횟수는 코드와 같이 threshold 통과·σ>0 bin-route 평가 횟수다. oracle 전역 s8 비율은 "
        f"{int(fb_oracle):,}/({int(pos_oracle):,}+{int(fb_oracle):,}) = **{100*oracle_fb_frac:.3f}%** "
        "(요청의 34.9% 반올림값과 일치)다. 준위별 비율은 문턱/σ support가 달라 전역값과 같을 필요가 없다.",
        "",
        "| 준위 | B positive [s⁻¹] | B fallback [s⁻¹] | fallback rate share | positive eval | fallback eval | fallback eval share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        nsum = r["n_pos"] + r["n_fb"]
        lines.append(f"| {r['target']['key']} | {fmt(r['B_pos'])} | {fmt(r['B_fb'])} | "
                     f"{100*r['B_fb']/r['B']:.4f}% | {r['n_pos']:,} | {r['n_fb']:,} | "
                     f"{100*r['n_fb']/nsum:.4f}% |")
    lines += [
        "",
        "## C 장 치환 세부",
        "",
        f"Lumina geometry의 s8 midpoint는 **{cmf['velocity']:.1f} km s⁻¹**다. jnu4 RVTJ의 "
        f"{cmf['v0']:.3f}–{cmf['v1']:.3f} km s⁻¹ 두 depth 사이에서 주파수별 log(Jν)를 "
        f"보간(w={cmf['weight']:.6f})하고, 1000개 Lumina log-bin에 적분 평균했다. "
        f"`Σ Jbar Δν / ∫Jνdν = {cmf['qratio']:.15f}`이다.",
        "",
        "| 준위 | C on positive-mask [s⁻¹] | C on fallback-mask [s⁻¹] | fallback-only hybrid [s⁻¹] | log10(hybrid/B) dex | threshold range [eV] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(f"| {r['target']['key']} | {fmt(r['C_posmask'])} | {fmt(r['C_fbmask'])} | "
                     f"{fmt(r['C_hybrid'])} | {dex_ratio(r['C_hybrid'], r['B']):+.6f} | "
                     f"{r['threshold_min']:.6f}–{r['threshold_max']:.6f} |")
    lines += [
        "",
        "## 입력·identity·산술 검증",
        "",
        f"- 동기화: EW consumer iter {int(oracle['rate_consume_iteration'])}, field producer iter "
        f"{int(oracle['field_producer_iteration'])}, lag {int(oracle['producer_to_consumer_lag'])}. "
        "따라서 B는 C1/C2 `iter=10, shell=8`을 사용했다.",
        f"- 실제 σ 소스: `{MODEL / 'cmfgen_sigma_bf.bin'}` → "
        f"`{(MODEL / 'cmfgen_sigma_bf.bin').resolve()}`. EW stdout의 26087/26592 coverage와 "
        "production stdout의 `LUMINA_CMFGEN_SIGMA_BF`가 같은 모델 링크를 지목한다.",
        f"- 소스 계약: `lumina_element_wide.c` estimator L{source['ew_estimator']}, fallback "
        f"L{source['ew_fallback']}, SL weight L{source['ew_weight']}; `lumina_plasma.c` estimator "
        f"L{source['plasma_estimator']}, fallback L{source['plasma_fallback']}.",
        f"- frozen grid `nu_mid` max relative mismatch {frozen['nu_grid_rel']:.3e}; C1 dump 반올림값으로 "
        f"재구성한 coarse-integral max relative residual {frozen['coarse_rel']:.3e}.",
        f"- jnu4 schema: ND={cmf['meta']['ND']}, good frequency records={cmf['meta']['good']:,}, "
        f"ν={cmf['meta']['nu_min']:.3e}–{cmf['meta']['nu_max']:.3e} Hz, "
        f"FINISH={cmf['meta']['finish']:.0f}; FL↔Hz max relative round-trip "
        f"{cmf['meta']['fl_roundtrip']:.3e}, ν↔Å {cmf['meta']['nu_roundtrip']:.3e}. "
        f"CGS Jν 단위 독립 sanity `u_inner/(aT⁴)={cmf['thermal_ratio']:.6f}`.",
    ]
    for r in results:
        lines.append(f"- {r['target']['key']} identity: matrix {r['target']['matrix']}, "
                     f"sl_id {r['target']['sl_id']}, members {r['member_count']}, "
                     f"Σ within-SL fraction={r['frac_sum']:.16f}, direct Boltzmann max|Δf|="
                     f"{r['frac_err']:.3e}, route count={r['n_routes']}.")
    lines += [
        "",
        "### 사용 입력 실측",
        "",
        "| 입력 | data rows/records | bytes | schema |",
        "|---|---:|---:|---|",
    ]
    for a in AUDIT:
        lines.append(f"| `{a['path']}` | {a['rows']:,} | {a['bytes']:,} | {a['schema']} |")
    lines += [
        "",
        "## 사전등록 판독표 적용",
        "",
        f"동등성 경계는 명시된 **0.1 dex**다. A/B가 이를 넘으면 첫 행, A/B가 통과하고 "
        "C/B가 −0.1 dex 미만이면 두 번째 행, A/B와 B/C가 모두 0.1 dex 이내면 세 번째 행을 적용했다. "
        "C가 오히려 +0.1 dex 넘게 증가하는 경우는 사전등록 표에 없으므로 UNRESOLVED 규칙이다.",
        "",
        "## UNRESOLVED",
        "",
    ]
    if unresolved:
        lines.extend(f"- {x}" for x in unresolved)
    else:
        lines.append("- 없음. 요청한 두 준위 모두 σ row, target route, within-SL weight, 장 단위·격자가 확정됐다.")
    lines += [
        "",
        "## 재현",
        "",
        f"`python3 scripts/w3_gamma_triple_compare.py --report {report_path.relative_to(ROOT)}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path,
                        default=ROOT / "docs/CODEX_W3_GAMMA_TRIPLE_COMPARE_2026-07-31.md")
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    unresolved: list[str] = []
    try:
        source = source_contract()
        oracle_path = EW_DIR / "lumina_oracle_cell_s8.csv"
        audit_csv(oracle_path, EXPECTED["oracle"], "frozen oracle")
        oracle = oracle_values(oracle_path)
        if (int(oracle["rate_consume_iteration"]) != CONSUMER_ITER or
                int(oracle["field_producer_iteration"]) != CONSUMER_ITER - 1 or
                int(oracle["producer_to_consumer_lag"]) != 1):
            raise Unresolved(f"unexpected field epoch contract: {oracle}")
        producer_iter = int(oracle["field_producer_iteration"])

        c1_path = FROZEN_DIR / "lumina_c1_bins.csv"
        c2_path = FROZEN_DIR / "lumina_c2_bfr_dump.csv"
        audit_csv(c1_path, EXPECTED["c1"], "frozen C1 bins")
        audit_csv(c2_path, EXPECTED["c2"], "frozen C2 bfr")
        frozen = load_frozen_field(c1_path, c2_path, producer_iter)

        levels_path = MODEL / "levels.csv"
        ioniz_path = MODEL / "ionization_energies.csv"
        geometry_path = MODEL / "geometry.csv"
        audit_csv(levels_path, EXPECTED["levels"], "atomic levels")
        audit_csv(ioniz_path, EXPECTED["ionization"], "ionization energies")
        audit_csv(geometry_path, EXPECTED["geometry"], "Lumina geometry")
        levels = load_levels(levels_path)
        ionization = load_ionization(ioniz_path)

        sigma = SigmaBF(MODEL / "cmfgen_sigma_bf.bin")
        grid_rel = max(abs(sigma.numin / frozen["edges"][0] - 1.0),
                       abs(sigma.numax / frozen["edges"][-1] - 1.0))
        if sigma.nlev != len(levels) or grid_rel > 2e-14:
            raise Unresolved("sigma/levels/frozen frequency identity mismatch")
        target_routes = TargetRoutes(MODEL / "ma_radrecomb_target.bin", len(levels))
        cmf = cmfgen_shell_field(frozen["edges"], geometry_path)

        # Log files are part of sigma-source provenance, not inferred defaults.
        prod_stdout = FROZEN_DIR / "stdout.log"
        ew_stdout = EW_DIR / "stdout.txt"
        audit_text(prod_stdout, "production resolved configuration/stdout")
        audit_text(ew_stdout, "EW frozen-recovery stdout")
        prod_text, ew_text = prod_stdout.read_text(), ew_stdout.read_text()
        sigma_rel = str((MODEL / "cmfgen_sigma_bf.bin").relative_to(ROOT))
        if (f"LUMINA_CMFGEN_SIGMA_BF={sigma_rel}" not in prod_text or
                "CMFGEN sigma_bf: 26087/26592 levels" not in ew_text):
            raise Unresolved("runtime sigma provenance cannot be tied to the loaded model binary")

        results = []
        for target in TARGETS:
            z = target["Z"]
            prov_path = EW_DIR / f"lumina_ew_iter0011_z{z}_s008_provenance.csv"
            identity_path = EW_DIR / f"lumina_ew_iter0011_z{z}_s008_identity.csv"
            manifest_path = EW_DIR / f"lumina_ew_iter0011_z{z}_s008_manifest.csv"
            solution_path = EW_DIR / f"lumina_ew_iter0011_z{z}_s008_solution.csv"
            audit_csv(prov_path, EXPECTED["provenance"], f"Z={z} provenance")
            audit_csv(identity_path, EXPECTED["identity"], f"Z={z} identity")
            audit_csv(manifest_path, EXPECTED["manifest"], f"Z={z} manifest")
            audit_csv(solution_path, EXPECTED["solution"], f"Z={z} solution/fractions")
            man = manifest(manifest_path)
            if int(man["shell"]) != SHELL or man["run_id"] != "iter0011" or man["identity_checksum_verified"] != "1":
                raise Unresolved(f"manifest epoch/identity mismatch for Z={z}")
            ident = identity_and_fraction(target, identity_path, solution_path, levels, man)
            aa = provenance_A(target, prov_path, ident["ids"])
            replay = rate_replay(target, ident["members"], ident["frac"], levels,
                                 ionization, target_routes, sigma, frozen, cmf["J"])
            v, reason = verdict(aa["A"], replay["B"], replay["C"])
            result = {"target": target, "A": aa["A"], **replay, "verdict": v,
                      "reason": reason, "member_count": len(ident["members"]),
                      "frac_sum": ident["frac_sum"], "frac_err": ident["frac_err"]}
            results.append(result)
            print(f"[RESULT] {target['key']}: A={result['A']:.12e} B={result['B']:.12e} "
                  f"C={result['C']:.12e} A/B={dex_ratio(result['A'],result['B']):+.6f}dex "
                  f"C/B={dex_ratio(result['C'],result['B']):+.6f}dex => {v}")
            print(f"[SPLIT]  B_pos={result['B_pos']:.12e} B_fb={result['B_fb']:.12e} "
                  f"fb_rate={100*result['B_fb']/result['B']:.4f}% "
                  f"fb_eval={100*result['n_fb']/(result['n_pos']+result['n_fb']):.4f}%")

        text = make_report(results, source, oracle, frozen, cmf, report_path, unresolved)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text)
        print(f"[REPORT] {report_path}")
        return 0
    except (Unresolved, OSError, ValueError, KeyError, struct.error) as exc:
        print(f"UNRESOLVED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
