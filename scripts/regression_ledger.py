#!/usr/bin/env python3
"""Append-only regression ledger for archived Lumina run directories.

This program is an offline reader.  It never launches Lumina, a model, or a GPU
kernel.  Metric definitions are schema data: changing them requires incrementing
LEDGER_SCHEMA_VERSION and must never be used to rewrite old JSONL rows.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import sys
import tempfile
from typing import Any, Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover - reported cleanly by main
    np = None


LEDGER_SCHEMA_VERSION = 2
C_A_S = 2.99792458e18
C_CGS = 2.99792458e10
EDDFACTOR_FL_TO_HZ = 1.0e15
C_A_PER_EDDFACTOR_FL = C_A_S / EDDFACTOR_FL_TO_HZ
SIGMA_T = 6.6524587321e-25
H_CGS = 6.62607015e-27
K_CGS = 1.380649e-16
K_B_EV = 8.617333262e-5
FOUR_PI_OVER_C = 4.0 * math.pi / C_CGS
DEFAULT_CMFGEN = Path("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4")

UV_FORMAL = (500.0, 3500.0)
FORMAL_TOTAL = (500.0, 20000.0)
EUV = (450.0, 918.0)
FUV = (918.0, 1290.0)

B_K_DEFINITION = (
    "b_k=(n_k/n_ground)/[(g_k/g_ground)*exp(-(E_k-E_ground)/(k_B*T_e))]; "
    "T_e is the local lumina_plasma_state electron temperature, ground is the first "
    "recorded level of the same (Z,0-based ion), and the LTE reference is the "
    "same-ion ratio of Saha-Boltzmann populations (the ion's Saha factor cancels); "
    "no T_rad, dilution W, g-weighted aggregation, floor, cap, or replacement is used"
)

METRIC_DEFINITIONS = {
    "uv_quotient": (
        "formal UV fraction = integral_[500,3500) F_lambda d_lambda / "
        "integral_[500,20000) F_lambda d_lambda from lumina_spectrum_formal.csv; "
        "trapezoids join the native bin-center samples selected inside each half-open "
        "wavelength band; no boundary extrapolation or invented endpoint is used"
    ),
    "electron_temperature": (
        "per shell: Lumina T_e from lumina_plasma_state.csv; CMFGEN truth is the "
        "TIME=19.48 d block of data/standart_data1/toy06/phys_toy06_cmfgen.txt "
        "linearly interpolated in shell midpoint velocity from model geometry.csv; "
        "difference_K=Lumina-CMFGEN and ratio=Lumina/CMFGEN"
    ),
    "radiation_energy_density": (
        "per shell and lane mc/cs/CMFGEN: u=(4*pi/c)*integral J_nu dnu over "
        "each source's complete finite native frequency grid; Lumina uses "
        "lumina_coevolve_field.csv and CMFGEN uses all valid EDDFACTOR data records "
        "after its 14 records of metadata, with the EDDFACTOR FL column interpreted in "
        "10^15 Hz; CMFGEN depth integrals are log-linearly interpolated in midpoint "
        "velocity; trapezoids join native samples and use no boundary extrapolation"
    ),
    "band_energy_ratio": (
        "per shell and lane mc/cs/CMFGEN: u_FUV/u_EUV with FUV=[918,1290] A and "
        "EUV=[450,918] A, where u=(4*pi/c)*integral J_nu dnu; also record each "
        "Lumina band divided by the matching CMFGEN band; the EDDFACTOR FL column is "
        "interpreted in 10^15 Hz; trapezoids join selected native bin centers only, "
        "and zero denominators are UNDEFINED"
    ),
    "optical_depth": (
        "per shell outward tau_es=sum_(j>=shell) n_e[j]*sigma_T*(r_outer-r_inner)[j]; "
        "when an LCMFCE01 payload exists, Lumina tau_Ross is the outward sum of local "
        "Rosseland harmonic-mean chi_tot at local T_e times shell width; CMFGEN Tau(es) "
        "and Tau(Ross) are read from MEANOPAC and linearly interpolated in midpoint velocity"
    ),
    "thermalization_and_clamps": (
        "when an LCMFCE01 payload exists, per shell chi_es/chi_tot=(sum chi_es*dnu)/"
        "(sum chi_tot*dnu) and epsilon_eff=[sum(eta_fixed/chi_tot)*dnu]/"
        "[sum(eta_total/chi_tot)*dnu] over its full frequency grid; clamp firings are "
        "the exact sum of recognized run-log counters (FLOORM clamped levels and "
        "fine-solver clamped lines), reported by counter family without inferred zeros"
    ),
    "ionization_fractions": (
        "per shell, element, and 0-based stage: Lumina fraction=n_ion/sum_stages(n_ion) "
        "from lumina_ion_pops.csv; CMFGEN truth is the matching ionfrac_*_toy06_cmfgen.txt "
        "TIME=19.48 d fraction linearly interpolated in midpoint velocity; record absolute "
        "difference and Lumina/CMFGEN only for a nonzero CMFGEN denominator"
    ),
    "departure_coefficients": (
        B_K_DEFINITION + "; per (shell,Z,ion), median_bk_unweighted is the ordinary "
        "median of all finite recorded b_k values and median_bk_population_weighted is "
        "the first sorted b_k whose cumulative n_k reaches >=50% of total n_k; max/min "
        "and frac outside [0.1,10] include nonpositive recorded b_k, count_bk_le_0 is "
        "explicit, and undefined/negative weights make only the weighted median UNDEFINED"
    ),
}

EXPECTED_METRICS = tuple(METRIC_DEFINITIONS)
ELEMENTS = {"ca": 20, "s": 16, "si": 14, "fe": 26, "co": 27, "ni": 28}
LCMF_HEADER = struct.Struct("<8sIIQQQQIId")


class LedgerError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def iso_mtime(value: float) -> str:
    return dt.datetime.fromtimestamp(value, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def undef(reason: str, **extra: Any) -> dict[str, Any]:
    result = {"status": "UNDEFINED", "reason": reason}
    result.update(extra)
    return result


def metric_status(parts: Iterable[bool]) -> str:
    values = list(parts)
    if values and all(values):
        return "DEFINED"
    if any(values):
        return "PARTIAL"
    return "UNDEFINED"


def finite(value: float) -> bool:
    return math.isfinite(value)


def ratio(numerator: float, denominator: float, reason: str) -> float | dict[str, Any]:
    if not finite(numerator) or not finite(denominator):
        return undef("non-finite operand: " + reason)
    if denominator == 0.0:
        return undef("zero denominator: " + reason)
    return numerator / denominator


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def require_columns(reader: csv.DictReader, names: Iterable[str], path: Path) -> None:
    missing = sorted(set(names) - set(reader.fieldnames or []))
    if missing:
        raise LedgerError(f"{path}: missing columns {missing}")


def trapz_selected(points: list[tuple[float, float]], lo: float, hi: float) -> float:
    """Trapezoid through finite native samples selected in [lo, hi)."""
    clean = sorted((x, y) for x, y in points if finite(x) and finite(y) and lo <= x < hi)
    if len(clean) < 2:
        raise LedgerError("fewer than two finite samples")
    total = 0.0
    for (x0, y0), (x1, y1) in zip(clean, clean[1:]):
        if x1 <= x0:
            raise LedgerError("duplicate or non-increasing coordinate")
        total += 0.5 * (y0 + y1) * (x1 - x0)
    return total


def trapz_native(points: list[tuple[float, float]]) -> float:
    clean = sorted((x, y) for x, y in points if finite(x) and finite(y))
    if len(clean) < 2:
        raise LedgerError("fewer than two finite samples")
    total = 0.0
    for (x0, y0), (x1, y1) in zip(clean, clean[1:]):
        if x1 <= x0:
            raise LedgerError("duplicate or non-increasing coordinate")
        total += 0.5 * (y0 + y1) * (x1 - x0)
    return total


def parse_time_table(path: Path, target_time: float) -> tuple[list[str], list[list[float]]]:
    active = False
    header: list[str] | None = None
    rows: list[list[float]] = []
    with path.open(errors="replace") as stream:
        for line in stream:
            if line.startswith("#TIME:"):
                active = abs(float(line.split()[1]) - target_time) < 1.0e-6
                if rows and not active:
                    break
                header = None
                continue
            if not active or not line.strip():
                continue
            if line.startswith("#"):
                if "vel_mid[km/s]" in line:
                    header = line[1:].split()
                continue
            if header is None:
                continue
            try:
                row = [float(part) for part in line.split()]
            except ValueError:
                continue
            if len(row) == len(header):
                rows.append(row)
    if header is None or not rows:
        raise LedgerError(f"{path}: TIME={target_time} table not found")
    return header, rows


def linear_interp(xs: list[float], ys: list[float], x: float) -> float:
    pairs = sorted(zip(xs, ys))
    if not pairs or x < pairs[0][0] or x > pairs[-1][0]:
        raise LedgerError(f"interpolation target {x} outside [{pairs[0][0]},{pairs[-1][0]}]")
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if x == x0:
            return y0
        if x0 <= x <= x1:
            if x1 == x0:
                raise LedgerError("duplicate interpolation coordinate")
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pairs[-1][1]


def log_interp(xs: list[float], ys: list[float], x: float) -> float:
    if any(y <= 0.0 or not finite(y) for y in ys):
        raise LedgerError("log interpolation received nonpositive/nonfinite oracle value")
    return 10.0 ** linear_interp(xs, [math.log10(y) for y in ys], x)


def parse_rvtj_block(path: Path, label: str, count: int) -> list[float]:
    lines = path.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != label:
            continue
        values: list[float] = []
        for following in lines[index + 1 :]:
            if len(values) >= count:
                break
            try:
                values.extend(float(token) for token in following.split())
            except ValueError:
                break
        if len(values) >= count:
            return values[:count]
    raise LedgerError(f"{path}: RVTJ block {label!r} not found")


class CMFOracle:
    """Lazy, process-wide oracle. EDDFACTOR is loaded once for all run rows."""

    eddfactor_fl_to_hz = EDDFACTOR_FL_TO_HZ
    eddfactor_c_a_per_fl = C_A_PER_EDDFACTOR_FL

    def __init__(self, cmf_dir: Path, truth_dir: Path):
        self.cmf_dir = cmf_dir
        self.truth_dir = truth_dir
        self._temperature: tuple[list[float], list[float]] | None = None
        self._meanopac: tuple[list[float], list[float], list[float]] | None = None
        self._edd: tuple[list[float], dict[str, list[float]]] | None = None
        self._ions: dict[str, tuple[list[float], list[list[float]]]] = {}

    def temperature(self, velocity: float) -> float:
        if self._temperature is None:
            header, rows = parse_time_table(self.truth_dir / "phys_toy06_cmfgen.txt", 19.48)
            vi, ti = header.index("vel_mid[km/s]"), header.index("temp[K]")
            self._temperature = ([row[vi] for row in rows], [row[ti] for row in rows])
        return linear_interp(*self._temperature, velocity)

    def meanopac(self, velocity: float) -> tuple[float, float]:
        if self._meanopac is None:
            velocities: list[float] = []
            ross: list[float] = []
            es: list[float] = []
            with (self.cmf_dir / "MEANOPAC").open(errors="replace") as stream:
                next(stream, None)
                for line in stream:
                    parts = line.split()
                    if len(parts) < 15:
                        continue
                    try:
                        ross.append(float(parts[2]))
                        es.append(float(parts[10]))
                        velocities.append(float(parts[14]))
                    except ValueError:
                        continue
            if not velocities:
                raise LedgerError("MEANOPAC contains no parseable rows")
            self._meanopac = velocities, ross, es
        velocities, ross, es = self._meanopac
        return linear_interp(velocities, ross, velocity), linear_interp(velocities, es, velocity)

    def _load_edd(self) -> None:
        if self._edd is not None:
            return
        if np is None:
            raise LedgerError("numpy is required to read EDDFACTOR")
        info_lines = (self.cmf_dir / "EDDFACTOR_INFO").read_text().splitlines()
        if len(info_lines) < 3:
            raise LedgerError("EDDFACTOR_INFO is truncated")
        tokens = info_lines[2].split()
        nd, recl, word = int(tokens[0]), int(tokens[1]), int(tokens[2])
        little = tokens[5].upper().startswith("T")
        nword = recl // word
        dtype = "<f8" if little else ">f8"
        raw = np.fromfile(self.cmf_dir / "EDDFACTOR", dtype=dtype)
        raw = raw[: (raw.size // nword) * nword].reshape(-1, nword)
        if raw.shape[0] <= 14 or nword < nd + 1:
            raise LedgerError("EDDFACTOR dimensions do not match INFO")
        data = raw[14:, : nd + 1]
        fl = np.asarray(data[:, nd], dtype=np.float64)
        jnu = np.asarray(data[:, :nd], dtype=np.float64)
        good = np.isfinite(fl) & (fl > 0.0) & np.isfinite(jnu).all(axis=1)
        fl, jnu = fl[good], jnu[good]
        nu = fl * self.eddfactor_fl_to_hz
        if not np.isfinite(nu).all():
            raise LedgerError("EDDFACTOR FL conversion produced non-finite frequency")
        order = np.argsort(nu)
        fl, nu, jnu = fl[order], nu[order], jnu[order]
        velocities = parse_rvtj_block(self.cmf_dir / "RVTJ", "Velocity (km/s)", nd)
        if len(nu) < 2:
            raise LedgerError("EDDFACTOR has fewer than two valid data samples")
        integrals = {
            "all": [FOUR_PI_OVER_C * float(value)
                    for value in np.trapezoid(jnu, nu, axis=0)]
        }
        # Match CMFGEN's native FL convention exactly: lambda[A]=2997.92458/FL.
        wavelength = self.eddfactor_c_a_per_fl / fl
        for name, (lam_lo, lam_hi) in {"euv": EUV, "fuv": FUV}.items():
            mask = (wavelength >= lam_lo) & (wavelength <= lam_hi)
            if int(mask.sum()) < 2:
                raise LedgerError(f"EDDFACTOR has fewer than two samples in {name}")
            values = np.trapezoid(jnu[mask], nu[mask], axis=0)
            integrals[name] = [FOUR_PI_OVER_C * float(value) for value in values]
        self._edd = velocities, integrals

    def energy(self, velocity: float, band: str) -> float:
        self._load_edd()
        assert self._edd is not None
        velocities, integrals = self._edd
        return log_interp(velocities, integrals[band], velocity)

    def ion_fractions(self, tag: str, velocity: float) -> list[float]:
        if tag not in self._ions:
            path = self.truth_dir / f"ionfrac_{tag}_toy06_cmfgen.txt"
            header, rows = parse_time_table(path, 19.48)
            velocity_index = header.index("vel_mid[km/s]")
            stage_indices = list(range(1, len(header)))
            self._ions[tag] = (
                [row[velocity_index] for row in rows],
                [[row[index] for row in rows] for index in stage_indices],
            )
        velocities, stage_columns = self._ions[tag]
        return [linear_interp(velocities, column, velocity) for column in stage_columns]


def read_geometry(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader, ("shell_id", "r_inner", "r_outer", "v_inner", "v_outer"), path)
        for row in reader:
            rows.append({
                "shell": int(row["shell_id"]),
                "r_inner": float(row["r_inner"]),
                "r_outer": float(row["r_outer"]),
                "velocity_kms": 0.5 * (float(row["v_inner"]) + float(row["v_outer"])) / 1.0e5,
            })
    rows.sort(key=lambda row: int(row["shell"]))
    if [row["shell"] for row in rows] != list(range(len(rows))):
        raise LedgerError(f"{path}: shell ids are not contiguous from zero")
    return rows


def read_plasma(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader, ("shell_id", "T_e", "n_e"), path)
        for row in reader:
            shell = int(row["shell_id"])
            result[shell] = {"T_e": float(row["T_e"]), "n_e": float(row["n_e"])}
    return result


def metric_uv(run: Path) -> dict[str, Any]:
    path = run / "lumina_spectrum_formal.csv"
    if not path.is_file():
        return undef(f"missing {path.name}", definition=METRIC_DEFINITIONS["uv_quotient"])
    try:
        points: list[tuple[float, float]] = []
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            require_columns(reader, ("wavelength_angstrom", "flux"), path)
            for row in reader:
                points.append((float(row["wavelength_angstrom"]), float(row["flux"])))
        uv = trapz_selected(points, *UV_FORMAL)
        total = trapz_selected(points, *FORMAL_TOTAL)
        q = ratio(uv, total, "formal total flux")
        if isinstance(q, dict):
            return undef(q["reason"], definition=METRIC_DEFINITIONS["uv_quotient"])
        if not 0.0 <= q <= 1.0:
            return undef(
                f"self-check rejected formal UV fraction outside [0,1]: {q}",
                definition=METRIC_DEFINITIONS["uv_quotient"], source=str(path),
                uv_integral=uv, total_integral=total,
            )
        return {
            "status": "DEFINED", "definition": METRIC_DEFINITIONS["uv_quotient"],
            "source": str(path), "uv_integral": uv, "total_integral": total,
            "uv_fraction": q, "sample_count": len(points),
        }
    except (OSError, ValueError, KeyError, LedgerError) as exc:
        return undef(str(exc), definition=METRIC_DEFINITIONS["uv_quotient"], source=str(path))


def metric_temperature(
    run: Path, geometry: list[dict[str, float | int]], oracle: CMFOracle
) -> dict[str, Any]:
    path = run / "lumina_plasma_state.csv"
    if not path.is_file():
        return undef(f"missing {path.name}", definition=METRIC_DEFINITIONS["electron_temperature"])
    try:
        plasma = read_plasma(path)
        rows = []
        for geo in geometry:
            shell, velocity = int(geo["shell"]), float(geo["velocity_kms"])
            if shell not in plasma:
                rows.append({"shell": shell, "status": "UNDEFINED", "reason": "missing plasma shell"})
                continue
            lumina = plasma[shell]["T_e"]
            cmfgen = oracle.temperature(velocity)
            rows.append({
                "shell": shell, "velocity_kms": velocity, "lumina_T_e_K": lumina,
                "cmfgen_T_e_K": cmfgen, "difference_K": lumina - cmfgen,
                "ratio": ratio(lumina, cmfgen, "CMFGEN T_e"),
            })
        return {
            "status": metric_status("status" not in row for row in rows),
            "definition": METRIC_DEFINITIONS["electron_temperature"],
            "source": str(path), "rows": rows,
        }
    except (OSError, ValueError, KeyError, LedgerError) as exc:
        return undef(str(exc), definition=METRIC_DEFINITIONS["electron_temperature"], source=str(path))


def read_field(path: Path) -> dict[int, dict[str, list[tuple[float, float]]]]:
    result: dict[int, dict[str, list[tuple[float, float]]]] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader, ("shell", "wavelength_A", "cs_J", "mc_J"), path)
        for row in reader:
            shell = int(row["shell"])
            wavelength = float(row["wavelength_A"])
            if wavelength <= 0.0:
                raise LedgerError("nonpositive wavelength in coevolve field")
            nu = C_A_S / wavelength
            bucket = result.setdefault(shell, {"mc": [], "cs": []})
            bucket["mc"].append((nu, float(row["mc_J"])))
            bucket["cs"].append((nu, float(row["cs_J"])))
    return result


def field_u(points: list[tuple[float, float]], wavelength_band: tuple[float, float]) -> float:
    lam_lo, lam_hi = wavelength_band
    # Select in wavelength so the documented closed boundaries remain exact,
    # then integrate the selected native centers in increasing frequency.
    selected = [(nu, value) for nu, value in points if lam_lo <= C_A_S / nu <= lam_hi]
    value = trapz_native(selected)
    return FOUR_PI_OVER_C * value


def metric_field(
    run: Path, geometry: list[dict[str, float | int]], oracle: CMFOracle
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run / "lumina_coevolve_field.csv"
    if not path.is_file():
        reason = f"missing {path.name}"
        return (
            undef(reason, definition=METRIC_DEFINITIONS["radiation_energy_density"]),
            undef(reason, definition=METRIC_DEFINITIONS["band_energy_ratio"]),
        )
    try:
        field = read_field(path)
        energy_rows, band_rows = [], []
        for geo in geometry:
            shell, velocity = int(geo["shell"]), float(geo["velocity_kms"])
            if shell not in field:
                missing = {"shell": shell, "status": "UNDEFINED", "reason": "missing field shell"}
                energy_rows.append(missing)
                band_rows.append(copy.deepcopy(missing))
                continue
            try:
                u_mc = FOUR_PI_OVER_C * trapz_native(field[shell]["mc"])
                u_cs = FOUR_PI_OVER_C * trapz_native(field[shell]["cs"])
                u_cmf = oracle.energy(velocity, "all")
                energy_rows.append({
                    "shell": shell, "velocity_kms": velocity, "u_mc": u_mc, "u_cs": u_cs,
                    "u_cmfgen": u_cmf, "u_mc_over_cmfgen": ratio(u_mc, u_cmf, "CMFGEN u"),
                    "u_cs_over_cmfgen": ratio(u_cs, u_cmf, "CMFGEN u"),
                    "u_mc_over_u_cs": ratio(u_mc, u_cs, "u_cs"),
                })
                values: dict[str, float] = {}
                for lane in ("mc", "cs"):
                    values[f"u_{lane}_euv"] = field_u(field[shell][lane], EUV)
                    values[f"u_{lane}_fuv"] = field_u(field[shell][lane], FUV)
                values["u_cmfgen_euv"] = oracle.energy(velocity, "euv")
                values["u_cmfgen_fuv"] = oracle.energy(velocity, "fuv")
                band_rows.append({
                    "shell": shell, "velocity_kms": velocity, **values,
                    "mc_fuv_over_euv": ratio(values["u_mc_fuv"], values["u_mc_euv"], "mc EUV"),
                    "cs_fuv_over_euv": ratio(values["u_cs_fuv"], values["u_cs_euv"], "cs EUV"),
                    "cmfgen_fuv_over_euv": ratio(values["u_cmfgen_fuv"], values["u_cmfgen_euv"], "CMFGEN EUV"),
                    "mc_euv_over_cmfgen": ratio(values["u_mc_euv"], values["u_cmfgen_euv"], "CMFGEN EUV"),
                    "mc_fuv_over_cmfgen": ratio(values["u_mc_fuv"], values["u_cmfgen_fuv"], "CMFGEN FUV"),
                    "cs_euv_over_cmfgen": ratio(values["u_cs_euv"], values["u_cmfgen_euv"], "CMFGEN EUV"),
                    "cs_fuv_over_cmfgen": ratio(values["u_cs_fuv"], values["u_cmfgen_fuv"], "CMFGEN FUV"),
                })
            except LedgerError as exc:
                missing = {"shell": shell, "status": "UNDEFINED", "reason": str(exc)}
                energy_rows.append(missing)
                band_rows.append(copy.deepcopy(missing))
        return (
            {"status": metric_status("status" not in row for row in energy_rows),
             "definition": METRIC_DEFINITIONS["radiation_energy_density"], "source": str(path), "rows": energy_rows},
            {"status": metric_status("status" not in row for row in band_rows),
             "definition": METRIC_DEFINITIONS["band_energy_ratio"], "source": str(path), "rows": band_rows},
        )
    except (OSError, ValueError, KeyError, LedgerError) as exc:
        return (
            undef(str(exc), definition=METRIC_DEFINITIONS["radiation_energy_density"], source=str(path)),
            undef(str(exc), definition=METRIC_DEFINITIONS["band_energy_ratio"], source=str(path)),
        )


def load_lcmf_payload(run: Path) -> dict[str, Any] | None:
    candidates = (run / "chieta_iter10", run / "emiss_ab_iter10.A")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None
    raw = path.read_bytes()
    if len(raw) < LCMF_HEADER.size:
        raise LedgerError(f"{path}: truncated LCMFCE01 header")
    header = LCMF_HEADER.unpack_from(raw)
    magic, endian, version, nr, nnu, iteration, generation, flags, reserved, texp = header
    if (magic, endian, version, reserved) != (b"LCMFCE01", 0x01020304, 1, 0):
        raise LedgerError(f"{path}: LCMFCE01 identity mismatch")
    lengths = [nr + 1, nnu, nnu] + [nr * nnu] * 6
    arrays, offset = [], LCMF_HEADER.size
    for length in lengths:
        size = int(length) * 8
        if offset + size > len(raw):
            raise LedgerError(f"{path}: truncated LCMFCE01 array")
        arrays.append(struct.unpack_from(f"<{int(length)}d", raw, offset))
        offset += size
    if offset != len(raw):
        raise LedgerError(f"{path}: trailing LCMFCE01 bytes")
    r_edge, nu, dnu, chi_tot, chi_es, eta_fixed, eta_coherent, eta_total, _ = arrays
    if not all(nu[i] > nu[i + 1] for i in range(len(nu) - 1)):
        raise LedgerError(f"{path}: frequency is not descending")
    if not all(value > 0.0 for value in dnu):
        raise LedgerError(f"{path}: nonpositive dnu")
    for a, b, total in zip(eta_fixed, eta_coherent, eta_total):
        if struct.pack("<d", a + b) != struct.pack("<d", total):
            raise LedgerError(f"{path}: eta decomposition is not bitwise exact")
    sidecar = Path(str(path) + ".manifest.json")
    if not sidecar.is_file():
        raise LedgerError(f"{path}: missing manifest")
    manifest = json.loads(sidecar.read_text())
    digest = hashlib.sha256(raw).hexdigest()
    if manifest.get("sha256") != digest:
        raise LedgerError(f"{path}: manifest sha256 mismatch")
    return {
        "path": path, "sha256": digest, "nr": int(nr), "nnu": int(nnu),
        "iteration": int(iteration), "generation": int(generation), "texp": texp,
        "r_edge": r_edge, "nu": nu, "dnu": dnu, "chi_tot": chi_tot,
        "chi_es": chi_es, "eta_fixed": eta_fixed, "eta_total": eta_total,
    }


def dplanck_dT(nu: float, temperature: float) -> float:
    if nu <= 0.0 or temperature <= 0.0:
        raise LedgerError("nonpositive nu or T in Rosseland weight")
    x = H_CGS * nu / (K_CGS * temperature)
    if x > 700.0:
        return 0.0
    ex = math.exp(x)
    denom = ex - 1.0
    if denom == 0.0:
        raise LedgerError("zero Planck derivative denominator")
    return (2.0 * H_CGS * nu**3 / C_CGS**2) * (x * ex / (denom * denom)) / temperature


def payload_shell_metrics(
    payload: dict[str, Any], plasma: dict[int, dict[str, float]] | None
) -> dict[int, dict[str, Any]]:
    nr, nnu = payload["nr"], payload["nnu"]
    if plasma is not None and nr != len(plasma):
        raise LedgerError(f"LCMFCE01 nr={nr} differs from plasma shells={len(plasma)}")
    nu, dnu = payload["nu"], payload["dnu"]
    result: dict[int, dict[str, Any]] = {}
    for shell in range(nr):
        start, stop = shell * nnu, (shell + 1) * nnu
        ct = payload["chi_tot"][start:stop]
        ce = payload["chi_es"][start:stop]
        ef = payload["eta_fixed"][start:stop]
        et = payload["eta_total"][start:stop]
        if any(value <= 0.0 or not finite(value) for value in ct):
            raise LedgerError(f"LCMFCE01 shell {shell} has nonpositive/nonfinite chi_tot")
        chi_tot_int = math.fsum(value * width for value, width in zip(ct, dnu))
        chi_es_int = math.fsum(value * width for value, width in zip(ce, dnu))
        fixed_source = math.fsum((a / c) * width for a, c, width in zip(ef, ct, dnu))
        total_source = math.fsum((a / c) * width for a, c, width in zip(et, ct, dnu))
        result[shell] = {
            "chi_es_over_chi_tot": chi_es_int / chi_tot_int,
            "epsilon_eff": (fixed_source / total_source if total_source != 0.0
                            else undef("zero total source integral")),
        }
        if plasma is None or shell not in plasma:
            result[shell]["chi_ross"] = undef("local T_e unavailable for Rosseland weighting")
        else:
            weights = [dplanck_dT(freq, plasma[shell]["T_e"]) * width for freq, width in zip(nu, dnu)]
            numerator = math.fsum(weights)
            denominator = math.fsum(weight / opacity for weight, opacity in zip(weights, ct))
            result[shell]["chi_ross"] = (
                numerator / denominator if denominator != 0.0
                else undef("zero Rosseland denominator")
            )
    return result


def parse_clamp_counts(run: Path) -> dict[str, Any]:
    counts = {"floorm_level_firings": 0, "fine_line_source_clamp_firings": 0}
    occurrences = {key: 0 for key in counts}
    sources = []
    patterns = (
        (re.compile(r"clamped levels: deep\(s0-2\)=(\d+) mid\(s3-6\)=(\d+) phot\(s>=7\)=(\d+)"),
         "floorm_level_firings"),
        (re.compile(r"S_l deposit:.*?clamped=(\d+)/\d+ lines"), "fine_line_source_clamp_firings"),
    )
    for name in ("stdout.log", "stderr.log"):
        path = run / name
        if not path.is_file():
            continue
        sources.append(str(path))
        with path.open(errors="replace") as stream:
            for line in stream:
                for pattern, family in patterns:
                    match = pattern.search(line)
                    if not match:
                        continue
                    counts[family] += sum(int(value) for value in match.groups())
                    occurrences[family] += 1
    observed = [family for family, number in occurrences.items() if number > 0]
    if not observed:
        return undef("no recognized clamp firing counters in run logs", sources=sources)
    return {"status": "DEFINED", "counts": counts, "counter_occurrences": occurrences, "sources": sources}


def metrics_tau_and_thermalization(
    run: Path, geometry: list[dict[str, float | int]], oracle: CMFOracle
) -> tuple[dict[str, Any], dict[str, Any]]:
    plasma_path = run / "lumina_plasma_state.csv"
    clamp = parse_clamp_counts(run)
    try:
        plasma_error: str | None = None
        try:
            plasma = read_plasma(plasma_path) if plasma_path.is_file() else None
            if plasma is None:
                plasma_error = f"missing {plasma_path.name}"
        except (OSError, ValueError, KeyError, LedgerError) as exc:
            plasma, plasma_error = None, str(exc)
        payload_error: str | None = None
        try:
            payload = load_lcmf_payload(run)
            if payload:
                expected_edges = [float(geometry[0]["r_inner"])] + [float(row["r_outer"]) for row in geometry]
                if len(expected_edges) != len(payload["r_edge"]) or any(
                    abs(a - b) > 1.0e-12 * max(abs(a), abs(b), 1.0)
                    for a, b in zip(expected_edges, payload["r_edge"])
                ):
                    raise LedgerError("LCMFCE01 r_edge differs from selected model geometry")
                payload_metrics = payload_shell_metrics(payload, plasma)
            else:
                payload_metrics = None
        except (OSError, ValueError, KeyError, json.JSONDecodeError, LedgerError) as exc:
            payload, payload_metrics, payload_error = None, None, str(exc)
        tau_es: dict[int, float] | None = None
        if plasma is not None:
            tau_es = {}
            running = 0.0
            for geo in reversed(geometry):
                shell = int(geo["shell"])
                if shell not in plasma:
                    raise LedgerError(f"plasma missing shell {shell}")
                width = float(geo["r_outer"]) - float(geo["r_inner"])
                running += plasma[shell]["n_e"] * SIGMA_T * width
                tau_es[shell] = running
        tau_ross: dict[int, float] = {}
        ross_available = payload_metrics is not None and all(
            isinstance(payload_metrics[shell]["chi_ross"], float)
            for shell in payload_metrics
        )
        if ross_available:
            running = 0.0
            for geo in reversed(geometry):
                shell = int(geo["shell"])
                width = float(geo["r_outer"]) - float(geo["r_inner"])
                running += payload_metrics[shell]["chi_ross"] * width
                tau_ross[shell] = running
        tau_rows = []
        for geo in geometry:
            shell, velocity = int(geo["shell"]), float(geo["velocity_kms"])
            cmf_ross, cmf_es = oracle.meanopac(velocity)
            lumina_es: float | dict[str, Any] = (
                tau_es[shell] if tau_es is not None else undef(plasma_error or "plasma unavailable")
            )
            lumina_ross: float | dict[str, Any]
            if ross_available:
                lumina_ross = tau_ross[shell]
            else:
                reason = payload_error or "no run-local LCMFCE01 chi_tot payload; no surrogate reconstruction"
                if payload_metrics is not None:
                    chi_ross = payload_metrics[shell]["chi_ross"]
                    if isinstance(chi_ross, dict):
                        reason = chi_ross["reason"]
                lumina_ross = undef(reason)
            tau_rows.append({
                "shell": shell, "velocity_kms": velocity, "lumina_tau_es": lumina_es,
                "cmfgen_tau_es": cmf_es, "tau_es_lumina_over_cmfgen": (
                    ratio(lumina_es, cmf_es, "CMFGEN tau_es")
                    if isinstance(lumina_es, float) else undef(lumina_es["reason"])
                ),
                "lumina_tau_ross": lumina_ross, "cmfgen_tau_ross": cmf_ross,
                "tau_ross_lumina_over_cmfgen": (
                    ratio(lumina_ross, cmf_ross, "CMFGEN tau_Ross")
                    if isinstance(lumina_ross, float) else undef(lumina_ross["reason"])
                ),
            })
        tau = {
            "status": "DEFINED" if tau_es is not None and ross_available else "PARTIAL",
            "definition": METRIC_DEFINITIONS["optical_depth"], "source": str(plasma_path),
            "opacity_payload": (str(payload["path"]) if payload else undef(payload_error or "absent")),
            "rows": tau_rows,
        }
        if payload_metrics is None:
            chi_eps: dict[str, Any] = undef(payload_error or "no run-local LCMFCE01 payload")
        else:
            chi_eps = {
                "status": "DEFINED", "payload": str(payload["path"]), "payload_sha256": payload["sha256"],
                "rows": [{
                    "shell": shell,
                    "chi_es_over_chi_tot": payload_metrics[shell]["chi_es_over_chi_tot"],
                    "epsilon_eff": payload_metrics[shell]["epsilon_eff"],
                } for shell in sorted(payload_metrics)],
            }
        thermal = {
            "status": metric_status((payload_metrics is not None, clamp["status"] == "DEFINED")),
            "definition": METRIC_DEFINITIONS["thermalization_and_clamps"],
            "chi_es_over_chi_tot_and_epsilon_eff": chi_eps, "clamp_firings": clamp,
        }
        return tau, thermal
    except (OSError, ValueError, KeyError, LedgerError) as exc:
        return (
            undef(str(exc), definition=METRIC_DEFINITIONS["optical_depth"], source=str(plasma_path)),
            {"status": "PARTIAL" if clamp["status"] == "DEFINED" else "UNDEFINED",
             "definition": METRIC_DEFINITIONS["thermalization_and_clamps"],
             "chi_es_over_chi_tot_and_epsilon_eff": undef(str(exc)), "clamp_firings": clamp},
        )


def metric_ionization(
    run: Path, geometry: list[dict[str, float | int]], oracle: CMFOracle
) -> dict[str, Any]:
    path = run / "lumina_ion_pops.csv"
    if not path.is_file():
        return undef(f"missing {path.name}", definition=METRIC_DEFINITIONS["ionization_fractions"])
    try:
        populations: dict[tuple[int, int], dict[int, float]] = {}
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            require_columns(reader, ("shell_id", "Z", "stage", "n_ion"), path)
            for row in reader:
                key = (int(row["shell_id"]), int(row["Z"]))
                populations.setdefault(key, {})[int(row["stage"])] = float(row["n_ion"])
        rows, missing_elements = [], []
        geometry_by_shell = {int(row["shell"]): row for row in geometry}
        for tag, z_value in ELEMENTS.items():
            truth_path = oracle.truth_dir / f"ionfrac_{tag}_toy06_cmfgen.txt"
            if not truth_path.is_file():
                missing_elements.append({"element": tag, "Z": z_value, "reason": f"missing {truth_path}"})
                continue
            for shell in sorted(geometry_by_shell):
                stages = populations.get((shell, z_value))
                if stages is None:
                    rows.append({"shell": shell, "Z": z_value, "element": tag,
                                 "status": "UNDEFINED", "reason": "element absent from Lumina ion pops"})
                    continue
                total = math.fsum(stages.values())
                if total == 0.0:
                    rows.append({"shell": shell, "Z": z_value, "element": tag,
                                 "status": "UNDEFINED", "reason": "zero Lumina element population"})
                    continue
                velocity = float(geometry_by_shell[shell]["velocity_kms"])
                truth = oracle.ion_fractions(tag, velocity)
                for stage in sorted(set(stages) | set(range(len(truth)))):
                    if stage not in stages or stage >= len(truth):
                        rows.append({"shell": shell, "velocity_kms": velocity, "Z": z_value,
                                     "element": tag, "stage": stage, "status": "UNDEFINED",
                                     "reason": "stage absent from one side"})
                        continue
                    lumina = stages[stage] / total
                    cmfgen = truth[stage]
                    rows.append({
                        "shell": shell, "velocity_kms": velocity, "Z": z_value, "element": tag,
                        "stage": stage, "lumina_fraction": lumina, "cmfgen_fraction": cmfgen,
                        "difference": lumina - cmfgen,
                        "ratio": ratio(lumina, cmfgen, "CMFGEN ion fraction"),
                    })
        return {
            "status": metric_status((bool(rows), not missing_elements)),
            "definition": METRIC_DEFINITIONS["ionization_fractions"], "source": str(path),
            "rows": rows, "missing_truth_elements": missing_elements,
        }
    except (OSError, ValueError, KeyError, LedgerError) as exc:
        return undef(str(exc), definition=METRIC_DEFINITIONS["ionization_fractions"], source=str(path))


def ordinary_median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def population_weighted_median(values: list[float], weights: list[float]) -> float | dict[str, Any]:
    if any(not finite(weight) or weight < 0.0 for weight in weights):
        return undef("nonfinite or negative n_k weight")
    total = math.fsum(weights)
    if total <= 0.0:
        return undef("sum n_k weight is not positive")
    cumulative = 0.0
    for value, weight in sorted(zip(values, weights)):
        cumulative += weight
        if cumulative >= 0.5 * total:
            return value
    raise LedgerError("weighted median cumulative sum did not close")


def summarize_bk_group(key: tuple[int, int, int], values: list[float], weights: list[float]) -> dict[str, Any]:
    if not values or any(not finite(value) for value in values):
        raise LedgerError(f"b_k group {key} empty or nonfinite")
    outside = sum(value < 0.1 or value > 10.0 for value in values)
    return {
        "shell": key[0], "Z": key[1], "ion": key[2], "count": len(values),
        "median_bk_unweighted": ordinary_median(values),
        "median_bk_population_weighted": population_weighted_median(values, weights),
        "max_bk": max(values), "min_bk": min(values),
        "frac_bk_outside_0p1_to_10": outside / len(values),
        "count_bk_le_0": sum(value <= 0.0 for value in values),
    }


def load_census(path: Path) -> dict[tuple[int, int, int, int], tuple[str, ...]]:
    result: dict[tuple[int, int, int, int], tuple[str, ...]] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = ("shell", "Z", "ion", "level_num", "E_eV", "g", "n_k", "n_ground", "b_k")
        require_columns(reader, fields, path)
        for row in reader:
            key = tuple(int(row[name]) for name in fields[:4])
            if key in result:
                raise LedgerError(f"{path}: duplicate census key {key}")
            result[key] = tuple(row[name] for name in fields[4:])
    return result


def metric_bk(run: Path, plasma: dict[int, dict[str, float]] | None) -> dict[str, Any]:
    path = run / "lumina_levelpop.csv"
    if not path.is_file():
        return undef(f"missing {path.name}", definition=METRIC_DEFINITIONS["departure_coefficients"])
    census_path = run / "lumina_census_bk.csv"
    try:
        census = load_census(census_path) if census_path.is_file() else None
        census_seen: set[tuple[int, int, int, int]] = set()
        census_mismatches: list[dict[str, Any]] = []
        census_mismatch_count = 0
        summaries, seen_groups = [], set()
        current: tuple[int, int, int] | None = None
        values: list[float] = []
        weights: list[float] = []
        formula_checked = formula_mismatch = 0
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = ("shell", "Z", "ion", "level_num", "E_eV", "g", "n_k", "n_ground", "b_k")
            require_columns(reader, fields, path)
            ground_meta: dict[tuple[int, int, int], tuple[float, int]] = {}
            for row in reader:
                group = (int(row["shell"]), int(row["Z"]), int(row["ion"]))
                if current is not None and group != current:
                    summaries.append(summarize_bk_group(current, values, weights))
                    seen_groups.add(current)
                    if group in seen_groups:
                        raise LedgerError(f"non-contiguous repeated b_k group {group}")
                    values, weights = [], []
                current = group
                bk, nk = float(row["b_k"]), float(row["n_k"])
                values.append(bk)
                weights.append(nk)
                level = int(row["level_num"])
                energy, g_value, ng = float(row["E_eV"]), int(row["g"]), float(row["n_ground"])
                if group not in ground_meta:
                    ground_meta[group] = (energy, max(g_value, 1))
                if plasma is not None and group[0] in plasma and ng > 0.0 and nk > 0.0:
                    eg, gg = ground_meta[group]
                    boltz = (max(g_value, 1) / gg) * math.exp(-(energy - eg) / (K_B_EV * plasma[group[0]]["T_e"]))
                    if boltz > 0.0:
                        expected = (nk / ng) / boltz
                        formula_checked += 1
                        if abs(expected - bk) > 2.5e-3 * max(abs(expected), abs(bk), 1.0e-300):
                            formula_mismatch += 1
                if census is not None:
                    full_key = (group[0], group[1], group[2], level)
                    if full_key in census:
                        census_seen.add(full_key)
                        actual = tuple(row[name] for name in fields[4:])
                        expected_strings = census[full_key]
                        if actual != expected_strings:
                            census_mismatch_count += 1
                            if len(census_mismatches) < 20:
                                census_mismatches.append({"key": full_key, "levelpop": actual, "census": expected_strings})
            if current is not None:
                summaries.append(summarize_bk_group(current, values, weights))
        summaries.sort(key=lambda row: (row["shell"], row["Z"], row["ion"]))
        if census is None:
            census_result = undef("missing lumina_census_bk.csv")
        else:
            missing = len(set(census) - census_seen)
            compared = len(census_seen)
            mismatch_count = census_mismatch_count
            census_result = {
                "status": "PASS" if missing == 0 and mismatch_count == 0 else "FAIL",
                "census_path": str(census_path), "keys_expected": len(census),
                "keys_compared": compared, "missing_keys": missing,
                "value_mismatch_count": mismatch_count,
                "mismatch_examples": census_mismatches,
                "comparison": "exact CSV field strings for E_eV,g,n_k,n_ground,b_k at identical key",
            }
        return {
            "status": "DEFINED", "definition": METRIC_DEFINITIONS["departure_coefficients"],
            "b_k_definition": B_K_DEFINITION, "source": str(path), "groups": summaries,
            "levelpop_formula_audit": {
                "checked_positive_rows": formula_checked, "mismatch_count": formula_mismatch,
                "relative_tolerance": 2.5e-3,
                "note": "tolerance covers CSV rounding of E_eV and b_k; no values are altered",
            },
            "census_crosscheck": census_result,
        }
    except (OSError, ValueError, KeyError, LedgerError) as exc:
        return undef(str(exc), definition=METRIC_DEFINITIONS["departure_coefficients"], source=str(path))


def extract_log_text(run: Path) -> tuple[str, list[str]]:
    pieces, sources = [], []
    for name in ("stdout.log", "stderr.log"):
        path = run / name
        if path.is_file():
            pieces.append(path.read_text(errors="replace"))
            sources.append(str(path))
    return "\n".join(pieces), sources


def extract_gates(run: Path, log_text: str, log_sources: list[str]) -> dict[str, Any]:
    entries: dict[str, str] = {}
    source = None
    starts = [match.start() for match in re.finditer(r"=== RUN FOOTER \(env/arg as actually used\) ===", log_text)]
    if starts:
        block = log_text[starts[-1]:]
        end = block.find("=== END RUN FOOTER")
        if end >= 0:
            block = block[:end]
        for line in block.splitlines():
            match = re.match(r"\s+([A-Z][A-Z0-9_]*)=(.*)$", line)
            if match:
                entries[match.group(1)] = match.group(2)
        source = "last RUN FOOTER in " + ",".join(log_sources)
    if not entries:
        for env_path in sorted(run.glob("*.env")):
            for line in env_path.read_text(errors="replace").splitlines():
                match = re.match(r"\s*export\s+([A-Z][A-Z0-9_]*)=(.*)$", line)
                if match:
                    entries[match.group(1)] = match.group(2).strip()
            if entries:
                source = str(env_path) + " (declared; no actual footer available)"
                break
    ordered = [[key, entries[key]] for key in sorted(entries)]
    return {
        "entries": ordered, "count": len(ordered), "sha256": json_hash(ordered),
        "source": source or "none", "status": "DEFINED" if ordered else "UNDEFINED",
        **({"reason": "no RUN FOOTER or parseable .env"} if not ordered else {}),
    }


def extract_binary(run: Path, log_text: str) -> dict[str, Any]:
    explicit = re.findall(r"(?:binary_sha256|BINARY_SHA256|sha256\(binary\))\s*[=:]\s*([0-9a-fA-F]{64})", log_text)
    argv_lines = re.findall(r"^\s*argv:\s*(.+)$", log_text, re.MULTILINE)
    argv = argv_lines[-1] if argv_lines else None
    name = None
    if argv:
        try:
            name = shlex.split(argv)[0]
        except ValueError:
            name = argv.split()[0] if argv.split() else None
    if explicit:
        return {"kind": "sha256_from_log", "sha256": explicit[-1].lower(), "name": name, "argv": argv}
    if name:
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = run / candidate.name
        try:
            resolved = candidate.resolve()
            resolved.relative_to(run.resolve())
            in_run = True
        except (OSError, ValueError):
            in_run = False
        if in_run and resolved.is_file():
            return {"kind": "sha256_run_local", "sha256": sha256_file(resolved),
                    "name": name, "archived_path": str(resolved), "argv": argv}
        return {
            "kind": "name_only", "name": name, "argv": argv,
            "reason": "executable was not archived inside run directory and no authenticated sha256 appears in logs; current workspace binary is not historical evidence",
        }
    env_names = []
    for env_path in sorted(run.glob("*.env")):
        env_names += re.findall(r"^\s*export\s+LUMINA_BIN=(.+)$", env_path.read_text(errors="replace"), re.MULTILINE)
    if env_names:
        return {"kind": "name_only", "name": env_names[-1].strip("\"'"),
                "reason": "binary name found in env deck but executable and authenticated hash were not archived in run directory"}
    return {"kind": "UNDEFINED", "reason": "no argv, LUMINA_BIN, or authenticated binary hash found"}


def run_kind(path: Path) -> str:
    parts = path.resolve().parts
    if "logs" in parts:
        return "logs"
    if "scratch" in parts:
        return "scratch"
    return "UNDEFINED"


def discover_model_dir(repo: Path, run: Path, log_text: str, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    argv_lines = re.findall(r"^\s*argv:\s*(.+)$", log_text, re.MULTILINE)
    if argv_lines:
        try:
            tokens = shlex.split(argv_lines[-1])
            if len(tokens) > 1:
                candidate = Path(tokens[1])
                if not candidate.is_absolute():
                    candidate = repo / candidate
                if (candidate / "geometry.csv").is_file():
                    return candidate.resolve()
        except ValueError:
            pass
    return (repo / "data/tardis_reference_toy06_19p48d").resolve()


def input_inventory(run: Path) -> dict[str, Any]:
    names = (
        "lumina_spectrum_formal.csv", "lumina_plasma_state.csv", "lumina_coevolve_field.csv",
        "lumina_ion_pops.csv", "lumina_levelpop.csv", "lumina_census_bk.csv", "chieta_iter10",
        "stdout.log", "stderr.log",
    )
    result = {}
    for name in names:
        path = run / name
        if path.is_file():
            stat = path.stat()
            result[name] = {"status": "PRESENT", "bytes": stat.st_size, "mtime": iso_mtime(stat.st_mtime)}
        else:
            result[name] = {"status": "MISSING"}
    return result


def build_row(
    run: Path, repo: Path, oracle: CMFOracle, model_override: Path | None = None
) -> dict[str, Any]:
    run = run.resolve()
    if not run.is_dir():
        raise LedgerError(f"run directory does not exist: {run}")
    stat = run.stat()
    measured = utc_now()
    log_text, log_sources = extract_log_text(run)
    model_dir = discover_model_dir(repo, run, log_text, model_override)
    geometry_path = model_dir / "geometry.csv"
    try:
        geometry = read_geometry(geometry_path)
        geometry_error = None
    except (OSError, ValueError, KeyError, LedgerError) as exc:
        geometry, geometry_error = [], str(exc)
    metrics: dict[str, Any] = {"uv_quotient": metric_uv(run)}
    if geometry_error:
        for name in ("electron_temperature", "radiation_energy_density", "band_energy_ratio",
                     "optical_depth", "ionization_fractions"):
            metrics[name] = undef(geometry_error, definition=METRIC_DEFINITIONS[name])
        clamp = parse_clamp_counts(run)
        metrics["thermalization_and_clamps"] = {
            "status": "PARTIAL" if clamp["status"] == "DEFINED" else "UNDEFINED",
            "definition": METRIC_DEFINITIONS["thermalization_and_clamps"],
            "chi_es_over_chi_tot_and_epsilon_eff": undef(geometry_error), "clamp_firings": clamp,
        }
        plasma = None
    else:
        metrics["electron_temperature"] = metric_temperature(run, geometry, oracle)
        metrics["radiation_energy_density"], metrics["band_energy_ratio"] = metric_field(run, geometry, oracle)
        metrics["optical_depth"], metrics["thermalization_and_clamps"] = metrics_tau_and_thermalization(run, geometry, oracle)
        metrics["ionization_fractions"] = metric_ionization(run, geometry, oracle)
        try:
            plasma = read_plasma(run / "lumina_plasma_state.csv")
        except (OSError, ValueError, KeyError, LedgerError):
            plasma = None
    metrics["departure_coefficients"] = metric_bk(run, plasma)
    metrics = {name: metrics[name] for name in EXPECTED_METRICS}
    row = {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "run_path": str(run), "run_kind": run_kind(run),
        "run_directory_mtime": {"epoch_seconds": stat.st_mtime, "utc": iso_mtime(stat.st_mtime)},
        "binary_identifier": extract_binary(run, log_text),
        "gate_set": extract_gates(run, log_text, log_sources),
        "measured_at": measured,
        "metric_definitions": copy.deepcopy(METRIC_DEFINITIONS),
        "metric_definitions_sha256": json_hash(METRIC_DEFINITIONS),
        "model_geometry": str(geometry_path),
        "cmfgen_oracle": {"directory": str(oracle.cmf_dir.resolve()), "truth_directory": str(oracle.truth_dir.resolve())},
        "input_inventory": input_inventory(run),
        "metrics": metrics,
    }
    validate_row(row)
    return row


def validate_row(row: dict[str, Any]) -> None:
    required = {
        "ledger_schema_version", "run_path", "run_kind", "run_directory_mtime",
        "binary_identifier", "gate_set", "measured_at", "metric_definitions",
        "metric_definitions_sha256", "input_inventory", "metrics",
    }
    missing = sorted(required - set(row))
    if missing:
        raise LedgerError(f"row missing fields {missing}")
    if row["ledger_schema_version"] != LEDGER_SCHEMA_VERSION:
        raise LedgerError("wrong ledger schema version")
    if row["metric_definitions"] != METRIC_DEFINITIONS or row["metric_definitions_sha256"] != json_hash(METRIC_DEFINITIONS):
        raise LedgerError("metric definitions differ from frozen schema")
    if tuple(row["metrics"]) != EXPECTED_METRICS:
        raise LedgerError(f"metric order/set mismatch: {tuple(row['metrics'])}")
    if row["run_kind"] not in ("logs", "scratch", "UNDEFINED"):
        raise LedgerError("invalid run_kind")
    uv = row["metrics"]["uv_quotient"]
    if uv["status"] == "DEFINED" and not (0.0 <= uv["uv_fraction"] <= 1.0):
        raise LedgerError(f"UV fraction outside [0,1]: {uv['uv_fraction']}")
    bk = row["metrics"]["departure_coefficients"]
    if bk["status"] == "DEFINED":
        keys = set()
        for group in bk["groups"]:
            key = (group["shell"], group["Z"], group["ion"])
            if key in keys:
                raise LedgerError(f"duplicate b_k summary group {key}")
            keys.add(key)
            if not group["min_bk"] <= group["median_bk_unweighted"] <= group["max_bk"]:
                raise LedgerError(f"b_k median outside extrema for {key}")
            weighted = group["median_bk_population_weighted"]
            if isinstance(weighted, float) and not group["min_bk"] <= weighted <= group["max_bk"]:
                raise LedgerError(f"weighted b_k median outside extrema for {key}")
            if not 0.0 <= group["frac_bk_outside_0p1_to_10"] <= 1.0:
                raise LedgerError(f"b_k outside fraction invalid for {key}")
            if not 0 <= group["count_bk_le_0"] <= group["count"]:
                raise LedgerError(f"b_k nonpositive count invalid for {key}")
    try:
        json.dumps(row, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"row is not strict JSON: {exc}") from exc


def prior_measurement_count(ledger: Path, run_path: str) -> int:
    if not ledger.is_file():
        return 0
    count = 0
    with ledger.open(errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise LedgerError(f"ledger line {line_number} lacks newline; refusing append")
            try:
                old = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"ledger line {line_number} invalid: {exc}") from exc
            if old.get("run_path") == run_path:
                count += 1
    return count


def append_row(ledger: Path, row: dict[str, Any]) -> dict[str, Any]:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(ledger, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        count = prior_measurement_count(ledger, row["run_path"])
        if count:
            row["recomputed_at"] = row["measured_at"]
            row["prior_measurement_count"] = count
        validate_row(row)
        encoded = (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False) + "\n").encode()
        view = memoryview(encoded)
        written = 0
        while written < len(encoded):
            count_now = os.write(descriptor, view[written:])
            if count_now <= 0:
                raise LedgerError(f"short append stopped at {written}/{len(encoded)} bytes")
            written += count_now
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return row


def write_fixture(repo: Path) -> tuple[Path, Path, Path, Path]:
    run = repo / "logs/fixture_run"
    model = repo / "data/model"
    truth = repo / "data/standart_data1/toy06"
    cmf = repo / "cmf"
    for path in (run, model, truth, cmf):
        path.mkdir(parents=True, exist_ok=True)
    (model / "geometry.csv").write_text(
        "shell_id,r_inner,r_outer,v_inner,v_outer\n"
        "0,1.0e14,2.0e14,4.0e8,5.0e8\n1,2.0e14,3.0e14,5.0e8,6.0e8\n"
    )
    (run / "lumina_spectrum_formal.csv").write_text(
        "wavelength_angstrom,flux\n500,1\n1000,2\n3500,2\n10000,1\n20000,1\n"
    )
    (run / "lumina_plasma_state.csv").write_text(
        "shell_id,W,T_rad,n_e,T_e\n0,0.3,10000,1e9,12000\n1,0.2,9000,5e8,10000\n"
    )
    wavelengths = [400.0, 450.0, 600.0, 700.0, 918.0, 1000.0, 1100.0, 1200.0,
                   1290.0, 3500.0, 20000.0, 25000.0]
    with (run / "lumina_coevolve_field.csv").open("w") as stream:
        stream.write("shell,bin,wavelength_A,cs_J,mc_J\n")
        for shell in range(2):
            for index, wavelength in enumerate(wavelengths):
                stream.write(f"{shell},{index},{wavelength},{2+shell},{1+shell}\n")
    (run / "lumina_ion_pops.csv").write_text(
        "shell_id,Z,stage,n_ion\n0,26,0,1\n0,26,1,3\n1,26,0,2\n1,26,1,2\n"
    )
    (run / "lumina_levelpop.csv").write_text(
        "shell,Z,ion,level_num,E_eV,g,n_k,n_ground,b_k,has_sigma,n_sig_pos\n"
        "0,26,2,0,0.0000,2,4.000000e+00,4.000000e+00,1.0000e+00,1,2\n"
        "0,26,2,1,1.0000,4,8.000000e+00,4.000000e+00,1.0000e+00,1,2\n"
        "0,26,2,2,2.0000,2,0.000000e+00,4.000000e+00,-1.0000e+00,0,0\n"
    )
    (run / "lumina_census_bk.csv").write_text(
        "shell,Z,ion,level_num,E_eV,g,n_k,n_ground,b_k\n"
        "0,26,2,0,0.0000,2,4.000000e+00,4.000000e+00,1.0000e+00\n"
        "0,26,2,1,1.0000,4,8.000000e+00,4.000000e+00,1.0000e+00\n"
        "0,26,2,2,2.0000,2,0.000000e+00,4.000000e+00,-1.0000e+00\n"
    )
    binary = run / "fixture_binary"
    binary.write_bytes(b"fixture executable bytes\n")
    (run / "stdout.log").write_text(
        "=== RUN FOOTER (env/arg as actually used) ===\n"
        "  LUMINA_ALPHA=1\n  LUMINA_BETA=0\n"
        f"  argv: {binary} data/model 1 1 spectrum nlte\n"
        "=== END RUN FOOTER (2 vars) ===\n"
        "[FLOORM] iter 1 clamped levels: deep(s0-2)=1 mid(s3-6)=2 phot(s>=7)=3\n"
    )
    (run / "stderr.log").write_text(
        "[cmf_fine] S_l deposit: max S_l/B=1 clamped=4/10 lines (sl_clamp=2) skipped weak(tau<1)=0\n"
    )
    phys_header = (
        "#NTIMES: 1\n#TIMES[d]: 19.480\n#TIME: 19.480\n#NVEL: 3\n"
        "#vel_mid[km/s] temp[K] rho[gcc] ne[/cm^3] natom[/cm^3]\n"
    )
    (truth / "phys_toy06_cmfgen.txt").write_text(
        phys_header + "4000 11000 1e-14 1e9 1e9\n5000 10000 1e-14 1e9 1e9\n6000 9000 1e-14 1e9 1e9\n"
    )
    ion_header = (
        "#NTIMES: 1\n#NSTAGES: 2\n#TIMES[d]: 19.480\n#TIME: 19.480\n#NVEL: 3\n"
        "#vel_mid[km/s] fe0 fe1\n"
        "4000 0.25 0.75\n5000 0.5 0.5\n6000 0.75 0.25\n"
    )
    (truth / "ionfrac_fe_toy06_cmfgen.txt").write_text(ion_header)
    (cmf / "MEANOPAC").write_text(
        "R I Tau(Ross) dTau RatR ChiR ChiR2 ChiF ChiES TauF TauES RatF RatES KappaR V(km/s)\n"
        "1 1 1.0 0 0 0 0 0 0 0 0.5 0 0 0 4000\n"
        "1 2 0.5 0 0 0 0 0 0 0 0.25 0 0 0 5000\n"
        "1 3 0.1 0 0 0 0 0 0 0 0.05 0 0 0 6000\n"
    )
    nd, nword = 3, 4
    (cmf / "EDDFACTOR_INFO").write_text(
        "fixture\nfixture\n          3          32           8           1           4           T\n"
    )
    velocities = [4000.0, 5000.0, 6000.0]
    rvtj = "Velocity (km/s)\n " + " ".join(str(value) for value in velocities) + "\n"
    (cmf / "RVTJ").write_text(rvtj)
    records = [[0.0] * nword for _ in range(14)]
    for wavelength in (25000.0, 20000.0, 3500.0, 1290.0, 1200.0, 1100.0, 1000.0,
                       918.0, 700.0, 600.0, 450.0, 400.0):
        records.append([1.0, 2.0, 3.0, C_A_S / (EDDFACTOR_FL_TO_HZ * wavelength)])
    with (cmf / "EDDFACTOR").open("wb") as stream:
        for record in records:
            stream.write(struct.pack("<4d", *record))

    payload_path = run / "chieta_iter10"
    payload_header = (b"LCMFCE01", 0x01020304, 1, 2, 4, 10, 10, 7, 0, 1.0e6)
    payload_arrays = [
        (1.0e14, 2.0e14, 3.0e14),
        (4.0e15, 3.0e15, 2.0e15, 1.0e15),
        (1.0e15, 1.0e15, 1.0e15, 1.0e15),
        tuple([2.0e-14] * 8),
        tuple([1.0e-14] * 8),
        tuple([2.0e-10] * 8),
        tuple([3.0e-10] * 8),
        tuple([5.0e-10] * 8),
        tuple([1.0e-4] * 8),
    ]
    payload_raw = LCMF_HEADER.pack(*payload_header) + b"".join(
        struct.pack(f"<{len(array)}d", *array) for array in payload_arrays
    )
    payload_path.write_bytes(payload_raw)
    (run / "chieta_iter10.manifest.json").write_text(json.dumps({
        "schema": "LCMFCE01-v1", "sha256": hashlib.sha256(payload_raw).hexdigest(),
        "iteration": 10, "field_generation": 10, "post_damping": True,
        "coherent_frozen": True, "frequency_descending": True,
        "eta_decomposition_bitwise": True, "eta_decomposition_max_abs": 0,
    }) + "\n")
    return run, model, truth, cmf


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="regression_ledger_fixture_") as temporary:
        repo = Path(temporary)
        run, model, truth, cmf = write_fixture(repo)
        oracle = CMFOracle(cmf, truth)
        row = build_row(run, repo, oracle, model)
        validate_row(row)
        energy = row["metrics"]["radiation_energy_density"]
        bands = row["metrics"]["band_energy_ratio"]
        if energy["status"] != "DEFINED" or bands["status"] != "DEFINED":
            raise LedgerError("fixture EDDFACTOR metrics did not define")
        fixture_u_cmfgen = energy["rows"][0]["u_cmfgen"]
        fixture_base_u = FOUR_PI_OVER_C * (
            C_A_S / 400.0 - C_A_S / 25000.0
        )
        fixture_expected = math.sqrt(2.0) * fixture_base_u
        if not math.isclose(fixture_u_cmfgen, fixture_expected, rel_tol=2.0e-15):
            raise LedgerError(
                f"fixture EDDFACTOR FL-unit gate mismatch: {fixture_u_cmfgen} != {fixture_expected}"
            )
        if row["metrics"]["thermalization_and_clamps"]["chi_es_over_chi_tot_and_epsilon_eff"]["status"] != "DEFINED":
            raise LedgerError("fixture LCMFCE01 thermalization path did not define")
        if row["metrics"]["optical_depth"]["status"] != "DEFINED":
            raise LedgerError("fixture LCMFCE01 tau_Ross path did not define")
        if row["metrics"]["departure_coefficients"]["census_crosscheck"]["status"] != "PASS":
            raise LedgerError("fixture census crosscheck did not pass")
        bk_probe = summarize_bk_group((0, 26, 2), [1.0, 2.0, 100.0], [100.0, 1.0, 1.0])
        if bk_probe["median_bk_unweighted"] != 2.0 or \
           bk_probe["median_bk_population_weighted"] != 1.0:
            raise LedgerError("b_k dual-median weighting fixture failed")
        ledger = repo / "validation/regression_ledger/ledger.jsonl"
        append_row(ledger, copy.deepcopy(row))
        prefix = ledger.read_bytes()
        second = copy.deepcopy(row)
        second["measured_at"] = utc_now()
        append_row(ledger, second)
        after = ledger.read_bytes()
        if not after.startswith(prefix) or after == prefix:
            raise LedgerError("append-only prefix invariant failed")
        lines = after.splitlines()
        if len(lines) != 2 or "recomputed_at" not in json.loads(lines[1]):
            raise LedgerError("recomputed_at invariant failed")
        injected = copy.deepcopy(row)
        injected["metrics"]["uv_quotient"]["uv_fraction"] = 1.5
        try:
            validate_row(injected)
        except LedgerError as exc:
            print(f"NEGATIVE CONTROL: FAIL (expected): injected uv_fraction=1.5 -> {exc}")
        else:
            raise LedgerError("negative control escaped self-check")
        broken_oracle = CMFOracle(cmf, truth)
        broken_oracle.eddfactor_fl_to_hz = 1.0
        broken_oracle.eddfactor_c_a_per_fl = C_A_S
        broken_row = build_row(run, repo, broken_oracle, model)
        try:
            broken_energy = broken_row["metrics"]["radiation_energy_density"]
            broken_bands = broken_row["metrics"]["band_energy_ratio"]
            if broken_energy["status"] != "DEFINED" or broken_bands["status"] != "DEFINED":
                reason = broken_energy.get("reason") or broken_bands.get("reason")
                if reason is None:
                    for metric in (broken_energy, broken_bands):
                        for candidate in metric.get("rows", []):
                            if "reason" in candidate:
                                reason = candidate["reason"]
                                break
                        if reason is not None:
                            break
                raise LedgerError(f"fixture EDDFACTOR metrics did not define: {reason}")
            broken_u = broken_energy["rows"][0]["u_cmfgen"]
            if not math.isclose(broken_u, fixture_expected, rel_tol=2.0e-15):
                raise LedgerError(
                    f"fixture EDDFACTOR FL-unit gate mismatch: {broken_u} != {fixture_expected}"
                )
        except LedgerError as exc:
            print(f"NEGATIVE CONTROL EDDFACTOR: FAIL (expected): FL treated as Hz -> {exc}")
        else:
            raise LedgerError("EDDFACTOR FL-unit negative control escaped self-check")
        print("PASS fixture metrics: all 8 metric objects present and strict-JSON valid")
        print("PASS fixture EDDFACTOR: FL decoded as 10^15 Hz; energy and FUV/EUV metrics defined")
        print("PASS b_k dual weighting: ordinary and n_k-weighted medians remain distinct")
        print("PASS fixture census: levelpop and census paths agree exactly")
        print("PASS append-only: first JSONL prefix preserved; recomputed_at added on second measurement")
        empty_run = repo / "logs/empty_run"
        empty_run.mkdir()
        empty_row = build_row(empty_run, repo, oracle, model)
        if tuple(empty_row["metrics"]) != EXPECTED_METRICS:
            raise LedgerError("empty-run fixture dropped a metric")
        if empty_row["metrics"]["uv_quotient"]["status"] != "UNDEFINED" or \
           empty_row["metrics"]["departure_coefficients"]["status"] != "UNDEFINED":
            raise LedgerError("empty-run fixture invented a missing run-side value")
        print("PASS missing-input fixture: row retained; unavailable run-side values are UNDEFINED")
        payload_only = repo / "scratch/payload_only"
        payload_only.mkdir(parents=True)
        for suffix in ("", ".manifest.json"):
            source = Path(str(run / "chieta_iter10") + suffix)
            Path(str(payload_only / "chieta_iter10") + suffix).write_bytes(source.read_bytes())
        payload_row = build_row(payload_only, repo, oracle, model)
        chi_eps = payload_row["metrics"]["thermalization_and_clamps"]["chi_es_over_chi_tot_and_epsilon_eff"]
        if chi_eps["status"] != "DEFINED" or payload_row["metrics"]["optical_depth"]["status"] != "PARTIAL":
            raise LedgerError("payload-only fixture did not separate chi/epsilon from T_e-dependent Rosseland path")
        print("PASS payload-only fixture: chi_es/chi_tot and epsilon_eff defined without plasma T_e")
        print("PASS --self-test")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path, help="one or more archived run directories")
    parser.add_argument("--ledger", type=Path,
                        default=Path("validation/regression_ledger/ledger.jsonl"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cmfgen-dir", type=Path, default=DEFAULT_CMFGEN)
    parser.add_argument("--truth-dir", type=Path,
                        default=Path("data/standart_data1/toy06"))
    parser.add_argument("--model-dir", type=Path,
                        help="override model geometry for every input run (primarily fixtures)")
    parser.add_argument("--no-append", action="store_true", help="print rows without changing ledger")
    parser.add_argument("--self-test", action="store_true", help="run generated-fixture tests only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        if args.run_dirs:
            print("FAIL: --self-test accepts no run directories", file=sys.stderr)
            return 2
        try:
            return self_test()
        except (OSError, ValueError, KeyError, LedgerError, json.JSONDecodeError) as exc:
            print(f"FAIL --self-test: {exc}", file=sys.stderr)
            return 1
    if not args.run_dirs:
        print("FAIL: at least one RUN_DIR is required", file=sys.stderr)
        return 2
    if np is None:
        print("FAIL: numpy is required for EDDFACTOR processing", file=sys.stderr)
        return 2
    repo = args.repo_root.resolve()
    truth = args.truth_dir if args.truth_dir.is_absolute() else repo / args.truth_dir
    ledger = args.ledger if args.ledger.is_absolute() else repo / args.ledger
    oracle = CMFOracle(args.cmfgen_dir.resolve(), truth.resolve())
    failures = 0
    for requested in args.run_dirs:
        run = requested if requested.is_absolute() else repo / requested
        try:
            row = build_row(run, repo, oracle, args.model_dir)
            if args.no_append:
                print(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False))
            else:
                append_row(ledger, row)
                print(f"APPENDED {row['run_path']} -> {ledger}", flush=True)
            census = row["metrics"]["departure_coefficients"].get("census_crosscheck", {})
            if census.get("status") == "FAIL":
                failures += 1
                print(
                    f"FAIL CENSUS {run}: missing={census.get('missing_keys')} "
                    f"mismatches={census.get('value_mismatch_count')}; row was retained",
                    file=sys.stderr, flush=True,
                )
        except (OSError, ValueError, KeyError, LedgerError, json.JSONDecodeError) as exc:
            failures += 1
            print(f"FAIL {run}: {exc}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
