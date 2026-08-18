#!/usr/bin/env python3
"""Parameterized copy of the 2026-07-19 coupled-root estimator.

This preserves the historical estimator's physics and numerical conventions,
but takes every run/model/reference path from the command line.  It also
evaluates the nested R/J/O operator cube used for the lever-additivity audit:

R: replace the committed temperature by a coupled root;
J: select CMFGEN J instead of the run's own cs.J when R is active;
O: add the historical CMFGEN-root-to-truth residual correction.

J has no effect while R is off: changing the field cannot alter a temperature
already committed in a CSV without invoking the root evaluator.  That nesting
is intentional and is the R x J interaction measured by this audit.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd

H = 6.62607015e-27
KB = 1.380649e-16
C = 2.99792458e10
EV = 1.602176634e-12
T_EXP_DEFAULT = 19.48 * 86400.0
SOB = 2.6540281e-2
NU_MIN = 1.5e14
NU_MAX = 3.0e16
H_PHOTO_HISTORICAL = 7.2e-7


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--deposition-file", type=Path, required=True)
    parser.add_argument("--cmfgen-jtable", type=Path, required=True)
    parser.add_argument("--cmfgen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--shell", type=int, default=0)
    parser.add_argument("--truth-temperature-k", type=float, default=18760.0)
    parser.add_argument("--time-days", type=float, default=19.48)
    parser.add_argument("--photo-heating", type=float, default=H_PHOTO_HISTORICAL)
    parser.add_argument("--baseline-gate", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def read_rvtj_block(path: Path, label: str, count: int) -> np.ndarray:
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip() != label:
            continue
        values: list[float] = []
        for candidate in lines[index + 1 :]:
            try:
                values.extend(float(token) for token in candidate.split())
            except ValueError:
                break
            if len(values) >= count:
                break
        if len(values) != count:
            raise ValueError(f"{path}: {label!r} has {len(values)} values, expected {count}")
        return np.asarray(values)
    raise KeyError(f"{path}: no block {label!r}")


def read_cmfgen_truth(cmfgen_dir: Path, target_velocity: float) -> float:
    info = (cmfgen_dir / "EDDFACTOR_INFO").read_text().splitlines()
    nd = int(info[2].split()[0])
    rvtj = cmfgen_dir / "RVTJ"
    velocity = read_rvtj_block(rvtj, "Velocity (km/s)", nd)
    temperature = 1.0e4 * read_rvtj_block(rvtj, "Temperature (10^4K)", nd)
    order = np.argsort(velocity)
    if not velocity[order][0] <= target_velocity <= velocity[order][-1]:
        raise ValueError(f"target velocity {target_velocity} is outside CMFGEN RVTJ")
    return float(np.interp(target_velocity, velocity[order], temperature[order]))


class CoupledRootEstimator:
    """Historical coupled-root estimator with path/state literals removed."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.shell = args.shell
        self.t_exp = args.time_days * 86400.0
        self.h_photo = args.photo_heating

        required = [
            args.input_dir / "lumina_plasma_state.csv",
            args.input_dir / "lumina_coevolve_field.csv",
            args.input_dir / "lumina_ion_pops.csv",
            args.input_dir / "lumina_levelpop.csv",
            args.model_dir / "line_list.csv",
            args.model_dir / "levels.csv",
            args.model_dir / "ionization_energies.csv",
            args.deposition_file,
            args.cmfgen_jtable,
            args.cmfgen_dir / "EDDFACTOR",
            args.cmfgen_dir / "EDDFACTOR_INFO",
            args.cmfgen_dir / "RVTJ",
            args.cmfgen_dir / "MEANOPAC",
        ]
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            raise FileNotFoundError("required inputs absent: " + ", ".join(absent))

        plasma = pd.read_csv(args.input_dir / "lumina_plasma_state.csv")
        p0 = plasma[plasma["shell_id"] == self.shell]
        if len(p0) != 1:
            raise ValueError(f"expected one plasma row for shell {self.shell}, found {len(p0)}")
        self.te0 = float(p0.iloc[0]["T_e"])
        self.ne = float(p0.iloc[0]["n_e"])

        dep = pd.read_csv(args.deposition_file)
        d0 = dep[dep["shell_id"] == self.shell]
        if len(d0) != 1:
            raise ValueError(f"expected one deposition row for shell {self.shell}, found {len(d0)}")
        self.h_dep = float(d0.iloc[0]["heating_rate"])

        field = pd.read_csv(args.input_dir / "lumina_coevolve_field.csv")
        f0 = field[field["shell"] == self.shell].sort_values("bin")
        bins = f0["bin"].to_numpy(dtype=int)
        if not np.array_equal(bins, np.arange(len(bins))):
            raise ValueError("field bins are missing or reordered")
        self.nfb = len(bins)
        self.dln = math.log(NU_MAX / NU_MIN) / self.nfb
        self.cs_j_bin = f0["cs_J"].to_numpy(dtype=float)
        self.mc_j_bin = f0["mc_J"].to_numpy(dtype=float)
        if np.any(~np.isfinite(self.cs_j_bin)) or np.any(self.cs_j_bin < 0):
            raise ValueError("cs_J contains non-finite or negative values")
        if np.any(~np.isfinite(self.mc_j_bin)) or np.any(self.mc_j_bin < 0):
            raise ValueError("mc_J contains non-finite or negative values")

        self.pops: dict[tuple[int, int], float] = {}
        with (args.input_dir / "lumina_ion_pops.csv").open() as handle:
            for row in csv.DictReader(handle):
                if int(row["shell_id"]) == self.shell:
                    self.pops[(int(row["Z"]), int(row["stage"]))] = float(row["n_ion"])
        elements = sorted({z for z, _ in self.pops})
        self.frac0: dict[int, np.ndarray] = {}
        self.nel: dict[int, float] = {}
        for zc in elements:
            stages = np.asarray([self.pops.get((zc, stage), 0.0) for stage in range(8)])
            total = float(stages.sum())
            if total > 0:
                self.nel[zc] = total
                self.frac0[zc] = stages / total
        if not self.nel:
            raise ValueError("no positive shell ion populations")

        self.chi: dict[tuple[int, int], float] = {}
        with (args.model_dir / "ionization_energies.csv").open() as handle:
            for row in csv.DictReader(handle):
                self.chi[(int(row["atomic_number"]), int(row["ion_number"]))] = float(
                    row["ionization_energy_eV"]
                )

        self._load_levels_and_lines()
        self.jb_cs = self.jbin(self.nu, self.cs_j_bin)
        self.jb_cmf = self._load_jtable()
        self._cache: dict[tuple[str, float], tuple[float, float, float]] = {}

    def _load_levels_and_lines(self) -> None:
        temporary: dict[tuple[int, int], dict[int, tuple[float, float]]] = {}
        with (self.args.model_dir / "levels.csv").open() as handle:
            for row in csv.DictReader(handle):
                key = (int(row["atomic_number"]), int(row["ion_number"]))
                temporary.setdefault(key, {})[int(row["level_number"])] = (
                    float(row["energy_eV"]), float(row["g"])
                )
        self.level_arrays: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for key, values in temporary.items():
            nlevel = max(values) + 1
            energies = np.zeros(nlevel)
            weights = np.ones(nlevel)
            for level, (energy, weight) in values.items():
                energies[level] = energy
                weights[level] = weight
            self.level_arrays[key] = energies, weights

        lines = pd.read_csv(
            self.args.model_dir / "line_list.csv",
            usecols=[
                "atomic_number", "ion_number", "level_number_lower",
                "level_number_upper", "f_lu", "A_ul", "nu",
            ],
        )
        z = lines["atomic_number"].to_numpy()
        ion = lines["ion_number"].to_numpy()
        lower = lines["level_number_lower"].to_numpy()
        upper = lines["level_number_upper"].to_numpy()
        oscillator = lines["f_lu"].to_numpy()
        a_ul = lines["A_ul"].to_numpy()
        nu = lines["nu"].to_numpy(dtype=float)
        count = len(z)
        e_lower = np.zeros(count)
        g_lower = np.ones(count)
        g_upper = np.ones(count)
        valid = np.zeros(count, dtype=bool)
        for key in set(zip(z, ion)):
            if key not in self.level_arrays:
                continue
            energies, weights = self.level_arrays[key]
            mask = (z == key[0]) & (ion == key[1])
            lo = lower[mask]
            up = upper[mask]
            okay = (lo < len(energies)) & (up < len(energies))
            index = np.where(mask)[0][okay]
            e_lower[index] = energies[lo[okay]]
            g_lower[index] = weights[lo[okay]]
            g_upper[index] = weights[up[okay]]
            valid[index] = True
        keep = valid & (nu > 0) & np.isin(z, list(self.nel))
        self.z = z[keep]
        self.ion = ion[keep]
        self.f_lu = oscillator[keep]
        self.a_ul = a_ul[keep]
        self.nu = nu[keep]
        self.e_lower = e_lower[keep]
        self.g_lower = g_lower[keep]
        self.g_upper = g_upper[keep]
        self.de = H * self.nu
        self.line_count = len(self.nu)
        if self.line_count == 0:
            raise ValueError("no lines survived the historical estimator selection")

        # Historical VR_STD definition.  The collision-strength minimum is an
        # inherited model term, not a repair/fallback introduced by this copy.
        ry_de = np.minimum(13.605693 / (self.de / EV), 136.0)
        vr = 8.63e-6 * 14.5 * 0.2 * self.f_lu * self.g_lower * ry_de
        omega_minimum = 8.63e-6
        self.coeff = np.where(
            self.f_lu > 1e-10, np.maximum(vr, omega_minimum), omega_minimum
        )
        self.ftau = SOB * self.f_lu * (C / self.nu) * self.t_exp
        self.bul = (C * C / (2 * H * self.nu**3)) * self.a_ul
        self.blu = self.bul * (self.g_upper / self.g_lower)
        self.groups: dict[tuple[int, int], np.ndarray] = {}
        for key in set(zip(self.z.tolist(), self.ion.tolist())):
            self.groups[key] = np.where((self.z == key[0]) & (self.ion == key[1]))[0]

    def _load_jtable(self) -> np.ndarray:
        with self.args.cmfgen_jtable.open("rb") as handle:
            magic, version, nshell, nfb = struct.unpack("4i", handle.read(16))
            raw = np.frombuffer(handle.read(), np.float64)
        if magic != 1247035714 or version != 1:
            raise ValueError(f"unexpected CMFGEN jtable header magic={magic} version={version}")
        if nfb != self.nfb or self.shell >= nshell or raw.size != nshell * nfb:
            raise ValueError("CMFGEN jtable dimensions do not match the selected field/shell")
        table = raw.reshape(nshell, nfb)[self.shell]
        positive = table > 0
        if positive.sum() < 2:
            raise ValueError("CMFGEN jtable has fewer than two positive bins")
        grid = np.asarray([NU_MIN * math.exp((b + 0.5) * self.dln) for b in range(nfb)])
        return np.interp(self.nu, grid[positive], table[positive])

    def jbin(self, frequencies: np.ndarray, values: np.ndarray) -> np.ndarray:
        index = np.floor(np.log(frequencies / NU_MIN) / self.dln).astype(int)
        in_grid = (frequencies > NU_MIN) & (frequencies < NU_MAX)
        output = np.full(len(frequencies), 1e-30)
        output[in_grid] = values[index[in_grid]]
        return output

    def fractions_at(self, zc: int, temperature: float) -> np.ndarray:
        base = self.frac0[zc]
        scale = (temperature / self.te0) ** 0.8
        result = np.zeros(8)
        result[0] = 1.0
        for stage in range(7):
            ratio = base[stage + 1] / base[stage] if base[stage] > 0 else 0.0
            result[stage + 1] = result[stage] * ratio * scale
        total = float(result.sum())
        if not total > 0:
            raise ValueError(f"undefined ion ladder for Z={zc} at T={temperature}")
        return result / total

    def populations_at(self, temperature: float) -> dict[tuple[int, int], float]:
        return {
            (zc, stage): self.nel[zc] * self.fractions_at(zc, temperature)[stage]
            for zc in self.nel for stage in range(8)
        }

    def partition(self, key: tuple[int, int], temperature: float) -> float:
        energies, weights = self.level_arrays[key]
        exponent = energies * EV / (KB * temperature)
        # exp(-positive) is safe without an exponent cap.
        value = float(np.sum(weights * np.exp(-exponent)))
        if not value > 0:
            raise ValueError(f"undefined partition function for {key} at T={temperature}")
        return value

    @staticmethod
    def beta_escape(tau: np.ndarray) -> np.ndarray:
        output = np.empty_like(tau)
        thin = tau <= 1e-6
        output[thin] = 1.0
        output[~thin] = -np.expm1(-tau[~thin]) / tau[~thin]
        return output

    @staticmethod
    def alpha_rr(zrec: int, temperature: float) -> float:
        return 2.6e-13 * zrec**1.6 * (temperature / 1e4) ** -0.8

    def residual(self, temperature: float, field_name: str) -> tuple[float, float, float]:
        cache_key = (field_name, float(temperature))
        if cache_key in self._cache:
            return self._cache[cache_key]
        jb = self.jb_cs if field_name == "own_cs" else self.jb_cmf
        populations = self.populations_at(temperature)
        nion = np.zeros(self.line_count)
        partition = np.ones(self.line_count)
        for key, index in self.groups.items():
            nion[index] = populations.get(key, 0.0)
            partition[index] = self.partition(key, temperature)
        excitation = self.e_lower * EV / (KB * temperature)
        nlower = nion * self.g_lower * np.exp(-excitation) / partition
        invsqrt = 1.0 / math.sqrt(temperature)
        qlu = self.coeff / self.g_lower * invsqrt * np.exp(-self.de / (KB * temperature))
        qul = self.coeff / self.g_upper * invsqrt
        clu = self.ne * qlu
        cul = self.ne * qul
        tau = self.ftau * nlower
        beta = self.beta_escape(tau)
        rul = (self.a_ul + self.bul * jb) * beta
        rlu = self.blu * jb * beta
        denominator = cul + rul
        if np.any(denominator <= 0) or np.any(~np.isfinite(denominator)):
            raise ValueError(f"undefined line denominator at T={temperature}")
        nupper = nlower * (clu + rlu) / denominator
        line = float(np.sum(self.de * (nlower * qlu * self.ne - nupper * qul * self.ne)))

        fb = 0.0
        for (zc, stage), _ in populations.items():
            next_pop = populations.get((zc, stage + 1), 0.0)
            if next_pop <= 0 or (zc, stage) not in self.chi:
                continue
            fb += (
                self.ne * next_pop * self.alpha_rr(stage + 1, temperature)
                * (self.chi[(zc, stage)] * EV + KB * temperature)
            )
        ff = 1.426e-27 * 1.2 * self.ne * self.ne * math.sqrt(temperature)
        ad = 1.5 * self.ne * KB * temperature * (3.0 / self.t_exp)
        value = (self.h_dep + self.h_photo) - (ff + ad + fb + line)
        answer = value, line, fb
        self._cache[cache_key] = answer
        return answer

    def lowest_root(self, field_name: str) -> float:
        temperatures = np.geomspace(3500.0, 140000.0, 25)
        previous: float | None = None
        lower: float | None = None
        for temperature in temperatures:
            value = self.residual(float(temperature), field_name)[0]
            if previous is not None and previous > 0 and value <= 0:
                assert lower is not None
                lo, hi = lower, float(temperature)
                for _ in range(40):
                    middle = 0.5 * (lo + hi)
                    trial = self.residual(middle, field_name)[0]
                    if trial > 0:
                        lo = middle
                    else:
                        hi = middle
                return 0.5 * (lo + hi)
            previous = value
            lower = float(temperature)
        low_value = self.residual(3500.0, field_name)[0]
        if low_value <= 0:
            raise RuntimeError(f"{field_name}: lowest-root undefined (pin_lo; r(3500)<=0)")
        raise RuntimeError(f"{field_name}: lowest-root undefined (no + to - crossing through 140000 K)")


def endpoint(committed: float, own: float, cmf: float, truth: float,
             r_on: int, j_on: int, o_on: int) -> float:
    base = committed if not r_on else (cmf if j_on else own)
    residual_correction = truth - cmf
    return base + o_on * residual_correction


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimator = CoupledRootEstimator(args)
    own = estimator.lowest_root("own_cs")
    cmf = estimator.lowest_root("cmfgen")
    committed = estimator.te0
    truth = args.truth_temperature_k
    target_velocity = 4264.0 + 728.0 * args.shell
    rvtj_truth = read_cmfgen_truth(args.cmfgen_dir, target_velocity)

    rounded = {
        "committed": round(committed), "own": round(own), "cmf": round(cmf), "truth": round(truth)
    }
    historical_levers = {
        "R_committed_to_own": own - committed,
        "J_own_to_cmf": cmf - own,
        "O_cmf_to_truth": truth - cmf,
        "whole_truth_minus_committed": truth - committed,
    }
    rounded_levers = {
        "R_committed_to_own": rounded["own"] - rounded["committed"],
        "J_own_to_cmf": rounded["cmf"] - rounded["own"],
        "O_cmf_to_truth": rounded["truth"] - rounded["cmf"],
        "whole_truth_minus_committed": rounded["truth"] - rounded["committed"],
    }

    plasma_path = args.input_dir / "lumina_plasma_state.csv"
    field_path = args.input_dir / "lumina_coevolve_field.csv"
    ion_path = args.input_dir / "lumina_ion_pops.csv"
    root_rows = [
        [args.label, "committed", committed, rounded["committed"], str(plasma_path), "T_e",
         "stored shell electron temperature"],
        [args.label, "own_cs_coupled_root", own, rounded["own"],
         f"{field_path}; {ion_path}; {plasma_path}; {args.model_dir}",
         "cs_J; n_ion; T_e,n_e; line_list/levels/ionization_energies",
         "historical lowest (+ to -) root; ion ladder ratios calibrated at committed state"],
        [args.label, "cmfgen_J_coupled_root", cmf, rounded["cmf"],
         f"{args.cmfgen_jtable}; {ion_path}; {plasma_path}; {args.model_dir}",
         "jtable J_nu; n_ion; T_e,n_e; line_list/levels/ionization_energies",
         "same historical coupled root with only line-pump J replaced by CMFGEN jtable"],
        [args.label, "truth", truth, rounded["truth"], str(args.cmfgen_dir / "RVTJ"),
         "Temperature (10^4K); Velocity (km/s)",
         f"campaign 18760 K anchor; direct linear RVTJ interpolation={rvtj_truth:.9f} K"],
    ]
    write_csv(args.output_dir / "coupled_roots.csv",
              ["dataset", "endpoint", "temperature_K", "rounded_temperature_K", "source_file",
               "source_field", "definition"], root_rows)

    lever_rows = []
    for name in historical_levers:
        lever_rows.append([
            args.label, name, historical_levers[name], rounded_levers[name],
            "successive endpoint difference (descriptive ladder; not assumed independent)",
        ])
    write_csv(args.output_dir / "historical_levers.csv",
              ["dataset", "quantity", "delta_T_K", "rounded_endpoint_delta_K", "definition"], lever_rows)

    cube: dict[tuple[int, int, int], float] = {}
    cube_rows = []
    for bits in itertools.product((0, 1), repeat=3):
        value = endpoint(committed, own, cmf, truth, *bits)
        cube[bits] = value
        cube_rows.append([args.label, *bits, value, value - committed,
                          "F(R,J,O)=(committed if R=0 else [own if J=0 else CMF]) + O*(truth-CMF)"])
    write_csv(args.output_dir / "factorial_endpoints.csv",
              ["dataset", "R_root", "J_cmf_field", "O_truth_residual", "temperature_K",
               "delta_from_committed_K", "definition"], cube_rows)

    factors = {"R": (1, 0, 0), "J": (0, 1, 0), "O": (0, 0, 1)}
    standalone = {name: cube[bits] - cube[(0, 0, 0)] for name, bits in factors.items()}
    whole = cube[(1, 1, 1)] - cube[(0, 0, 0)]
    if whole == 0:
        raise ValueError("directional closure is undefined because committed equals truth")
    required_direction = 1.0 if whole > 0 else -1.0
    directional_closure = {name: required_direction * value for name, value in standalone.items()}
    standalone_sum = sum(standalone.values())
    directional_sum = sum(directional_closure.values())
    interaction_rj = cube[(1, 1, 0)] - cube[(1, 0, 0)] - cube[(0, 1, 0)] + cube[(0, 0, 0)]
    standalone_rows = [
        [args.label, factor, value, "F(single factor)-F(0,0,0)"] for factor, value in standalone.items()
    ]
    standalone_rows.extend([
        [args.label, "sum_standalone", standalone_sum, "R_alone + J_alone + O_alone"],
        [args.label, "whole", whole, "F(1,1,1)-F(0,0,0) = truth-committed"],
        [args.label, "sum_standalone_minus_whole", standalone_sum - whole,
         "primary non-additivity statistic"],
        [args.label, "R_x_J_interaction", interaction_rj,
         "F(1,1,0)-F(1,0,0)-F(0,1,0)+F(0,0,0)"],
        [args.label, "sum_directional_closure", directional_sum,
         "sum sign(truth-committed)*standalone_delta; positive means closes the original gap"],
        [args.label, "total_discrepancy_magnitude", abs(whole),
         "absolute value of truth-committed"],
        [args.label, "sum_directional_closure_minus_total_discrepancy", directional_sum - abs(whole),
         "additivity statistic expressed as closure of the original discrepancy"],
    ])
    write_csv(args.output_dir / "standalone_additivity.csv",
              ["dataset", "quantity", "delta_T_K", "definition"], standalone_rows)

    order_rows = []
    contributions: dict[str, list[float]] = {name: [] for name in factors}
    final_errors: list[float] = []
    for order in itertools.permutations(("R", "J", "O")):
        state = [0, 0, 0]
        before = cube[tuple(state)]
        cumulative = 0.0
        order_name = "->".join(order)
        for step_index, factor in enumerate(order, 1):
            bit_index = {"R": 0, "J": 1, "O": 2}[factor]
            state[bit_index] = 1
            after = cube[tuple(state)]
            contribution = after - before
            cumulative += contribution
            contributions[factor].append(contribution)
            order_rows.append([
                args.label, order_name, step_index, factor, before, after, contribution, cumulative,
                after - committed,
            ])
            before = after
        final_errors.append(cumulative - whole)
    write_csv(args.output_dir / "cumulative_orders.csv",
              ["dataset", "order", "step", "factor", "before_K", "after_K", "step_delta_K",
               "cumulative_delta_K", "endpoint_minus_committed_K"], order_rows)

    width_rows = []
    for factor, values in contributions.items():
        width_rows.append([args.label, factor, min(values), max(values), max(values) - min(values),
                           "range of marginal attribution over all 6 orders"])
    width_rows.append([args.label, "cumulative_final_minus_whole", min(final_errors), max(final_errors),
                       max(final_errors) - min(final_errors),
                       "must be zero for each telescoping order; not an additivity proof"])
    write_csv(args.output_dir / "order_dependence.csv",
              ["dataset", "quantity", "minimum_K", "maximum_K", "width_K", "definition"], width_rows)

    gate_rows: list[list[object]] = []
    gate_pass: bool | None = None
    if args.baseline_gate:
        expected = {"cmf": 18277, "R": 3497, "J": 1660, "O": 483}
        actual = {"cmf": rounded["cmf"], "R": rounded_levers["R_committed_to_own"],
                  "J": rounded_levers["J_own_to_cmf"], "O": rounded_levers["O_cmf_to_truth"]}
        gate_pass = True
        for quantity, expected_value in expected.items():
            passed = actual[quantity] == expected_value
            gate_pass = gate_pass and passed
            gate_rows.append([quantity, expected_value, actual[quantity], "PASS" if passed else "FAIL",
                              "integer gate uses differences of rounded historical endpoints"])
        write_csv(args.output_dir / "baseline_gate.csv",
                  ["quantity", "expected_K", "actual_K", "status", "definition"], gate_rows)

    provenance = [
        ["committed T_e,n_e", str(plasma_path), "T_e;n_e", "consumed"],
        ["own pump", str(field_path), "bin;cs_J", "consumed"],
        ["ion calibration", str(ion_path), "Z;stage;n_ion", "consumed"],
        ["level capture", str(args.input_dir / "lumina_levelpop.csv"),
         "presence checked; fields not consumed by historical estimator", "gate-only"],
        ["atomic model", str(args.model_dir), "line_list.csv;levels.csv;ionization_energies.csv", "consumed"],
        ["deposition", str(args.deposition_file), "heating_rate", "consumed"],
        ["CMFGEN pump", str(args.cmfgen_jtable), "J_nu sampled on 1000-bin grid", "consumed"],
        ["CMFGEN J source", str(args.cmfgen_dir / "EDDFACTOR"), "J_nu", "jtable lineage"],
        ["CMFGEN depth/truth", str(args.cmfgen_dir / "RVTJ"),
         "Velocity (km/s);Temperature (10^4K)", "consumed for truth cross-check"],
        ["CMFGEN opacity", str(args.cmfgen_dir / "MEANOPAC"),
         "presence checked; no coupled-root term depends on opacity table", "gate-only"],
        ["historical photoheating scalar", "CLI --photo-heating", str(args.photo_heating), "consumed"],
    ]
    write_csv(args.output_dir / "provenance.csv",
              ["quantity", "source_file", "source_field", "use"], provenance)

    summary = {
        "dataset": args.label,
        "committed_K": committed,
        "own_cs_root_K": own,
        "cmfgen_J_root_K": cmf,
        "truth_K": truth,
        "rvtj_linear_truth_K": rvtj_truth,
        "historical_levers_K": historical_levers,
        "rounded_endpoints_K": rounded,
        "rounded_historical_levers_K": rounded_levers,
        "standalone_K": standalone,
        "standalone_directional_closure_K": directional_closure,
        "sum_standalone_K": standalone_sum,
        "sum_directional_closure_K": directional_sum,
        "total_discrepancy_magnitude_K": abs(whole),
        "sum_directional_closure_minus_total_discrepancy_K": directional_sum - abs(whole),
        "whole_K": whole,
        "sum_standalone_minus_whole_K": standalone_sum - whole,
        "R_x_J_interaction_K": interaction_rj,
        "cumulative_final_minus_whole_min_K": min(final_errors),
        "cumulative_final_minus_whole_max_K": max(final_errors),
        "order_attribution_width_K": {
            factor: max(values) - min(values) for factor, values in contributions.items()
        },
        "baseline_gate": gate_pass,
        "line_count_used": estimator.line_count,
        "definition": (
            "historical 2026-07-19 analytic coupled-root estimator; lowest +to- root on "
            "24 geometric intervals from 3500 to 140000 K and 40 bisections; ion adjacent-stage "
            "ratios scale as (T/T_committed)^0.8; H_photo is the historical fixed CLI scalar"
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if gate_pass is False:
        raise SystemExit("baseline reproduction gate FAILED")


if __name__ == "__main__":
    run(arguments())
