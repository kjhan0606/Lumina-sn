#!/usr/bin/env python3
"""Offline audit for CODEX_EPS_THIN investigations 1 and 2.

This consumer reads existing CSV, LINEPOP metadata, and imported CMFGEN atomic
tables.  It does not run transport, a plasma model, or GPU code.  Invalid or
missing inputs abort; no clamp, floor, cap, fallback, or replacement value is
used by this analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import uv_t2n9_offline as linepop_reader  # noqa: E402


CAPTURE = Path("/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932")
MODEL = ROOT / "data/tardis_reference_toy06_19p48d_sivcaiv"
SCHEME = CAPTURE / "scheme_fracture_s0/scheme_fracture_s0_line_rank.csv"
LINEPOP = CAPTURE / "linepop_iter10"
PLASMA = CAPTURE / "lumina_plasma_state.csv"
OLD_DIR = (ROOT / "validation/cmfgen_toy06_19p48d/analysis/"
           "reddening_localization")
ARTIS_COL_CONST = 8.629e-6
TREF_K = 10400.0
IGC_MAGIC = 0x49474331


TARGETS = {
    776418: {
        "ion": "Fe IV", "Z": 26, "ion0": 3, "lo": 0, "up": 5,
        "lower_config": "3d5_6Se[5/2]", "upper_config": "3d5_4Pe[5/2]",
        "osc": Path("/gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/19apr23/osc_data"),
        "col": Path("/gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/19apr23/col_data"),
        "collision_reference": "ZP97_FeIV_col (Zhang and Pradhan)",
    },
    748621: {
        "ion": "Co IV", "Z": 27, "ion0": 3, "lo": 0, "up": 20,
        "lower_config": "3d6_5De[4]", "upper_config": "3d6_3De[3]",
        "osc": Path("/gpfs/kjhan/cmfgen_21jun23/atomic/COB/IV/19apr23/osc_data"),
        "col": Path("/gpfs/kjhan/cmfgen_21jun23/atomic/COB/IV/19apr23/col_data"),
        "collision_reference": "Zha96_FeIII_col; CMFGEN file says 'Using FeIII values?'",
    },
    774507: {
        "ion": "Fe IV", "Z": 26, "ion0": 3, "lo": 0, "up": 6,
        "lower_config": "3d5_6Se[5/2]", "upper_config": "3d5_4Pe[3/2]",
        "osc": Path("/gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/19apr23/osc_data"),
        "col": Path("/gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/19apr23/col_data"),
        "collision_reference": "ZP97_FeIV_col (Zhang and Pradhan)",
    },
    635410: {
        "ion": "Ni IV", "Z": 28, "ion0": 3, "lo": 0, "up": 16,
        "lower_config": "3d7_4Fe[9/2]", "upper_config": "3d7_2Fe[7/2]",
        "osc": Path("/gpfs/kjhan/cmfgen_21jun23/atomic/NICK/IV/19apr23/osc_data"),
        "col": Path("/gpfs/kjhan/cmfgen_21jun23/atomic/NICK/IV/19apr23/col_data"),
        "collision_reference": "Fernandez-Menchero et al. 2019, MNRAS 483, 2154",
    },
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), f"empty CSV: {path}")
    return rows


def source_line(path: Path, first: str, second: str | None = None) -> dict[str, Any]:
    matches = []
    # Count physical newline records.  ``str.splitlines`` also splits CMFGEN's
    # form-feed page markers and would report a line number different from rg/sed.
    for number, text in enumerate(path.read_text(errors="replace").split("\n"), 1):
        if first in text and (second is None or second in text):
            matches.append((number, text.strip()))
    require(len(matches) == 1,
            f"expected one source row in {path} for {first!r}/{second!r}, got {len(matches)}")
    number, text = matches[0]
    return {"path": str(path), "line": number, "text": text}


def parse_osc_triplet(text: str) -> tuple[float, float, float]:
    # First three scientific/decimal fields after the second configuration are f, A, lambda.
    fields = re.findall(r"(?<![A-Za-z0-9_/])[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?", text)
    # Configuration strings contain digits, so locate the upper configuration terminator first.
    closing = text.find("]", text.find("]") + 1)
    require(closing >= 0, f"cannot locate two configurations: {text}")
    tail = text[closing + 1:]
    nums = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?", tail)
    require(len(nums) >= 3, f"cannot parse f/A/lambda: {text}")
    return tuple(float(x.replace("D", "E").replace("d", "e")) for x in nums[:3])


def read_collision_binary(path: Path, z: int, ion0: int,
                          lo: int, up: int) -> tuple[list[float], list[float]]:
    with path.open("rb") as stream:
        raw = stream.read(28)
        require(len(raw) == 28, f"truncated collision header: {path}")
        magic, version, bz, bi, ntr, nt, nlev = struct.unpack("<IIiiiii", raw)
        require((magic, version, bz, bi) == (IGC_MAGIC, 1, z, ion0),
                f"collision identity mismatch: {path}")
        require(ntr > 0 and nt >= 2 and nlev > max(lo, up),
                f"invalid collision dimensions: {path}")
        tgrid = list(struct.unpack(f"<{nt}d", stream.read(8 * nt)))
        target = (min(lo, up), max(lo, up))
        found: list[float] | None = None
        for _ in range(ntr):
            pair_raw = stream.read(8)
            omega_raw = stream.read(8 * nt)
            require(len(pair_raw) == 8 and len(omega_raw) == 8 * nt,
                    f"truncated collision record: {path}")
            pair = struct.unpack("<ii", pair_raw)
            omega = list(struct.unpack(f"<{nt}d", omega_raw))
            if pair == target:
                require(found is None, f"duplicate collision pair {target}: {path}")
                found = omega
        require(stream.read(1) == b"", f"collision file has trailing bytes: {path}")
    require(found is not None, f"collision pair {target} absent: {path}")
    return tgrid, found


def interpolate_linear_clamped(tgrid: list[float], values: list[float],
                               temperature: float) -> float:
    require(len(tgrid) == len(values) and len(tgrid) >= 2,
            "invalid interpolation arrays")
    require(all(math.isfinite(x) for x in tgrid + values),
            "nonfinite collision table value")
    require(all(tgrid[i] < tgrid[i + 1] for i in range(len(tgrid) - 1)),
            "collision temperature grid is not strictly increasing")
    i = 0
    while i < len(tgrid) - 2 and temperature > tgrid[i + 1]:
        i += 1
    frac = (temperature - tgrid[i]) / (tgrid[i + 1] - tgrid[i])
    frac = min(1.0, max(0.0, frac))
    return values[i] + frac * (values[i + 1] - values[i])


def beta_escape(tau: float) -> float:
    require(math.isfinite(tau) and tau > 0.0, "tau must be finite and positive")
    return -math.expm1(-tau) / tau


def rates(ne: float, te: float, coeff: float, g_up: int,
          a_ul: float, tau: float) -> dict[str, float]:
    require(ne > 0.0 and te > 0.0 and coeff > 0.0 and g_up > 0 and a_ul > 0.0,
            "rate input outside defined positive domain")
    c_ul = ne * coeff / (g_up * math.sqrt(te))
    beta = beta_escape(tau)
    eps = c_ul / (c_ul + a_ul * beta)
    eps_thin_limit = c_ul / (c_ul + a_ul)
    return {
        "C_ul_s-1": c_ul, "A_ul_s-1": a_ul, "C_ul_over_A_ul": c_ul / a_ul,
        "beta_escape": beta, "eps_prime_at_tau": eps,
        "eps_tau_to_zero": eps_thin_limit,
    }


def plasma_states() -> tuple[dict[str, float], dict[str, float]]:
    rows = read_csv_rows(PLASMA)
    row = next((x for x in rows if int(x["shell_id"]) == 0), None)
    require(row is not None, "shell 0 absent from plasma state")
    final = {"n_e_cm-3": float(row["n_e"]), "T_e_K": float(row["T_e"])}
    lp = linepop_reader.parse_linepop(LINEPOP)
    slots = [i for i, shell in enumerate(lp.shells) if int(shell) == 0]
    require(len(slots) == 1, "LINEPOP shell 0 selection is not unique")
    state = lp.shell_state[slots[0]]
    iteration10 = {"n_e_cm-3": float(state[2]), "T_e_K": float(state[0])}
    return final, iteration10


def investigation1() -> dict[str, Any]:
    scheme = {int(row["line_id"]): row for row in read_csv_rows(SCHEME)
              if int(row["line_id"]) in TARGETS}
    require(set(scheme) == set(TARGETS), "target set absent/incomplete in scheme CSV")
    lines = {int(row["line_id"]): row
             for row in read_csv_rows(MODEL / "line_list.csv")
             if int(row["line_id"]) in TARGETS}
    require(set(lines) == set(TARGETS), "target set absent/incomplete in line_list.csv")
    levels = {(int(row["atomic_number"]), int(row["ion_number"]),
               int(row["level_number"])): row
              for row in read_csv_rows(MODEL / "levels.csv")}
    final_state, iter10_state = plasma_states()
    output = []
    replay_errors = []
    wrong_g_errors = []
    final_vs_capture = []
    for line_id, meta in TARGETS.items():
        sr = scheme[line_id]
        ar = lines[line_id]
        require((int(ar["atomic_number"]), int(ar["ion_number"]),
                 int(ar["level_number_lower"]), int(ar["level_number_upper"])) ==
                (meta["Z"], meta["ion0"], meta["lo"], meta["up"]),
                f"line identity mismatch for {line_id}")
        upper = levels[(meta["Z"], meta["ion0"], meta["up"])]
        lower = levels[(meta["Z"], meta["ion0"], meta["lo"])]
        g_up = int(upper["g"])
        a_ul = float(ar["A_ul"])
        binary = MODEL / f"ige_col_{meta['Z']}_{meta['ion0']}_cmfgen.bin"
        tgrid, omega_grid = read_collision_binary(
            binary, meta["Z"], meta["ion0"], meta["lo"], meta["up"])
        upsilon = interpolate_linear_clamped(tgrid, omega_grid, TREF_K)
        coeff = ARTIS_COL_CONST * upsilon

        osc_evidence = source_line(meta["osc"], meta["lower_config"],
                                   meta["upper_config"])
        f_original, a_original, lambda_original = parse_osc_triplet(osc_evidence["text"])
        require(math.isclose(a_original, a_ul, rel_tol=0.0, abs_tol=1e-12),
                f"A_ul disagrees with original osc_data for {line_id}")
        require(math.isclose(lambda_original, float(ar["wavelength"]),
                             rel_tol=0.0, abs_tol=5e-4),
                f"wavelength disagrees with original osc_data for {line_id}")
        col_evidence = source_line(meta["col"], meta["lower_config"],
                                   meta["upper_config"])

        tau = float(sr["tau_used"])
        observed_eps = float(sr["eps_l"])
        final_rates = rates(final_state["n_e_cm-3"], final_state["T_e_K"],
                            coeff, g_up, a_ul, tau)
        replay = rates(iter10_state["n_e_cm-3"], iter10_state["T_e_K"],
                       coeff, g_up, a_ul, tau)
        wrong_g = rates(iter10_state["n_e_cm-3"], iter10_state["T_e_K"],
                        coeff, g_up + 1, a_ul, tau)
        replay_errors.append(abs(replay["eps_prime_at_tau"] - observed_eps))
        wrong_g_errors.append(abs(wrong_g["eps_prime_at_tau"] - observed_eps))
        final_vs_capture.append(abs(final_rates["eps_prime_at_tau"] - observed_eps))
        eta_reconstructed = float(sr["w"]) * observed_eps * float(sr["S_l_used"])
        require(math.isclose(eta_reconstructed, float(sr["eta_l"]),
                             rel_tol=2e-15, abs_tol=0.0),
                f"eta=w*eps*S round trip failed for {line_id}")
        output.append({
            "line_id": line_id, "ion": meta["ion"],
            "lambda_A": float(sr["lambda_A"]), "tau_used": tau,
            "eps_l_captured": observed_eps,
            "level_pair": [meta["lo"], meta["up"]],
            "lower_energy_eV": float(lower["energy_eV"]),
            "upper_energy_eV": float(upper["energy_eV"]),
            "g_up": g_up, "f_lu_line_list": float(ar["f_lu"]),
            "A_ul_line_list_s-1": a_ul,
            "original_osc_data": {
                "evidence": osc_evidence, "f": f_original,
                "A_ul_s-1": a_original, "lambda_A": lambda_original,
            },
            "transition_character": {
                "lower_configuration": meta["lower_config"],
                "upper_configuration": meta["upper_config"],
                "same_even_parity": True,
                "spin_multiplicity_changes": True,
                "classification": "E1-forbidden/intercombination-character; exact M1/E2/mixing label is not encoded",
            },
            "collision_data": {
                "binary_path": str(binary), "source_evidence": col_evidence,
                "reference": meta["collision_reference"], "tier": 1,
                "Tref_K": TREF_K, "upsilon_at_Tref": upsilon,
                "coeff_cm3_s-1_K0p5": coeff,
                "prescription": "coeff=8.629e-6*Upsilon_CMFGEN(Tref), linear-in-T interpolation clamped to table endpoints",
                "van_Regemorter_used": False,
            },
            "rates_at_requested_final_state": final_rates,
            "rates_at_iter10_eps_generation_state": replay,
            "eps_replay_abs_error": abs(replay["eps_prime_at_tau"] - observed_eps),
            "source_product_check": {
                "S_l_used": float(sr["S_l_used"]),
                "eps_times_S_l_used": observed_eps * float(sr["S_l_used"]),
                "w": float(sr["w"]), "eta_l_captured": float(sr["eta_l"]),
                "w_times_eps_times_S_reconstructed": eta_reconstructed,
            },
        })

    require(max(replay_errors) < 2e-9,
            f"iter10 eps replay failed: max error {max(replay_errors)}")
    require(min(wrong_g_errors) > 1e-3,
            "negative control failed to reject injected g_up+1")
    require(max(final_vs_capture) > 1e-4,
            "epoch negative control unexpectedly reproduced captured eps")
    return {
        "status": "PASS", "schema": "codex-eps-thin-audit-v1",
        "definitions": {
            "requested_rate": "C_ul=n_e*coeff/(g_up*sqrt(T_e)); final-state n_e,T_e are shell 0 fields n_e,T_e in lumina_plasma_state.csv",
            "eps_replay": "C_ul/(C_ul+A_ul*((1-exp(-tau))/tau)); replay uses shell 0 [T_e,T_rad,n_e,t_exp] in LINEPOP iter10 selected_shell_state",
            "atomic_A": "line_list.A_ul compared to the matched transition row A field in original CMFGEN osc_data",
            "collision_coeff": "8.629e-6 times the imported binary transition Upsilon linearly interpolated at production Tref=10400 K",
            "source_product": "eta_l=w*eps_l*S_l_used, using the four fields in scheme_fracture_s0_line_rank.csv",
            "clamp": 0, "floor": 0, "cap": 0, "fallback": 0,
            "undefined_policy": "abort; no replacement value",
        },
        "sources": {
            "scheme_csv": str(SCHEME), "line_list_csv": str(MODEL / "line_list.csv"),
            "levels_csv": str(MODEL / "levels.csv"),
            "plasma_state_csv": str(PLASMA), "linepop": str(LINEPOP),
        },
        "states": {"requested_final": final_state,
                   "iter10_eps_generation": iter10_state},
        "self_checks": {
            "iter10_eps_replay_max_abs_error": max(replay_errors),
            "eta_product_round_trip": "PASS",
            "atomic_A_round_trip": "PASS",
            "collision_pair_present_in_imported_binary": "PASS",
        },
        "negative_controls": {
            "injected_g_up_plus_one": {
                "status": "PASS-rejected", "minimum_eps_abs_error": min(wrong_g_errors)},
            "wrong_epoch_final_state_used_to_replay_iter10_eps": {
                "status": "PASS-rejected", "maximum_eps_abs_error": max(final_vs_capture)},
        },
        "lines": output,
    }


def investigation2() -> dict[str, Any]:
    ledger_path = OLD_DIR / "taskB_band_ledger.csv"
    ions_path = OLD_DIR / "taskB_top_ions.csv"
    ledger = read_csv_rows(ledger_path)
    ions = read_csv_rows(ions_path)
    deep = [row for row in ledger if row["group"] == "s0-2"]
    require(bool(deep), "old deep ledger is empty")
    total = math.fsum(float(row["emitE"]) for row in deep)
    pile_rows = [row for row in deep if row["band"] == "NUV_1290_2000"]
    require(len(pile_rows) == 1, "old NUV pile row is not unique")
    pile = float(pile_rows[0]["emitE"])
    co_rows = [row for row in ions if row["group"] == "s0-2"
               and row["role"] == "EMIT_NUVpile_1290_2000"
               and row["Z"] == "27" and row["ion_idx"] == "3"]
    require(len(co_rows) == 1, "old Co IV ion row is not unique")
    co = float(co_rows[0]["E"])
    return {
        "status": "REPRODUCED", "schema": "codex-c11-origin-v1",
        "definitions": {
            "event_archive": "logs/coevolve_consume_a10_kx_gphall/lumina_events.bin, stored CAP128M iter11 prefix",
            "emission": "EventRec etype in {2 line-emit,4 kpkt-ff,5 kpkt-fb}",
            "event_wavelength": "2.99792458e18/EventRec.nu_comov Angstrom",
            "deep": "EventRec.shell in {0,1,2}",
            "pile": "1290 <= event_wavelength_A < 2000",
            "co_iv_pile_numerator": "sum EventRec.energy for pile/deep/emission records with valid line_id whose emitted line table has Z=27, ion_number=3",
            "all_deep_emission_denominator": "sum EventRec.energy for all deep/emission records in the 11 exhaustive taskB ledger wavelength bands; continuum etype 4/5 remains in this denominator",
            "clamp": 0, "floor": 0, "cap": 0, "fallback": 0,
        },
        "sources": {
            "band_ledger": str(ledger_path), "band_ledger_fields": ["group", "band", "emitE"],
            "ion_table": str(ions_path), "ion_table_fields": ["group", "role", "Z", "ion_idx", "E", "frac_of_role"],
            "producer_script": str(OLD_DIR / "taskB_event_forensics.py"),
            "claim": str(ROOT / "validation/cmfgen_toy06_19p48d/analysis/criminal_record/CRIMINAL_RECORD.md") + ":64",
        },
        "values": {
            "all_deep_emission_energy": total,
            "deep_1290_2000_emission_energy": pile,
            "co_iv_deep_1290_2000_emission_energy": co,
            "pile_fraction_of_all_deep_emission": pile / total,
            "co_iv_fraction_of_deep_pile": co / pile,
            "co_iv_deep_pile_fraction_of_all_deep_emission": co / total,
            "product_of_reported_factors": (pile / total) * (co / pile),
        },
        "quantity_class": "event-ledger packet comoving energy weight; not mc_J field dump",
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "validation/codex_eps_thin")
    args = parser.parse_args()
    audit1 = investigation1()
    audit2 = investigation2()
    write_json(args.outdir / "investigation1_eps_thin.json", audit1)
    write_json(args.outdir / "investigation2_c11_origin.json", audit2)
    print(json.dumps({
        "status": "PASS", "investigation1_lines": len(audit1["lines"]),
        "investigation1_eps_replay_max_abs_error":
            audit1["self_checks"]["iter10_eps_replay_max_abs_error"],
        "investigation2_fraction":
            audit2["values"]["co_iv_deep_pile_fraction_of_all_deep_emission"],
        "outdir": str(args.outdir),
    }, indent=2))


if __name__ == "__main__":
    main()
