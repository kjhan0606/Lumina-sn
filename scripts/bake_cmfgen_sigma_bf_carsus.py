#!/usr/bin/env python3
"""Bake CMFGEN sigma_bf(nu) onto LUMINA's BF grid for the carsus-rebuilt levels.

The previous baker (in expand_atomic_data_cmfgen.py) keyed phot_data lookups by
the original CMFGEN config strings, which are stripped during carsus ingest.
This script rebuilds the binary by:

  1. Reading every level from data/tardis_reference/levels.csv (the carsus
     output).
  2. For each (Z, ion) covered by cmfgen_config_lumina.yml, parsing the CMFGEN
     osc_data to recover (config, E_cm, g) per level.
  3. Matching each carsus level to a CMFGEN level by (energy, g) — both are
     CMFGEN-sourced for ions with priority=30, so the energies should agree
     to ~10 ueV.
  4. Parsing phot_data and interpolating sigma_bf(nu) onto the LUMINA BF grid
     (BF_NU_MIN..BF_NU_MAX, BF_N_FREQ_BIN bins, identical to lumina.h).
  5. Writing data/atomic/cmfgen_sigma_bf.bin in the same binary layout the
     C-side load_cmfgen_sigma_bf() expects (magic 0x434D4644, version 1).

Levels for which no CMFGEN match is found get sigma_bf == 0 across the grid
and the C-side opacity loop falls back to the Kramers form.
"""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
sys.path.insert(0, str(ROOT / "scripts"))
from cmfgen_parser import parse_osc, parse_phot  # noqa: E402

LEVELS_CSV  = ROOT / "data" / "tardis_reference" / "levels.csv"
CMFGEN_ROOT = ROOT / "data" / "atomic" / "cmfgen"
CONFIG_YML  = ROOT / "scripts" / "cmfgen_config_lumina.yml"
OUT_BIN     = ROOT / "data" / "atomic" / "cmfgen_sigma_bf.bin"

# Must match BF grid in src/lumina.h
BF_NU_MIN     = 1.5e14
BF_NU_MAX     = 3.0e16
BF_N_FREQ_BIN = 1000

C_CGS     = 2.99792458e10
H_CGS     = 6.62607015e-27
EV_TO_ERG = 1.602176634e-12
CM2EV     = 1.239841984e-4

# Energy match tolerance (eV).  Carsus rounds level energies to ~10 digits,
# CMFGEN levels are 8-digit.  1e-3 eV is generous but still distinguishes
# adjacent fine-structure components in iron-peak ions.
E_MATCH_TOL_EV = 1.0e-3

CMFGEN_DIRS = {
    "H":"HYD", "He":"HE", "C":"CARB", "N":"NIT", "O":"OXY", "Ne":"NEON",
    "Na":"NA", "Mg":"MG", "Al":"AL", "Si":"SIL", "P":"PHOS", "S":"SUL",
    "Cl":"CHL", "Ar":"ARG", "K":"POT", "Ca":"CA", "Sc":"SCAN", "Ti":"TIT",
    "V":"VAN", "Cr":"CHRO", "Mn":"MAN", "Fe":"FE", "Co":"COB", "Ni":"NICK",
}
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
SYM_TO_Z = {"H":1,"He":2,"C":6,"N":7,"O":8,"Ne":10,"Na":11,"Mg":12,"Al":13,
            "Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,
            "Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_yml_ions() -> list[tuple[int, int, Path, str, list[str]]]:
    """Return [(Z, ion_charge, date_dir, osc_filename, pho_filenames), ...]."""
    with open(CONFIG_YML) as f:
        cfg = yaml.safe_load(f)
    out = []
    for sym, atom_block in cfg["atom"].items():
        Z = SYM_TO_Z.get(sym)
        eldir = CMFGEN_DIRS.get(sym)
        if Z is None or eldir is None:
            continue
        for ion_charge, entry in atom_block["ion_charge"].items():
            stage = ion_charge + 1
            date  = entry["date"]
            ion_dir = CMFGEN_ROOT / eldir / ROMAN[stage] / date
            if not ion_dir.is_dir():
                log(f"  SKIP {sym} {ROMAN[stage]} {date} — dir missing")
                continue
            osc_name = entry.get("osc", "osc_data")
            pho_list = entry.get("pho") or []
            out.append((Z, ion_charge, ion_dir, osc_name, list(pho_list)))
    return out


def load_carsus_levels() -> pd.DataFrame:
    df = pd.read_csv(LEVELS_CSV)
    df["global_idx"] = np.arange(len(df))
    return df


def match_levels_to_cmfgen(carsus_sub: pd.DataFrame, cmfgen_levels: np.ndarray
                           ) -> dict[int, str]:
    """Return {global_idx -> cmfgen config string}."""
    cmf_E = cmfgen_levels["E_cm"] * CM2EV
    cmf_g = cmfgen_levels["g"]
    cmf_cfg = cmfgen_levels["config"]
    used = np.zeros(len(cmf_E), dtype=bool)

    out: dict[int, str] = {}
    for _, row in carsus_sub.iterrows():
        E = float(row.energy_eV)
        g = float(row.g)
        # candidates within energy tolerance + same g
        dE = np.abs(cmf_E - E)
        cand = np.where((dE < E_MATCH_TOL_EV) & (np.abs(cmf_g - g) < 0.5)
                        & (~used))[0]
        if cand.size == 0:
            # relax g (some carsus rounds; rare)
            cand = np.where((dE < E_MATCH_TOL_EV) & (~used))[0]
        if cand.size == 0:
            continue
        # pick closest in energy
        k = cand[np.argmin(dE[cand])]
        used[k] = True
        out[int(row.global_idx)] = str(cmf_cfg[k])
    return out


def bake_one_level(entry, nu_grid: np.ndarray, nu_thresh: float) -> np.ndarray:
    """Evaluate sigma_bf(nu) for a single phot entry over the BF grid."""
    nb = nu_grid.size
    sigma = np.zeros(nb, dtype="f8")
    idx = np.searchsorted(nu_grid, nu_thresh)
    if idx >= nb:
        return sigma

    if entry.cs_type in (20, 21, 22):
        if entry.energy.size == 0 or entry.sigma_Mb.size == 0:
            return sigma
        nu_pts = entry.energy * nu_thresh
        s_Mb   = entry.sigma_Mb * 1.0e-18
        sigma[idx:] = np.interp(nu_grid[idx:], nu_pts, s_Mb,
                                left=0.0, right=s_Mb[-1])
    elif entry.cs_type == 1 and entry.sigma_Mb.size >= 3:
        sigma_0 = float(entry.sigma_Mb[0])
        beta    = float(entry.sigma_Mb[1])
        s_exp   = float(entry.sigma_Mb[2])
        y = nu_grid[idx:] / nu_thresh
        sigma[idx:] = sigma_0 * (beta + (1.0 - beta) / y) * (y ** (-s_exp)) * 1e-18
    elif entry.cs_type in (2, 3, 7, 8, 9) and entry.sigma_Mb.size >= 1:
        sigma_0 = float(entry.sigma_Mb[0]) * 1.0e-18
        if sigma_0 <= 0:
            return sigma
        y = nu_grid[idx:] / nu_thresh
        sigma[idx:] = sigma_0 / (y ** 3)
    return sigma


def main() -> None:
    log(f"output  : {OUT_BIN}")
    log(f"levels  : {LEVELS_CSV}")
    log(f"config  : {CONFIG_YML}")

    carsus = load_carsus_levels()
    n_levels = len(carsus)
    log(f"carsus  : {n_levels} levels across "
        f"{carsus.groupby(['atomic_number','ion_number']).ngroups} ions")

    log_min = np.log(BF_NU_MIN)
    log_max = np.log(BF_NU_MAX)
    d_log_nu = (log_max - log_min) / BF_N_FREQ_BIN
    nu_grid = BF_NU_MIN * np.exp((np.arange(BF_N_FREQ_BIN) + 0.5) * d_log_nu)

    sigma_grid = np.zeros((n_levels, BF_N_FREQ_BIN), dtype="f8")
    has_cmfgen = np.zeros(n_levels, dtype="i1")

    yml_ions = load_yml_ions()
    log(f"cmfgen ions in yml: {len(yml_ions)}")

    n_baked_total = 0
    for (Z, ion_csv, date_dir, osc_name, pho_list) in yml_ions:
        sym = next(s for s, z in SYM_TO_Z.items() if z == Z)
        sub = carsus[(carsus.atomic_number == Z) &
                     (carsus.ion_number == ion_csv)].copy()
        if sub.empty:
            log(f"  {sym} {ROMAN[ion_csv+1]:4s}: no carsus levels — skip")
            continue
        osc_p = date_dir / osc_name
        if not osc_p.exists():
            log(f"  {sym} {ROMAN[ion_csv+1]:4s}: osc file '{osc_name}' missing — skip")
            continue
        try:
            osc = parse_osc(osc_p)
        except Exception as e:
            log(f"  {sym} {ROMAN[ion_csv+1]:4s}: osc parse FAIL: {e}")
            continue
        if osc.n_levels == 0:
            continue
        E_ion_eV = osc.ionization_eV

        cfg_to_global = match_levels_to_cmfgen(sub, osc.levels)

        phot_paths = [date_dir / nm for nm in pho_list
                      if (date_dir / nm).exists()]
        if not phot_paths:
            log(f"  {sym} {ROMAN[ion_csv+1]:4s}: matched "
                f"{len(cfg_to_global)}/{len(sub)} but no phot files "
                f"({pho_list}) — skip")
            continue
        # Aggregate entries across all listed phot files
        all_entries = []
        for phot_p in phot_paths:
            try:
                p = parse_phot(phot_p)
                all_entries.extend(p.entries)
            except Exception as e:
                log(f"  {sym} {ROMAN[ion_csv+1]:4s}: phot {phot_p.name} "
                    f"parse FAIL: {e}")
        if not all_entries:
            continue

        # Wrap to mimic PhotIon for the rest of the loop
        class _PhotWrap:
            entries = all_entries
        phot = _PhotWrap()

        # Phot entries can repeat the same config for different J levels.
        # Build {normalized_config -> [PhotEntry]}.  For the J-resolved
        # configs in osc_data we want exact match; for term-grouped phot
        # entries we fall back to a stripped term lookup.
        from collections import defaultdict

        def norm(s: str) -> str: return s.strip().lower()

        import re
        TERM_RE = re.compile(r"\[[^\[\]]*\]\s*$")
        def term(s: str) -> str: return TERM_RE.sub("", norm(s))

        cfg_phot: dict[str, list] = defaultdict(list)
        term_phot: dict[str, list] = defaultdict(list)
        for e in phot.entries:
            cfg_phot[norm(e.config)].append(e)
            term_phot[term(e.config)].append(e)

        n_baked_ion = 0
        # Reverse map cmfgen-config -> [global_idx]
        gi_to_cfg = cfg_to_global  # {global_idx: cfg_str}
        for gi, cfg in gi_to_cfg.items():
            cfg_n = norm(cfg)
            entries = cfg_phot.get(cfg_n)
            if not entries:
                entries = term_phot.get(term(cfg))
            if not entries:
                continue
            E_level_eV = float(carsus.iloc[gi].energy_eV)
            E_thresh = E_ion_eV - E_level_eV
            if E_thresh <= 0:
                continue
            nu_thresh = E_thresh * EV_TO_ERG / H_CGS

            # Sum over all entries that route from this level (rare; usually 1)
            sig = np.zeros(BF_N_FREQ_BIN, dtype="f8")
            for e in entries:
                sig += bake_one_level(e, nu_grid, nu_thresh)
            if np.any(sig > 0):
                sigma_grid[gi] = sig
                has_cmfgen[gi] = 1
                n_baked_ion += 1

        n_baked_total += n_baked_ion
        log(f"  {sym} {ROMAN[ion_csv+1]:4s}: matched "
            f"{len(cfg_to_global)}/{len(sub)} levels, baked {n_baked_ion}")

    log(f"total baked: {n_baked_total} / {n_levels} levels "
        f"({100.0 * n_baked_total / n_levels:.1f}%)")

    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BIN, "wb") as f:
        f.write(struct.pack("<II", 0x434D4644, 1))
        f.write(struct.pack("<ii", n_levels, BF_N_FREQ_BIN))
        f.write(struct.pack("<dd", BF_NU_MIN, BF_NU_MAX))
        f.write(has_cmfgen.tobytes())
        pad = (8 - (n_levels % 8)) % 8
        f.write(b"\x00" * pad)
        f.write(sigma_grid.tobytes(order="C"))
    log(f"wrote {OUT_BIN}  size = {OUT_BIN.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
