#!/usr/bin/env python3
"""Bake per-level SPIN MULTIPLICITY (2S+1) for the LUMINA spin-allowed Milne
recombination gate (LUMINA_ALPHA_SPINGATE=1).

The runtime levels.csv (carsus output) carries NO term label, and the C code
never reads the CMFGEN OSC files -- so the multiplicity has to be produced
offline here and shipped as a companion table (level_multiplicity.csv) keyed by
(atomic_number, ion_number, level_number).  The C loader (load_atomic_data,
gated) reads it and fills atom->level_mult.

Method (mirrors scripts/bake_cmfgen_sigma_bf_carsus.py so the level identity is
provenance-consistent with the baked sigma_bf.bin):
  1. Read every level from levels.csv.
  2. For each (Z, ion) in cmfgen_config_lumina.yml, parse the CMFGEN osc_data
     to recover (config, E_cm, g) per level.
  3. Match each carsus level to a CMFGEN level by (energy, g).
  4. Parse the multiplicity from the matched CMFGEN config term label.

Multiplicity parse rule (ROBUST -- the naive "leading integer after last '_'"
breaks on real CMFGEN labels):
  * strip trailing J designation  "...[4]" / "...[9/2]"
  * remove parent-term parentheses "(3P2)" / "(F<9/2>)"
  * take the token after the last '_'
  * multiplicity = the digit immediately BEFORE the uppercase term letter
    e.g. "3d6_5De"    -> 5D  -> 5
         "3d7_a4Fe"   -> 4F  -> 4   (naive: 'a' -> 0, WRONG)
         "3s21Se"     -> 1S  -> 1   (naive: leading '3' of 3s2, WRONG)
  * unparseable -> 0 (unknown; the gate never skips unknown levels)

Output columns: atomic_number, ion_number, level_number, multiplicity
Only matched+parsed levels are written; everything else defaults to 0 in C.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
sys.path.insert(0, str(ROOT / "scripts"))
from cmfgen_parser import parse_osc  # noqa: E402

# Read/write the reference dir the runtime actually loads.  The keying is
# (Z, ion, level_number) so the table is portable across reference dirs that
# share the CMFGEN level ordering; point LUMINA_SPINGATE_MULT at this file for
# any dir that lacks its own copy.
REF_DIR    = ROOT / "data" / "tardis_reference_toy06_19p48d"
LEVELS_CSV = REF_DIR / "levels.csv"
CONFIG_YML = ROOT / "scripts" / "cmfgen_config_lumina.yml"
OUT_CSV    = REF_DIR / "level_multiplicity.csv"

CMFGEN_ROOT = ROOT / "data" / "atomic" / "cmfgen"
CM2EV = 1.239841984e-4
E_MATCH_TOL_EV = 1.0e-3

CMFGEN_DIRS = {
    "H": "HYD", "He": "HE", "C": "CARB", "N": "NIT", "O": "OXY", "Ne": "NEON",
    "Na": "NA", "Mg": "MG", "Al": "AL", "Si": "SIL", "P": "PHOS", "S": "SUL",
    "Cl": "CHL", "Ar": "ARG", "K": "POT", "Ca": "CA", "Sc": "SCAN", "Ti": "TIT",
    "V": "VAN", "Cr": "CHRO", "Mn": "MAN", "Fe": "FE", "Co": "COB", "Ni": "NICK",
}
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
SYM_TO_Z = {"H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "Ne": 10, "Na": 11,
            "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
            "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
            "Fe": 26, "Co": 27, "Ni": 28}

_JBRK = re.compile(r"\[[^\[\]]*\]\s*$")
_PAREN = re.compile(r"\([^()]*\)")
_TERM = re.compile(r"(\d)[A-Z]")


def parse_mult(config: str) -> int:
    """Spin multiplicity (2S+1) from a CMFGEN config/term label (see header)."""
    s = _JBRK.sub("", str(config).strip())
    s = _PAREN.sub("", s)
    tok = s.split("_")[-1]
    m = _TERM.search(tok)
    return int(m.group(1)) if m else 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_yml_ions():
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
            ion_dir = CMFGEN_ROOT / eldir / ROMAN[stage] / entry["date"]
            if not ion_dir.is_dir():
                continue
            osc_name = entry.get("osc", "osc_data")
            out.append((Z, ion_charge, ion_dir, osc_name))
    return out


def match_levels_to_cmfgen(carsus_sub: pd.DataFrame, cmfgen_levels: np.ndarray):
    """{global_idx -> cmfgen config string} by (energy, g) matching."""
    cmf_E = cmfgen_levels["E_cm"] * CM2EV
    cmf_g = cmfgen_levels["g"]
    cmf_cfg = cmfgen_levels["config"]
    used = np.zeros(len(cmf_E), dtype=bool)
    out = {}
    for _, row in carsus_sub.iterrows():
        E = float(row.energy_eV); g = float(row.g)
        dE = np.abs(cmf_E - E)
        cand = np.where((dE < E_MATCH_TOL_EV) & (np.abs(cmf_g - g) < 0.5)
                        & (~used))[0]
        if cand.size == 0:
            cand = np.where((dE < E_MATCH_TOL_EV) & (~used))[0]
        if cand.size == 0:
            continue
        k = cand[int(np.argmin(dE[cand]))]
        used[k] = True
        out[int(row.global_idx)] = str(cmf_cfg[k])
    return out


def main() -> None:
    log(f"levels : {LEVELS_CSV}")
    log(f"config : {CONFIG_YML}")
    log(f"output : {OUT_CSV}")

    carsus = pd.read_csv(LEVELS_CSV)
    carsus["global_idx"] = np.arange(len(carsus))
    n_levels = len(carsus)

    rows = []           # (Z, ion, level_number, multiplicity)
    cov = {}            # (Z,ion) -> (matched, parsed, total)
    for (Z, ion_csv, date_dir, osc_name) in load_yml_ions():
        sub = carsus[(carsus.atomic_number == Z)
                     & (carsus.ion_number == ion_csv)].copy()
        if sub.empty:
            continue
        osc_p = date_dir / osc_name
        if not osc_p.exists():
            continue
        try:
            osc = parse_osc(osc_p)
        except Exception as e:
            log(f"  Z={Z} ion={ion_csv}: osc parse FAIL: {e}")
            continue
        if osc.n_levels == 0:
            continue
        gi_to_cfg = match_levels_to_cmfgen(sub, osc.levels)
        nparsed = 0
        for gi, cfg in gi_to_cfg.items():
            m = parse_mult(cfg)
            if m > 0:
                nparsed += 1
            lnum = int(carsus.iloc[gi].level_number)
            rows.append((Z, ion_csv, lnum, m))
        cov[(Z, ion_csv)] = (len(gi_to_cfg), nparsed, len(sub))

    df = pd.DataFrame(rows, columns=["atomic_number", "ion_number",
                                     "level_number", "multiplicity"])
    df = df.sort_values(["atomic_number", "ion_number", "level_number"])
    df.to_csv(OUT_CSV, index=False)

    n_nonzero = int((df.multiplicity > 0).sum())
    log(f"wrote {len(df)} rows ({n_nonzero} with known multiplicity) / "
        f"{n_levels} total levels -> {OUT_CSV}")

    # coverage table for the report (the diagnostic + analog ions)
    print("\n  per-ion coverage (matched / parsed>0 / total levels):")
    name = {14: "Si", 16: "S", 20: "Ca", 26: "Fe", 27: "Co", 28: "Ni"}
    for (Z, ion) in sorted(cov):
        if Z not in name:
            continue
        mt, pa, tot = cov[(Z, ion)]
        print(f"    {name[Z]} {ROMAN[ion+1]:4s}: matched {mt:4d}/{tot:<4d} "
              f"parsed {pa:4d} ({100.0*pa/max(tot,1):5.1f}% of ion)")


if __name__ == "__main__":
    main()
