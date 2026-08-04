#!/usr/bin/env python3
"""Parse AUTOSTRUCTURE IC olg output → levels.csv + line_list.csv per ion.

Reads the IC level table ('K  LV  T  K*CM  2*S+1  L  2J  CF  (EK-E1)/RY')
and the IC E1-DATA block that follows it.  Writes per-ion CSV files in the
capraise schema (atomic_number, ion_number, level_number, energy_eV, g,
metastable) and (atomic_number, ion_number, level_number_lower,
level_number_upper, line_id, wavelength, f_ul, f_lu, nu, B_lu, B_ul, A_ul,
wavelength_cm).

Level indices in the AS output start at 1 (level K=1 is ground).  The
output CSV uses level_number = K-1 so the ground state is level 0
(matches capraise convention).

Usage:
    parse_as_olg_levels_bb.py <olg> <Z> <ion_number> <out_levels.csv>
                              <out_lines.csv>

ion_number is 0-indexed (neutral=0, singly ionized=1, doubly ionized=2,
…).  This matches the capraise schema.
"""

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RYD_TO_EV   = 13.605693122994  # eV per Rydberg
HC_EVA      = 12398.41984       # h c in eV·Angstrom
C_CGS       = 2.99792458e10     # cm/s
H_CGS       = 6.62607015e-27    # erg·s
M_E_CGS     = 9.1093837015e-28  # g
E_CGS       = 4.80320451e-10    # esu (statcoulomb)
PI          = np.pi


def parse_ic_levels(olg_path: Path):
    """Yield (K, T_cm, twoS_plus_1, L, twoJ, CF, EK_minus_E1_Ryd, E_eV) per
    IC level.  Reads the LAST occurrence of the level table (which is the
    IC level-level block, not the term block).
    """
    with olg_path.open() as f:
        text = f.read()
    hdr_re = re.compile(
        r'^\s*K\s+LV\s+T\s+K\*CM\s+2\*S\+1\s+L\s+2J\s+CF\s+\(EK-E1\)/RY',
        re.MULTILINE)
    matches = list(hdr_re.finditer(text))
    if not matches:
        raise RuntimeError(f"No IC level table found in {olg_path}")
    m = matches[-1]
    # Skip the rest of the header line.
    nl = text.find('\n', m.end())
    start = nl + 1 if nl >= 0 else m.end()
    body = text[start:start + 5_000_000]
    rows = []
    started = False
    for line in body.splitlines():
        line = line.rstrip()
        if not line.strip():
            if started:
                break
            continue
        toks = line.split()
        if len(toks) < 9:
            if started:
                break
            continue
        try:
            K = int(toks[0])
            LV = int(toks[1])
            T = float(toks[2])
            kcm = float(toks[3])
            twoS1 = int(toks[4])
            L = int(toks[5])
            twoJ = int(toks[6])
            CF = int(toks[7])
            EK = float(toks[8])
        except (ValueError, IndexError):
            if started:
                break
            continue
        started = True
        rows.append((K, kcm, twoS1, L, twoJ, CF, EK, EK * RYD_TO_EV))
    return rows


def parse_e1_block(olg_path: Path, nlevels: int):
    """Yield (K_upper, KP_lower, |A_ul|, wavelength_A) per E1 transition.

    AS dumps the E1 table after the IC level table.  Some rows have signed
    A values (phase); we take the absolute value.  Wavelengths come straight
    from the WAVEL/AE column.
    """
    with olg_path.open() as f:
        text = f.read()
    e1_hdr_re = re.compile(
        r'^\s*E1-DATA\s+K\s+KP\s+A\(EK\)\*SEC',
        re.MULTILINE)
    matches = list(e1_hdr_re.finditer(text))
    if not matches:
        raise RuntimeError(f"No IC E1-DATA block found in {olg_path}")
    m = matches[-1]
    nl = text.find('\n', m.end())
    start = nl + 1 if nl >= 0 else m.end()
    body = text[start:]
    rows = []
    started = False
    for line in body.splitlines():
        if not line.strip():
            if started:
                # E1 block has internal blanks between K groups — keep going.
                continue
            continue
        toks = line.split()
        # Expected: idx K KP A(EK)*SEC S G*F F(ABS) -F(EMI) WAVEL GF(VEL) ALPHA(POL)
        if len(toks) < 9:
            if started:
                break
            continue
        try:
            K = int(toks[1])
            KP = int(toks[2])
            A_signed = float(toks[3])
            wavel = float(toks[8])
        except (ValueError, IndexError):
            if started:
                # Skip non-numeric continuation lines (e.g., headers between K
                # groups).  Only stop if we've already started AND see clearly
                # non-numeric content.
                continue
            continue
        if K < 1 or K > nlevels or KP < 1 or KP > nlevels:
            if started:
                break
            continue
        started = True
        rows.append((K, KP, abs(A_signed), abs(wavel)))
    return rows


def main():
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)
    olg = Path(sys.argv[1])
    Z = int(sys.argv[2])
    ion = int(sys.argv[3])
    out_levels = Path(sys.argv[4])
    out_lines = Path(sys.argv[5])

    print(f"[parse] olg={olg}")
    print(f"[parse] Z={Z} ion={ion}")

    lv_rows = parse_ic_levels(olg)
    nL = len(lv_rows)
    print(f"[parse] IC levels: {nL}")

    if nL == 0:
        raise RuntimeError("no levels parsed")

    lv = pd.DataFrame(lv_rows,
        columns=['K', 'K_cm', 'twoS1', 'L', 'twoJ', 'CF', 'EK_Ry', 'energy_eV'])
    lv['level_number'] = lv['K'] - 1
    lv['g'] = lv['twoJ'] + 1
    lv['metastable'] = 0
    lv.loc[0, 'metastable'] = 1  # ground = metastable by convention

    out_lv = lv[['level_number', 'energy_eV', 'g', 'metastable']].copy()
    out_lv.insert(0, 'atomic_number', Z)
    out_lv.insert(1, 'ion_number', ion)
    out_levels.parent.mkdir(parents=True, exist_ok=True)
    out_lv.to_csv(out_levels, index=False)
    print(f"[write] {out_levels}: {len(out_lv)} levels, "
          f"E_max={out_lv.energy_eV.max():.3f} eV")

    e1_rows = parse_e1_block(olg, nL)
    print(f"[parse] E1 transitions (raw): {len(e1_rows)}")

    # K = upper, KP = lower (per AS convention in level-level block)
    e1 = pd.DataFrame(e1_rows, columns=['K_upper', 'KP_lower', 'A_ul', 'wavelength'])

    # Build per-K g and energy
    g_per_K = {row.K: row.twoJ + 1 for row in lv.itertuples(index=False)}
    E_per_K = {row.K: row.energy_eV for row in lv.itertuples(index=False)}

    e1['level_number_lower'] = e1['KP_lower'] - 1
    e1['level_number_upper'] = e1['K_upper']  - 1
    e1['g_l'] = e1['KP_lower'].map(g_per_K)
    e1['g_u'] = e1['K_upper' ].map(g_per_K)
    e1['E_l_eV'] = e1['KP_lower'].map(E_per_K)
    e1['E_u_eV'] = e1['K_upper' ].map(E_per_K)
    e1['dE_eV']  = e1['E_u_eV'] - e1['E_l_eV']

    # Drop downward-wrong rows (KP must be lower energy)
    mask = e1['dE_eV'] > 1e-6
    e1 = e1[mask].copy()
    print(f"[parse] E1 transitions (upper→lower, dE>0): {len(e1)}")

    # Use energy-derived wavelength (vacuum, Å) — more reliable than AS's column
    e1['wavelength_A'] = HC_EVA / e1['dE_eV']
    e1['wavelength_cm'] = e1['wavelength_A'] * 1e-8
    e1['nu'] = C_CGS / e1['wavelength_cm']

    # f_lu (oscillator strength, absorption) from A_ul:
    #   A_ul = (8 π² e² ν² / (m c³)) (g_l / g_u) f_lu
    pref = (8.0 * PI**2 * E_CGS**2) / (M_E_CGS * C_CGS**3)
    e1['f_lu'] = e1['A_ul'] / (pref * e1['nu']**2 * (e1['g_l'] / e1['g_u']))
    e1['f_ul'] = -(e1['g_l'] / e1['g_u']) * e1['f_lu']

    # Einstein B coefficients:
    #   B_lu = (π e² / (m c h ν)) f_lu  (cgs, per unit J_ν)
    Bpref = (PI * E_CGS**2) / (M_E_CGS * C_CGS * H_CGS)
    e1['B_lu'] = Bpref * e1['f_lu'] / e1['nu']
    e1['B_ul'] = (e1['g_l'] / e1['g_u']) * e1['B_lu']

    # Drop tiny / zero / non-finite A
    keep = (e1['A_ul'] > 0) & np.isfinite(e1['A_ul']) & np.isfinite(e1['f_lu'])
    e1 = e1[keep].copy()
    e1 = e1.sort_values(['level_number_lower', 'level_number_upper']).reset_index(drop=True)
    e1['line_id'] = np.arange(len(e1))

    out_ll = pd.DataFrame({
        'atomic_number': Z,
        'ion_number': ion,
        'level_number_lower': e1['level_number_lower'],
        'level_number_upper': e1['level_number_upper'],
        'line_id': e1['line_id'],
        'wavelength': e1['wavelength_A'],
        'f_ul': e1['f_ul'],
        'f_lu': e1['f_lu'],
        'nu': e1['nu'],
        'B_lu': e1['B_lu'],
        'B_ul': e1['B_ul'],
        'A_ul': e1['A_ul'],
        'wavelength_cm': e1['wavelength_cm'],
    })
    out_ll.to_csv(out_lines, index=False)
    print(f"[write] {out_lines}: {len(out_ll)} lines")
    print(f"[summary] λ range: {e1['wavelength_A'].min():.1f}–{e1['wavelength_A'].max():.1f} Å")
    print(f"[summary] A_ul range: {e1['A_ul'].min():.2e}–{e1['A_ul'].max():.2e} s^-1")


if __name__ == '__main__':
    main()
