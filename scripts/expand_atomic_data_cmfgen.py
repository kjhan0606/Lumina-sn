#!/usr/bin/env python3
"""Convert CMFGEN atomic data to LUMINA's CSV/HDF5 schema.

For each (Z, ion) in the SN Ia critical list:
  - parse osc_data         → levels + lines
  - parse phot_data_A      → photoionization sigma_bf(nu) per level
  - parse col_data         → Omega(T) per transition
Apply per-ion level caps (GPU memory tractability) before emission.

Outputs:
  data/tardis_reference_cmfgen/
    levels.csv, line_list.csv,
    macro_atom_data.csv, macro_atom_references.csv,
    ionization_energies.csv, atom_masses.csv, abundances.csv
  data/atomic/atomic_data_cmfgen.h5
    /Z{ZZ}_ion{N}/phot/L{LLLL}/{cs_type, n_points, energy?, sigma_Mb}
    /Z{ZZ}_ion{N}/col/{T_grid_kK, level_pairs, omega}

Designed to be loaded by a (forthcoming) CMFGEN-aware lumina_atomic.c path.
The CSV trio is fully drop-in compatible with the existing TARDIS-derived data.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmfgen_parser import parse_osc, parse_phot, parse_col, parse_f_to_s  # noqa: E402

ROOT        = Path('/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn')
CMFGEN_ROOT = ROOT / 'data' / 'atomic' / 'cmfgen'
import os
_OUT_SUFFIX = os.environ.get('CMFGEN_OUT_SUFFIX', '')
OUT_DIR     = ROOT / 'data' / f'tardis_reference_cmfgen{_OUT_SUFFIX}'
OUT_H5      = ROOT / 'data' / 'atomic' / f'atomic_data_cmfgen{_OUT_SUFFIX}.h5'

CMFGEN_DIRS = {
    1:'HYD', 2:'HE', 6:'CARB', 7:'NIT', 8:'OXY', 10:'NEON', 11:'NA',
    12:'MG', 13:'AL', 14:'SIL', 15:'PHOS', 16:'SUL', 17:'CHL', 18:'ARG',
    19:'POT', 20:'CA', 21:'SCAN', 22:'TIT', 23:'VAN', 24:'CHRO',
    25:'MAN', 26:'FE', 27:'COB', 28:'NICK',
}
ROMAN = ['', 'I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII']
SYM = {6:'C',7:'N',8:'O',10:'Ne',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',
       16:'S',17:'Cl',18:'Ar',19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',
       24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni'}

ATOM_MASS_AMU = {
    6:12.0107, 8:15.999, 12:24.305, 13:26.9815385, 14:28.0855,
    16:32.065, 20:40.078, 21:44.955908, 22:47.867, 23:50.9415,
    24:51.9961, 25:54.938044, 26:55.845, 27:58.933194, 28:58.6934,
}

# Level caps tuned for GPU memory.  Fe II 500x500 rate matrix x 30 shells x
# FP64 ~= 60 MB; full 2698 levels would be 1.7 GB just for one ion.
# `None` keeps the full CMFGEN level list.
ION_LEVEL_CAPS: dict[tuple[int,int], int | None] = {
    # (Z, stage 1-based): cap (None = full)
    # V4 (#219b): caps raised for iron-peak II/III to recover line counts lost
    # in V3 build.  GPU NLTE rate matrices stay well under H100 96 GB even at
    # the doubled / tripled caps (Co III 1500^2 x 30 x 8 = 540 MB; total over
    # all ions ~3 GB).
    (6, 1):None, (6, 2):None, (6, 3):None,
    (8, 1):None, (8, 2):None, (8, 3):None,
    # #24 ionfix (2026-06-01): add the missing TOP ion stage for elements whose
    # nebular-Saha dominant stage at SN-ejecta shell conditions lies ABOVE the
    # previous ceiling (over-populated lower ion -> spurious opacity). Sc/Mg are
    # trapped at II across ALL shells (4700A carriers); Al/Ti/Cr trapped in outer
    # shells only. CMFGEN tracks all these stages, so this = faithfulness.
    (12, 1):None, (12, 2):None, (12, 3):None,      # +Mg IV(III idx) MG/III 201 lvl
    (13, 1): 80, (13, 2): 80, (13, 3): 80, (13, 4):None,   # +Al IV AL/IV 201 lvl
    (14, 1):None, (14, 2):None, (14, 3):None, (14, 4):None,
    (16, 1):None, (16, 2):None, (16, 3):None,
    (20, 1):None, (20, 2):None, (20, 3): 200,
    (21, 1): 60, (21, 2): 500, (21, 3):None,       # +Sc III SCAN/III 87 lvl
    (22, 1): 200, (22, 2): 600, (22, 3): 600, (22, 4):None, # +Ti IV TIT/IV 126 lvl
    (23, 1): 60, (23, 2): 200,
    # #219g (2026-05-25): iron-peak III caps raised to full CMFGEN level lists
    # (probe whether 800-cap was masking HST UV[1700,3000]=0.297 floor).
    # Fe III 1500, Co III 3917 (None=full), Cr/Mn/Ni III 1000 (None=full).
    (24, 1): 200, (24, 2): 600, (24, 3):None, (24, 4): 200, # +Cr IV CHRO/IV cap 200
    (25, 1): 60, (25, 2): 600, (25, 3):None,
    (26, 1): 300, (26, 2): 800, (26, 3):None, (26, 4): 200, (26, 5): 200,
    (27, 1): 200, (27, 2):1500, (27, 3):None, (27, 4): 200,
    (28, 1): 800, (28, 2): 800, (28, 3):None, (28, 4): 200,
}

# CMFGEN super-level scheme (XzV_F_TO_S): full levels (FL) carry line opacity,
# collapsed to super-levels (SL) for the NLTE statistical-equilibrium solve;
# FL populations are distributed within an SL by Boltzmann at the local T.
# This is CMFGEN's *actual published method* (Hillier & Miller 1998) and is
# Phase-1-faithful (matching the paper's map, NOT a tuning knob).  When enabled
# (env CMFGEN_SUPER_LEVELS=1) the listed ions keep ALL full levels (no lowest-N
# truncation) and attach the f_to_s super-level index per level.  All other
# ions get the identity map (each FL is its own SL) so the downstream solver
# reproduces current behaviour for them.
# Value = f_to_s filename inside the chosen date dir.
SUPER_LEVEL_ENABLED = os.environ.get('CMFGEN_SUPER_LEVELS', '0') not in ('0', '', 'false')
SUPER_LEVEL_IONS: dict[tuple[int, int], str] = {
    (26, 2): 'f_to_s_342',   # Fe II  2698 FL -> 342 SL (E%LS 1%)
    (27, 2): 'f_to_s_252',   # Co II  2747 FL -> 252 SL (E%LS 1.5%)
    (28, 2): 'f_to_s_88',    # Ni II  1000 FL ->  88 SL
    # level-cap sweep (2026-06-04): lift the last capped iron-peak II ions from
    # the 600 cap to full 1000 FL via super-levels (Fe/Co/Ni II were already
    # full). Closes the level-cap hypothesis for the macroatom over-redshift.
    (22, 2): 'f_to_s_92',    # Ti II  1000 FL ->  92 SL
    (24, 2): 'f_to_s_84',    # Cr II  1000 FL ->  84 SL
    (25, 2): 'f_to_s_92',    # Mn II  1000 FL ->  92 SL
}

N_SHELLS = 30
DEFAULT_ABUNDANCES = {
    6:0.02, 8:0.0394, 12:0.005, 13:0.0005, 14:0.1, 16:0.05,
    20:0.05, 21:0.00003, 22:0.0003, 23:0.0002, 24:0.003,
    25:0.0015, 26:0.5, 27:0.05, 28:0.13,
}

C_CGS  = 2.99792458e10
H_CGS  = 6.62607015e-27
K_CGS  = 1.380649e-16
EV_TO_ERG = 1.602176634e-12

# Pre-bake target grid: must match NLTE_NU_{MIN,MAX} / NLTE_N_FREQ_BINS in
# src/lumina.h.  If those constants change, regenerate the binary.
BF_NU_MIN     = 1.5e14   # c / 20000 A
BF_NU_MAX     = 3.0e16   # c / 100 A
BF_N_FREQ_BIN = 1000
OUT_SIGMA_BIN = ROOT / 'data' / 'atomic' / f'cmfgen_sigma_bf{_OUT_SUFFIX}.bin'
_DATE_RE = re.compile(r'^\d{1,2}[a-z]{3}\d{2}$')
_MONTHS = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
           'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}


def _pick_latest(ion_dir: Path) -> Path | None:
    dates = [d for d in ion_dir.iterdir()
             if d.is_dir() and _DATE_RE.match(d.name)]
    if not dates:
        return None
    def key(d: Path) -> tuple[int, int, int]:
        m = re.match(r'^(\d{1,2})([a-z]{3})(\d{2})$', d.name)
        if not m: return (0, 0, 0)
        day, mon, yy = m.groups()
        yr = int(yy); yr = 2000 + yr if yr < 50 else 1900 + yr
        return (yr, _MONTHS.get(mon, 0), int(day))
    return max(dates, key=key)


def _phot_path(date_dir: Path) -> Path | None:
    for nm in ('phot_data_A', 'phot_data_a', 'phot_data_B',
               'phot_data_b', 'phot_data'):
        p = date_dir / nm
        if p.exists(): return p
    cands = list(date_dir.glob('phot_data*'))
    return cands[0] if cands else None


def _norm_cfg(s) -> str:
    return str(s).strip().lower()


_TERM_RE = re.compile(r'\[[^\[\]]*\]\s*$')

def _term_cfg(s) -> str:
    """Strip the trailing [J/2] (or [J]) bracket — phot_data groups J-levels."""
    return _TERM_RE.sub('', _norm_cfg(s))


def parse_all_ions() -> dict:
    """Parse every (Z, stage) listed in ION_LEVEL_CAPS that has CMFGEN data."""
    print("=== Phase 1: parse CMFGEN ions ===")
    out: dict[tuple[int,int], dict] = {}
    for (Z, stage), cap in sorted(ION_LEVEL_CAPS.items()):
        eldir = CMFGEN_DIRS.get(Z)
        if eldir is None: continue
        ion_dir = CMFGEN_ROOT / eldir / ROMAN[stage]
        if not ion_dir.is_dir():
            print(f"  miss {SYM[Z]:2s} {ROMAN[stage]:4s}: dir absent")
            continue
        date_dir = _pick_latest(ion_dir)
        if date_dir is None:
            print(f"  miss {SYM[Z]:2s} {ROMAN[stage]:4s}: no date subdir")
            continue
        osc_p = date_dir / 'osc_data'
        if not osc_p.exists():
            cands = list(date_dir.glob('*osc*'))
            if not cands: continue
            osc_p = cands[0]

        try:
            osc = parse_osc(osc_p)
        except Exception as e:
            print(f"  ERR {SYM[Z]:2s} {ROMAN[stage]:4s} osc: {e}")
            continue
        if osc.n_levels == 0:
            continue

        # Super-level ions keep ALL full levels (FL carry opacity); the NLTE
        # solve later collapses them to f_to_s super-levels.
        ftos = None
        use_superlev = SUPER_LEVEL_ENABLED and (Z, stage) in SUPER_LEVEL_IONS
        if use_superlev:
            fts_name = SUPER_LEVEL_IONS[(Z, stage)]
            fts_p = date_dir / fts_name
            if not fts_p.exists():
                print(f"  WARN {SYM[Z]:2s} {ROMAN[stage]:4s}: f_to_s "
                      f"{fts_name} absent in {date_dir.name}; using cap fallback")
                use_superlev = False
            else:
                try:
                    ftos = parse_f_to_s(fts_p)
                except Exception as e:
                    print(f"  WARN {SYM[Z]:2s} {ROMAN[stage]:4s} f_to_s: {e}")
                    use_superlev = False
                else:
                    if ftos.n_levels != osc.n_levels:
                        print(f"  WARN {SYM[Z]:2s} {ROMAN[stage]:4s}: f_to_s "
                              f"n_levels {ftos.n_levels} != osc {osc.n_levels}; "
                              f"using cap fallback")
                        ftos = None
                        use_superlev = False

        if use_superlev:
            n_kept = osc.n_levels   # keep everything for opacity
        else:
            n_kept = osc.n_levels if cap is None else min(osc.n_levels, cap)
        levels = osc.levels[:n_kept]
        # CMFGEN encodes Kurucz-computed (theoretical, no observed lambda)
        # transitions with negative lam_A. |lam_A| is the correct rest
        # wavelength; the previous `> 0.0` filter dropped 86%/99% of
        # Fe III / Co III bb network and starved the rate matrix (#266).
        tmask = (osc.transitions['i'] >= 1) & (osc.transitions['j'] >= 1) & \
                (osc.transitions['i'] <= n_kept) & \
                (osc.transitions['j'] <= n_kept) & \
                (osc.transitions['lam_A'] != 0.0)
        trans = osc.transitions[tmask].copy()
        trans['lam_A'] = np.abs(trans['lam_A'])

        phot = None
        pp = _phot_path(date_dir)
        if pp is not None:
            try: phot = parse_phot(pp)
            except Exception as e:
                print(f"  WARN {SYM[Z]:2s} {ROMAN[stage]:4s} phot: {e}")

        col = None
        cp = date_dir / 'col_data'
        if cp.exists():
            try: col = parse_col(cp)
            except Exception as e:
                print(f"  WARN {SYM[Z]:2s} {ROMAN[stage]:4s} col: {e}")

        out[(Z, stage)] = dict(osc=osc, levels=levels, trans=trans,
                               phot=phot, col=col, date=date_dir.name,
                               n_kept=n_kept, ftos=ftos)
        sl_tag = f" SL={ftos.n_super}" if ftos is not None else ""
        print(f"  {SYM[Z]:2s} {ROMAN[stage]:4s}: "
              f"{n_kept:4d}/{osc.n_levels:5d} lev{sl_tag}, "
              f"{len(trans):6d}/{osc.n_transitions:6d} trn, "
              f"phot={'Y' if phot and phot.entries else '-'} "
              f"col={'Y' if col and col.entries else '-'}  ({date_dir.name})")
    return out


def build_global_levels(ion_data: dict):
    """Emit per-ion level metadata into a flat global table.

    Returns
    -------
    levels_rows : list of (Z, ion_csv, lvl_csv, E_eV, g, metastable, super_level)
    level_lookup_global : dict (Z, stage, cmfgen_id_1based) -> global_idx
    per_ion_g : dict (Z, stage) -> ndarray of g[lvl_csv]

    super_level is a per-ion 0-based super-level index.  For CMFGEN super-level
    ions it is the f_to_s assignment; for all other ions it equals lvl_csv
    (identity map: each full level is its own super level).
    """
    rows = []
    lookup = {}
    per_ion_g = {}
    n_total = 0
    for (Z, stage), d in sorted(ion_data.items()):
        ion_csv = stage - 1
        levs = d['levels']
        n = len(levs)
        ftos = d.get('ftos')
        gs = np.empty(n, dtype='f8')
        # Bug #2 fix: physical metastable = no allowed downward E1 transition.
        # CMFGEN d['trans'] lists only E1 lines; sum A_ul where this level is upper.
        # Lucy 2002 §4 definition: A_rad_sum == 0 ⇒ metastable (n_lower=n_meta, weight=1).
        A_rad_sum = np.zeros(n, dtype='f8')
        for t in d['trans']:
            i = int(t['i']); j = int(t['j'])
            if i > j: i, j = j, i
            upper_k = j - 1
            if 0 <= upper_k < n:
                A_rad_sum[upper_k] += float(t['A'])
        for k in range(n):
            cmfgen_id = int(levs['ID'][k])     # 1-based
            E_cm = float(levs['E_cm'][k])
            # CMFGEN column 'E_eV' is the photoionization threshold (E_ion-E_level).
            # LUMINA needs level energy from ground: E_cm in cm^-1 -> eV.
            E_eV = E_cm * 1.239841984e-4
            g_val = float(levs['g'][k])
            gs[k] = g_val
            metastable = 1 if A_rad_sum[k] == 0.0 else 0
            if ftos is not None:
                super_level = int(ftos.sl_of_fl[cmfgen_id - 1])
            else:
                super_level = k   # identity: each full level is its own SL
            rows.append((Z, ion_csv, k, E_eV, int(round(g_val)), metastable,
                         super_level))
            lookup[(Z, stage, cmfgen_id)] = n_total
            n_total += 1
        per_ion_g[(Z, stage)] = gs
    return rows, lookup, per_ion_g


def build_lines(ion_data: dict, level_lookup: dict, per_ion_g: dict):
    """Collect all transitions and sort by descending nu."""
    Zs, ions, los, ups, lams, fs, As = [], [], [], [], [], [], []
    for (Z, stage), d in sorted(ion_data.items()):
        ion_csv = stage - 1
        gs = per_ion_g[(Z, stage)]
        for t in d['trans']:
            i = int(t['i']); j = int(t['j'])
            if i > j: i, j = j, i           # ensure lower < upper
            if i < 1 or j > d['n_kept']:    continue
            if (Z, stage, i) not in level_lookup: continue
            if (Z, stage, j) not in level_lookup: continue
            Zs.append(Z); ions.append(ion_csv)
            los.append(i - 1); ups.append(j - 1)
            lams.append(float(t['lam_A']))
            fs.append(float(t['f']))   # f_lu (absorption oscillator)
            As.append(float(t['A']))   # A_ul (s^-1)

    Z_arr   = np.array(Zs, dtype='i4')
    ion_arr = np.array(ions, dtype='i4')
    lo_arr  = np.array(los, dtype='i4')
    up_arr  = np.array(ups, dtype='i4')
    lam_arr = np.array(lams, dtype='f8')
    f_arr   = np.array(fs, dtype='f8')
    A_arr   = np.array(As, dtype='f8')
    nu_arr  = np.where(lam_arr > 0, C_CGS / (lam_arr * 1e-8), 0.0)

    valid = (nu_arr > 0) & (lam_arr > 0)
    Z_arr   = Z_arr[valid];   ion_arr = ion_arr[valid]
    lo_arr  = lo_arr[valid];  up_arr  = up_arr[valid]
    lam_arr = lam_arr[valid]; f_arr   = f_arr[valid]
    A_arr   = A_arr[valid];   nu_arr  = nu_arr[valid]

    order = np.argsort(-nu_arr)
    return dict(Z=Z_arr[order], ion=ion_arr[order],
                lo=lo_arr[order], up=up_arr[order],
                lam=lam_arr[order], f_lu=f_arr[order],
                A_ul=A_arr[order], nu=nu_arr[order])


def write_levels_csv(rows, path: Path) -> None:
    with open(path, 'w') as fp:
        fp.write("atomic_number,ion_number,level_number,energy_eV,g,metastable,"
                 "super_level\n")
        for r in rows:
            fp.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.10f},{r[4]},{r[5]},{r[6]}\n")


def write_line_list_csv(L, levels_rows, path: Path) -> None:
    n = L['Z'].size
    # Build per-ion (Z, ion_csv, lvl_csv) -> g lookup directly from levels_rows
    g_map = {}
    for r in levels_rows:
        g_map[(r[0], r[1], r[2])] = float(r[4])
    g_lo = np.array([g_map[(int(L['Z'][i]), int(L['ion'][i]), int(L['lo'][i]))]
                     for i in range(n)], dtype='f8')
    g_up = np.array([g_map[(int(L['Z'][i]), int(L['ion'][i]), int(L['up'][i]))]
                     for i in range(n)], dtype='f8')
    f_lu = L['f_lu']
    f_ul = f_lu * g_lo / g_up
    nu   = L['nu']
    A_ul = L['A_ul']
    B_lu = A_ul * (C_CGS ** 2) / (8 * np.pi * H_CGS * (nu ** 3)) * (g_up / g_lo)
    B_ul = B_lu * g_lo / g_up
    wl_cm = L['lam'] * 1e-8
    with open(path, 'w') as fp:
        fp.write("atomic_number,ion_number,level_number_lower,level_number_upper,"
                 "line_id,wavelength,f_ul,f_lu,nu,B_lu,B_ul,A_ul,wavelength_cm\n")
        for i in range(n):
            fp.write(
                f"{int(L['Z'][i])},{int(L['ion'][i])},"
                f"{int(L['lo'][i])},{int(L['up'][i])},{i},"
                f"{L['lam'][i]:.6f},{f_ul[i]:.6e},{f_lu[i]:.6e},"
                f"{nu[i]:.6e},{B_lu[i]:.6e},{B_ul[i]:.6e},{A_ul[i]:.6e},"
                f"{wl_cm[i]:.10e}\n"
            )


def write_macro_atom(L, levels_rows, level_lookup_global,
                     ma_path: Path, mr_path: Path) -> tuple[int, int]:
    """Emit macro_atom_data.csv + macro_atom_references.csv.

    Per CMFGEN line we generate three macro-atom transitions:
      (-1) emission        : src=upper, dst=lower (line photon)
      ( 0) internal-down   : src=upper, dst=lower (no photon)
      ( 1) internal-up     : src=lower, dst=upper (excitation)
    Static transition_probability = A_ul-weighted radiative rates at reference
    photospheric conditions (NOT flat 1/N): emit & internal-down ∝ A_ul,
    internal-up ∝ B_lu·W·B_ν(T_rad). finalize_cmfgen_ref_npy normalizes each
    source-level block to Σp=1. This is the static fallback used when the
    per-shell dynamic recompute (LUMINA_DYNAMIC_TRANSPROB) is off; the old flat
    1/N over-weighted internal transitions (red-pileup placeholder, #257).
    Reference: T_rad_ref/W_ref are representative photospheric values; the live
    path overrides per shell when dynamic transprob is enabled.
    """
    T_RAD_REF = 10000.0   # K, ≈ converged DDC15 T_inner
    W_REF     = 0.5        # dilute factor near photosphere

    n_levels = len(levels_rows)
    n_lines  = L['Z'].size

    # global-idx of lower/upper for every line
    glo = np.empty(n_lines, dtype=np.int64)
    gup = np.empty(n_lines, dtype=np.int64)
    for i in range(n_lines):
        Z = int(L['Z'][i]); ion_csv = int(L['ion'][i])
        glo[i] = level_lookup_global[(Z, ion_csv + 1, int(L['lo'][i]) + 1)]
        gup[i] = level_lookup_global[(Z, ion_csv + 1, int(L['up'][i]) + 1)]

    # Per-line radiative-rate weights (relative; finalize normalizes per block).
    g_map = {}
    for r in levels_rows:
        g_map[(r[0], r[1], r[2])] = float(r[4])
    g_lo = np.array([g_map[(int(L['Z'][i]), int(L['ion'][i]), int(L['lo'][i]))]
                     for i in range(n_lines)], dtype='f8')
    g_up = np.array([g_map[(int(L['Z'][i]), int(L['ion'][i]), int(L['up'][i]))]
                     for i in range(n_lines)], dtype='f8')
    nu_l  = L['nu'].astype('f8')
    A_ul  = L['A_ul'].astype('f8')
    B_lu  = A_ul * (C_CGS ** 2) / (8 * np.pi * H_CGS * (nu_l ** 3)) * (g_up / g_lo)
    x = H_CGS * nu_l / (K_CGS * T_RAD_REF)
    B_nu = (2 * H_CGS * nu_l ** 3 / C_CGS ** 2) / np.expm1(np.clip(x, 1e-30, 700.0))
    w_emit = A_ul                    # ttype -1  (β folded into per-block norm)
    w_idn  = A_ul                    # ttype  0
    w_iup  = B_lu * W_REF * B_nu     # ttype +1

    # Bucket transitions by source level: (line_id, ttype, dst_global, weight)
    src_buckets: list[list[tuple[int, int, int, float]]] = [[] for _ in range(n_levels)]
    # tuple = (line_id, ttype, dst_global_idx)
    for i in range(n_lines):
        src_buckets[gup[i]].append((i, -1, int(glo[i]), float(w_emit[i])))  # emission
        src_buckets[gup[i]].append((i,  0, int(glo[i]), float(w_idn[i])))   # internal-down
        src_buckets[glo[i]].append((i,  1, int(gup[i]), float(w_iup[i])))   # internal-up

    n_trans = sum(len(b) for b in src_buckets)
    count_down  = np.zeros(n_levels, dtype=np.int64)
    count_up    = np.zeros(n_levels, dtype=np.int64)
    count_total = np.zeros(n_levels, dtype=np.int64)
    block_refs  = np.zeros(n_levels, dtype=np.int64)
    cum = 0
    for lvl in range(n_levels):
        b = src_buckets[lvl]
        for (_, ttype, _, _) in b:
            if ttype == 1: count_up[lvl] += 1
            else:          count_down[lvl] += 1
        count_total[lvl] = len(b)
        block_refs[lvl] = cum
        cum += len(b)

    with open(ma_path, 'w') as fp:
        fp.write(",atomic_number,ion_number,source_level_number,"
                 "destination_level_number,transition_type,"
                 "transition_probability,transition_line_id,"
                 "lines_idx,destination_level_idx,source_level_idx\n")
        idx = 0
        for lvl in range(n_levels):
            r = levels_rows[lvl]
            Z, ion_csv, src_lvl_csv = r[0], r[1], r[2]
            b = src_buckets[lvl]
            for (line_id, ttype, dst_global, weight) in b:
                # destination level_number_csv = (Z, ion_csv) row 2 of dst
                d_row = levels_rows[dst_global]
                dst_lvl_csv = d_row[2]
                fp.write(f"{idx},{Z},{ion_csv},{src_lvl_csv},{dst_lvl_csv},"
                         f"{ttype},{weight:.10e},{line_id},{line_id},"
                         f"{dst_global},{lvl}\n")
                idx += 1

    with open(mr_path, 'w') as fp:
        fp.write("atomic_number,ion_number,source_level_number,"
                 "count_down,count_up,count_total,"
                 "block_references,references_idx\n")
        for lvl in range(n_levels):
            r = levels_rows[lvl]
            fp.write(f"{r[0]},{r[1]},{r[2]},"
                     f"{count_down[lvl]},{count_up[lvl]},{count_total[lvl]},"
                     f"{block_refs[lvl]},{lvl}\n")
    return n_trans, n_levels


def write_ionization_csv(ion_data: dict, path: Path) -> int:
    rows = sorted(((Z, stage - 1, d['osc'].ionization_eV)
                   for (Z, stage), d in ion_data.items()),
                  key=lambda r: (r[0], r[1]))
    with open(path, 'w') as fp:
        fp.write("atomic_number,ion_number,ionization_energy_eV\n")
        for Z, ion_csv, E in rows:
            fp.write(f"{Z},{ion_csv},{E:.10f}\n")
    return len(rows)


def write_atom_masses(elements_used, path: Path) -> None:
    with open(path, 'w') as fp:
        fp.write("atomic_number,mass_amu\n")
        for Z in elements_used:
            if Z in ATOM_MASS_AMU:
                fp.write(f"{Z},{ATOM_MASS_AMU[Z]:.10f}\n")


def write_abundances(elements_used, path: Path) -> None:
    with open(path, 'w') as fp:
        header = "atomic_number," + ",".join(str(i) for i in range(N_SHELLS))
        fp.write(header + "\n")
        for Z in elements_used:
            x = DEFAULT_ABUNDANCES.get(Z, 0.0)
            vals = ",".join(f"{x}" for _ in range(N_SHELLS))
            fp.write(f"{Z},{vals}\n")


def bake_sigma_bf_grid(ion_data: dict, ion_data_index: dict,
                       levels_rows, level_lookup, path: Path) -> tuple[int, int]:
    """Pre-bake sigma_bf(nu) onto the fixed LUMINA bf opacity grid.

    Output binary layout (little-endian):
      uint32 magic        = 0x434D4644  ('CMFD')
      uint32 version      = 1
      int32  n_levels
      int32  n_freq_bins
      double nu_min_Hz, nu_max_Hz
      int8   has_cmfgen[n_levels]   (1 if level has CMFGEN data, 0 else)
      pad to 8-byte align
      double sigma_cm2[n_levels * n_freq_bins]   (row-major; level-major)

    Levels without CMFGEN data have sigma_cm2 == 0 across the grid; the C-side
    bf opacity loop falls back to Kramers for those.
    """
    n_levels = len(levels_rows)
    nb = BF_N_FREQ_BIN
    log_min = np.log(BF_NU_MIN)
    log_max = np.log(BF_NU_MAX)
    d_log_nu = (log_max - log_min) / nb
    # Bin centers (matches lumina_plasma.c: nu_bin[b] = nu_min * exp((b+0.5)*dlog))
    nu_grid = BF_NU_MIN * np.exp((np.arange(nb) + 0.5) * d_log_nu)

    sigma_grid = np.zeros((n_levels, nb), dtype='f8')
    has_cmfgen = np.zeros(n_levels, dtype='i1')

    # Build (Z, ion_csv, level_num_csv) -> ionization_eV map for thresholds
    ion_E = {(Z, stage - 1): d['osc'].ionization_eV
             for (Z, stage), d in ion_data.items()}

    n_baked = 0
    for (Z, stage), d in ion_data.items():
        ion_csv = stage - 1
        levs = d['levels']
        n_kept = d['n_kept']
        E_ion = d['osc'].ionization_eV  # eV
        # Level-config -> level_num_csv map (J-resolved + term-level)
        cfg_to_lvl: dict[str, int] = {}
        term_to_lvls: dict[str, list[int]] = {}
        for k in range(n_kept):
            cfg_to_lvl[_norm_cfg(levs['config'][k])] = k
            term_to_lvls.setdefault(_term_cfg(levs['config'][k]), []).append(k)

        phot = d['phot']
        if phot is None or not phot.entries:
            continue

        for entry in phot.entries:
            cfg_norm = _norm_cfg(entry.config)
            if cfg_norm in cfg_to_lvl:
                targets = [cfg_to_lvl[cfg_norm]]
            else:
                targets = term_to_lvls.get(_term_cfg(entry.config), [])
            if not targets:
                continue

            # Two evaluation paths:
            #   - Tabulated (20-22): interpolate (E_ratio, sigma_Mb) pairs.
            #   - Modified Seaton (type 1): sigma = sigma_0 * [beta + (1-beta)/y]
            #         * y^(-s)  where y = nu/nu_threshold and parameters are
            #         [sigma_0_Mb, beta, s].  Hillier & Miller 1998 eq.7.
            #   - Other fit types (2,3,7,8,9): use the first parameter as
            #         sigma_0 and apply the Kramers form (nu_0/nu)^3.  This
            #         is an approximation that preserves order-of-magnitude;
            #         a fully faithful implementation requires CMFGEN's
            #         sub_phot_gen.f for each type.
            for lvl_csv in targets:
                E_level_eV = float(levs['E_cm'][lvl_csv]) * 1.239841984e-4
                E_thresh   = E_ion - E_level_eV
                if E_thresh <= 0:
                    continue
                nu_thresh = E_thresh * EV_TO_ERG / H_CGS
                idx = np.searchsorted(nu_grid, nu_thresh)
                if idx >= nb:
                    continue
                # When baking against a ref-merged level table, the ref may
                # have fewer levels for this ion than CMFGEN provides; skip
                # CMFGEN levels that don't exist in the ref.
                gidx = level_lookup.get((Z, stage, lvl_csv + 1))
                if gidx is None:
                    continue

                if entry.cs_type in (20, 21, 22):
                    if entry.energy.size == 0 or entry.sigma_Mb.size == 0:
                        continue
                    nu_pts = entry.energy * nu_thresh
                    sigma  = entry.sigma_Mb * 1.0e-18
                    sigma_on_grid = np.interp(nu_grid[idx:], nu_pts, sigma,
                                              left=0.0, right=sigma[-1])
                    sigma_grid[gidx, idx:] = sigma_on_grid
                    has_cmfgen[gidx] = 1
                    n_baked += 1
                elif entry.cs_type == 1 and entry.sigma_Mb.size >= 3:
                    sigma_0 = float(entry.sigma_Mb[0])  # Mb
                    beta    = float(entry.sigma_Mb[1])
                    s_exp   = float(entry.sigma_Mb[2])
                    y = nu_grid[idx:] / nu_thresh        # y >= 1
                    sigma = sigma_0 * (beta + (1.0 - beta) / y) * (y ** (-s_exp))
                    sigma_grid[gidx, idx:] = sigma * 1.0e-18
                    has_cmfgen[gidx] = 1
                    n_baked += 1
                elif entry.cs_type in (2, 3, 7, 8, 9) and entry.sigma_Mb.size >= 1:
                    # Approximate as Kramers using the first parameter.
                    sigma_0 = float(entry.sigma_Mb[0]) * 1.0e-18
                    if sigma_0 <= 0:
                        continue
                    y = nu_grid[idx:] / nu_thresh
                    sigma_grid[gidx, idx:] = sigma_0 / (y ** 3)
                    has_cmfgen[gidx] = 1
                    n_baked += 1

    # Write flat binary
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        import struct
        f.write(struct.pack('<II', 0x434D4644, 1))           # magic, version
        f.write(struct.pack('<ii', n_levels, nb))
        f.write(struct.pack('<dd', BF_NU_MIN, BF_NU_MAX))
        f.write(has_cmfgen.tobytes())
        # Pad to 8-byte alignment
        pad = (8 - (n_levels % 8)) % 8
        f.write(b'\x00' * pad)
        f.write(sigma_grid.tobytes(order='C'))

    return n_baked, int(has_cmfgen.sum())


def write_phot_col_h5(ion_data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as h5:
        for (Z, stage), d in sorted(ion_data.items()):
            ion_csv = stage - 1
            grp = h5.create_group(f"Z{Z:02d}_ion{ion_csv}")
            grp.attrs['ionization_eV'] = d['osc'].ionization_eV
            grp.attrs['n_levels'] = d['n_kept']
            grp.attrs['z_screen'] = d['osc'].z_screen

            # Direct (J-resolved) lookup: osc config string -> level idx.
            cfg_to_lvl = {_norm_cfg(d['levels']['config'][k]): k
                          for k in range(len(d['levels']))}
            # Term lookup: term-level config (no [J/2] suffix) -> list of level idx.
            term_to_lvls: dict[str, list[int]] = {}
            for k in range(len(d['levels'])):
                term = _term_cfg(d['levels']['config'][k])
                term_to_lvls.setdefault(term, []).append(k)

            phot = d['phot']
            if phot is not None and phot.entries:
                pgrp = grp.create_group('phot')
                pgrp.attrs['final_state'] = phot.final_state.encode()
                pgrp.attrs['final_state_excit_eV'] = phot.final_state_excit_eV
                pgrp.attrs['final_state_g'] = phot.final_state_g
                matched = 0
                shared = 0
                for entry in phot.entries:
                    cfg_norm = _norm_cfg(entry.config)
                    targets: list[int] = []
                    if cfg_norm in cfg_to_lvl:
                        targets = [cfg_to_lvl[cfg_norm]]
                    else:
                        term = _term_cfg(entry.config)
                        targets = term_to_lvls.get(term, [])
                        if len(targets) > 1:
                            shared += 1
                    for lvl_csv in targets:
                        ds_name = f"L{lvl_csv:04d}"
                        if ds_name in pgrp:
                            continue
                        egrp = pgrp.create_group(ds_name)
                        egrp.attrs['cs_type']  = entry.cs_type
                        egrp.attrs['n_points'] = entry.n_points
                        if entry.energy.size:
                            egrp.create_dataset('energy', data=entry.energy)
                        egrp.create_dataset('sigma_Mb', data=entry.sigma_Mb)
                        matched += 1
                pgrp.attrs['n_matched'] = matched
                pgrp.attrs['n_shared'] = shared

            col = d['col']
            if col is not None and col.entries:
                cgrp = grp.create_group('col')
                cgrp.create_dataset('T_grid_kK', data=col.T_grid_kK)
                cgrp.attrs['n_T'] = col.n_T
                cgrp.attrs['scale_factor'] = col.scale_factor
                pairs = []
                omegas = []
                for cfg_l, cfg_u, om in col.entries:
                    cl = _norm_cfg(cfg_l); cu = _norm_cfg(cfg_u)
                    if cl in cfg_to_lvl and cu in cfg_to_lvl:
                        pairs.append((cfg_to_lvl[cl], cfg_to_lvl[cu]))
                        omegas.append(om)
                if pairs:
                    cgrp.create_dataset('level_pairs',
                                        data=np.asarray(pairs, dtype='i4'))
                    cgrp.create_dataset('omega',
                                        data=np.asarray(omegas, dtype='f8'))
                cgrp.attrs['n_matched'] = len(pairs)


def copy_zeta_files() -> None:
    """Copy zeta_*.{csv,npy} from existing tardis_reference (TARDIS-specific
    Saha-Boltzmann corrections; not provided by CMFGEN). Falls back to a unit
    zeta table if the source dir is missing."""
    import shutil
    src_dir = ROOT / 'data' / 'tardis_reference'
    files = ['zeta_data.npy', 'zeta_ions.csv', 'zeta_temps.csv']
    for f in files:
        s = src_dir / f
        d = OUT_DIR / f
        if s.exists():
            shutil.copy(s, d)
        else:
            print(f"  WARN missing source zeta file {s}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ion_data = parse_all_ions()
    if not ion_data:
        print("No ions parsed; aborting.")
        sys.exit(1)

    print(f"\n=== Phase 2: build global tables ({len(ion_data)} ions) ===")
    levels_rows, level_lookup, per_ion_g = build_global_levels(ion_data)
    print(f"  global levels: {len(levels_rows)}")

    L = build_lines(ion_data, level_lookup, per_ion_g)
    print(f"  global lines:  {L['Z'].size}")

    print("\n=== Phase 3: write CSVs ===")
    write_levels_csv(levels_rows, OUT_DIR / 'levels.csv')
    print(f"  levels.csv: {len(levels_rows)} rows")
    write_line_list_csv(L, levels_rows, OUT_DIR / 'line_list.csv')
    print(f"  line_list.csv: {L['Z'].size} rows")
    n_trans, n_lvls = write_macro_atom(
        L, levels_rows, level_lookup,
        OUT_DIR / 'macro_atom_data.csv',
        OUT_DIR / 'macro_atom_references.csv',
    )
    print(f"  macro_atom_data.csv: {n_trans} rows")
    print(f"  macro_atom_references.csv: {n_lvls} rows")
    n_ions = write_ionization_csv(ion_data, OUT_DIR / 'ionization_energies.csv')
    print(f"  ionization_energies.csv: {n_ions} rows")
    elements_used = sorted({Z for (Z, _) in ion_data.keys()})
    write_atom_masses(elements_used, OUT_DIR / 'atom_masses.csv')
    write_abundances(elements_used, OUT_DIR / 'abundances.csv')
    print(f"  atom_masses.csv + abundances.csv ({len(elements_used)} Z)")
    copy_zeta_files()
    print("  zeta_*.{csv,npy} copied from tardis_reference")

    print("\n=== Phase 4: write HDF5 phot/col (browsable) ===")
    write_phot_col_h5(ion_data, OUT_H5)
    print(f"  wrote {OUT_H5}")

    print("\n=== Phase 5: bake sigma_bf grid for C loader ===")
    n_baked, n_levels_with_data = bake_sigma_bf_grid(
        ion_data, ion_data, levels_rows, level_lookup, OUT_SIGMA_BIN
    )
    print(f"  baked {n_baked} curves -> {n_levels_with_data}/{len(levels_rows)} "
          f"levels covered ({100*n_levels_with_data/len(levels_rows):.1f}%)")
    print(f"  wrote {OUT_SIGMA_BIN} "
          f"({len(levels_rows)} x {BF_N_FREQ_BIN} doubles = "
          f"{len(levels_rows)*BF_N_FREQ_BIN*8 / (1024*1024):.0f} MB)")

    print("\n" + "=" * 64)
    print(f"SUMMARY  {len(ion_data)} ions  "
          f"{len(levels_rows):,} lev  {L['Z'].size:,} lines  "
          f"{n_trans:,} macro-atom trans")
    print(f"CSVs:  {OUT_DIR}")
    print(f"HDF5:  {OUT_H5}")
    print("=" * 64)


if __name__ == '__main__':
    main()
