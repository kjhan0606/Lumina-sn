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
    ionization_energies.csv, atom_masses.csv
  data/atomic/atomic_data_cmfgen.h5
    /Z{ZZ}_ion{N}/phot/L{LLLL}/{cs_type, n_points, energy?, sigma_Mb}
    /Z{ZZ}_ion{N}/col/{T_grid_kK, level_pairs, omega}

Designed to be loaded by a (forthcoming) CMFGEN-aware lumina_atomic.c path.
The CSV trio is fully drop-in compatible with the existing TARDIS-derived data.
"""

from __future__ import annotations

import csv
import re
import shlex
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
    # 2026-07-28: the four ions the campaign has been missing since 2026-07-19.
    # Without them S IV/V and Ca IV/V are level-less destination rungs, so
    # LUMINA_SIMUL_CAP_TOPION truncates the ladder with r:=0 and their population
    # is identically zero. Measured consequence against the published CMFGEN
    # toy06 @19.48d: Ca III sits at exactly 1.0000 in every shell where the truth
    # has 2.5-16% Ca IV, and the cap breaks the energy ledger (photo-heating is
    # counted against the element while the fb cooling term is identically zero
    # because nion[upper] == 0). CMFGEN carries all four with osc_data +
    # phot_data_A + col_data in 19apr23, so importing them is faithfulness, not
    # an extension. Level counts available: S IV 194, S V 307, Ca IV 378,
    # Ca V 613. S IV/Ca IV take the full list -- a 100-level cap was measured to
    # drop 73% of the S IV lines. The new top rungs (S VI / Ca VI) stay absent,
    # which is where the cap belongs: those are negligible at SN-ejecta
    # conditions. Ladder headroom checked first: sum(npop-1) 49 -> 53, SIM_MAXP
    # is 96, so no element gets silently dropped (plasma.c:8315 does that with no
    # warning, and it would eat this very repair on a larger reference).
    (16, 1):None, (16, 2):None, (16, 3):None, (16, 4):None, (16, 5): 200,
    (20, 1):None, (20, 2):None, (20, 3): 200, (20, 4):None, (20, 5): 200,
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

# --------------------------------------------------------------------------
# GATE: off by default.  CMFGEN_FULL_LEVELS=1 repairs the two LUMINA-atom data
# defects the coverage-extension certification measured
# (validation/cmfgen_toy06_19p48d/analysis/rates_certification/coverage_extension
# /REPORT.md, item 3 "[데이터] LUMINA 원자 준위 절단" and item 7):
#
#  (A) LEVEL TRUNCATION.  Six ions carry a 200-level cap while the published
#      CMFGEN toy06 @19.48d reference run models many more.  The missing levels
#      hold ~0 population but carry 20-95 % of Gamma: measured D/C at s0 is
#      Ni IV 0.4723, Fe IV 0.773, Fe V 0.207 -- a defect the population-share
#      yardstick cannot see, only the rate yardstick.
#  (B) ABSENT IONS.  Six ions the reference run carries have zero rows in
#      levels.csv, so layer D is structurally 0 for them.
#
# Caps below = the reference run's own NF, read from
# /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL_SPEC (<ion>_ISF third field),
# NOT guessed:  SkV 203, FeSIX 2000, CoV 1000, CoSIX 1000, NkV 1000, NkSIX 1000,
# NkIV 1000, CoIV 1000, FeIV 1000, FeV 1000, CaV 528, CaIII 232.
# min(osc.n_levels, cap) still applies, so an ion whose LUMINA-side osc file is
# shorter than NF simply takes its full list.
#
# STRUCTURE-CHANGING by design: levels.csv, line_list.csv, macro_atom_*.csv and
# the sigma bin all grow.  Gate OFF reproduces the shipped bake bit for bit
# (the dict is not touched at all).
FULL_LEVELS = os.environ.get('CMFGEN_FULL_LEVELS', '0') not in ('0', '', 'false')

# (A) truncated ions: 200 -> reference NF
FULL_LEVEL_CAPS_RAISE: dict[tuple[int, int], int] = {
    (20, 3):  232,   # Ca III  CaIII_ISF   200 -> 232
    (20, 5):  528,   # Ca V    CaV_ISF     200 -> 528
    (26, 4): 1000,   # Fe IV   FeIV_ISF    200 -> 1000
    (26, 5): 1000,   # Fe V    FeV_ISF     200 -> 1000
    (27, 4): 1000,   # Co IV   CoIV_ISF    200 -> 1000
    (28, 4): 1000,   # Ni IV   NkIV_ISF    200 -> 1000
}
# (B) ions with no rows at all: added at the reference NF
FULL_LEVEL_CAPS_NEW: dict[tuple[int, int], int] = {
    (14, 5):  203,   # Si V    SkV_ISF
    (26, 6): 2000,   # Fe VI   FeSIX_ISF
    (27, 5): 1000,   # Co V    CoV_ISF
    (27, 6): 1000,   # Co VI   CoSIX_ISF
    (28, 5): 1000,   # Ni V    NkV_ISF
    (28, 6): 1000,   # Ni VI   NkSIX_ISF
}
if FULL_LEVELS:
    ION_LEVEL_CAPS.update(FULL_LEVEL_CAPS_RAISE)
    ION_LEVEL_CAPS.update(FULL_LEVEL_CAPS_NEW)

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
# R4 identity gate: consume every f_to_s source selected by CMFGEN_LINKS, not
# only the three legacy iron-group-II entries below.  Default OFF is a strict
# compatibility requirement for the existing _links deck bake.
LINK_FTOS_ENABLED = os.environ.get('CMFGEN_LINK_FTOS', '0').strip().lower() \
    not in ('0', '', 'false')
SUPER_LEVEL_IONS: dict[tuple[int, int], str] = {
    (26, 2): 'f_to_s_342',   # Fe II  2698 FL -> 342 SL (E%LS 1%)
    (27, 2): 'f_to_s_252',   # Co II  2747 FL -> 252 SL (E%LS 1.5%)
    (28, 2): 'f_to_s_88',    # Ni II  1000 FL ->  88 SL
    # level-cap sweep (2026-06-04): lift the last capped iron-peak II ions from
    # the 600 cap to full 1000 FL via super-levels (Fe/Co/Ni II were already
    # full). Closes the level-cap hypothesis for the macroatom over-redshift.
    # 2026-07-28 HELD BACK for single-variable control: these three were added to
    # the dict on 2026-06-04 but the reference in use
    # (tardis_reference_cmfgen_superlev_ionfix_ddc15strat) predates them, so
    # enabling them here would ride along with the S IV/V + Ca IV/V import and
    # make the A/B two-variable (they lift Ti/Cr/Mn II from the 600 cap to the
    # full 1000 FL -- a separate, judgeable change). Re-enable and judge on its
    # own once the four-ion import is settled.
    # (22, 2): 'f_to_s_92',    # Ti II  1000 FL ->  92 SL
    # (24, 2): 'f_to_s_84',    # Cr II  1000 FL ->  84 SL
    # (25, 2): 'f_to_s_92',    # Mn II  1000 FL ->  92 SL
}

# --------------------------------------------------------------------------
# GATE: off by default.  CMFGEN_VINTAGE_MATCH=1 makes the baker read, for every
# ion the published CMFGEN toy06 @19.48d certification run carries, the SAME
# osc/phot files that run actually used, instead of _pick_latest.  Ions that run
# does not carry keep _pick_latest.  With the gate off this file reproduces the
# shipped bake bit for bit.
#
# Provenance (2026-07-29): read off the reference run's own symlinks,
#   /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/{<Sp>_F_OSCDAT, PHOT<Sp>_A}
#     -> /gpfs/kjhan/cmfgen_21jun23/atomic/<EL>/<ion>/<vintage>/<file>
# and the file names are given explicitly, NOT re-derived by globbing: several
# of these vintages carry more than one candidate (FE/IV/18oct00 alone has
# feiv_osc{,_rev,_rev2,_rev3}.dat and phot_data_tst.dat, which is what
# _phot_path's `phot_data*` glob would have picked).
#
# NOT switched by this gate:
#   * col_data -- keeps reading from the _pick_latest dir.  The reference run's
#     collision files are `col_guess.dat` stubs (0 parsed entries) for most of
#     these ions, so switching them would delete collision data; the gate's
#     variable is the photoionization data.
#   * f_to_s for SUPER_LEVEL_IONS, except where the configured file does not
#     exist in the matched vintage (4th tuple slot; see the two entries below).
#
# (Z, stage) -> (date_dir, osc file, phot file, f_to_s override or None)
TOY06_VINTAGE: dict[tuple[int, int], tuple[str, str, str, str | None]] = {
    (14, 2): ('19apr23', 'osc_data',         'phot_data_A',      None),
    (14, 3): ('19apr23', 'osc_data',         'phot_data_A',      None),
    (14, 4): ('5dec96',  'osc_op_split.dat', 'phot_op.dat',      None),
    (16, 2): ('19apr23', 'osc_data',         'phot_data_A',      None),
    (16, 3): ('3oct00',  'siiiosc_fin.dat',  'phot_sm_3000.dat', None),
    (16, 4): ('3oct00',  'sivosc_fin.dat',   'phot_sm_3000.dat', None),
    (16, 5): ('3oct00',  'svosc_fin.dat',    'phot_sm_3000.dat', None),
    (20, 2): ('19apr23', 'osc_data',         'phot_data_A',      None),
    (20, 3): ('10apr99', 'osc_op_sp.dat',    'phot_smooth.dat',  None),
    (20, 4): ('10apr99', 'osc_op_sp.dat',    'phot_smooth.dat',  None),
    (20, 5): ('10apr99', 'osc_op_sp.dat',    'phot_smooth.dat',  None),
    (26, 2): ('19apr23', 'osc_data',         'phot_data_A',      None),
    (26, 3): ('19apr23', 'osc_data',         'phot_data_A',      None),
    (26, 4): ('18oct00', 'feiv_osc.dat',     'phot_sm_3000.dat', None),
    (26, 5): ('18oct00', 'fev_osc.dat',      'phot_sm_3000.dat', None),
    # Co II 18oct00 has no 252-super-level map (that is a 19apr23-only file);
    # f_to_s_55.dat is the one the reference run itself used for this atom.
    (27, 2): ('18oct00', 'coii_osc.dat',     'phot_data.dat',    'f_to_s_55.dat'),
    (27, 3): ('18oct00', 'coiii_osc.dat',    'phot_data.dat',    None),
    (27, 4): ('18oct00', 'coiv_osc.dat',     'phot_data.dat',    None),
    # same 88-super-level grouping as SUPER_LEVEL_IONS, just the .dat spelling.
    (28, 2): ('18oct00', 'nkii_osc.dat',     'phot_data.dat',    'f_to_s_88.dat'),
    (28, 3): ('18oct00', 'nkiii_osc.dat',    'phot_data.dat',    None),
    (28, 4): ('18oct00', 'nkiv_osc.dat',     'phot_data.dat',    None),
}
#
# CMFGEN_VINTAGE_MATCH=1     -- FULL match (osc+phot) for all 21 ions above.
#                               Changes the model atom: Co II 2747->1000 and
#                               Co III 3917->1000 levels, -44% of the line list.
#                               Diagnostic only (see bakefix3).
# CMFGEN_VINTAGE_MATCH=phot  -- RESTRICTED match: swap ONLY the phot file, and
#                               only for VINTAGE_PHOT_IONS.  levels/lines stay
#                               on _pick_latest, so levels.csv / line_list.csv /
#                               macro_atom_*.csv / ionization_energies.csv are
#                               byte-identical to the gate-off bake by
#                               construction (none of them reads phot data).
_VM_ENV = os.environ.get('CMFGEN_VINTAGE_MATCH', '0').strip().lower()
VINTAGE_MATCH = _VM_ENV not in ('0', '', 'false')
VINTAGE_PHOT_ONLY = _VM_ENV == 'phot'

# Ions admitted to the restricted set.  Membership is measured, not assumed:
# an ion is in only if the reference-vintage phot file covers EVERY emitted
# level that the current phot file covers (no level may lose sigma), where
# "covers" applies the same config/term name match and the same
# E_thresh > 0 / nu_thresh < BF_NU_MAX filters the baker itself uses.
# Measured 2026-07-29 (lost -> excluded):
#   Si IV  66/66 emitted levels lost -- 5dec96 uses a completely different
#          config naming convention (osc_op_split vs 19apr23), 0 names match.
#   S V    35 lost -- 3oct00 is a different atom here, not a revision
#          (216 vs 307 levels, max|dE| 7.4e4 cm^-1, g differs on 93 levels).
#   Co IV  1 lost -- level 36 (0-based 35) '3d6_1Se[0]' has no 18oct00 entry.
# For every ion below the emitted-level energies are identical between the two
# vintages (max|dE| = 0, except S IV 4.9e-7 and Fe V 0.8 cm^-1), so the
# phot-only swap reproduces the full-vintage sigma for them.
#
# Measured effect of the swap, over the whole baked grid (INT sigma dnu, summed
# over the ion's emitted levels), bakefix4 / bakefix2:
#   S III 0.9961   S IV 1.0008   Ni II 0.00042
#   Ca III, Ca IV, Ca V, Fe IV, Fe V, Ni III, Ni IV -- ZERO rows changed; five of
#   those seven phot files are md5-IDENTICAL across the two vintages, the other
#   two (Ca IV, Fe IV) differ only outside the baked grid.  For those seven the
#   "vintage mismatch" was a directory label, not data.
# Ni II is the one real regression: 18oct00 is an all-Seaton (type 1, 3
# parameters, sigma_0 ~ 3.4 Mb) table, which 19apr23 replaced with a 2166-point
# tabulated OP table (type 20).  Matching the reference run there means throwing
# away the modern tabulation for a 1/2358 smooth fit, and NO certified ion can
# see it (the harness carries no Ni).  Hence CMFGEN_VINTAGE_PHOT_DROP below.
VINTAGE_PHOT_IONS: set[tuple[int, int]] = {
    (16, 3), (16, 4),                     # S III, S IV      (3oct00)
    (20, 3), (20, 4), (20, 5),            # Ca III, Ca IV, Ca V (10apr99)
    (26, 4), (26, 5),                     # Fe IV, Fe V      (18oct00)
    (28, 2), (28, 3), (28, 4),            # Ni II, Ni III, Ni IV (18oct00)
}
# CMFGEN_VINTAGE_PHOT_DROP='28:2,...' removes ions from the restricted set
# without editing this file, so a per-ion adoption choice stays reproducible
# from one code state.  Empty by default -- no effect on any existing bake.
for _tok in os.environ.get('CMFGEN_VINTAGE_PHOT_DROP', '').replace(' ', '').split(','):
    if _tok:
        _z, _s = _tok.split(':')
        VINTAGE_PHOT_IONS.discard((int(_z), int(_s)))

C_CGS  = 2.99792458e10
H_CGS  = 6.62607015e-27
K_CGS  = 1.380649e-16
EV_TO_ERG = 1.602176634e-12

# I20 수리 (docs/I20_AIR_WAVELENGTH_REPAIR_CONTRACT.md).
# 전하는 SI-2019 정의값에서 유도한 참값을 쓴다: e[C]=1.602176634e-19 (정확),
# c (정확) ⟹ e[esu] = e[C]*c/10.  구값 4.80320425e-10 은 계보 불명이었고
# CMFGEN 의 4.80320427e-10 은 CODATA-2006 이라 둘 다 참값이 아니다.
ME_CGS = 9.1093837015e-28                     # g
E_CGS  = 1.602176634e-19 * C_CGS / 10.0       # = 4.803204712...e-10 esu
# A_ul = A_PREFACTOR * f_lu * (g_lo/g_up) * nu^2   —  genosc_v6.f:313-317 과 동형
A_PREFACTOR = 8.0 * np.pi**2 * E_CGS**2 / (ME_CGS * C_CGS**3)

# Pre-bake target grid: must match NLTE_NU_{MIN,MAX} / NLTE_N_FREQ_BINS in
# src/lumina.h.  If those constants change, regenerate the binary.
BF_NU_MIN     = 1.5e14   # c / 20000 A
BF_NU_MAX     = 3.0e16   # c / 100 A
BF_N_FREQ_BIN = 1000
OUT_SIGMA_BIN = ROOT / 'data' / 'atomic' / f'cmfgen_sigma_bf{_OUT_SUFFIX}.bin'
_DATE_RE = re.compile(r'^\d{1,2}[a-z]{3}\d{2}$')
_MONTHS = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
           'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

CMFGEN_LINKS_PATH = (Path(os.environ['CMFGEN_LINKS']).expanduser()
                     if os.environ.get('CMFGEN_LINKS') else None)
_LINK_KINDS = ('osc', 'f_to_s', 'phot', 'col')


def _atomic_path_identity(path: Path) -> tuple[tuple[int, int], str]:
    """Return ((Z, stage), vintage) encoded in an atomic-tree path."""
    parts = path.parts
    positions = [i for i, part in enumerate(parts) if part == 'atomic']
    if not positions:
        raise ValueError(f"atomic tree component absent from link source: {path}")
    i = positions[-1]
    if i + 1 < len(parts) and parts[i + 1] == 'cmfgen':
        i += 1
    if i + 4 >= len(parts):
        raise ValueError(f"incomplete atomic-tree link source: {path}")
    element, roman, vintage = parts[i + 1:i + 4]
    z_by_dir = {name: z for z, name in CMFGEN_DIRS.items()}
    if element not in z_by_dir or roman not in ROMAN:
        raise ValueError(f"unknown element/ion in link source: {path}")
    if not _DATE_RE.match(vintage):
        raise ValueError(f"invalid vintage directory in link source: {path}")
    return (z_by_dir[element], ROMAN.index(roman)), vintage


def _link_kind(target: str) -> str | None:
    if target.endswith('_F_OSCDAT'):
        return 'osc'
    if target.endswith('_F_TO_S'):
        return 'f_to_s'
    if target.startswith('PHOT') and target.endswith('_A'):
        return 'phot'
    if target.endswith('_COL_DATA'):
        return 'col'
    return None


def load_cmfgen_links(path: Path) -> dict[tuple[int, int], dict[str, Path]]:
    """Parse the four atomic inputs selected by a CMFGEN run.

    Recognised rows are ``ln -sf SOURCE TARGET`` commands.  Every represented
    ion must provide osc, f_to_s, phot and col; a partial or duplicate mapping
    is an error because silently falling back would mix vintages.
    """
    if not path.is_file():
        raise FileNotFoundError(f"CMFGEN_LINKS does not exist: {path}")
    result: dict[tuple[int, int], dict[str, Path]] = {}
    for lineno, line in enumerate(path.read_text(encoding='latin-1').splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        fields = shlex.split(stripped, comments=True)
        if not fields or fields[0] != 'ln':
            continue
        operands = [field for field in fields[1:] if not field.startswith('-')]
        if len(operands) != 2:
            raise ValueError(f"{path}:{lineno}: expected ln SOURCE TARGET")
        source, target = Path(operands[0]), operands[1]
        kind = _link_kind(target)
        if kind is None:
            continue
        key, _vintage = _atomic_path_identity(source)
        slot = result.setdefault(key, {})
        if kind in slot:
            raise ValueError(f"{path}:{lineno}: duplicate {kind} link for {key}")
        slot[kind] = source

    for key, sources in sorted(result.items()):
        missing = [kind for kind in _LINK_KINDS if kind not in sources]
        if missing:
            raise ValueError(f"{path}: incomplete links for {key}; missing {missing}")
        identities = {_atomic_path_identity(source)[0] for source in sources.values()}
        if identities != {key}:
            raise ValueError(f"{path}: cross-ion source set for {key}: {identities}")
        absent = [str(source) for source in sources.values() if not source.is_file()]
        if absent:
            raise FileNotFoundError(f"{path}: linked source files absent for {key}: {absent}")
    return result


if CMFGEN_LINKS_PATH is not None and VINTAGE_MATCH:
    raise RuntimeError("CMFGEN_LINKS and CMFGEN_VINTAGE_MATCH are mutually exclusive")
CMFGEN_LINK_MAP = (load_cmfgen_links(CMFGEN_LINKS_PATH)
                   if CMFGEN_LINKS_PATH is not None else {})
if LINK_FTOS_ENABLED and CMFGEN_LINKS_PATH is None:
    raise RuntimeError("CMFGEN_LINK_FTOS=1 requires CMFGEN_LINKS")


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
        latest_dir = _pick_latest(ion_dir)
        if latest_dir is None:
            print(f"  miss {SYM[Z]:2s} {ROMAN[stage]:4s}: no date subdir")
            continue
        date_dir = latest_dir
        col_dir = latest_dir
        linked = CMFGEN_LINK_MAP.get((Z, stage))
        selection_source = 'links' if linked is not None else 'auto'
        vm = None
        vm_osc = vm_phot = None
        linked_ftos = None
        if linked is not None:
            osc_p = linked['osc']
            linked_ftos = linked['f_to_s']
            pp = linked['phot']
            cp = linked['col']
            date_dir = osc_p.parent
        else:
            vm = TOY06_VINTAGE.get((Z, stage)) if VINTAGE_MATCH else None
            if vm is not None and VINTAGE_PHOT_ONLY and (Z, stage) not in VINTAGE_PHOT_IONS:
                vm = None
            if vm is not None:
                vdir = ion_dir / vm[0]
                if (vdir / vm[1]).exists() and (vdir / vm[2]).exists():
                    vm_phot = vdir / vm[2]
                    if not VINTAGE_PHOT_ONLY:
                        vm_osc = vdir / vm[1]
                        date_dir = vdir
                    selection_source = 'vintage_match'
                else:
                    print(f"  WARN {SYM[Z]:2s} {ROMAN[stage]:4s}: vintage-match "
                          f"{vm[0]} incomplete; keeping {date_dir.name}")
                    vm = None
            osc_p = vm_osc if vm_osc is not None else date_dir / 'osc_data'
            if not osc_p.exists():
                cands = list(date_dir.glob('*osc*'))
                if not cands:
                    continue
                osc_p = cands[0]
            pp = vm_phot if vm_phot is not None else _phot_path(date_dir)
            cp = col_dir / 'col_data'

        try:
            osc = parse_osc(osc_p)
        except Exception as e:
            if linked is not None:
                raise RuntimeError(
                    f"linked osc parse failed for {SYM[Z]} {ROMAN[stage]}: {osc_p}") from e
            print(f"  ERR {SYM[Z]:2s} {ROMAN[stage]:4s} osc: {e}")
            continue
        if osc.n_levels == 0:
            continue

        # Super-level ions keep ALL full levels (FL carry opacity); the NLTE
        # solve later collapses them to f_to_s super-levels.
        ftos = None
        use_link_ftos = LINK_FTOS_ENABLED and linked_ftos is not None
        use_superlev = (use_link_ftos or
                        (SUPER_LEVEL_ENABLED and (Z, stage) in SUPER_LEVEL_IONS))
        if use_superlev:
            if linked_ftos is not None:
                fts_p = linked_ftos
            else:
                fts_name = SUPER_LEVEL_IONS[(Z, stage)]
                if vm_osc is not None and vm[3]:
                    fts_name = vm[3]
                fts_p = date_dir / fts_name
            if not fts_p.exists():
                if linked is not None:
                    raise FileNotFoundError(
                        f"linked f_to_s absent for {SYM[Z]} {ROMAN[stage]}: {fts_p}")
                print(f"  WARN {SYM[Z]:2s} {ROMAN[stage]:4s}: f_to_s "
                      f"{fts_p.name} absent in {date_dir.name}; using cap fallback")
                use_superlev = False
            else:
                try:
                    ftos = parse_f_to_s(fts_p)
                except Exception as e:
                    if linked is not None:
                        raise RuntimeError(
                            f"linked f_to_s parse failed for {SYM[Z]} {ROMAN[stage]}: {fts_p}") from e
                    print(f"  WARN {SYM[Z]:2s} {ROMAN[stage]:4s} f_to_s: {e}")
                    use_superlev = False
                else:
                    if ftos.n_levels != osc.n_levels:
                        if linked is not None:
                            raise RuntimeError(
                                f"linked f_to_s n_levels {ftos.n_levels} != osc "
                                f"{osc.n_levels} for {SYM[Z]} {ROMAN[stage]}")
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
        if pp is not None:
            try:
                phot = parse_phot(pp)
            except Exception as e:
                if linked is not None:
                    raise RuntimeError(
                        f"linked phot parse failed for {SYM[Z]} {ROMAN[stage]}: {pp}") from e
                print(f"  WARN {SYM[Z]:2s} {ROMAN[stage]:4s} phot: {e}")

        col = None
        if cp.exists():
            try:
                col = parse_col(cp)
            except Exception as e:
                if linked is not None:
                    raise RuntimeError(
                        f"linked col parse failed for {SYM[Z]} {ROMAN[stage]}: {cp}") from e
                print(f"  WARN {SYM[Z]:2s} {ROMAN[stage]:4s} col: {e}")

        fts_source = linked_ftos if linked_ftos is not None else (
            fts_p if use_superlev else None)
        provenance = {
            'selection_source': selection_source,
            'latest_vintage': latest_dir.name,
            'osc_path': osc_p,
            'f_to_s_path': fts_source,
            'phot_path': pp,
            'col_path': cp,
        }
        out[(Z, stage)] = dict(osc=osc, levels=levels, trans=trans,
                               phot=phot, col=col, date=date_dir.name,
                               n_kept=n_kept, ftos=ftos,
                               provenance=provenance)
        sl_tag = f" SL={ftos.n_super}" if ftos is not None else ""
        print(f"  {SYM[Z]:2s} {ROMAN[stage]:4s}: "
              f"{n_kept:4d}/{osc.n_levels:5d} lev{sl_tag}, "
              f"{len(trans):6d}/{osc.n_transitions:6d} trn, "
              f"phot={'Y' if phot and phot.entries else '-'} "
              f"col={'Y' if col and col.entries else '-'}  ({date_dir.name}; "
              f"{selection_source})"
              + (f"  VM[{'phot' if VINTAGE_PHOT_ONLY else 'full'}] "
                 f"osc={osc_p.name}({date_dir.name}) "
                 f"phot={pp.name if pp else '-'}({vm[0]}) "
                 f"col<-{col_dir.name}" if vm is not None else ""))
    return out


def build_global_levels(ion_data: dict):
    """Emit per-ion level metadata into a flat global table.

    Returns
    -------
    levels_rows : list of (Z, ion_csv, lvl_csv, E_eV, g, metastable,
                           super_level, configuration)
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
                         super_level, str(levs['config'][k])))
            lookup[(Z, stage, cmfgen_id)] = n_total
            n_total += 1
        per_ion_g[(Z, stage)] = gs
    return rows, lookup, per_ion_g


def build_lines(ion_data: dict, level_lookup: dict, per_ion_g: dict):
    """Collect all transitions and sort by descending nu.

    I20 수리 (docs/I20_AIR_WAVELENGTH_REPAIR_CONTRACT.md):
    CMFGEN osc 파일의 lambda 열은 lambda>2000A 에서 **공기파장**이고 A 열은 f 열과
    ~1e-5 어긋난다.  CMFGEN 자신은 둘 다 읽지 않는다 — genosc_v6.f:278-286 이 f 와
    준위 인덱스만 읽고, :205 에서 nu 를 준위 에너지(진공)로 재계산하며, :313-317 에서
    A 를 f 로부터 만든다.  여기서도 같은 방식으로 산출한다:

        nu   = (E_up - E_lo) * c                     [준위 에너지, 진공]
        f_lu = 원본 osc f 열 (abs — genosc_v6.f:305-309)
        A_ul = A_PREFACTOR * f_lu * (g_lo/g_up) * nu^2

    구현 이전에는 lam <- t['lam_A'](공기), nu <- c/lam, A <- t['A'] 였고,
    그 결과 45/58 이온의 635,169선(덱 28.6%)이 82-85 km/s 어긋나 있었다.
    """
    Zs, ions, los, ups, fs, nus, glos, gups = [], [], [], [], [], [], [], []
    for (Z, stage), d in sorted(ion_data.items()):
        ion_csv = stage - 1
        gs = per_ion_g[(Z, stage)]
        E_cm = d['levels']['E_cm']
        for t in d['trans']:
            i = int(t['i']); j = int(t['j'])
            if i > j: i, j = j, i           # ensure lower < upper
            if i < 1 or j > d['n_kept']:    continue
            if (Z, stage, i) not in level_lookup: continue
            if (Z, stage, j) not in level_lookup: continue
            dE_cm = float(E_cm[j - 1]) - float(E_cm[i - 1])
            if not (dE_cm > 0.0):           # 축퇴/역전은 선이 아니다
                continue
            Zs.append(Z); ions.append(ion_csv)
            los.append(i - 1); ups.append(j - 1)
            fs.append(abs(float(t['f'])))   # f_lu (absorption oscillator)
            nus.append(dE_cm * C_CGS)       # Hz, 진공
            glos.append(float(gs[i - 1])); gups.append(float(gs[j - 1]))

    Z_arr   = np.array(Zs, dtype='i4')
    ion_arr = np.array(ions, dtype='i4')
    lo_arr  = np.array(los, dtype='i4')
    up_arr  = np.array(ups, dtype='i4')
    f_arr   = np.array(fs, dtype='f8')
    nu_arr  = np.array(nus, dtype='f8')
    glo_arr = np.array(glos, dtype='f8')
    gup_arr = np.array(gups, dtype='f8')
    lam_arr = np.where(nu_arr > 0, C_CGS / nu_arr * 1e8, 0.0)   # Angstrom, 진공
    A_arr   = A_PREFACTOR * f_arr * (glo_arr / gup_arr) * nu_arr**2

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
    with open(path, 'w', newline='') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        writer.writerow(("atomic_number", "ion_number", "level_number",
                         "energy_eV", "g", "metastable", "super_level",
                         "configuration"))
        for r in rows:
            writer.writerow((r[0], r[1], r[2], f"{r[3]:.10f}", r[4], r[5],
                             r[6], r[7]))


def write_atomic_vintage_manifest(ion_data: dict, path: Path) -> None:
    """Record the exact per-ion source choice used by this deck bake."""
    fields = [
        'atomic_number', 'ion_stage', 'ion_number', 'ion',
        'selection_source', 'latest_vintage',
        'osc_vintage', 'osc_path', 'f_to_s_vintage', 'f_to_s_path',
        'phot_vintage', 'phot_path', 'col_vintage', 'col_path',
    ]
    if LINK_FTOS_ENABLED:
        fields.extend([
            'f_to_s_format', 'f_to_s_format_basis',
            'f_to_s_declared_full_levels', 'f_to_s_declared_super_levels',
        ])
    with open(path, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for (z, stage), data in sorted(ion_data.items()):
            provenance = data['provenance']
            row = {
                'atomic_number': z,
                'ion_stage': stage,
                'ion_number': stage - 1,
                'ion': f"{SYM[z]} {ROMAN[stage]}",
                'selection_source': provenance['selection_source'],
                'latest_vintage': provenance['latest_vintage'],
            }
            for kind in _LINK_KINDS:
                source = provenance[f'{kind}_path']
                row[f'{kind}_path'] = str(source) if source is not None else ''
                row[f'{kind}_vintage'] = (
                    _atomic_path_identity(source)[1] if source is not None else '')
            if LINK_FTOS_ENABLED:
                ftos = data.get('ftos')
                if ftos is None and provenance['selection_source'] == 'links':
                    raise RuntimeError(
                        f"CMFGEN_LINK_FTOS missing parsed map for {row['ion']}")
                row.update({
                    'f_to_s_format': ftos.format_name if ftos is not None else '',
                    'f_to_s_format_basis': ftos.format_basis if ftos is not None else '',
                    'f_to_s_declared_full_levels': (
                        ftos.n_levels if ftos is not None else ''),
                    'f_to_s_declared_super_levels': (
                        ftos.n_super if ftos is not None else ''),
                })
            writer.writerow(row)


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


# --------------------------------------------------------------------------
# sigma_bf(nu): faithful CMFGEN evaluators + bin-AVERAGED bake
#
# BAKEFIX (2026-07-29), driven by
#   validation/cmfgen_toy06_19p48d/analysis/rates_certification/VERDICT.md
#   sections 3.1 (fit-type stand-in) and 2.2 (point-sampling bias).
#
# (1) The old code lumped CMFGEN fit types 2, 3, 7, 8, 9 into a single
#     sigma = params[0] * 1e-18 * (nu_th/nu)^3 stand-in.  That is wrong in two
#     distinct ways, both measured against CMFGEN's own converged Gamma:
#       * type 7 (modified Seaton, sub_phot_gen.f:505-512) is
#             RU = (nu_th + A3)/nu ;  sigma = 0 identically unless RU <= 1
#         i.e. the edge sits at nu_th + A3, NOT at nu_th.  For the 399 Fe III
#         levels carrying type 7, A3 = 10.2..12.3 (CMFGEN units of 1e15 Hz) =
#         42-50 eV above threshold, so the true edge is near 150-250 A where J
#         is negligible.  The stand-in instead opened a ~1.2 Mb edge at
#         1000-4000 A on every one of them: +2.0% of Gamma(Fe III).
#         params[0] IS a cross-section in Mb for type 7, so the shape was the
#         only defect; it is now evaluated exactly.
#       * types 2, 3, 8 (hydrogenic, sub_phot_gen.f:267/421/308) take
#         params[0] = principal quantum number n (types 2/8) or a scale that
#         multiplies ALPHA_BF/NEF (type 3) -- none of them a cross-section in
#         Mb.  Type 9 is a Verner fit.  Reading n as "sigma_0 in Mb" is a
#         fabricated number.  It is nevertheless STILL USED here, deliberately,
#         and the reason is measured, not assumed:
#           - VERDICT 3.1 scoped this defect as "9 S II levels".  It is not:
#             10876 of the 26592 baked levels take that path, concentrated in
#             exactly the campaign's ions (Co III 3344, Co II 2700, Fe II 1973).
#           - Removing it (has_cmfgen=0, so the C loader's per-ion Kramers
#             fallback owns them) was built and put through the certification:
#             Co III collapses to D/PRRR = 0.111 (gate FAIL) because 88.5 % of
#             the Co III population at s6 then sits on a sigma==0 row.  The
#             fallback is a single per-ION sigma_0 (lumina_plasma.c:6117-6122;
#             2.0 Mb for Co III), not a per-level one, so it cannot stand in
#             for 1441 hydrogenic levels.
#           - Against CMFGEN's own type-8 value at threshold (evaluated from
#             HYD_L_DATA) neither stand-in is right, and which one is less
#             wrong depends on the ion: legacy/CMFGEN vs fallback/CMFGEN =
#             0.85 vs 0.34 (Co III), 0.41 vs 0.53 (Fe II), 0.41 vs 1.03 (Co II),
#             0.09 vs 0.12 (Sc I).  Swapping one for the other is not a repair.
#         So the stand-in stays, LABELLED, until types 2/3/8 are evaluated
#         exactly.  That is feasible and is the real fix: it needs CMFGEN's
#         HYD_L_DATA / GBF_N_DATA (present at
#         /gpfs/kjhan/cmfgen_runs/toy06_19.48d/), whose units were verified
#         here -- 10^BF_L_CROSS(1,0,u=1) * 1e-10 = 6.3034e-18 cm^2 vs the
#         analytic H ground-state 6.30e-18.  Type 2 additionally needs NEF and
#         type 3 needs ALPHA_BF/GBF_N_DATA.  Deliberately NOT done in this
#         change: the certification harness has no exact evaluator for these
#         types either (VERDICT caveat 2), so it would ship untestable.
#
# (2) The old code point-sampled sigma at the geometric bin centre.  Where the
#     CMFGEN table is finer than the 1588 km/s bin (unsmoothed OP data, e.g.
#     S II at 2.0e-3 in dln(nu) vs the grid's 5.3e-3) that biases Gamma HIGH by
#     +1.4%..+6.4%.  sigma is now the bin AVERAGE (1/dnu) INT_bin sigma dnu,
#     which is the quantity the 1000-bin GEMM weight
#     (Gamma = SUM_b sigma_b * Jbar_b * 4pi/(h nu_c,b) * dnu_b) actually wants.
#     The quadrature matches the certification's "Bav" layer node-for-node:
#     nodes = bin edges + the level's own structure nodes + 5 log
#     subdivisions per bin, trapezoid, accumulated into the owning bin.
#
# NOT changed here (VERDICT 3.2, "latent, not live"): the tabulated
# convention sigma=0 below the first table node and sigma=const above the last,
# where CMFGEN uses CROSS_A[1] and CROSS_A[N]*(u_N/u)^3.  Measured effect
# -0.24% (S II s0).  Kept so this bake is a single-variable change.
# --------------------------------------------------------------------------

# Fit types this baker can evaluate exactly (CMFGEN newsubs/sub_phot_gen.f).
_SIGMA_TABULATED = (20, 21, 22)
_SIGMA_EXACT_FIT = (1, 7)
# Hydrogenic (2/3/8) and Verner (9): no exact evaluator offline.  Carried by
# the known-wrong params[0]-as-sigma_0 Kramers stand-in -- see the head note for
# why it is kept rather than removed, and what replacing it would take.
_SIGMA_STANDIN_FIT = (2, 3, 8, 9)

# --------------------------------------------------------------------------
# BAKEFIX2 (2026-07-29) -- EXACT evaluators for CMFGEN fit types 2, 3, 8, 9.
#
# Ported statement-for-statement from newsubs/sub_phot_gen.f (the routine
# CMFGEN itself calls; branch structure, constants and units unchanged):
#     type 2 : lines 267-304   hydrogenic, split l
#     type 8 : lines 308-363   hydrogenic split l, offset edge
#     type 3 : lines 421-451   hydrogenic pure n, bf gaunt factor
#     type 9 : lines 518-530   Verner et al. multi-shell ground-state fit
# and cross-checked line-by-line against spec_plt/subs/raw_subphot_v2.f, which
# carries the same four branches verbatim and ends with
#     PHOT(1:NCF)=1.0E+08*PHOT(1:NCF)    !"Convert from CMFGEN units to MB"
# -- that statement is what fixes the unit convention used below: SUB_PHOT_GEN
# returns sigma in PROGRAM UNITS of 1e-10 cm^2 (= 1e8 Mb ... 1 Mb = 1e-18 cm^2),
# consistent with CONV_FAC = 1.0E-08 (phot_data_mod.f:116) multiplying the
# Megabarn table of types 1/4/5/7/20-22, and with the 1.0E-08 hard-wired into
# the type-9 branch.  So sigma[cm^2] = PHOT[program units] * 1e-10.
#
# HYD_L_DATA / GBF_N_DATA (module HYD_BF_PHOT_DATA, read by
# newsubs/rd_hyd_bf_data.f) are already inside the imported CMFGEN tree at
# cmfgen/HYD/I/5dec96/{hyd_l_data,gbf_n_data}.dat -- md5-identical to
# /gpfs/kjhan/cmfgen_21jun23/atomic/HYD/I/5dec96/ which is what the toy06
# 19.48d run actually symlinks.  Header states "Cross-sctions are in program
# units (i.e. cgs unit*10^10)", i.e. BF_L_CROSS = log10(sigma_cgs * 1e10),
# which is the same 1e-10 convention.
#
# Two per-ion/per-level quantities the fits need, both from
# newsubs/rdphot_gen_v2.f:
#     ZION = ZXzV, the ion's screened nuclear charge (osc_data header;
#            1 for neutral, 3 for Fe III, ...)                    [line 177]
#     NEF  = ZION*SQRT(NU_INF/EDGE), NU_INF = 1e-15*109737.31*c
#            / (1 + 5.48597e-4/(2*AT_NO))   (AT_NO=1: /(1+5.48597e-4))
#                                                              [lines 465-478]
#
# GATE: off by default.  CMFGEN_EXACT_HYD=1 switches types 2/3/8/9 from the
# legacy params[0]-as-sigma_0 stand-in to these evaluators.  With the gate off
# this file reproduces the shipped bakefix bit for bit.
# --------------------------------------------------------------------------
SIGMA_EXACT_HYD = os.environ.get('CMFGEN_EXACT_HYD', '0') not in ('0', '', 'false')

_HYD_BF_DIR = CMFGEN_ROOT / 'HYD' / 'I' / '5dec96'
_HYD_BF: dict | None = None

# rdphot_gen_v2.f:465 -- 1e-15 * R_inf[cm^-1] * c[cm/s], in CMFGEN's 1e15 Hz.
_NU_INF_INF = 1.0e-15 * 109737.31 * C_CGS
_EV_TO_HZ = 0.241798840766          # sub_phot_gen.f:129, eV -> 1e15 Hz


def _cmfgen_nef(Z: int, zion: float, nu_th: float) -> float:
    """NEF for one level, verbatim from rdphot_gen_v2.f:465-478.

    nu_th in Hz (the level's own EDGE(I)+EXC_FREQ(K), which for phot_data_A is
    just the threshold the baker already computes).
    """
    nu_inf = _NU_INF_INF / (1.0 + 5.48597e-04) if Z == 1 else \
             _NU_INF_INF / (1.0 + 5.48597e-04 / (2 * Z))
    t1 = nu_inf / (nu_th * 1.0e-15)
    return zion * np.sqrt(t1) if t1 > 0.0 else 0.0


def _read_hyd_bf_data() -> dict:
    """Port of newsubs/rd_hyd_bf_data.f: HYD_L_DATA + GBF_N_DATA -> arrays.

    Arrays are 1-BASED padded (index 0 unused) so the Fortran index arithmetic
    (BF_L_INDX(N,L), J+1, J-1, ...) transcribes without an off-by-one rewrite.
    """
    def _hdr(lines, i, key):
        while key not in lines[i]:
            i += 1
        return lines[i].split('!')[0].strip(), i + 1

    out: dict = {}

    txt = (_HYD_BF_DIR / 'hyd_l_data.dat').read_text().splitlines()
    v, i = _hdr(txt, 0, 'Maximum principal quantum number'); max_l_pqn = int(v)
    v, i = _hdr(txt, i, 'Number of values');                 n_per_l   = int(v)
    v, i = _hdr(txt, i, 'L_ST_U');                           l_st_u    = _f(v)
    v, i = _hdr(txt, i, 'L_DEL_U');                          l_del_u   = _f(v)
    tok = ' '.join(txt[i:]).split()
    p = 0
    cross = np.zeros(1 + n_per_l * max_l_pqn * (max_l_pqn + 1) // 2, dtype='f8')
    indx = np.zeros((max_l_pqn + 1, max_l_pqn), dtype='i8')
    cnt = 0
    for N in range(1, max_l_pqn + 1):
        for L in range(0, N):
            rd_n, rd_l, npts = int(tok[p]), int(tok[p + 1]), int(tok[p + 2])
            p += 3
            if npts != n_per_l or rd_n != N or rd_l != L:
                raise ValueError(f'hyd_l_data.dat: bad block {rd_n},{rd_l},{npts}')
            for k in range(1, n_per_l + 1):
                cross[cnt + k] = _f(tok[p]); p += 1
            indx[N, L] = cnt + 1
            cnt += n_per_l
    out.update(MAX_L_PQN=max_l_pqn, N_PER_L=n_per_l, L_ST_U=l_st_u,
               L_DEL_U=l_del_u, BF_L_CROSS=cross, BF_L_INDX=indx)

    txt = (_HYD_BF_DIR / 'gbf_n_data.dat').read_text().splitlines()
    v, i = _hdr(txt, 0, 'Maximum principal quantum number'); max_n_pqn = int(v)
    v, i = _hdr(txt, i, 'Number of values');                 n_per_n   = int(v)
    v, i = _hdr(txt, i, 'N_ST_U');                           n_st_u    = _f(v)
    v, i = _hdr(txt, i, 'N_DEL_U');                          n_del_u   = _f(v)
    tok = ' '.join(txt[i:]).split()
    p = 0
    gaunt = np.zeros(1 + n_per_n * max_n_pqn, dtype='f8')
    gindx = np.zeros(max_n_pqn + 1, dtype='i8')
    cnt = 0
    for N in range(1, max_n_pqn + 1):
        rd_n, npts = int(tok[p]), int(tok[p + 1]); p += 2
        if npts != n_per_n or rd_n != N:
            raise ValueError(f'gbf_n_data.dat: bad block {rd_n},{npts}')
        for k in range(1, n_per_n + 1):
            gaunt[cnt + k] = _f(tok[p]); p += 1
        gindx[N] = cnt + 1
        cnt += n_per_n
    out.update(MAX_N_PQN=max_n_pqn, N_PER_N=n_per_n, N_ST_U=n_st_u,
               N_DEL_U=n_del_u, BF_N_GAUNT=gaunt, BF_N_INDX=gindx)
    return out


def _f(s: str) -> float:
    return float(s.replace('D', 'E').replace('d', 'e'))


def _hyd_bf():
    global _HYD_BF
    if _HYD_BF is None:
        _HYD_BF = _read_hyd_bf_data()
    return _HYD_BF


def _hyd_l_block(u: np.ndarray, N: int, LST: int, LEND: int) -> np.ndarray:
    """sub_phot_gen.f:285-304 == 343-360 (identical text in both branches).

    Returns SUM/((LEND-LST+1)*(LEND+LST+1)) in CMFGEN program units, i.e.
    everything except the (NEF/(N*ZION))**2 [type 2] / 1/ZION**2 [type 8] factor.
    Requires u >= 1 (the caller masks on FREQ >= EDGE, as CMFGEN does).
    """
    D = _hyd_bf()
    n_per_l = D['N_PER_L']; l_del_u = D['L_DEL_U']
    XC = D['BF_L_CROSS']; IDX = D['BF_L_INDX']

    X  = np.log10(u)
    RJ = X / l_del_u
    J  = RJ.astype('i8') + 1            # Fortran: J=RJ (truncate) ; J=J+1
    T1 = RJ - (J - 1)
    inside = J < n_per_l                # IF(J .LT. N_PER_L)
    total = np.zeros_like(u)
    for L in range(LST, LEND + 1):
        base = int(IDX[N, L])
        T2 = np.empty_like(u)
        if inside.any():
            JJ = J[inside] + base - 1
            T2[inside] = T1[inside] * XC[JJ + 1] + (1.0 - T1[inside]) * XC[JJ]
        if not inside.all():
            o = ~inside
            JJ = base + n_per_l - 1
            T2[o] = (XC[JJ] - XC[JJ - 1]) * (RJ[o] - n_per_l) + XC[JJ]
        total += (2 * L + 1) * 10.0 ** T2
        # Fortran restores J=RJ+1 at the end of every l iteration, so `J`
        # (and hence `inside`) is the same for every L -- reproduced by not
        # mutating J here.
    return total / ((LEND - LST + 1) * (LEND + LST + 1))


def _gbf_n(u: np.ndarray, N: int) -> np.ndarray:
    """Bound-free gaunt factor, sub_phot_gen.f:425-445."""
    if N > 30:
        return np.ones_like(u)
    D = _hyd_bf()
    n_per_n = D['N_PER_N']; n_del_u = D['N_DEL_U']
    G = D['BF_N_GAUNT']; IDX = D['BF_N_INDX']
    X  = np.log10(u)
    RJ = X / n_del_u
    J  = RJ.astype('i8') + 1
    T1 = RJ - (J - 1)
    base = int(IDX[N])
    out = np.empty_like(u)
    inside = J < n_per_n
    if inside.any():
        JJ = J[inside] + base - 1
        out[inside] = T1[inside] * G[JJ + 1] + (1.0 - T1[inside]) * G[JJ]
    if not inside.all():
        o = ~inside
        JJ = base + n_per_n - 1
        t = np.log10(G[JJ - 1] / G[JJ])
        out[o] = G[JJ] * 10.0 ** (t * (n_per_n - RJ[o]))
    return out


def _sigma_model(cs_type: int, energy: np.ndarray, params: np.ndarray,
                 nu_th: float, zion: float | None = None,
                 nef: float | None = None):
    """Return (sigma_fn, nodes, nu_start, tag) for one CMFGEN phot entry, or None.

    sigma_fn(nu) -> sigma in cm^2 (vectorised);  nodes = abscissae where the
    cross-section has a kink or an edge;  nu_start = lowest frequency at which
    sigma is nonzero (the integral starts there, so a step is never smeared
    across a bin);  tag names the evaluation path actually taken.
    """
    if nu_th <= 0:
        return None

    if cs_type in _SIGMA_TABULATED:
        if energy.size == 0 or params.size == 0:
            return None
        nu_pts = np.asarray(energy, dtype='f8') * nu_th
        sig_pts = np.asarray(params, dtype='f8') * 1.0e-18
        # np.interp needs ascending abscissae; CMFGEN tables are ascending in
        # u (verified: 0/5437 entries non-monotonic across the whole tree).
        if nu_pts.size >= 2 and np.any(np.diff(nu_pts) < 0):
            return None
        lo = max(nu_th, float(nu_pts[0]))   # left=0.0 convention -> start at node 1

        def fn(nu, _p=nu_pts, _s=sig_pts):
            return np.interp(nu, _p, _s, left=0.0, right=_s[-1])

        return fn, nu_pts, lo, 'tab'

    if cs_type == 1 and params.size >= 3:
        # Seaton fit, sub_phot_gen.f:412-419.
        s0, beta, s_exp = (float(params[0]), float(params[1]), float(params[2]))
        if s0 == 0.0:
            return None

        def fn(nu, _s0=s0, _b=beta, _e=s_exp, _n=nu_th):
            out = np.zeros(np.shape(nu))
            m = nu >= _n
            if np.any(m):
                ru = _n / np.asarray(nu)[m]
                out[m] = 1.0e-18 * _s0 * (_b + (1.0 - _b) * ru) * ru ** _e
            return out

        return fn, np.array([nu_th]), nu_th, 'seaton'

    # ---- BAKEFIX2: exact hydrogenic / Verner, gated ----------------------
    if SIGMA_EXACT_HYD and cs_type in _SIGMA_STANDIN_FIT and zion is not None \
            and nef is not None and zion > 0:
        m = _sigma_hydrogenic_exact(cs_type, params, nu_th, zion, nef)
        if m is not None:
            return m
        # falls through to the legacy stand-in when the entry is malformed
        # (n out of the HYD_L_DATA range, l > n-1, wrong parameter count, ...),
        # which is what CMFGEN itself refuses to evaluate (it STOPs).

    if cs_type in _SIGMA_STANDIN_FIT and params.size >= 1:
        # KNOWN-WRONG STAND-IN, kept deliberately (see head note): params[0] is
        # a principal quantum number (2/8) / an ALPHA_BF scale (3) / a Verner
        # parameter (9), NOT a cross-section in Mb.  Kramers shape from nu_th.
        s0 = float(params[0]) * 1.0e-18
        if s0 <= 0.0:
            return None

        def fn(nu, _s0=s0, _n=nu_th):
            out = np.zeros(np.shape(nu))
            m = nu >= _n
            if np.any(m):
                y = np.asarray(nu)[m] / _n
                out[m] = _s0 / (y ** 3)
            return out

        return fn, np.array([nu_th]), nu_th, 'standin'

    if cs_type == 7 and params.size >= 4:
        # Modified Seaton, sub_phot_gen.f:505-512.  A3 is in CMFGEN frequency
        # units of 1e15 Hz and SHIFTS the edge; sigma is identically 0 below it.
        s0, beta, s_exp = (float(params[0]), float(params[1]), float(params[2]))
        if s0 == 0.0:
            return None
        edge = nu_th + float(params[3]) * 1.0e15

        def fn(nu, _s0=s0, _b=beta, _e=s_exp, _E=edge):
            out = np.zeros(np.shape(nu))
            m = nu >= _E
            if np.any(m):
                ru = _E / np.asarray(nu)[m]
                out[m] = 1.0e-18 * _s0 * (_b + (1.0 - _b) * ru) * ru ** _e
            return out

        return fn, np.array([edge]), edge, 'modseaton'

    return None


# Program units -> cm^2 (raw_subphot_v2.f: "Convert from CMFGEN units to MB",
# PHOT*1e8 = Mb = 1e-18 cm^2, so PHOT*1e-10 = cm^2).
_PROG_TO_CM2 = 1.0e-10


def _sigma_hydrogenic_exact(cs_type: int, params: np.ndarray, nu_th: float,
                            zion: float, nef: float):
    """Exact CMFGEN types 2/3/8/9.  Returns (fn, nodes, nu_start, tag) or None."""
    D = _hyd_bf()

    if cs_type == 2 and params.size >= 3:
        # sub_phot_gen.f:267-304.
        N    = int(round(float(params[0])))
        LST  = int(round(float(params[1])))
        LEND = int(round(float(params[2])))
        if not (1 <= N <= D['MAX_L_PQN']) or LST < 0 or LEND < LST \
                or LEND + 1 > N:
            return None                       # CMFGEN STOPs on these
        scale = (nef / (N * zion)) ** 2 * _PROG_TO_CM2

        def fn(nu, _N=N, _a=LST, _b=LEND, _s=scale, _e=nu_th):
            out = np.zeros(np.shape(nu))
            m = np.asarray(nu) >= _e
            if np.any(m):
                out[m] = _s * _hyd_l_block(np.asarray(nu)[m] / _e, _N, _a, _b)
            return out

        return fn, _hyd_l_nodes(nu_th, D['N_PER_L'], D['L_DEL_U']), nu_th, 'hyd2'

    if cs_type == 8 and params.size >= 4:
        # sub_phot_gen.f:308-363.  CROSS_A(LMIN+3) offsets the edge; sigma is
        # identically zero below EDGE+offset (and below EDGE, from the enclosing
        # IF(FREQ_VEC(I) .GE. EDGE) guard).
        N    = int(round(float(params[0])))
        LST  = int(round(float(params[1])))
        LEND = int(round(float(params[2])))
        if not (1 <= N <= D['MAX_L_PQN']) or LST < 0 or LEND < LST \
                or LEND > D['MAX_L_PQN'] or LEND + 1 > N:
            return None
        edge = nu_th + float(params[3]) * 1.0e15
        start = max(nu_th, edge)
        if edge <= 0:
            return None
        scale = _PROG_TO_CM2 / (zion * zion)

        def fn(nu, _N=N, _a=LST, _b=LEND, _s=scale, _E=edge, _st=start):
            out = np.zeros(np.shape(nu))
            m = np.asarray(nu) >= _st
            if np.any(m):
                out[m] = _s * _hyd_l_block(np.asarray(nu)[m] / _E, _N, _a, _b)
            return out

        return fn, _hyd_l_nodes(edge, D['N_PER_L'], D['L_DEL_U']), start, 'hyd8'

    if cs_type == 3 and params.size >= 2:
        # sub_phot_gen.f:421-451.  ALPHA_BF = 2.815e-6*ZION^4 (rdphot_gen_v2.f
        # :503) already carries ZION; FREQ is in CMFGEN's 1e15 Hz.
        cross = float(params[0])
        N = int(float(params[1]))             # Fortran INTEGER <- REAL: truncate
        if N < 1 or nef <= 0.0 or cross == 0.0:
            return None
        alpha_bf = 2.815e-06 * zion ** 4
        pre = alpha_bf * cross / nef / N / (nef ** 3) * _PROG_TO_CM2

        def fn(nu, _N=N, _p=pre, _e=nu_th):
            out = np.zeros(np.shape(nu))
            m = np.asarray(nu) >= _e
            if np.any(m):
                x = np.asarray(nu)[m]
                out[m] = _p * _gbf_n(x / _e, _N) / ((x * 1.0e-15) ** 3)
            return out

        nodes = _hyd_l_nodes(nu_th, D['N_PER_N'], D['N_DEL_U']) \
            if N <= 30 else np.array([nu_th])
        return fn, nodes, nu_th, 'hyd3'

    if cs_type == 9 and params.size >= 8 and params.size % 8 == 0:
        # sub_phot_gen.f:518-530.  8 params per shell:
        #   0 n, 1 l, 2 E_th[eV], 3 E_0[eV], 4 sigma_0[Mb], 5 y_a, 6 P, 7 y_w.
        # Shell 1 uses the LEVEL's edge; shells 2.. use their own tabulated E_th.
        nsh = params.size // 8
        sh = []
        for j in range(nsh):
            p = np.asarray(params[8 * j:8 * j + 8], dtype='f8')
            if p[3] <= 0.0 or p[5] <= 0.0:
                return None
            e_j = nu_th if j == 0 else _EV_TO_HZ * p[2] * 1.0e15
            sh.append((e_j, p))
        if not sh:
            return None

        def fn(nu, _sh=sh, _e=nu_th):
            nu = np.asarray(nu)
            out = np.zeros(np.shape(nu))
            glob = nu >= _e                   # enclosing IF(FREQ .GE. EDGE)
            for e_j, p in _sh:
                m = glob & (nu >= e_j)
                if not np.any(m):
                    continue
                U  = nu[m] * 1.0e-15 / p[3] / _EV_TO_HZ
                T1 = (U - 1.0) ** 2 + p[7] ** 2
                T2 = U ** (5.5 + p[1] - 0.5 * p[6])
                T3 = (1.0 + np.sqrt(U / p[5])) ** p[6]
                out[m] += 1.0e-08 * T1 * p[4] / T2 / T3 * _PROG_TO_CM2
            return out

        return fn, np.array([e for e, _ in sh]), nu_th, 'verner9'

    return None


def _hyd_l_nodes(edge: float, n_pts: int, del_u: float) -> np.ndarray:
    """Interpolation knots of the HYD_L_DATA / GBF_N_DATA tables, in Hz.

    The tables are linear in log10(u) with step del_u, so sigma has a kink at
    every u = 10**(k*del_u); handing those to the bin-average quadrature keeps
    the trapezoid exact between knots.
    """
    return edge * 10.0 ** (np.arange(n_pts) * del_u)


def _bin_average_sigma(fn, nodes: np.ndarray, edges: np.ndarray,
                       d_nu: np.ndarray, nu_start: float,
                       n_sub: int = 6) -> np.ndarray:
    """(1/dnu_b) INT_{bin b} sigma dnu, for a log-uniform bin grid.

    Node set = {nu_start, bin edges, sigma structure nodes, n_sub-1 log
    subdivisions per bin}, so every trapezoid segment lies inside one bin.
    """
    nb = edges.size - 1
    out = np.zeros(nb)
    lo = max(nu_start, float(edges[0]))
    hi = float(edges[-1])
    if lo >= hi:
        return out
    parts = [np.array([lo, hi]), edges[(edges > lo) & (edges < hi)]]
    if nodes is not None and nodes.size:
        ex = nodes[(nodes > lo) & (nodes < hi)]
        if ex.size:
            parts.append(ex)
    k0 = max(int(np.searchsorted(edges, lo)) - 1, 0)
    le = np.log(edges[k0:])
    if le.size >= 2 and n_sub > 1:
        frac = (np.arange(1, n_sub) / n_sub)[None, :]
        sub = np.exp(le[:-1, None] + (le[1:, None] - le[:-1, None]) * frac).ravel()
        sub = sub[(sub > lo) & (sub < hi)]
        if sub.size:
            parts.append(sub)
    x = np.unique(np.concatenate(parts))
    s = fn(x)
    seg = 0.5 * (s[1:] + s[:-1]) * np.diff(x)
    mid = 0.5 * (x[1:] + x[:-1])
    b = np.clip(np.searchsorted(edges, mid) - 1, 0, nb - 1)
    np.add.at(out, b, seg)
    return out / d_nu


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
    # Bin edges + widths for the bin-AVERAGE bake (BAKEFIX 2).  The C loader's
    # bin centres are nu_min*exp((b+0.5)*dlog) (lumina_plasma.c nu_bin[b]); the
    # baked value is the average over [edge_b, edge_{b+1}], not a sample there.
    nu_edges = BF_NU_MIN * np.exp(np.arange(nb + 1) * d_log_nu)
    d_nu = np.diff(nu_edges)

    sigma_grid = np.zeros((n_levels, nb), dtype='f8')
    has_cmfgen = np.zeros(n_levels, dtype='i1')
    n_skip_type: dict[int, int] = {}
    skip_by_ion: dict[tuple[int, int], int] = {}
    n_exact_type: dict[int, int] = {}
    exact_by_ion: dict[tuple[int, int], int] = {}

    # Build (Z, ion_csv, level_num_csv) -> ionization_eV map for thresholds
    ion_E = {(Z, stage - 1): d['osc'].ionization_eV
             for (Z, stage), d in ion_data.items()}

    n_baked = 0
    for (Z, stage), d in ion_data.items():
        ion_csv = stage - 1
        levs = d['levels']
        n_kept = d['n_kept']
        E_ion = d['osc'].ionization_eV  # eV
        # ZXzV = "Screened nuclear charge" (osc_data header); the phot file
        # repeats it and CMFGEN cross-checks the two, so do the same.
        zion = float(d['osc'].z_screen)
        if SIGMA_EXACT_HYD and d['phot'] is not None \
                and d['phot'].z_screen > 0 \
                and abs(d['phot'].z_screen - zion) > 1e-9:
            print(f"  [bakefix2] WARNING Z={Z} stage={stage}: ZION mismatch "
                  f"osc={zion} phot={d['phot'].z_screen}; using osc value")
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

            # Evaluation paths (see the BAKEFIX head note above):
            #   - Tabulated (20-22)      : linear interpolation in nu.
            #   - Seaton (1)             : sub_phot_gen.f:412, edge at nu_th.
            #   - Modified Seaton (7)    : sub_phot_gen.f:505, edge at nu_th+A3.
            #   - Hydrogenic/Verner (2,3,8,9): known-wrong params[0] stand-in,
            #     kept and counted; see the head note.
            # Every baked curve is stored as the BIN AVERAGE of sigma, not a
            # point sample at the bin centre.
            for lvl_csv in targets:
                E_level_eV = float(levs['E_cm'][lvl_csv]) * 1.239841984e-4
                E_thresh   = E_ion - E_level_eV
                if E_thresh <= 0:
                    continue
                nu_thresh = E_thresh * EV_TO_ERG / H_CGS
                if nu_thresh >= nu_edges[-1]:
                    continue
                # When baking against a ref-merged level table, the ref may
                # have fewer levels for this ion than CMFGEN provides; skip
                # CMFGEN levels that don't exist in the ref.
                gidx = level_lookup.get((Z, stage, lvl_csv + 1))
                if gidx is None:
                    continue

                nef = _cmfgen_nef(Z, zion, nu_thresh) if SIGMA_EXACT_HYD else None
                model = _sigma_model(entry.cs_type, entry.energy,
                                     entry.sigma_Mb, nu_thresh,
                                     zion=zion, nef=nef)
                if model is None:
                    if entry.cs_type in _SIGMA_STANDIN_FIT:
                        n_skip_type[entry.cs_type] = n_skip_type.get(entry.cs_type, 0) + 1
                        skip_by_ion[(Z, stage)] = skip_by_ion.get((Z, stage), 0) + 1
                    continue
                fn, nodes, nu_start, tag = model
                if tag == 'standin':
                    n_skip_type[entry.cs_type] = n_skip_type.get(entry.cs_type, 0) + 1
                    skip_by_ion[(Z, stage)] = skip_by_ion.get((Z, stage), 0) + 1
                elif tag in ('hyd2', 'hyd3', 'hyd8', 'verner9'):
                    n_exact_type[entry.cs_type] = n_exact_type.get(entry.cs_type, 0) + 1
                    exact_by_ion[(Z, stage)] = exact_by_ion.get((Z, stage), 0) + 1
                sigma_grid[gidx, :] = _bin_average_sigma(
                    fn, nodes, nu_edges, d_nu, nu_start)
                # has_cmfgen stays 1 even when the row is all zeros: for type 7
                # "sigma == 0 over the whole grid" is CMFGEN's answer, and must
                # NOT be overwritten by the C-side Kramers fallback.
                has_cmfgen[gidx] = 1
                n_baked += 1

    if n_exact_type:
        tot = sum(n_exact_type.values())
        print(f"  [bakefix2] {tot} levels evaluated EXACTLY on CMFGEN fit types "
              f"{dict(sorted(n_exact_type.items()))} (sub_phot_gen.f port; "
              f"gate CMFGEN_EXACT_HYD=1).")
        top = sorted(exact_by_ion.items(), key=lambda kv: -kv[1])[:8]
        print("  [bakefix2]   top ions (Z,stage):count = "
              + ", ".join(f"({Z},{st}):{c}" for (Z, st), c in top))
    if n_skip_type:
        tot = sum(n_skip_type.values())
        print(f"  [bakefix] WARNING: {tot} levels carry the known-wrong "
              f"params[0]-as-sigma_0 stand-in (CMFGEN fit types "
              f"{dict(sorted(n_skip_type.items()))}); exact evaluation needs "
              f"HYD_L_DATA/GBF_N_DATA -- see the head note in this file.")
        top = sorted(skip_by_ion.items(), key=lambda kv: -kv[1])[:8]
        print("  [bakefix]   most affected ions (Z,stage):count = "
              + ", ".join(f"({Z},{st}):{c}" for (Z, st), c in top))

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
    write_atomic_vintage_manifest(
        ion_data, OUT_DIR / 'atomic_vintage_manifest.csv')
    print(f"  atomic_vintage_manifest.csv: {len(ion_data)} ions")
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
    print(f"  atom_masses.csv ({len(elements_used)} Z)")
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
