#!/usr/bin/env python3
"""certify_coverage.py -- coverage extension of the rate-machine certification.

The certification harness (../certify_rate_machine.py) is NOT modified: its
sha256 must stay 25d36f43c731917534f154e68ea9da16f2416693ca7e634e39f41a8869533ac1.
This driver imports it as a module, replaces the hard-coded `IONS` table with a
larger one built from the SAME schema, and calls its `main()` unchanged.  Every
convention (depth points s0..s8 + d42/45/47/49/50, the A/Ag/Bpt/Bav/C/C2/D
ladder, the D/PRRR gate) is therefore identical by construction, and the
existing ions are a byte-level regression test (see --group regress).

Ion table entries are exactly the harness's schema:
    lab, Z, ion(=stage-1), cmf(CMFGEN species name), pop(POP* file),
    osc/fts/phot (files the RUN wrote, i.e. the reference vintage),
    lum_osc/lum_phot (LUMINA-side atomic data = what the baker picks).

usage:  certify_coverage.py --group new --bake bakefix4b
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
sys.path.insert(0, PARENT)

import certify_rate_machine as crm    # noqa: E402  (the untouched yardstick)

# --------------------------------------------------------------- F_TO_S shim
# SIL/IV/5dec96/f_to_s_split.dat (what toy06 ran for Si IV) has SEVEN columns:
#   name  g  E_cm  <col>  <col>  SL_index  flag
# i.e. no trailing 1-based level index, so the shipped parser's
# `idx = int(t[7])` guard raises IndexError on every line and returns an empty
# array (crm.main() then dies in `SL[:NF].max()`).  The fallback below is used
# ONLY when the shipped parser returns fewer than nlev entries, so it is a
# provable no-op for every file that already parses (see --group regress).
#
# Verified independently for SkIV, not assumed:
#   * the names it returns reproduce SkIV_F_OSCDAT's level order 1:1 (the
#     harness re-checks this itself and aborts if it fails), and
#   * max(SL[:NF=61]) = 50 = the number of rows in the "SkIV Photoionization
#     Rates" block of SkIVPRRR, i.e. the super-level count the run itself used.
_ORIG_FTS = crm.parse_f_to_s


def parse_f_to_s_compat(path, nlev):
    sl, nm = _ORIG_FTS(path, nlev)
    if len(sl) >= nlev:
        return sl, nm
    sl2, nm2 = [], []
    with open(path) as fh:
        for ln in fh:
            t = ln.split()
            if len(t) < 7:
                continue
            try:
                float(t[1]); float(t[2]); float(t[3]); float(t[4])
                int(t[5]); int(t[6])
            except ValueError:
                continue
            sl2.append(int(t[5]))
            nm2.append(t[0])
            if len(sl2) >= nlev:
                break
    if len(sl2) >= nlev:
        return np.array(sl2), nm2
    return sl, nm


crm.parse_f_to_s = parse_f_to_s_compat


def E(lab, Z, ion, cmf, pop, osc, fts, phot, ldir):
    return dict(lab=lab, Z=Z, ion=ion, cmf=cmf, pop=pop,
                osc=osc, fts=fts, phot=phot,
                lum_osc=f'data/atomic/cmfgen/{ldir}/19apr23/osc_data',
                lum_phot=f'data/atomic/cmfgen/{ldir}/19apr23/phot_data_A')


# ---- the four ions the shipped harness carries, verbatim (regression set) ----
REGRESS = crm.IONS

# ---- the rest of the ions already run through the harness in earlier sessions
EXISTING_EXT = [
    E('Co II', 27, 1, 'Co2', 'POPCOB', 'Co2_F_OSCDAT', 'Co2_F_TO_S', 'PHOTCo2_A', 'COB/II'),
    E('Fe II', 26, 1, 'Fe2', 'POPIRON', 'Fe2_F_OSCDAT', 'Fe2_F_TO_S', 'PHOTFe2_A', 'FE/II'),
]
EXISTING_VM = [
    E('Ca II', 20, 1, 'Ca2', 'POPCAL', 'Ca2_F_OSCDAT', 'Ca2_F_TO_S', 'PHOTCa2_A', 'CA/II'),
    E('Si II', 14, 1, 'Sk2', 'POPSIL', 'Sk2_F_OSCDAT', 'Sk2_F_TO_S', 'PHOTSk2_A', 'SIL/II'),
    E('Si III', 14, 2, 'SkIII', 'POPSIL', 'SkIII_F_OSCDAT', 'SkIII_F_TO_S', 'PHOTSkIII_A', 'SIL/III'),
]

# ---- NEW: every remaining ion the toy06 reference run carries AND the LUMINA
#      atom has rows for.  Priority order per the task.
NEW = [
    E('Ni II', 28, 1, 'Nk2', 'POPNICK', 'Nk2_F_OSCDAT', 'Nk2_F_TO_S', 'PHOTNk2_A', 'NICK/II'),
    E('Ni III', 28, 2, 'NkIII', 'POPNICK', 'NkIII_F_OSCDAT', 'NkIII_F_TO_S', 'PHOTNkIII_A', 'NICK/III'),
    E('Ni IV', 28, 3, 'NkIV', 'POPNICK', 'NkIV_F_OSCDAT', 'NkIV_F_TO_S', 'PHOTNkIV_A', 'NICK/IV'),
    E('Co IV', 27, 3, 'CoIV', 'POPCOB', 'CoIV_F_OSCDAT', 'CoIV_F_TO_S', 'PHOTCoIV_A', 'COB/IV'),
    E('Fe IV', 26, 3, 'FeIV', 'POPIRON', 'FeIV_F_OSCDAT', 'FeIV_F_TO_S', 'PHOTFeIV_A', 'FE/IV'),
    E('Fe V', 26, 4, 'FeV', 'POPIRON', 'FeV_F_OSCDAT', 'FeV_F_TO_S', 'PHOTFeV_A', 'FE/V'),
    E('S IV', 16, 3, 'SIV', 'POPSUL', 'SIV_F_OSCDAT', 'SIV_F_TO_S', 'PHOTSIV_A', 'SUL/IV'),
    E('S V', 16, 4, 'SV', 'POPSUL', 'SV_F_OSCDAT', 'SV_F_TO_S', 'PHOTSV_A', 'SUL/V'),
    E('Ca III', 20, 2, 'CaIII', 'POPCAL', 'CaIII_F_OSCDAT', 'CaIII_F_TO_S', 'PHOTCaIII_A', 'CA/III'),
    E('Ca IV', 20, 3, 'CaIV', 'POPCAL', 'CaIV_F_OSCDAT', 'CaIV_F_TO_S', 'PHOTCaIV_A', 'CA/IV'),
    E('Ca V', 20, 4, 'CaV', 'POPCAL', 'CaV_F_OSCDAT', 'CaV_F_TO_S', 'PHOTCaV_A', 'CA/V'),
    E('Si IV', 14, 3, 'SkIV', 'POPSIL', 'SkIV_F_OSCDAT', 'SkIV_F_TO_S', 'PHOTSkIV_A', 'SIL/IV'),
]

# ---- ions the reference run carries but the LUMINA atom does NOT (no rows in
#      levels.csv).  Layer D is structurally 0; layers PRRR/A/C still measure
#      whether the missing ion matters and whether CMFGEN's own numbers close.
ABSENT = [
    E('Ni V', 28, 4, 'NkV', 'POPNICK', 'NkV_F_OSCDAT', 'NkV_F_TO_S', 'PHOTNkV_A', 'NICK/V'),
    E('Co V', 27, 4, 'CoV', 'POPCOB', 'CoV_F_OSCDAT', 'CoV_F_TO_S', 'PHOTCoV_A', 'COB/V'),
    E('Si V', 14, 4, 'SkV', 'POPSIL', 'SkV_F_OSCDAT', 'SkV_F_TO_S', 'PHOTSkV_A', 'SIL/V'),
]

# ---- ions the LUMINA atom gains under CMFGEN_FULL_LEVELS=1 (bakefix5).  Same
#      six as ABSENT plus Fe VI / Co VI / Ni VI, which ABSENT never listed.
#      ABSENT itself is left untouched so earlier runs stay reproducible.
NEW6 = [
    E('Si V',  14, 4, 'SkV',   'POPSIL',  'SkV_F_OSCDAT',   'SkV_F_TO_S',   'PHOTSkV_A',   'SIL/V'),
    E('Fe VI', 26, 5, 'FeSIX', 'POPIRON', 'FeSIX_F_OSCDAT', 'FeSIX_F_TO_S', 'PHOTFeSIX_A', 'FE/VI'),
    E('Co V',  27, 4, 'CoV',   'POPCOB',  'CoV_F_OSCDAT',   'CoV_F_TO_S',   'PHOTCoV_A',   'COB/V'),
    E('Co VI', 27, 5, 'CoSIX', 'POPCOB',  'CoSIX_F_OSCDAT', 'CoSIX_F_TO_S', 'PHOTCoSIX_A', 'COB/VI'),
    E('Ni V',  28, 4, 'NkV',   'POPNICK', 'NkV_F_OSCDAT',   'NkV_F_TO_S',   'PHOTNkV_A',   'NICK/V'),
    E('Ni VI', 28, 5, 'NkSIX', 'POPNICK', 'NkSIX_F_OSCDAT', 'NkSIX_F_TO_S', 'PHOTNkSIX_A', 'NICK/VI'),
]

GROUPS = dict(regress=REGRESS, existing_ext=EXISTING_EXT, existing_vm=EXISTING_VM,
              new=NEW, absent=ABSENT, new6=NEW6,
              all=REGRESS + EXISTING_EXT + EXISTING_VM + NEW,
              all27=REGRESS + EXISTING_EXT + EXISTING_VM + NEW + NEW6)

BAKES = {
    'shipped': ('data/atomic/cmfgen_sigma_bf_superlev_ionfix_ddc15strat_sivcaiv.bin',
                'data/tardis_reference_cmfgen_superlev_ionfix_ddc15strat_sivcaiv'),
    'bakefix': ('data/atomic/cmfgen_sigma_bf_sivcaiv_bakefix.bin',
                'data/tardis_reference_cmfgen_sivcaiv_bakefix'),
    'bakefix2': ('data/atomic/cmfgen_sigma_bf_sivcaiv_bakefix2.bin',
                 'data/tardis_reference_cmfgen_sivcaiv_bakefix2'),
    'bakefix4': ('data/atomic/cmfgen_sigma_bf_sivcaiv_bakefix4.bin',
                 'data/tardis_reference_cmfgen_sivcaiv_bakefix4'),
    'bakefix4b': ('data/atomic/cmfgen_sigma_bf_sivcaiv_bakefix4b.bin',
                  'data/tardis_reference_cmfgen_sivcaiv_bakefix4b'),
    'bakefix5': ('data/atomic/cmfgen_sigma_bf_sivcaiv_bakefix5.bin',
                 'data/tardis_reference_cmfgen_sivcaiv_bakefix5'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True, choices=sorted(GROUPS))
    ap.add_argument('--bake', required=True, choices=sorted(BAKES))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    bin_, ref = BAKES[a.bake]
    out = a.out or os.path.join(HERE, f'{a.bake}__{a.group}')
    os.makedirs(out, exist_ok=True)

    crm.IONS = GROUPS[a.group]
    sys.argv = ['certify_rate_machine.py',
                '--bin', os.path.join(ROOT, bin_),
                '--ref', os.path.join(ROOT, ref),
                '--out', out]
    crm.main()


if __name__ == '__main__':
    main()
