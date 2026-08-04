#!/usr/bin/env python3
"""Kromer decomposition of the EMERGENT UV — cycle-breaker diagnostic (2026-07-07).

Question (user's meta-observation): every macro-atom/field patch moves UV a few
points. Are we even attacking the right carrier? Decompose the escaping packets
by their LAST interaction:
  emit_Z < 0  => continuum / thermal (ff/fb/e-scatter) — NOT line fluorescence
  emit_Z > 0  => line emission by that ion (fluorescence-processed)

If emergent UV is mostly continuum/thermal or direct-escape, the whole
line/macro-atom patch campaign was the wrong domain.

Usage: python3 scripts/kromer_uv_decomp.py [lumina_kromer.csv]
"""
import sys, csv, collections
import numpy as np

ROMAN = ['I','II','III','IV','V','VI','VII','VIII']
ELEM = {6:'C',7:'N',8:'O',11:'Na',12:'Mg',13:'Al',14:'Si',16:'S',18:'Ar',20:'Ca',
        22:'Ti',24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni'}
def ion_label(Z, ion):
    if Z < 0: return 'CONT/THERM'
    return f"{ELEM.get(Z,'Z%d'%Z)} {ROMAN[ion] if 0<=ion<len(ROMAN) else ion}"

path = sys.argv[1] if len(sys.argv) > 1 else 'lumina_kromer.csv'
esc_lam=[]; emitZ=[]; emitI=[]; inLam=[]; inZ=[]; en=[]
for r in csv.DictReader(open(path)):
    try:
        esc_lam.append(float(r['escape_lambda_A'])); emitZ.append(int(r['emit_Z']))
        emitI.append(int(r['emit_ion'])); inLam.append(float(r['in_lambda_A']))
        inZ.append(int(r['in_Z'])); en.append(float(r['energy']))
    except (KeyError, ValueError): pass
esc_lam=np.array(esc_lam); emitZ=np.array(emitZ); emitI=np.array(emitI)
inLam=np.array(inLam); inZ=np.array(inZ); en=np.array(en)
print(f"kromer packets: {len(en)}  total escaped energy: {en.sum():.3e}")

BANDS = [('UV 2500-3500', 2500, 3500), ('blue 4200-4700', 4200, 4700),
         ('green 4400-5500', 4400, 5500), ('red 5500-7000', 5500, 7000),
         ('NIR 7000-10000', 7000, 10000)]

for name, lo, hi in BANDS:
    m = (esc_lam >= lo) & (esc_lam < hi)
    Etot = en[m].sum()
    if Etot <= 0:
        print(f"\n=== {name}: (no escaped energy) ==="); continue
    cont = en[m & (emitZ < 0)].sum()
    line = en[m & (emitZ >= 0)].sum()
    print(f"\n=== {name}: E={Etot:.3e} ({100*Etot/en.sum():.1f}% of total) ===")
    print(f"  last-emit CONTINUUM/THERMAL : {100*cont/Etot:5.1f}%")
    print(f"  last-emit LINE (fluoresced) : {100*line/Etot:5.1f}%")
    # break line-emitted by ion
    byion = collections.defaultdict(float)
    mm = m & (emitZ >= 0)
    for z, i, e in zip(emitZ[mm], emitI[mm], en[mm]):
        byion[(z, i)] += e
    top = sorted(byion.items(), key=lambda x: -x[1])[:6]
    if top:
        print("  top line emitters: " + '  '.join(
            f"{ion_label(z,i)}={100*e/Etot:.1f}%" for (z,i), e in top))
    # where were UV escapers ABSORBED (in-band vs bluer)? only for UV
    if lo == 2500:
        mi = m & (inLam > 0)
        if mi.sum():
            same = en[mi & (inLam >= 2500) & (inLam < 3500)].sum()
            bluer = en[mi & (inLam < 2500)].sum()
            redder = en[mi & (inLam >= 3500)].sum()
            noabs = en[m & (inLam <= 0)].sum()
            print(f"  UV escaper LAST-ABSORPTION: same-UV={100*same/Etot:.1f}%  "
                  f"bluer(<2500)={100*bluer/Etot:.1f}%  redder={100*redder/Etot:.1f}%  "
                  f"never-absorbed(direct)={100*noabs/Etot:.1f}%")

print("\n[VERDICT] If UV last-emit is mostly CONTINUUM/THERMAL or never-absorbed")
print("(direct escape), the line/macro-atom fluorescence patches were the wrong")
print("domain. If mostly LINE (Fe/Co/etc.), fluorescence IS the mechanism but")
print("under-converts — then the reprocessing-depth / redistribution is the lever.")
