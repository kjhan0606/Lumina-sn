#!/usr/bin/env python3
"""ARTIS toy06_nlte_bk baseline extractor: Fe III per-level b_k + cell Te/ionfrac at a timestep.
Comparison truth for the ARTIS-parity Lumina run (target: Lumina Fe III b_k -> ARTIS ~1).
Usage: artis_baseline_bk.py [timestep=20] [Z=26] [ionstage=3]"""
import sys, glob, os
AD="/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk"
TS=int(sys.argv[1]) if len(sys.argv)>1 else 27  # [사례 18] 19.48d=ts27 (mid 20.25d); ts20=11.24d — timesteps.out이 정본
Z=int(sys.argv[2]) if len(sys.argv)>2 else 26
ION=int(sys.argv[3]) if len(sys.argv)>3 else 3   # ionstage 1=neutral -> FeIII=3
# nlte: timestep modelgridindex Z ionstage level n_LTE n_NLTE ion_popfrac
cell_lev={}   # cell -> {lev: b_k}
cell_ionfrac={}  # cell -> {stage: popfrac}
for F in sorted(glob.glob(f"{AD}/nlte_000[1-7].out")):
    for ln in open(F):
        p=ln.split()
        if len(p)<8 or p[0]!=str(TS) or p[2]!=str(Z): continue
        c=int(p[1])
        if p[3]==str(ION):
            lev=int(p[4]); nL=float(p[5]); nN=float(p[6])
            if nL>0: cell_lev.setdefault(c,{})[lev]=nN/nL
        if p[4]=="0":
            try: cell_ionfrac.setdefault(c,{})[int(p[3])]=float(p[7])
            except: pass
cells=sorted(cell_lev)
print(f"# ARTIS toy06_nlte_bk ts={TS} (19.49d) Z={Z} ionstage={ION}(FeIII)  cells with FeIII: {cells}")
print(f"# photosphere ~ cell10-11.  f(FeIV)=ionfrac stage4 / (stage3+4+...)")
for c in cells:
    lv=cell_lev[c]
    # trap-band levels
    trap={l:lv[l] for l in (17,25,31,32,28) if l in lv}
    ifr=cell_ionfrac.get(c,{})
    tot=sum(v for v in ifr.values() if v==v)  # skip nan
    fIV=(ifr.get(4,0)/tot) if tot>0 else float('nan')
    print(f"cell{c:>3}: f(FeIV)={fIV:.3f}  trap b_k: "+" ".join(f"lev{l}={trap[l]:.2f}" for l in sorted(trap)))
