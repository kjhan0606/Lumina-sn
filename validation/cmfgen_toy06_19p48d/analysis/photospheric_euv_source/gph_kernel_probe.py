#!/usr/bin/env python3
"""Which EUV band drives the Fe III photoionization at the photosphere?
Crude GROUND-state Kramers kernel on the actual field the Gph loop reads (mc_J
where sampled else cs_J), integrated above the Fe III ground edge (404.5A, 30.65eV).
w(nu) ~ sig(nu)*J/(h nu)*dnu ; sig ~ sig_edge*(nu_edge/nu)^3 (hydrogenic, code's
Kramers fallback form). This is the pop=1 ground anchor -- excited levels add edges
redward but this isolates the deep-EUV driver. Compares LUMINA vs CMFGEN J at 300A.
"""
import csv, math
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
FCSV=f"{REPO}/logs/coevolve_consume_a10_kx_kpr5/lumina_coevolve_field.csv"
AN=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis"
C=2.99792458e18; H=6.62607015e-27; FLOOR=1.01e-30
EDGE_A=404.5   # Fe III ground edge (30.65 eV)
nu_edge=C/EDGE_A
rows={}
with open(FCSV) as f:
    for r in csv.DictReader(f):
        s=int(r['shell']); rows.setdefault(s,[]).append(
            (float(r['wavelength_A']),float(r['cs_J']),float(r['mc_J'])))

def gph_ground(s):
    """crude ground Kramers Gph on the field Gph reads; split by band contribution."""
    # sort by wavelength ascending to get dnu
    band_w={'300-450':0.0,'450-912':0.0,'<300':0.0}
    tot=0.0
    lst=sorted(rows[s])  # by wl asc
    for i,(wl,cs,mc) in enumerate(lst):
        nu=C/wl
        if nu<nu_edge: continue          # below threshold, no ionization from ground
        J = mc if mc>FLOOR else cs        # exactly the Gph-loop selection
        if J<=0: continue
        # dnu from neighbor spacing
        if i+1<len(lst): dnu=abs(C/lst[i+1][0]-nu)
        else: dnu=abs(nu-C/lst[i-1][0])
        sig=(nu_edge/nu)**3               # hydrogenic shape (edge-normalized=1)
        w=sig*J/(H*nu)*dnu
        tot+=w
        if wl<300: band_w['<300']+=w
        elif wl<450: band_w['300-450']+=w
        else: band_w['450-912']+=w
    return tot,band_w

print("Crude GROUND Fe III Kramers Gph (edge-normalized sig; relative band shares):")
print(f"  {'shell':6}{'v_kms':>7}{'Gph_gnd(rel)':>14}{'%<300':>8}{'%300-450':>10}{'%450-912':>10}")
VMAP={0:4264,2:5720,4:7176,5:7904,6:8632,7:9360,8:10088,9:10816}
for s in [0,4,6,7,8,9]:
    tot,bw=gph_ground(s);
    if tot<=0: continue
    print(f"  {s:<6}{VMAP.get(s,0):>7}{tot:>14.3e}{100*bw['<300']/tot:>8.1f}"
          f"{100*bw['300-450']/tot:>10.1f}{100*bw['450-912']/tot:>10.1f}")

# per-bin LUMINA vs CMFGEN at 300A and near 404A (edge)
print("\nLUMINA mc_J vs CMFGEN J at the Fe III edge region (photosphere shell 8):")
def nearest(s,target):
    best=min(rows[s],key=lambda t:abs(t[0]-target)); return best
for tgt in [300.0,350.0,404.5,450.0]:
    wl,cs,mc=nearest(8,tgt); Jread=mc if mc>FLOOR else cs
    print(f"  ~{tgt:6.1f}A: LUMINA(shell8) wl={wl:.2f} mc_J={mc:.3e} cs_J={cs:.3e} -> Gph reads {Jread:.3e}")
# CMFGEN J300 at shell 8 from gradient_budget
gb={}
with open(f"{AN}/gradient_budget_shells.csv") as f:
    for r in csv.DictReader(f): gb[int(r['shell'])]=r
print(f"  CMFGEN J300_am(shell8)={float(gb[8]['CMFGEN_J300_am']):.3e}  "
      f"J300_gm(shell0)={float(gb[0]['CMFGEN_J300_gm']):.3e}")
wl,cs,mc=nearest(8,300.0); Jl=mc if mc>FLOOR else cs
print(f"  => LUMINA/CMFGEN field ratio @~300A (shell8) = {Jl/float(gb[8]['CMFGEN_J300_am']):.2e}")
