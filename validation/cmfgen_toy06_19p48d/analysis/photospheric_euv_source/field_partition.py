#!/usr/bin/env python3
"""Field-level partition: which estimator does the Gph loop read in the EUV, and
what is LUMINA's radial EUV profile vs CMFGEN's?

Gph loop (lumina_plasma.c:5845-5868, 5901-5924): per bin
    J = mc_J   if g_photoion_mc_count>0   (alpha=1 => full mc override)
      = cs_J   (nlte->J_nu) otherwise.
The field CSV (lumina_cuda.cu:6131 'shell,bin,wavelength_A,cs_J,mc_J') holds both.
mc_J at floor 1e-30 <=> zero-count bin <=> Gph uses cs_J.

CMFGEN contrast from gradient_budget_shells.csv (CMFGEN_J300_am / _J918_gm) and
jnu_918_1290_formingshells.csv (918A only; <912 not in that extract -> limitation).
"""
import csv, math, numpy as np
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
AN=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis"
FCSV=f"{REPO}/logs/coevolve_consume_a10_kx_kpr5/lumina_coevolve_field.csv"
OUT=f"{AN}/photospheric_euv_source"
FLOOR=1.01e-30
C=2.99792458e18
# load field: shell -> list of (wl, cs, mc)
rows={}
with open(FCSV) as f:
    for r in csv.DictReader(f):
        s=int(r['shell']); rows.setdefault(s,[]).append(
            (float(r['wavelength_A']),float(r['cs_J']),float(r['mc_J'])))

def band_partition(s,lo,hi):
    """return (J_from_cs, J_from_mc, n_cs, n_mc) integrating the field the Gph
    loop actually reads (mc where count>0 else cs), summed as J*dnu over the band."""
    Jcs=Jmc=0.0; ncs=nmc=0
    for wl,cs,mc in rows[s]:
        if wl<lo or wl>=hi: continue
        nu=C/wl
        # dnu weight ~ nu (log grid); use plain J-sum proxy (relative shares robust)
        if mc>FLOOR:  Jmc+=mc; nmc+=1
        else:         Jcs+=cs; ncs+=1
    return Jcs,Jmc,ncs,nmc

print("="*90)
print("FIELD PARTITION at photosphere: J the Gph loop reads = mc(count>0) else cs")
print("  (share of the band J-sum sourced from cs_J vs mc_J)")
print("="*90)
out=[]
for label,(lo,hi) in [("EUV<912",(100,912)),("300-450",(300,450)),("450-912",(450,912))]:
    print(f"\n[{label}]")
    print(f"  {'shell':6}{'cs_share%':>10}{'mc_share%':>10}{'n_cs':>7}{'n_mc':>7}{'J_cs':>12}{'J_mc':>12}")
    for s in [0,2,4,6,7,8,9]:
        Jcs,Jmc,ncs,nmc=band_partition(s,lo,hi); T=Jcs+Jmc
        cshare=100*Jcs/T if T>0 else float('nan'); mshare=100*Jmc/T if T>0 else float('nan')
        print(f"  {s:<6}{cshare:>10.1f}{mshare:>10.1f}{ncs:>7}{nmc:>7}{Jcs:>12.3e}{Jmc:>12.3e}")
        out.append([label,s,cshare,mshare,ncs,nmc,Jcs,Jmc])

print("\n"+"="*90)
print("RADIAL EUV PROFILE (cs_J the Gph reads is dominated by cs; report cs band-sum)")
print("  LUMINA decline s0->s8 vs CMFGEN's steep outward decline")
print("="*90)
def cssum(s,lo,hi): return sum(cs for wl,cs,mc in rows[s] if lo<=wl<hi)
for label,(lo,hi) in [("EUV<912",(100,912)),("300-450",(300,450)),("450-912",(450,912))]:
    v0=cssum(0,lo,hi); v8=cssum(8,lo,hi)
    fac=v0/v8 if v8>0 else float('inf')
    trend="RISE" if v8>v0 else "decline"
    tau=math.log(fac) if fac>0 else float('nan')
    print(f"  [{label}] cs_J s0={v0:.3e}  s8={v8:.3e}  -> s0/s8={fac:.2g} ({trend}); "
          f"tau_eff~ln={tau:+.1f}")
    out.append([f"radial_{label}",0,v0,v8,fac,tau,0,0])

# ---- CMFGEN contrast ----
print("\n"+"="*90)
print("CMFGEN CONTRAST  (gradient_budget_shells.csv: CMFGEN_J300_am, CMFGEN_J918_gm)")
print("="*90)
gb={}
with open(f"{AN}/gradient_budget_shells.csv") as f:
    for r in csv.DictReader(f): gb[int(r['shell'])]=r
def g(s,k):
    try: return float(gb[s][k])
    except: return float('nan')
print(f"  {'shell':6}{'v_kms':>7}{'CMF_J300_am':>13}{'CMF_J918_gm':>13}{'LUM_J300_am':>13}{'LUM_J918_gm':>13}")
for s in [0,2,4,5,6,7,8,9]:
    if s not in gb: continue
    print(f"  {s:<6}{g(s,'v_kms'):>7.0f}{g(s,'CMFGEN_J300_am'):>13.3e}"
          f"{g(s,'CMFGEN_J918_gm'):>13.3e}{g(s,'LUM_J300_am'):>13.3e}{g(s,'LUM_J918_gm'):>13.3e}")
c0=g(0,'CMFGEN_J300_am'); c8=g(8,'CMFGEN_J300_am')
print(f"\n  CMFGEN J300 s0->s8 decline factor = {c0/c8:.3g}  (tau_eff~ln={math.log(c0/c8):.1f})")
print(f"  => CMFGEN EUV(300A) is optically THICK s0->s8 (steep absorption) AND faint at phot.")
l0=cssum(0,300,450); l8=cssum(8,300,450)
print(f"  LUMINA(kpr5) cs_J(300-450) s0->s8 decline = {l0/l8:.3g}  (tau_eff~ln={math.log(l0/l8):.1f})")
print(f"  LUMINA(kpr5) cs_J(450-912) s0->s8 = {cssum(0,450,912)/cssum(8,450,912):.3g} (RISE=local source)")

# jnu CSV 918A note
jn={}
with open(f"{AN}/jnu_918_1290_formingshells.csv") as f:
    for r in csv.DictReader(f):
        v=int(round(float(r['velocity_km_s']))); w=float(r['wavelength_A'])
        if abs(w-918.0)<0.5: jn[v]=float(r['J_nu'])
print(f"\n  CMFGEN jnu(918A) forming shells (limitation: extract stops at 918A, no <912):")
for v in sorted(jn): print(f"    v={v:6d}  J_nu(918)={jn[v]:.3e}")

with open(f"{OUT}/field_partition.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(['band_or_kind','shell','a','b','c','d','e','f']); w.writerows(out)
print(f"\n[out] {OUT}/field_partition.csv")
