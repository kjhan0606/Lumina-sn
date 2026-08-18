#!/usr/bin/env python3
"""f1_arithmetic_closure.py -- OFFLINE F1 arithmetic closure (no new data).

Decompose the measured deep FUV deficit (s0: Lumina mc_J vs CMFGEN) into
COLOR (cold T_e Wien penalty) + DILUTION (geometry / CMFGEN deeper thermal
ceiling) + RESIDUAL (genuine transport trapping), using:
  - exact band-integrated Planck ratios B(13120)/B(18900) per band,
  - the dilution profile W(r) normalized at s0,
  - the J values already measured by f0b (read from f0b_thermalization_shells.csv).

Identity used (per band, at a shell):
  Delta = log10(J_C / J_L)                              [measured deficit]
        = log10[B(T_C)/B(T_L)]        (COLOR)
        + log10[f_C / W_L]            (DILUTION: CMFGEN thermalizes above W_L*B)
        + log10[W_L / f_L]            (RESIDUAL: Lumina vs its own W_L*B ceiling)
  where f = J/B(T_local) is the thermalization fraction, W_L the Lumina dilution.
"""
import csv, math
import numpy as np

REPO='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
OUT=f'{REPO}/validation/cmfgen_toy06_19p48d/analysis'
H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; CLIGHT_A=2.99792458e18

BANDS=[("EUV_300_450",300.,450.),("FUV_918_1290",918.,1290.),
       ("flank_1290_2000",1290.,2000.),("flank_2000_4000",2000.,4000.)]

def Bnu(nu,T):
    x=np.minimum(H*nu/(KB*T),700.0); return 2*H*nu**3/C**2/np.expm1(x)
def bandB(lo_A,hi_A,T,npt=2000):
    nu=np.linspace(CLIGHT_A/hi_A,CLIGHT_A/lo_A,npt)
    return np.trapz(Bnu(nu,T),nu)/(nu[-1]-nu[0])

# ---- load f0b measured J table ----
rows={}
for r in csv.DictReader(open(f'{OUT}/f0b_thermalization_shells.csv')):
    rows[(r['band'],int(r['shell']))]=r

TL=float(rows[('FUV_918_1290',0)]['B_Te'])     # Lumina s0 T_e (13120, not pinned)
TC=float(rows[('FUV_918_1290',0)]['C_Tc'])      # CMFGEN s0 T (18900)
WL=float(rows[('FUV_918_1290',0)]['B_W'])       # Lumina s0 dilution

print("="*96)
print("F1 ARITHMETIC CLOSURE  (deep deficit at s0; OFFLINE, no new data)")
print("="*96)
print(f"  s0:  T_e(Lumina)={TL:.0f} K   T(CMFGEN)={TC:.0f} K   W_L={WL:.4f}")

# ---- (A) exact band-integrated Planck color ratios ----
print("\n(A) EXACT band-integrated Planck color factor  log10[ B_band(T_C=%.0f)/B_band(T_L=%.0f) ]:" % (TC,TL))
print(f"    {'band':>18} {'Bbar(T_L)':>12} {'Bbar(T_C)':>12} {'ratio':>10} {'dex':>7}")
color={}
for bn,lo,hi in BANDS:
    bl=bandB(lo,hi,TL); bc=bandB(lo,hi,TC); rt=bc/bl; color[bn]=math.log10(rt)
    print(f"    {bn:>18} {bl:>12.4e} {bc:>12.4e} {rt:>10.3e} {math.log10(rt):>+7.2f}")

# ---- (B) dilution-only profile W(r) normalized at s0 ----
print("\n(B) DILUTION-only profile W(r)/W(s0)  (geometric dilution, Lumina plasma W):")
print(f"    {'sh':>3} {'v':>6} {'W':>9} {'W/W(s0)':>9} {'dex':>7}")
for s in range(0,11):
    r=rows[('FUV_918_1290',s)]; W=float(r['B_W']); rel=W/WL
    print(f"    {s:>3} {r['v_kms']:>6} {W:>9.5f} {rel:>9.4f} {math.log10(rel):>+7.2f}")
print("    -> if the field were dilution-only (J proportional to W), s0->s8 would fall "
      f"{math.log10(float(rows[('FUV_918_1290',8)]['B_W'])/WL):+.2f} dex.")
print("       (CMFGEN FUV s0->s8 actually RISES inward by ~+2.4 dex; pure-dilution cannot make it flat.)")

# ---- (C) s0 decomposition per band (energy-int metric) ----
print("\n(C) s0 DEFICIT DECOMPOSITION  Delta = COLOR + DILUTION + RESIDUAL  [energy-integrated J]")
print(f"    {'band':>18} {'J_L':>10} {'J_C':>10} {'Delta':>7} | {'color':>7} {'dilut':>7} {'resid':>7} {'sum':>7} {'f_L':>6} {'f_C':>6}")
closure_csv=[]
for bn,lo,hi in BANDS:
    r=rows[(bn,0)]
    JL=float(r['B_JmeanE']); JC=float(r['C_JmeanE'])
    BL=float(r['B_Bnu_Te']); BC=float(r['C_Bnu_Tc'])
    if JL<=0 or JC<=0: continue
    fL=JL/BL; fC=JC/BC
    Delta=math.log10(JC/JL)
    col=color[bn]
    dil=math.log10(fC/WL)
    res=math.log10(WL/fL)
    ssum=col+dil+res
    print(f"    {bn:>18} {JL:>10.3e} {JC:>10.3e} {Delta:>+7.2f} | {col:>+7.2f} {dil:>+7.2f} {res:>+7.2f} {ssum:>+7.2f} {fL:>6.3f} {fC:>6.3f}")
    closure_csv.append([bn,'energy',f"{JL:.4e}",f"{JC:.4e}",f"{Delta:+.3f}",
                        f"{col:+.3f}",f"{dil:+.3f}",f"{res:+.3f}",f"{fL:.4f}",f"{fC:.4f}"])

# ---- (C') FUV s0 also with GEOM-mean J (the doc's -2.3 dex metric) ----
print("\n(C') FUV s0 with GEOM-mean J (matches doc's -2.3 dex decomposition metric):")
rf=rows[('FUV_918_1290',0)]
JLg=float(rf['B_Jgeom']); BL=float(rf['B_Bnu_Te']); BC=float(rf['C_Bnu_Tc'])
# CMFGEN geom J for FUV s0 from gradient_budget_shells.csv (self-run, geom-mean)
JCg=None
try:
    for r in csv.DictReader(open(f'{OUT}/gradient_budget_shells.csv')):
        if int(r['shell'])==0: JCg=float(r['CMFGEN_J918_gm']); break
except FileNotFoundError:
    pass
if JCg:
    fLg=JLg/BL; fCg=JCg/BC
    Delta=math.log10(JCg/JLg); col=color['FUV_918_1290']; dil=math.log10(fCg/WL); res=math.log10(WL/fLg)
    print(f"    J_L(geom)={JLg:.3e}  J_C(geom)={JCg:.3e}  Delta={Delta:+.2f} dex")
    print(f"    = COLOR {col:+.2f} + DILUTION {dil:+.2f} + RESIDUAL {res:+.2f}  (sum {col+dil+res:+.2f})")
    print(f"    f_L(geom)={fLg:.3f}  f_C(geom)={fCg:.3f}  W_L={WL:.3f}")
    closure_csv.append(['FUV_918_1290','geom',f"{JLg:.4e}",f"{JCg:.4e}",f"{Delta:+.3f}",
                        f"{col:+.3f}",f"{dil:+.3f}",f"{res:+.3f}",f"{fLg:.4f}",f"{fCg:.4f}"])

print("\nINTERPRETATION:")
print("  COLOR (cold T_e Wien penalty) dominates the FUV/EUV deficit.")
print("  DILUTION>0 = CMFGEN's deep gas thermalizes ABOVE a W_L-dilute Planck (optically")
print("             thicker because hotter/more-ionized) -- an INDIRECT T_e effect.")
print("  RESIDUAL ~ 0 (energy) to +0.5 dex (geom continuum) = the only GENUINE transport")
print("             trapping deficit; << the color term -> not the dominant axis.")

# ---- CSV ----
outp=f'{OUT}/f1_closure_table.csv'
with open(outp,'w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['band','metric','J_L','J_C','deficit_dex','color_dex','dilution_dex','residual_dex','f_L','f_C'])
    w.writerows(closure_csv)
print(f"\n[out] -> {outp}")
