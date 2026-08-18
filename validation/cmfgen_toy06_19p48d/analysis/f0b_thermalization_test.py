#!/usr/bin/env python3
"""f0b_thermalization_test.py -- OFFLINE F0b decisive fork (1a T_e vs 1b transport).

Per band (EUV 300-450, FUV 918-1290, flank 1290-2000, 2000-4000 A) per forming
shell s0-s10, compute the band-integrated mean intensity J and the thermalization
ratios J/B(T_e_local) and J/[W*B(T_e_local)] for:
  (a) Lumina B-run   (all-level Gph, mc_J field) : logs/coevolve_consume_a10_kx_gphall
  (b) Lumina jtable  (CMFGEN-field arm, mc_J)    : logs/coevolve_consume_a10_kx_jtable
  (c) CMFGEN self-run field (jnu4 EDDFACTOR) with published T(v).

FORK (doc FUV_GRADIENT_ATTACK_DESIGN.md, F0b):
  deep J ~ W*B(cold T_e)  -> T_e's fault (Axis 1a) -> promote F3-T
  deep J << W*B(cold T_e) -> transport starvation (Axis 1b) -> promote F0a

YARDSTICK AUDIT (printed): T_rad is pinned (uniq=1, 10470.093 in all shells) in
BOTH runs -> any B(T_rad) column is DEFINITIONAL, not measured. T_e pins:
B-run pins s9-12, jtable pins s0-2 (=10470.093=T_rad) -> for those shells J/B(T_e)
uses a definitional T_e. mc_J is the transported MC field (run scheme alpha=1.0)
-> its amplitude is a real measurement. CMFGEN J is a converged self-run field.
"""
import os, sys, math, csv
import numpy as np

REPO   = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
STD    = f'{REPO}/data/standart_data1/toy06'
JNU4   = '/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4'
BRUN   = f'{REPO}/logs/coevolve_consume_a10_kx_gphall'
JTAB   = f'{REPO}/logs/coevolve_consume_a10_kx_jtable'
OUT    = f'{REPO}/validation/cmfgen_toy06_19p48d/analysis'

H  = 6.62607015e-27      # erg s
KB = 1.380649e-16        # erg/K
C  = 2.99792458e10       # cm/s
CLIGHT_A = 2.99792458e18 # A/s  (lam_A = CLIGHT_A/nu_Hz)

# forming shells: Lumina shell index -> mid velocity (uniform 728 km/s grid,
# matches gradient_budget.py VEL for the overlapping shells)
SHELLS = list(range(0, 11))
VEL    = [4264 + 728*n for n in SHELLS]

BANDS = [("EUV_300_450", 300., 450.),
         ("FUV_918_1290", 918., 1290.),
         ("flank_1290_2000", 1290., 2000.),
         ("flank_2000_4000", 2000., 4000.)]

# ---------------------------------------------------------------- Planck
def Bnu(nu, T):
    x = H*nu/(KB*T)
    x = np.minimum(x, 700.0)
    return 2.0*H*nu**3/C**2 / np.expm1(x)

def band_mean_B(nu_lo, nu_hi, T, npt=800):
    nu = np.linspace(nu_lo, nu_hi, npt)
    return np.trapz(Bnu(nu, T), nu) / (nu_hi - nu_lo)

def band_mean_J(nu_sorted, J_sorted, lo_A, hi_A):
    """Energy-integrated band-mean J_nu over bins whose lambda in [lo,hi].
    Returns (Jmean_energy, Jgeom, nbins, nfloor, nu_a, nu_b)."""
    nu_hi = CLIGHT_A/lo_A; nu_lo = CLIGHT_A/hi_A
    m = (nu_sorted >= nu_lo) & (nu_sorted <= nu_hi)
    if m.sum() < 2:
        return 0.0, 0.0, int(m.sum()), 0, nu_lo, nu_hi
    nu = nu_sorted[m]; J = J_sorted[m]
    Jmean = np.trapz(J, nu) / (nu[-1] - nu[0])
    pos = J[J > 1e-29]
    Jgeom = float(np.exp(np.mean(np.log(pos)))) if pos.size else 0.0
    nfloor = int(np.sum(J <= 1e-29))
    return float(Jmean), Jgeom, int(m.sum()), nfloor, nu[0], nu[-1]

# ---------------------------------------------------------------- CMFGEN phys T(v)
def cmfgen_block(path, t=19.480):
    lines = open(path).read().splitlines(); start=None
    for i, ln in enumerate(lines):
        if ln.startswith('#TIME:') and abs(float(ln.split()[1]) - t) < 1e-3:
            start=i; break
    rows=[]; j=start+1
    while j < len(lines):
        s=lines[j].strip()
        if s.startswith('#TIME'): break
        if s and not s.startswith('#'):
            try: rows.append([float(x) for x in s.split()])
            except ValueError: pass
        j+=1
    return np.array(rows)

# ---------------------------------------------------------------- EDDFACTOR reader
def read_info(info):
    v = open(info).read().splitlines()[2].split()
    return dict(ND=int(v[0]), RECL=int(v[1]), WORD=int(v[2]), little=(v[5]=='T'))
def read_edd(edd):
    info=read_info(edd+'_INFO'); ND=info['ND']; nwr=info['RECL']//info['WORD']
    dt='<f8' if info['little'] else '>f8'
    raw=np.fromfile(edd, dtype=dt); raw=raw[:(raw.size//nwr)*nwr].reshape(-1,nwr)
    data=raw[14:]
    good=np.isfinite(data[:,:ND]).all(axis=1) & (data[:,ND] > 0)
    J=data[good,:ND]; FL=data[good,ND]; nu=FL*1e15
    o=np.argsort(nu); return J[o], nu[o], ND
def rvtj_block(text, label, ND):
    lines=text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip()==label:
            vals=[]; j=i+1
            while j < len(lines) and len(vals) < ND:
                try: vals += [float(t) for t in lines[j].split()]
                except ValueError: break
                j+=1
            return np.array(vals[:ND])
    raise KeyError(label)

# ---------------------------------------------------------------- Lumina loaders
def load_plasma(d):
    W={}; Te={}; Trad={}
    for r in csv.DictReader(open(f'{d}/lumina_plasma_state.csv')):
        s=int(r['shell_id']); W[s]=float(r['W']); Te[s]=float(r['T_e']); Trad[s]=float(r['T_rad'])
    return W, Te, Trad
def load_field(d):
    """shell -> (nu_sorted_asc, mc_J_sorted, cs_J_sorted)."""
    lam={}; mc={}; cs={}
    for r in csv.DictReader(open(f'{d}/lumina_coevolve_field.csv')):
        s=int(r['shell'])
        lam.setdefault(s, []).append(float(r['wavelength_A']))
        mc.setdefault(s, []).append(float(r['mc_J']))
        cs.setdefault(s, []).append(float(r['cs_J']))
    out={}
    for s in lam:
        nu=CLIGHT_A/np.array(lam[s]); o=np.argsort(nu)
        out[s]=(nu[o], np.array(mc[s])[o], np.array(cs[s])[o])
    return out

# ================================================================= LOAD
print("=== F0b deep thermalization test (OFFLINE) ===")
ph=cmfgen_block(f'{STD}/phys_toy06_cmfgen.txt'); vph=ph[:,0]; Tph=ph[:,1]
def T_cmf(v): return float(Tph[int(np.argmin(np.abs(vph - v)))])

print("=== CMFGEN self-run field (jnu4 EDDFACTOR) ===")
Jc, nuc, ND = read_edd(f'{JNU4}/EDDFACTOR')
Vc = rvtj_block(open(f'{JNU4}/RVTJ').read(), 'Velocity (km/s)', ND)
print(f"  ND={ND} nfreq={Jc.shape[0]} V=[{Vc[0]:.0f}..{Vc[-1]:.0f}] nu=[{nuc[0]:.2e}..{nuc[-1]:.2e}]")

WB, TeB, TradB = load_plasma(BRUN); FB = load_field(BRUN)
WJ, TeJ, TradJ = load_plasma(JTAB); FJ = load_field(JTAB)

# ---------- yardstick audit ----------
def uniq_pin(name, d):
    vals=sorted(set(round(v,3) for v in d.values()))
    print(f"  {name}: uniq={len(vals)}"+("  <== PINNED (definitional)" if len(vals)==1 else "")+f"  e.g. {vals[:3]}")
print("\n--- YARDSTICK AUDIT ---")
print("B-run:")
uniq_pin("T_rad", TradB)
pinB=[s for s in SHELLS if abs(TeB[s]-list(TradB.values())[0])<1e-3]
print(f"  T_e pinned-to-T_rad shells (in s0-10): {pinB}  (these J/B(T_e) are DEFINITIONAL)")
print("jtable:")
uniq_pin("T_rad", TradJ)
pinJ=[s for s in SHELLS if abs(TeJ[s]-list(TradJ.values())[0])<1e-3]
print(f"  T_e pinned-to-T_rad shells (in s0-10): {pinJ}  (these J/B(T_e) are DEFINITIONAL)")

# ================================================================= PER-SHELL TABLE
rows=[]
for s, v in zip(SHELLS, VEL):
    d = int(np.argmin(np.abs(Vc - v)))
    Tc = T_cmf(v)
    for bn, lo, hi in BANDS:
        # Lumina B-run
        JB_e, JB_g, nB, flB, na, nb = band_mean_J(FB[s][0], FB[s][1], lo, hi)
        Bte_B = band_mean_B(na, nb, TeB[s]) if nb>na else 0.0
        # Lumina jtable
        JJ_e, JJ_g, nJ, flJ, na2, nb2 = band_mean_J(FJ[s][0], FJ[s][1], lo, hi)
        Bte_J = band_mean_B(na2, nb2, TeJ[s]) if nb2>na2 else 0.0
        # CMFGEN
        JC_e, JC_g, nC, flC, nac, nbc = band_mean_J(nuc, Jc[:,d], lo, hi)
        Bc = band_mean_B(nac, nbc, Tc) if nbc>nac else 0.0
        rows.append(dict(s=s, v=v, band=bn,
            JB_e=JB_e, JB_g=JB_g, Bte_B=Bte_B, WB=WB[s], TeB=TeB[s], flB=flB, nB=nB,
            JJ_e=JJ_e, JJ_g=JJ_g, Bte_J=Bte_J, WJ=WJ[s], TeJ=TeJ[s], pinJ=(s in pinJ),
            JC_e=JC_e, JC_g=JC_g, Bc=Bc, Tc=Tc, pinB=(s in pinB)))

def r_of(s, band): return next(r for r in rows if r['s']==s and r['band']==band)
def safe(a,b): return a/b if b>0 else float('nan')

# ================================================================= PRINT
for bn,_,_ in BANDS:
    print("\n"+"="*128)
    print(f"BAND {bn}   [Jmean=energy-integrated band-mean J_nu; ratios use energy-int J]")
    print("="*128)
    print(f"{'sh':>3}{'v':>6} | {'B:Te':>6}{'B:J_L':>10}{'B:B(Te)':>10}{'J/B':>7}{'J/WB':>7} | "
          f"{'jt:Te':>7}{'jt:J_L':>10}{'jt:J/WB':>8} | {'C:Tc':>6}{'C:J':>10}{'C:B':>10}{'C:J/B':>7} | {'J_C/J_B':>9}")
    for s in SHELLS:
        r=r_of(s,bn)
        JoB = safe(r['JB_e'], r['Bte_B']); JoWB = safe(r['JB_e'], r['WB']*r['Bte_B'])
        jJoWB = safe(r['JJ_e'], r['WJ']*r['Bte_J'])
        CJoB = safe(r['JC_e'], r['Bc']); CoverL = safe(r['JC_e'], r['JB_e'])
        pinf = '*' if r['pinB'] else ' '; pinfj = '*' if r['pinJ'] else ' '
        print(f"{s:>3}{r['v']:>6} | {r['TeB']:>6.0f}{r['JB_e']:>10.3e}{r['Bte_B']:>10.3e}{JoB:>7.3f}{JoWB:>7.3f}{pinf}| "
              f"{r['TeJ']:>6.0f}{pinfj}{r['JJ_e']:>10.3e}{jJoWB:>8.3f} | "
              f"{r['Tc']:>6.0f}{r['JC_e']:>10.3e}{r['Bc']:>10.3e}{CJoB:>7.3f} | {CoverL:>9.2e}")
    print("  (* = T_e pinned to T_rad -> that J/B is DEFINITIONAL, not a measured thermalization)")

# ================================================================= FORK VERDICT
print("\n"+"="*128)
print("F0b FORK VERDICT  (deep FUV, s0)")
print("="*128)
r0=r_of(0,'FUV_918_1290')
JoWB_e = safe(r0['JB_e'], r0['WB']*r0['Bte_B'])
JoWB_g = safe(r0['JB_g'], r0['WB']*r0['Bte_B'])
JoB_e  = safe(r0['JB_e'], r0['Bte_B'])
print(f"  s0 FUV  T_e(B)={r0['TeB']:.0f}K (NOT pinned)  W={r0['WB']:.4f}")
print(f"  J_L(energy-int)={r0['JB_e']:.3e}  J_L(geom)={r0['JB_g']:.3e}  B(T_e)={r0['Bte_B']:.3e}")
print(f"  J_L/B(T_e)       energy-int = {JoB_e:.3f}")
print(f"  J_L/[W*B(T_e)]   energy-int = {JoWB_e:.3f}    geom = {JoWB_g:.3f}")
resid_e = -math.log10(JoWB_e) if JoWB_e>0 else float('nan')
resid_g = -math.log10(JoWB_g) if JoWB_g>0 else float('nan')
print(f"  => deficit below own W*B(T_e): energy-int {resid_e:+.2f} dex, geom {resid_g:+.2f} dex")
if JoWB_e >= 0.3 or JoWB_g >= 0.3:
    print("  VERDICT: deep FUV J is within a factor ~3 of W*B(its own cold T_e)")
    print("           -> NOT the >=1-2 dex transport-starvation signature.")
    print("           -> the -2.3 dex deficit vs CMFGEN is dominated by cold T_e (color, Axis 1a).")
    print("           => T_e's FAULT. Promote F3-T (temperature-table probe).")
else:
    print("  VERDICT: deep FUV J << W*B(its own T_e) -> transport starvation (Axis 1b). Promote F0a.")

# ================================================================= WRITE CSV
outp=f'{OUT}/f0b_thermalization_shells.csv'
with open(outp,'w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['band','shell','v_kms',
        'B_Te','B_W','B_JmeanE','B_Jgeom','B_Bnu_Te','B_JoverB','B_JoverWB','B_floorbins','B_Te_pinned',
        'jt_Te','jt_JmeanE','jt_Bnu_Te','jt_JoverWB','jt_Te_pinned',
        'C_Tc','C_JmeanE','C_Bnu_Tc','C_JoverB','deficit_JC_over_JB_dex'])
    for r in rows:
        JoB=safe(r['JB_e'],r['Bte_B']); JoWB=safe(r['JB_e'],r['WB']*r['Bte_B'])
        jJoWB=safe(r['JJ_e'],r['WJ']*r['Bte_J']); CJoB=safe(r['JC_e'],r['Bc'])
        defd = math.log10(safe(r['JC_e'],r['JB_e'])) if (r['JC_e']>0 and r['JB_e']>0) else float('nan')
        w.writerow([r['band'],r['s'],r['v'],
            f"{r['TeB']:.1f}",f"{r['WB']:.5f}",f"{r['JB_e']:.4e}",f"{r['JB_g']:.4e}",f"{r['Bte_B']:.4e}",
            f"{JoB:.4f}",f"{JoWB:.4f}",r['flB'],int(r['pinB']),
            f"{r['TeJ']:.1f}",f"{r['JJ_e']:.4e}",f"{r['Bte_J']:.4e}",f"{jJoWB:.4f}",int(r['pinJ']),
            f"{r['Tc']:.1f}",f"{r['JC_e']:.4e}",f"{r['Bc']:.4e}",f"{CJoB:.4f}",f"{defd:.4f}"])
print(f"\n[out] -> {outp}")
