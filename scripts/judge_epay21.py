#!/usr/bin/env python3
"""5-criteria judgment for the Kirchhoff per-bin EPAY run (epay21).
Usage: python3 scripts/judge_epay21.py [rundir]"""
import sys, csv, collections, math
import numpy as np
H=6.62607015e-27; KB=1.380649e-16; EV=1.602176634e-12; C=2.99792458e10
SOB=0.02654; T_EXP=19.48*86400
run = sys.argv[1] if len(sys.argv)>1 else 'logs/stage1_toy06_epay21'

# --- 1. plasma: hot band + inner/valley vs epay7 baseline ---
ps={int(r['shell_id']):(float(r['T_e']),float(r['n_e']),float(r['T_rad']),float(r['W']))
    for r in csv.DictReader(open(f'{run}/lumina_plasma_state.csv'))}
base={int(r['shell_id']):float(r['T_e'])
      for r in csv.DictReader(open('logs/stage1_toy06_epay7/lumina_plasma_state.csv'))}
band=[ps[s][0] for s in range(36,41)]
print(f"[1] hot band s36-40: mean={np.mean(band):.0f} max={max(band):.0f} "
      f"({'DEAD ✓' if max(band)<25000 else 'RE-IGNITED ✗'})")
dev=[(s,ps[s][0]/base[s]-1) for s in range(50)]
big=[(s,d) for s,d in dev if abs(d)>0.15]
print(f"    vs epay7: shells with |dT|>15%: {[(s,round(d,2)) for s,d in big][:8]}")

# --- 2. J(16.35 eV) at s5/8/12 ---
d=np.genfromtxt(f'{run}/lumina_cmfgen_jnu.csv',delimiter=',',names=True)
for s in [5,8,12]:
    sub=d[d['shell']==s]; ev=sub['nu']*H/EV
    i=np.argmin(np.abs(ev-16.35))
    T=ps[s][0]; nu=16.35*EV/H
    B=2*H*nu**3/C**2*math.exp(-H*nu/(KB*T))
    print(f"[2] J(16.35eV,s{s}) = {sub['J'][i]:.2e}  (B(T_e)={B:.2e}, ratio {sub['J'][i]/B:.2f}; epay20 was ~1e-11-e-12)")

# --- 3. blanket tau (Si II 5981 / Co II 5821 at s5,s8) ---
lev=collections.defaultdict(dict)
for r in csv.DictReader(open('data/tardis_reference_toy06_19p48d/levels.csv')):
    lev[(int(r['atomic_number']),int(r['ion_number']))][int(r['level_number'])]=(float(r['energy_eV']),float(r['g']))
def U_of(key,T):
    E=np.array([v[0] for v in lev[key].values()]); g=np.array([v[1] for v in lev[key].values()])
    return float(np.sum(g*np.exp(-np.minimum(E*EV/(KB*T),300))))
lm={}
for r in csv.DictReader(open('data/tardis_reference_toy06_19p48d/line_list.csv')):
    k=(int(r['atomic_number']),int(r['ion_number']),round(float(r['wavelength']),1))
    if k not in lm: lm[k]=(float(r['f_lu']),int(r['level_number_lower']),float(r['wavelength_cm']))
pops=collections.defaultdict(dict)
for r in csv.DictReader(open('lumina_ion_pops.csv')):
    pops[(int(r['shell_id']),int(r['Z']))][int(r['stage'])]=float(r['n_ion'])
for (Z,ion,wl) in [(14,1,5980.6),(27,1,5820.9),(16,1,5661.6)]:
    f_lu,lo,lam_cm=lm[(Z,ion,wl)]
    row=[]
    for s in [5,8,12]:
        Trad,W=ps[s][2],ps[s][3]
        n_ion=pops.get((s,Z),{}).get(ion,0.0)
        E_lo,g_lo=lev[(Z,ion)][lo]
        U=U_of((Z,ion),Trad)
        n_lo=n_ion*W*g_lo*math.exp(-min(E_lo*EV/(KB*Trad),300))/U
        tau=SOB*f_lu*lam_cm*T_EXP*n_lo
        row.append(f"s{s}:{tau:.1f}")
    print(f"[3] tau Z{Z}i{ion}@{wl}: " + ' '.join(row) + "  (epay20: 1928/806, 1078, 1954; CMFGEN: 0-7)")

# --- 4+5 run compare_narrowband separately on MC csv ---
print("[4/5] run: python3 scripts/compare_narrowband.py "
      f"{run}/lumina_spectrum.csv epay21_MC")
