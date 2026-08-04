#!/usr/bin/env python3
"""T0: model T_e/n_e vs DDC15 gold (CMFGEN) directly, vs velocity.
Usage: compare_plasma_vs_gold.py <run_dir>
  run_dir must contain lumina_plasma_state.csv and ref/geometry.csv
Gold = data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d (T in 10^4 K, n_e /cm^3, v km/s).
"""
import sys, numpy as np, csv, os
RUN = sys.argv[1] if len(sys.argv)>1 else '.'
GOLD = 'data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d'

lines = open(GOLD).read().split('\n')
def block(key, n=115):
    vals=[]; grab=False
    for L in lines:
        if key in L: grab=True; continue
        if grab:
            try: row=[float(x) for x in L.split()]
            except:
                if vals: break
                else: continue
            vals+=row
            if len(vals)>=n: break
    return np.array(vals[:n])
gv=block('Velocity (km/s)'); gT=block('Temperature (10^4 K)')*1e4; gne=block('Electron density')
o=np.argsort(gv); gv,gT,gne=gv[o],gT[o],gne[o]

mTe=[];mne=[]
for r in csv.DictReader(open(os.path.join(RUN,'lumina_plasma_state.csv'))):
    mTe.append(float(r['T_e'])); mne.append(float(r['n_e']))
vin=[];vout=[]
for r in csv.DictReader(open(os.path.join(RUN,'ref','geometry.csv'))):
    vin.append(float(r['v_inner'])); vout.append(float(r['v_outer']))
mTe=np.array(mTe); mne=np.array(mne)
mv=(np.array(vin)+np.array(vout))/2/1e5  # cm/s -> km/s

gT_i=np.interp(mv,gv,gT); gne_i=np.interp(mv,gv,gne)
rt=mTe/gT_i; rn=mne/gne_i
print(f"RUN={RUN}")
print(f"T0a T_e model/gold: median={np.median(rt):.3f}  photosphere(v<25k)={np.median(rt[mv<25000]):.3f}  outer(v>50k)={np.median(rt[mv>50000]):.3f}")
print(f"T0b n_e model/gold: median={np.median(rn):.3f}  photosphere={np.median(rn[mv<25000]):.3f}  outer={np.median(rn[mv>50000]):.2e}  dex-RMS={np.sqrt(np.mean(np.log10(np.clip(rn,1e-12,1e12))**2)):.2f}")
print("  v(km/s) goldTe modelTe Tratio | goldne modelne neratio")
for i in range(0,len(mTe),max(1,len(mTe)//10)):
    print(f"  {mv[i]:7.0f} {gT_i[i]:6.0f} {mTe[i]:6.0f} {rt[i]:.2f}  | {gne_i[i]:.2e} {mne[i]:.2e} {rn[i]:.3f}")
