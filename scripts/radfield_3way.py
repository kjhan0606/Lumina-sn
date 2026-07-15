#!/usr/bin/env python3
"""3-way UV field judge: Lumina mc_J / cs_J vs ARTIS radfield J_nu (toy06 19.48d).
ARTIS whitebox radfield_000*.out: per-cell per-bin J estimator, ts27=19.42-21.13d.
Grid is 1:1 with Lumina shells (mid v = 4264 + 728*mgi for both).
Usage: radfield_3way.py [lumina_run_dir] [ts]
"""
import sys, csv, glob
import numpy as np

C=2.99792458e10
ART='/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_whitebox_run'
LUM=sys.argv[1] if len(sys.argv)>1 else 'logs/coevolve_consume_a10_kx_gphground'
TS=int(sys.argv[2]) if len(sys.argv)>2 else 27

# ---- ARTIS radfield: {mgi: [(nu_lo,nu_hi,J_nu_avg,ncontrib)]} ----
art={}
for fn in glob.glob(f'{ART}/radfield_*.out'):
    for ln in open(fn):
        p=ln.split()
        if not p or p[0]=='timestep': continue
        try: ts=int(p[0])
        except: continue
        if ts!=TS: continue
        mgi=int(p[1]); b=int(p[2])
        if b<0: continue                 # bin -1 = full integral row
        # data rows have 10 fields (no ncontrib): ts mgi bin nu_lo nu_hi nuJ J J_nu_avg T_R W
        nu_lo=float(p[3]); nu_hi=float(p[4]); javg=float(p[7]); nc=999
        art.setdefault(mgi,[]).append((nu_lo,nu_hi,javg,nc))
for m in art: art[m].sort()

# ---- Lumina field ----
lum={}
for r in csv.DictReader(open(f'{LUM}/lumina_coevolve_field.csv')):
    s=int(r['shell'])
    lum.setdefault(s,[]).append((float(r['wavelength_A']),float(r['cs_J']),float(r['mc_J'])))

def art_J(mgi,lam_A):
    nu=C/(lam_A*1e-8)
    for nu_lo,nu_hi,j,nc in art.get(mgi,[]):
        if nu_lo<=nu<nu_hi:
            wide=(nu_hi/nu_lo>2.0)
            return j,nc,wide,(nu_lo,nu_hi)
    return None,0,False,None

def lum_J(s,lam_A):
    rows=lum.get(s)
    if not rows: return None,None
    lam=np.array([x[0] for x in rows])
    i=int(np.argmin(np.abs(lam-lam_A)))
    return rows[i][1],rows[i][2]   # cs_J, mc_J

WLS=[1000,1100,1200,1300,1500,2000,3000,5000]
print(f"# 3-way field: Lumina({LUM}) vs ARTIS radfield ts{TS} (19.42-21.13d bracket of 19.48d)")
print(f"# grid: shell s == mgi (mid v = 4264+728*s km/s). ratios = Lumina/ARTIS J_nu")
hdr=f"{'sh':>3} {'v':>6} |" + "".join(f"{w:>9}A" for w in WLS)
print(hdr+"   (mc_J/ART)")
for s in [4,6,8,10,12,15,20,25,30]:
    cells=[]
    for w in WLS:
        aj,nc,wide,rng=art_J(s,w)
        cj,mj=lum_J(s,w)
        if aj is None or aj<=0 or mj is None:
            cells.append(f"{'--':>10}"); continue
        mark='w' if wide else (' ' if nc>=10 else '?')
        cells.append(f"{mj/aj:>9.2f}{mark}")
    v=4264+728*s
    print(f"{s:>3} {v:>6} |"+"".join(cells))
print()
print(hdr+"   (cs_J/ART)")
for s in [4,6,8,10,12,15,20,25,30]:
    cells=[]
    for w in WLS:
        aj,nc,wide,rng=art_J(s,w)
        cj,mj=lum_J(s,w)
        if aj is None or aj<=0 or cj is None:
            cells.append(f"{'--':>10}"); continue
        mark='w' if wide else (' ' if nc>=10 else '?')
        cells.append(f"{cj/aj:>9.2f}{mark}")
    v=4264+728*s
    print(f"{s:>3} {v:>6} |"+"".join(cells))
print("\n# marks: 'w'=ARTIS bin wideband(nu_hi/nu_lo>2, smeared; 913A falls there), '?'=ncontrib<10 (noisy)")
print("# absolute J sanity (s10):")
for w in [1000,1200,1500,3000]:
    aj,nc,wide,rng=art_J(10,w)
    cj,mj=lum_J(10,w)
    print(f"  {w}A: ARTIS={aj:.3e}(n={nc}) mc_J={mj:.3e} cs_J={cj:.3e}")
