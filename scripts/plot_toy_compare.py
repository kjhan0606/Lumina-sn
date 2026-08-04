import numpy as np, sys, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
tag=sys.argv[1]; d=f"logs/toy/{tag}"
def load(f):
    if not os.path.exists(f): return None,None
    a=np.genfromtxt(f,delimiter=',',names=True); return a['wavelength_angstrom'],a['flux']
cl,cf=load(f"{d}/cmf_spectrum.csv"); ml,mf=load(f"{d}/mc_spectrum.csv")
fig,ax=plt.subplots(1,1,figsize=(12,6))
if cl is not None: ax.plot(cl,cf/np.nanmax(cf[cf>0]) if np.any(cf>0) else cf,color='steelblue',lw=1.5,label='pure-CMFGEN (comoving)')
if ml is not None: ax.plot(ml,mf/np.nanmax(mf[mf>0]) if np.any(mf>0) else mf,color='crimson',lw=1.2,alpha=0.8,label='MC macro-atom')
for w,c in [(4475,'green'),(6590,'purple')]: ax.axvspan(w-80,w+80,color=c,alpha=0.1)
ax.set_xlim(2500,10000); ax.set_xlabel('wavelength [Å]'); ax.set_ylabel('normalized flux')
ax.legend(); ax.grid(alpha=0.3); ax.set_title(f'Toy {tag}: pure-CMFGEN vs MC macro-atom (peak-normalized)')
plt.tight_layout(); out=f"/tmp/toy_{tag}_compare.png"; plt.savefig(out,dpi=110)
print(f"saved {out}")
# 수치 요약
for nm,l,f in [("CMF",cl,cf),("MC",ml,mf)]:
    if l is None: print(f"  {nm}: (없음)"); continue
    fin=np.isfinite(f); print(f"  {nm}: n={fin.sum()}, max={np.nanmax(f):.2e}, optical(4000-7000) frac={np.nansum(f[(l>4000)&(l<7000)])/np.nansum(f[fin]):.2f}")
