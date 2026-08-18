#!/usr/bin/env python3
"""plot_rT.py -- r(T) bistability curves (frozen vs coupled) for s8/s6/s2."""
import csv, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/bistability_test"
CMFTe={2:16351.43,6:11929.14,8:10383.0}
KPRTe={2:19479.6,6:13362.1,8:12207.8}
d=collections.defaultdict(list)
for r in csv.DictReader(open(f"{OUT}/rT_curves.csv")):
    d[int(r['s'])].append((float(r['T']),float(r['rA']),float(r['rB']),float(r['f4B'])))
shells=[8,6,2]
fig,axes=plt.subplots(2,3,figsize=(15,8))
for i,s in enumerate(shells):
    rows=sorted(d[s]); T=np.array([x[0] for x in rows]); rA=np.array([x[1] for x in rows])
    rB=np.array([x[2] for x in rows]); f4=np.array([x[3] for x in rows])
    ax=axes[0][i]
    ax.axhline(0,color='k',lw=0.8)
    ax.plot(T,rA,'o-',ms=3,label='(a) frozen ioniz',color='#3898EC')
    ax.plot(T,rB,'s-',ms=3,label='(b) coupled re-solve',color='#D97757')
    ax.axvline(CMFTe[s],color='#4EC9B0',ls='--',lw=1.2,label=f'CMFGEN {CMFTe[s]:.0f}K')
    ax.axvline(KPRTe[s],color='#FFC107',ls=':',lw=1.2,label=f'kpr5 {KPRTe[s]:.0f}K')
    ax.set_title(f's{s}  r(T)=H-C'); ax.set_xlabel('T_e [K]'); ax.set_ylabel('residual [erg/cm3/s]')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax2=axes[1][i]
    ax2.plot(T,f4,'d-',ms=3,color='#8858C8')
    ax2.axvline(CMFTe[s],color='#4EC9B0',ls='--',lw=1.2)
    ax2.axvline(KPRTe[s],color='#FFC107',ls=':',lw=1.2)
    ax2.axhline(0.022,color='r',ls='-',lw=0.8,label='CMFGEN f(FeIV)=0.022')
    ax2.set_title(f's{s} coupled f(FeIV)(T)'); ax2.set_xlabel('T_e [K]'); ax2.set_ylabel('f(FeIV)')
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3); ax2.set_ylim(-0.05,1.05)
plt.tight_layout(); plt.savefig(f"{OUT}/rT_bistability.png",dpi=110)
print(f"[out] {OUT}/rT_bistability.png")
