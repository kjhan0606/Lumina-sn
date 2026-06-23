#!/usr/bin/env python3
# Falsifier: does the analytic Sobolev P-Cygni, when fed the SAME weak-Ca-line
# forest as S4 (all lines tau>1e-6 from allcalines.csv), reproduce S4's red
# emission at ~8956? If yes -> forest is physical, S4 correct, analytic-3-line
# was just incomplete. If no -> S4 has a real weak-line bug.
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=2.99792458e10; h=6.62607015e-27; k=1.380649e-16
t_exp=84326.4; r_in=1.6e14; r_out=1.68e14; inv_ct=1.0/(C*t_exp); T_inner=4430.0
Bnu=lambda nu,T:(2*h*nu**3/C**2)/(np.exp(h*nu/(k*T))-1.0)

# load firing lines (rest A, tau) ; restrict to those that can resonate in 7700-9300
L=np.genfromtxt("logs/toy/R2/allcalines.csv",delimiter=",")
L=L[L[:,1]>1e-4]                                  # drop negligible (each <0.01%)
L=L[np.argsort(L[:,0])]
rest=L[:,0]*1e-8; tau=L[:,1]                      # cm, tauS
print(f"forest lines: {len(rest)}  (tau range {tau.min():.1e}..{tau.max():.1e})")

def Wdil(r):
    if r<=r_in: return 0.5
    a=1.0-(r_in*r_in)/(r*r); return 0.5*(1.0-np.sqrt(a if a>0 else 0.0))

nu_line_all=None  # set per call
def spec(lines_rest, lines_tau, NLAM=600, NP=4000):
    lamA=np.linspace(7600,9300,NLAM); pg=np.linspace(r_out/NP*0.5,r_out,NP); dp=pg[1]-pg[0]
    F=np.zeros(NLAM)
    nu_line=C/lines_rest                              # per-line rest freq (=nu_comoving at resonance)
    for il in range(NLAM):
        lo=lamA[il]*1e-8; s=0.0
        for p in pg:
            core=p<r_in
            zl=np.sqrt(max(r_in*r_in-p*p,0.)) if core else 0.0
            zh=np.sqrt(max(r_out*r_out-p*p,0.)) if p<r_out else 0.0
            if zh<=0: continue
            zres=C*t_exp*(1.0-lo/lines_rest)          # array
            I=0.0
            order=np.argsort(zres)                     # ascending z (outer->inner on far side)
            for j in order:
                z=zres[j]
                if (-zh<z< -zl):
                    r=np.sqrt(p*p+z*z); mu=z/r; beta=r*inv_ct
                    q=(1-mu*beta)/np.sqrt(1-beta*beta); D=1.0/q
                    # S4-faithful source: W(r)*B(nu_comoving=nu_line, T_inner)
                    ex=np.exp(-lines_tau[j]); Sl=Wdil(r)*Bnu(nu_line[j],T_inner)
                    I=I*ex+(D**3)*Sl*(1-ex)
            if core:
                zc=np.sqrt(r_in*r_in-p*p); b=r_in*inv_ct
                qc=(1-zc/r_in*b)/np.sqrt(1-b*b); I=(1.0/qc)**3*Bnu(qc*C/lo,T_inner)
            for j in order:
                z=zres[j]
                if ((zl if core else 0.0)<z<zh):
                    r=np.sqrt(p*p+z*z); mu=z/r; beta=r*inv_ct
                    q=(1-mu*beta)/np.sqrt(1-beta*beta); D=1.0/q
                    ex=np.exp(-lines_tau[j]); Sl=Wdil(r)*Bnu(nu_line[j],T_inner)
                    I=I*ex+(D**3)*Sl*(1-ex)
            s+=I*p*dp
        F[il]=s
    return lamA,F

# (b) full forest (same as S4 default) + its OWN line-free continuum
la,Ff=spec(rest,tau)
la,Fcont_ana=spec(np.array([]),np.array([]))       # analytic continuum = core disk only
# UNIFORM normalization recipe applied identically to ALL methods (apple-to-apple):
# divide by a linear envelope through the same two line-free windows.
def ncont(l,F):
    bl=np.mean(F[(l>7620)&(l<7720)]); rr=np.mean(F[(l>9180)&(l<9290)])
    return F/(bl+(rr-bl)*(l-7670)/(9235-7670))
def load(f):d=np.genfromtxt(f,delimiter=',',names=True);n=d.dtype.names;l,fl=d[n[0]],d[n[1]];o=np.argsort(l);return l[o],fl[o]
lf,Sf=load("logs/toy/R2/clean_lined.csv")          # S4 full forest (raw L_lam)
lm,Fm=load("logs/toy/R2/mc_spectrum.csv")          # MC (raw L_lam, erg/s/cm)
# TRUEST apple-to-apple: divide each by its OWN continuum (no shared envelope).
lcc,Fcc=load("logs/toy/R2/clean_cont.csv")         # S4 true continuum-only run
A =Ff/Fcont_ana                                    # analytic / its core-disk continuum
S =np.interp(la,lf,Sf)/np.interp(la,lcc,Fcc)       # S4 / S4 continuum
M =ncont(la,np.interp(la,lm,Fm))                   # MC: no cont run -> envelope (caveat)
import os
Shi=None
if os.path.exists("logs/toy/R2/s4_hires.csv"):
    lh,Fh=load("logs/toy/R2/s4_hires.csv"); Shi=ncont(la,np.interp(la,lh,Fh))

fig,ax=plt.subplots(figsize=(13,7))
ax.plot(la,A,'r',lw=2.4,label='ANALYTIC + forest (Sobolev truth)')
ax.plot(la,S,'g',lw=1.7,alpha=.7,label='S4 (NRAY=256, DVRES=30)')
if Shi is not None: ax.plot(la,Shi,color='darkgreen',lw=2.0,ls='--',label='S4 hi-res (NRAY=1024, DVRES=8)')
ax.plot(la,M,'b',lw=1.3,alpha=.7,label='MC (TARDIS-like)')
ax.axhline(1,color='gray',lw=.6);ax.axvline(8664.5,color='purple',ls=':',lw=.8)
ax.set_xlim(7700,9300);ax.set_ylim(0.55,1.15);ax.grid(alpha=.3);ax.legend(fontsize=11,loc='lower right')
ax.set_xlabel('wavelength [Angstrom]');ax.set_ylabel('flux / own continuum')
ax.set_title('Apple-to-apple: same Ca lines, same source W(r)B(nu_line), each / its own continuum')
plt.tight_layout();plt.savefig('figures/2026-06-23_R2_apple_3way.png',dpi=120)
def ft(l,F,lo=8560,hi=9250):
    a=(l>7800)&(l<8500);p=(l>lo)&(l<hi);return F[a].min(),l[a][F[a].argmin()],F[p].max(),l[p][F[p].argmax()]
rows=[("ANALYTIC",A),("S4-256",S),("MC",M)]
if Shi is not None: rows.insert(2,("S4-hires",Shi))
for lab,F in rows:
    t,tw,pk,pw=ft(la,F);print(f"  {lab:9s}: trough={t:.3f}@{tw:.0f}  emission_peak={pk:.3f}@{pw:.0f}")
print("saved figures/2026-06-23_R2_apple_3way.png")
