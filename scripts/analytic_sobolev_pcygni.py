#!/usr/bin/env python3
# Analytic Sobolev P-Cygni for the R2 toy (thin shell + Ca II triplet), the
# DEBUG TRUTH for S4/MC. Homologous: iso-velocity surfaces = planes z=v_obs*t_exp.
# Saturated line -> jump I = I*e^-tau + S_l*(1-e^-tau) ~ S_l at each resonance plane
# crossed within the shell along each tangent ray; disk integral F=int I p dp.
# Overlay logs/toy/R2/cmf_sobolev_v5.csv (S4) + mc_spectrum.csv (MC).
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=2.99792458e10; h=6.62607015e-27; k=1.380649e-16
t_exp=84326.4; r_in=1.6e14; r_out=1.68e14; T_inner=4430.0; T_e=4430.0; W=0.5
lines=[8498e-8,8542e-8,8662e-8]; tauS=8.5e6
Bnu=lambda nu,T:(2*h*nu**3/C**2)/(np.exp(h*nu/(k*T))-1.0)
Bc=Bnu(C/8542e-8,T_inner); Sl=W*Bnu(C/8542e-8,T_e); ex=np.exp(-tauS)
lamA=np.linspace(7600,9300,500); lam=lamA*1e-8
NP=3000; pg=np.linspace(r_out/NP*0.5,r_out,NP); dp=pg[1]-pg[0]
F=np.zeros_like(lam)
for il,lo in enumerate(lam):
    s=0.0
    for p in pg:
        core=p<r_in; zin=np.sqrt(max(r_in*r_in-p*p,0.)); zout=np.sqrt(max(r_out*r_out-p*p,0.)) if p<r_out else 0.
        if zout<=0: continue
        zres=[C*(1.-lo/ll)*t_exp for ll in lines]
        I=0.0
        for zr in sorted(zres):       # far side z<0 (emission)
            if -zout<zr<-(zin if core else 0.): I=I*ex+Sl*(1-ex)
        if core: I=Bc
        for zr in sorted(zres):       # near side z>0 (absorption/emission)
            if (zin if core else 0.)<zr<zout: I=I*ex+Sl*(1-ex)
        s+=I*p*dp
    F[il]=s
def load(f):
    d=np.genfromtxt(f,delimiter=',',names=True);n=d.dtype.names;l,fl=d[n[0]],d[n[1]];o=np.argsort(l)
    return np.interp(lamA,l[o],fl[o])
def nc(F):
    b=np.mean(F[(lamA>7620)&(lamA<7720)]);r=np.mean(F[(lamA>9150)&(lamA<9280)])
    return F/(b+(r-b)*(lamA-7670)/(9215-7670))
fig,ax=plt.subplots(figsize=(13,7))
ax.plot(lamA,nc(F),'k',lw=2.6,label='ANALYTIC Sobolev (truth)')
ax.plot(lamA,nc(load("logs/toy/R2/cmf_sobolev_v5.csv")),'g',lw=1.6,label='S4')
ax.plot(lamA,nc(load("logs/toy/R2/mc_spectrum.csv")),'r',lw=1.4,alpha=.8,label='MC')
ax.set_xlim(7600,9300);ax.set_ylim(0.4,1.2);ax.grid(alpha=.3);ax.legend();ax.axvline(8542,color='purple',ls=':',lw=.7)
ax.set_xlabel('wavelength [A]');ax.set_ylabel('flux / local continuum')
plt.tight_layout();plt.savefig('figures/2026-06-23_R2_analytic_clean.png',dpi=115);print("saved")
