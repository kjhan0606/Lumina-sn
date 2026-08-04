#!/usr/bin/env python3
"""CMF formal (paper-method NLTE line source fn, converged run 160908) vs
scatter baseline (160767), overlaid on SN 2002bo B-max.

Both anchor-normalized on [4000,6000]A against the dereddened observed spectrum.
The CMF spectrum (lumina_spectrum_cmf.csv) uses the NLTE two-level line source
function S_l = (2hv^3/c^2)/(g_u n_l/(g_l n_u) - 1) -- fluorescence-bearing,
the paper's actual mechanism -- NOT the old S=J coherent-scatter.
Phase (1) test: does following the paper's CMF map reproduce 2002bo B-max?
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
CMF  = f"{ROOT}/logs/paperDDC15_paperli_einsteinFix_scatter_2002bo_vi9019_L1p0_prod_160908/lumina_spectrum_cmf.csv"
SCAT = f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_ofix_160767/lumina_spectrum_formal.csv"
OUT  = f"{ROOT}/figures/2026-05-31_cmf_nlte_source_vs_scatter_sn2002bo.png"

EBV, RV = 0.41, 3.1; A_V = RV*EBV

def ccm(w):
    x = 1e4/w; a=np.zeros_like(x); b=np.zeros_like(x)
    s=(x>=1.1)&(x<=3.3); y=x[s]-1.82
    a[s]=1+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
    b[s]=1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
    s=(x>=0.3)&(x<1.1); a[s]=0.574*x[s]**1.61; b[s]=-0.527*x[s]**1.61
    s=(x>3.3)&(x<=8.0); xs=x[s]
    Fa=np.where(xs>=5.9,-0.04473*(xs-5.9)**2-0.009779*(xs-5.9)**3,0.0)
    Fb=np.where(xs>=5.9,0.2130*(xs-5.9)**2+0.1207*(xs-5.9)**3,0.0)
    a[s]=1.752-0.316*xs-0.104/((xs-4.67)**2+0.341)+Fa
    b[s]=-3.090+1.825*xs+1.206/((xs-4.62)**2+0.263)+Fb
    return a+b/RV

obs = pd.read_csv(f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam = obs["wavelength_angstrom"].values
oflu = obs["flux_erg_s_cm2_angstrom"].values * 10**(0.4*A_V*ccm(olam))
ALO, AHI = 4000., 6000.
I_obs = float(np.trapezoid(oflu[(olam>=ALO)&(olam<=AHI)], olam[(olam>=ALO)&(olam<=AHI)]))

def load(p):
    m=pd.read_csv(p); lam=m["wavelength_angstrom"].values; fr=m["flux"].values
    sa=(lam>=ALO)&(lam<=AHI); K=I_obs/float(np.trapezoid(fr[sa],lam[sa]))
    return lam, fr*K

clam, cflu = load(CMF)
slam, sflu = load(SCAT)

# --- Blondin+2013 Fig 6 / Table 3 metric: anchor-pin [4000,6000] => F_scl==1,
#     Q_uvoir = Imod_UVOIR / Iobs_UVOIR over [1500,10200]; paper |F_scl-Q_uvoir|=0.01 ---
UVOIR_LO, UVOIR_HI = 1500., 10200.
so_uv = (olam >= max(UVOIR_LO, olam[0])) & (olam <= UVOIR_HI)
I_obs_uvoir = float(np.trapezoid(oflu[so_uv], olam[so_uv]))
def blondin_metric(lam, flu, name):
    sm = (lam >= UVOIR_LO) & (lam <= UVOIR_HI)
    I_mod = float(np.trapezoid(flu[sm], lam[sm]))
    Q = I_mod / I_obs_uvoir
    s_uvonly = (lam >= 1500) & (lam <= 3356)
    s_all = (lam >= 1500) & (lam <= 10200)
    fuv_frac = float(np.trapezoid(flu[s_uvonly], lam[s_uvonly])) / max(float(np.trapezoid(flu[s_all], lam[s_all])), 1e-30)
    print(f"  [{name}] F_scl=1.000 Q_uvoir={Q:.3f} |F_scl-Q_uvoir|={abs(1.0-Q):.3f} (paper 0.01)  FUV[1500,3356]frac={fuv_frac*100:.1f}% (paper-ok ~15%)")
    return Q
print("=== Blondin Fig 6 / Table 3 SED metric (obs starts at %.0fA) ===" % olam[0])
Qs = blondin_metric(slam, sflu, "scatter 160767")
Qc = blondin_metric(clam, cflu, "CMF 160908")

RUNS = [("scatter baseline 160767 (S=coherent), Q_uvoir=%.2f"%Qs, slam, sflu, "darkorange"),
        ("CMF NLTE source 160908 (paper-method), Q_uvoir=%.2f"%Qc, clam, cflu, "navy")]

BANDS=[("UV [1800,3000]",1800,3000),("blue [3000,3700]",3000,3700),
       ("4300 peak [4250,4400]",4250,4400),("opt [4400,5700]",4400,5700),
       ("Si6355 [5700,6500]",5700,6500),("red [6500,7400]",6500,7400),
       ("CaNIR [7400,8800]",7400,8800),("NIR2 [9000,12000]",9000,12000)]
print(f"{'band':24s}{'scatter/obs':>16s}{'CMF/obs':>16s}")
for ttl,lo,hi in BANDS:
    so=(olam>=lo)&(olam<=hi)
    Io=float(np.trapezoid(oflu[so],olam[so])) if so.sum()>1 else float('nan')
    sm=(slam>=lo)&(slam<=hi); Is=float(np.trapezoid(sflu[sm],slam[sm]))
    cm=(clam>=lo)&(clam<=hi); Ic=float(np.trapezoid(cflu[cm],clam[cm]))
    print(f"{ttl:24s}{Is/Io:>16.3f}{Ic/Io:>16.3f}")

fig,ax=plt.subplots(figsize=(13,7))
ax.plot(olam,oflu,lw=1.1,color="black",alpha=0.85,label="SN 2002bo B-max (dereddened, E(B-V)=0.41)")
for lbl,lam,flu,c in RUNS:
    ax.plot(lam,flu,lw=1.0,color=c,alpha=0.85,label=lbl)
ax.set_xlim(1800,12000); ax.set_xlabel("wavelength [A]"); ax.set_ylabel("flux (anchor-norm [4000,6000])")
ax.axvspan(ALO,AHI,color="gray",alpha=0.08)
ax.set_title("Phase-1 CMF (NLTE line source fn) vs scatter baseline -- SN 2002bo B-max")
ax.legend(loc="upper right",fontsize=9)
fig.tight_layout(); fig.savefig(OUT,dpi=130)
print(f"\nsaved {OUT}")
