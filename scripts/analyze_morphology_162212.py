#!/usr/bin/env python3
"""Morphology defect analysis for run 162212 (scatter mode) vs SN 2002bo +0d.
Zoom [3700,5400]: 3900 peak, missing 5100 peak, spurious 5200 trough.
Per-ion opacity decomposition from reference tau_sobolev + line_list."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_dbj_A_162212"
OBS = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/sn2002bo/epochs/sn2002bo_m0d0.csv"
FIG = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/figures/2026-06-03_162212_morphology_3700-5400_vs_2002bo.png"

# --- model spectrum (formal = emergent flux; lumina_spectrum is MC binned) ---
mod = np.loadtxt(f"{RUN}/lumina_spectrum_formal.csv", delimiter=",", skiprows=1)
mw, mf = mod[:,0], mod[:,1]
mf = mf / mw**2          # F_nu -> F_lambda shape (c factor absorbed by normalization)
# also MC spectrum
mc = np.loadtxt(f"{RUN}/lumina_spectrum.csv", delimiter=",", skiprows=1)
mcw, mcf = mc[:,0], mc[:,1]
mcf = mcf / mcw**2

# --- observed ---
_rows=[]
for ln in open(OBS):
    ln=ln.strip()
    if not ln or ln.startswith("#") or ln.startswith("wavelength"): continue
    a,b=ln.split(",")[:2]; _rows.append((float(a),float(b)))
obs=np.array(_rows)
ow, of = obs[:,0], obs[:,1]

# restrict to overlap for normalization
def norm_band(w, f, lo=4000, hi=8000):
    m = (w>=lo)&(w<=hi)
    return np.trapz(f[m], w[m])

# normalize model (both) to obs over a broad common band
A_obs = norm_band(ow, of)
mf_n  = mf  * A_obs/norm_band(mw, mf)
mcf_n = mcf * A_obs/norm_band(mcw, mcf)

# --- defect metrics ---
def peakval(w,f,lo,hi):
    m=(w>=lo)&(w<=hi)
    i=np.argmax(f[m]); return w[m][i], f[m][i]
# 3900 peak
o_wl, o_pk = peakval(ow, of, 3800, 4050)
m_wl, m_pk = peakval(mw, mf_n, 3800, 4050)
print(f"[3900] obs peak  wl={o_wl:.1f} val={o_pk:.3e}")
print(f"[3900] model pk  wl={m_wl:.1f} val={m_pk:.3e}")
print(f"[3900] height ratio model/obs = {m_pk/o_pk:.3f}")
print(f"[3900] wl offset model-obs = {m_wl-o_wl:.1f} A (negative=bluer)")
# 5100 peak (obs)
o5_wl,o5_pk = peakval(ow,of,5000,5250)
m5 = (mw>=5000)&(mw<=5250)
print(f"[5100] obs peak wl={o5_wl:.1f}; model local max in [5000,5250]={mf_n[m5].max():.3e} vs obs {o5_pk:.3e}")
# 5200 trough
def troughval(w,f,lo,hi):
    m=(w>=lo)&(w<=hi); i=np.argmin(f[m]); return w[m][i],f[m][i]
mt_wl,mt_v = troughval(mw,mf_n,5100,5350)
ot_wl,ot_v = troughval(ow,of,5100,5350)
print(f"[5200] model trough wl={mt_wl:.1f} v={mt_v:.3e}; obs min wl={ot_wl:.1f} v={ot_v:.3e}")

# --- per-ion opacity decomposition in [3700,5400] ---
ll = np.genfromtxt(f"{RUN}/ref/line_list.csv", delimiter=",", names=True)
tau = np.load(f"{RUN}/ref/tau_sobolev.npy")  # (Nlines,30)
lam = ll["wavelength"]
Z   = ll["atomic_number"].astype(int)
ion = ll["ion_number"].astype(int)
# shell 0 (photosphere) tau
tau0 = tau[:, 0]
ELEM={1:'H',6:'C',8:'O',11:'Na',12:'Mg',13:'Al',14:'Si',16:'S',20:'Ca',
      21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni'}
ROM={1:'I',2:'II',3:'III',4:'IV'}
flu = ll["f_lu"]
# strongest-tau line shell0 AND f_lu-weighted line density (carrier identification)
def ion_in_window(lo,hi):
    m=(lam>=lo)&(lam<=hi)
    dt={}; df={}
    for zz,ii,tt,ff in zip(Z[m],ion[m],tau0[m],flu[m]):
        k=(zz,ii)
        dt[k]=dt.get(k,0.0)+tt
        df[k]=df.get(k,0.0)+ff
    return dt,df
for name,lo,hi in [("3900 peak",3800,4000),("5100 region",5000,5200),("5200 trough",5150,5300)]:
    dt,df=ion_in_window(lo,hi)
    print(f"\n=== {name} [{lo},{hi}]: top ions by tau_sobolev(shell0) | by Sigma f_lu ===")
    print("  -- by tau(shell0) --")
    for (zz,ii),tt in sorted(dt.items(),key=lambda kv:-kv[1])[:5]:
        print(f"     {ELEM.get(zz,zz)} {ROM.get(ii,ii)}: tau_sum={tt:.3e}")
    print("  -- by Sigma f_lu (line strength density) --")
    for (zz,ii),ff in sorted(df.items(),key=lambda kv:-kv[1])[:5]:
        print(f"     {ELEM.get(zz,zz)} {ROM.get(ii,ii)}: Sigma_flu={ff:.3e}")

# --- plot ---
fig,ax=plt.subplots(figsize=(13,6))
ax.plot(ow,of,'k-',lw=1.6,label='SN 2002bo +0d (obs)')
ax.plot(mw,mf_n,'-',color='crimson',lw=1.3,label='162212 formal (scatter, norm)')
ax.plot(mcw,mcf_n,'-',color='royalblue',lw=0.8,alpha=0.6,label='162212 MC binned')
ax.set_xlim(3700,5400)
band=(ow>=3700)&(ow<=5400)
ax.set_ylim(0, max(of[band].max(), mf_n[(mw>=3700)&(mw<=5400)].max())*1.15)
for wl,txt,col in [(3900,'(a) 3900 peak\ntoo tall/blue','orange'),
                   (5100,'(b) missing\n5100 peak','green'),
                   (5200,'(c) spurious\n5200 trough','purple')]:
    ax.axvline(wl,color=col,ls='--',alpha=0.6)
    ax.text(wl,ax.get_ylim()[1]*0.92,txt,color=col,fontsize=9,ha='center',va='top')
ax.set_xlabel('Wavelength (A)'); ax.set_ylabel('Flux (norm)')
ax.set_title('Run 162212 (scatter) line-shape defects vs SN 2002bo +0d')
ax.legend(loc='upper right',fontsize=9)
plt.tight_layout(); plt.savefig(FIG,dpi=130)
print(f"\nSAVED {FIG}")
