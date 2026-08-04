#!/usr/bin/env python3
"""First observational-validation harness: DDC15 17.02d gold (B-max DDT model)
vs SN 2011fe B-max (HST UV + optical, phase +0.4d). Shape comparison (band-
normalized); focus on the UV 2000-3500A iron-curtain band that breaks the
optical degeneracy. NO LUMINA run / no IC build — establishes the harness."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BG=(0x14/255,0x0D/255,0x44/255); CARD=(0x20/255,0x18/255,0x58/255)
BLUE=(0x38/255,0x98/255,0xEC/255); CORAL=(0xD9/255,0x77/255,0x57/255)
TEAL=(0x4E/255,0xC9/255,0xB0/255); GOLD=(0xFF/255,0xC1/255,0x07/255)
WHITE=(0xFA/255,0xF9/255,0xF5/255); DIM=(0x70/255,0x7E/255,0x9A/255)
R='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'

def load(p, sep=None):
    w=[];f=[]
    for ln in open(p):
        if ln[0] in '#wW': continue
        s=ln.replace(',',' ').split()
        if len(s)<2: continue
        try: a=float(s[0]); b=float(s[1])
        except: continue
        w.append(a); f.append(b)
    w=np.array(w); f=np.array(f)
    o=np.argsort(w); return w[o], f[o]

wg,fg = load(f'{R}/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_017.020d.dat')   # 10pc flux
wo,fo = load(f'{R}/data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv')           # observed flux

def band_int(w,f,a,b):
    m=(w>=a)&(w<b)
    return np.trapz(f[m],w[m]) if m.sum()>1 else np.nan

# normalize model to obs by optical band-integral (4500-5800) — shape comparison
norm = band_int(wo,fo,4500,5800)/band_int(wg,fg,4500,5800)
fg_n = fg*norm

def smooth(w,f,frac=0.004):
    # simple running mean in index space ~ resolution match
    n=max(3,int(len(f)*frac)); k=np.ones(n)/n
    return np.convolve(f,k,'same')

# resample model onto obs grid for residuals
fg_i = np.interp(wo, wg, fg_n)
def rms(a,b):  # log-RMS over a band (shape)
    m=(a>0)&(b>0);
    return np.sqrt(np.mean((np.log10(a[m])-np.log10(b[m]))**2)) if m.sum()>2 else np.nan
def band_metric(lo,hi):
    m=(wo>=lo)&(wo<hi)
    r = band_int(wo,fg_i,lo,hi)/band_int(wo,fo,lo,hi)  # model/obs band ratio
    return r, rms(fg_i[m], fo[m])

bands = [('NUV (HST iron curtain)',2000,3300),('blue',3300,4500),
         ('optical',4500,5800),('red',5800,7500),('NIR',7500,9000)]
print("=== DDC15 17.02d  vs  SN2011fe B-max (band-normalized at 4500-5800) ===")
print(f"{'band':28s} {'model/obs':>10s} {'logRMS':>8s}")
for nm,lo,hi in bands:
    r,e=band_metric(lo,hi); print(f"{nm:28s} {r:10.2f} {e:8.3f}")

# ---- figure ----
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,9),facecolor=BG,
                            gridspec_kw={'height_ratios':[2,1],'hspace':0.28})
for ax in (ax1,ax2): ax.set_facecolor(CARD)
mo=(wo>=1600)&(wo<=9500)
ax1.plot(wo[mo], smooth(wo,fo)[mo], color=WHITE, lw=1.8, label='SN 2011fe B-max (HST UV+opt, +0.4d)')
mg=(wg>=1600)&(wg<=9500)
ax1.plot(wg[mg], smooth(wg,fg_n)[mg], color=GOLD, lw=1.6, label='DDC15 17.02d gold (norm @4500-5800)', alpha=0.9)
ax1.axvspan(2000,3300,color=BLUE,alpha=0.10)
ax1.text(2650, ax1.get_ylim()[1]*0.9,'NUV\niron curtain\n(breaks degeneracy)',color=BLUE,fontsize=9,ha='center',va='top')
ax1.set_xlim(1600,9500); ax1.set_yscale('log')
ax1.set_ylabel('flux (band-normalized, log)',color=WHITE)
ax1.set_title('DDC15 delayed-detonation (17.02d, B-max) vs SN 2011fe at B-max — does a real DDT model match the UV?',
              color=WHITE,fontsize=12.5,loc='left')
leg=ax1.legend(facecolor=BG,edgecolor=DIM,fontsize=10,loc='lower right')
for t in leg.get_texts(): t.set_color(WHITE)

# residual ratio
ratio = fg_i/np.maximum(fo,1e-30)
ax2.plot(wo[mo], smooth(wo,ratio)[mo], color=TEAL, lw=1.5)
ax2.axhline(1.0,color=DIM,ls='--',lw=0.8)
ax2.axvspan(2000,3300,color=BLUE,alpha=0.10)
ax2.set_xlim(1600,9500); ax2.set_ylim(0.1,10); ax2.set_yscale('log')
ax2.set_xlabel('wavelength [Å]',color=WHITE); ax2.set_ylabel('model / obs',color=WHITE)
ax2.set_title('residual ratio (1 = perfect; UV deficit = model over-absorbed by iron curtain)',color=WHITE,fontsize=10,loc='left')
for ax in (ax1,ax2):
    ax.tick_params(colors=WHITE)
    for sp in ax.spines.values(): sp.set_color(DIM)
out=f'{R}/figures/2026-06-21_ddc15_17d_vs_sn2011fe_bmax.png'
fig.savefig(out,dpi=130,facecolor=BG,bbox_inches='tight')
print("saved",out)
