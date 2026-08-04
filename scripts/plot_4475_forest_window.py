#!/usr/bin/env python3
"""4475 forest-window verdict figure: scatter A/B + macroatom vs gold.
Shows (a) full-optical SED comparison, (b) 4475 zoom with O II opacity windows,
(c) the two-knob bar chart (window contrast vs color)."""
import numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BG=(0x14/255,0x0D/255,0x44/255); CARD=(0x20/255,0x18/255,0x58/255)
BLUE=(0x38/255,0x98/255,0xEC/255); CORAL=(0xD9/255,0x77/255,0x57/255)
TEAL=(0x4E/255,0xC9/255,0xB0/255); GOLD=(0xFF/255,0xC1/255,0x07/255)
WHITE=(0xFA/255,0xF9/255,0xF5/255); DIM=(0x70/255,0x7E/255,0x9A/255)

def load(p):
    w=[];f=[]
    for ln in open(p):
        s=ln.replace(',',' ').split()
        if len(s)<2: continue
        try: a=float(s[0]); b=float(s[1])
        except: continue
        w.append(a); f.append(b)
    w=np.array(w); f=np.array(f)
    if w[0]>w[-1]: w=w[::-1]; f=f[::-1]
    return w,f

R='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
runs={
 'GOLD (DDC15 0.976d CMFGEN)':(f'{R}/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat',GOLD,2.4),
 'SCATTER hot BB (T_inner 9677K)':(f'{R}/logs/ddc15_pc_phase3_jnul1_radls1_linere1_ratio1.0_pi1_fz1_167721/lumina_spectrum.csv',BLUE,1.4),
 'SCATTER cool BB (T_inner 4434K)':(f'{R}/logs/ddc15_pc_phase3_jnul1_radls1_linere1_ratio1.0_pi1_fz1_167722/lumina_spectrum.csv',TEAL,1.4),
 'MACROATOM (fluorescence)':(f'{R}/logs/ddc15_pc_phase3_jnul1_radls1_linere1_ratio1.0_pi1_fz1_167694/lumina_spectrum.csv',CORAL,1.6),
}
data={}
for nm,(p,c,lw) in runs.items():
    if os.path.exists(p): data[nm]=(load(p),c,lw)

# normalize each to unit mean over 4000-9000 for shape comparison
def norm(w,f,a=4000,b=9000):
    m=(w>=a)&(w<b); s=np.trapz(f[m],w[m])
    return f/ s if s>0 else f

# O II opacity windows from line list (zero-line bins): 4500-4550, 4900-4950
WINDOWS=[(4500,4550),(4900,4950)]

fig=plt.figure(figsize=(15,9)); fig.patch.set_facecolor(BG)
gs=GridSpec(2,2,figure=fig,height_ratios=[1,1],hspace=0.32,wspace=0.22,
            left=0.07,right=0.97,top=0.90,bottom=0.09)

def smooth(f,n=15):
    if n<=1: return f
    k=np.ones(n)/n
    return np.convolve(f,k,'same')

# ---- (a) full optical SED, shape-normalized ----
axA=fig.add_subplot(gs[0,:]); axA.set_facecolor(CARD)
for nm,((w,f),c,lw) in data.items():
    fn=norm(w,f)
    sm = 1 if nm.startswith('GOLD') else 21
    fn=smooth(fn,sm)
    m=(w>=3500)&(w<=10000)
    axA.plot(w[m],fn[m],color=c,lw=lw+0.4,label=nm,alpha=0.95)
axA.axvline(4475,color=WHITE,ls=':',lw=1,alpha=0.5)
axA.text(4475,axA.get_ylim()[1]*0.92,'4475',color=WHITE,fontsize=9,ha='center')
axA.axvspan(8500,9600,color=CORAL,alpha=0.08)
axA.text(9050,axA.get_ylim()[1]*0.80,'our peak\n~8500-9500 (too red)',color=CORAL,fontsize=8.5,ha='center')
axA.set_xlim(3500,10000)
axA.set_xlabel('wavelength [Å]',color=WHITE); axA.set_ylabel('shape-normalized flux',color=WHITE)
axA.set_title('(a)  Optical SED — gold is blue-weighted, our models too-red',color=WHITE,fontsize=12,loc='left')
leg=axA.legend(facecolor=BG,edgecolor=DIM,fontsize=9,loc='upper right')
for t in leg.get_texts(): t.set_color(WHITE)

# ---- (b) 4475 zoom, with O II windows ----
axB=fig.add_subplot(gs[1,0]); axB.set_facecolor(CARD)
for nm,((w,f),c,lw) in data.items():
    fn=norm(w,f,4200,5200)
    sm = 1 if nm.startswith('GOLD') else 9
    fn=smooth(fn,sm)
    m=(w>=4100)&(w<=5200)
    axB.plot(w[m],fn[m],color=c,lw=lw+0.4,alpha=0.95)
for (lo,hi) in WINDOWS:
    axB.axvspan(lo,hi,color=GOLD,alpha=0.12)
axB.text(4525,axB.get_ylim()[1]*0.95,'O II window\n4500-4550',color=GOLD,fontsize=8,ha='center',va='top')
axB.text(4925,axB.get_ylim()[1]*0.95,'O II window\n4900-4950',color=GOLD,fontsize=8,ha='center',va='top')
axB.axvline(4475,color=WHITE,ls=':',lw=1,alpha=0.6)
axB.axvline(4990,color=CORAL,ls=':',lw=1,alpha=0.6)
axB.set_xlim(4100,5200)
axB.set_xlabel('wavelength [Å]',color=WHITE); axB.set_ylabel('normalized flux (4200-5200)',color=WHITE)
axB.set_title('(b)  4475 zoom: gold peaks at the BLUE window,\n     macroatom at the RED window (4990)',color=WHITE,fontsize=11,loc='left')

# ---- (c) two-knob bars ----
axC=fig.add_subplot(gs[1,1]); axC.set_facecolor(CARD)
labels=['knob1:\n4475-window\ncontrast','knob2:\ncolor\nwin4475/win4950']
gold_vals=[1.87,9.36]; our_vals=[2.22,0.99]
x=np.arange(2); width=0.34
axC.bar(x-width/2,gold_vals,width,color=GOLD,label='GOLD',alpha=0.95)
axC.bar(x+width/2,our_vals,width,color=CORAL,label='OURS (macroatom)',alpha=0.95)
for i,(g,o) in enumerate(zip(gold_vals,our_vals)):
    axC.text(i-width/2,g+0.15,f'{g:.2f}',color=GOLD,ha='center',fontsize=10,fontweight='bold')
    axC.text(i+width/2,o+0.15,f'{o:.2f}',color=CORAL,ha='center',fontsize=10,fontweight='bold')
axC.axhline(1.0,color=DIM,ls='--',lw=0.8)
axC.set_xticks(x); axC.set_xticklabels(labels,color=WHITE,fontsize=9)
axC.set_ylabel('ratio',color=WHITE); axC.set_ylim(0,10.5)
axC.set_title('(c)  Two knobs separated:\n     window OK (knob1), color is the ~9× defect (knob2)',color=WHITE,fontsize=11,loc='left')
leg=axC.legend(facecolor=BG,edgecolor=DIM,fontsize=9,loc='upper left')
for t in leg.get_texts(): t.set_color(WHITE)
axC.annotate('window reproduced\n(2.22 vs 1.87)',xy=(0,2.22),xytext=(0.35,4.3),color=TEAL,fontsize=8.5,
             ha='center',arrowprops=dict(color=TEAL,arrowstyle='->',lw=1.2))
axC.annotate('too-red:\nblue window\nnot lit',xy=(1+width/2,0.99),xytext=(1.25,5.0),color=BLUE,fontsize=8.5,
             ha='center',arrowprops=dict(color=BLUE,arrowstyle='->',lw=1.2))

for ax in (axA,axB,axC):
    ax.tick_params(colors=WHITE);
    for sp in ax.spines.values(): sp.set_color(DIM)

fig.suptitle('gold 0.976d 4475Å  =  O II forest WINDOW (not fluorescence emission, not Mg II) — defect is too-red COLOR',
             color=WHITE,fontsize=13.5,fontweight='bold',y=0.965)
out=f'{R}/figures/2026-06-21_4475_forest_window_verdict.png'
os.makedirs(f'{R}/figures',exist_ok=True)
fig.savefig(out,dpi=130,facecolor=BG,bbox_inches='tight')
print("saved",out)
