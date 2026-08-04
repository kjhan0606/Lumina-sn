#!/usr/bin/env python3
"""Differential emergent-spectrum comparison: line-resolved fluorescence (B) vs
binned (A) vs DDC15 gold. The MC blue-tilt + other absolute confounds are
COMMON to A and B, so B-A isolates the fluorescence effect; gold gives the
shape target. Usage: compare_fluor_spectrum.py A_virtual.csv B_virtual.csv gold.dat out.png
"""
import sys, numpy as np, csv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def load_csv(f):
    w=[];fl=[]
    for r in csv.reader(open(f)):
        try: w.append(float(r[0])); fl.append(float(r[1]))
        except: pass
    return np.array(w), np.array(fl)
def load_gold(f):
    w=[];fl=[]
    for line in open(f):
        p=line.split()
        if len(p)>=2:
            try: w.append(float(p[0])); fl.append(float(p[1]))
            except: pass
    return np.array(w), np.array(fl)

wa,fa = load_csv(sys.argv[1]); wb,fb = load_csv(sys.argv[2]); wg,fg = load_gold(sys.argv[3])
out = sys.argv[4] if len(sys.argv)>4 else 'figures/fluor_spectrum.png'

# resample all to a common optical grid
grid = np.arange(3500, 9000, 5.0)
def rs(w,fl):
    m = fl>0
    return np.interp(grid, w[m], fl[m], left=0, right=0)
Fa,Fb,Fg = rs(wa,fa), rs(wb,fb), rs(wg,fg)
# normalize each to its mean over 4000-8000 (shape comparison; absolute differs)
band=(grid>=4000)&(grid<=8000)
def nrm(F): m=F[band].mean(); return F/m if m>0 else F
na,nb,ng = nrm(Fa),nrm(Fb),nrm(Fg)

# blue/red color metric
def color(F):
    b=F[(grid>=4000)&(grid<5000)].mean(); r=F[(grid>=6000)&(grid<7000)].mean()
    return b/r if r>0 else np.nan
print(f"blue/red (4000-5000 / 6000-7000):  A(binned)={color(Fa):.3f}  B(line-res)={color(Fb):.3f}  GOLD={color(Fg):.3f}")
print(f"  -> gold is {'BLUER' if color(Fg)>color(Fa) else 'redder'} than binned; does B move toward gold? {'YES' if color(Fb)>color(Fa) else 'no'}")
# where does B differ from A (fluorescence footprint)?
diff = nb - na
for lo,hi,nm in [(4000,5000,'blue'),(4400,4600,'4475'),(5000,6000,'green'),(6500,6700,'6590'),(6000,7000,'red')]:
    m=(grid>=lo)&(grid<hi)
    print(f"  {nm:5s} {lo}-{hi}: B/A flux ratio = {nb[m].mean()/max(na[m].mean(),1e-30):.3f}  (>1 = fluorescence raised this band)")

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11,8),sharex=True,height_ratios=[2,1])
ax1.plot(grid,ng,label='DDC15 gold',color='k',lw=1.5)
ax1.plot(grid,na,label='A: binned-J',color='#5bc0de',lw=1)
ax1.plot(grid,nb,label='B: line-resolved fluorescence',color='#d9534f',lw=1)
ax1.set_ylabel('normalized flux (mean 4000-8000=1)'); ax1.legend(); ax1.set_title('Emergent spectrum: line-resolved fluorescence vs binned vs gold')
ax1.set_ylim(0,max(3,np.percentile(ng,99)*1.2))
ax2.plot(grid,nb-na,color='#d9534f',lw=1); ax2.axhline(0,color='gray',ls='--')
ax2.set_ylabel('B - A (fluor effect)'); ax2.set_xlabel('wavelength (A)')
for x in (4475,6590): ax1.axvline(x,color='gold',ls=':',alpha=0.6); ax2.axvline(x,color='gold',ls=':',alpha=0.6)
plt.tight_layout(); plt.savefig(out,dpi=110); print('saved',out)
