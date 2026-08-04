#!/usr/bin/env python3
"""3-way toy06 19.48d emergent-spectrum compare: Lumina vs CMFGEN vs ARTIS, feature-level.
Usage: cmp3_artis.py <run_dir>"""
import sys, csv
import numpy as np
run = sys.argv[1]
lw=[];lf=[]
for r in csv.DictReader(open(f"{run}/lumina_spectrum_formal.csv")):
    lw.append(float(r['wavelength_angstrom'])); lf.append(float(r['flux']))
lw=np.array(lw); lf=np.array(lf); o=np.argsort(lw); lw,lf=lw[o],lf[o]
def load(path,col):
    wl=[];fx=[]
    for line in open(path):
        if line.startswith("#"): continue
        p=line.split(); wl.append(float(p[0])); fx.append(float(p[col]))
    return np.array(wl),np.array(fx)
cwl,cfx=load("data/standart_data1/toy06/spectra_toy06_cmfgen.txt",1+26)   # 19.48d
awl,afx=load("data/standart_data1/toy06/spectra_toy06_artis.txt",77)      # 19.61d
grid=np.linspace(2500,10000,750)
L=np.interp(grid,lw,lf); C=np.interp(grid,cwl,cfx); A=np.interp(grid,awl,afx)
def nrm(f): return f/np.trapz(f,grid)
Ln,Cn,An=nrm(L),nrm(C),nrm(A)
def frac(f,a,b): m=(grid>=a)&(grid<b); return 100*float(np.trapz(f[m],grid[m])/np.trapz(f,grid))
print(f"=== {run} : Lumina vs CMFGEN vs ARTIS (toy06 19.48d) ===")
print(f"peak: L {grid[np.argmax(L)]:.0f}  C {grid[np.argmax(C)]:.0f}  A {grid[np.argmax(A)]:.0f}")
print(f"corr(L,C)={np.corrcoef(Ln,Cn)[0,1]:.3f}  corr(L,A)={np.corrcoef(Ln,An)[0,1]:.3f}  corr(C,A)={np.corrcoef(Cn,An)[0,1]:.3f}")
print(f"{'band':16} {'Lum':>7} {'CMF':>7} {'ART':>7} | {'L-C':>6} {'L-A':>6}")
for nm,a,b in [("UV 2500-3000",2500,3000),("UVb 3000-3500",3000,3500),("blue 3500-5000",3500,5000),
               ("green 5000-6500",5000,6500),("red 6500-9000",6500,9000)]:
    fl,fc,fa=frac(L,a,b),frac(C,a,b),frac(A,a,b)
    print(f"{nm:16} {fl:6.1f}% {fc:6.1f}% {fa:6.1f}% | {fl-fc:+6.1f} {fl-fa:+6.1f}")
# feature windows (where is Lumina most off vs CMFGEN)
print("--- 250A-window flux fraction: |Lum-CMFGEN| top divergences ---")
edges=np.arange(2500,9000,250); div=[]
for e in edges:
    m=(grid>=e)&(grid<e+250)
    div.append((abs(frac(L,e,e+250)-frac(C,e,e+250)), int(e), frac(L,e,e+250), frac(C,e,e+250), frac(A,e,e+250)))
for d,e,fl,fc,fa in sorted(div,reverse=True)[:6]:
    print(f"  [{e}-{e+250}]A: L {fl:.1f}% C {fc:.1f}% A {fa:.1f}%  (|L-C|={d:.1f})")
