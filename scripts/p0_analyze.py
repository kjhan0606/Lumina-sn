#!/usr/bin/env python3
"""P0 falsifier analysis: does forcing S II presence light the green / drop UV?
Usage: p0_analyze.py <run_dir> [label]
Reads run_dir/lumina_ion_pops.csv (f(II)), lumina_levelpop.csv (b_k),
lumina_spectrum_formal.csv (bands). Compares to CMFGEN toy06 19.48d."""
import sys, csv, numpy as np
run = sys.argv[1]; label = sys.argv[2] if len(sys.argv)>2 else run

# ---- CMFGEN reference ion fractions at 19.48d (stage col s1 = II) ----
def cmfgen_fII(elem):
    path=f"data/standart_data1/toy06/ionfrac_{elem}_toy06_cmfgen.txt"
    lines=open(path).read().splitlines()
    # find block "#TIME:  19.48"
    i=0; vel=[]; fII=[]
    while i<len(lines):
        if lines[i].startswith("#TIME:") and abs(float(lines[i].split(":")[1])-19.48)<0.05:
            i+=1
            while i<len(lines) and not lines[i].startswith("#vel"): i+=1
            i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith("#"):
                p=lines[i].split()
                vel.append(float(p[0])); fII.append(float(p[2]))  # p[1]=s0(I) p[2]=s1(II)
                i+=1
            break
        i+=1
    return np.array(vel), np.array(fII)

# ---- Lumina f(II) per shell from ion_pops (stage is 0-based: 0=I,1=II,2=III) ----
def lumina_fII(Z):
    tot={}; sec={}
    try:
        for r in csv.DictReader(open(f"{run}/lumina_ion_pops.csv")):
            if int(r['Z'])!=Z: continue
            s=int(r['shell_id']); n=float(r['n_ion']); st=int(r['stage'])
            tot[s]=tot.get(s,0)+n
            if st==1: sec[s]=sec.get(s,0)+n   # stage 1 = singly-ionized (II)
    except FileNotFoundError:
        return None
    return {s: (sec.get(s,0)/tot[s] if tot[s]>0 else 0) for s in sorted(tot)}

# ---- Lumina b_k (median low levels L1-7, Rydberg-safe) ----
def lumina_bk(Z,ion):
    vals=[]
    try:
        for r in csv.DictReader(open(f"{run}/lumina_levelpop.csv")):
            if int(r['Z'])==Z and int(r['ion'])==ion and 1<=int(r['level_num'])<=7:
                b=float(r['b_k'])
                if 0<b<1e3: vals.append(b)
    except FileNotFoundError:
        return float('nan')
    return float(np.median(vals)) if vals else float('nan')

print(f"\n========== P0: {label}  ({run}) ==========")
elems={'s':(16,'S II'),'si':(14,'Si II'),'ca':(20,'Ca II')}
print(f"{'ion':6} | {'CMFGEN f(II) range':22} | Lumina f(II) @ shells 6/9/15/25/33 | b_k")
for e,(Z,nm) in elems.items():
    cv,cf=cmfgen_fII(e)
    # CMFGEN line-forming velocity window ~ 9000-16000 km/s
    m=(cv>=9000)&(cv<=16000)
    crange=f"{cf[m].min():.2f}-{cf[m].max():.2f}" if m.any() else "n/a"
    lf=lumina_fII(Z)
    if lf:
        cells=" ".join(f"{lf.get(s,float('nan')):.2f}" for s in [6,9,15,25,33])
    else:
        cells="(no ion_pops)"
    bk=lumina_bk(Z,2)
    print(f"{nm:6} | {crange:22} | {cells:34} | {bk:.2f}")

# ---- spectrum bands vs CMFGEN ----
def load(path,col):
    wl=[];fx=[]
    for line in open(path):
        if line.startswith("#"): continue
        p=line.split(); wl.append(float(p[0])); fx.append(float(p[col]))
    return np.array(wl),np.array(fx)
try:
    lw=[];lf=[]
    for r in csv.DictReader(open(f"{run}/lumina_spectrum_formal.csv")):
        lw.append(float(r['wavelength_angstrom'])); lf.append(float(r['flux']))
    lw=np.array(lw);lf=np.array(lf);o=np.argsort(lw);lw,lf=lw[o],lf[o]
    cwl,cfx=load("data/standart_data1/toy06/spectra_toy06_cmfgen.txt",1+26)
    grid=np.linspace(2500,10000,750)
    L=np.interp(grid,lw,lf);C=np.interp(grid,cwl,cfx)
    Ln=L/np.trapz(L,grid);Cn=C/np.trapz(C,grid)
    def frac(f,a,b):m=(grid>=a)&(grid<b);return 100*float(np.trapz(f[m],grid[m])/np.trapz(f,grid))
    corr=np.corrcoef(Ln,Cn)[0,1]
    uv=frac(L,2500,3000);grn=frac(L,5000,6500)
    print(f"spectrum: corr(L,C)={corr:.3f}  UV[2500-3000]={uv:.1f}% (C 8.5, target<15)  "
          f"green[5000-6500]={grn:.1f}% (C 22.7, target>18)")
    print(f"VERDICT: {'green UP + UV DOWN => ionization is binding ✓' if (grn>12 and uv<20) else 'no move => re-plan' if grn<9 else 'partial'}")
except FileNotFoundError:
    print("(no spectrum yet)")
