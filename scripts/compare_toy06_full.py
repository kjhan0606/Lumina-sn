#!/usr/bin/env python3
"""toy06 19.48d: Lumina run vs CMFGEN — T_e / n_e profiles + emergent spectrum.
Usage: compare_toy06_full.py <run_dir e.g. logs/stage1_toy06_fix2> [label]"""
import sys, csv, math
import numpy as np

run=sys.argv[1]; label=sys.argv[2] if len(sys.argv)>2 else run.split('_')[-1]
BASE="data/tardis_reference_toy06_19p48d"

# --- Lumina profiles ---
sid=[];Te=[];ne=[]
for r in csv.DictReader(open(f"{run}/lumina_plasma_state.csv")):
    sid.append(int(r['shell_id'])); Te.append(float(r['T_e'])); ne.append(float(r['n_e']))
Te=np.array(Te); ne=np.array(ne)
vmid=[]
for r in csv.DictReader(open(f"{BASE}/geometry.csv")):
    vmid.append(0.5*(float(r['v_inner'])+float(r['v_outer']))/1e5)
vmid=np.array(vmid)

# --- CMFGEN truth at 19.48 ---
cv=[];cT=[];cne=[]
intime=False
for line in open("data/standart_data1/toy06/phys_toy06_cmfgen.txt"):
    if line.startswith("#TIME:"):
        intime=abs(float(line.split()[1])-19.48)<0.01; continue
    if line.startswith("#") or not line.strip(): continue
    if intime:
        p=line.split(); cv.append(float(p[0])); cT.append(float(p[1])); cne.append(float(p[3]))
cv=np.array(cv); cT=np.array(cT); cne=np.array(cne)
cT_i=np.interp(vmid,cv,cT); cne_i=np.interp(vmid,cv,cne)

print(f"===== {label}: T_e / n_e vs CMFGEN (19.48d) =====")
print(f"{'s':>3} {'v':>7} | {'T_e':>7} {'T_cmf':>7} {'ratio':>6} | {'n_e':>9} {'ne_cmf':>9} {'dex':>6}")
for s in [0,5,10,15,20,25,30,35,40,43,46,49]:
    dex=math.log10(ne[s]/cne_i[s]) if ne[s]>0 and cne_i[s]>0 else float('nan')
    print(f"{s:>3} {vmid[s]:7.0f} | {Te[s]:7.0f} {cT_i[s]:7.0f} {Te[s]/cT_i[s]:6.2f} | {ne[s]:9.2e} {cne_i[s]:9.2e} {dex:6.2f}")
rms=float(np.sqrt(np.mean((Te/cT_i-1)**2)))
rms_o=float(np.sqrt(np.mean((Te[35:]/cT_i[35:]-1)**2)))
dex_all=float(np.sqrt(np.mean((np.log10(ne/np.maximum(cne_i,1e-30)))**2)))
print(f"T_e %RMS all={100*rms:.1f}%  outer(s>=35)={100*rms_o:.1f}%   n_e RMS dex={dex_all:.3f}")

# --- spectrum ---
try:
    lw=[];lf=[]
    for r in csv.DictReader(open(f"{run}/lumina_spectrum_formal.csv")):
        lw.append(float(r['wavelength_angstrom'])); lf.append(float(r['flux']))
    lw=np.array(lw); lf=np.array(lf)
    o=np.argsort(lw); lw,lf=lw[o],lf[o]
    cwl=[];cfx=[]
    for line in open("data/standart_data1/toy06/spectra_toy06_cmfgen.txt"):
        if line.startswith("#"): continue
        p=line.split()
        cwl.append(float(p[0])); cfx.append(float(p[1+26]))  # 19.48d = col 26
    cwl=np.array(cwl); cfx=np.array(cfx)
    grid=np.linspace(2500,24000,800)
    L=np.interp(grid,lw,lf); Cf=np.interp(grid,cwl,cfx)
    def frac(f,a,b): m=(grid>=a)&(grid<b); return float(np.trapz(f[m],grid[m])/np.trapz(f,grid))
    print(f"\n--- spectrum (2500-24000A rebin) ---")
    print(f"peak λ:      Lumina {grid[np.argmax(L)]:.0f}  CMFGEN {grid[np.argmax(Cf)]:.0f}")
    for nm,a,b in [("UV 2500-3500",2500,3500),("blue 3500-5000",3500,5000),
                   ("green 5000-6500",5000,6500),("red 6500-9000",6500,9000),
                   ("NIR 9000-24000",9000,24000)]:
        print(f"{nm:16s}: L {100*frac(L,a,b):5.1f}%  C {100*frac(Cf,a,b):5.1f}%")
    Ln=L/np.trapz(L,grid); Cn=Cf/np.trapz(Cf,grid)
    corr=float(np.corrcoef(Ln,Cn)[0,1])
    lum_ratio=float(np.trapz(L,grid)/np.trapz(Cf,grid))
    print(f"shape corr={corr:.3f}   L_total ratio(L/C)={lum_ratio:.2f}")
except Exception as e:
    print(f"(spectrum skip: {e})")
