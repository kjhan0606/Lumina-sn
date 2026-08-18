#!/usr/bin/env python3
"""STAGE4 ROUND2 -- f(IV) depth crossover: where is the deep drain enhancement
CMFGEN-warranted, and where does LTE already suffice?

Per-shell f(IV) for Fe/Co/Ni: CMFGEN truth (interp to Lumina v_mid) vs the
LTE-all-level baseline (gphall run) vs the round-1 blowup (stage4 run), with W.
Locates the depth-gate threshold for the round-2 recommendation.  Read-only.
"""
import csv,re,numpy as np
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
D=f"{REPO}/data/standart_data1/toy06"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/stage4_round2"
# uniform velocity grid: v_inner[0]=3900, v_outer[49]=40300 km/s (stdout.log:106)
vmid=np.array([3900+(i+0.5)*(40300-3900)/50 for i in range(50)])
W={}
for row in csv.DictReader(open(f"{REPO}/logs/coevolve_consume_a10_kx_stage4/lumina_plasma_state.csv")):
    W[int(row["shell_id"])]=float(row["W"])
def fiv(run,Z):
    tot={};niv={}
    for row in csv.DictReader(open(f"{REPO}/logs/coevolve_consume_a10_kx_{run}/lumina_ion_pops.csv")):
        if int(row["Z"])!=Z: continue
        s=int(row["shell_id"]);tot[s]=tot.get(s,0)+float(row["n_ion"])
        if int(row["stage"])==3: niv[s]=float(row["n_ion"])
    return {s:(niv.get(s,0)/tot[s] if tot[s]>0 else 0) for s in tot}
def cmf(elem,st):
    L=open(f"{D}/ionfrac_{elem}_toy06_cmfgen.txt").read().splitlines()
    times=[];starts=[]
    for i,l in enumerate(L):
        m=re.match(r'#TIME:\s*([\d.]+)',l)
        if m: times.append(float(m.group(1)));starts.append(i)
    k=int(np.argmin(abs(np.array(times)-19.48)));i0=starts[k];i1=starts[k+1]
    cols=None;vel=[];dat=[]
    for l in L[i0:i1]:
        if l.startswith('#vel_mid'):cols=l.strip().lstrip('#').split()[1:];continue
        if l.startswith('#'):continue
        try: vals=[float(x) for x in l.split()]
        except: continue
        vel.append(vals[0]);dat.append(vals[1:])
    vel=np.array(vel);dat=np.array(dat);d={c:dat[:,j] for j,c in enumerate(cols)}
    tot=sum(d[f'{elem}{s}'] for s in range(6) if f'{elem}{s}' in d)
    return vel,d[f'{elem}{st}']/tot
rows=[["Z","elem","shell","v_km_s","W","cmfgen_fIV","gphall_fIV_LTE","stage4_fIV_blowup"]]
for elem,Z in [("fe",26),("co",27),("ni",28)]:
    gp=fiv("gphall",Z); s4=fiv("stage4",Z); vc,fc=cmf(elem,3)
    for s in range(50):
        rows.append([Z,elem.upper(),s,f"{vmid[s]:.0f}",f"{W.get(s,float('nan')):.4f}",
                     f"{np.interp(vmid[s],vc,fc):.4f}",
                     f"{gp.get(s,float('nan')):.4f}",f"{s4.get(s,float('nan')):.4f}"])
with open(f"{OUT}/fiv_depth_crossover.csv","w",newline="") as f: csv.writer(f).writerows(rows)
print(f"[out] {OUT}/fiv_depth_crossover.csv  ({len(rows)-1} rows)")
# print the crossover summary
print("\nCo crossover (W where CMFGEN f(IV) drops below the LTE baseline):")
gp=fiv("gphall",27); vc,fc=cmf("co",3)
for s in range(12):
    c=np.interp(vmid[s],vc,fc); g=gp.get(s,0)
    tag="DRAIN-NEEDED" if c>g+0.02 else ("LTE-OK" if abs(c-g)<0.05 else "LTE-over")
    print(f"  s{s:2d} v={vmid[s]:5.0f} W={W.get(s,0):.3f}  CMFGEN={c:.3f} LTE(gphall)={g:.3f}  {tag}")
