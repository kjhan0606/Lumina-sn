#!/usr/bin/env python3
"""PART 2 -- stage4 failure mechanics.
(a) b_k of Co IV lev 50/144 and Fe III comb, stage4 vs gphall, s0 & s8.
(b) ionization f(IV) for Fe/Co/Ni/Si, stage4 vs gphall, s0 & s8 (photospheric blowup).
(d) Ni f(IV)=0 and Si nan diagnosis from ion_pops rows.
Read-only.
"""
import numpy as np, csv, os
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUNS={"gphall":f"{REPO}/logs/coevolve_consume_a10_kx_gphall",
      "stage4":f"{REPO}/logs/coevolve_consume_a10_kx_stage4"}
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/crime_reconstruction"
ELEM={14:"Si",26:"Fe",27:"Co",28:"Ni",22:"Ti",24:"Cr",13:"Al"}

def load_levelpop(path, wantZion, shells):
    """return dict[(shell,Z,ion)] -> list of (level_num,b_k,n_k,g,E,has_sigma)."""
    out={}
    with open(f"{path}/lumina_levelpop.csv") as f:
        for row in csv.DictReader(f):
            s=int(row["shell"])
            if s not in shells: continue
            Z=int(row["Z"]); ion=int(row["ion"])
            if (Z,ion) not in wantZion: continue
            out.setdefault((s,Z,ion),[]).append(
                (int(row["level_num"]),float(row["b_k"]),float(row["n_k"]),
                 int(row["g"]),float(row["E_eV"]),int(row["has_sigma"])))
    return out

def load_ionpops(path, shells):
    out={}
    with open(f"{path}/lumina_ion_pops.csv") as f:
        for row in csv.DictReader(f):
            s=int(row["shell_id"])
            if s not in shells: continue
            out[(s,int(row["Z"]),int(row["stage"]))]=float(row["n_ion"])
    return out

shells={0,8}
wantZion={(27,3),(26,2),(27,2),(28,3),(14,3),(26,3)}
lp={r:load_levelpop(p,wantZion,shells) for r,p in RUNS.items()}
ip={r:load_ionpops(p,shells) for r,p in RUNS.items()}

print("="*78)
print("(a) b_k -- Co IV level 50 (metastable trap) & 144 (funnel transit)")
print("="*78)
print(f"{'run':8}{'shell':6}{'lev50 b_k':>14}{'lev50 n_k':>14}{'lev144 b_k':>14}{'lev144 n_k':>14}")
rows_a=[]
for r in RUNS:
    for s in sorted(shells):
        d={l[0]:l for l in lp[r].get((s,27,3),[])}
        b50 =d.get(50,(50,float('nan'),float('nan')))[1]; n50=d.get(50,(0,0,float('nan')))[2]
        b144=d.get(144,(144,float('nan'),float('nan')))[1]; n144=d.get(144,(0,0,float('nan')))[2]
        print(f"{r:8}{s:<6}{b50:>14.4e}{n50:>14.4e}{b144:>14.4e}{n144:>14.4e}")
        rows_a.append([r,s,"CoIV",b50,n50,b144,n144])

print("\n"+"="*78)
print("(a) Fe III comb -- b_k distribution over the 1500 Fe III super-levels")
print("="*78)
print(f"{'run':8}{'shell':6}{'nlev':>6}{'b_k min':>12}{'b_k med':>12}{'b_k max':>12}{'b_k>10 %':>10}{'b_k(gnd)':>12}")
rows_comb=[]
for r in RUNS:
    for s in sorted(shells):
        L=lp[r].get((s,26,2),[])
        if not L:
            print(f"{r:8}{s:<6} (no Fe III rows)"); continue
        bk=np.array([x[1] for x in L]); gnd=[x for x in L if x[0]==0]
        bgnd=gnd[0][1] if gnd else float('nan')
        print(f"{r:8}{s:<6}{len(bk):>6}{np.nanmin(bk):>12.3e}{np.nanmedian(bk):>12.3e}"
              f"{np.nanmax(bk):>12.3e}{100*np.mean(bk>10):>10.1f}{bgnd:>12.3e}")
        rows_comb.append([r,s,"FeIII",len(bk),float(np.nanmin(bk)),float(np.nanmedian(bk)),
                          float(np.nanmax(bk)),float(np.mean(bk>10)),bgnd])

print("\n"+"="*78)
print("(b,d) ionization fraction f(stage) = n_ion / sum_stages, stage4 vs gphall")
print("="*78)
def frac(ipd,s,Z,stage):
    tot=sum(v for (ss,zz,st),v in ipd.items() if ss==s and zz==Z)
    n=ipd.get((s,Z,stage),float('nan'))
    return (n/tot if tot>0 else float('nan')), n, tot
rows_b=[]
for Z in (26,27,28,14):
    print(f"\n-- {ELEM[Z]} (Z={Z}) --")
    print(f"{'run':8}{'shell':6}{'f(III)':>12}{'f(IV)':>12}{'f(V)':>12}{'n(IV)':>14}{'n_tot':>14}")
    for r in RUNS:
        for s in sorted(shells):
            fIII,_,_=frac(ip[r],s,Z,2); fIV,nIV,ntot=frac(ip[r],s,Z,3); fV,_,_=frac(ip[r],s,Z,4)
            print(f"{r:8}{s:<6}{fIII:>12.4f}{fIV:>12.4f}{fV:>12.4f}{nIV:>14.4e}{ntot:>14.4e}")
            rows_b.append([r,s,ELEM[Z],Z,fIII,fIV,fV,nIV,ntot])

# raw ion_pops rows for Ni(28) and Si(14): zero vs missing vs reservoir
print("\n"+"="*78)
print("(d) RAW ion_pops rows Ni(28) & Si(14): stage4 vs gphall, all stages, s0 & s8")
print("="*78)
for Z in (28,14):
    print(f"\n-- {ELEM[Z]} raw n_ion by stage --")
    print(f"{'run':8}{'shell':6}"+"".join(f"{'st'+str(st):>13}" for st in range(6)))
    for r in RUNS:
        for s in sorted(shells):
            vals=[ip[r].get((s,Z,st),None) for st in range(6)]
            print(f"{r:8}{s:<6}"+"".join((f"{v:>13.4e}" if v is not None else f"{'MISSING':>13}") for v in vals))

with open(f"{OUT}/part2_bk_coiv.csv","w",newline="") as f:
    csv.writer(f).writerows([["run","shell","ion","b50","n50","b144","n144"]]+rows_a)
with open(f"{OUT}/part2_feiii_comb.csv","w",newline="") as f:
    csv.writer(f).writerows([["run","shell","ion","nlev","bk_min","bk_med","bk_max","frac_bk_gt10","bk_gnd"]]+rows_comb)
with open(f"{OUT}/part2_ionfrac.csv","w",newline="") as f:
    csv.writer(f).writerows([["run","shell","elem","Z","fIII","fIV","fV","nIV","ntot"]]+rows_b)
print(f"\n[out] part2_bk_coiv.csv, part2_feiii_comb.csv, part2_ionfrac.csv")
