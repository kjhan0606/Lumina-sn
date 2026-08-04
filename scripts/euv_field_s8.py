#!/usr/bin/env python3
# At a given shell, compare MC field vs thermal field in the Fe III-ionizing EUV.
# field csv: shell,bin,wavelength_A,cs_J,mc_J.  Fe III ground edge ~404A; excited ~461A.
import sys, csv, collections
def load(path, shell):
    rows=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r['shell'])!=shell: continue
            rows.append((float(r['wavelength_A']), float(r['cs_J']), float(r['mc_J'])))
    return sorted(rows)
def band(rows, lo, hi):
    cs=sum(c for w,c,m in rows if lo<=w<hi); mc=sum(m for w,c,m in rows if lo<=w<hi)
    return cs, mc
shell=int(sys.argv[1]) if len(sys.argv)>1 else 8
paths=sys.argv[2:] or ['logs/coevolve_consume_a10_kx_tepop1/lumina_coevolve_field.csv']
bands=[(50,300,'deepEUV<300'),(300,404,'EUV 300-404(FeIII grnd)'),(404,461,'404-461(FeIII exc)'),
       (461,912,'461-912(HI/FUV)'),(912,1500,'FUV 912-1500'),(1500,4000,'opt/NUV')]
for p in paths:
    tag=p.split('/')[-2] if '/' in p else p
    rows=load(p,shell)
    print(f"\n== shell {shell}  [{tag}] ==")
    print(f"  {'band':28} {'cs_J(thermal)':>14} {'mc_J(MC)':>14} {'mc/cs':>8}")
    for lo,hi,nm in bands:
        cs,mc=band(rows,lo,hi)
        r=(mc/cs) if cs>0 else float('inf')
        print(f"  {nm:28} {cs:14.3e} {mc:14.3e} {r:8.2f}")
