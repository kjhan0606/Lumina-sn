#!/usr/bin/env python3
# Per-shell 404-461A field (mc_J, cs_J) profile — where does the FeIII-excited ionizing field live?
import sys, csv, collections
path=sys.argv[1] if len(sys.argv)>1 else 'logs/coevolve_consume_a10_kx_tepop1/lumina_coevolve_field.csv'
lo,hi=404.0,461.0
mc=collections.defaultdict(float); cs=collections.defaultdict(float)
with open(path) as f:
    for r in csv.DictReader(f):
        w=float(r['wavelength_A'])
        if lo<=w<hi:
            s=int(r['shell']); mc[s]+=float(r['mc_J']); cs[s]+=float(r['cs_J'])
print(f"{'shell':>5} {'mc_J(404-461)':>14} {'cs_J':>12} {'mc/cs':>8}")
for s in sorted(mc):
    r=(mc[s]/cs[s]) if cs[s]>0 else 0
    if mc[s]>1e-30 or cs[s]>1e-30:
        print(f"{s:>5} {mc[s]:14.3e} {cs[s]:12.3e} {r:8.2f}")
