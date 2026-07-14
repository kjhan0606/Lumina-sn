#!/usr/bin/env python3
"""GPH A/B ion-state extractor: <q> and f(IV) per element per shell.
Usage: gph_ab_ionstate.py <ion_pops.csv> [label]
"""
import sys, csv
from collections import defaultdict

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path

# Z -> name
ZN = {14:'Si', 16:'S', 20:'Ca', 26:'Fe', 27:'Co', 28:'Ni'}
# n_ion[shell][Z][stage]
pops = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
with open(path) as f:
    r = csv.DictReader(f)
    for row in r:
        s = int(row['shell_id']); Z = int(row['Z']); st = int(row['stage'])
        pops[s][Z][st] += float(row['n_ion'])

def qbar(d):
    tot = sum(d.values())
    if tot <= 0: return float('nan'), float('nan')
    # Verified convention: stage index = ion charge (0=neutral I, 2=III, 3=IV, 4=V).
    q = sum(st*n for st,n in d.items())/tot
    f4 = d.get(3,0.0)/tot   # stage 3 = IV
    return q, f4

CORE = list(range(0,5))
print(f"### {label}")
print(f"{'Z':>4} {'shell':>5} {'<q>':>6} {'f(IV)':>7}  top-stages")
for Z in [26,27,28,14,16,20]:
    for s in CORE:
        d = pops[s][Z]
        q,f4 = qbar(d)
        tot = sum(d.values())
        top = sorted(d.items(), key=lambda x:-x[1])[:3]
        tops = " ".join(f"st{st}:{n/tot:.2f}" for st,n in top) if tot>0 else "-"
        print(f"{ZN[Z]:>4} {s:>5} {q:>6.2f} {f4:>7.3f}  {tops}")
    print()

# core-averaged summary
print(f"--- {label} CORE(s0-4) avg ---")
for Z in [26,27,28,14,16,20]:
    qs=[]; f4s=[]
    for s in CORE:
        q,f4 = qbar(pops[s][Z])
        if q==q: qs.append(q); f4s.append(f4)
    if qs:
        print(f"{ZN[Z]:>4}: <q>={sum(qs)/len(qs):.2f}  f(IV)={sum(f4s)/len(f4s):.3f}")
