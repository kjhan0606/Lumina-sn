#!/usr/bin/env python3
# f(FeIV) per shell from lumina_ion_pops.csv (long: shell_id,Z,stage,n_ion). Fe IV = Z26 stage3.
import sys, csv, collections
def frac(path, Z=26, want=3):
    tot=collections.defaultdict(float); wan=collections.defaultdict(float)
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            if int(row['Z'])!=Z: continue
            s=int(row['shell_id']); n=float(row['n_ion'])
            tot[s]+=n
            if int(row['stage'])==want: wan[s]+=n
    return {s:(wan[s]/tot[s] if tot[s]>0 else 0.0) for s in sorted(tot)}
if __name__=='__main__':
    paths=sys.argv[1:] or ['lumina_ion_pops.csv']
    res={p:frac(p) for p in paths}
    shells=sorted(next(iter(res.values())).keys())
    hdr="shell "+" ".join(f"{p.split('/')[-2] if '/' in p else p:>16.16}" for p in paths)
    print(hdr)
    for s in shells:
        print(f"{s:>5} "+" ".join(f"{res[p][s]:16.4f}" for p in paths))
