#!/usr/bin/env python3
"""Compare Lumina A/B IME (Si/S/Ca) ionization vs CMFGEN/ARTIS benchmark, per shell.
Is B over-ionizing Si/S relative to benchmark (regression)?
"""
import sys, csv, glob
from collections import defaultdict
C=2.99792458e10
ZN={14:'Si',16:'S',20:'Ca'}

def load_ion(path):
    p=defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for row in csv.DictReader(open(path)):
        p[int(row['shell_id'])][int(row['Z'])][int(row['stage'])]+=float(row['n_ion'])
    return p
def qf(d):
    t=sum(d.values())
    if t<=0: return None
    return sum(st*n for st,n in d.items())/t

# shell mid velocities (km/s) from geometry.csv
vel={}
for row in csv.DictReader(open('data/tardis_reference_toy06_19p48d/geometry.csv')):
    s=int(row['shell_id']); vel[s]=(float(row['v_inner'])+float(row['v_outer']))/2/1e5

pA=load_ion('logs/coevolve_consume_a10_kx_gphground/lumina_ion_pops.csv')
pB=load_ion('logs/coevolve_consume_a10_kx_gphall/lumina_ion_pops.csv')

# benchmark loader
def bench_q(el,ep,v):
    fn=f'data/standart_data1/toy06/ionfrac_{el}_toy06_cmfgen.txt'
    lines=open(fn).read().splitlines()
    blocks=[(float(l.split(':')[1].split()[0]),i) for i,l in enumerate(lines) if l.strip().startswith('#TIME:')]
    bt,bi=min(blocks,key=lambda x:abs(x[0]-ep))
    rows=[]; j=bi+1
    while j<len(lines):
        s=lines[j].strip()
        if s.startswith('#TIME'): break
        if not s or s.startswith('#'): j+=1; continue
        parts=s.split()
        try: vv=float(parts[0])
        except: j+=1; continue
        rows.append((vv,[float(x) for x in parts[1:]])); j+=1
    vr=min(rows,key=lambda x:abs(x[0]-v))
    frac=vr[1]; t=sum(frac)
    return (sum(i*f for i,f in enumerate(frac))/t if t>0 else float('nan'), vr[0])

print(f"{'sh':>3} {'v_km/s':>7} {'el':>3} {'A<q>':>6} {'B<q>':>6} {'CMFGEN':>7} {'(v_bench)':>9}  verdict")
for s in range(4,12):
    v=vel[s]
    for Z,el in [(14,'si'),(16,'s'),(20,'ca')]:
        qa=qf(pA[s][Z]); qb=qf(pB[s][Z])
        qc,vb=bench_q(el,19.48,v)
        if qa is None: continue
        # verdict: is B closer to bench than A? and is B over-ionized?
        da=abs(qa-qc); db=abs(qb-qc)
        over = "B OVER" if qb>qc+0.3 else ("B under" if qb<qc-0.3 else "B~ok")
        better = "B better" if db<da-0.05 else ("A better" if da<db-0.05 else "same")
        print(f"{s:>3} {v:>7.0f} {ZN[Z]:>3} {qa:>6.2f} {qb:>6.2f} {qc:>7.2f} {vb:>9.0f}  {over}/{better}")
    print()
