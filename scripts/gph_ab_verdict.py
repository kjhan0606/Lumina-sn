#!/usr/bin/env python3
"""GPH A/B verdict: pre-registered prediction check, B(all-level) vs A(ground).
Usage: gph_ab_verdict.py <dirA_ground> <dirB_all>
Reads lumina_ion_pops.csv + lumina_plasma_state.csv from each dir.
"""
import sys, csv, math
from collections import defaultdict

dA, dB = sys.argv[1], sys.argv[2]
ZN = {14:'Si',16:'S',20:'Ca',26:'Fe',27:'Co',28:'Ni'}
CORE = list(range(0,5))
# CMFGEN benchmark core <q> targets (IGE ~ IV = q3): Fe/Co ~3.0, from census
CMFGEN_Q = {26:3.0, 27:3.0, 28:3.0}

def load_ion(path):
    pops = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    with open(path) as f:
        for row in csv.DictReader(f):
            s=int(row['shell_id']); Z=int(row['Z']); st=int(row['stage'])
            pops[s][Z][st]+=float(row['n_ion'])
    return pops

def load_plasma(path):
    ps={}
    with open(path) as f:
        for row in csv.DictReader(f):
            ps[int(row['shell_id'])]={'W':float(row['W']),'T_rad':float(row['T_rad']),
                                      'n_e':float(row['n_e']),'T_e':float(row['T_e'])}
    return ps

def qf(d):
    tot=sum(d.values())
    if tot<=0: return None,None
    return sum(st*n for st,n in d.items())/tot, d.get(3,0.0)/tot

def core_avg(pops,Z):
    qs=[];fs=[]
    for s in CORE:
        q,f=qf(pops[s][Z])
        if q is not None: qs.append(q);fs.append(f)
    return (sum(qs)/len(qs),sum(fs)/len(fs)) if qs else (float('nan'),float('nan'))

pA=load_ion(f"{dA}/lumina_ion_pops.csv"); psA=load_plasma(f"{dA}/lumina_plasma_state.csv")
pB=load_ion(f"{dB}/lumina_ion_pops.csv"); psB=load_plasma(f"{dB}/lumina_plasma_state.csv")

print("="*68)
print("GPH A/B VERDICT — B(all-level) vs A(ground)")
print("="*68)
print("\n[P2/P3] Core(s0-4) IGE ionization: <q> and f(IV)")
print(f"{'el':>4} {'A<q>':>6} {'B<q>':>6} {'d<q>':>6} | {'Af(IV)':>7} {'Bf(IV)':>7} | {'CMFGEN':>6}")
for Z in [26,27,28,14,16,20]:
    qA,fA=core_avg(pA,Z); qB,fB=core_avg(pB,Z)
    tgt=CMFGEN_Q.get(Z,None)
    ts=f"{tgt:.1f}" if tgt else "  -"
    dq = qB-qA if (qA==qA and qB==qB) else float('nan')
    print(f"{ZN[Z]:>4} {qA:>6.2f} {qB:>6.2f} {dq:>+6.2f} | {fA:>7.3f} {fB:>7.3f} | {ts:>6}")

print("\n[P4] Core n_e (should RISE in B toward CMFGEN):")
print(f"{'sh':>3} {'A n_e':>11} {'B n_e':>11} {'ratio B/A':>10}")
for s in CORE:
    ne_a=psA[s]['n_e']; ne_b=psB[s]['n_e']
    print(f"{s:>3} {ne_a:>11.3e} {ne_b:>11.3e} {ne_b/ne_a:>10.2f}")

print("\n[P5] Runaway check — T_e core(s0-4) + far-edge(s40-49):")
print(f"{'sh':>3} {'A T_e':>9} {'B T_e':>9} {'dT':>8}")
for s in list(range(0,5))+list(range(40,50)):
    if s in psA and s in psB:
        ta=psA[s]['T_e']; tb=psB[s]['T_e']
        flag=" <-- HOT" if tb>40000 else ""
        print(f"{s:>3} {ta:>9.0f} {tb:>9.0f} {tb-ta:>+8.0f}{flag}")

# Verdict summary
print("\n"+"="*68)
qFeA,fFeA=core_avg(pA,26); qFeB,fFeB=core_avg(pB,26)
qCoA,fCoA=core_avg(pA,27); qCoB,fCoB=core_avg(pB,27)
maxTeB=max(psB[s]['T_e'] for s in CORE)
print("PRE-REGISTERED PREDICTION CHECK:")
print(f" P2 Fe/Co core f(IV) rise (A~0.01-0.10 -> B>=0.3): Fe {fFeA:.3f}->{fFeB:.3f}, Co {fCoA:.3f}->{fCoB:.3f}  {'PASS' if (fFeB>=0.3 and fCoB>=0.3) else 'FAIL/partial'}")
print(f" P3 Fe/Co core <q> toward 3.0 (CMFGEN): Fe {qFeA:.2f}->{qFeB:.2f}, Co {qCoA:.2f}->{qCoB:.2f}  {'PASS' if (qFeB>=2.7 and qCoB>=2.7) else 'FAIL/partial'}")
print(f" P5 core NOT runaway (T_e<40kK): max core T_e_B={maxTeB:.0f}  {'PASS' if maxTeB<40000 else 'FAIL runaway'}")
print("="*68)
