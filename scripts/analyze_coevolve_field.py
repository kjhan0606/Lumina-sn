#!/usr/bin/env python3
"""Analyze lumina_coevolve_field.csv (shell,bin,wavelength_A,cs_J,mc_J): the Stage-4
discriminator. Per shell, is the MC shadow field (i) BLUER and (ii) STRONGER in the UV
than the deterministic cs.J? amp_UV>1 => a blend injects trapped UV the deterministic
field misses => Stage-4 can open the optical channel. amp_UV~1 => field is not the lever;
the down-branch/co-evolution architecture is.
Usage: analyze_coevolve_field.py <run_dir or csv_path>"""
import sys, csv
import numpy as np

p = sys.argv[1]
path = p if p.endswith('.csv') else f"{p}/lumina_coevolve_field.csv"
rows = list(csv.DictReader(open(path)))
shells = sorted(set(int(r['shell']) for r in rows))
nb = 1 + max(int(r['bin']) for r in rows)
lam = np.zeros(nb); csJ = {}; mcJ = {}
for s in shells: csJ[s] = np.zeros(nb); mcJ[s] = np.zeros(nb)
for r in rows:
    s = int(r['shell']); b = int(r['bin'])
    lam[b] = float(r['wavelength_A']); csJ[s][b] = float(r['cs_J']); mcJ[s][b] = float(r['mc_J'])

def bandmask(a, b): return (lam >= a) & (lam < b)
UV = bandmask(2500, 3500); OPT = bandmask(3500, 6500)
FLOOR = 1e-29  # nlte_normalize floors to 1e-30; treat <=this as "no MC packets"

print(f"field: {path}  ({len(shells)} shells x {nb} bins)")
print(f"{'s':>3} {'csUV/csOpt':>10} {'mcUV/mcOpt':>10} {'bluer':>5} | "
      f"{'ampUV(mc/cs)':>12} {'ampOpt':>8} {'UV_str':>6} {'mc_UV_dead':>10}")
amp_uv_all = []
for s in shells:
    cu = csJ[s][UV].sum(); co = csJ[s][OPT].sum()
    mu = mcJ[s][UV].sum(); mo = mcJ[s][OPT].sum()
    cs_col = cu/co if co > 0 else 0.0
    mc_col = mu/mo if mo > 0 else 0.0
    amp_uv = mu/cu if cu > 0 else 0.0
    amp_opt = mo/co if co > 0 else 0.0
    dead = mcJ[s][UV].max() <= FLOOR
    amp_uv_all.append((s, amp_uv, dead))
    if s % 5 == 0 or s in (shells[0], shells[-1]):
        print(f"{s:>3} {cs_col:10.3e} {mc_col:10.3e} {'YES' if mc_col>cs_col else 'no':>5} | "
              f"{amp_uv:12.3e} {amp_opt:8.3e} {'YES' if amp_uv>1 else 'no':>6} {'DEAD' if dead else '-':>10}")

live = [(s,a) for s,a,d in amp_uv_all if not d]
strong = [s for s,a in live if a > 1.0]
lf = [s for s,a,d in amp_uv_all if 6 <= s <= 35]
lf_live = [(s,a) for s,a,d in amp_uv_all if 6 <= s <= 35 and not d]
lf_strong = [s for s,a in lf_live if a > 1.0]
print("\n=== STAGE-4 DISCRIMINATOR ===")
print(f"shells with live MC UV (not floored): {len(live)}/{len(shells)}")
print(f"shells where MC UV STRONGER than det (amp>1): {len(strong)}/{len(shells)}  "
      f"[line-forming s6-35: {len(lf_strong)}/{len(lf)}]")
if lf_live:
    med = np.median([a for _,a in lf_live])
    print(f"median amp_UV (MC/det) in live line-forming shells: {med:.3f}")
    if med > 1.3:
        print("  => VERDICT: MC field carries substantially MORE UV -> Stage-4 blend is the lever.")
    elif med > 0.7:
        print("  => VERDICT: MC ~ det in UV amplitude -> blend adds little; DOWN-BRANCH/co-evolution is the lever.")
    else:
        print("  => VERDICT: MC field UV-POORER than det -> injection/sampling limited (Stage-2 first).")
dead_lf = [s for s in lf if any(x[0]==s and x[2] for x in amp_uv_all)]
if dead_lf:
    print(f"  line-forming shells with DEAD MC UV (injection/undersampling hole): {dead_lf}")
