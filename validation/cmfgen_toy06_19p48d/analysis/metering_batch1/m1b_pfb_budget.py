#!/usr/bin/env python3
"""M1b: what the Kramers->Milne swap does to the k-packet fb BRANCH PROBABILITY.

Reconstructs, per shell, the exact denominator of
  p_fb = C_fb / (C_ff + C_fb + C_collexc)          (lumina_plasma.c:4351-4353)
from parity42's own dumps:
  C_ff        = 1.426e-27 sqrt(Te) ne sum_z2n      (recomputed here)
  C_collexc   = "tot" printed by [KPD] (stderr, last iteration)
  C_fb(Kram)  = sum_ions alpha_K  n_ion ne kTe     (recomputed here)
  C_fb(Milne) = sum_ions alpha_RR n_ion ne kTe     (frozenin_alpha_rr, gated+DR)
and validates the reconstruction against the printed [KPD-UP] p_ff/p_fb at s8.
"""
import csv, math, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location(
    'm1', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'm1_kramers_vs_milne.py'))

ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
RUN = os.path.join(ROOT, 'logs/coevolve_consume_parity42')
OUT = os.path.join(ROOT, 'validation/cmfgen_toy06_19p48d/analysis/metering_batch1')
KB = 1.380649e-16

# import the machinery from m1 without re-running its main block
src = open(spec.origin).read()
head = src.split('os.makedirs(OUT, exist_ok=True)')[0]
ns = {'__name__': 'm1mod'}
exec(compile(head, spec.origin, 'exec'), ns)
Cont, alpha_dr, Te, lv_Z, lv_ion = (ns['Cont'], ns['alpha_dr'], ns['Te'],
                                    ns['lv_Z'], ns['lv_ion'])
ROM = ns['ROM']

# ---- run dumps --------------------------------------------------------------
ne = []
with open(os.path.join(RUN, 'lumina_plasma_state.csv')) as f:
    for r in csv.DictReader(f):
        ne.append(float(r['n_e']))
ne = np.array(ne)
ND = len(ne)

nion = {}
with open(os.path.join(RUN, 'lumina_ion_pops.csv')) as f:
    for r in csv.DictReader(f):
        nion[(int(r['shell_id']), int(r['Z']), int(r['stage']))] = float(r['n_ion'])

tot = {}
pat = re.compile(r'^\[KPD\] s(\d+) tot=([0-9.eE+-]+)')
for line in open(os.path.join(RUN, 'stderr.log')):
    m = pat.match(line)
    if m:
        tot[int(m.group(1))] = float(m.group(2))   # keep the LAST occurrence
pup = None
for line in open(os.path.join(RUN, 'stderr.log')):
    m = re.match(r'^\[KPD-UP\] p_ff\[8\]=([0-9.eE+-]+) p_fb\[8\]=([0-9.eE+-]+)', line)
    if m:
        pup = (float(m.group(1)), float(m.group(2)))

# ---- the ion-pop table the C loop walks ------------------------------------
pops = sorted({(int(z), int(i)) for z, i in zip(lv_Z, lv_ion)})
conts = {}
for (Z, k) in pops:
    if k >= 1:
        conts[(Z, k)] = Cont(Z, k - 1)     # recombining (Z,k) -> product (Z,k-1)

rows = []
for s in range(ND):
    T, n_e = Te[s], ne[s]
    kTe = KB * T
    te4 = (T / 1e4) ** -0.75
    sum_z2n = 0.0
    Cfb_K = Cfb_M = Cfb_Mfull = 0.0
    for (Z, k) in pops:
        if k < 1:
            continue
        n = nion.get((s, Z, k), 0.0)
        if n <= 0:
            continue
        sum_z2n += k * k * n
        Cfb_K += 2.6e-13 * k * k * te4 * n * n_e * kTe
        c = conts[(Z, k)]
        aM = c.alpha(T, gated=True) + alpha_dr(Z, k, T)
        aF = c.alpha(T, gated=False) + alpha_dr(Z, k, T)
        Cfb_M += aM * n * n_e * kTe
        Cfb_Mfull += aF * n * n_e * kTe
    Cff = 1.426e-27 * math.sqrt(T) * n_e * sum_z2n
    Cco = tot.get(s, float('nan'))
    den_K = Cff + Cfb_K + Cco
    rows.append(dict(shell=s, T_e=T, n_e=n_e, C_ff=Cff, C_collexc=Cco,
                     C_fb_kramers=Cfb_K, C_fb_milne=Cfb_M,
                     C_fb_milne_ungated=Cfb_Mfull,
                     p_ff_kramers=Cff / den_K, p_fb_kramers=Cfb_K / den_K,
                     p_fb_milne=Cfb_M / (Cff + Cfb_M + Cco),
                     p_fb_milne_ungated=Cfb_Mfull / (Cff + Cfb_Mfull + Cco),
                     fb_boost=Cfb_M / Cfb_K if Cfb_K > 0 else float('nan')))

with open(os.path.join(OUT, 'm1b_pfb_budget.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

r8 = rows[8]
print('=== reconstruction check @ s8 (production printed vs recomputed) ===')
print(f'  printed  [KPD-UP] p_ff[8]={pup[0]:.3e}  p_fb[8]={pup[1]:.3e}')
print(f'  recomputed        p_ff[8]={r8["p_ff_kramers"]:.3e}  '
      f'p_fb[8]={r8["p_fb_kramers"]:.3e}')
print(f'  ratio             p_ff {r8["p_ff_kramers"]/pup[0]:.4f}   '
      f'p_fb {r8["p_fb_kramers"]/pup[1]:.4f}')
print(f'  C_ff={r8["C_ff"]:.4e}  C_fb(K)={r8["C_fb_kramers"]:.4e}  '
      f'C_collexc={r8["C_collexc"]:.4e}')
print()
print('shell   T_e     C_fb^Milne/C_fb^Kram   p_fb(Kram)   p_fb(Milne)   p_ff')
for s in (0, 5, 8, 20, 30, 45, 49):
    r = rows[s]
    print(f'  s{s:<3d} {r["T_e"]:7.0f}   {r["fb_boost"]:8.2f}x           '
          f'{r["p_fb_kramers"]:.3e}    {r["p_fb_milne"]:.3e}   '
          f'{r["p_ff_kramers"]:.3e}')
b = np.array([r['fb_boost'] for r in rows])
print(f'\nfb_boost over 50 shells: min {b.min():.2f}  median {np.median(b):.2f}  '
      f'max {b.max():.2f}')
pk = np.array([r['p_fb_kramers'] for r in rows])
pm = np.array([r['p_fb_milne'] for r in rows])
print(f'p_fb(Kramers): min {pk.min():.2e} med {np.median(pk):.2e} max {pk.max():.2e}')
print(f'p_fb(Milne)  : min {pm.min():.2e} med {np.median(pm):.2e} max {pm.max():.2e}')
