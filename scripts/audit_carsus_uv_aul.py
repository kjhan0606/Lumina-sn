#!/usr/bin/env python3
"""Audit carsus UV A_ul values for Si II / Ni II / Fe II / Co II / Cr II vs NIST.

Phase B audit (#131) only spot-checked Ca II H/K, Si II 6347/6371, Fe II m42.
K1 saturates Ni II UV A_ul at f=0.3 (×0.3 better than carsus); L1 monotone
Si II UV A_ul down to f=0.05 (×20 too strong?). Suggests systematic over-
strength in carsus UV line A_ul. This script:
  1. Loads line_list.csv
  2. Reports strongest A_ul lines λ<4000Å for each iron-peak II ion
  3. Compares against published NIST ASD values (hard-coded reference set)
  4. Flags lines where carsus ≥ 2× NIST
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
LL = ROOT/"data/tardis_reference_strat6_2011fe_physical/line_list.csv"
LV = ROOT/"data/tardis_reference_strat6_2011fe_physical/levels.csv"

# NIST ASD reference values for strong UV/optical lines (A_ki in s^-1).
# Format: (Z, ion, λ_NIST_Å, A_ki_NIST, label)  -- ion is ion stage (0=I, 1=II, ...)
# Hand-picked from NIST ASD (https://physics.nist.gov/asd) — strongest UV
# resonance / near-resonance lines for each iron-peak II.
NIST = [
    # Si II (Z=14, ion=1) — strong UV
    (14, 1, 1304.37, 6.53e8,  "Si II 1304"),
    (14, 1, 1808.01, 2.54e6,  "Si II 1808"),  # intercombination
    (14, 1, 1816.93, 2.65e6,  "Si II 1817"),
    (14, 1, 2335.32, 4.20e8,  "Si II 2335"),
    (14, 1, 2350.17, 3.95e8,  "Si II 2350"),
    (14, 1, 3856.02, 4.59e6,  "Si II 3856"),
    (14, 1, 3862.59, 1.36e6,  "Si II 3863"),
    # Ni II (Z=28, ion=1) — strong UV
    (28, 1, 1317.22, 2.81e8,  "Ni II 1317"),
    (28, 1, 1370.13, 6.42e8,  "Ni II 1370"),
    (28, 1, 1454.84, 2.61e8,  "Ni II 1455"),
    (28, 1, 1502.15, 5.55e8,  "Ni II 1502"),
    (28, 1, 1709.60, 3.51e8,  "Ni II 1710"),
    (28, 1, 1741.55, 3.31e8,  "Ni II 1742"),
    (28, 1, 1751.91, 3.07e8,  "Ni II 1752"),
    # Fe II (Z=26, ion=1) — UV multiplets
    (26, 1, 2260.08, 3.18e6,  "Fe II 2260"),
    (26, 1, 2344.21, 1.73e8,  "Fe II 2344 UV3"),
    (26, 1, 2382.76, 3.13e8,  "Fe II 2383 UV2"),
    (26, 1, 2586.65, 8.61e7,  "Fe II 2587 UV1"),
    (26, 1, 2600.17, 2.35e8,  "Fe II 2600 UV1"),
    # Co II (Z=27, ion=1)
    (27, 1, 1466.21, 4.20e8,  "Co II 1466"),
    (27, 1, 1574.55, 3.95e8,  "Co II 1575"),
    # Cr II (Z=24, ion=1)
    (24, 1, 2056.26, 2.99e8,  "Cr II 2056"),
    (24, 1, 2061.58, 2.30e8,  "Cr II 2062"),
    (24, 1, 2065.50, 1.34e8,  "Cr II 2066"),
]

print(f"Loading {LL}...")
df = pd.read_csv(LL)
print(f"  {len(df):,} lines total\n")

df["lam_A"] = df["wavelength_cm"] * 1e8
print("=== UV line counts (λ<4000Å) per iron-peak II ion ===")
print(f"  {'Z':<4s}{'ion':<5s}{'n_total':>10s}{'n_λ<4000':>12s}{'n_λ<2500':>12s}{'sum_A_ul':>12s}")
for Z, label in [(14,"Si II"),(20,"Ca II"),(22,"Ti II"),(24,"Cr II"),
                  (25,"Mn II"),(26,"Fe II"),(27,"Co II"),(28,"Ni II")]:
    sub = df[(df.atomic_number==Z)&(df.ion_number==1)]
    sub_uv = sub[sub.lam_A<4000]
    sub_fuv = sub[sub.lam_A<2500]
    sumA = sub_uv.A_ul.sum() if len(sub_uv) else 0
    print(f"  {Z:<4d}{1:<5d}{len(sub):>10d}{len(sub_uv):>12d}{len(sub_fuv):>12d}{sumA:>12.2e}  {label}")

print("\n=== Carsus vs NIST A_ki comparison ===")
print(f"  {'label':<20s}{'NIST_A':>11s}{'λ_NIST':>10s}{'cars_A':>11s}{'cars_λ':>10s}{'Δλ_Å':>8s}{'ratio':>8s}{'flag':>6s}")
for Z, ion, lam_N, A_N, label in NIST:
    sub = df[(df.atomic_number==Z)&(df.ion_number==ion)]
    sub = sub[(sub.lam_A > lam_N-3.0)&(sub.lam_A < lam_N+3.0)]
    if len(sub)==0:
        print(f"  {label:<20s}{A_N:>11.2e}{lam_N:>10.2f}   not found within ±3Å")
        continue
    # pick strongest within window
    idx = sub.A_ul.idxmax()
    A_c = float(sub.loc[idx, "A_ul"])
    lam_c = float(sub.loc[idx, "lam_A"])
    dlam = lam_c - lam_N
    ratio = A_c / A_N if A_N > 0 else float('nan')
    flag = "**" if ratio > 2.0 or ratio < 0.5 else ""
    print(f"  {label:<20s}{A_N:>11.2e}{lam_N:>10.2f}{A_c:>11.2e}{lam_c:>10.2f}{dlam:>+8.2f}{ratio:>8.2f}{flag:>6s}")

print("\n=== Strongest carsus UV lines λ<4000Å (top 10 per ion) ===")
for Z, label in [(14,"Si II"),(26,"Fe II"),(27,"Co II"),(28,"Ni II")]:
    sub = df[(df.atomic_number==Z)&(df.ion_number==1)&(df.lam_A<4000)]
    if len(sub)==0: continue
    top = sub.nlargest(10, "A_ul")
    print(f"\n  --- {label} (Z={Z}) top-10 strongest A_ul λ<4000Å ---")
    print(f"  {'λ_Å':>10s}{'A_ul (s⁻¹)':>14s}{'f_lu':>10s}{'lev_lo→up':>12s}")
    for _, r in top.iterrows():
        print(f"  {r.lam_A:>10.2f}{r.A_ul:>14.3e}{r.f_lu:>10.3e}  {int(r.level_number_lower):>4d}→{int(r.level_number_upper):<5d}")
