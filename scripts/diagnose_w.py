#!/usr/bin/env python3
"""Diagnose W discrepancy by computing W from TARDIS j_estimator,
then comparing LUMINA j_estimator against TARDIS j_estimator."""
import numpy as np
import json

REF = "data/tardis_reference"

# Load TARDIS reference
plasma = np.genfromtxt(f"{REF}/plasma_state.csv", delimiter=',', names=True)
W_ref = plasma['W']
T_rad_ref = plasma['T_rad']

# Load TARDIS j/nu_bar estimators
mc_est = np.genfromtxt(f"{REF}/mc_estimators.csv", delimiter=',', names=True)
j_tardis = mc_est['j_estimator']
nu_bar_tardis = mc_est['nu_bar_estimator']

# Load geometry
geo = np.genfromtxt(f"{REF}/geometry.csv", delimiter=',', names=True)
r_inner = geo['r_inner']
r_outer = geo['r_outer']
n_shells = len(r_inner)

# Compute volume
volume = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)

# Constants
SIGMA_SB = 5.670374419e-05
T_RAD_CONST = 1.2523374827e-11

# Load config
with open(f"{REF}/config.json") as f:
    cfg = json.load(f)
T_inner = cfg['T_inner_K']
t_exp = cfg['time_explosion_s']

L_inner = 4.0 * np.pi * r_inner[0]**2 * SIGMA_SB * T_inner**4
t_sim = 1.0 / L_inner
n_packets_tardis = cfg.get('n_packets', 200000)

print(f"T_inner = {T_inner:.2f} K")
print(f"L_inner = {L_inner:.6e} erg/s")
print(f"t_sim = {t_sim:.6e} s")
print(f"n_packets (TARDIS) = {n_packets_tardis}")

# STEP 1: Recompute W from TARDIS j_estimator → should match W_ref
print(f"\n{'='*70}")
print("STEP 1: Verify W formula with TARDIS j values")
print(f"{'='*70}")
print(f"{'Shell':>5} {'W_formula':>10} {'W_TARDIS':>10} {'Ratio':>8} {'T_form':>10} {'T_TARDIS':>10} {'T_ratio':>8}")
print("-" * 70)

for s in range(n_shells):
    j = j_tardis[s]
    nb = nu_bar_tardis[s]
    T_rad_c = T_RAD_CONST * nb / j
    W_c = j / (4.0 * SIGMA_SB * T_rad_c**4 * t_sim * volume[s])
    w_ratio = W_c / W_ref[s]
    t_ratio = T_rad_c / T_rad_ref[s]
    if s % 5 == 0 or abs(w_ratio - 1.0) > 0.01:
        print(f"{s:5d} {W_c:10.6f} {W_ref[s]:10.6f} {w_ratio:8.4f} {T_rad_c:10.2f} {T_rad_ref[s]:10.2f} {t_ratio:8.4f}")

# STEP 2: Load LUMINA j_estimator (from lumina_plasma_state.csv — need to add j output)
# For now, compute what j_estimator LUMINA would need to get its W values
print(f"\n{'='*70}")
print("STEP 2: Infer LUMINA j_estimator from its W values")
print(f"{'='*70}")

lumina_plasma = np.genfromtxt("lumina_plasma_state.csv", delimiter=',', names=True)
W_lumina = lumina_plasma['W']
T_rad_lumina = lumina_plasma['T_rad']

# j = W * 4 * sigma * T^4 * t_sim * V
j_lumina_inferred = W_lumina * 4.0 * SIGMA_SB * T_rad_lumina**4 * t_sim * volume

print(f"\n{'Shell':>5} {'j_LUMINA':>16} {'j_TARDIS':>16} {'j_ratio':>8} {'W_L':>8} {'W_T':>8} {'T_L':>10} {'T_T':>10}")
print("-" * 95)

for s in range(n_shells):
    j_ratio = j_lumina_inferred[s] / j_tardis[s]
    print(f"{s:5d} {j_lumina_inferred[s]:16.4e} {j_tardis[s]:16.4e} {j_ratio:8.4f} "
          f"{W_lumina[s]:8.4f} {W_ref[s]:8.4f} {T_rad_lumina[s]:10.2f} {T_rad_ref[s]:10.2f}")

# STEP 3: Decompose W error into T_rad and j components
print(f"\n{'='*70}")
print("STEP 3: Decompose W error = j_error / T_rad^4_error")
print(f"{'='*70}")
print(f"  W = j / (4σT⁴ × t_sim × V)")
print(f"  W_L/W_T = (j_L/j_T) × (T_T/T_L)^4")
print()

print(f"{'Shell':>5} {'W_L/W_T':>8} {'j_L/j_T':>8} {'(T_T/T_L)^4':>12} {'j×T4_product':>14}")
print("-" * 55)

for s in range(n_shells):
    W_ratio = W_lumina[s] / W_ref[s]
    j_ratio = j_lumina_inferred[s] / j_tardis[s]
    T_ratio_4 = (T_rad_ref[s] / T_rad_lumina[s])**4
    product = j_ratio * T_ratio_4
    if s % 3 == 0:
        print(f"{s:5d} {W_ratio:8.4f} {j_ratio:8.4f} {T_ratio_4:12.4f} {product:14.4f}")

# Note: t_sim changes across iterations if T_inner changes!
# At iteration 0 (using TARDIS T_inner), t_sim is the same
# But at later iterations, LUMINA T_inner may differ!
# The W formula at iteration N uses iteration N's t_sim (computed from T_inner_N)

# STEP 4: Check t_sim difference
print(f"\n{'='*70}")
print("STEP 4: Check if t_sim differs between iterations")
print(f"{'='*70}")
# For the first iteration, LUMINA uses TARDIS T_inner → same t_sim
# But the comparison output showed iter 5 results, where T_inner has changed
# The W at iter 1 (hold iteration) should use TARDIS T_inner
# Let's check what T_inner LUMINA used
print(f"  TARDIS T_inner = {T_inner:.2f} K")
print(f"  TARDIS t_sim = {t_sim:.6e} s")
print(f"  TARDIS L_inner = {L_inner:.6e} erg/s")
print(f"  If T_inner = 13577K → L_inner = {4*np.pi*r_inner[0]**2*SIGMA_SB*13577**4:.6e}")
print(f"  → t_sim = {1.0/(4*np.pi*r_inner[0]**2*SIGMA_SB*13577**4):.6e}")
print()
print(f"  CRITICAL: t_sim at T_inner=13577K is {t_sim / (1.0/(4*np.pi*r_inner[0]**2*SIGMA_SB*13577**4)):.4f}x smaller")
print(f"  This means W is inflated by factor (T_inner_L/T_inner_T)^4 = {(13577/T_inner)**4:.4f}")
print(f"  But this only affects iterations after T_inner starts changing")

# STEP 5: What's the W at iteration 1 (before T_inner changes)?
# From the run output, iter 1 is a hold iteration, so T_inner stays at 10521.52
# The W values reported are from solve_radiation_field which uses the CURRENT T_inner's t_sim
# At iter 1, t_sim is still from T_inner=10521.52
# So the W at iter 1 reflects JUST the j_estimator discrepancy
