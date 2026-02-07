#!/usr/bin/env python3
"""Investigate partition function discrepancy.

Key observation: all lines within (Z, ion) have the SAME tau ratio.
This means the error is in n_ion * w / Z_partition (the common factor).
Since n_ion matches TARDIS perfectly, the error must be in the
partition function or the W weighting.

tau_ratio = (n_lower_ours / n_lower_tardis)
          = (n_ion * w * g * exp(-E/kT) / Z_ours) / (n_ion * w * g * exp(-E/kT) / Z_tardis)
          = Z_tardis / Z_ours

So: tau_ratio = Z_tardis / Z_ours
     Z_ours / Z_tardis = 1 / tau_ratio
"""
import numpy as np

REF = "data/tardis_reference"
K_B = 1.380649e-16
EV_TO_ERG = 1.602176634e-12

# Load
plasma = np.genfromtxt(f"{REF}/plasma_state.csv", delimiter=',', names=True)
W_ref = plasma['W']
T_rad_ref = plasma['T_rad']

levels = np.genfromtxt(f"{REF}/levels.csv", delimiter=',', names=True)
ioniz = np.genfromtxt(f"{REF}/ionization_energies.csv", delimiter=',', names=True)

ion_npy = np.load(f"{REF}/ion_number_density.npy")
tau_ref = np.load(f"{REF}/tau_sobolev.npy")

line_list = np.genfromtxt(f"{REF}/line_list.csv", delimiter=',', names=True)

import json
with open(f"{REF}/config.json") as f:
    config = json.load(f)
t_exp = config['time_explosion_s']

ELEMENTS = [6, 8, 14, 16, 20, 26, 27, 28]
ELEM_NAMES = {6: 'C', 8: 'O', 14: 'Si', 16: 'S', 20: 'Ca', 26: 'Fe', 27: 'Co', 28: 'Ni'}
SOBOLEV_COEFF = 2.6540281e-02

# Build ion map
ion_map = []
for z in ELEMENTS:
    mask = ioniz['atomic_number'].astype(int) == z
    n_ioniz = np.sum(mask)
    for stage in range(n_ioniz + 1):
        ion_map.append((z, stage))

def get_levels_for(z, ion):
    mask = (levels['atomic_number'].astype(int) == z) & (levels['ion_number'].astype(int) == ion)
    return levels[mask]

s = 0
T_rad = T_rad_ref[s]
T_e = 0.9 * T_rad
W = W_ref[s]

print("=" * 80)
print("Partition Function Investigation")
print(f"Shell 0: T_rad={T_rad:.2f} K, T_e={T_e:.2f} K, W={W:.6f}")
print("=" * 80)

# For each (Z, ion) with significant lines, infer the TARDIS Z_partition from tau
print(f"\n{'Ion':>10} {'Z_dilute':>14} {'Z_inferred':>14} {'Z_LTE(Trad)':>14} {'tau_ratio':>10} {'Z_d/Z_inf':>10}")
print("-" * 80)

for z in ELEMENTS:
    max_ion = 0
    for zz, st in ion_map:
        if zz == z:
            max_ion = max(max_ion, st)

    for ion in range(max_ion + 1):
        # Get ion number density
        ip = -1
        for i, (zz, st) in enumerate(ion_map):
            if zz == z and st == ion:
                ip = i
                break
        if ip < 0:
            continue
        n_ion = ion_npy[ip, s]
        if n_ion < 1.0:
            continue

        # Find lines for this (Z, ion) with significant tau
        line_mask = ((line_list['atomic_number'].astype(int) == z) &
                     (line_list['ion_number'].astype(int) == ion))
        line_indices = np.where(line_mask)[0]
        sig_mask = tau_ref[line_indices, s] > 1e-5
        sig_indices = line_indices[sig_mask]

        if len(sig_indices) < 3:
            continue

        # Compute our partition function
        lvls = get_levels_for(z, ion)
        Z_meta = 0.0
        Z_non = 0.0
        beta_rad = 1.0 / (K_B * T_rad)
        beta_e = 1.0 / (K_B * T_e)

        for l in lvls:
            E = l['energy_eV'] * EV_TO_ERG
            g = int(l['g'])
            meta = int(l['metastable'])
            if meta:
                boltz = E / (K_B * T_e)
                if boltz < 500:
                    Z_meta += g * np.exp(-boltz)
            else:
                boltz = E / (K_B * T_rad)
                if boltz < 500:
                    Z_non += g * np.exp(-boltz)

        Z_dilute = max(Z_meta + W * Z_non, 1e-300)

        # LTE partition at T_rad
        Z_lte = 0.0
        for l in lvls:
            E = l['energy_eV'] * EV_TO_ERG
            g = int(l['g'])
            boltz = E / (K_B * T_rad)
            if boltz < 500:
                Z_lte += g * np.exp(-boltz)
        Z_lte = max(Z_lte, 1e-300)

        # Infer TARDIS Z from tau ratio
        # Pick a metastable line (w=1.0 so we don't need to figure out W)
        # tau = SOBOLEV * f_lu * lam * t_exp * (n_ion * w * g * exp(-E/kT) / Z)
        # For a metastable lower level: w=1
        # tau = SOBOLEV * f_lu * lam * t_exp * n_ion * g * exp(-E_lo*beta_rad) / Z
        # Z_tardis = SOBOLEV * f_lu * lam * t_exp * n_ion * g * exp(-E_lo*beta_rad) / tau_tardis

        # Try to infer Z from multiple lines
        Z_inferred_list = []
        for i in sig_indices[:20]:  # up to 20 lines
            lev_lo = int(line_list['level_number_lower'][i])
            f_lu = line_list['f_lu'][i]
            lam_cm = line_list['wavelength_cm'][i]

            lo_mask = lvls['level_number'].astype(int) == lev_lo
            lo = lvls[lo_mask]
            if len(lo) == 0:
                continue

            E_lo = lo['energy_eV'][0] * EV_TO_ERG
            g_lo = int(lo['g'][0])
            meta_lo = int(lo['metastable'][0])
            w_lo = 1.0 if meta_lo else W

            # Our n_lower
            n_lo_ours = n_ion * w_lo * g_lo * np.exp(-E_lo * beta_rad) / Z_dilute

            # Implied n_lower from TARDIS tau (ignoring stim correction for simplicity)
            # Actually need stim correction...
            lev_up = int(line_list['level_number_upper'][i])
            up_mask = lvls['level_number'].astype(int) == lev_up
            up = lvls[up_mask]
            if len(up) == 0:
                continue
            E_up = up['energy_eV'][0] * EV_TO_ERG
            g_up = int(up['g'][0])
            meta_up = int(up['metastable'][0])
            w_up = 1.0 if meta_up else W

            n_up_ours = n_ion * w_up * g_up * np.exp(-E_up * beta_rad) / Z_dilute
            stim = 1.0 - (g_lo * n_up_ours) / (g_up * n_lo_ours) if n_lo_ours > 0 and n_up_ours > 0 else 1.0
            stim = max(stim, 0)

            tau_ours = SOBOLEV_COEFF * f_lu * lam_cm * t_exp * n_lo_ours * stim
            tau_tardis = tau_ref[i, s]

            if tau_tardis > 0 and tau_ours > 0:
                # tau_ratio = tau_ours / tau_tardis ≈ Z_tardis / Z_ours
                ratio = tau_ours / tau_tardis
                Z_inf = Z_dilute * ratio  # Z_tardis ≈ Z_ours * ratio
                Z_inferred_list.append(Z_inf)

        if len(Z_inferred_list) > 0:
            Z_inferred = np.median(Z_inferred_list)
            tau_ratio_med = np.median([tau_ours / tau_tardis])
            # Actually compute tau ratio fresh
            tau_ours_list = []
            tau_tardis_list = []
            for i in sig_indices[:20]:
                tau_ours_list.append(1.0)  # placeholder
                tau_tardis_list.append(1.0)

            # Just use our Z_dilute / Z_inferred to get the ratio
            z_ratio = Z_dilute / Z_inferred
            print(f"{ELEM_NAMES[z]:>2} {ion:2d} ({ELEM_NAMES[z]} {'I'*(ion+1):>5}) "
                  f"{Z_dilute:14.6e} {Z_inferred:14.6e} {Z_lte:14.6e} "
                  f"{Z_dilute/Z_inferred:10.6f} {Z_dilute/Z_inferred:10.6f}")

# ================================================================
# Key test: What if we DON'T use dilute partition function?
# What if TARDIS uses LTE partition function for level populations too?
# ================================================================
print("\n" + "=" * 80)
print("Test: What partition function gives exact TARDIS tau?")
print("=" * 80)

# For S I (worst case: 2.4755x):
z, ion = 16, 0
print(f"\nS I analysis:")
lvls = get_levels_for(z, ion)
print(f"  Number of levels: {len(lvls)}")

# Count metastable vs non-metastable
n_meta = np.sum(lvls['metastable'].astype(int))
n_non = len(lvls) - n_meta
print(f"  Metastable: {n_meta}, Non-metastable: {n_non}")

# Show first few levels
print(f"  {'lev':>4} {'E_eV':>10} {'g':>3} {'meta':>5}")
for l in lvls[:10]:
    print(f"  {int(l['level_number']):4d} {l['energy_eV']:10.4f} {int(l['g']):3d} {int(l['metastable']):5d}")

# Compute partition sums
Z_meta_Te = 0.0
Z_meta_Trad = 0.0
Z_non_Te = 0.0
Z_non_Trad = 0.0
Z_all_Trad = 0.0
Z_all_Te = 0.0

for l in lvls:
    E = l['energy_eV'] * EV_TO_ERG
    g = int(l['g'])
    meta = int(l['metastable'])

    b_rad = E / (K_B * T_rad)
    b_e = E / (K_B * T_e)

    if b_rad < 500:
        Z_all_Trad += g * np.exp(-b_rad)
    if b_e < 500:
        Z_all_Te += g * np.exp(-b_e)

    if meta:
        if b_Te := b_e < 500:
            pass
        if b_e < 500:
            Z_meta_Te += g * np.exp(-b_e)
        if b_rad < 500:
            Z_meta_Trad += g * np.exp(-b_rad)
    else:
        if b_e < 500:
            Z_non_Te += g * np.exp(-b_e)
        if b_rad < 500:
            Z_non_Trad += g * np.exp(-b_rad)

Z_dilute = Z_meta_Te + W * Z_non_Trad
Z_lte_rad = Z_all_Trad
Z_lte_e = Z_all_Te

print(f"\n  Partition sums:")
print(f"    Z_meta(Te)    = {Z_meta_Te:.6f}")
print(f"    Z_meta(Trad)  = {Z_meta_Trad:.6f}")
print(f"    Z_non(Te)     = {Z_non_Te:.6f}")
print(f"    Z_non(Trad)   = {Z_non_Trad:.6f}")
print(f"    Z_dilute = Z_meta(Te) + W*Z_non(Trad) = {Z_dilute:.6f}")
print(f"    Z_LTE(Trad)   = {Z_lte_rad:.6f}")
print(f"    Z_LTE(Te)     = {Z_lte_e:.6f}")
print(f"    Z_dilute / Z_LTE(Trad) = {Z_dilute / Z_lte_rad:.6f}")

# The tau ratio for S I is 2.4755
# So Z_tardis / Z_ours = 2.4755
# Z_tardis = Z_ours * 2.4755 = {Z_dilute * 2.4755}
Z_implied = Z_dilute * 2.4755
print(f"\n  TARDIS implied Z = Z_dilute * 2.4755 = {Z_implied:.6f}")
print(f"  This is very close to Z_LTE(Trad) = {Z_lte_rad:.6f} (ratio {Z_implied/Z_lte_rad:.6f})")

# Actually check: TARDIS uses LTE partition for level populations too??
# No — let me compute tau with LTE partition
ip = -1
for i, (zz, st) in enumerate(ion_map):
    if zz == 16 and st == 0:
        ip = i
        break
n_ion = ion_npy[ip, s]

# Pick the first significant S I line
line_mask = ((line_list['atomic_number'].astype(int) == 16) &
             (line_list['ion_number'].astype(int) == 0))
line_indices = np.where(line_mask)[0]
sig = [i for i in line_indices if tau_ref[i, 0] > 1e-5]

if sig:
    i = sig[0]
    lev_lo = int(line_list['level_number_lower'][i])
    lo = lvls[lvls['level_number'].astype(int) == lev_lo]
    E_lo = lo['energy_eV'][0] * EV_TO_ERG
    g_lo = int(lo['g'][0])
    meta_lo = int(lo['metastable'][0])
    w_lo = 1.0 if meta_lo else W

    # With dilute partition
    n_lo_dilute = n_ion * w_lo * g_lo * np.exp(-E_lo / (K_B * T_rad)) / Z_dilute

    # With LTE partition at T_rad
    n_lo_lte = n_ion * w_lo * g_lo * np.exp(-E_lo / (K_B * T_rad)) / Z_lte_rad

    # With LTE partition at T_e
    n_lo_lte_Te = n_ion * w_lo * g_lo * np.exp(-E_lo / (K_B * T_rad)) / Z_lte_e

    print(f"\n  Example S I line {i} (lev_lo={lev_lo}, meta={meta_lo}):")
    print(f"    n_lo (dilute Z): {n_lo_dilute:.6e}")
    print(f"    n_lo (LTE@Trad): {n_lo_lte:.6e}")
    print(f"    n_lo (LTE@Te):   {n_lo_lte_Te:.6e}")

    # Compute tau with each
    f_lu = line_list['f_lu'][i]
    lam = line_list['wavelength_cm'][i]
    tau_dilute = SOBOLEV_COEFF * f_lu * lam * t_exp * n_lo_dilute
    tau_lte = SOBOLEV_COEFF * f_lu * lam * t_exp * n_lo_lte
    tau_lte_Te = SOBOLEV_COEFF * f_lu * lam * t_exp * n_lo_lte_Te
    tau_tardis_val = tau_ref[i, 0]

    print(f"    tau (dilute Z): {tau_dilute:.6e}")
    print(f"    tau (LTE@Trad): {tau_lte:.6e}")
    print(f"    tau (LTE@Te):   {tau_lte_Te:.6e}")
    print(f"    tau TARDIS:     {tau_tardis_val:.6e}")
    print(f"    ratio dilute:   {tau_dilute / tau_tardis_val:.4f}")
    print(f"    ratio LTE@Trad: {tau_lte / tau_tardis_val:.4f}")
    print(f"    ratio LTE@Te:   {tau_lte_Te / tau_tardis_val:.4f}")

# ================================================================
# Now check C I (1.99x discrepancy)
# ================================================================
print(f"\n\nC I analysis:")
z, ion = 6, 0
lvls = get_levels_for(z, ion)
n_meta = np.sum(lvls['metastable'].astype(int))
n_non = len(lvls) - n_meta
print(f"  Levels: {len(lvls)}, metastable: {n_meta}, non-metastable: {n_non}")

# Partition functions
Z_meta_Te = 0.0
Z_non_Trad = 0.0
Z_all_Trad = 0.0

for l in lvls:
    E = l['energy_eV'] * EV_TO_ERG
    g = int(l['g'])
    meta = int(l['metastable'])
    b_rad = E / (K_B * T_rad)
    b_e = E / (K_B * T_e)

    if b_rad < 500:
        Z_all_Trad += g * np.exp(-b_rad)
    if meta:
        if b_e < 500:
            Z_meta_Te += g * np.exp(-b_e)
    else:
        if b_rad < 500:
            Z_non_Trad += g * np.exp(-b_rad)

Z_dilute = Z_meta_Te + W * Z_non_Trad
print(f"  Z_meta(Te)={Z_meta_Te:.6f}, Z_non(Trad)={Z_non_Trad:.6f}")
print(f"  Z_dilute = {Z_dilute:.6f}")
print(f"  Z_LTE(Trad) = {Z_all_Trad:.6f}")
print(f"  Z_dilute / Z_LTE(Trad) = {Z_dilute / Z_all_Trad:.6f}")
print(f"  1 / tau_ratio (expect Z_dilute/Z_tardis) = {1.0 / 1.9881:.6f}")

# ================================================================
# KEY INSIGHT: For METASTABLE levels (w=1), level population =
#   n_k = n_ion * (1.0) * g_k * exp(-E_k / kT_?) / Z
#
# With dilute Z: Z = Z_meta(Te) + W * Z_non(Trad)
# With LTE Z:   Z = Z_all(T_rad)
#
# For NON-METASTABLE levels (w=W), level population =
#   n_k = n_ion * W * g_k * exp(-E_k / kT_rad) / Z
#
# If the lower level is metastable:
#   tau ∝ n_ion * g_lo * exp(-E_lo/kT_?) / Z
#   Using T_rad for Boltzmann: n_lo_dilute = n_ion * g * exp(-E/kT_rad) / Z_dilute
#   Using T_e for Boltzmann:   n_lo_dilute = n_ion * g * exp(-E/kT_e) / Z_dilute
#
# If the lower level is non-metastable:
#   tau ∝ n_ion * W * g_lo * exp(-E_lo/kT_rad) / Z
# ================================================================

# For S I with all-metastable lower levels, check:
print("\n\n" + "=" * 80)
print("FINAL CHECK: Try using T_e for metastable Boltzmann factor")
print("(even though we thought TARDIS uses T_rad for all)")
print("=" * 80)

for z_check, ion_check, name, expected_ratio in [(16, 0, 'S I', 2.4755), (6, 0, 'C I', 1.9881)]:
    lvls = get_levels_for(z_check, ion_check)

    Z_meta_Te = 0.0
    Z_non_Trad = 0.0
    for l in lvls:
        E = l['energy_eV'] * EV_TO_ERG
        g = int(l['g'])
        meta = int(l['metastable'])
        if meta:
            if E / (K_B * T_e) < 500:
                Z_meta_Te += g * np.exp(-E / (K_B * T_e))
        else:
            if E / (K_B * T_rad) < 500:
                Z_non_Trad += g * np.exp(-E / (K_B * T_rad))
    Z_dilute = Z_meta_Te + W * Z_non_Trad

    ip = -1
    for i, (zz, st) in enumerate(ion_map):
        if zz == z_check and st == ion_check:
            ip = i
            break
    n_ion = ion_npy[ip, s]

    # Pick a metastable line
    line_mask = ((line_list['atomic_number'].astype(int) == z_check) &
                 (line_list['ion_number'].astype(int) == ion_check))
    line_indices = np.where(line_mask)[0]
    for idx in line_indices:
        if tau_ref[idx, 0] < 1e-5:
            continue
        lev_lo = int(line_list['level_number_lower'][idx])
        lo = lvls[lvls['level_number'].astype(int) == lev_lo]
        if len(lo) == 0 or int(lo['metastable'][0]) == 0:
            continue

        E_lo = lo['energy_eV'][0] * EV_TO_ERG
        g_lo = int(lo['g'][0])

        # Using T_rad (our current approach)
        n_lo_Trad = n_ion * g_lo * np.exp(-E_lo / (K_B * T_rad)) / Z_dilute

        # Using T_e (alternative)
        n_lo_Te = n_ion * g_lo * np.exp(-E_lo / (K_B * T_e)) / Z_dilute

        f_lu = line_list['f_lu'][idx]
        lam = line_list['wavelength_cm'][idx]
        tau_Trad = SOBOLEV_COEFF * f_lu * lam * t_exp * n_lo_Trad
        tau_Te = SOBOLEV_COEFF * f_lu * lam * t_exp * n_lo_Te
        tau_tardis_val = tau_ref[idx, s]

        print(f"\n{name} line {idx} (meta lower level {lev_lo}, E={lo['energy_eV'][0]:.4f} eV):")
        print(f"  tau (T_rad for Boltz): {tau_Trad:.6e} (ratio={tau_Trad/tau_tardis_val:.4f})")
        print(f"  tau (T_e for Boltz):   {tau_Te:.6e} (ratio={tau_Te/tau_tardis_val:.4f})")
        print(f"  TARDIS tau:            {tau_tardis_val:.6e}")
        break

print("\nDone.")
