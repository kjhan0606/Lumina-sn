#!/usr/bin/env python3
"""Deep investigation of tau_sobolev discrepancy.

Focus: Why are Si and C lines showing 2-3x tau overestimate?
Hypothesis: The level population formula differs from TARDIS.
"""
import numpy as np

REF = "data/tardis_reference"
N_SHELLS = 30

K_B = 1.380649e-16
H_PLANCK = 6.62607015e-27
EV_TO_ERG = 1.602176634e-12
AMU = 1.660539066e-24
M_E = 9.1093837015e-28
SOBOLEV_COEFF = 2.6540281e-02
C_LIGHT = 2.99792458e10
M_PI = 3.14159265358979323846

# Load data
plasma = np.genfromtxt(f"{REF}/plasma_state.csv", delimiter=',', names=True)
W_ref = plasma['W']
T_rad_ref = plasma['T_rad']

ne_csv = np.genfromtxt(f"{REF}/electron_densities.csv", delimiter=',', names=True)
n_e_ref = ne_csv['n_e']

rho_csv = np.genfromtxt(f"{REF}/density.csv", delimiter=',', names=True)
rho = rho_csv['rho']

levels = np.genfromtxt(f"{REF}/levels.csv", delimiter=',', names=True)
ioniz = np.genfromtxt(f"{REF}/ionization_energies.csv", delimiter=',', names=True)

ion_npy = np.load(f"{REF}/ion_number_density.npy")
tau_ref = np.load(f"{REF}/tau_sobolev.npy")

# Load line list
print("Loading line list...")
line_list = np.genfromtxt(f"{REF}/line_list.csv", delimiter=',', names=True)
n_lines = len(line_list)

# Config
import json
with open(f"{REF}/config.json") as f:
    config = json.load(f)
t_exp = config['time_explosion_s']

# Build ion map
ELEMENTS = [6, 8, 14, 16, 20, 26, 27, 28]
ELEM_NAMES = {6: 'C', 8: 'O', 14: 'Si', 16: 'S', 20: 'Ca', 26: 'Fe', 27: 'Co', 28: 'Ni'}

ion_map = []
for z in ELEMENTS:
    mask = ioniz['atomic_number'].astype(int) == z
    n_ioniz = np.sum(mask)
    for stage in range(n_ioniz + 1):
        ion_map.append((z, stage))

def get_levels_for(z, ion):
    mask = (levels['atomic_number'].astype(int) == z) & (levels['ion_number'].astype(int) == ion)
    return levels[mask]

def compute_dilute_partition(z, ion, T_rad, W):
    T_e = 0.9 * T_rad
    lvls = get_levels_for(z, ion)
    Z_meta = 0.0
    Z_non = 0.0
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
    return max(Z_meta + W * Z_non, 1e-300)

# ================================================================
# Pick specific lines with large discrepancy to debug
# ================================================================
print("\n" + "=" * 70)
print("Investigating tau discrepancy for specific lines")
print("=" * 70)

# Load LUMINA tau from validation CSV
lumina_tau_s0 = None
import os
if os.path.exists("lumina_tau_validation.csv"):
    vals = []
    with open("lumina_tau_validation.csv") as f:
        for line in f:
            if line.startswith('#') or line.startswith('line'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 2:
                vals.append(float(parts[1]))
    lumina_tau_s0 = np.array(vals)

# Find Si II 6355 line
si_mask = (line_list['atomic_number'].astype(int) == 14) & (line_list['ion_number'].astype(int) == 1)
si_lines = np.where(si_mask)[0]
print(f"\nSi II lines: {len(si_lines)}")

# Find the 6355A line
for idx in si_lines:
    wl = line_list['wavelength_cm'][idx] * 1e8  # to Angstrom
    if 6300 < wl < 6400:
        nu = line_list['nu'][idx]
        tau_t = tau_ref[idx, 0]
        tau_l = lumina_tau_s0[idx] if lumina_tau_s0 is not None else None
        f_lu = line_list['f_lu'][idx]
        lev_lo = int(line_list['level_number_lower'][idx])
        lev_up = int(line_list['level_number_upper'][idx])
        print(f"\n  Si II 6355: line_idx={idx}, wl={wl:.2f}A, f_lu={f_lu:.6f}")
        print(f"    tau_TARDIS={tau_t:.6e}, tau_LUMINA={tau_l:.6e}" if tau_l else f"    tau_TARDIS={tau_t:.6e}")
        if tau_l:
            print(f"    ratio={tau_l/tau_t:.4f}")
        print(f"    level_lower={lev_lo}, level_upper={lev_up}")

# ================================================================
# Manual tau computation for the worst-case lines
# ================================================================
print("\n" + "=" * 70)
print("Manual tau computation for selected lines (shell 0)")
print("=" * 70)

s = 0
T_rad = T_rad_ref[s]
T_e = 0.9 * T_rad
W = W_ref[s]

# Find WORST lines by discrepancy ratio
if lumina_tau_s0 is not None:
    mask = tau_ref[:, 0] > 1e-5
    ratios = np.where(mask, lumina_tau_s0 / tau_ref[:, 0], 1.0)
    worst = np.argsort(np.abs(np.log(ratios)))[::-1]

    print(f"\nTop 20 worst-matching lines (tau > 1e-5):")
    print(f"{'idx':>6} {'Z':>3} {'ion':>4} {'wl_A':>10} {'f_lu':>10} {'lev_lo':>6} {'lev_up':>6} {'tau_LUM':>12} {'tau_TAR':>12} {'ratio':>8}")
    print("-" * 95)

    count = 0
    for i in worst:
        if not mask[i]:
            continue
        z = int(line_list['atomic_number'][i])
        ion = int(line_list['ion_number'][i])
        wl = line_list['wavelength_cm'][i] * 1e8
        f_lu = line_list['f_lu'][i]
        lev_lo = int(line_list['level_number_lower'][i])
        lev_up = int(line_list['level_number_upper'][i])
        tau_l = lumina_tau_s0[i]
        tau_t = tau_ref[i, 0]
        r = tau_l / tau_t if tau_t > 0 else 0

        print(f"{i:6d} {z:3d} {ion:4d} {wl:10.2f} {f_lu:10.6f} {lev_lo:6d} {lev_up:6d} {tau_l:12.4e} {tau_t:12.4e} {r:8.4f}")
        count += 1
        if count >= 20:
            break

# ================================================================
# Compare TARDIS level populations vs our formula
# ================================================================
print("\n" + "=" * 70)
print("TARDIS Level Population Check")
print("=" * 70)

# For the Si II 6355 line, manually compute n_lower and n_upper
# and compare against what produces the TARDIS tau
z = 14
ion = 1  # Si II

# Find the line
for idx in si_lines:
    wl = line_list['wavelength_cm'][idx] * 1e8
    if 6300 < wl < 6400:
        lev_lo = int(line_list['level_number_lower'][idx])
        lev_up = int(line_list['level_number_upper'][idx])
        f_lu = line_list['f_lu'][idx]
        lam_cm = line_list['wavelength_cm'][idx]

        # Get ion number density (from TARDIS reference)
        ip = -1
        for i, (zz, st) in enumerate(ion_map):
            if zz == z and st == ion:
                ip = i
                break
        n_ion = ion_npy[ip, s]

        # Get level data
        lvls = get_levels_for(z, ion)
        Z_part = compute_dilute_partition(z, ion, T_rad, W)

        lo_mask = lvls['level_number'].astype(int) == lev_lo
        up_mask = lvls['level_number'].astype(int) == lev_up

        lo_data = lvls[lo_mask]
        up_data = lvls[up_mask]

        if len(lo_data) == 0 or len(up_data) == 0:
            continue

        E_lo = lo_data['energy_eV'][0]
        E_up = up_data['energy_eV'][0]
        g_lo = int(lo_data['g'][0])
        g_up = int(up_data['g'][0])
        meta_lo = int(lo_data['metastable'][0])
        meta_up = int(up_data['metastable'][0])

        beta_rad = 1.0 / (K_B * T_rad)
        beta_e = 1.0 / (K_B * T_e)

        # Our formula: BOTH levels use T_rad
        w_lo = 1.0 if meta_lo else W
        w_up = 1.0 if meta_up else W

        boltz_lo = E_lo * EV_TO_ERG * beta_rad
        boltz_up = E_up * EV_TO_ERG * beta_rad

        n_lo = n_ion * w_lo * g_lo * np.exp(-boltz_lo) / Z_part
        n_up = n_ion * w_up * g_up * np.exp(-boltz_up) / Z_part

        stim = 1.0 - (g_lo * n_up) / (g_up * n_lo) if n_lo > 0 and n_up > 0 else 1.0
        if stim < 0:
            stim = 0.0

        tau_manual = SOBOLEV_COEFF * f_lu * lam_cm * t_exp * n_lo * stim
        tau_tardis = tau_ref[idx, 0]

        print(f"\nSi II 6355A (line {idx}):")
        print(f"  n_ion(Si II) = {n_ion:.6e}")
        print(f"  Z_part(dilute) = {Z_part:.6f}")
        print(f"  Level {lev_lo}: E={E_lo:.4f} eV, g={g_lo}, meta={meta_lo}")
        print(f"  Level {lev_up}: E={E_up:.4f} eV, g={g_up}, meta={meta_up}")
        print(f"  beta_rad = {beta_rad:.6e}")
        print(f"  boltz_lo = {boltz_lo:.6f}")
        print(f"  boltz_up = {boltz_up:.6f}")
        print(f"  w_lo = {w_lo}, w_up = {w_up}")
        print(f"  n_lower = {n_lo:.6e}")
        print(f"  n_upper = {n_up:.6e}")
        print(f"  stim_corr = {stim:.10f}")
        print(f"  tau_manual = {tau_manual:.6e}")
        print(f"  tau_TARDIS = {tau_tardis:.6e}")
        print(f"  ratio = {tau_manual/tau_tardis:.6f}")

        # Try the alternative: what tau_TARDIS implies for n_lower
        n_lo_implied = tau_tardis / (SOBOLEV_COEFF * f_lu * lam_cm * t_exp * stim)
        print(f"\n  n_lower implied by TARDIS tau: {n_lo_implied:.6e}")
        print(f"  Our n_lower / TARDIS implied: {n_lo / n_lo_implied:.6f}")

        # What if TARDIS uses T_e for metastable Boltzmann?
        boltz_lo_Te = E_lo * EV_TO_ERG * beta_e
        n_lo_alt = n_ion * w_lo * g_lo * np.exp(-boltz_lo_Te) / Z_part
        tau_alt = SOBOLEV_COEFF * f_lu * lam_cm * t_exp * n_lo_alt * stim
        print(f"\n  Alternative (T_e for metastable Boltzmann):")
        print(f"  n_lower_alt = {n_lo_alt:.6e}")
        print(f"  tau_alt = {tau_alt:.6e}")
        print(f"  ratio_alt = {tau_alt/tau_tardis:.6f}")

        break

# ================================================================
# Check SPECIFIC Si lines with large discrepancy
# ================================================================
print("\n" + "=" * 70)
print("Si lines tau detail (top 10 by |log(ratio)|)")
print("=" * 70)

si_all_mask = (line_list['atomic_number'].astype(int) == 14)
si_indices = np.where(si_all_mask)[0]

if lumina_tau_s0 is not None:
    si_tau_mask = (tau_ref[si_indices, 0] > 1e-5)
    si_filtered = si_indices[si_tau_mask]
    si_ratios = lumina_tau_s0[si_filtered] / tau_ref[si_filtered, 0]

    # Group by ion
    for ion_check in [0, 1, 2, 3]:
        ion_sub = [i for i in si_filtered if int(line_list['ion_number'][i]) == ion_check]
        if len(ion_sub) == 0:
            continue
        sub_ratios = [lumina_tau_s0[i] / tau_ref[i, 0] for i in ion_sub]
        print(f"\n  Si ion={ion_check}: {len(ion_sub)} lines with tau>1e-5")
        print(f"    median ratio: {np.median(sub_ratios):.4f}")
        print(f"    min/max ratio: {np.min(sub_ratios):.4f} / {np.max(sub_ratios):.4f}")

        # Show a few lines
        worst_sub = sorted(range(len(ion_sub)), key=lambda j: abs(np.log(sub_ratios[j])), reverse=True)
        for j in worst_sub[:5]:
            i = ion_sub[j]
            wl = line_list['wavelength_cm'][i] * 1e8
            lev_lo = int(line_list['level_number_lower'][i])
            meta = int(get_levels_for(14, ion_check)[
                get_levels_for(14, ion_check)['level_number'].astype(int) == lev_lo
            ]['metastable'][0]) if len(get_levels_for(14, ion_check)[
                get_levels_for(14, ion_check)['level_number'].astype(int) == lev_lo
            ]) > 0 else -1
            print(f"    line {i}: wl={wl:.1f}A, lev_lo={lev_lo}, meta={meta}, ratio={sub_ratios[j]:.4f}")

# ================================================================
# Check C lines (1.99x median discrepancy)
# ================================================================
print("\n" + "=" * 70)
print("C lines tau detail")
print("=" * 70)

c_all_mask = (line_list['atomic_number'].astype(int) == 6)
c_indices = np.where(c_all_mask)[0]

if lumina_tau_s0 is not None:
    for ion_check in [0, 1, 2, 3]:
        ion_sub = [i for i in c_indices if int(line_list['ion_number'][i]) == ion_check and tau_ref[i, 0] > 1e-5]
        if len(ion_sub) == 0:
            continue
        sub_ratios = [lumina_tau_s0[i] / tau_ref[i, 0] for i in ion_sub]
        print(f"\n  C ion={ion_check}: {len(ion_sub)} lines with tau>1e-5")
        print(f"    median ratio: {np.median(sub_ratios):.4f}")

        # Detail one line
        i = ion_sub[0]
        wl = line_list['wavelength_cm'][i] * 1e8
        lev_lo = int(line_list['level_number_lower'][i])
        lev_up = int(line_list['level_number_upper'][i])
        z = 6
        ion = ion_check

        # Compute manually
        ip = -1
        for ii, (zz, st) in enumerate(ion_map):
            if zz == z and st == ion:
                ip = ii
                break
        n_ion_val = ion_npy[ip, s]
        Z_part = compute_dilute_partition(z, ion, T_rad, W)

        lvls = get_levels_for(z, ion)
        lo = lvls[lvls['level_number'].astype(int) == lev_lo]
        up = lvls[lvls['level_number'].astype(int) == lev_up]

        if len(lo) > 0 and len(up) > 0:
            E_lo = lo['energy_eV'][0]
            g_lo = int(lo['g'][0])
            meta_lo = int(lo['metastable'][0])
            w_lo = 1.0 if meta_lo else W

            beta_rad = 1.0 / (K_B * T_rad)
            n_lo = n_ion_val * w_lo * g_lo * np.exp(-E_lo * EV_TO_ERG * beta_rad) / Z_part

            print(f"    Example: line {i}, wl={wl:.1f}A, lev_lo={lev_lo} (meta={meta_lo})")
            print(f"      n_ion={n_ion_val:.4e}, Z_part={Z_part:.4f}, n_lower={n_lo:.4e}")
            print(f"      E_lo={E_lo:.4f} eV, g_lo={g_lo}, w_lo={w_lo}")
            n_lo_implied = tau_ref[i, 0] / (SOBOLEV_COEFF * line_list['f_lu'][i] *
                                              line_list['wavelength_cm'][i] * t_exp)
            print(f"      n_lower from TARDIS tau: {n_lo_implied:.4e}")
            print(f"      Our/TARDIS ratio: {n_lo/n_lo_implied:.4f}")

print("\nDone.")
