#!/usr/bin/env python3
"""Validate LUMINA plasma solver against TARDIS reference data.

Compares:
1. n_e (electron density) — LUMINA computed vs TARDIS reference
2. Ion number densities — LUMINA vs TARDIS reference (ion_number_density.npy)
3. tau_sobolev — LUMINA vs TARDIS reference (tau_sobolev.npy)
4. Partition functions — LUMINA vs TARDIS (computed from levels data)

Usage: python validate_plasma.py
"""
import numpy as np
import os

REF = "data/tardis_reference"
N_SHELLS = 30

# Elements in our model
ELEMENTS = {6: 'C', 8: 'O', 14: 'Si', 16: 'S', 20: 'Ca', 26: 'Fe', 27: 'Co', 28: 'Ni'}

# Physical constants (CGS)
K_B = 1.380649e-16      # erg/K
H_PLANCK = 6.62607015e-27  # erg*s
EV_TO_ERG = 1.602176634e-12
AMU = 1.660539066e-24    # g
M_E = 9.1093837015e-28   # g
SOBOLEV_COEFF = 2.6540281e-02
M_PI = 3.14159265358979323846

print("=" * 70)
print("LUMINA Plasma Solver Validation vs TARDIS Reference")
print("=" * 70)

# --- Load TARDIS reference data ---
print("\n--- Loading TARDIS reference ---")

# Plasma state (W, T_rad)
plasma = np.genfromtxt(f"{REF}/plasma_state.csv", delimiter=',', names=True)
W_ref = plasma['W']
T_rad_ref = plasma['T_rad']
print(f"  W[0]={W_ref[0]:.6f}, T_rad[0]={T_rad_ref[0]:.2f} K")

# Electron density
ne_csv = np.genfromtxt(f"{REF}/electron_densities.csv", delimiter=',', names=True)
n_e_ref = ne_csv['n_e']
print(f"  n_e[0]={n_e_ref[0]:.6e}, n_e[29]={n_e_ref[29]:.6e}")

# Density
rho_csv = np.genfromtxt(f"{REF}/density.csv", delimiter=',', names=True)
rho = rho_csv['rho']

# Ion number density from TARDIS
ion_npy = np.load(f"{REF}/ion_number_density.npy")
print(f"  ion_number_density shape: {ion_npy.shape}")

# tau_sobolev from TARDIS
tau_ref = np.load(f"{REF}/tau_sobolev.npy")
print(f"  tau_sobolev shape: {tau_ref.shape}")

# Abundances
abund_csv = np.genfromtxt(f"{REF}/abundances.csv", delimiter=',', skip_header=0, dtype=None, encoding='utf-8')
# Parse abundances CSV manually
with open(f"{REF}/abundances.csv") as f:
    hdr = f.readline().strip().split(',')
    abundances = {}
    for line in f:
        parts = line.strip().split(',')
        z = int(parts[0])
        vals = [float(x) for x in parts[1:]]
        abundances[z] = np.array(vals)

# Levels
levels = np.genfromtxt(f"{REF}/levels.csv", delimiter=',', names=True)
print(f"  Levels: {len(levels)} entries")

# Ionization energies
ioniz = np.genfromtxt(f"{REF}/ionization_energies.csv", delimiter=',', names=True)
print(f"  Ionization: {len(ioniz)} entries")

# Zeta data
zeta_ions_csv = np.genfromtxt(f"{REF}/zeta_ions.csv", delimiter=',', names=True)
zeta_temps_csv = np.genfromtxt(f"{REF}/zeta_temps.csv", delimiter=',', names=True)
zeta_data = np.load(f"{REF}/zeta_data.npy")
zeta_temps = zeta_temps_csv['temperature']
print(f"  Zeta: {zeta_data.shape[0]} ions x {zeta_data.shape[1]} temps")

# Atom masses
masses_csv = np.genfromtxt(f"{REF}/atom_masses.csv", delimiter=',', names=True)
atom_masses = {}
for row in masses_csv:
    atom_masses[int(row['atomic_number'])] = row['mass_amu']

# Line list (just first few columns for tau validation)
print("  Loading line list (this may take a moment)...")
line_list = np.genfromtxt(f"{REF}/line_list.csv", delimiter=',', names=True, max_rows=None)
print(f"  Lines: {len(line_list)} entries")

# ================================================================
# STEP 1: Reproduce TARDIS partition functions
# ================================================================
print("\n" + "=" * 70)
print("STEP 1: Partition Functions (Dilute)")
print("=" * 70)

def get_levels(z, ion):
    mask = (levels['atomic_number'].astype(int) == z) & (levels['ion_number'].astype(int) == ion)
    return levels[mask]

def compute_dilute_partition(z, ion, T_rad, W):
    """TARDIS dilute partition function: Z_meta(T_e) + W * Z_non_meta(T_rad)"""
    T_e = 0.9 * T_rad
    lvls = get_levels(z, ion)
    if len(lvls) == 0:
        return 1e-300

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

def compute_lte_partition(z, ion, T):
    """LTE partition function at temperature T"""
    lvls = get_levels(z, ion)
    if len(lvls) == 0:
        return 1e-300
    Z = 0.0
    for l in lvls:
        E = l['energy_eV'] * EV_TO_ERG
        g = int(l['g'])
        boltz = E / (K_B * T)
        if boltz < 500:
            Z += g * np.exp(-boltz)
    return max(Z, 1e-300)

# ================================================================
# STEP 2: Reproduce TARDIS ionization (nebular Saha)
# ================================================================
print("\n" + "=" * 70)
print("STEP 2: Ion Number Densities (Nebular Saha)")
print("=" * 70)

def get_ioniz_energy(z, ion):
    mask = (ioniz['atomic_number'].astype(int) == z) & (ioniz['ion_number'].astype(int) == ion)
    matched = ioniz[mask]
    if len(matched) > 0:
        return matched['ionization_energy_eV'][0]
    return 1e10

def interpolate_zeta(z, ion, T):
    """Interpolate zeta factor for (Z, ion) at temperature T"""
    for i in range(len(zeta_ions_csv)):
        if int(zeta_ions_csv['atomic_number'][i]) == z and int(zeta_ions_csv['ion_number'][i]) == ion:
            vals = zeta_data[i, :]
            if T <= zeta_temps[0]:
                return vals[0]
            if T >= zeta_temps[-1]:
                return vals[-1]
            idx = np.searchsorted(zeta_temps, T) - 1
            idx = max(0, min(idx, len(zeta_temps) - 2))
            frac = (T - zeta_temps[idx]) / (zeta_temps[idx + 1] - zeta_temps[idx])
            return vals[idx] + frac * (vals[idx + 1] - vals[idx])
    return 1.0  # no zeta data -> LTE

def get_n_ion_stages(z):
    """Count ion stages for element Z"""
    mask = ioniz['atomic_number'].astype(int) == z
    n_ioniz = np.sum(mask)
    return n_ioniz + 1  # n_ioniz energies -> n_ioniz+1 populations

def compute_ions_for_element(z, shell, n_e, W, T_rad, rho_s, abund):
    """Compute ion populations using TARDIS nebular Saha"""
    T_e = 0.9 * T_rad
    mass = atom_masses[z]
    n_element = (abund * rho_s) / (mass * AMU)

    n_stages = get_n_ion_stages(z)

    g_electron = (2.0 * M_PI * M_E * K_B * T_rad / (H_PLANCK * H_PLANCK)) ** 1.5
    beta_rad = 1.0 / (K_B * T_rad)
    beta_el = 1.0 / (K_B * T_e)

    ratios = []
    for k in range(n_stages - 1):
        # LTE partition functions at T_rad for Saha
        Z_cur = compute_lte_partition(z, k, T_rad)
        Z_next = compute_lte_partition(z, k + 1, T_rad)

        chi_eV = get_ioniz_energy(z, k)
        chi_erg = chi_eV * EV_TO_ERG

        phi_lte = (Z_next / Z_cur) * 2.0 * g_electron * np.exp(-chi_erg * beta_rad)
        delta = (T_e / T_rad) * np.exp(chi_erg * (beta_rad - beta_el))
        zeta = interpolate_zeta(z, k, T_rad)

        phi_neb = phi_lte * W * (zeta * delta + W * (1.0 - zeta)) * np.sqrt(T_e / T_rad)

        ratio = phi_neb / n_e if n_e > 0 else 1e10
        ratio = min(ratio, 1e30)
        ratios.append(ratio)

    # Normalize
    product = 1.0
    sum_norm = 1.0
    for r in ratios:
        product *= r
        if product > 1e30:
            product = 1e30
            sum_norm += product
            break
        sum_norm += product

    n_0 = n_element / sum_norm
    pops = [n_0]
    product = 1.0
    for r in ratios:
        product *= r
        pops.append(max(n_0 * product, 1e-300))

    return pops

def compute_n_e_iterative(shell, W, T_rad, rho_s):
    """Compute self-consistent n_e via iteration"""
    n_e = n_e_ref[shell]  # initialize from TARDIS reference

    elem_list = sorted(ELEMENTS.keys())

    for iteration in range(20):
        n_e_old = n_e
        n_e_new = 0.0

        for z in elem_list:
            abund = abundances[z][shell]
            pops = compute_ions_for_element(z, shell, n_e, W, T_rad, rho_s, abund)
            for stage, n_ion in enumerate(pops):
                n_e_new += stage * n_ion

        n_e = max(n_e_new, 1.0)

        if n_e_old > 0 and abs(n_e - n_e_old) / n_e_old < 1e-6:
            break

    return n_e

# ================================================================
# Compare n_e across all shells
# ================================================================
print("\nComputing LUMINA n_e for all shells...")
n_e_lumina = np.zeros(N_SHELLS)
for s in range(N_SHELLS):
    n_e_lumina[s] = compute_n_e_iterative(s, W_ref[s], T_rad_ref[s], rho[s])

print(f"\n{'Shell':>5} {'n_e_LUMINA':>14} {'n_e_TARDIS':>14} {'Ratio':>8} {'Error%':>8}")
print("-" * 55)
total_ne_err = 0.0
for s in range(N_SHELLS):
    ratio = n_e_lumina[s] / n_e_ref[s] if n_e_ref[s] > 0 else 0
    err = (n_e_lumina[s] - n_e_ref[s]) / n_e_ref[s] * 100 if n_e_ref[s] > 0 else 0
    total_ne_err += abs(err)
    print(f"{s:5d} {n_e_lumina[s]:14.6e} {n_e_ref[s]:14.6e} {ratio:8.4f} {err:+8.2f}")
print(f"\nMean |n_e error|: {total_ne_err / N_SHELLS:.2f}%")

# ================================================================
# Compare ion densities for key ions
# ================================================================
print("\n" + "=" * 70)
print("STEP 3: Ion Number Densities — Key Ions")
print("=" * 70)

# Decode ion_number_density.npy shape
# TARDIS stores this as (n_ions, n_shells) where n_ions follows a specific ordering
# The ion_number_density.npy contains ALL ions for the model elements
# Shape is typically (n_total_ion_stages, n_shells) = (153, 30)
print(f"\nTARDIS ion_number_density shape: {ion_npy.shape}")

# Build index mapping: TARDIS ion_npy is ordered by (Z, ion_stage)
# Build the same ordering as our C code
ion_map = []
elem_list = sorted(ELEMENTS.keys())
for z in elem_list:
    n_stages = get_n_ion_stages(z)
    for stage in range(n_stages):
        ion_map.append((z, stage))

print(f"Total ion populations in our model: {len(ion_map)}")
print(f"Expected to match ion_npy rows: {ion_npy.shape[0]}")

# If shapes match, compare directly
if len(ion_map) == ion_npy.shape[0]:
    print("\nKey ions comparison (shell 0, using TARDIS reference n_e):")
    print(f"{'Ion':>10} {'n_LUMINA':>14} {'n_TARDIS':>14} {'Ratio':>8} {'Error%':>8}")
    print("-" * 60)

    key_ions = [(14, 1, 'Si I'), (14, 2, 'Si II'), (14, 3, 'Si III'),
                (26, 1, 'Fe I'), (26, 2, 'Fe II'), (26, 3, 'Fe III'),
                (20, 2, 'Ca II'), (20, 3, 'Ca III'),
                (16, 2, 'S II'), (16, 3, 'S III'),
                (8, 1, 'O I'), (8, 2, 'O II')]

    s = 0  # shell 0
    for z_check, stage_check, name in key_ions:
        # Find index in ion_map
        idx = -1
        for i, (z, st) in enumerate(ion_map):
            if z == z_check and st == stage_check:
                idx = i
                break
        if idx < 0:
            continue

        # Compute LUMINA value
        abund = abundances[z_check][s]
        pops = compute_ions_for_element(z_check, s, n_e_ref[s], W_ref[s], T_rad_ref[s], rho[s], abund)
        n_lumina = pops[stage_check] if stage_check < len(pops) else 0.0
        n_tardis = ion_npy[idx, s]

        if n_tardis > 1e-10:
            ratio = n_lumina / n_tardis
            err = (n_lumina - n_tardis) / n_tardis * 100
            print(f"{name:>10} {n_lumina:14.6e} {n_tardis:14.6e} {ratio:8.4f} {err:+8.2f}")
        else:
            print(f"{name:>10} {n_lumina:14.6e} {n_tardis:14.6e} {'N/A':>8} {'N/A':>8}")

# ================================================================
# Compare across ALL shells for dominant ions
# ================================================================
print("\n" + "=" * 70)
print("STEP 4: Ion Densities Across All Shells (Fe II, Si II)")
print("=" * 70)

dominant_ions = [(26, 2, 'Fe II'), (14, 2, 'Si II'), (20, 2, 'Ca II')]
for z_check, stage_check, name in dominant_ions:
    idx = -1
    for i, (z, st) in enumerate(ion_map):
        if z == z_check and st == stage_check:
            idx = i
            break
    if idx < 0:
        continue

    print(f"\n{name}:")
    print(f"{'Shell':>5} {'n_LUMINA':>14} {'n_TARDIS':>14} {'Ratio':>8} {'Error%':>8}")
    print("-" * 55)
    total_err = 0.0
    for s in range(N_SHELLS):
        abund = abundances[z_check][s]
        pops = compute_ions_for_element(z_check, s, n_e_ref[s], W_ref[s], T_rad_ref[s], rho[s], abund)
        n_lumina = pops[stage_check] if stage_check < len(pops) else 0.0
        n_tardis = ion_npy[idx, s]

        if n_tardis > 1e-10:
            ratio = n_lumina / n_tardis
            err = (n_lumina - n_tardis) / n_tardis * 100
            total_err += abs(err)
            if s % 5 == 0 or abs(err) > 5:  # Print every 5th shell or big errors
                print(f"{s:5d} {n_lumina:14.6e} {n_tardis:14.6e} {ratio:8.4f} {err:+8.2f}")
    print(f"  Mean |error|: {total_err / N_SHELLS:.2f}%")

# ================================================================
# tau_sobolev comparison
# ================================================================
print("\n" + "=" * 70)
print("STEP 5: tau_sobolev Comparison")
print("=" * 70)

# For tau, we need to compute level populations → tau
# Let's do a quick summary: compare tau from LUMINA C code (the validation CSV) against reference
if os.path.exists("lumina_tau_validation.csv"):
    val = np.genfromtxt("lumina_tau_validation.csv", delimiter=',', skip_header=2, usecols=(1,))
    # tau_ref shape: (n_lines, n_shells)
    tau_tardis_s0 = tau_ref[:, 0]
    tau_lumina_s0 = val[:len(tau_tardis_s0)]  # might need truncating

    # Only compare lines with significant tau
    mask = tau_tardis_s0 > 1e-10
    ratios = tau_lumina_s0[mask] / tau_tardis_s0[mask]

    print(f"\nLines with tau > 1e-10: {np.sum(mask)} / {len(tau_tardis_s0)}")
    print(f"  Median ratio (LUMINA/TARDIS): {np.median(ratios):.4f}")
    print(f"  Mean ratio: {np.mean(ratios):.4f}")
    print(f"  Std ratio: {np.std(ratios):.4f}")
    print(f"  P10-P90: [{np.percentile(ratios, 10):.4f}, {np.percentile(ratios, 90):.4f}]")

    # Break down by element
    Z_line = line_list['atomic_number'].astype(int)
    for z in sorted(ELEMENTS.keys()):
        elem_mask = mask & (Z_line[:len(mask)] == z)
        if np.sum(elem_mask) > 0:
            elem_ratios = tau_lumina_s0[elem_mask] / tau_tardis_s0[elem_mask]
            print(f"  {ELEMENTS[z]:>2} (Z={z:2d}): N={np.sum(elem_mask):6d}, "
                  f"median={np.median(elem_ratios):.4f}, "
                  f"mean={np.mean(elem_ratios):.4f}")
else:
    print("  lumina_tau_validation.csv not found. Run with LUMINA_VALIDATE_PLASMA=1")

# ================================================================
# DETAILED n_e INVESTIGATION
# ================================================================
print("\n" + "=" * 70)
print("STEP 6: n_e Investigation — Where Does the Difference Come From?")
print("=" * 70)

# Compute contribution to n_e from each element
s = 0  # innermost shell
print(f"\nShell 0: n_e contribution breakdown")
print(f"{'Element':>8} {'Ion':>8} {'n_LUMINA':>14} {'n_TARDIS':>14} {'contrib_LUMINA':>16} {'contrib_TARDIS':>16}")
print("-" * 85)

n_e_parts_lumina = {}
n_e_parts_tardis = {}

for z in sorted(ELEMENTS.keys()):
    n_stages = get_n_ion_stages(z)
    abund = abundances[z][s]
    pops_lumina = compute_ions_for_element(z, s, n_e_ref[s], W_ref[s], T_rad_ref[s], rho[s], abund)

    for stage in range(n_stages):
        idx = -1
        for i, (zz, st) in enumerate(ion_map):
            if zz == z and st == stage:
                idx = i
                break
        if idx < 0:
            continue

        n_l = pops_lumina[stage] if stage < len(pops_lumina) else 0.0
        n_t = ion_npy[idx, s]

        contrib_l = stage * n_l
        contrib_t = stage * n_t

        if contrib_l > 1e3 or contrib_t > 1e3:
            n_e_parts_lumina[(z, stage)] = contrib_l
            n_e_parts_tardis[(z, stage)] = contrib_t
            print(f"{ELEMENTS[z]:>8} {stage:>8d} {n_l:14.6e} {n_t:14.6e} {contrib_l:16.6e} {contrib_t:16.6e}")

total_l = sum(n_e_parts_lumina.values())
total_t = sum(n_e_parts_tardis.values())
print(f"{'TOTAL':>8} {'':>8} {'':>14} {'':>14} {total_l:16.6e} {total_t:16.6e}")
print(f"\nn_e LUMINA = {total_l:.6e}")
print(f"n_e TARDIS = {n_e_ref[s]:.6e}")
print(f"n_e error = {(total_l - n_e_ref[s]) / n_e_ref[s] * 100:.2f}%")

# ================================================================
# Check whether the issue is with partition function or ionization
# ================================================================
print("\n" + "=" * 70)
print("STEP 7: Partition Function Check — Dilute vs LTE")
print("=" * 70)

print(f"\nShell 0: T_rad={T_rad_ref[0]:.2f} K, T_e={0.9*T_rad_ref[0]:.2f} K, W={W_ref[0]:.6f}")
print(f"{'Ion':>10} {'Z_dilute':>14} {'Z_LTE(Trad)':>14} {'Z_LTE(Te)':>14} {'Dilute/LTE':>12}")
print("-" * 70)

key_ions_pf = [(14, 1, 'Si I'), (14, 2, 'Si II'), (14, 3, 'Si III'),
               (26, 1, 'Fe I'), (26, 2, 'Fe II'), (26, 3, 'Fe III'),
               (20, 1, 'Ca I'), (20, 2, 'Ca II')]

for z, ion, name in key_ions_pf:
    Z_dil = compute_dilute_partition(z, ion, T_rad_ref[0], W_ref[0])
    Z_lte_rad = compute_lte_partition(z, ion, T_rad_ref[0])
    Z_lte_e = compute_lte_partition(z, ion, 0.9 * T_rad_ref[0])
    ratio = Z_dil / Z_lte_rad if Z_lte_rad > 0 else 0
    print(f"{name:>10} {Z_dil:14.6e} {Z_lte_rad:14.6e} {Z_lte_e:14.6e} {ratio:12.6f}")

print("\nDone.")
