#!/usr/bin/env python3
"""Debug the ~2.5x tau discrepancy for neutral atoms."""
import numpy as np

REF = "data/tardis_reference"
K_B = 1.380649e-16
EV_TO_ERG = 1.602176634e-12
AMU = 1.660539066e-24
M_E = 9.1093837015e-28
SOBOLEV_COEFF = 2.6540281e-02
M_PI = 3.14159265358979323846
H_PLANCK = 6.62607015e-27

plasma = np.genfromtxt(f"{REF}/plasma_state.csv", delimiter=',', names=True)
W_ref = plasma['W']
T_rad_ref = plasma['T_rad']
ne_csv = np.genfromtxt(f"{REF}/electron_densities.csv", delimiter=',', names=True)
n_e_ref = ne_csv['n_e']

ion_npy = np.load(f"{REF}/ion_number_density.npy")
tau_ref = np.load(f"{REF}/tau_sobolev.npy")
levels = np.genfromtxt(f"{REF}/levels.csv", delimiter=',', names=True)
line_list = np.genfromtxt(f"{REF}/line_list.csv", delimiter=',', names=True)
ioniz = np.genfromtxt(f"{REF}/ionization_energies.csv", delimiter=',', names=True)

import json
with open(f"{REF}/config.json") as f:
    config = json.load(f)
t_exp = config['time_explosion_s']

ELEMENTS = [6, 8, 14, 16, 20, 26, 27, 28]
ion_map = []
for z in ELEMENTS:
    mask = ioniz['atomic_number'].astype(int) == z
    n_ioniz = np.sum(mask)
    for stage in range(n_ioniz + 1):
        ion_map.append((z, stage))

s = 0  # Shell 0
T_rad = T_rad_ref[s]
W = W_ref[s]

def get_levels_for(z, ion):
    mask = (levels['atomic_number'].astype(int) == z) & (levels['ion_number'].astype(int) == ion)
    return levels[mask]

# STEP 1: Compare n_ion for Fe I from TARDIS reference vs our computation
print("=" * 70)
print("Fe I tau analysis")
print("=" * 70)

# TARDIS Fe I ion density
for i, (z, st) in enumerate(ion_map):
    if z == 26 and st == 0:
        fe0_idx = i
    if z == 26 and st == 1:
        fe1_idx = i
    if z == 26 and st == 2:
        fe2_idx = i

n_fe0_tardis = ion_npy[fe0_idx, s]
n_fe1_tardis = ion_npy[fe1_idx, s]
n_fe2_tardis = ion_npy[fe2_idx, s]

print(f"TARDIS: Fe I = {n_fe0_tardis:.6e}, Fe II = {n_fe1_tardis:.6e}, Fe III = {n_fe2_tardis:.6e}")
print(f"TARDIS: Fe I / Fe II = {n_fe0_tardis/n_fe1_tardis:.6e}")

# Now let's check: what partition function does TARDIS use?
# The key is: for tau, TARDIS uses:
#   n_lower = n_ion * bf_lower / Z_partition
# where bf_lower and Z_partition BOTH use the dilute Boltzmann factors

# Our tau for Fe I lines uses:
#   Z_part = Z_meta(T_rad) + W * Z_non(T_rad)  [the FIXED version]
# TARDIS uses: Z = sum(bf) = sum_meta(g*exp(-E/kT_rad)) + W*sum_nonmeta(g*exp(-E/kT_rad))

# These should be identical now. So where does 2.1x come from?

# Pick a Fe I line
fe0_lines = np.where((line_list['atomic_number'].astype(int) == 26) &
                     (line_list['ion_number'].astype(int) == 0))[0]
sig_fe0 = [i for i in fe0_lines if tau_ref[i, 0] > 1e-5]

if sig_fe0:
    idx = sig_fe0[0]
    lev_lo = int(line_list['level_number_lower'][idx])
    lev_up = int(line_list['level_number_upper'][idx])
    f_lu = line_list['f_lu'][idx]
    lam = line_list['wavelength_cm'][idx]
    wl = lam * 1e8

    lvls = get_levels_for(26, 0)

    # Compute partition function (FIXED: T_rad for all)
    Z_meta = 0.0
    Z_non = 0.0
    beta_rad = 1.0 / (K_B * T_rad)
    for l in lvls:
        E = l['energy_eV'] * EV_TO_ERG
        g = int(l['g'])
        meta = int(l['metastable'])
        boltz = E * beta_rad
        if boltz < 500:
            bf = g * np.exp(-boltz)
            if meta:
                Z_meta += bf
            else:
                Z_non += bf
    Z_dilute = Z_meta + W * Z_non

    lo = lvls[lvls['level_number'].astype(int) == lev_lo]
    up = lvls[lvls['level_number'].astype(int) == lev_up]
    E_lo = lo['energy_eV'][0] * EV_TO_ERG
    g_lo = int(lo['g'][0])
    meta_lo = int(lo['metastable'][0])
    w_lo = 1.0 if meta_lo else W

    n_lower = n_fe0_tardis * w_lo * g_lo * np.exp(-E_lo * beta_rad) / Z_dilute

    stim = 1.0
    if len(up) > 0:
        E_up = up['energy_eV'][0] * EV_TO_ERG
        g_up = int(up['g'][0])
        meta_up = int(up['metastable'][0])
        w_up = 1.0 if meta_up else W
        n_upper = n_fe0_tardis * w_up * g_up * np.exp(-E_up * beta_rad) / Z_dilute
        stim = 1.0 - (g_lo * n_upper) / (g_up * n_lower)
        stim = max(stim, 0)

    tau_ours = SOBOLEV_COEFF * f_lu * lam * t_exp * n_lower * stim
    tau_tardis = tau_ref[idx, 0]

    print(f"\nFe I example line {idx}: wl={wl:.1f}A, lev_lo={lev_lo}, meta_lo={meta_lo}")
    print(f"  Z_meta(T_rad) = {Z_meta:.6f}")
    print(f"  Z_non(T_rad)  = {Z_non:.6f}")
    print(f"  Z_dilute      = {Z_dilute:.6f}")
    print(f"  n_Fe_I (TARDIS) = {n_fe0_tardis:.6e}")
    print(f"  n_lower = {n_lower:.6e}")
    print(f"  stim    = {stim:.8f}")
    print(f"  tau_ours  = {tau_ours:.6e}")
    print(f"  tau_TARDIS = {tau_tardis:.6e}")
    print(f"  ratio = {tau_ours/tau_tardis:.6f}")

    # What n_lower gives TARDIS tau?
    n_lo_implied = tau_tardis / (SOBOLEV_COEFF * f_lu * lam * t_exp * stim)
    print(f"\n  n_lower implied by TARDIS tau: {n_lo_implied:.6e}")
    print(f"  Our n_lower / implied: {n_lower / n_lo_implied:.6f}")

    # What Z_part gives TARDIS tau?
    Z_implied = n_fe0_tardis * w_lo * g_lo * np.exp(-E_lo * beta_rad) * stim * \
                SOBOLEV_COEFF * f_lu * lam * t_exp / tau_tardis
    print(f"\n  Z_part implied by TARDIS tau: {Z_implied:.6f}")
    print(f"  Our Z_dilute / implied: {Z_dilute / Z_implied:.6f}")

# STEP 2: Check if the issue is stimulated emission
print("\n\n" + "=" * 70)
print("Check stimulated emission correction")
print("=" * 70)

# The stim_corr = 1 - (g_lo * n_up) / (g_up * n_lo)
# If n_up/n_lo is different, stim_corr differs
# For ground state transitions with high upper level, stim ≈ 1.0

# STEP 3: Maybe TARDIS uses a DIFFERENT partition function for tau
# TARDIS has ThermalLTEPartitionFunction which uses T_e
# Maybe TARDIS uses that for level populations of neutrals?
# Let me check what "general_level_boltzmann_factor" is actually used for tau
print("\n\nChecking if TARDIS tau_sobolev uses different level populations...")
print("TARDIS tau = sobolev_coeff * f_lu * lambda * t_exp * n_lower * stim")
print("n_lower = n_ion * bf_lower / Z_part")
print("where bf and Z use the dilute formula")

# Actually, let me try: what if tau uses bf from LTE (no dilute correction)?
# n_lower_lte = n_ion * g * exp(-E/kT_rad) / Z_lte(T_rad)
Z_lte_rad = 0.0
for l in lvls:
    E = l['energy_eV'] * EV_TO_ERG
    g = int(l['g'])
    boltz = E * beta_rad
    if boltz < 500:
        Z_lte_rad += g * np.exp(-boltz)

n_lower_lte = n_fe0_tardis * g_lo * np.exp(-E_lo * beta_rad) / Z_lte_rad
tau_lte = SOBOLEV_COEFF * f_lu * lam * t_exp * n_lower_lte * stim
print(f"\n  With LTE partition (no W weighting):")
print(f"  Z_lte = {Z_lte_rad:.6f}")
print(f"  n_lower_lte = {n_lower_lte:.6e}")
print(f"  tau_lte = {tau_lte:.6e}")
print(f"  ratio = {tau_lte/tau_tardis:.6f}")

# STEP 4: Key test — What if the level population uses bf/Z where:
#   bf_lower = g * exp(-E/kT_rad)  for metastable (w=1, as in dilute)
#   Z = Z_meta(T_rad) + W * Z_non(T_rad)  (dilute partition)
# Then: n_lower = n_ion * bf_lower / Z = n_ion * g * exp(-E/kT_rad) / Z_dilute
# This IS what we compute. So why does it differ?

# Wait — n_ion itself differs! Let me check n_Fe_I TARDIS vs C code
# C code computes n_e = 1.81e9 (vs TARDIS 1.76e9)
# With higher n_e, Fe II→Fe I ratio changes

# Let me recompute Fe I with TARDIS n_e=1.76e9 vs our n_e=1.81e9
print("\n\n" + "=" * 70)
print("Effect of n_e difference on Fe I density")
print("=" * 70)

# Read abundance
with open(f"{REF}/abundances.csv") as f:
    f.readline()
    for line in f:
        parts = line.strip().split(',')
        if int(parts[0]) == 26:
            fe_abund = float(parts[1])
            break

rho_csv = np.genfromtxt(f"{REF}/density.csv", delimiter=',', names=True)
rho = rho_csv['rho'][s]
masses_csv = np.genfromtxt(f"{REF}/atom_masses.csv", delimiter=',', names=True)
fe_mass = masses_csv[masses_csv['atomic_number'] == 26]['mass_amu'][0]

n_fe_total = (fe_abund * rho) / (fe_mass * AMU)
print(f"Fe total: {n_fe_total:.6e}")

T_e = 0.9 * T_rad
g_electron = (2.0 * M_PI * M_E * K_B * T_rad / (H_PLANCK * H_PLANCK)) ** 1.5
beta_e = 1.0 / (K_B * T_e)

# Ioniz energies for Fe
chi_fe0 = ioniz[(ioniz['atomic_number'].astype(int) == 26) & (ioniz['ion_number'].astype(int) == 0)]['ionization_energy_eV'][0]
chi_fe1 = ioniz[(ioniz['atomic_number'].astype(int) == 26) & (ioniz['ion_number'].astype(int) == 1)]['ionization_energy_eV'][0]

print(f"Chi(Fe I→II) = {chi_fe0:.4f} eV")
print(f"Chi(Fe II→III) = {chi_fe1:.4f} eV")

# LTE partition functions at T_rad
def lte_part(z, ion):
    lvls = get_levels_for(z, ion)
    Z = 0.0
    for l in lvls:
        E = l['energy_eV'] * EV_TO_ERG
        g = int(l['g'])
        boltz = E / (K_B * T_rad)
        if boltz < 500:
            Z += g * np.exp(-boltz)
    return max(Z, 1e-300)

Z_fe0_lte = lte_part(26, 0)
Z_fe1_lte = lte_part(26, 1)
Z_fe2_lte = lte_part(26, 2)

print(f"Z_Fe_I(LTE) = {Z_fe0_lte:.6f}")
print(f"Z_Fe_II(LTE) = {Z_fe1_lte:.6f}")

# Zeta factors
zeta_ions_csv = np.genfromtxt(f"{REF}/zeta_ions.csv", delimiter=',', names=True)
zeta_temps = np.genfromtxt(f"{REF}/zeta_temps.csv", delimiter=',', names=True)['temperature']
zeta_data = np.load(f"{REF}/zeta_data.npy")

def get_zeta(z, ion, T):
    for i in range(len(zeta_ions_csv)):
        if int(zeta_ions_csv['atomic_number'][i]) == z and int(zeta_ions_csv['ion_number'][i]) == ion:
            vals = zeta_data[i, :]
            idx = np.searchsorted(zeta_temps, T) - 1
            idx = max(0, min(idx, len(zeta_temps) - 2))
            frac = (T - zeta_temps[idx]) / (zeta_temps[idx + 1] - zeta_temps[idx])
            return vals[idx] + frac * (vals[idx + 1] - vals[idx])
    return 1.0

zeta_fe0 = get_zeta(26, 0, T_rad)
zeta_fe1 = get_zeta(26, 1, T_rad)

# Compute Fe0→Fe1 ratio for different n_e
for ne_label, ne_val in [("TARDIS n_e", n_e_ref[s]), ("LUMINA n_e", 1.8098e9)]:
    # Fe0→Fe1
    chi_erg = chi_fe0 * EV_TO_ERG
    phi_lte = (Z_fe1_lte / Z_fe0_lte) * 2.0 * g_electron * np.exp(-chi_erg * beta_rad)
    delta = (T_e / T_rad) * np.exp(chi_erg * (beta_rad - beta_e))
    phi_neb = phi_lte * W * (zeta_fe0 * delta + W * (1.0 - zeta_fe0)) * np.sqrt(T_e / T_rad)
    ratio_01 = phi_neb / ne_val

    # Fe1→Fe2
    chi_erg = chi_fe1 * EV_TO_ERG
    phi_lte2 = (Z_fe2_lte / Z_fe1_lte) * 2.0 * g_electron * np.exp(-chi_erg * beta_rad)
    delta2 = (T_e / T_rad) * np.exp(chi_erg * (beta_rad - beta_e))
    phi_neb2 = phi_lte2 * W * (zeta_fe1 * delta2 + W * (1.0 - zeta_fe1)) * np.sqrt(T_e / T_rad)
    ratio_12 = phi_neb2 / ne_val

    # n_Fe0
    sum_norm = 1.0 + ratio_01 + ratio_01 * ratio_12
    n_fe0 = n_fe_total / sum_norm
    n_fe1 = n_fe0 * ratio_01
    n_fe2 = n_fe1 * ratio_12

    print(f"\n  {ne_label} = {ne_val:.6e}:")
    print(f"    ratio_01 = {ratio_01:.6e}, ratio_12 = {ratio_12:.6e}")
    print(f"    n_Fe0 = {n_fe0:.6e}")
    print(f"    n_Fe1 = {n_fe1:.6e}")
    print(f"    n_Fe2 = {n_fe2:.6e}")
    print(f"    TARDIS n_Fe0 = {n_fe0_tardis:.6e}")
    print(f"    ratio n_Fe0/TARDIS = {n_fe0/n_fe0_tardis:.6f}")

print(f"\n  TARDIS Fe0 = {n_fe0_tardis:.6e} but computed Fe0 = {n_fe0:.6e}")
print(f"  n_Fe0 error explains tau ratio: n_Fe0_ours/n_Fe0_TARDIS = {n_fe0/n_fe0_tardis:.4f}")
print(f"  Compare with C code tau ratio for Fe I: 2.13")
print(f"  Conclusion: n_Fe0 error is {n_fe0/n_fe0_tardis:.4f}, NOT 2.13")
print(f"  So the 2x factor must come from the partition function OR level pop formula")

print("\n\n" + "=" * 70)
print("FINAL: Compute Fe I tau with TARDIS n_ion directly")
print("=" * 70)

# Use TARDIS n_Fe0 directly, plus our partition function
# tau = SOBOLEV * f * lam * t_exp * n_lower * stim
# n_lower = n_Fe0 * (w_lo * g_lo * exp(-E_lo/kTrad)) / Z_dilute

# The question: does TARDIS divide by Z_dilute or Z_lte?
# If Z_dilute: n_lo = n_Fe0 * 1 * g * exp(-E/kTrad) / (Z_meta(Trad) + W*Z_non(Trad))
# If Z_lte:    n_lo = n_Fe0 * 1 * g * exp(-E/kTrad) / Z_all(Trad)

idx = sig_fe0[0]
lev_lo = int(line_list['level_number_lower'][idx])
lo = lvls[lvls['level_number'].astype(int) == lev_lo]
E_lo = lo['energy_eV'][0] * EV_TO_ERG
g_lo = int(lo['g'][0])
meta_lo = int(lo['metastable'][0])
f_lu = line_list['f_lu'][idx]
lam = line_list['wavelength_cm'][idx]

print(f"Fe I line {idx}: lev_lo={lev_lo}, meta={meta_lo}, E_lo={lo['energy_eV'][0]:.4f} eV")

# Dilute partition (our formula)
Z_dilute_fe0 = Z_meta + W * Z_non  # already computed above
# LTE partition
Z_lte_fe0 = Z_lte_rad  # already computed above

# Boltzmann factor for lower level (metastable, w=1)
bf_lo = g_lo * np.exp(-E_lo * beta_rad)

n_lo_dilute = n_fe0_tardis * 1.0 * bf_lo / Z_dilute_fe0
n_lo_lte = n_fe0_tardis * 1.0 * bf_lo / Z_lte_fe0

tau_dilute = SOBOLEV_COEFF * f_lu * lam * t_exp * n_lo_dilute
tau_lte = SOBOLEV_COEFF * f_lu * lam * t_exp * n_lo_lte
tau_tardis = tau_ref[idx, 0]

print(f"\n  Z_dilute = {Z_dilute_fe0:.6f}")
print(f"  Z_lte    = {Z_lte_fe0:.6f}")
print(f"  Z_dilute / Z_lte = {Z_dilute_fe0/Z_lte_fe0:.6f}")
print(f"  n_lower (dilute Z) = {n_lo_dilute:.6e}")
print(f"  n_lower (LTE Z)    = {n_lo_lte:.6e}")
print(f"  tau (dilute Z) = {tau_dilute:.6e}")
print(f"  tau (LTE Z)    = {tau_lte:.6e}")
print(f"  tau TARDIS     = {tau_tardis:.6e}")
print(f"  ratio dilute   = {tau_dilute/tau_tardis:.6f}")
print(f"  ratio LTE      = {tau_lte/tau_tardis:.6f}")

# Hmm, even with TARDIS n_Fe0 directly, the ratio is ~2.1x.
# The only explanation: TARDIS stores n_Fe0 * g * exp(-E/kT) / Z differently
# OR: TARDIS tau uses a DIFFERENT n_lower formula

# Let me back-calculate what TARDIS n_lower must be:
n_lo_tardis = tau_tardis / (SOBOLEV_COEFF * f_lu * lam * t_exp)
print(f"\n  n_lower implied by TARDIS tau (no stim): {n_lo_tardis:.6e}")
print(f"  n_lower (dilute) / implied = {n_lo_dilute / n_lo_tardis:.6f}")

# What Z_part would give n_lo = n_lo_tardis?
Z_needed = n_fe0_tardis * bf_lo / n_lo_tardis
print(f"  Z_partition needed: {Z_needed:.6f}")
print(f"  This is {Z_needed/Z_dilute_fe0:.6f}x our Z_dilute")
print(f"  This is {Z_needed/Z_lte_fe0:.6f}x Z_lte")

# What if they DON'T divide by partition function at all for tau?
# n_lower = n_ion_density (from Saha) * fraction_in_level?
# Or: what if n_lower is stored differently in TARDIS?
print("\nDone.")
