#!/usr/bin/env python3
"""
Probe A — LTE consistency gate for LUMINA's NLTE bound-free ionization rates.

Question: at LTE (J_nu = B_nu(T_e), T_e = T_rad, W = 1) does the rate-balance
ionization the code computes reproduce the analytic LTE Saha ratio?

Key algebra: in the bf channel the code sets
    R_rec_i = R_bf_i * nstar_i           (plasma.c:3337, Milne detailed balance)
    nstar_i = n_e * lambda^3 * g_i / (2 g_hi) * exp((chi - E_i)/kT_e)   (plasma.c:3318-3334)
so in steady state n_i = n_hi * nstar_i and R_bf CANCELS. Hence the LTE ion ratio
depends ONLY on n_star_ratio summed over levels — no sigma_bf, no field needed.

    (n_lo/n_hi)_code = sum_i nstar_i = n_e lambda^3/(2 g_hi) * exp(chi/kT_e) * U_lo(T_e)

Full LTE Saha (correct):
    (n_lo/n_hi)_saha = n_e lambda^3 * U_lo/(2 U_hi) * exp(chi/kT_e)

So the code reproduces Saha ONLY IF U_hi == g_hi, i.e. the upper ion is treated as
its ground LEVEL only. Discrepancy factor = U_hi(T_e) / g_hi.

This probe quantifies that factor for the iron-peak pairs and reports the resulting
ion fractions so we can see whether the bf rates alone over- or under-ionize at LTE.
"""
import sys, math, csv, os

# --- constants verbatim from src/lumina.h ---
H_PLANCK   = 6.62607015e-27   # erg*s
K_BOLTZMANN= 1.380649e-16     # erg/K
M_ELECTRON = 9.1093837015e-28 # g
EV_TO_ERG  = 1.602176634e-12

REF = sys.argv[1] if len(sys.argv) > 1 else \
    "data/tardis_reference_cmfgen_superlev_ionfix_ddc15strat"

def load_levels(path):
    """returns dict[(Z, ion0)] -> list of (E_eV, g)"""
    d = {}
    with open(os.path.join(path, "levels.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            Z = int(row["atomic_number"]); ion0 = int(row["ion_number"])
            E = float(row["energy_eV"]);   g = int(float(row["g"]))
            d.setdefault((Z, ion0), []).append((E, g))
    return d

def load_chi(path):
    """ion_number n -> energy to ionize stage n (0-indexed: 0=Fe I->II)"""
    d = {}
    with open(os.path.join(path, "ionization_energies.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            d[(int(row["atomic_number"]), int(row["ion_number"]))] = \
                float(row["ionization_energy_eV"])
    return d

def partition(levels, T_e, e_cut_eV=None):
    """U = sum g exp(-E/kT_e); optionally only levels with E < e_cut (bound)."""
    kT = K_BOLTZMANN * T_e / EV_TO_ERG  # in eV
    U = 0.0
    for E, g in levels:
        if e_cut_eV is not None and E >= e_cut_eV:
            continue
        U += g * math.exp(-E / kT)
    return U

def thermal_debroglie3(T_e):
    return (H_PLANCK*H_PLANCK / (2.0*math.pi*M_ELECTRON*K_BOLTZMANN*T_e))**1.5

def code_ratio_lo_over_hi(lower, g_hi, chi_eV, T_e, n_e):
    """sum_i nstar_i, verbatim per-level Milne form (plasma.c:3318-3334).
    Only levels with chi - E_i > 0 contribute (nu_thresh>0, line 3289)."""
    lam3 = thermal_debroglie3(T_e)
    kT_erg = K_BOLTZMANN * T_e
    chi_erg = chi_eV * EV_TO_ERG
    s = 0.0
    for E_eV, g in lower:
        chi_lev = chi_erg - E_eV*EV_TO_ERG
        if chi_lev <= 0.0:
            continue
        nstar = n_e * lam3 * g / (2.0*g_hi) * math.exp(chi_lev/kT_erg)
        if nstar > 1e30: nstar = 1e30
        s += nstar
    return s

def saha_ratio_lo_over_hi(U_lo, U_hi, chi_eV, T_e, n_e):
    lam3 = thermal_debroglie3(T_e)
    chi_erg = chi_eV * EV_TO_ERG
    return n_e * lam3 * U_lo/(2.0*U_hi) * math.exp(chi_erg/(K_BOLTZMANN*T_e))

PAIRS = [  # (Z, lower_ion0, upper_ion0, label)
    (26, 1, 2, "Fe II/III"),
    (14, 1, 2, "Si II/III"),
    (27, 1, 2, "Co II/III"),
    (28, 1, 2, "Ni II/III"),
    (20, 1, 2, "Ca II/III"),
]

def frac_upper(r_lo_hi):
    return 1.0/(1.0 + r_lo_hi)

def main():
    levels = load_levels(REF); chi = load_chi(REF)
    print(f"# Probe A  ref={REF}")
    for (T_e, n_e) in [(8000.0, 1e8), (10000.0, 1e9), (6000.0, 1e7)]:
        print(f"\n=== T_e={T_e:.0f} K   n_e={n_e:.0e} cm^-3 ===")
        print(f"{'pair':<11} {'chi_eV':>7} {'U_hi/g_hi':>9} "
              f"{'fIII_code':>10} {'fIII_saha':>10} {'code/saha r':>12}")
        for (Z, lo0, hi0, label) in PAIRS:
            lo = levels.get((Z, lo0)); hi = levels.get((Z, hi0))
            if not lo or not hi:
                print(f"{label:<11}  (missing data)"); continue
            chi_eV = chi[(Z, lo0)]      # ionize lower stage
            g_hi = hi[0][1]             # upper-ion ground level g
            U_lo = partition(lo, T_e, e_cut_eV=chi_eV)
            U_hi = partition(hi, T_e)
            r_code = code_ratio_lo_over_hi(lo, g_hi, chi_eV, T_e, n_e)
            r_saha = saha_ratio_lo_over_hi(U_lo, U_hi, chi_eV, T_e, n_e)
            print(f"{label:<11} {chi_eV:7.3f} {U_hi/g_hi:9.3f} "
                  f"{frac_upper(r_code):10.4f} {frac_upper(r_saha):10.4f} "
                  f"{(r_code/r_saha if r_saha>0 else float('nan')):12.4f}")
        print("  fIII = n_upper/(n_lo+n_upper).  code/saha r = (n_lo/n_hi)_code / _saha"
              "  (should be U_hi/g_hi if formula is otherwise exact)")

if __name__ == "__main__":
    main()
