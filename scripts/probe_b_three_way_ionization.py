#!/usr/bin/env python3
"""
Probe B — three-way ionization discriminator on a converged LUMINA run.

For each iron-peak II/III pair at selected shells, compute the III fraction THREE ways:
  (1) NLTE rate solve   : sum of nlte_level_populations over levels (column n_pop)
  (2) nebular phi_neb    : the dilute-Saha ion stage (column n_ion_total) that feeds OPACITY
  (3) LTE Saha @ T_e     : phi_neb / A, where A is the partition-function-independent
                           amplification factor phi_neb / Saha@Te.

Key algebra (from compute_ion_populations, plasma.c:454-469):
  ratio_neb = n(III)/n(II) = phi_neb / n_e,
  phi_neb = W*sqrt(Te/Tr)*( zeta*(Te/Tr)*phi_LTE(Te) + W*(1-zeta)*phi_LTE(Tr) )
  phi_LTE(T) = (Z_next/Z_cur)*2*g_e(Tr)*exp(-chi*beta_T)   [g_e uses Tr in BOTH terms]
The clean LTE Saha at T_e is phi_Saha(Te) = (Z_next/Z_cur)*2*g_e(Te)*exp(-chi*beta_e).
Dividing, the (Z_next/Z_cur)*2 prefactor CANCELS and g_e(Tr)/g_e(Te) = (Tr/Te)^1.5, giving
  A = phi_neb/phi_Saha(Te) = W*zeta + W^2*(1-zeta)*(Tr/Te)*exp(chi*(beta_e - beta_rad))
which needs only W, zeta, Tr, Te, chi -- no n_e, no partition functions.
Hence ratio_Saha@Te = ratio_neb / A.

FORK the probe decides:
  - If A >> 1: phi_neb over-ionizes BEYOND Saha@Te -> cure = feed NLTE/Saha@Te ion stage
    into opacity instead of phi_neb (a phi_neb-replacement fix).
  - If A ~ 1 but fIII still high: T_e/T_rad themselves too hot (T_inner self-pin overshoot)
    -> cure is upstream temperature, not phi_neb.
  - Probe A predicts ratio_NLTE ~ ratio_Saha@Te (NLTE bf pins to Saha@Te). Verify here.
"""
import sys, os, csv, math
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_ddc15strat_scatter_161922"
_ITER = sys.argv[2] if len(sys.argv) > 2 else "004"
DUMP = os.path.join(RUN, f"nlte_levels_iter{_ITER}.csv")
REF  = os.path.join(RUN, "ref")

K_EV = 8.617333262e-5  # Boltzmann in eV/K

PAIRS = [(26, "Fe"), (14, "Si"), (27, "Co"), (28, "Ni"), (20, "Ca"), (16, "S"), (24, "Cr")]
SHELLS = [0, 5, 15, 29]

def load_chi(ref):
    d = {}
    with open(os.path.join(ref, "ionization_energies.csv")) as f:
        for row in csv.DictReader(f):
            d[(int(row["atomic_number"]), int(row["ion_number"]))] = \
                float(row["ionization_energy_eV"])
    return d

def load_zeta(ref):
    Z = []; ion = []
    with open(os.path.join(ref, "zeta_ions.csv")) as f:
        for row in csv.DictReader(f):
            Z.append(int(row["atomic_number"])); ion.append(int(row["ion_number"]))
    temps = []
    with open(os.path.join(ref, "zeta_temps.csv")) as f:
        for row in csv.DictReader(f):
            temps.append(float(row["temperature"]))
    data = np.load(os.path.join(ref, "zeta_data.npy"))
    temps = np.array(temps)
    idx = {(Z[i], ion[i]): i for i in range(len(Z))}
    def zeta(Zel, ion_stage, T):
        if (Zel, ion_stage) not in idx:
            return 1.0
        v = data[idx[(Zel, ion_stage)]]
        if T <= temps[0]:  return float(v[0])
        if T >= temps[-1]: return float(v[-1])
        return float(np.interp(T, temps, v))
    return zeta

def parse_dump(dump):
    # agg[(Z,ion,shell)] = [sum_npop, n_ion_total, Te, Tr, W]
    agg = {}
    with open(dump) as f:
        r = csv.reader(f); next(r)
        for row in r:
            Z = int(row[0]); ion = int(row[1]); s = int(row[2])
            npop = float(row[7]); Te = float(row[8]); Tr = float(row[9])
            W = float(row[10]); nion = float(row[11])
            k = (Z, ion, s)
            if k not in agg:
                agg[k] = [0.0, nion, Te, Tr, W]
            agg[k][0] += npop
    return agg

def amp_factor(W, zeta, Tr, Te, chi_eV):
    beta_e = 1.0/(K_EV*Te); beta_r = 1.0/(K_EV*Tr)
    return W*zeta + W*W*(1.0-zeta)*(Tr/Te)*math.exp(chi_eV*(beta_e - beta_r))

def f_from_ratio(r):
    if not math.isfinite(r): return float('nan')
    return r/(1.0+r)

def main():
    chi = load_chi(REF); zeta_fn = load_zeta(REF)
    agg = parse_dump(DUMP)
    print(f"# Probe B  run={RUN}  dump=iter{_ITER}")
    print(f"# fIII = n(III)/(n(II)+n(III)).  A = phi_neb/Saha@Te amplification.")
    print(f"# (1)NLTE = Sum n_pop ; (2)neb = phi_neb n_ion_total ; (3)Saha@Te = ratio_neb/A")
    for s in SHELLS:
        # representative Te,Tr,W for this shell
        any_k = next((k for k in agg if k[2] == s), None)
        if any_k is None: continue
        Te = agg[any_k][2]; Tr = agg[any_k][3]; W = agg[any_k][4]
        print(f"\n=== shell {s}:  T_e={Te:.0f}  T_rad={Tr:.0f}  T_e/T_rad={Te/Tr:.3f}  W={W:.3f} ===")
        print(f"{'pair':<8}{'chi_eV':>7}{'zeta':>6}{'A':>8}"
              f"{'fIII_NLTE':>10}{'fIII_neb':>9}{'fIII_Saha@Te':>13}")
        for (Z, name) in PAIRS:
            kII  = (Z, 1, s); kIII = (Z, 2, s)
            if kII not in agg or kIII not in agg:
                continue
            nII_nlte, nII_neb  = agg[kII][0],  agg[kII][1]
            nIII_nlte, nIII_neb = agg[kIII][0], agg[kIII][1]
            chi_eV = chi.get((Z, 1), float('nan'))  # ionize II -> III (0-indexed ion=1)
            z = zeta_fn(Z, 1, Tr)
            A = amp_factor(W, z, Tr, Te, chi_eV)
            r_nlte = nIII_nlte/nII_nlte if nII_nlte > 0 else float('inf')
            r_neb  = nIII_neb /nII_neb  if nII_neb  > 0 else float('inf')
            r_saha = r_neb / A if A > 0 else float('nan')
            print(f"{name+' II/III':<8}{chi_eV:7.2f}{z:6.2f}{A:8.2f}"
                  f"{f_from_ratio(r_nlte):10.4f}{f_from_ratio(r_neb):9.4f}"
                  f"{f_from_ratio(r_saha):13.4f}")

if __name__ == "__main__":
    main()
