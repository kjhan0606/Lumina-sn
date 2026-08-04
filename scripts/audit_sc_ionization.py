#!/usr/bin/env python3
"""Path-A audit: independent Sc ionization (Saha + TARDIS-nebular) vs LUMINA.

Reads CMFGEN SCAN I/II/III osc_data (g, E, ionization energy), builds
partition functions U(T), and computes Sc II<->III (and I<->II) ionization
balance at shell-0 conditions, comparing to LUMINA's reported ~100% Sc II.

No fit knobs; pure physics cross-check using CMFGEN's own atomic data.
"""
import numpy as np

# --- physical constants (cgs) ---
k_B   = 1.380649e-16      # erg/K
h     = 6.62607015e-27    # erg s
m_e   = 9.1093837e-28     # g
hc_k  = 1.4387768775e0    # cm*K  (hc/k_B, for E in cm^-1 -> use E_cm*hc_k/T)
eV_cm = 8065.544          # cm^-1 per eV

CMF = "/gpfs/kjhan/cmfgen_21jun23/atomic/SCAN"
ION_FILES = {1:"I/19apr23/osc_data", 2:"II/19apr23/osc_data", 3:"III/19apr23/osc_data"}

def parse_osc(path):
    """Return (ionization_energy_cm, list[(g, E_cm)])."""
    ion_E = None; nlev = None; levels = []
    with open(path, errors="replace") as fh:
        lines = fh.readlines()
    # header scalars
    for ln in lines:
        if "!Number of energy levels" in ln: nlev = int(ln.split()[0])
        if "!Ionization energy" in ln:       ion_E = float(ln.split()[0])
        if "!Number of transitions" in ln:   break
    # level block: lines after the transitions-count header, fixed columns
    # format: name  g  E(cm^-1)  10^15Hz  eV  Lam(A)  ID ...
    started = False; count = 0
    for ln in lines:
        s = ln.rstrip("\n")
        if "!Number of transitions" in ln: started = True; continue
        if not started: continue
        if not s.strip(): continue
        parts = s.split()
        # need: ... g E ... ; name may contain no spaces (underscored) -> parts[0]=name
        try:
            g = float(parts[1]); E = float(parts[2])
        except (IndexError, ValueError):
            continue
        levels.append((g, E)); count += 1
        if nlev and count >= nlev: break
    return ion_E, levels

def U(levels, T):
    g = np.array([l[0] for l in levels]); E = np.array([l[1] for l in levels])
    return float(np.sum(g * np.exp(-E * hc_k / T)))

def saha_ratio(U_lo, U_hi, chi_cm, T, n_e):
    """n_hi/n_lo for LTE Saha at temperature T, electron density n_e."""
    fac = (2.0*np.pi*m_e*k_B*T/h**2)**1.5
    return (2.0 * U_hi/U_lo / n_e) * fac * np.exp(-chi_cm * hc_k / T)

def main():
    data = {}
    for z, rel in ION_FILES.items():
        ionE, lev = parse_osc(f"{CMF}/{rel}")
        data[z] = dict(ionE=ionE, lev=lev, n=len(lev))
        print(f"Sc {['','I','II','III'][z]:>3}: nlev={len(lev):5d}  ionE={ionE:.1f} cm^-1 = {ionE/eV_cm:.3f} eV")
    print()

    # shell-0 conditions (job 161633, super ref, einsteinFix-floor X_Sc=1e-5)
    T_rad = 8942.6
    W     = 0.8842
    n_e   = 5.206e9          # init estimate
    n_Sc  = 1.62e4           # total Sc number density (X_Sc*rho/(45 m_H))
    chi_I_II  = data[1]['ionE']   # Sc I->II
    chi_II_III= data[2]['ionE']   # Sc II->III

    print(f"shell-0: T_rad={T_rad:.0f}K  W={W:.3f}  n_e={n_e:.2e}  n_Sc={n_Sc:.2e}")
    print(f"LUMINA reports: Sc ~100% Sc II (n_ScII={1.632e4:.3e} ~= n_Sc)\n")

    for label, T_e, neb in [
        ("LTE Saha @ T_e=0.9*T_rad", 0.9*T_rad, False),
        ("LTE Saha @ T_e=T_rad",     T_rad,     False),
        ("LTE Saha @ T_e=12000",     12000.,    False),
        ("Nebular  @ T_rad (W,Te=0.9Tr)", 0.9*T_rad, True),
    ]:
        U1 = U(data[1]['lev'], T_e); U2 = U(data[2]['lev'], T_e); U3 = U(data[3]['lev'], T_e)
        if not neb:
            r12 = saha_ratio(U1, U2, chi_I_II,   T_e, n_e)   # II/I
            r23 = saha_ratio(U2, U3, chi_II_III, T_e, n_e)   # III/II
        else:
            # TARDIS nebular: Phi_neb = W*(zeta + W(1-zeta))*(Te/Trad)^0.5 * Phi_LTE(Trad)
            # bound with zeta in {1 (all-to-ground), 0.1}; use Trad in Saha, scale.
            U1r=U(data[1]['lev'],T_rad);U2r=U(data[2]['lev'],T_rad);U3r=U(data[3]['lev'],T_rad)
            base12 = saha_ratio(U1r,U2r,chi_I_II,T_rad,n_e)
            base23 = saha_ratio(U2r,U3r,chi_II_III,T_rad,n_e)
            zeta=1.0
            scale = W*(zeta + W*(1-zeta))*(T_e/T_rad)**0.5
            r12 = base12*scale; r23 = base23*scale
        # solve fractions: n_I:n_II:n_III = 1 : r12 : r12*r23
        a = 1.0; b = r12; c = r12*r23
        tot = a+b+c
        fI,fII,fIII = a/tot, b/tot, c/tot
        print(f"[{label}]  T_e={T_e:.0f}")
        print(f"    II/I={r12:.3e}  III/II={r23:.3e}")
        print(f"    f(ScI)={fI:.3e}  f(ScII)={fII:.3e}  f(ScIII)={fIII:.3e}")
        print()

if __name__ == "__main__":
    main()
