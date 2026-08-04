#!/usr/bin/env python3
"""Offline single-cell macro-atom calculator: apple-to-apple of the excitation-rate root
(Divergence 1). Frozen cell + atomic data -> compares Lumina's up-field (B_lu*J-bar, which
thermalizes to B(T_e) in thick UV lines) vs ARTIS's transported field, WITHOUT a slurm run.

PHASE 1 (this): is the transported binned field cs.J super-thermal (> B(T_e)) at the UV
Fe/S lines? That is exactly the quantity that decides whether switching the up-rate from the
thermal J-bar to the transported field lifts b_k above 1 (=fluorescence). Runs in seconds.

Usage: offline_macroatom_calc.py <field_csv> <plasma_csv> [shell]
"""
import sys, csv
import numpy as np

H = 6.62607015e-27      # erg s
KB = 1.380649e-16       # erg/K
C = 2.99792458e10       # cm/s

field_csv = sys.argv[1] if len(sys.argv) > 1 else "logs/coevolve_s01/lumina_coevolve_field.csv"
plasma_csv = sys.argv[2] if len(sys.argv) > 2 else "logs/stage1_toy06_epay27/lumina_plasma_state.csv"
shell = int(sys.argv[3]) if len(sys.argv) > 3 else 5

def planck_nu(nu, T):
    x = H * nu / (KB * T)
    x = np.clip(x, 1e-10, 700)
    return (2 * H * nu**3 / C**2) / (np.expm1(x))

# --- cell state ---
pl = {int(r['shell_id']): r for r in csv.DictReader(open(plasma_csv))}
Te = float(pl[shell]['T_e']); Tr = float(pl[shell]['T_rad']); W = float(pl[shell]['W'])
ne = float(pl[shell]['n_e'])
print(f"=== OFFLINE MACRO-ATOM CALC: shell {shell}  T_e={Te:.0f}K  T_rad={Tr:.0f}K  W={W:.4f}  n_e={ne:.2e} ===")

# --- binned transported field cs.J for this shell: lam[A] -> J ---
lam=[]; J=[]
for r in csv.DictReader(open(field_csv)):
    if int(r['shell']) != shell: continue
    lam.append(float(r['wavelength_A'])); J.append(float(r['cs_J']))
lam = np.array(lam); J = np.array(J)
o = np.argsort(lam); lam, J = lam[o], J[o]
nu_grid = C / (lam * 1e-8)

def transported_J(nu):
    return np.interp(nu, nu_grid[::-1], J[::-1])  # nu ascending

# --- the ROOT quantity, per wavelength band: transported J vs B(T_e) and W*B(T_rad) ---
print(f"\n{'band[A]':14} {'<J_transp>':>11} {'<B(T_e)>':>11} {'J/B(Te)':>9} {'W*B(Tr)/B(Te)':>13}")
bands = [("UV 2000-2800",2000,2800),("UV 2800-3500",2800,3500),("blue 3500-5000",3500,5000),
         ("grn 5000-6500",5000,6500),("red 6500-9000",6500,9000)]
for nm,a,b in bands:
    m = (lam>=a)&(lam<b)
    if m.sum()==0: continue
    nu = C/(0.5*(a+b)*1e-8)
    Jt = J[m].mean()
    Bte = planck_nu(nu, Te)
    WBtr = W*planck_nu(nu, Tr)
    print(f"{nm:14} {Jt:11.3e} {Bte:11.3e} {Jt/Bte:9.3f} {WBtr/Bte:13.3f}")

print("\n--- ROOT VERDICT (Divergence 1) ---")
# UV super-thermality: is the transported field > B(T_e) where Lumina's J-bar thermalizes?
muv = (lam>=2000)&(lam<3500)
nu_uv = C/(2650e-8)
ratio_uv = J[muv].mean()/planck_nu(nu_uv, Te)
print(f"transported J / B(T_e) in the UV (2000-3500A) = {ratio_uv:.3f}")
if ratio_uv > 1.15:
    print("  => transported field is SUPER-THERMAL in the UV: switching the up-rate from")
    print("     the thermal J-bar (->B(T_e)) to this field will drive b_k>1 => FLUORESCENCE.")
    print("     Divergence-1 fix is validated at the field level (offline).")
elif ratio_uv > 0.85:
    print("  => transported ~ thermal in UV: field switch alone won't lift b_k. Injection/")
    print("     deep-field (Stage-2) needed to make the UV field super-thermal first.")
else:
    print("  => transported field is UV-POOR: injection-limited, not a rate-form issue.")

# W*B(T_rad) proxy check (what IUP_TRAD dilute-BB would give)
print(f"W*B(T_rad)/B(T_e) in UV = {W*planck_nu(nu_uv,Tr)/planck_nu(nu_uv,Te):.3f}  "
      f"(this is what IUP_TRAD feeds; W-crush kills it => why that knob failed)")
