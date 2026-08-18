#!/usr/bin/env python3
"""Split-field quantifier for the B-run coevolve_consume_a10_kx_gphall.

Consumer census (established from src/lumina_plasma.c, see VERDICT.md):
  (a) Gph photoionization rate  -> mc_J  (alpha=1.0 blend, plasma.c:5489-5501/5434-5442)
  (b) bf photo-heating  Hex     -> mc_J  (same Gph loop, Hx accumulator 5513/5454)
  (c) line cooling + line PUMP  -> cs_J  (nlte->J_nu via nlte_get_J_at_nu, 5562/5572-73)
  (d) fb / C_fb recomb cooling  -> no J  (rate-based alpha*(chi+kT), 4900-4909)
  (e) NLTE level-pop line Jbar  -> per-line Sobolev field (jbar_line_det / cs_J fallback)
  (f) T_rad / W fit             -> MC estimators (nu_bar_estimator/j_estimator, solve_radiation_field)

This script quantifies, per shell (s0,s1,s2) and per band, how far apart the two
BINNED fields the thermal ledger (cs_J) and the ionization ledger (mc_J) actually
consume are.  Read-only.  No source edits.
"""
import csv, math
import numpy as np

C = 2.99792458e10
REPO = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN = f"{REPO}/logs/coevolve_consume_a10_kx_gphall"
FIELD = f"{RUN}/lumina_coevolve_field.csv"

# committed run state (plasma_state.csv)
STATE = {0: (13119.874754, 4.426076e9, 0.2978587262, 10470.093240),
         1: (13592.133079, 3.615404e9, 0.1878908694, 10470.093240),
         2: (13911.516798, 2.864163e9, 0.1342391605, 10470.093240)}

# load field for s0,s1,s2
lam = {s: {} for s in (0, 1, 2)}
cs = {s: {} for s in (0, 1, 2)}
mc = {s: {} for s in (0, 1, 2)}
for r in csv.DictReader(open(FIELD)):
    s = int(r['shell'])
    if s not in (0, 1, 2):
        continue
    b = int(r['bin'])
    lam[s][b] = float(r['wavelength_A'])
    cs[s][b] = float(r['cs_J'])
    mc[s][b] = float(r['mc_J'])

def arrs(s):
    bs = sorted(lam[s])
    L = np.array([lam[s][b] for b in bs])
    Cs = np.array([cs[s][b] for b in bs])
    Mc = np.array([mc[s][b] for b in bs])
    nu = C / (L * 1e-8)
    o = np.argsort(nu)
    return nu[o], L[o], Cs[o], Mc[o]

# bands: (name, lam_lo, lam_hi, dominant consumer that lives here)
BANDS = [
    ("EUV <912",        0.0,   912.0,  "Gph(mc) + bf-heat(mc)"),
    ("FUV 912-1290",    912.0, 1290.0, "Gph excited-lev(mc) / pump(cs)"),
    ("res-pump 1490-1650", 1490.0, 1650.0, "line PUMP (cs)  [Co IV/Fe-Co II]"),
    ("res-pump 1700-2100", 1700.0, 2100.0, "line PUMP (cs)  [Fe/Co/Ni II forest]"),
    ("NUV 2100-3000",   2100.0, 3000.0, "line cool/pump (cs)"),
    ("opt 3000-7000",   3000.0, 7000.0, "line cool (cs)"),
    ("IR >7000",        7000.0, 1e9,    "far-IR forbidden cool (cs)"),
]

def uband(nu, J, lam_arr, lo, hi):
    m = (lam_arr >= lo) & (lam_arr < hi) & (J > 1e-29)
    if m.sum() < 2:
        return 0.0
    return 4 * math.pi / C * np.trapezoid(J[m], nu[m])

print("# SPLIT-FIELD band table: cs_J (thermal-ledger field) vs mc_J (ionization-ledger field)")
print("# energy density u_band [erg/cm3] integrated over each band; ratio mc/cs")
print("#")
rows = []
for s in (0, 1, 2):
    Te, ne, W, Trad = STATE[s]
    nu, L, Cs, Mc = arrs(s)
    print(f"## shell s{s}: Te={Te:.0f} K  ne={ne:.3e}  W={W:.4f}  T_rad={Trad:.0f}")
    print(f"#  {'band':22s} {'consumer(field)':34s} {'u_cs':>11s} {'u_mc':>11s} {'mc/cs':>8s}")
    for nm, lo, hi, cons in BANDS:
        ucs = uband(nu, Cs, L, lo, hi)
        umc = uband(nu, Mc, L, lo, hi)
        rat = umc / ucs if ucs > 0 else float('nan')
        print(f"#  {nm:22s} {cons:34s} {ucs:11.3e} {umc:11.3e} {rat:8.3f}")
        rows.append((s, Te, nm, cons, ucs, umc, rat))
    print("#")

# probe wavelengths (single-bin lookup, exact like nlte_get_J_at_nu)
PROBES = [(1526.0, "flag: 39x claim"), (1856.0, "top Fe/Co II pump line"),
          (2500.0, "NUV super-thermal check"), (838.0, "EUV ionizing"),
          (404.0, "Fe III / Co III bf threshold"), (6000.0, "optical")]
print("# PROBE wavelengths (nearest-bin cs_J vs mc_J), s0 only")
print(f"#  {'lam_A':>8s} {'note':28s} {'cs_J':>11s} {'mc_J':>11s} {'mc/cs':>9s} {'cs/mc':>9s}")
nu0, L0, Cs0, Mc0 = arrs(0)
for lp, note in PROBES:
    i = int(np.argmin(np.abs(L0 - lp)))
    r1 = Mc0[i] / Cs0[i] if Cs0[i] > 0 else float('nan')
    r2 = Cs0[i] / Mc0[i] if Mc0[i] > 0 else float('nan')
    print(f"#  {L0[i]:8.1f} {note:28s} {Cs0[i]:11.3e} {Mc0[i]:11.3e} {r1:9.4f} {r2:9.1f}")

# write CSV
with open(f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/splitfield_audit/consumer_table.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["shell", "Te_K", "band", "dominant_consumer(field_read)",
                "u_cs_J", "u_mc_J", "ratio_mc_over_cs"])
    for row in rows:
        w.writerow([row[0], f"{row[1]:.0f}", row[2], row[3],
                    f"{row[4]:.4e}", f"{row[5]:.4e}", f"{row[6]:.4f}"])
print("#\n# wrote consumer_table.csv")
