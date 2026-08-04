#!/usr/bin/env python3
"""Per-(Z,ion) Sobolev τ attribution at Si II 6355 trough window.

EXACT LUMINA-nebular physics (no approximation):

  Z_part(Z,ion,s) = Σ_meta g·exp(-E/kT_rad) + W·Σ_non-meta g·exp(-E/kT_rad)
  weight_l        = 1.0 if level is metastable else W
  n_lower(l,s)    = n_ion(Z,ion,s) · weight_l · g_lo · exp(-E_lo/kT_rad) / Z_part
  τ_Sob(l,s)      = (π e² / m_e c) · f_lu · λ_rest · t_exp · n_lower

Source: src/lumina_plasma.c:282-313 (Z_part), :682-694 (n_lower).

Schema note: this LUMINA run uses TARDIS-schema ion populations via symlink:
  ion_number_density.npy → tardis_reference_strat6_higherL_aulboost/...
TARDIS schema = 8 elements [C,O,Si,S,Ca,Fe,Co,Ni], 153 ion-pops total.
LUMINA atom_masses.csv expands to 15 (adds Mg, Al, Sc, Ti, V, Cr, Mn).
For the Si II 6355 trough window the dominant ions are Si II/III/IV, Fe II/III,
Co II/III, Ni II/III, O I/II, Ca I/II — all present in the TARDIS schema.
Lines whose (Z,ion) is not in the 8-element schema are skipped (no fake
contribution).

bbfix vs asE share the SAME ion_number_density.npy (identical symlink target).
The attribution difference comes ENTIRELY from (a) added Fe III/Co III bb lines
in line_list.csv and (b) added high-E_lower levels in levels.csv → Z_part shift.
That is the precise question the user asked: which ions carry the τ in bbfix?
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
RUNS = {
    "bbfix(158558)": ROOT/"logs/paperDDC15v3asEfloor_bbfix_2002bo_vi9019_L1p0_prod_158558",
    "asE(157921)":   ROOT/"logs/paperDDC15v3asEfloorEPS0p0_2002bo_vi9019_L1p0_prod_157921",
}

LAM_LO, LAM_HI = 6100., 6650.  # rest-frame lines that absorb in the Si II 6355
                                # OBSERVED trough [6050,6300] at v ∈ [3,15]e3 km/s.
                                # λ_L = λ_obs / (1 − v/c) gives this range.

# CGS constants (matches src/lumina_plasma.c)
EC = 4.8032068e-10        # statcoulomb
ME = 9.1093837e-28        # g
C_CGS = 2.99792458e10     # cm/s
KB_ERG = 1.380649e-16     # erg/K
EV_ERG = 1.60218e-12      # eV → erg
SIGMA_PRE = np.pi * EC**2 / (ME * C_CGS)

# TARDIS ion_number_density.npy schema — ordered list (verified by inspection)
TARDIS_ELEMENTS = [6, 8, 14, 16, 20, 26, 27, 28]

ELEM = {1:"H",2:"He",6:"C",7:"N",8:"O",10:"Ne",11:"Na",12:"Mg",13:"Al",14:"Si",
        16:"S",18:"Ar",19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",
        26:"Fe",27:"Co",28:"Ni"}
ROMAN = {0:"I",1:"II",2:"III",3:"IV",4:"V",5:"VI",6:"VII",7:"VIII",8:"IX"}


def build_tardis_ion_index(ref):
    """Reconstruct the TARDIS ion_number_density.npy ordering.

    For each Z in TARDIS_ELEMENTS (sorted), n_ioniz = #rows with that Z in
    ionization_energies.csv, then ion stages = 0..n_ioniz (n_ioniz+1 total).
    Indexed flat in sorted Z order.

    Verified to match the 153-row npy shape.
    """
    ion_e = pd.read_csv(ref/"ionization_energies.csv")
    idx = {}
    ip = 0
    for Z in TARDIS_ELEMENTS:
        n_ioniz = int((ion_e.atomic_number == Z).sum())
        for ion in range(n_ioniz + 1):
            idx[(Z, ion)] = ip
            ip += 1
    return idx, ip


def analyze(label, runD):
    ref = runD/"ref"
    ll = pd.read_csv(ref/"line_list.csv")
    lev = pd.read_csv(ref/"levels.csv")
    ps = pd.read_csv(ref/"plasma_state.csv").sort_values("shell_id")
    cfg = json.loads((ref/"config.json").read_text())
    t_exp = cfg["time_explosion_s"]
    n_shells = cfg["n_shells"]

    ip_idx, n_ion_pops = build_tardis_ion_index(ref)
    n_ion_arr = np.load(ref/"ion_number_density.npy")
    assert n_ion_arr.shape == (n_ion_pops, n_shells), \
        f"shape mismatch: npy {n_ion_arr.shape} vs computed ({n_ion_pops},{n_shells})"

    T_rad = ps.T_rad.values
    W = ps.W.values
    beta_eV = 1.0 / (KB_ERG / EV_ERG * T_rad)

    # ---- Build Z_part(Z,ion,s) — LUMINA-exact nebular formula ----
    Z_part = np.zeros((n_ion_pops, n_shells))
    for (Z, ion), g in lev.groupby(["atomic_number", "ion_number"]):
        if (Z, ion) not in ip_idx:
            continue
        ip = ip_idx[(Z, ion)]
        E = g.energy_eV.values
        gw = g.g.values.astype(float)
        meta = g.metastable.values.astype(bool)
        boltz = E[:, None] * beta_eV[None, :]
        boltz = np.clip(boltz, 0, 500.0)
        bf = gw[:, None] * np.exp(-boltz)
        z_meta = bf[meta, :].sum(axis=0) if meta.any() else np.zeros(n_shells)
        z_non = bf[~meta, :].sum(axis=0) if (~meta).any() else np.zeros(n_shells)
        Z_part[ip, :] = z_meta + W * z_non
    Z_part = np.maximum(Z_part, 1e-300)

    # ---- Filter line list to trough window ----
    sel = (ll.wavelength >= LAM_LO) & (ll.wavelength <= LAM_HI)
    lw = ll[sel].reset_index(drop=True)
    n_in_window = len(lw)
    n_outside_schema = sum(1 for z, i in zip(lw.atomic_number, lw.ion_number)
                           if int(z) not in TARDIS_ELEMENTS)

    # Lookup E_lower, g_lower, meta_lower per line (drop lines whose ion is
    # not in TARDIS schema — they have no n_ion entry)
    lev_idx = lev.set_index(["atomic_number", "ion_number", "level_number"])
    keys = list(zip(lw.atomic_number.astype(int),
                    lw.ion_number.astype(int),
                    lw.level_number_lower.astype(int)))
    have_level = np.array([k in lev_idx.index for k in keys])
    have_ion = np.array([(int(z), int(i)) in ip_idx
                         for z, i in zip(lw.atomic_number, lw.ion_number)])
    keep = have_level & have_ion
    lw_v = lw[keep].reset_index(drop=True)
    keys_v = [k for k, kp in zip(keys, keep) if kp]
    rows = lev_idx.loc[keys_v]
    E_lo = rows.energy_eV.values
    g_lo = rows.g.values.astype(float)
    meta_lo = rows.metastable.values.astype(bool)
    Z_arr = lw_v.atomic_number.values.astype(int)
    ion_arr = lw_v.ion_number.values.astype(int)
    ip_arr = np.array([ip_idx[(int(z), int(i))]
                       for z, i in zip(Z_arr, ion_arr)])

    print(f"\n=== {label} ===")
    print(f"  Lines in [{LAM_LO:.0f},{LAM_HI:.0f}] Å: {n_in_window:,}")
    print(f"  Outside 8-element TARDIS schema (dropped): {n_outside_schema:,}")
    print(f"  No level match (dropped): {int((~have_level & have_ion).sum()):,}")
    print(f"  Valid lines for τ attribution: {len(lw_v):,}")

    # ---- τ_Sob per line per shell ----
    lam_cm = lw_v.wavelength.values * 1e-8
    f_lu = lw_v.f_lu.values
    n_ion_per_line = n_ion_arr[ip_arr, :]
    Z_part_per_line = Z_part[ip_arr, :]
    boltz_l = E_lo[:, None] * beta_eV[None, :]
    boltz_l = np.clip(boltz_l, 0, 500.0)
    weight = np.where(meta_lo[:, None], 1.0, W[None, :])
    n_lower = n_ion_per_line * weight * g_lo[:, None] * np.exp(-boltz_l) / Z_part_per_line
    tau_sob = SIGMA_PRE * f_lu[:, None] * lam_cm[:, None] * t_exp * n_lower
    tau_max_per_line = tau_sob.max(axis=1)

    df = pd.DataFrame(dict(Z=Z_arr, ion=ion_arr,
                           lam=lw_v.wavelength.values, flu=f_lu,
                           E_lo=E_lo, tau_max=tau_max_per_line))
    grp = df.groupby(["Z", "ion"]).agg(
        n_lines=("lam", "count"),
        n_thick=("tau_max", lambda x: int((x > 1.0).sum())),
        n_modthick=("tau_max", lambda x: int((x > 0.1).sum())),
        tau_max_sum=("tau_max", "sum"),
        tau_max_top=("tau_max", "max"),
    ).sort_values("tau_max_sum", ascending=False)
    # Strongest line in each group
    top_line = df.loc[df.groupby(["Z", "ion"])["tau_max"].idxmax(),
                      ["Z", "ion", "lam", "flu", "E_lo"]]
    top_line = top_line.set_index(["Z", "ion"])
    grp = grp.join(top_line.rename(columns={"lam": "lam_top",
                                            "flu": "flu_top",
                                            "E_lo": "E_top"}))
    total = grp.tau_max_sum.sum()
    grp["frac"] = 100. * grp.tau_max_sum / max(total, 1e-300)

    print(f"\n  Top-15 by Σ shell-max τ_Sob:")
    print(f"  {'ion':10s}  {'n_lines':>7s}  {'n_thk':>5s}  {'n_mthk':>6s}  "
          f"{'frac%':>6s}  {'τ_top':>9s}  {'λ_top':>7s}  {'f_lu':>9s}  {'E_lo[eV]':>9s}")
    for (Z, ion), row in grp.head(15).iterrows():
        name = f"{ELEM.get(Z, f'Z{Z}')} {ROMAN.get(ion, str(ion+1))}"
        print(f"  {name:10s}  {int(row.n_lines):7d}  {int(row.n_thick):5d}  "
              f"{int(row.n_modthick):6d}  {row.frac:6.2f}  {row.tau_max_top:9.2e}  "
              f"{row.lam_top:7.1f}  {row.flu_top:9.2e}  {row.E_top:9.3f}")
    return grp


grp_bb = analyze("bbfix(158558)", RUNS["bbfix(158558)"])
grp_ae = analyze("asE(157921)",   RUNS["asE(157921)"])

print(f"\n=== Side-by-side: Σ shell-max τ_Sob per (Z,ion) in [{LAM_LO:.0f},{LAM_HI:.0f}] Å ===")
print(f"  {'ion':10s}  {'asE frac%':>10s}  {'bb frac%':>10s}  {'Δ (bb−asE)':>12s}  "
      f"{'bb/asE Στ':>12s}  {'asE n':>7s}  {'bb n':>7s}  {'bb n_thk':>9s}")
all_keys = sorted(set(grp_bb.index) | set(grp_ae.index),
                  key=lambda k: -max(grp_bb.frac.get(k, 0),
                                     grp_ae.frac.get(k, 0)))
for k in all_keys[:20]:
    Z, ion = k
    name = f"{ELEM.get(Z, f'Z{Z}')} {ROMAN.get(ion, str(ion+1))}"
    fae = grp_ae.frac.get(k, 0.0)
    fbb = grp_bb.frac.get(k, 0.0)
    tae = grp_ae.tau_max_sum.get(k, 0.0)
    tbb = grp_bb.tau_max_sum.get(k, 0.0)
    n_ae = int(grp_ae.n_lines.get(k, 0))
    n_bb = int(grp_bb.n_lines.get(k, 0))
    n_bb_thk = int(grp_bb.n_thick.get(k, 0))
    ratio = (tbb / tae) if tae > 0 else float('inf')
    delta = fbb - fae
    flag = "  ★" if abs(delta) > 1.0 else ""
    if ratio > 5 and tae > 1e-30:
        flag += "  bb-boost"
    print(f"  {name:10s}  {fae:9.3f}%  {fbb:9.3f}%  {delta:+11.3f}%  "
          f"{ratio:12.3e}  {n_ae:6d}  {n_bb:6d}  {n_bb_thk:8d}{flag}")
