#!/usr/bin/env python3
"""Build a LUMINA reference that FAITHFULLY reproduces the DDC15 initial epoch
(0.976 d) directly from the public hydro file — the most controlled possible
test: same model, same epoch, compare to DDC15's own emergent spectrum
(DDC15_spec_2500_25500_interp5_000.976d.dat).

The hydro file specifies, on a 115-point grid at 0.976 d: radius, velocity, gas
temperature, density, atom/electron density, Rosseland opacity chi, kappa, and
the mass fractions of 26 elements + 105 isotopes (isotopes resolved for Z=17..28).
Native grid points (>= photosphere) are used directly as shells — no re-gridding.

Faithfulness fixes vs the first cut:
  * ISOTOPE-AWARE composition: Z=17..28 are built by summing the resolved
    isotopes and decaying them t0=0.976d -> target_epoch (default 0.976 = identity,
    so the machinery is exercised but the 0.976d composition is unchanged). This
    is required so OTHER SNe / later epochs can be evolved from their ICs.
  * Ne (and any LUMINA-untracked element) is kept as INERT MASS: tracked mass
    fractions are normalized by the FULL 26-element sum, NOT renormalized to the
    tracked subset. LUMINA has no Ne atomic data (levels/lines/ionization), so Ne
    cannot form lines, but dropping+renorming it would spuriously inflate the
    number densities of the tracked species (n = X*rho/(A*m_u) uses absolute X).
    Free electrons from Ne are already captured by the file's electron density.
  * ATOM-DENSITY cross-check: the re-derived total atom number density
    Sum_Z X_Z*rho/(A_Z*m_u) is compared to the file's own "Atom density" column.

Information NOT in the file, derived here:
  * inner boundary = photosphere: grid point where the Rosseland optical depth
    tau = integral(chi dr) from the outer edge reaches TAU_PHOT (default 2/3).
  * L_inner = 4 pi r_inner^2 sigma T_inner^4 with T_inner the gas T there.

Usage:
  build_ddc15_initial_epoch.py KEEPER_REF OUT_REF [tau_phot] [target_epoch_d]
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HYDRO = Path("data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d")
T0_D = 0.976
SIGMA_SB = 5.670374e-5
AMU = 1.66053906660e-24  # g
LN2 = np.log(2.0)

# LUMINA line-forming elements (must have atomic data in the keeper ref)
Z_LIST = [6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]

NAME2Z = {
    "HYD": 1, "HE": 2, "CARB": 6, "NIT": 7, "OXY": 8, "FLU": 9, "NEON": 10,
    "SOD": 11, "MAG": 12, "ALUM": 13, "SIL": 14, "PHOS": 15, "SUL": 16,
    "CHL": 17, "ARG": 18, "POT": 19, "CAL": 20, "SCAN": 21, "TIT": 22,
    "VAN": 23, "CHRO": 24, "MAN": 25, "IRON": 26, "COB": 27, "NICK": 28,
    "BAR": 56,
}
Z2NAME = {v: k for k, v in NAME2Z.items()}

# atomic masses (amu) for ALL 26 elements present in the file (atom-density check)
A_AMU = {1: 1.008, 2: 4.0026, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
         10: 20.180, 11: 22.990, 12: 24.305, 13: 26.982, 14: 28.085,
         15: 30.974, 16: 32.06, 17: 35.45, 18: 39.948, 19: 39.098,
         20: 40.078, 21: 44.956, 22: 47.867, 23: 50.942, 24: 51.996,
         25: 54.938, 26: 55.845, 27: 58.933, 28: 58.693, 56: 137.33}

# radioactive decay table: (Z_parent, A) -> (Z_daughter, half_life_days)
DECAY = {
    (28, 56): (27, 6.075),     (27, 56): (26, 77.236),
    (28, 57): (27, 1.483),     (27, 57): (26, 271.74),
    (27, 55): (26, 17.53),     (26, 55): (25, 1002.0),
    (26, 52): (25, 0.3448),    (25, 52): (24, 5.591),
    (24, 48): (23, 0.898),     (23, 48): (22, 15.973),
    (22, 44): (21, 21915.0),   (21, 44): (20, 0.1654),
    (25, 51): (24, 0.0321),    (24, 51): (23, 27.70),
    (24, 49): (23, 0.0294),    (23, 49): (22, 330.0),
    (26, 53): (25, 0.0059),
    (22, 45): (21, 0.1283),
    (20, 47): (21, 4.536),     (21, 47): (22, 3.3492),
    (21, 46): (22, 83.79),
    (21, 43): (20, 0.1626),
}


def parse_hydro(path):
    blocks, name, vals = {}, None, []

    def isnum(s):
        t = s.split()
        if not t:
            return False
        try:
            [float(x) for x in t]
            return True
        except ValueError:
            return False

    for ln in path.read_text().splitlines():
        if isnum(ln):
            vals.extend(float(x) for x in ln.split())
        else:
            if name and vals:
                blocks[name] = np.array(vals)
            name, vals = ln.strip(), []
    if name and vals:
        blocks[name] = np.array(vals)
    return blocks


def build_isotope_elements(blocks, n, target_epoch_d):
    """Return {Z: X_Z[n]} for Z=17..28 by summing resolved isotopes and decaying
    t0 -> target_epoch_d. At target=t0 the decay loop is a no-op (machinery only)."""
    iso = {}
    for Z in range(17, 29):
        nm = Z2NAME[Z]
        for h in blocks:
            toks = h.split()
            if len(toks) >= 2 and toks[0] == nm and toks[1].isdigit():
                iso[(Z, int(toks[1]))] = blocks[h][:n].copy()

    dt = 0.002
    nstep = int(round((target_epoch_d - T0_D) / dt))
    lam = {k: LN2 / DECAY[k][1] for k in DECAY}
    for _ in range(max(nstep, 0)):
        dX = {k: np.zeros(n) for k in iso}
        for k, arr in iso.items():
            if k in lam:
                loss = lam[k] * arr * dt
                dX[k] -= loss
                kd = (DECAY[k][0], k[1])
                if kd in dX:
                    dX[kd] += loss
        for k in iso:
            iso[k] = np.clip(iso[k] + dX[k], 0.0, None)

    Xiso = {}
    for Z in range(17, 29):
        tot = np.zeros(n)
        for (Zi, A), arr in iso.items():
            if Zi == Z:
                tot += arr
        Xiso[Z] = tot
    return Xiso, nstep


def main(keeper, out, tau_phot, target_epoch_d):
    b = parse_hydro(HYDRO)

    def g(k, n=115):
        for h, a in b.items():
            if h.startswith(k):
                return a[:n]
        raise KeyError(k)

    R = g("Radius grid") * 1e10          # cm   (outer -> inner)
    v = g("Velocity (km/s)") * 1e5       # cm/s
    T = g("Temperature (10^4 K)") * 1e4  # K
    rho = g("Density")                   # g/cm^3
    ne = g("Electron density")           # /cm^3
    chi = g("Rosseland mean opacity") * 1e-10  # cm^-1
    atomdens_file = g("Atom density")    # /cm^3  (file's own value)
    npt = len(R)

    # Rosseland optical depth from the outer edge inward
    tau = np.zeros_like(R)
    for i in range(1, len(R)):
        tau[i] = tau[i - 1] + 0.5 * (chi[i - 1] + chi[i]) * abs(R[i - 1] - R[i])
    i_phot = int(np.searchsorted(tau, tau_phot))
    v_inner = v[i_phot]
    T_inner = T[i_phot]
    r_inner = R[i_phot]
    L_inner = 4 * np.pi * r_inner ** 2 * SIGMA_SB * T_inner ** 4
    t_exp_s = target_epoch_d * 86400.0

    # --- element mass fractions: isotope-summed (decayed) for Z>=17 ---
    Xiso, nstep = build_isotope_elements(b, npt, target_epoch_d)
    Xel, iso_check = {}, {}
    for Z in Z_LIST:
        if Z <= 16:
            Xel[Z] = g(Z2NAME[Z] + " mass fraction").copy()
        else:
            Xel[Z] = Xiso[Z]
            elem_lvl = g(Z2NAME[Z] + " mass fraction")   # cross-check at t0
            iso_check[Z] = float(np.max(np.abs(Xiso[Z] - elem_lvl)))

    # FULL 26-element sum per grid point (for true-fraction normalization)
    Xfull = np.zeros(npt)
    for nm, Z in NAME2Z.items():
        try:
            Xfull += g(nm + " mass fraction")
        except KeyError:
            pass

    # --- domain = native grid points photosphere..outer, re-ordered inner->outer ---
    sel = np.arange(0, i_phot + 1)[::-1]
    n_shells = len(sel)
    R_s, v_s, T_s, rho_s, ne_s = R[sel], v[sel], T[sel], rho[sel], ne[sel]
    Xfull_s = Xfull[sel]
    r_edge = np.empty(n_shells + 1)
    r_edge[1:-1] = 0.5 * (R_s[:-1] + R_s[1:])
    r_edge[0] = r_inner
    r_edge[-1] = R_s[-1] + (R_s[-1] - r_edge[-2])
    v_edge = r_edge / t_exp_s

    # TRUE mass fractions: normalize by FULL 26-element sum (NOT tracked subset).
    # Untracked species (Ne, Ar, ...) thereby remain as inert mass -> tracked
    # number densities n = X*rho/(A*m_u) stay physically correct.
    M = np.vstack([Xel[Z][sel] for Z in Z_LIST])
    M = M / np.where(Xfull_s > 0, Xfull_s, 1.0)
    tracked_sum = M.sum(0)            # < 1 wherever inert species are present

    # --- atom-density cross-check: re-derive vs file column ---
    n_atom_rederived = np.zeros(npt)
    for nm, Z in NAME2Z.items():
        try:
            n_atom_rederived += g(nm + " mass fraction") * rho / (A_AMU[Z] * AMU)
        except KeyError:
            pass
    ratio = n_atom_rederived / np.where(atomdens_file > 0, atomdens_file, np.nan)

    # --- write ref ---
    keeper, out = Path(keeper), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rewrite = {"geometry.csv", "density.csv", "abundances.csv", "abundances.npy",
               "electron_densities.csv", "plasma_state.csv", "config.json"}
    for f in keeper.iterdir():
        if f.name in rewrite:
            continue
        link = out / f.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(f.resolve())

    sid = np.arange(n_shells)
    pd.DataFrame({"shell_id": sid, "r_inner": r_edge[:-1], "r_outer": r_edge[1:],
                  "v_inner": v_edge[:-1], "v_outer": v_edge[1:]}).to_csv(
        out / "geometry.csv", index=False)
    pd.DataFrame({"shell_id": sid, "rho": rho_s}).to_csv(out / "density.csv", index=False)
    pd.DataFrame({"shell_id": sid, "n_e": ne_s}).to_csv(out / "electron_densities.csv", index=False)
    cols = ["atomic_number"] + [str(s) for s in sid]
    pd.DataFrame([[Z] + list(M[i]) for i, Z in enumerate(Z_LIST)],
                 columns=cols).to_csv(out / "abundances.csv", index=False)

    W_init = 0.5 * (r_inner / R_s) ** 2
    pd.DataFrame({"shell_id": sid, "W": W_init, "T_rad": T_s}).to_csv(
        out / "plasma_state.csv", index=False)

    with open(keeper / "config.json") as fh:
        cfg = json.load(fh)
    cfg.update(n_shells=int(n_shells), time_explosion_s=float(t_exp_s),
               luminosity_inner_erg_s=float(L_inner),
               v_inner_min_cm_s=float(v_inner), v_outer_max_cm_s=float(v_s[-1]),
               T_inner_K=round(float(T_inner), -1))
    with open(out / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    # --- report ---
    print(f"[ddc15-init] epoch t0={T0_D}d -> target={target_epoch_d}d  ({nstep} decay steps)"
          f"  tau_phot={tau_phot}")
    print(f"[ddc15-init] photosphere: shell0 v={v_inner/1e5:.0f} km/s  T_inner={T_inner:.0f}K"
          f"  r={r_inner:.3e}cm  tau_total={tau[-1]:.0f}")
    print(f"[ddc15-init] L_inner(SB) = {L_inner:.3e} erg/s")
    print(f"[ddc15-init] domain: {n_shells} shells  v=[{v_inner/1e5:.0f},{v_s[-1]/1e5:.0f}] km/s"
          f"  T=[{T_s[0]:.0f},{T_s[-1]:.0f}]K")
    print(f"[ddc15-init] ISOTOPE cross-check (max |iso_sum - elem_level| at t0, must be ~0):")
    print("             " + "  ".join(f"Z{Z}={iso_check[Z]:.1e}" for Z in iso_check))
    print(f"[ddc15-init] ATOM-DENSITY cross-check (re-derived / file):")
    for j in [0, i_phot // 2, i_phot, npt - 1]:
        print(f"             grid[{j:3d}] v={v[j]/1e5:6.0f}km/s  re-derived={n_atom_rederived[j]:.3e}"
              f"  file={atomdens_file[j]:.3e}  ratio={ratio[j]:.3f}")
    print(f"             domain mean ratio = {np.nanmean(ratio[sel]):.3f}"
          f"  (std {np.nanstd(ratio[sel]):.3f})")
    print(f"[ddc15-init] INERT-mass deficit (1 - tracked_sum): "
          f"shell0={1-tracked_sum[0]:.3f}  max={1-tracked_sum.min():.3f}"
          f"  (mostly Ne; was renormalized away before)")
    print(f"[ddc15-init] inner-shell tracked composition (TRUE fractions):")
    for i, Z in enumerate(Z_LIST):
        if M[i, 0] > 1e-3:
            print(f"             Z={Z:2d} {M[i,0]:.4f}")
    print(f"[ddc15-init] wrote {out}")


if __name__ == "__main__":
    keeper = sys.argv[1]
    out = sys.argv[2]
    tau_phot = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0 / 3.0
    target_epoch_d = float(sys.argv[4]) if len(sys.argv) > 4 else T0_D
    main(keeper, out, tau_phot, target_epoch_d)
