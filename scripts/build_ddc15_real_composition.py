#!/usr/bin/env python3
"""Build a reference dir with the REAL DDC15 composition + density, replacing the
hand-rolled ddc15_profile_v3 approximation.

Source: data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d (Blondin+2015 public release).
It gives, at 0.976 d past explosion on a 115-point velocity grid: radius,
velocity, density, atom/electron density, Rosseland tau, kappa, and the mass
fractions of 26 elements + 105 isotopes (isotopes resolved for Z=17..28).

Pipeline:
  1. parse the hydro file
  2. decay the iron-peak isotopes (Z=17..28) from 0.976 d to the target epoch
     (default 17.76 d) through the standard SN Ia chains (56Ni->56Co->56Fe,
     55Co->55Fe, 48Cr->48V->48Ti, 52Mn->52Cr, ...), re-summing to elements
  3. interpolate element mass fractions X_Z(v) onto our shell-midpoint velocities
  4. density at the target epoch via homologous scaling rho(v,t)=rho(v,t0)(t0/t)^3
  5. write a self-consistent ref (geometry/density/abundances/electron_densities/
     plasma_state/config) and symlink all atomic data from the keeper ref

The element mass fractions are renormalized per shell over the LUMINA-tracked set
(untracked species H/He/Ne/Na/Ar/K/P/Cl/Ba/N/F are dropped), matching the
existing pipeline convention.

Usage:
  build_ddc15_real_composition.py KEEPER_REF OUT_REF [t_exp_d] [vi_kms] [n_shells]
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HYDRO = Path("data/ddc15_hydro/DDC15_SN_HYDRO_DATA_0.976d")
T0_D = 0.976                       # epoch of the hydro file
M_SUN = 1.989e33

# LUMINA-tracked elements (must match the atomic data in the keeper ref)
Z_LIST = [6, 8, 12, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28]

# element name in the hydro file -> Z
NAME2Z = {
    "HYD": 1, "HE": 2, "CARB": 6, "NIT": 7, "OXY": 8, "FLU": 9, "NEON": 10,
    "SOD": 11, "MAG": 12, "ALUM": 13, "SIL": 14, "PHOS": 15, "SUL": 16,
    "CHL": 17, "ARG": 18, "POT": 19, "CAL": 20, "SCAN": 21, "TIT": 22,
    "VAN": 23, "CHRO": 24, "MAN": 25, "IRON": 26, "COB": 27, "NICK": 28,
    "BAR": 56,
}
Z2NAME = {v: k for k, v in NAME2Z.items()}

# radioactive decay table: (Z_parent, A) -> (Z_daughter, half_life_days).
# Daughter has the same A (beta/EC keeps nucleon number). Isotopes absent here
# are treated as stable. Covers every chain with non-negligible parent mass and
# half-life within ~2 orders of the 0.976->17.76 d window.
LN2 = np.log(2.0)
DECAY = {
    (28, 56): (27, 6.075),     (27, 56): (26, 77.236),    # 56Ni->56Co->56Fe
    (28, 57): (27, 1.483),     (27, 57): (26, 271.74),    # 57Ni->57Co->57Fe
    (27, 55): (26, 17.53),     (26, 55): (25, 1002.0),    # 55Co->55Fe->55Mn
    (26, 52): (25, 0.3448),    (25, 52): (24, 5.591),     # 52Fe->52Mn->52Cr
    (24, 48): (23, 0.898),     (23, 48): (22, 15.973),    # 48Cr->48V->48Ti
    (22, 44): (21, 21915.0),   (21, 44): (20, 0.1654),    # 44Ti->44Sc->44Ca
    (25, 51): (24, 0.0321),    (24, 51): (23, 27.70),     # 51Mn->51Cr->51V
    (24, 49): (23, 0.0294),    (23, 49): (22, 330.0),     # 49Cr->49V->49Ti
    (26, 53): (25, 0.0059),                               # 53Fe->53Mn (Mn53~stable)
    (22, 45): (21, 0.1283),    (21, 45): (20, 0.1827),    # 45Ti->45Sc->45Ca(stable@45 actually Sc45 stable) -> keep as Sc
    (20, 47): (21, 4.536),     (21, 47): (22, 3.3492),    # 47Ca->47Sc->47Ti
    (21, 46): (22, 83.79),                                # 46Sc->46Ti
    (21, 43): (20, 0.1626),                               # 43Sc->43Ca
}
# 45Sc is stable; drop the spurious Sc45->Ca45 line above by overriding:
DECAY.pop((21, 45), None)


def parse_hydro(path):
    """Return (v_kms[115], rho0[115], named_blocks dict header->array)."""
    lines = path.read_text().splitlines()
    blocks = {}
    cur_name, cur_vals = None, []

    def is_num_line(s):
        toks = s.split()
        if not toks:
            return False
        try:
            [float(t) for t in toks]
            return True
        except ValueError:
            return False

    for ln in lines:
        if is_num_line(ln):
            cur_vals.extend(float(t) for t in ln.split())
        else:
            if cur_name is not None and cur_vals:
                blocks[cur_name] = np.array(cur_vals)
            cur_name = ln.strip()
            cur_vals = []
    if cur_name is not None and cur_vals:
        blocks[cur_name] = np.array(cur_vals)
    return blocks


def get_block(blocks, key, n=115):
    for h, arr in blocks.items():
        if h.startswith(key):
            return arr[:n]
    raise KeyError(key)


def main(keeper, out, t_exp_d, vi_kms, n_shells, L_inner):
    t_exp_s = t_exp_d * 86400.0
    blocks = parse_hydro(HYDRO)
    v_kms = get_block(blocks, "Velocity (km/s)")
    rho0 = get_block(blocks, "Density (gm/cm^3)")
    npt = len(v_kms)

    # --- collect isotope mass fractions for Z=17..28, element-level for Z<=16 ---
    iso = {}  # (Z,A) -> array(npt)
    for Z in range(17, 29):
        nm = Z2NAME[Z]
        for h in blocks:
            toks = h.split()
            if len(toks) >= 2 and toks[0] == nm and toks[1].isdigit():
                A = int(toks[1])
                iso[(Z, A)] = blocks[h][:npt].copy()

    # --- decay isotopes t0 -> t_exp_d (forward Euler, conserves mass per A) ---
    dt = 0.002
    nstep = int(round((t_exp_d - T0_D) / dt))
    lam = {k: LN2 / DECAY[k][1] for k in DECAY}
    for _ in range(nstep):
        dX = {k: np.zeros(npt) for k in iso}
        for (Z, A), arr in iso.items():
            k = (Z, A)
            if k in lam:
                loss = lam[k] * arr * dt
                dX[k] -= loss
                Zd = DECAY[k][0]
                kd = (Zd, A)
                if kd in dX:
                    dX[kd] += loss
                # if daughter not tracked as isotope, mass leaves the iron-peak
                # bookkeeping (negligible for our chains; all daughters tracked)
        for k in iso:
            iso[k] = np.clip(iso[k] + dX[k], 0.0, None)

    # --- element mass fractions on the DDC15 grid at the target epoch ---
    Xel = {}
    for Z in Z_LIST:
        if Z <= 16:
            Xel[Z] = get_block(blocks, Z2NAME[Z] + " mass fraction").copy()
        else:
            tot = np.zeros(npt)
            for (Zi, A), arr in iso.items():
                if Zi == Z:
                    tot += arr
            Xel[Z] = tot

    # --- our shell grid (identical to the run grid) ---
    keeper = Path(keeper)
    geo_k = pd.read_csv(keeper / "geometry.csv")
    v_outer_max = geo_k["v_outer"].iloc[-1] / 1e5  # km/s
    v_edges = np.linspace(vi_kms, v_outer_max, n_shells + 1)
    v_in, v_out = v_edges[:-1], v_edges[1:]
    v_mid = 0.5 * (v_in + v_out)

    # interpolate X_Z(v) and rho(v) onto shell midpoints (v ascending)
    order = np.argsort(v_kms)
    v_s = v_kms[order]
    X_shell = {Z: np.interp(v_mid, v_s, Xel[Z][order]) for Z in Z_LIST}
    rho0_mid = np.interp(v_mid, v_s, rho0[order])
    rho_target = rho0_mid * (T0_D / t_exp_d) ** 3  # homologous

    # renormalize per shell over the tracked set
    M = np.vstack([X_shell[Z] for Z in Z_LIST])  # (nZ, n_shells)
    colsum = M.sum(axis=0)
    M = M / np.where(colsum > 0, colsum, 1.0)

    # --- write the ref dir ---
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # symlink all atomic data from keeper EXCEPT the structure files we rewrite
    rewrite = {"geometry.csv", "density.csv", "abundances.csv", "abundances.npy",
               "electron_densities.csv", "plasma_state.csv", "config.json"}
    for f in keeper.iterdir():
        if f.name in rewrite:
            continue
        link = out / f.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(f.resolve())

    r_in = v_in * 1e5 * t_exp_s
    r_out = v_out * 1e5 * t_exp_s
    pd.DataFrame({"shell_id": np.arange(n_shells), "r_inner": r_in,
                  "r_outer": r_out, "v_inner": v_in * 1e5,
                  "v_outer": v_out * 1e5}).to_csv(out / "geometry.csv", index=False)
    pd.DataFrame({"shell_id": np.arange(n_shells),
                  "rho": rho_target}).to_csv(out / "density.csv", index=False)

    cols = ["atomic_number"] + [str(s) for s in range(n_shells)]
    rows = [[Z] + list(M[i]) for i, Z in enumerate(Z_LIST)]
    pd.DataFrame(rows, columns=cols).to_csv(out / "abundances.csv", index=False)

    # config + init plasma/electron (same convention as the slurm PYEOF block)
    with open(keeper / "config.json") as fh:
        cfg = json.load(fh)
    cfg["n_shells"] = n_shells
    cfg["time_explosion_s"] = float(t_exp_s)
    cfg["luminosity_inner_erg_s"] = float(L_inner)
    cfg["v_inner_min_cm_s"] = vi_kms * 1e5
    cfg["v_outer_max_cm_s"] = v_outer_max * 1e5
    sigma_SB = 5.670374e-5
    T_inner = (L_inner / (4 * np.pi * r_in[0] ** 2 * sigma_SB)) ** 0.25
    cfg["T_inner_K"] = round(float(T_inner), -1)
    with open(out / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    W_init = 0.5 * (r_in[0] / r_in) ** 2
    pd.DataFrame({"shell_id": np.arange(n_shells), "W": W_init,
                  "T_rad": np.full(n_shells, T_inner)}).to_csv(
        out / "plasma_state.csv", index=False)
    m_H = 1.6726e-24
    ne_init = (rho_target / m_H) / 14.0
    pd.DataFrame({"shell_id": np.arange(n_shells),
                  "n_e": ne_init}).to_csv(out / "electron_densities.csv", index=False)

    # report
    print(f"[ddc15] t0={T0_D}d -> target={t_exp_d}d  ({nstep} decay steps)")
    print(f"[ddc15] grid: {n_shells} shells  v=[{vi_kms:.0f},{v_outer_max:.0f}] km/s"
          f"  T_inner_init={T_inner:.0f}K")
    print(f"[ddc15] density(17.76d): inner={rho_target[0]:.3e} outer={rho_target[-1]:.3e} g/cm^3")
    print(f"[ddc15] mass fractions (inner shell0 / outer shell{n_shells-1}):")
    for i, Z in enumerate(Z_LIST):
        print(f"          Z={Z:2d}  {M[i,0]:.4e}   {M[i,-1]:.4e}")
    # decay sanity: total Ni+Co+Fe mass conservation across the chain
    print(f"[ddc15] wrote {out}")


if __name__ == "__main__":
    keeper = sys.argv[1]
    out = sys.argv[2]
    t_exp_d = float(sys.argv[3]) if len(sys.argv) > 3 else 1534464 / 86400.0
    vi_kms = float(sys.argv[4]) if len(sys.argv) > 4 else 9019.0
    n_shells = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    L_inner = 1.22e43
    main(keeper, out, t_exp_d, vi_kms, n_shells, L_inner)
