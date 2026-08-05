#!/usr/bin/env python3
"""Build a LUMINA reference for the StaNdaRT *toy06* SN Ia benchmark at a near-max
epoch, by HOMOLOGOUS expansion of the toy06 input model + radioactive decay, so
Lumina's plasma (T_e, ionization) + emergent spectrum can be compared to the
StaNdaRT multi-code references (CMFGEN / ARTIS / TARDIS / ...).

Mirrors scripts/build_ddc15_epoch.py (same homologous-expansion + decay +
Lumina-reference-output structure); differences:
  * input = StaNdaRT toy06 ASCII table (202 zones) instead of the DDC15 hydro file
  * decay = analytic Bateman for the single 56Ni->56Co->56Fe chain (toy06 has no
    other unstable isotopes, stable IGE = 0)
  * the native zones above the photosphere can exceed ~70 -> resample to ~50 shells

Homologous expansion (R = v t):  v invariant; R(t)=v*t; rho(t)=rho_model*(t_model/t)^3.
  t_model = the time at which the toy06 mass fractions / radii are tabulated
  (header "tend = 4.1667e-02 DAYS"); 56Ni/stable-IGE (cols 5-6) are at t=0.
Energy: L_inner = toy06 bolometric L at the epoch = integral of the StaNdaRT
  CMFGEN emergent spectrum column nearest the epoch over wavelength.  The CMFGEN
  spectra are tabulated as L_lambda in erg/s/Ang (values ~1e36-1e39, integral
  ~1e43 erg/s = SN Ia near-max bolometric), i.e. luminosity-like already -- NO
  4*pi*(10pc)^2 factor is applied.  T_inner = (L/(4 pi r_inner^2 sigma))^(1/4).

toy06 column layout (1-indexed, per StaNdaRT read_snia_toy_model):
  (1)idx (2)vel[km/s] (3)dmass (4)mass (5)X_IGE0(stable Fe @t=0) (6)X_56Ni0 @t=0
  (7)X_IME (8)X_Ti (9)X_CO (10)rad[cm] (11)dens[g/cc] (12)temp[K]
  (13)X_56Ni (14)X_Ni(incl 56Ni) (15)X_Co(=56Co) (16)X_Fe(incl decayed 56Fe)
  (17)X_Ca (18)X_S (19)X_Si (20)X_O (21)X_C       (mass fractions @ tend)

Usage: build_toy06_epoch.py KEEPER_REF OUT_REF [TARGET_EPOCH_D] [tau_phot] [n_shells]
  KEEPER_REF = a template Lumina ref dir to inherit atomic-data symlinks + config
               base (e.g. data/tardis_reference_ddc15_0p976d).
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

MODEL = Path("data/standart_data1/input_models/snia_toy06_1h_lowres.dat")
SPEC_CMFGEN = Path("data/standart_data1/toy06/spectra_toy06_cmfgen.txt")
SIGMA_SB = 5.670374e-5
SIGMA_T  = 6.652458e-25       # Thomson cross section [cm^2]
N_A      = 6.02214076e23
PC       = 3.085677581e18
DAY      = 86400.0
LN2      = np.log(2.0)
THALF_NI56 = 6.075          # days (56Ni)
THALF_CO56 = 77.236         # days (56Co)

# elements written to the Lumina reference (Z -> name), all covered by the
# symlinked atomic data (C,O,Mg,Al,Si,S,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni).
Z_LIST = [28, 27, 26, 20, 16, 14, 8, 6]
Z2NAME = {28: "Ni", 27: "Co", 26: "Fe", 20: "Ca", 16: "S", 14: "Si", 8: "O", 6: "C"}
A_AMU  = {6: 12.011, 8: 15.999, 14: 28.085, 16: 32.06, 20: 40.078,
          26: 55.845, 27: 58.933, 28: 58.693}
# toy06 mass-fraction columns (0-indexed) for the stable species
COL = {20: 16, 16: 17, 14: 18, 8: 19, 6: 20}    # Ca,S,Si,O,C  (cols 17-21)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_OUTPUT = REPO_ROOT / "data/tardis_reference_toy06_19p48d"
PLACEHOLDER_MODE = "PLACEHOLDER_ZBAR_ONE"
TRUE_MODE = "CMFGEN_CHARGE_BALANCE"
NE_DISPOSITION = "A"
NE_DIAGNOSTIC_APPROVAL = "NE-NAMING-A-DIAGNOSTIC-2026-08-05"


class OutputSafetyError(RuntimeError):
    """Raised before input files are opened when an output path is unsafe."""


class ElectronDensityContractError(RuntimeError):
    """Raised before ``tau_i``/``i_phot`` when NE-NAMING is not authorized."""


def _resolved(path):
    """Resolve aliases and ``..`` without requiring the leaf to exist."""
    return Path(path).expanduser().resolve(strict=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def true_path_specification():
    """Return disposition-B schema only; no CMFGEN value path is implemented."""
    return {
        "status": "SPECIFICATION_ONLY_DISPOSITION_B_NOT_APPROVED",
        "electron_density_mode": TRUE_MODE,
        "formula": "n_e(v) = sum_Z sum_q q * n_Z_q(v)",
        "source": "original CMFGEN RVTJ",
        "required_identity": {
            "epoch": "same as deck target_epoch_d",
            "composition": "same isotope/element mixture as deck",
            "velocity_frame": "same homologous rest frame as geometry.csv",
        },
        "required_metadata": {
            "units": "explicit for velocity, radius, n_Z_q, and n_e",
            "ND": "record original depth count and ordering",
            "interpolation": "declare coordinate and algorithm before use",
            "duplicates": "reject or declare deterministic consolidation",
            "non_monotonic": "reject; never silently sort physical depths",
            "coverage": "record covered velocity interval and zone mask",
            "outside_grid_policy": "FATAL; never substitute Zbar=1",
        },
    }


def authorize_ne_boundary(mode, output_class, approval_token, provenance):
    """Fail closed before the first optical-depth or photosphere calculation."""
    required = ("builder_path", "builder_sha256", "input_hashes")
    missing = [name for name in required if not provenance.get(name)]
    if missing:
        raise ElectronDensityContractError(
            "[NE-NAMING][FATAL] missing provenance before i_phot: "
            + ",".join(missing)
        )
    if mode == PLACEHOLDER_MODE:
        if approval_token != NE_DIAGNOSTIC_APPROVAL:
            raise ElectronDensityContractError(
                "[NE-NAMING][FATAL] unapproved placeholder before i_phot"
            )
        if output_class != "diagnostic":
            raise ElectronDensityContractError(
                "[NE-NAMING][FATAL] placeholder production blocked by disposition A"
            )
        print(
            "[NE-NAMING][WARN] approved placeholder diagnostic "
            f"mode={mode} disposition={NE_DISPOSITION}"
        )
        return
    if mode == TRUE_MODE:
        raise ElectronDensityContractError(
            "[NE-NAMING][FATAL] CMFGEN true path is specification-only under disposition A"
        )
    raise ElectronDensityContractError(
        f"[NE-NAMING][FATAL] unsupported mode {mode!r}"
    )


def guard_output_paths(keeper, out):
    """Validate output identity and freshness before any model file is opened.

    The canonical deck itself, aliases of it, and descendants of its tree are
    forbidden.  Rejecting descendants is necessary for the stronger contract
    that the canonical tree hash remain unchanged.
    """
    keeper_path = Path(keeper).expanduser()
    out_path = Path(out).expanduser()
    keeper_real = _resolved(keeper_path)
    out_real = _resolved(out_path)
    canonical_real = _resolved(CANONICAL_OUTPUT)

    if out_real == keeper_real:
        raise OutputSafetyError(
            f"refusing input==output (including aliases): {out_path}"
        )
    if out_real == canonical_real or canonical_real in out_real.parents:
        raise OutputSafetyError(
            "refusing canonical output tree or alias: "
            f"{out_path} -> {out_real}"
        )
    if out_path.exists() or out_path.is_symlink():
        raise OutputSafetyError(f"output must be a new directory: {out_path}")
    return keeper_path, out_path


def parse_t_model(path):
    for ln in path.read_text().splitlines():
        m = re.search(r"tend\s*=\s*([0-9.eE+-]+)\s*DAYS", ln)
        if m:
            return float(m.group(1))
    raise RuntimeError("could not find tend in model header")


def parse_spec_times(path):
    for ln in path.read_text().splitlines():
        if ln.startswith("#TIMES"):
            return np.array([float(x) for x in ln.split(":")[1].split()])
    raise RuntimeError("no #TIMES in spectrum file")


def bateman(X0_ni56, t_day, stable_fe):
    """Decay X_56Ni0 -> 56Co -> 56Fe over t_day; return (Ni, Co, Fe) mass fractions.
    A=56 throughout so number-fraction Bateman == mass-fraction Bateman."""
    l1 = LN2 / THALF_NI56
    l2 = LN2 / THALF_CO56
    fNi = np.exp(-l1 * t_day)
    fCo = l1 / (l2 - l1) * (np.exp(-l1 * t_day) - np.exp(-l2 * t_day))
    fFe = 1.0 - fNi - fCo
    Ni = X0_ni56 * fNi                       # stable Ni = 0 for toy06
    Co = X0_ni56 * fCo
    Fe = X0_ni56 * fFe + stable_fe           # stable Fe = X_IGE0 (= 0 here)
    return Ni, Co, Fe


def main(keeper, out, target_epoch_d, tau_phot, n_shells_req,
         electron_density_mode, output_class, ne_approval_token):
    # GEN-GUARD: this must remain the first operation.  In particular, it must
    # precede np.loadtxt(), Path.read_text(), keeper.iterdir(), and open().
    keeper, out = guard_output_paths(keeper, out)

    d = np.loadtxt(MODEL)                    # (202, 21), inner->outer (v ascending)
    t_model = parse_t_model(MODEL)
    n = d.shape[0]
    v_kms   = d[:, 1]
    v       = v_kms * 1e5                     # cm/s, invariant
    x_ige0  = d[:, 4]                         # stable IGE (Fe) @ t=0
    x_ni56_0 = d[:, 5]                        # 56Ni @ t=0
    rho_m   = d[:, 10]                        # density @ t_model
    t_ratio = target_epoch_d / t_model
    t_exp_s = target_epoch_d * DAY

    # --- homologous expansion to target epoch ---
    r   = v * (target_epoch_d * DAY)         # R = v t  [cm]
    rho = rho_m * (t_model / target_epoch_d) ** 3

    # --- decayed composition (per native zone) ---
    Xel = {}
    Ni, Co, Fe = bateman(x_ni56_0, target_epoch_d, x_ige0)
    Xel[28], Xel[27], Xel[26] = Ni, Co, Fe
    for Z, c in COL.items():
        Xel[Z] = d[:, c].copy()
    Xtot = sum(Xel[Z] for Z in Z_LIST)

    # --- electron density (singly-ionized estimate) for the photosphere search ---
    inv_Abar = np.zeros(n)                    # sum_Z X_Z / A_Z  [per amu]
    for Z in Z_LIST:
        inv_Abar += Xel[Z] / A_AMU[Z]
    n_atom = rho * N_A * inv_Abar             # atoms / cm^3
    ne = n_atom * 1.0                         # <Z_ion> ~ 1 (singly ionized)

    # NE-NAMING disposition A: establish machine-readable provenance and
    # authorization before tau_i or i_phot exists.  The default production
    # invocation therefore stops here; only an explicitly approved scratch
    # diagnostic may continue through the placeholder branch.
    ne_provenance = {
        "builder_path": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "input_hashes": {
            str(MODEL): sha256_file(MODEL),
            str(SPEC_CMFGEN): sha256_file(SPEC_CMFGEN),
            str(keeper / "config.json"): sha256_file(keeper / "config.json"),
        },
    }
    authorize_ne_boundary(
        electron_density_mode, output_class, ne_approval_token, ne_provenance
    )

    # --- electron-scattering optical depth from the OUTER surface inward ---
    tau = np.zeros(n)
    for i in range(n - 2, -1, -1):
        tau[i] = tau[i + 1] + 0.5 * (ne[i] + ne[i + 1]) * SIGMA_T * (r[i + 1] - r[i])
    above = np.where(tau >= tau_phot)[0]
    i_phot = int(above.max()) if above.size else 0   # outermost zone with tau>=2/3
    v_inner = v[i_phot]
    r_inner = r[i_phot]
    tau_total = tau[0]

    # --- domain: photosphere..outer (inner->outer) ---
    sel = np.arange(i_phot, n)
    n_above = sel.size
    v_max = v[-1]

    # --- energy: L_inner from CMFGEN bolometric at the epoch ---
    times = parse_spec_times(SPEC_CMFGEN)
    j = int(np.argmin(np.abs(times - target_epoch_d)))
    sp = np.loadtxt(SPEC_CMFGEN)
    L_inner = float(np.trapezoid(sp[:, j + 1], sp[:, 0]))
    L_src = f"int CMFGEN L_lambda d_lambda @ {times[j]}d (erg/s, no 10pc factor)"
    T_inner = (L_inner / (4 * np.pi * r_inner ** 2 * SIGMA_SB)) ** 0.25

    # --- build shell grid (resample if too many native zones) ---
    resampled = n_above > 70
    if resampled:
        n_shells = int(n_shells_req)
        v_edge = np.linspace(v_inner, v_max, n_shells + 1)
        v_cen = 0.5 * (v_edge[:-1] + v_edge[1:])
        rho_s = np.interp(v_cen, v, rho)
        ne_s = np.interp(v_cen, v, ne)
        n_atom_s = np.interp(v_cen, v, n_atom)
        Xs = {Z: np.interp(v_cen, v, Xel[Z]) for Z in Z_LIST}
    else:
        n_shells = n_above
        v_cen = v[sel]
        v_edge = np.empty(n_shells + 1)
        v_edge[1:-1] = 0.5 * (v_cen[:-1] + v_cen[1:])
        v_edge[0] = v_inner
        v_edge[-1] = v_cen[-1] + (v_cen[-1] - v_edge[-2])
        rho_s = rho[sel]
        ne_s = ne[sel]
        n_atom_s = n_atom[sel]
        Xs = {Z: Xel[Z][sel] for Z in Z_LIST}
    r_edge = v_edge * t_exp_s
    r_inner = r_edge[0]

    # element mass-fraction matrix, renormalized to sum=1 per shell
    M = np.vstack([Xs[Z] for Z in Z_LIST])
    col_sum = M.sum(axis=0)
    M = M / np.where(col_sum > 0, col_sum, 1.0)

    # --- write reference dir (inherit atomic-data symlinks from keeper) ---
    out.mkdir(parents=True, exist_ok=False)
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
                  "v_inner": v_edge[:-1], "v_outer": v_edge[1:]}
                 ).to_csv(out / "geometry.csv", index=False)
    pd.DataFrame({"shell_id": sid, "rho": rho_s}).to_csv(out / "density.csv", index=False)
    pd.DataFrame({"shell_id": sid, "n_e": ne_s}).to_csv(out / "electron_densities.csv", index=False)
    cols = ["atomic_number"] + [str(s) for s in sid]
    pd.DataFrame([[Z] + list(M[i]) for i, Z in enumerate(Z_LIST)], columns=cols
                 ).to_csv(out / "abundances.csv", index=False)

    # plasma_state init: geometric dilution W, T_rad = T_inner * W^0.25
    r_cen = 0.5 * (r_edge[:-1] + r_edge[1:])
    W = 0.5 * (1.0 - np.sqrt(np.clip(1.0 - (r_inner / r_cen) ** 2, 0.0, 1.0)))
    T_rad = T_inner * W ** 0.25
    pd.DataFrame({"shell_id": sid, "W": W, "T_rad": T_rad}).to_csv(out / "plasma_state.csv", index=False)

    with open(keeper / "config.json") as fh:
        cfg = json.load(fh)
    cfg.update(n_shells=int(n_shells), time_explosion_s=float(t_exp_s),
               luminosity_inner_erg_s=float(L_inner), v_inner_min_cm_s=float(v_inner),
               v_outer_max_cm_s=float(v_max), T_inner_K=round(float(T_inner), -1),
               source_model="StaNdaRT toy06 (snia_toy06_1h_lowres)",
               target_epoch_d=float(target_epoch_d), t_model_d=float(t_model))
    with open(out / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    companion_names = (
        "config.json", "geometry.csv", "electron_densities.csv", "plasma_state.csv"
    )
    generation_id = "NE-DIAGNOSTIC-UNCOMMITTED"
    ne_manifest = {
        "schema": "lumina.ne-naming/v1",
        "electron_density_mode": electron_density_mode,
        "formula": "n_e = n_atom * 1.0",
        "applicable_zones": {
            "native_pre_photosphere_search": list(range(n)),
            "output_shells": list(map(int, sid)),
            "velocity_frame": "homologous rest frame; v=r/target_epoch",
        },
        "builder": {
            "path": ne_provenance["builder_path"],
            "sha256": ne_provenance["builder_sha256"],
            "producer_status": "REGISTERED_DIAGNOSTIC_BUILDER",
        },
        "inputs": {
            path: {"sha256": digest}
            for path, digest in ne_provenance["input_hashes"].items()
        },
        "source": {
            "epoch_days": float(target_epoch_d),
            "composition": "StaNdaRT toy06 analytic decay at target epoch",
            "velocity_frame": "homologous rest frame",
            "units": {"radius": "cm", "velocity": "cm/s", "n_e": "cm^-3"},
        },
        "tau_phot": float(tau_phot),
        "sigma_T_cm2": float(SIGMA_T),
        "approved_disposition": NE_DISPOSITION,
        "approval": {
            "token": ne_approval_token,
            "scope": output_class,
        },
        "generation_id": generation_id,
        "companions": {
            name: {
                "sha256": sha256_file(out / name),
                "generation_id": generation_id,
            }
            for name in companion_names
        },
        "boundary_reproduction": {
            "radius_cm": list(map(float, r)),
            "n_e_cm3": list(map(float, ne)),
            "n_atom_cm3": list(map(float, n_atom)),
            "Zbar_s": list(map(float, ne / n_atom)),
            "tau_i": list(map(float, tau)),
            "i_phot": int(i_phot),
            "v_inner_cm_s": float(v_inner),
            "r_inner_cm": float(r_inner),
            "tau_total": float(tau_total),
        },
        "output_shell_diagnostics": {
            "n_atom_cm3": list(map(float, n_atom_s)),
            "Zbar_s": list(map(float, ne_s / n_atom_s)),
        },
        "boundary_impact_magnitude": "UNQUANTIFIED_PENDING_CLEAN_ZBAR",
        "true_path_specification": true_path_specification(),
    }
    with (out / "ne_naming_manifest.json").open("w") as stream:
        json.dump(ne_manifest, stream, indent=2)
        stream.write("\n")

    # --- report ---
    ix = {Z: i for i, Z in enumerate(Z_LIST)}
    print(f"[toy06-epoch] t_model={t_model:.4f}d -> target={target_epoch_d}d  (t_ratio={t_ratio:.1f})")
    print(f"[toy06-epoch] photosphere tau_es={tau_phot:.3f}: v_inner={v_inner/1e5:.0f} km/s  "
          f"r_inner={r_inner:.3e} cm  tau_es_total={tau_total:.2f}")
    print(f"[toy06-epoch] L_inner={L_inner:.4e} erg/s  ({L_src})  T_inner={T_inner:.0f} K")
    print(f"[toy06-epoch] domain: {n_shells} shells "
          f"({'resampled from %d native' % n_above if resampled else 'native'})  "
          f"v=[{v_inner/1e5:.0f}, {v_max/1e5:.0f}] km/s")
    print(f"[toy06-epoch] inner-shell composition (decayed): " +
          "  ".join(f"{Z2NAME[Z]}={M[ix[Z],0]:.3f}" for Z in Z_LIST if M[ix[Z], 0] > 1e-3))
    print(f"[toy06-epoch] Fe-group inner shell: "
          f"Ni={M[ix[28],0]:.3f} Co={M[ix[27],0]:.3f} Fe={M[ix[26],0]:.3f}")
    print(f"[toy06-epoch] wrote {out}")
    print(
        "[NE-NAMING][WARN] manifest=ne_naming_manifest.json "
        f"mode={electron_density_mode} formula='n_e = n_atom * 1.0' "
        f"zones=native:0-{n - 1},output:0-{n_shells - 1} "
        f"tau_phot={tau_phot:.17g} disposition={NE_DISPOSITION}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keeper", help="existing template reference directory")
    parser.add_argument(
        "out",
        help="required, new output directory (canonical deck and aliases forbidden)",
    )
    parser.add_argument("target_epoch_d", nargs="?", type=float, default=19.48)
    parser.add_argument("tau_phot", nargs="?", type=float, default=2.0 / 3.0)
    parser.add_argument("n_shells", nargs="?", type=int, default=50)
    parser.add_argument(
        "--electron-density-mode",
        default=PLACEHOLDER_MODE,
        help=(
            f"mode value ({PLACEHOLDER_MODE}; {TRUE_MODE} is schema-only under "
            "disposition A)"
        ),
    )
    parser.add_argument(
        "--output-class",
        choices=("production", "canonical", "diagnostic"),
        default="production",
        help="placeholder is permitted only for an approved scratch diagnostic",
    )
    parser.add_argument("--ne-approval-token")
    args = parser.parse_args()
    try:
        main(
            args.keeper,
            args.out,
            args.target_epoch_d,
            args.tau_phot,
            args.n_shells,
            args.electron_density_mode,
            args.output_class,
            args.ne_approval_token,
        )
    except (OutputSafetyError, ElectronDensityContractError) as exc:
        parser.error(str(exc))
