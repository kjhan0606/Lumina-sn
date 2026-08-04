#!/usr/bin/env python3
"""Inventory UV opacity carriers in the Si II 1808 Å pump band.

For each iron-peak II ion (Fe/Co/Cr/Ni/Mn II), sum n_pop at shell 0 for levels
whose photoionization threshold falls in [1700, 1900] Å. Also count bound-bound
line transitions whose wavelength falls in [1700, 1900] Å.

The hypothesis (project #302 → priority #2): some species can absorb UV at
1808 Å, reducing the J_ν that pumps Si II level 5 (b_l=27 → ionization current
×27). Find the species with significant n_pop in this band.
"""
import numpy as np
import pandas as pd
from pathlib import Path

REF = Path(
    "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/"
    "tardis_reference_v3_femerge_capraise_bbfix_metafix_asE"
)
DUMP = Path(
    "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/"
    "paperDDC15v3asEfloor_bbfix_2002bo_vi9019_L1p0_nltedump_158780/"
    "nlte_levels_iter003.csv"
)
OUT_DIR = Path(
    "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/layer2_bl_diagnosis"
)

# (Z, ion, label, IP_eV) — NIST IPs
SPECIES = [
    (12, 1, "Mg II", 15.0353),
    (13, 1, "Al II", 18.829),
    (14, 1, "Si II", 16.346),
    (20, 1, "Ca II", 11.8717),
    (24, 1, "Cr II", 16.486),
    (25, 1, "Mn II", 15.640),
    (26, 1, "Fe II", 16.188),
    (27, 1, "Co II", 17.084),
    (28, 1, "Ni II", 18.169),
]

PUMP_LO, PUMP_HI = 1700.0, 1900.0  # Si II pump band


def load_has_cmfgen(path):
    with open(path, "rb") as f:
        magic = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        assert magic == 0x434D4644
        f.read(4)
        n_lev = int(np.frombuffer(f.read(4), dtype="<i4")[0])
        n_freq = int(np.frombuffer(f.read(4), dtype="<i4")[0])
        f.read(16)
        has = np.frombuffer(f.read(n_lev), dtype=np.int8).astype(bool)
    return has


def main():
    levels = pd.read_csv(REF / "levels.csv")
    has_cmf = load_has_cmfgen(REF / "cmfgen_sigma_bf.bin")
    levels["has_cmfgen"] = has_cmf
    print(f"levels.csv {len(levels)} rows, CMFGEN coverage {has_cmf.sum()} ({has_cmf.mean():.1%})")

    # Load NLTE dump shell 0 only — much smaller subset
    print(f"\nLoading NLTE dump shell 0 ...")
    df = pd.read_csv(DUMP, usecols=["Z", "ion", "shell", "global_idx", "n_pop"])
    shell0 = df[df["shell"] == 0].copy()
    print(f"  shell0 rows: {len(shell0):,}")

    # BF audit per species
    print(f"\n=== BF threshold inventory at λ ∈ [{PUMP_LO}, {PUMP_HI}] Å ===")
    print(f"  {'species':<8} {'n_lev_band':>10} {'g_sum':>8} {'n_CMF':>5} "
          f"{'n_Kram':>6} {'n_pop_sum':>11} {'n_ion_tot':>11} {'frac_band':>9}")

    bf_rows = []
    for Z, ion, label, IP in SPECIES:
        sub = levels[(levels["atomic_number"] == Z) & (levels["ion_number"] == ion)].copy()
        if sub.empty:
            continue
        sub["E_thresh"] = IP - sub["energy_eV"]
        sub["lam_thresh_A"] = np.where(sub["E_thresh"] > 0, 12397.6 / sub["E_thresh"], np.nan)
        band = sub[(sub["lam_thresh_A"] >= PUMP_LO) & (sub["lam_thresh_A"] < PUMP_HI)]

        # cross with NLTE dump
        dump_sub = shell0[(shell0["Z"] == Z) & (shell0["ion"] == ion)]
        dump_band = dump_sub[dump_sub["global_idx"].isin(band.index)]
        n_pop_sum = float(dump_band["n_pop"].sum())
        n_ion_tot = float(dump_sub["n_pop"].sum())

        n_cmf = int(band["has_cmfgen"].sum())
        n_kram = int((~band["has_cmfgen"]).sum())
        g_sum = float(band["g"].sum())
        frac = n_pop_sum / max(n_ion_tot, 1e-30)

        print(f"  {label:<8} {len(band):>10} {g_sum:>8.0f} {n_cmf:>5} {n_kram:>6} "
              f"{n_pop_sum:>11.3e} {n_ion_tot:>11.3e} {frac:>9.2%}")
        bf_rows.append(dict(species=label, Z=Z, ion=ion, n_lev_band=len(band),
                            n_pop_band=n_pop_sum, n_pop_total=n_ion_tot, frac=frac))

    # BB audit: count line transitions with rest wavelength in pump band, per (Z,ion)
    lines_path = REF / "line_list.csv"
    if lines_path.exists():
        print(f"\n=== BB line inventory in [{PUMP_LO}, {PUMP_HI}] Å ===")
        lines = pd.read_csv(lines_path, usecols=["atomic_number", "ion_number",
                                                  "wavelength", "f_lu"])
        # 'wavelength' is in Å (LUMINA convention)
        bb_band = lines[(lines["wavelength"] >= PUMP_LO) & (lines["wavelength"] < PUMP_HI)]
        print(f"  Total lines in band: {len(bb_band):,}")
        print(f"  {'species':<8} {'N_lines':>8} {'Σf_lu':>10} {'max_f_lu':>10}")
        for Z, ion, label, _ in SPECIES:
            sel = bb_band[(bb_band["atomic_number"] == Z) & (bb_band["ion_number"] == ion)]
            if len(sel) == 0:
                continue
            f_sum = float(sel["f_lu"].sum())
            f_max = float(sel["f_lu"].max())
            print(f"  {label:<8} {len(sel):>8} {f_sum:>10.3e} {f_max:>10.3e}")
    else:
        print(f"\n(no lines.csv at {lines_path})")

    # Top BB lines in band, regardless of species
    if lines_path.exists():
        top_bb = bb_band.nlargest(20, "f_lu")
        print(f"\nTop 20 BB lines in [{PUMP_LO}, {PUMP_HI}] Å by f_lu:")
        print(top_bb.to_string(index=False, float_format=lambda x: f"{x:.4e}"))


if __name__ == "__main__":
    main()
