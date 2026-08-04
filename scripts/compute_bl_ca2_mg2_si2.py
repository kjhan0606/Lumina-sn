#!/usr/bin/env python3
"""Compute NLTE departure coefficient b_l = n_NLTE / n_LTE for Ca II, Mg II, Si II.

For each (Z, ion, shell, level):
    n_LTE(l) = n_ion_total * (g_l / U_ion(T_e)) * exp(-E_l / kT_e)
    b_l(l)   = n_NLTE(l) / n_LTE(l)

The partition function U_ion(T_e) is computed self-consistently from the same
NLTE level list (treating the dump's energies/degeneracies as the ion model
truth — which they are, since the LTE Saha-Boltzmann internally references the
same levels in lumina_plasma.c).

Output: per-level table sorted by b_l descending for shells 0/5/10 (photosphere,
mid-shell, outer line-forming).

This is Layer-2 final diagnosis (project #302): which excited levels have b_l>>1
and therefore drive the Σ n(l)·σ_bf(l)·J_ν ionization current to inflated values?
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

KB_EV = 8.617333262e-5  # Boltzmann constant in eV/K

DUMP = Path(
    "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/"
    "paperDDC15v3asEfloor_bbfix_2002bo_vi9019_L1p0_nltedump_158780/"
    "nlte_levels_iter003.csv"
)

OUT_DIR = Path(
    "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/layer2_bl_diagnosis"
)
OUT_DIR.mkdir(exist_ok=True)

TARGETS = [
    (20, 1, "Ca II"),
    (12, 1, "Mg II"),
    (14, 1, "Si II"),
]
SHELLS_TO_REPORT = [0, 5, 10]  # photosphere, mid, outer line-forming
TOP_N = 15  # top levels by b_l


def main():
    print(f"Loading {DUMP.name} ...")
    df = pd.read_csv(DUMP)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    # We expect 30 shells × N_NLTE_levels rows. Sanity check.
    n_shells = df["shell"].nunique()
    print(f"  n_shells = {n_shells}")

    summary_rows = []

    for Z, ion, label in TARGETS:
        sub = df[(df["Z"] == Z) & (df["ion"] == ion)].copy()
        if sub.empty:
            print(f"\n=== {label} (Z={Z}, ion={ion}) — NO ROWS — possibly not NLTE-active")
            continue

        # Levels per shell
        n_lev = sub["level_idx"].nunique()
        print(f"\n=== {label}: {n_lev} NLTE levels × {n_shells} shells = {len(sub):,} rows")

        for shell in SHELLS_TO_REPORT:
            shell_df = sub[sub["shell"] == shell].copy()
            if shell_df.empty:
                continue
            T_e = float(shell_df["T_e"].iloc[0])
            n_ion = float(shell_df["n_ion_total"].iloc[0])
            W = float(shell_df["W"].iloc[0])
            T_rad = float(shell_df["T_rad"].iloc[0])

            # LTE Saha-Boltzmann reference: n_LTE(l) = n_ion * g_l/U * exp(-E_l/kT_e)
            # Compute partition function U at this shell's T_e from the dump levels:
            kT = KB_EV * T_e
            boltz = shell_df["g"].astype(float) * np.exp(
                -shell_df["E_eV"].astype(float) / kT
            )
            U = float(boltz.sum())
            shell_df["n_LTE"] = n_ion * boltz / U
            shell_df["b_l"] = np.where(
                shell_df["n_LTE"] > 0,
                shell_df["n_pop"] / shell_df["n_LTE"],
                np.nan,
            )
            shell_df["nl_sigmabf_prox"] = shell_df["n_pop"]  # n(l) drives ionization current

            # Rank by b_l × n(l) — this is what matters for the ionization current.
            # Pure b_l>>1 on empty levels doesn't actually move R_bf much.
            shell_df["impact"] = shell_df["b_l"] * shell_df["n_LTE"]  # = n_pop, sanity
            # Re-sort by n_pop / n_pop(ground) to see relative excitation.
            n_ground = shell_df.loc[shell_df["level_idx"] == 0, "n_pop"].iloc[0] \
                if (shell_df["level_idx"] == 0).any() else float(shell_df["n_pop"].max())
            shell_df["n_pop_frac"] = shell_df["n_pop"] / max(n_ion, 1e-30)

            print(
                f"\n  shell={shell}  T_e={T_e:.0f}K  T_rad={T_rad:.0f}K  "
                f"W={W:.4f}  n_ion={n_ion:.3e}  U(T_e)={U:.4f}"
            )

            # Top by b_l (interesting departures)
            top_bl = shell_df.nlargest(TOP_N, "b_l")[
                ["level_idx", "E_eV", "g", "n_pop", "n_LTE", "b_l", "n_pop_frac"]
            ]
            print(f"  Top {TOP_N} levels by b_l:")
            print(top_bl.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

            # Save full per-level table for this (Z, ion, shell)
            out_path = OUT_DIR / f"bl_Z{Z}_ion{ion}_shell{shell}.csv"
            shell_df_out = shell_df[
                ["level_idx", "E_eV", "g", "n_pop", "n_LTE", "b_l", "n_pop_frac"]
            ].sort_values("level_idx").reset_index(drop=True)
            shell_df_out.to_csv(out_path, index=False)

            # Summary row: ground b_l, max b_l, mean b_l (excited), n_high_bl
            ground = shell_df[shell_df["level_idx"] == 0]
            b_ground = float(ground["b_l"].iloc[0]) if len(ground) else np.nan
            excited = shell_df[shell_df["level_idx"] > 0]
            b_max = float(excited["b_l"].max()) if len(excited) else np.nan
            n_b10 = int((excited["b_l"] > 10).sum())
            n_b100 = int((excited["b_l"] > 100).sum())
            # Mean excited b_l weighted by n_pop (the levels that actually carry density)
            w = excited["n_pop"].clip(lower=0).fillna(0).values
            bl = excited["b_l"].fillna(0).values
            wbl = float((w * bl).sum() / max(w.sum(), 1e-30))
            summary_rows.append(
                dict(
                    species=label,
                    Z=Z,
                    ion=ion,
                    shell=shell,
                    T_e=T_e,
                    T_rad=T_rad,
                    W=W,
                    n_ion=n_ion,
                    U=U,
                    b_ground=b_ground,
                    b_excited_max=b_max,
                    b_excited_n_pop_weighted=wbl,
                    n_levels_b_gt_10=n_b10,
                    n_levels_b_gt_100=n_b100,
                )
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "bl_summary_ca2_mg2_si2.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nWrote summary → {summary_path}")
    print(f"Per-(Z,ion,shell) tables → {OUT_DIR}/bl_Z*_ion*_shell*.csv")


if __name__ == "__main__":
    main()
