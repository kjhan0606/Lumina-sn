#!/usr/bin/env python3
"""#299 Fe II NLTE-solved n_lower vs nebular W·ζ formula — quantitative comparison.

Method:
  - Read LUMINA's nlte_levels_iter*.csv dump (LUMINA_NLTE_LEVEL_DUMP=1).
  - For each (Z, ion, shell, level) in the dump, compute the nebular n_lower
    using the EXACT LUMINA formula (src/lumina_plasma.c:282-313, 682-694):
      Z_part = Σ_meta g·exp(-E·eV/kT_rad) + W·Σ_non_meta g·exp(-E·eV/kT_rad)
      n_lower = n_ion · weight · g · exp(-E·eV/kT_rad) / Z_part
        weight = 1 if metastable else W
  - Match dump levels to levels.csv by (Z, ion) + closest E_eV match.
  - Compute ratio NLTE/nebular per (Z, ion, shell, level).

The verdict:
  - If |log10(ratio)| < 0.3 (factor 2) for τ-relevant Fe II abs levels →
    NLTE machinery is essentially nebular for this ion. Bb-fog must come elsewhere.
  - If |log10(ratio)| > 0.5 (factor 3+) for the dominant abs levels →
    NLTE is a real lever; CMFGEN-vs-LUMINA NLTE quality is plausible #1 cause.

Output: figure + per-level CSV ranked by Σ_shell n_pop (τ proxy).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
REF  = ROOT / "data/tardis_reference_v3_femerge_capraise_bbfix_asE"

K_BOLTZMANN = 1.380649e-16   # erg/K
EV_TO_ERG   = 1.602176634e-12

# Which NLTE iter dump to analyze (final converged iter)
ITER_TAG = "003"  # iter000=NLTE iter1, ..., iter003=NLTE iter4 (last of N_ITER=6 with START=2)

# Target — Fe II is the dominant τ_Sob carrier (95.8% per #298 attribution)
TARGET_Z   = 26
TARGET_ION = 1   # Fe II


def load_levels(ref_dir):
    df = pd.read_csv(ref_dir / "levels.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


def compute_nebular_pop_for_dump(dump_df, levels_df):
    """For every (Z, ion, shell, dumped level) row, compute nebular n_lower
    using EXACT LUMINA formula. Returns dump_df with extra columns:
      meta : metastable flag (matched from levels.csv)
      Z_part : partition function for (Z, ion, shell)
      n_neb : nebular n_lower
      ratio_nlte_over_neb : n_pop / n_neb
    """
    out = []

    # Group dump by (Z, ion). For each (Z, ion), grab the FULL level table
    # for that ion from levels.csv and compute Z_part per shell from the
    # FULL spectrum of levels (not just NLTE-tracked ones).
    for (Z, ion), grp in dump_df.groupby(["Z", "ion"]):
        ion_levs = levels_df[
            (levels_df["atomic_number"] == Z) &
            (levels_df["ion_number"]    == ion)
        ].copy()
        if len(ion_levs) == 0:
            print(f"[WARN] no levels.csv data for Z={Z} ion={ion}")
            continue
        ion_levs = ion_levs.reset_index(drop=True)
        E_full   = ion_levs["energy_eV"].values
        g_full   = ion_levs["g"].values.astype(int)
        meta_full = ion_levs["metastable"].values.astype(int)

        # Match each dump row's (E_eV, g) to a levels.csv row to recover meta.
        # Use g exact + E_eV closest.
        for _, row in grp.iterrows():
            E_d, g_d = float(row["E_eV"]), int(row["g"])
            mask_g = (g_full == g_d)
            if mask_g.sum() == 0:
                # fallback: just closest E
                cand_idx = int(np.argmin(np.abs(E_full - E_d)))
            else:
                cand = np.where(mask_g)[0]
                cand_idx = int(cand[np.argmin(np.abs(E_full[cand] - E_d))])
            meta = int(meta_full[cand_idx])

            T_rad = float(row["T_rad"])
            W     = float(row["W"])
            n_ion = float(row["n_ion_total"])

            beta = EV_TO_ERG / (K_BOLTZMANN * T_rad)

            boltz_full = E_full * beta
            keep = boltz_full < 500.0
            bf = np.zeros_like(E_full)
            bf[keep] = g_full[keep] * np.exp(-boltz_full[keep])
            Z_meta = float(bf[meta_full == 1].sum())
            Z_non  = float(bf[meta_full == 0].sum())
            Z_part = Z_meta + W * Z_non
            if Z_part < 1e-300:
                Z_part = 1e-300

            boltz_d = E_d * beta
            if boltz_d < 500.0:
                weight = 1.0 if meta == 1 else W
                n_neb = n_ion * weight * g_d * np.exp(-boltz_d) / Z_part
            else:
                n_neb = 0.0

            n_pop = float(row["n_pop"])
            ratio = (n_pop / n_neb) if (n_neb > 0 and n_pop > 0) else np.nan

            out.append({
                "Z": Z, "ion": ion, "shell": int(row["shell"]),
                "level_idx": int(row["level_idx"]),
                "global_idx": int(row["global_idx"]),
                "E_eV": E_d, "g": g_d, "meta": meta,
                "T_rad": T_rad, "W": W, "n_ion_total": n_ion,
                "Z_part": Z_part, "n_pop_nlte": n_pop, "n_neb": n_neb,
                "ratio_nlte_over_neb": ratio,
                "log10_ratio": np.log10(ratio) if (ratio > 0) else np.nan,
            })

    return pd.DataFrame(out)


def main():
    if len(sys.argv) < 2:
        print("usage: compare_nlte_vs_nebular_fe2.py <run_log_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    dump_file = run_dir / f"nlte_levels_iter{ITER_TAG}.csv"
    if not dump_file.exists():
        # try iter000 if iter003 not yet present
        for tag in ["003", "002", "001", "000"]:
            cand = run_dir / f"nlte_levels_iter{tag}.csv"
            if cand.exists():
                dump_file = cand
                print(f"using {dump_file.name}")
                break
        else:
            print(f"ERROR: no nlte_levels_iter*.csv in {run_dir}")
            sys.exit(1)

    dump = pd.read_csv(dump_file)
    print(f"loaded {len(dump)} dump rows  (unique Z={sorted(dump.Z.unique())})")

    levels = load_levels(REF)
    print(f"levels.csv: {len(levels)} rows  ({levels.atomic_number.nunique()} elements)")

    result = compute_nebular_pop_for_dump(dump, levels)
    print(f"computed nebular pops for {len(result)} (Z,ion,shell,level) tuples")

    # Focus on Fe II
    fe2 = result[(result.Z == TARGET_Z) & (result.ion == TARGET_ION)].copy()
    print(f"\n=== Fe II results: {len(fe2)} rows ({fe2.shell.nunique()} shells × "
          f"{fe2.level_idx.nunique()} levels) ===")

    if len(fe2) == 0:
        print("No Fe II data — check NLTE solver ran for Fe II.")
        sys.exit(0)

    # Per-shell-summed n_pop ranking (τ_Sob proxy — for fixed f_lu, λ, t_exp,
    # τ ∝ n_lower so highest summed pop = highest τ contribution)
    per_lvl = fe2.groupby("level_idx").agg(
        E_eV=("E_eV","first"), g=("g","first"), meta=("meta","first"),
        sum_n_pop_nlte=("n_pop_nlte","sum"),
        sum_n_neb=("n_neb","sum"),
        median_log10_ratio=("log10_ratio","median"),
        min_log10_ratio=("log10_ratio","min"),
        max_log10_ratio=("log10_ratio","max"),
    ).sort_values("sum_n_pop_nlte", ascending=False)

    print("\n--- Top 20 Fe II absorption levels by Σshell n_NLTE (τ proxy) ---")
    print(f"  {'lvl':>4} {'E_eV':>7} {'g':>3} {'meta':>4} "
          f"{'Σn_NLTE':>10} {'Σn_neb':>10} {'med log10':>9} {'minmax':>14}")
    for lvl, r in per_lvl.head(20).iterrows():
        mr = (f"[{r['min_log10_ratio']:+.2f},{r['max_log10_ratio']:+.2f}]"
              if not np.isnan(r['min_log10_ratio']) else "[nan]")
        print(f"  {lvl:4d} {r['E_eV']:7.3f} {int(r['g']):3d} {int(r['meta']):4d} "
              f"{r['sum_n_pop_nlte']:10.2e} {r['sum_n_neb']:10.2e} "
              f"{r['median_log10_ratio']:+9.3f} {mr:>14}")

    # Save per-level summary
    out_csv = run_dir / "fe2_nlte_vs_nebular_per_level.csv"
    per_lvl.to_csv(out_csv)
    print(f"\nsaved: {out_csv}")

    # Save raw merged data
    raw_csv = run_dir / "fe2_nlte_vs_nebular_raw.csv"
    fe2.to_csv(raw_csv, index=False)
    print(f"saved: {raw_csv}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. log10(ratio) histogram for all Fe II rows
    ax = axes[0,0]
    vals = fe2["log10_ratio"].dropna().values
    ax.hist(vals, bins=80, color="steelblue", alpha=0.8)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.axvline(-0.3, color="orange", ls=":", lw=0.8); ax.axvline(0.3, color="orange", ls=":", lw=0.8)
    ax.set_xlabel("log10(n_NLTE / n_neb)")
    ax.set_ylabel("# (Z=26 ion=1, all shells × levels)")
    ax.set_title(f"Fe II n_lower: NLTE vs nebular  ({len(vals)} samples)\n"
                 f"median = {np.nanmedian(vals):+.3f}   p16/p84 = "
                 f"{np.nanpercentile(vals,16):+.2f}/{np.nanpercentile(vals,84):+.2f}")
    ax.grid(alpha=0.3)

    # 2. Per-level median ratio (top 30 abs levels by Σn_NLTE)
    ax = axes[0,1]
    top = per_lvl.head(30).reset_index()
    bars = np.arange(len(top))
    colors = ["crimson" if abs(x) > 0.3 else "steelblue" for x in top["median_log10_ratio"]]
    ax.barh(bars, top["median_log10_ratio"], color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(-0.3, color="orange", ls=":", lw=0.6); ax.axvline(0.3, color="orange", ls=":", lw=0.6)
    ax.set_yticks(bars)
    ax.set_yticklabels([f"l{int(i)} E={e:.2f}{'m' if m==1 else ''}"
                        for i,e,m in zip(top["level_idx"], top["E_eV"], top["meta"])],
                       fontsize=7)
    ax.set_xlabel("median log10(n_NLTE/n_neb) across shells")
    ax.set_title("Top 30 Fe II levels by Σshell n_NLTE\n(τ-contributing levels)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # 3. Scatter: n_neb vs n_NLTE, all Fe II rows
    ax = axes[1,0]
    valid = (fe2["n_pop_nlte"] > 0) & (fe2["n_neb"] > 0)
    x = fe2.loc[valid,"n_neb"].values
    y = fe2.loc[valid,"n_pop_nlte"].values
    ax.scatter(x, y, s=4, alpha=0.3, color="steelblue")
    lo = max(min(x.min(), y.min()), 1e-30)
    hi = max(x.max(), y.max()) * 2
    ax.plot([lo,hi],[lo,hi], color="k", ls="--", lw=0.8, label="1:1")
    ax.plot([lo,hi],[lo*2,hi*2], color="orange", ls=":", lw=0.6)
    ax.plot([lo,hi],[lo/2,hi/2], color="orange", ls=":", lw=0.6, label="2×")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n_neb [cm⁻³]")
    ax.set_ylabel("n_NLTE [cm⁻³]")
    ax.set_title("Fe II per-(shell,level): NLTE vs nebular n_lower")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 4. log10(ratio) vs E_eV (color = shell)
    ax = axes[1,1]
    sc = ax.scatter(fe2["E_eV"], fe2["log10_ratio"], s=6,
                    c=fe2["shell"], alpha=0.5, cmap="viridis")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.axhline(-0.3, color="orange", ls=":", lw=0.5); ax.axhline(0.3, color="orange", ls=":", lw=0.5)
    plt.colorbar(sc, ax=ax, label="shell")
    ax.set_xlabel("E_lower [eV]")
    ax.set_ylabel("log10(n_NLTE/n_neb)")
    ax.set_title("Fe II: NLTE/nebular departure vs lower energy")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = run_dir / "fe2_nlte_vs_nebular_diagnostic.png"
    plt.savefig(out_png, dpi=130)
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
