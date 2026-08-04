#!/usr/bin/env python3
"""Phase A: 4-cell NLTE × σ_bf ablation analyzer.

Reads `nlte_levels_iter*.csv` from each cell and computes
  ratio_NLTE_LTE = n_NLTE / n_LTE_Boltzmann
where n_LTE = n_ion_total · g · exp(-E/kT_e) / Z_partition.

Special focus on Si II 4p²P° and Ca II 4p²P° upper levels. Saves a
combined diagnostic table + plot.

Usage: python analyze_phaseA_ablation.py <jobid>
"""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_BOLTZMANN_eV = 8.617333e-5  # eV/K

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
JOBID = sys.argv[1] if len(sys.argv) > 1 else "154184"
CELLS = ["NN", "NF", "FN", "FF"]
LABELS = {"NN": "NLTE+σ_bf", "NF": "NLTE only",
          "FN": "σ_bf only (LTE)", "FF": "baseline (LTE)"}

# Si II 4p²P° = level energies ~10.07 eV (²P°₁/₂) and 10.07 eV (²P°₃/₂)
# These are the upper of Si II 6347/6371 (= 6355 blend)
# Ca II 4p²P° = ~3.12 eV (²P°₁/₂) and ~3.15 eV (²P°₃/₂) — upper of IR triplet
TARGETS = [
    ("Si II 4p²P°", 14, 1, 9.5, 10.5),
    ("Ca II 4p²P°", 20, 1, 3.0, 3.3),
    ("Fe II low excitation", 26, 1, 0.0, 1.5),
]

def latest_dump(cell_dir):
    files = sorted(glob.glob(str(cell_dir / "nlte_levels_iter*.csv")))
    return Path(files[-1]) if files else None

def load_cell(cell_dir):
    f = latest_dump(cell_dir)
    if f is None:
        return None
    df = pd.read_csv(f)
    return df

def compute_lte_ratio(df):
    """Add n_LTE column via Saha-Boltzmann.
    n_LTE(level i, shell s) = n_ion(s) · g_i · exp(-E_i/kT_e(s)) / Z_part(ion, s)
    Z_part = Σ_j g_j · exp(-E_j/kT_e(s))
    """
    out = df.copy()
    # group by (Z, ion, shell) to compute partition function
    out["boltzmann"] = out["g"] * np.exp(-out["E_eV"] / (K_BOLTZMANN_eV * out["T_e"]))
    Z_part = out.groupby(["Z","ion","shell"])["boltzmann"].transform("sum")
    out["n_LTE"] = out["n_ion_total"] * out["boltzmann"] / Z_part.replace(0, np.nan)
    out["ratio"] = out["n_pop"] / out["n_LTE"]
    return out

def main():
    print(f"=== Phase A ablation analysis (job {JOBID}) ===\n")
    cells_data = {}
    for cell in CELLS:
        cell_dir = ROOT / f"logs/phaseA_ablate_{cell}_{JOBID}"
        if not cell_dir.is_dir():
            print(f"  {cell}: dir missing ({cell_dir})")
            continue
        df = load_cell(cell_dir)
        if df is None:
            print(f"  {cell}: no nlte_levels dump (NLTE off cell — expected for FN/FF)")
            continue
        df = compute_lte_ratio(df)
        cells_data[cell] = df
        print(f"  {cell} ({LABELS[cell]}): {len(df)} rows from {latest_dump(cell_dir).name}")

    if "NN" not in cells_data:
        print("\nERROR: NN cell missing — cannot continue.")
        return 1

    print(f"\n{'cell':<6s} {'target':<22s} {'shell':>5s} {'levelE':>8s} "
          f"{'g':>4s} {'n_NLTE':>11s} {'n_LTE':>11s} {'ratio':>7s}")
    print("-"*100)

    rows = []
    for cell in ["NN", "NF"]:
        if cell not in cells_data:
            continue
        df = cells_data[cell]
        for name, Z, ion, E_lo, E_hi in TARGETS:
            sub = df[(df["Z"]==Z) & (df["ion"]==ion) &
                     (df["E_eV"]>=E_lo) & (df["E_eV"]<=E_hi) &
                     (df["shell"]==0)]
            if sub.empty:
                continue
            for _, r in sub.iterrows():
                print(f"{cell:<6s} {name:<22s} {int(r['shell']):>5d} "
                      f"{r['E_eV']:>8.3f} {int(r['g']):>4d} "
                      f"{r['n_pop']:>11.3e} {r['n_LTE']:>11.3e} {r['ratio']:>7.3f}")
                rows.append({"cell": cell, "target": name,
                             "shell": int(r["shell"]),
                             "E_eV": r["E_eV"], "g": int(r["g"]),
                             "n_pop": r["n_pop"], "n_LTE": r["n_LTE"],
                             "ratio": r["ratio"]})

    out_csv = ROOT / f"data/phaseA_ablation_{JOBID}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Spectrum comparison
    print(f"\n{'cell':<6s} {'spectrum present':<20s}")
    for cell in CELLS:
        f = ROOT / f"logs/phaseA_ablate_{cell}_{JOBID}/lumina_spectrum_formal.csv"
        print(f"  {cell}: {'YES' if f.exists() else 'NO ':<3s}  ({f})")

    # Plot Si II / Ca II ratio across shells (NN vs NF)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(5*len(TARGETS), 4))
    if len(TARGETS) == 1: axes = [axes]
    for ax, (name, Z, ion, E_lo, E_hi) in zip(axes, TARGETS):
        for cell in ["NN", "NF"]:
            if cell not in cells_data: continue
            df = cells_data[cell]
            sub = df[(df["Z"]==Z) & (df["ion"]==ion) &
                     (df["E_eV"]>=E_lo) & (df["E_eV"]<=E_hi)]
            if sub.empty: continue
            agg = sub.groupby("shell")["ratio"].mean()
            ax.plot(agg.index, agg.values, "-o", label=LABELS[cell], lw=1.5)
        ax.axhline(1.0, color="k", ls=":", lw=0.5)
        ax.set_xlabel("shell"); ax.set_ylabel("⟨n_NLTE / n_LTE⟩ (level-mean)")
        ax.set_title(f"{name}\nE∈[{E_lo}, {E_hi}] eV", fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Phase A NLTE/LTE ratio — job {JOBID}", fontweight="bold")
    plt.tight_layout()
    out_png = ROOT / f"figures/phaseA_ablation_{JOBID}.png"
    plt.savefig(out_png, dpi=140); plt.close()
    print(f"Wrote {out_png}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
