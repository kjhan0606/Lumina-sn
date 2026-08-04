#!/usr/bin/env python3
"""H3: macro-atom fate attribution analysis (per-(Z, ion, entry_band, exit_band)).
Ranks species that drive UV->red and optical->red cascade flux. Compares the
no-bypass (eps=0.0) vs full-bypass (eps=0.9) cases to see how the gate
re-routes traffic.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
JOB = sys.argv[1] if len(sys.argv) > 1 else "PLACEHOLDER"

BAND_NAMES = ["UVblnk[1700-3000)","CaIIKb[3000-3300)","UVtgt[3300-3700)",
              "fluor[3700-4400)","green[4400-5500)","red[5500-7000)",
              "NIR1[7000-10000)","NIR2[>=10000)"]
RED_BANDS = {5, 6}              # 5500-10000 A
UV_BANDS  = {0, 1, 2}           # 1700-3700 A
OPT_BANDS = {3, 4}              # 3700-5500 A

ION_NAME = ["I","II","III","IV"]

def load_csv(p):
    if not p.exists(): return None
    return pd.read_csv(p, comment="#")

def species_label(row):
    return f"{row['Z_name']} {ION_NAME[int(row['ion'])]}"

def attribution(df, entry_set, exit_set, label):
    sel = df[df["entry_band"].isin(entry_set) & df["exit_band"].isin(exit_set)]
    if sel.empty:
        print(f"  ({label}: empty)"); return
    total = sel["count"].sum()
    by_sp = sel.groupby(["Z","Z_name","ion"])["count"].sum().reset_index()
    by_sp["pct"] = 100.0 * by_sp["count"] / total
    by_sp = by_sp.sort_values("count", ascending=False)
    print(f"\n  --- {label} (total events {total:,}) ---")
    print(f"  {'species':<10} {'count':>14} {'pct':>7}")
    for _, r in by_sp.head(12).iterrows():
        sp = f"{r['Z_name']} {ION_NAME[int(r['ion'])]}"
        print(f"  {sp:<10} {int(r['count']):>14,} {r['pct']:>6.1f}%")

print(f"\n=== H3 macro-atom fate attribution (job {JOB}) ===\n")

for eps in ("0.0", "0.9"):
    csv = ROOT/f"logs/ddc15H3_{JOB}_ddc15H3_epsUV{eps}_fate/ma_fate_zihist.csv"
    df = load_csv(csv)
    if df is None:
        print(f"\n[ε={eps}] MISSING {csv}")
        continue
    print(f"\n##############  EPS_UV={eps}  ##############")
    print(f"  csv  : {csv.relative_to(ROOT)}")
    print(f"  rows : {len(df):,}, total events: {int(df['count'].sum()):,}")
    # Headline: which (Z, ion) feed UV->red and optical->red
    attribution(df, UV_BANDS,  RED_BANDS, "UV entry -> red/NIR1 exit (cascade harm channel)")
    attribution(df, OPT_BANDS, RED_BANDS, "optical entry -> red/NIR1 exit")
    attribution(df, RED_BANDS, RED_BANDS, "red entry -> red exit (self-coupling, baseline)")
    # UV+blanket-only specifics
    attribution(df, {0},       RED_BANDS, "band 0 UV-blanket entry -> red exit")
    attribution(df, {0},       {3, 4},    "band 0 UV-blanket entry -> blue/green (useful)")
    # Per-band exit decomposition
    print(f"\n  --- band 0 (UV-blanket) entry: exit-band breakdown ---")
    e0 = df[df["entry_band"] == 0]
    total0 = e0["count"].sum()
    if total0 > 0:
        for b in range(8):
            n = e0[e0["exit_band"] == b]["count"].sum()
            print(f"    exit {b} {BAND_NAMES[b]:<22} {int(n):>12,} {100.0*n/total0:>5.1f}%")

print()
