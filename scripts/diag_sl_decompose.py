#!/usr/bin/env python3
"""FAITHFUL in-run decomposition of super-thermal S_l (job 166121).

S_l/B(Te) = (S_l/J_line) * (J_line/B(Te))     [exact, same run]

  - J_line/B >> 1  AND  S_l/J_line ~ 1  -> S_l super-thermal because the
      (binned, non-local) FIELD the line sees is hot. Root = field/escape:
      the 2-level source faithfully tracks a hot J. (codex+claude locus)
  - J_line/B ~ 1   AND  S_l/J_line >> 1 -> upper level OVER-FED beyond the
      local field. For an isolated 2-level atom with J>B, collisions only
      push S_l DOWN toward B, so S_l>J_line REQUIRES multi-level cascade
      (fluorescence) or a pumping bug.  Decides A-vs-B.
"""
import sys, numpy as np, pandas as pd

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
DUMP = sys.argv[1] if len(sys.argv)>1 else \
    f"{ROOT}/logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_166121/lumina_sl_vs_B.csv"

d = pd.read_csv(DUMP)
print(f"rows={len(d)}  cols={list(d.columns)}")
d = d[(d.Sl>0)&(d.Jline>0)&(d.B_Te>0)].copy()
d["Sl_over_J"] = d.Sl/d.Jline
d["J_over_B"]  = d.Jline/d.B_Te
# sanity: Sl_over_B should == Sl_over_J * J_over_B
chk = np.nanmedian(np.abs(d.Sl_over_B - d.Sl_over_J*d.J_over_B)/d.Sl_over_B)
print(f"identity check |Sl/B - (Sl/J)(J/B)|/(Sl/B) median = {chk:.2e}\n")

def stat(name, x):
    x = x[np.isfinite(x)&(x>0)]
    return (f"{name:14s} N={len(x):7d}  median={np.median(x):11.3g}  "
            f"p25={np.percentile(x,25):10.3g}  p75={np.percentile(x,75):10.3g}")

bands = [("UV     <4000A", d[d.lambda_A<4000]),
         ("OPT 4000-7000", d[(d.lambda_A>=4000)&(d.lambda_A<7000)]),
         ("NIR 7000-12000",d[(d.lambda_A>=7000)&(d.lambda_A<12000)]),
         ("ALL          ", d)]
for name, sub in bands:
    print(f"### {name}   (lines x shells = {len(sub)})")
    print("   "+stat("S_l/B(Te)",  sub.Sl_over_B.values))
    print("   "+stat("S_l/J_line", sub.Sl_over_J.values))
    print("   "+stat("J_line/B(Te)",sub.J_over_B.values))
    # fraction where field explains it (Sl~J) vs overfeed (Sl>>J)
    r = sub.Sl_over_J.values; r=r[np.isfinite(r)&(r>0)]
    print(f"   frac S_l/J in[0.3,3]={np.mean((r>0.3)&(r<3)):.2f}  "
          f"frac S_l/J>3={np.mean(r>3):.2f}  frac S_l/J<0.3={np.mean(r<0.3):.2f}\n")

# Which band carries the super-thermal flux excess? weight by tau (deeper lines
# imprint more). Report tau-weighted S_l/B per band.
print("tau-weighted median S_l/B (which band over-emits):")
for name, sub in bands[:3]:
    w = sub.tau.values; sb = sub.Sl_over_B.values
    m = np.isfinite(sb)&(sb>0)&np.isfinite(w)
    if m.sum():
        order = np.argsort(sb[m]); cw = np.cumsum(w[m][order])
        med = sb[m][order][np.searchsorted(cw, cw[-1]/2)]
        print(f"   {name}: tau-wtd median S_l/B = {med:.3g}")
