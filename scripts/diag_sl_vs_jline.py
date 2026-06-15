#!/usr/bin/env python3
"""Two non-destructive diagnostics for the super-thermal S_l question.

(1) Einstein-relation consistency of the DDC15 line data (Claude-agent caveat):
    A_ul == (2 h nu^3 / c^2) * B_ul   and   g_lo*B_lu == g_up*B_ul.
    (g not in line_list -> check the A/B ratio, which is g-independent.)
(2) S_l / J_line (codex decisive test): is the line source just tracking the
    local binned mean intensity (radiative-scattering limit) or is it pumped
    far above it (cascade/coupling overfeed)?  S_l from the S_l dump, J_line
    from the config-identical JDUMP (cross-run, same converged plasma).
"""
import sys, numpy as np, pandas as pd

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
C = 2.99792458e10; H = 6.62607015e-27
LL = f"{ROOT}/data/tardis_reference_ddc15_0p976d/line_list.csv"
SLDUMP = f"{ROOT}/logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_165958/lumina_sl_vs_B.csv"
JDUMP  = f"{ROOT}/logs/ddc15_pc_phase3_jnul0_radls1_linere1_ratio1.0_pi1_fz1_165878/lumina_cmfgen_jnu.csv"

def einstein_check():
    print("="*70)
    print("(1) EINSTEIN-RELATION CONSISTENCY  (A_ul vs (2hnu^3/c^2)B_ul)")
    print("="*70)
    ll = pd.read_csv(LL, usecols=["nu","B_lu","B_ul","A_ul","f_lu","f_ul"])
    nu = ll.nu.values; Bul = ll.B_ul.values; Aul = ll.A_ul.values; Blu = ll.B_lu.values
    pref = 2*H*nu**3/C**2
    m = (Bul>0)&(Aul>0)
    rA = Aul[m]/(pref[m]*Bul[m])              # should be ~1
    print(f"  N lines              : {len(nu)}")
    print(f"  A_ul/((2hnu^3/c^2)Bul): median={np.median(rA):.4f}  "
          f"p5={np.percentile(rA,5):.4f}  p95={np.percentile(rA,95):.4f}")
    # g_lo*B_lu = g_up*B_ul -> B_lu/B_ul = g_up/g_lo (integer ratio). Also
    # f_lu/f_ul = -g_up/g_lo (TARDIS sign convention). Cross-check the two.
    m2 = (Bul>0)&(Blu>0)&(ll.f_ul.values!=0)
    r_BB = (Blu[m2]/Bul[m2])
    r_ff = np.abs(ll.f_lu.values[m2]/ll.f_ul.values[m2])
    dev = np.abs(r_BB - r_ff)/np.maximum(r_ff,1e-30)
    print(f"  B_lu/B_ul vs |f_lu/f_ul| (both = g_up/g_lo): "
          f"median rel-dev={np.median(dev):.2e}  p95={np.percentile(dev,95):.2e}")
    bad = (rA<0.5)|(rA>2.0)
    print(f"  lines with A/B ratio outside [0.5,2]: {bad.sum()} "
          f"({100*bad.mean():.2f}%)")
    print("  -> if median~1 and dev~0: Einstein data SELF-CONSISTENT (rule out)\n")

def sl_vs_j():
    print("="*70)
    print("(2) S_l / J_line   (codex decisive test, cross-run config-identical)")
    print("="*70)
    sl = pd.read_csv(SLDUMP)
    jd = pd.read_csv(JDUMP)
    # JDUMP: per (shell,bin) nu,J. Build per-shell nu->J interpolator (log-log).
    sl = sl[sl.Sl>0].copy()
    sl["nu"] = C/(sl.lambda_A.values*1e-8)
    out = []
    for s, g in sl.groupby("shell"):
        d = jd[jd.shell==s].sort_values("nu")
        if len(d)<2: continue
        lnu = np.log(d.nu.values); lJ = np.log(np.clip(d.J.values,1e-99,None))
        Jline = np.exp(np.interp(np.log(g.nu.values), lnu, lJ))
        gg = g.copy(); gg["Jline"]=Jline; out.append(gg)
    a = pd.concat(out, ignore_index=True)
    a["Sl_over_J"] = a.Sl/np.clip(a.Jline,1e-99,None)
    # optical band 4000-7000 A
    opt = a[(a.lambda_A>4000)&(a.lambda_A<7000)]
    uv  = a[(a.lambda_A<4000)]
    for name, sub in [("OPTICAL 4000-7000A",opt),("UV <4000A",uv),("ALL",a)]:
        r = sub.Sl_over_J.values; r=r[np.isfinite(r)&(r>0)]
        b = sub.Sl_over_B.values; b=b[np.isfinite(b)&(b>0)]
        print(f"  [{name}]  N={len(sub)}")
        print(f"     S_l/J_line : median={np.median(r):.3f}  "
              f"p25={np.percentile(r,25):.3f}  p75={np.percentile(r,75):.3f}  "
              f"frac(0.3..3)={np.mean((r>0.3)&(r<3)):.2f}")
        print(f"     S_l/B(Te)  : median={np.median(b):.2f}")
    print("\n  INTERPRETATION:")
    print("   - S_l/J_line ~ 1  => S_l just tracks local binned J (scattering")
    print("       limit, no thermalization). Super-thermal S_l/B then means the")
    print("       binned J the line sees is hot/non-local. ROOT=escape weighting.")
    print("   - S_l/J_line >> 1 => upper level overfed beyond the field")
    print("       (cascade/coupling). ROOT=population coupling, not field.")

if __name__ == "__main__":
    einstein_check()
    sl_vs_j()
