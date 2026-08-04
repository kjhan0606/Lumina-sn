#!/usr/bin/env python3
"""[IUP-JBLUE] pre-registered verdict battery (sbatch_kpkt_jblue.sh predictions 1-7).
Usage: jbl_verdict.py <run_dir> [<baseline_dir>]
Metrics computed identically for run and baseline (default = iupb 176699 dir) so
the comparison is apples-to-apples regardless of band-definition choices."""
import csv, os, sys
import numpy as np

BASE = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
run = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/logs/coevolve_consume_a10_kx_iupb"

def load_mc(d):
    wl, fl = [], []
    p = f"{d}/lumina_spectrum_coevolve_mc.csv"
    if not os.path.exists(p): return None, None
    for row in csv.reader(open(p)):
        if not row or row[0].startswith("wave"): continue
        try: w, x = float(row[0]), float(row[1])
        except ValueError: continue
        wl.append(w); fl.append(x)
    o = np.argsort(wl)
    return np.array(wl)[o], np.array(fl)[o]

def load_artis():
    wl, fx = [], []
    for line in open(f"{BASE}/data/standart_data1/toy06/spectra_toy06_artis.txt"):
        if line.startswith("#"): continue
        p = line.split(); wl.append(float(p[0])); fx.append(float(p[77]))
    return np.array(wl), np.array(fx)

def bands(wl, fl):
    # Definitions calibrated against the pre-registered iupb numbers in
    # sbatch_kpkt_jblue.sh: blue 32.1 / redNIR 36.8 (of 2500-10000A) and
    # spike 1.6 (4300-4600 of 2000-20000A).
    def integ(a, b):
        m = (wl >= a) & (wl < b)
        return float(np.trapz(fl[m], wl[m])) if m.sum() > 1 else 0.0
    t1 = integ(2500, 10000); t2 = integ(2000, 20000)
    if t1 <= 0 or t2 <= 0: return {}
    return {"UV 2500-3000/t1": integ(2500, 3000)/t1, "blue 3000-5000/t1": integ(3000, 5000)/t1,
            "green 5000-6500/t1": integ(5000, 6500)/t1, "redNIR 6500-1e4/t1": integ(6500, 10000)/t1,
            "spike 4300-4600/t2": integ(4300, 4600)/t2}

def corr_vs_artis(wl, fl):
    awl, afx = load_artis()
    grid = np.linspace(2500, 10000, 750)
    L = np.interp(grid, wl, fl); A = np.interp(grid, awl, afx)
    Ln = L/np.trapz(L, grid); An = A/np.trapz(A, grid)
    return float(np.corrcoef(Ln, An)[0, 1])

def ion_fII(d, Z, shells):
    p = f"{d}/lumina_ion_pops.csv"
    if not os.path.exists(p): return {}
    pops = {}
    for row in csv.DictReader(open(p)):
        try:
            z = int(row["Z"]); s = int(row["shell_id"])
            st = int(row["stage"]); n = float(row["n_ion"])
        except (ValueError, KeyError): continue
        if z == Z: pops.setdefault(s, {})[st] = n
    out = {}
    for s in shells:
        st = pops.get(s, {})
        tot = sum(st.values())
        out[s] = st.get(1, 0.0)/tot if tot > 0 else float("nan")
    return out

def kromer_share(d):
    p = f"{d}/lumina_kromer_coevolve.csv"
    if not os.path.exists(p): return {}
    tot = 0.0; by = {}; si2_abs = 0.0
    for row in csv.DictReader(open(p)):
        try:
            Z = int(row["emit_Z"]); ion = int(row["emit_ion"])
            e = float(row["energy"])
            inZ = int(row["in_Z"]); inion = int(row["in_ion"])
        except (ValueError, KeyError): continue
        tot += e; by[(Z, ion)] = by.get((Z, ion), 0.0) + e
        if inZ == 14 and inion == 1: si2_abs += e
    if tot > 0: by[("SiII", "absorbed")] = si2_abs
    if tot <= 0: return {}
    return {k: v/tot for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:8]}

def euv_ratio(d, shell=15):
    p = f"{d}/lumina_coevolve_field.csv"
    if not os.path.exists(p): return float("nan")
    num = den = 0.0
    for row in csv.DictReader(open(p)):
        try:
            s = int(row["shell"]); lam = float(row["wavelength_A"])
            cs = float(row["cs_J"]); mc = float(row["mc_J"])
        except (ValueError, KeyError): continue
        if s == shell and 531 <= lam <= 758:
            num += mc; den += cs
    return num/den if den > 0 else float("nan")

def jblue_counters(d):
    p = f"{d}/stdout.log"
    if not os.path.exists(p): return []
    return [l.strip() for l in open(p, errors="ignore") if "[IUP-JBLUE]" in l][:20]

for tag, d in [("RUN", run), ("REF(iupb)", ref)]:
    print(f"\n===== {tag}: {d} =====")
    wl, fl = load_mc(d)
    if wl is None:
        print("  (no MC emergent csv)"); continue
    b = bands(wl, fl)
    for k, v in b.items(): print(f"  {k:18} {100*v:6.1f}%")
    print(f"  corr(MC,ARTIS)     {corr_vs_artis(wl, fl):.3f}")
    print(f"  EUV s15 mc/cs      {euv_ratio(d):.2f}")
    for el, Z in [("Si", 14), ("S", 16)]:
        f2 = ion_fII(d, Z, [6, 9, 15, 25])
        print(f"  {el} f(II) " + "  ".join(f"s{s}={v:.3f}" for s, v in f2.items()))
    ks = kromer_share(d)
    if ks:
        print("  Kromer top emitters: " + "  ".join(
            (f"SiII_abs={100*v:.1f}%" if z == "SiII" else f"Z{z}ion{i}={100*v:.1f}%")
            for (z, i), v in ks.items()))
cl = jblue_counters(run)
print("\n===== [IUP-JBLUE] counters (RUN) =====")
print("\n".join(cl) if cl else "  (none found — gate arm FAILED?)")
