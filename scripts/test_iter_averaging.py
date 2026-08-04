#!/usr/bin/env python3
"""FREE TEST (no new run): does averaging the per-iteration T_e over the last K
iterations reduce the outer shell-to-shell scatter as ~1/sqrt(K)?  If yes, the
spike is iteration-independent MC shot noise that "feedback/averaging" kills
(at K x cost). If RMS plateaus, the per-iteration noise is correlated and
averaging won't help -> deterministic (CMFGEN) field needed."""
import numpy as np, csv, glob, os

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
LOG  = f"{ROOT}/logs/ddc15_radeq_blanket_164032"
REF  = f"{ROOT}/data/tardis_reference_ddc15_0p976d"

def rd(p):
    r = csv.reader(open(p)); h = next(r); c = {k: [] for k in h}
    for row in r:
        for k, v in zip(h, row): c[k].append(float(v))
    return {k: np.array(v) for k, v in c.items()}

# CMFGEN reference gas temperature
ref_te = rd(f"{REF}/plasma_state.csv")["T_rad"]
geo = rd(f"{REF}/geometry.csv")
v = 0.5 * (geo["v_inner"] + geo["v_outer"]) / 1e5
o = slice(24, 49)                       # outer MC-starved shells
r = ref_te[o]; nsh = len(ref_te)

# Extract per-iteration T_e[shell] from the nlte_levels dumps (T_e is col index 8,
# constant per shell, repeated on every level row).
files = sorted(glob.glob(f"{LOG}/nlte_levels_iter*.csv"))
print(f"found {len(files)} iteration dumps")
Te_iter = []                            # list of arrays Te[shell]
for f in files:
    Te = np.full(nsh, np.nan)
    with open(f) as fh:
        rd2 = csv.reader(fh); hdr = next(rd2)
        si = hdr.index("shell"); ti = hdr.index("T_e")
        for row in rd2:
            sh = int(row[si])
            if np.isnan(Te[sh]): Te[sh] = float(row[ti])
    Te_iter.append(Te)
    it = os.path.basename(f).split("iter")[1].split(".")[0]
    rms = np.sqrt(np.mean(((Te[o]-r)/r)**2))*100
    print(f"  iter{it}: outer mean={np.mean(Te[o]):6.0f}K  RMS={rms:5.1f}%")
Te_iter = np.array(Te_iter)             # (n_iter, n_shell)

# --- 1/sqrt(K) test: cumulative average over the LAST K converged iterations ---
# Drop the first transient iterations; treat the rest as independent samples.
n_iter = len(Te_iter)
burn = max(0, n_iter - 6)               # use up to last 6 as "converged" samples
conv = Te_iter[burn:]
print(f"\nusing last {len(conv)} iterations as converged samples")
print("K  avg-over-last-K  outer-RMS%  (expect ~RMS1/sqrt(K) if independent noise)")
rms1 = None
for K in range(1, len(conv)+1):
    avg = np.mean(conv[-K:], axis=0)
    rms = np.sqrt(np.mean(((avg[o]-r)/r)**2))*100
    if rms1 is None: rms1 = rms
    expect = rms1/np.sqrt(K)
    print(f"  K={K}  RMS={rms:5.1f}%   ideal-1/sqrt(K)={expect:5.1f}%")

# also: temporal jitter of a single far-outer shell across iterations
sh = 48
print(f"\nshell {sh} T_e trace across iters: " +
      " ".join(f"{t:.0f}" for t in Te_iter[:, sh]))
print(f"  std across converged iters = {np.std(conv[:, sh]):.0f}K "
      f"(mean {np.mean(conv[:, sh]):.0f}K)")
