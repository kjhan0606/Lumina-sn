"""Y2 step 2: is bfr the nu-weighted twin of J_raw?  (independent normalisation check)

ARTIS radfield.cc:850  Gamma_bf density  = sum(e*dist/nu) / (V*t*h)
ARTIS J estimator      J_raw[f]          = sum(e*dist)    / (4pi*V*t*dnu_f)
=> sigma*bfr[f]  ==  4pi*sigma*J_raw[f]*dnu_f/(h*nu_eff[f])  ==  pref[f]*J_raw[f]
   with nu_eff[f] the packet-weighted harmonic mean nu inside bin f.
Since the fine bins are log-spaced with dlog=5.3e-3, nu_eff must sit inside
[nu_lo, nu_hi] => the ratio  R[f] = sigma*bfr / (pref*J_raw) = nu_mid/nu_eff
must lie in [exp(-dlog/2), exp(+dlog/2)] = [0.99736, 1.00265].
Any excursion outside that window is a normalisation defect, not statistics.

FORMAL (clean complete run, iters 0-11); the per-shell block is iter 11.
"""
import os
import numpy as np
import pandas as pd
import y2_common as Y

J_raw = np.load(os.path.join(Y.OUT, "_cache_Jraw.npy"))
bfr = np.load(os.path.join(Y.OUT, "_cache_bfr.npy"))
cnt = np.load(os.path.join(Y.OUT, "_cache_cnt.npy"))
ni, ns, nb = J_raw.shape

# R = bfr * h * nu_mid / (4pi * J_raw * dnu)   (sigma cancels)
den = 4.0 * np.pi * J_raw * Y.DNU[None, None, :]
num = bfr * Y.H_PLANCK * Y.NU_MID[None, None, :]
m = (J_raw > 0) & (bfr > 0)
R = np.where(m, num / np.where(den > 0, den, 1.0), np.nan)

lo, hi = np.exp(-Y.DLOG / 2), np.exp(Y.DLOG / 2)
Rv = R[m]
print(f"[bfr-identity] paired bins = {m.sum()}  (of {R.size})")
print(f"  allowed window from log-bin geometry: [{lo:.6f}, {hi:.6f}]")
print(f"  R = sigma*bfr/(pref*J_raw):  min={Rv.min():.6f} p1={np.percentile(Rv,1):.6f} "
      f"median={np.median(Rv):.6f} p99={np.percentile(Rv,99):.6f} max={Rv.max():.6f}")
out_of = int(((Rv < lo - 1e-6) | (Rv > hi + 1e-6)).sum())
print(f"  outside window: {out_of} bins ({100*out_of/Rv.size:.4f}%)")
print(f"  => the C2 estimator IS the raw-MC field integral to <{max(abs(1-lo),abs(hi-1))*100:.3f}% "
      f"in-bin nu weighting; it is NOT an independent physical quantity.")

# per-shell summary at the last iteration
rows = []
for s in range(ns):
    ms = m[-1, s]
    if ms.sum() == 0:
        continue
    r = R[-1, s][ms]
    rows.append(dict(shell=s, n=int(ms.sum()), Rmin=float(r.min()),
                     Rmed=float(np.median(r)), Rmax=float(r.max()),
                     n_out=int(((r < lo - 1e-6) | (r > hi + 1e-6)).sum())))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(Y.OUT, "y2_bfr_identity.csv"), index=False)
print("\nper-shell (iter 11 = last block):")
print(df.loc[df.shell.isin([0, 8, 20, 30, 45, 49])].round(6).to_string(index=False))
print(f"\nshells with any out-of-window bin: {int((df.n_out>0).sum())} / {len(df)}")
