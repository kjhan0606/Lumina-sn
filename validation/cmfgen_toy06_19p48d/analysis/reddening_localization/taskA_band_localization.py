#!/usr/bin/env python3
"""TASK A -- spectral localization of the deep-shell reddening.

Band-by-band comparison of the radiation energy density contribution
  u_band = (4pi/c) INT_band J_nu dnu   [erg/cm3]
between CMFGEN toy06 @19.48d (EDDFACTOR J_nu, direct CGS) and the Lumina B-run
(logs/coevolve_consume_a10_kx_gphall/lumina_coevolve_field.csv, mc_J), at the
deep/mid shells s0(v=4264), s2(v=5720), s4(v=7176).

Method:
  CMFGEN: EDDFACTOR reader (validated, extract_jnu.py:22-35). Per-frequency
    log10(J) linear-interp in velocity to the target v (same as extract_jnu:93-98),
    vectorised as a 2-point interp on the shared depth grid. Gated on reproducing
    the FUV band-geo anchors s0=2.023e-4, s8=7.729e-7 to <1%.
  Lumina: mc_J on its native 1000-bin grid, per shell 0/2/4 directly.
  Both integrated in nu (trapz) over identical band edges.

Deliverables (this dir):
  taskA_band_table.csv     -- per (shell,band): u_cmfgen, u_lumina, ratio, fractions
  taskA_overlay_spectrum.csv -- per-bin lambda, J_cmfgen, J_lumina, ratio (for plots)
  taskA_crossing.csv       -- ratio=1 crossing wavelength per shell
No source edits, no commit. Read-only on /gpfs and logs/.
"""
import numpy as np, csv, os

RUN   = "/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"
EDD   = RUN + "/EDDFACTOR"
RVTJ  = RUN + "/RVTJ"
FIELD = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_gphall/lumina_coevolve_field.csv"
OUT   = os.path.dirname(os.path.abspath(__file__))

CLIGHT_A = 2997.92458          # lam_A = CLIGHT_A / FL(1e15 Hz)
C_CM  = 2.99792458e10
C_A   = 2.99792458e18          # A/s
FOURPI_OVER_C = 4*np.pi/C_CM

# task shells (field shell index == task s-index; verified via T_e=13120 @ shell0)
SHELLS = [(0, 4264, 's0'), (2, 5720, 's2'), (4, 7176, 's4')]

# band edges (A). 100 = Lumina blue limit; 19933 = Lumina red limit.
# CMFGEN covers 3..8.6e5 A, so the 19933-25000 tail is CMFGEN-only (noted separately).
EDGES = [100, 300, 450, 918, 1290, 2000, 3000, 4500, 7000, 10000, 19933]
BAND_LABELS = ['soft_100_300', 'EUV_300_450', 'xuv_450_918', 'FUV_918_1290',
               'NUV_1290_2000', 'UV_2000_3000', 'blue_3000_4500', 'opt_4500_7000',
               'red_7000_10000', 'NIR_10000_19933']

# ---------- CMFGEN EDDFACTOR reader (extract_jnu.py:17-35) ----------
def read_info(info):
    L = open(info).read().splitlines(); v = L[2].split()
    return dict(ND=int(v[0]), RECL=int(v[1]), WORD=int(v[2]), little=(v[5] == 'T'))

def read_eddfactor(edd):
    info = read_info(edd + '_INFO')
    ND = info['ND']; nwr = info['RECL'] // info['WORD']
    dt = '<f8' if info['little'] else '>f8'
    raw = np.fromfile(edd, dtype=dt); n = (raw.size // nwr) * nwr
    raw = raw[:n].reshape(-1, nwr)
    data = raw[14:]
    good = np.isfinite(data[:, :ND]).all(axis=1) & (data[:, ND] > 0)
    J = data[good, :ND]; FL = data[good, ND]
    return J, FL, ND

def parse_rvtj_block(text, label, ND):
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == label:
            vals = []; j = i + 1
            while j < len(lines) and len(vals) < ND:
                toks = lines[j].split()
                try: vals += [float(t) for t in toks]
                except ValueError: break
                j += 1
            return np.array(vals[:ND])
    raise KeyError(label)

J, FL, ND = read_eddfactor(EDD)
nu_c = FL * 1e15                       # Hz (descending)
lam_c = CLIGHT_A / FL                  # A
rt = open(RVTJ).read()
V = parse_rvtj_block(rt, 'Velocity (km/s)', ND)
dv = np.argsort(V); Vasc = V[dv]       # ascending velocity
Jv = J[:, dv]                          # [nfreq, ND] reordered ascending V

def cmf_J_at_v(vt):
    """per-frequency log10(J) linear-interp in velocity to vt (vectorised 2-pt)."""
    idx = np.searchsorted(Vasc, vt)
    idx = np.clip(idx, 1, ND - 1)
    v0, v1 = Vasc[idx - 1], Vasc[idx]
    f = (vt - v0) / (v1 - v0)
    a = Jv[:, idx - 1]; b = Jv[:, idx]
    aa = np.where(a > 0, a, np.nan); bb = np.where(b > 0, b, np.nan)
    lg = (1 - f) * np.log10(aa) + f * np.log10(bb)
    Jt = 10 ** lg
    return np.where(np.isfinite(Jt), Jt, 0.0)

# ---------- anchor gate ----------
def band_geo(vt, lo=918., hi=1290.):
    Jt = cmf_J_at_v(vt); m = (lam_c >= lo) & (lam_c <= hi) & (Jt > 0)
    return np.exp(np.nanmean(np.log(Jt[m])))
a0, a8 = band_geo(4264), band_geo(10088)
print(f"[ANCHOR gate] CMFGEN FUV geo-mean  s0={a0:.4e} (t 2.023e-4, {a0/2.023e-4:.3f}x)  "
      f"s8={a8:.4e} (t 7.729e-7, {a8/7.729e-7:.3f}x)")
assert 0.98 < a0/2.023e-4 < 1.02 and 0.98 < a8/7.729e-7 < 1.02, "ANCHOR FAIL"
print("[ANCHOR gate] PASS\n")

# ---------- Lumina field reader ----------
def load_field(path):
    d = {}
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            s = int(row[0]); lam = float(row[2]); mc = float(row[4])
            d.setdefault(s, []).append((lam, mc))
    out = {}
    for s, rows in d.items():
        a = np.array(rows); o = np.argsort(a[:, 0])
        out[s] = (a[o, 0], a[o, 1])       # lam ascending, mc_J
    return out
LF = load_field(FIELD)

# Lumina FUV band arithmetic-mean anchor (VERDICT: 5.809e-6 @ s0)
l0lam, l0mc = LF[0]
mfuv = (l0lam >= 918) & (l0lam <= 1290)
print(f"[ANCHOR gate] Lumina FUV arith-mean s0={l0mc[mfuv].mean():.4e} (t 5.81e-6)\n")

# ---------- band integrator: u_band = (4pi/c) INT J dnu ----------
def band_u(lam_A, J_lam, lo, hi):
    """integrate J over [lo,hi] A in nu (trapz). endpoints interpolated in lam."""
    m = (lam_A >= lo) & (lam_A <= hi)
    if m.sum() < 2:
        return 0.0
    lam_b = lam_A[m]; Jb = J_lam[m]
    nu = C_A / lam_b
    o = np.argsort(nu); nu = nu[o]; Jb = Jb[o]
    return FOURPI_OVER_C * np.trapz(Jb, nu)

def total_u(lam_A, J_lam, lo=100., hi=19933.):
    m = (lam_A >= lo) & (lam_A <= hi)
    lam_b = lam_A[m]; Jb = J_lam[m]; nu = C_A / lam_b
    o = np.argsort(nu)
    return FOURPI_OVER_C * np.trapz(Jb[o], nu[o])

# ---------- per-shell band table ----------
rows = []
print("# per-band u contribution (erg/cm3) and Lumina/CMFGEN ratio")
for sidx, vt, slab in SHELLS:
    Jc = cmf_J_at_v(vt)                         # CMFGEN J at target v (full grid)
    lamc, Jcm = lam_c, Jc
    laml, Jlm = LF[sidx]
    tot_c = total_u(lamc, Jcm); tot_l = total_u(laml, Jlm)
    print(f"\n## {slab} (v={vt}): total u(100-19933A) CMFGEN={tot_c:.3e}  "
          f"Lumina={tot_l:.3e}  ratio={tot_l/tot_c:.3f} ({np.log10(tot_l/tot_c):+.2f} dex)")
    print(f"   {'band':>16} {'lo':>6} {'hi':>6} {'u_cmf':>10} {'u_lum':>10} "
          f"{'ratio':>7} {'dex':>6} {'f_cmf':>6} {'f_lum':>6}")
    for k in range(len(BAND_LABELS)):
        lo, hi = EDGES[k], EDGES[k + 1]
        uc = band_u(lamc, Jcm, lo, hi); ul = band_u(laml, Jlm, lo, hi)
        rat = ul / uc if uc > 0 else float('nan')
        dex = np.log10(rat) if (uc > 0 and ul > 0) else float('nan')
        fc = uc / tot_c; fl = ul / tot_l
        print(f"   {BAND_LABELS[k]:>16} {lo:>6} {hi:>6} {uc:>10.3e} {ul:>10.3e} "
              f"{rat:>7.3f} {dex:>+6.2f} {fc:>6.3f} {fl:>6.3f}")
        rows.append([slab, vt, BAND_LABELS[k], lo, hi, uc, ul, rat, dex, fc, fl, tot_c, tot_l])
    # CMFGEN-only red tail 19933-25000 (Lumina has no data there)
    uc_tail = band_u(lamc, Jcm, 19933., 25000.)
    print(f"   [CMFGEN-only tail 19933-25000A: u={uc_tail:.3e} (Lumina grid ends at 19933)]")
    rows.append([slab, vt, 'CMFtail_19933_25000', 19933, 25000, uc_tail, 0.0,
                 float('nan'), float('nan'), uc_tail/tot_c, 0.0, tot_c, tot_l])

with open(f"{OUT}/taskA_band_table.csv", 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['shell', 'v_kms', 'band', 'lo_A', 'hi_A', 'u_cmfgen', 'u_lumina',
                'ratio_L_over_C', 'dex', 'frac_cmfgen', 'frac_lumina',
                'tot_u_cmfgen_100_19933', 'tot_u_lumina_100_19933'])
    w.writerows(rows)
print(f"\n[out] {OUT}/taskA_band_table.csv")

# ---------- overlay spectrum + crossing wavelength ----------
# resample CMFGEN onto the Lumina grid (log-interp J in lambda) for a clean ratio.
cross_rows = []
ov_rows = []
for sidx, vt, slab in SHELLS:
    Jc = cmf_J_at_v(vt)
    # CMFGEN lam ascending
    oc = np.argsort(lam_c); lc = lam_c[oc]; jc = Jc[oc]
    laml, Jlm = LF[sidx]
    good = jc > 0
    jc_i = 10 ** np.interp(laml, lc[good], np.log10(jc[good]))   # CMFGEN on Lumina grid
    ratio = np.where(jc_i > 0, Jlm / jc_i, np.nan)
    for lam, jl, jcx, r in zip(laml, Jlm, jc_i, ratio):
        ov_rows.append([slab, vt, f"{lam:.3f}", f"{jl:.6e}", f"{jcx:.6e}", f"{r:.6e}"])
    # crossing: find lambda where ratio crosses 1 (blue<1 -> red>1). scan blue->red.
    lr = np.log10(ratio)
    valid = np.isfinite(lr)
    lam_v = laml[valid]; lr_v = lr[valid]
    order = np.argsort(lam_v); lam_v = lam_v[order]; lr_v = lr_v[order]
    crossings = []
    for i in range(1, len(lr_v)):
        if lr_v[i - 1] == 0:
            crossings.append(lam_v[i - 1]); continue
        if lr_v[i - 1] * lr_v[i] < 0:
            # linear interp in lambda for lr=0
            lam_x = lam_v[i - 1] + (0 - lr_v[i - 1]) * (lam_v[i] - lam_v[i - 1]) / (lr_v[i] - lr_v[i - 1])
            crossings.append(lam_x)
    print(f"[crossing] {slab}: ratio=1 at lambda = "
          + (", ".join(f"{c:.0f}A" for c in crossings) if crossings else "none"))
    cross_rows.append([slab, vt, ";".join(f"{c:.1f}" for c in crossings)])

with open(f"{OUT}/taskA_overlay_spectrum.csv", 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['shell', 'v_kms', 'wavelength_A', 'J_lumina', 'J_cmfgen_interp', 'ratio_L_over_C'])
    w.writerows(ov_rows)
with open(f"{OUT}/taskA_crossing.csv", 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['shell', 'v_kms', 'ratio1_crossings_A'])
    w.writerows(cross_rows)
print(f"[out] {OUT}/taskA_overlay_spectrum.csv")
print(f"[out] {OUT}/taskA_crossing.csv")
