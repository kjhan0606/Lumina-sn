#!/usr/bin/env python3
"""build_cmfgen_jtable.py -- #33 GRADIENT-TRANSPLANT diagnostic table builder.

Reads the self-run CMFGEN toy06 @19.48d radiation field (EDDFACTOR = J_nu directly)
and re-samples it onto LUMINA's Gph photoionization frequency grid, producing a binary
table J_nu[50 shells x 1000 bins] that the C gate LUMINA_GPH_JTABLE injects into the
Gph rate integral (surgical ionization-causality probe).

Grid (authoritative source src/lumina.h:328-330, set in src/lumina_plasma.c:8857-8860):
    NLTE_N_FREQ_BINS = 1000
    NLTE_NU_MIN      = 1.5e14 Hz  (c/20000 A)
    NLTE_NU_MAX      = 3.0e16 Hz  (c/100 A)
    d_log_nu = log(NLTE_NU_MAX/NLTE_NU_MIN)/1000
    bin bb: edges [nu_min*exp(bb*dln), nu_min*exp((bb+1)*dln)], center exp((bb+0.5)*dln)
    (the exact loop the Gph integral uses at src/lumina_plasma.c:5294-5296 etc.)

Mapping:
  * velocity: CMFGEN's 90 RVTJ depths (1025..35975 km/s) -> Lumina's 50 shell mid-
    velocities (data/tardis_reference_toy06_19p48d/geometry.csv, 4264..39936 km/s);
    log-interpolate J in velocity per CMFGEN frequency; Lumina shells beyond the
    outermost CMFGEN depth (35975 km/s) HOLD the outermost CMFGEN value.
  * frequency: bin-average CMFGEN's fine grid (166k freqs) into Lumina's log-nu bins;
    Lumina bins outside CMFGEN's frequency coverage get 0 (and the C side SKIPs 0 bins,
    leaving the run's own field there).

Reader logic copied from toy06_19.48d_jnu4/extract_jnu.py (validated aba40a30) -- the
original is NOT modified.

Output: data/cmfgen_jtable_toy06_19p48d.bin  (int32 magic,ver,nshells,nfb + f64 grid)
        data/cmfgen_jtable_toy06_19p48d.json (sidecar metadata + sanity numbers)

Sanity gates (ABORT on failure): band-avg J(918-1290A) s0~2.0e-4, s8~7.7e-7 (2.4-dex
decline); EUV(300-450A) s0/s8 decline ~6-7 dex.
"""
import sys, os, json, argparse
import numpy as np

_trapz = getattr(np, 'trapezoid', np.trapz)   # numpy>=2 renamed trapz->trapezoid

CLIGHT_A  = 2997.92458        # lam_A = CLIGHT_A / FL(1e15 Hz)
CLIGHT_A_HZ = 2.99792458e18   # lam_A = CLIGHT_A_HZ / nu(Hz)

# Lumina Gph grid (must match src/lumina.h + lumina_plasma.c exactly)
NLTE_NU_MIN = 1.5e14
NLTE_NU_MAX = 3.0e16
NFB         = 1000
NSHELLS     = 50

MAGIC = 0x4A544142            # 'JTAB' -- must match the C loader
VERSION = 1

FUV_LO, FUV_HI = 918.0, 1290.0    # dominant all-level Gph band
EUV_LO, EUV_HI = 300.0, 450.0     # ground-threshold ionizing band

# ---- reader (copied from extract_jnu.py) ------------------------------------
def read_info(info):
    L = open(info).read().splitlines()
    v = L[2].split()
    return dict(ND=int(v[0]), RECL=int(v[1]), WORD=int(v[2]), little=(v[5] == 'T'))

def read_eddfactor(edd):
    info = read_info(edd + '_INFO')
    ND = info['ND']; nwr = info['RECL'] // info['WORD']   # ND+1
    dt = '<f8' if info['little'] else '>f8'
    raw = np.fromfile(edd, dtype=dt)
    n = (raw.size // nwr) * nwr
    raw = raw[:n].reshape(-1, nwr)
    finish = raw[4, 0]
    data = raw[14:]                                       # records 15..end
    good = np.isfinite(data[:, :ND]).all(axis=1) & (data[:, ND] > 0)
    J = data[good, :ND]                                   # [nfreq, ND] erg/cm2/s/Hz/sr
    FL = data[good, ND]                                   # 1e15 Hz
    return J, FL, ND, finish, int((~good).sum()), data.shape[0]

def parse_rvtj_block(text, label, ND):
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == label:
            vals = []; j = i + 1
            while j < len(lines) and len(vals) < ND:
                toks = lines[j].split()
                try:
                    vals += [float(t) for t in toks]
                except ValueError:
                    break
                j += 1
            return np.array(vals[:ND])
    raise KeyError(label)

# ---- Lumina geometry --------------------------------------------------------
def lumina_shell_velocities(geom_csv):
    import csv
    rows = list(csv.DictReader(open(geom_csv)))
    mids = np.array([(float(r['v_inner']) + float(r['v_outer'])) / 2.0 / 1e5
                     for r in rows])            # cm/s -> km/s
    return mids

# ---- velocity log-interpolation (vectorized over all freqs) -----------------
def interp_logJ_at_v(logJ_perm, Vasc, v):
    """logJ_perm[nfreq, ND] = log(J) with columns ascending in V (nan where J<=0).
    Returns J[nfreq] at velocity v; clamp to endpoints (hold outermost beyond max V)."""
    nd = Vasc.size
    if v <= Vasc[0]:
        out = logJ_perm[:, 0].copy()
    elif v >= Vasc[-1]:
        out = logJ_perm[:, -1].copy()          # HOLD outermost CMFGEN depth
    else:
        j = int(np.searchsorted(Vasc, v) - 1)
        j = max(0, min(j, nd - 2))
        w = (v - Vasc[j]) / (Vasc[j + 1] - Vasc[j])
        lo = logJ_perm[:, j]; hi = logJ_perm[:, j + 1]
        both = np.isfinite(lo) & np.isfinite(hi)
        out = np.where(np.isfinite(lo), lo, hi).copy()   # one-sided fallback
        out[both] = (1.0 - w) * lo[both] + w * hi[both]
    return np.exp(out)                          # nan stays nan -> no CMFGEN data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edd',  default='/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR')
    ap.add_argument('--rvtj', default='/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ')
    ap.add_argument('--geom', default='data/tardis_reference_toy06_19p48d/geometry.csv')
    ap.add_argument('--out',  default='data/cmfgen_jtable_toy06_19p48d.bin')
    a = ap.parse_args()

    # --- read CMFGEN field + depths ---
    J, FL, ND, finish, nbad, ntot = read_eddfactor(a.edd)
    nu_cmf = FL * 1e15                               # Hz
    lam_cmf = CLIGHT_A / FL                          # A
    print(f"[edd] {a.edd}: ND={ND} nfreq={J.shape[0]} nbad={nbad}/{ntot} FINISH_REC={finish}")
    print(f"[edd] CMFGEN freq coverage: {nu_cmf.min():.3e}..{nu_cmf.max():.3e} Hz "
          f"= {lam_cmf.min():.1f}..{lam_cmf.max():.1f} A")
    if not np.isfinite(finish) or finish == 0:
        print("  WARNING: FINISH_REC==0/NaN -> field may be incomplete (non-converged "
              "4-iter snapshot; relative gradient is the diagnostic target).")

    V = parse_rvtj_block(open(a.rvtj).read(), 'Velocity (km/s)', ND)
    T = parse_rvtj_block(open(a.rvtj).read(), 'Temperature (10^4K)', ND) * 1e4
    print(f"[rvtj] V={V[0]:.0f}..{V[-1]:.0f} km/s  T={T[0]:.0f}..{T[-1]:.0f} K")

    # sort CMFGEN by nu ascending; build ascending-V permutation of log(J)
    fo = np.argsort(nu_cmf)
    nu_s  = nu_cmf[fo]
    J_s   = J[fo, :]                                 # [nfreq, ND]
    dv    = np.argsort(V)                            # ascending V (deep->outer)
    Vasc  = V[dv]
    Jperm = J_s[:, dv]                               # [nfreq, ND] cols ascending V
    logJ_perm = np.where(Jperm > 0.0, Jperm, np.nan)
    logJ_perm = np.log(logJ_perm)

    nu_cov_min, nu_cov_max = nu_s.min(), nu_s.max()

    # --- Lumina Gph grid ---
    dln = np.log(NLTE_NU_MAX / NLTE_NU_MIN) / NFB
    bb = np.arange(NFB)
    lo_edge = NLTE_NU_MIN * np.exp(bb * dln)
    hi_edge = NLTE_NU_MIN * np.exp((bb + 1) * dln)
    nu_ctr  = NLTE_NU_MIN * np.exp((bb + 0.5) * dln)
    lam_ctr = CLIGHT_A_HZ / nu_ctr

    vmids = lumina_shell_velocities(a.geom)
    assert vmids.size == NSHELLS, f"geometry has {vmids.size} shells, expected {NSHELLS}"
    print(f"[geom] Lumina shell mids: s0={vmids[0]:.0f} s8={vmids[8]:.0f} "
          f"s49={vmids[-1]:.0f} km/s; beyond {Vasc[-1]:.0f}: shells "
          f"{list(np.where(vmids > Vasc[-1])[0])}")

    # precompute which CMFGEN fine freqs fall in each Lumina bin (coverage geometry
    # is velocity-independent -> compute once)
    bin_idx = np.searchsorted(lo_edge, nu_s, side='right') - 1   # bin owning each nu_s
    covered = (hi_edge > nu_cov_min) & (lo_edge < nu_cov_max)    # bins overlapping cov.

    # --- build the table: interp-in-velocity (per fine freq) then bin-average ---
    table = np.zeros((NSHELLS, NFB), dtype=np.float64)
    n_interp_fallback = 0
    for s in range(NSHELLS):
        Jfine = interp_logJ_at_v(logJ_perm, Vasc, vmids[s])      # [nfreq] at shell v
        fin = np.isfinite(Jfine)
        # bin-average (dnu-weighted trapz) over fine freqs in each bin
        row = np.zeros(NFB)
        # accumulate per bin using np.add.at on finite points
        for b in np.unique(bin_idx[fin]):
            if b < 0 or b >= NFB:
                continue
            m = fin & (bin_idx == b)
            xk = nu_s[m]; yk = Jfine[m]
            if xk.size >= 2:
                row[b] = _trapz(yk, xk) / (xk[-1] - xk[0])
            else:
                row[b] = yk[0]
        # covered bins with no fine point: log-interp J at bin center
        empty_cov = covered & (row <= 0.0)
        if empty_cov.any():
            xs = nu_s[fin]; ys = np.log(Jfine[fin])
            row[empty_cov] = np.exp(np.interp(nu_ctr[empty_cov], xs, ys))
            n_interp_fallback += int(empty_cov.sum())
        row[~covered] = 0.0
        table[s, :] = row

    # ---- SANITY GATES ----------------------------------------------------------
    # (a) extract-faithful reproduction (validates the reader independently of the
    #     table build): geometric-mean band-avg over the FINE grid at s0,s8.
    def extract_band_avg(v, lo_A, hi_A):
        Jf = interp_logJ_at_v(logJ_perm, Vasc, v)
        m = (lam_cmf[fo] >= lo_A) & (lam_cmf[fo] <= hi_A) & np.isfinite(Jf)
        return np.exp(np.nanmean(np.log(Jf[m])))
    ef_s0_fuv = extract_band_avg(vmids[0], FUV_LO, FUV_HI)
    ef_s8_fuv = extract_band_avg(vmids[8], FUV_LO, FUV_HI)
    ef_s0_euv = extract_band_avg(vmids[0], EUV_LO, EUV_HI)
    ef_s8_euv = extract_band_avg(vmids[8], EUV_LO, EUV_HI)
    fuv_decl = np.log10(ef_s0_fuv / ef_s8_fuv)
    euv_decl = np.log10(ef_s0_euv / ef_s8_euv)

    # (b) table-derived band-avg (validates the built table carries the gradient)
    def table_band_avg(s, lo_A, hi_A):
        m = (lam_ctr >= lo_A) & (lam_ctr <= hi_A) & (table[s] > 0.0)
        return np.exp(np.nanmean(np.log(table[s][m]))) if m.any() else 0.0
    tb_s0_fuv = table_band_avg(0, FUV_LO, FUV_HI)
    tb_s8_fuv = table_band_avg(8, FUV_LO, FUV_HI)
    tb_fuv_decl = np.log10(tb_s0_fuv / tb_s8_fuv) if tb_s8_fuv > 0 else float('nan')
    n_fuv_bins = int(((lam_ctr >= FUV_LO) & (lam_ctr <= FUV_HI)).sum())
    n_euv_bins = int(((lam_ctr >= EUV_LO) & (lam_ctr <= EUV_HI)).sum())

    print("\n=== SANITY GATES ===")
    print(f"[extract-faithful] FUV(918-1290A) band-avg  s0={ef_s0_fuv:.3e}  "
          f"s8={ef_s8_fuv:.3e}  decline={fuv_decl:+.3f} dex  (target s0~2.0e-4 "
          f"s8~7.7e-7, 2.41 dex)")
    print(f"[extract-faithful] EUV(300-450A) band-avg    s0={ef_s0_euv:.3e}  "
          f"s8={ef_s8_euv:.3e}  decline={euv_decl:+.3f} dex  (target ~6-7 dex)")
    print(f"[table-derived]    FUV band-avg              s0={tb_s0_fuv:.3e}  "
          f"s8={tb_s8_fuv:.3e}  decline={tb_fuv_decl:+.3f} dex  "
          f"({n_fuv_bins} FUV bins, {n_euv_bins} EUV bins)")
    print(f"[table] nonzero cells={int((table>0).sum())}/{NSHELLS*NFB}  "
          f"covered bins/shell~{int((table[0]>0).sum())}  "
          f"interp-fallback fills={n_interp_fallback}")

    fails = []
    if not (1.0e-4 <= ef_s0_fuv <= 4.0e-4):
        fails.append(f"FUV s0 band-avg {ef_s0_fuv:.3e} out of [1e-4,4e-4]")
    if not (2.0 <= fuv_decl <= 2.9):
        fails.append(f"FUV decline {fuv_decl:+.3f} out of [2.0,2.9] dex")
    if not (euv_decl >= 5.0):
        fails.append(f"EUV decline {euv_decl:+.3f} < 5.0 dex")
    if fails:
        print("\n[ABORT] sanity gate FAILED:")
        for f in fails:
            print("   - " + f)
        sys.exit(1)
    print("[OK] all sanity gates passed.")

    # ---- WRITE ----------------------------------------------------------------
    out = a.out
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'wb') as f:
        np.array([MAGIC, VERSION, NSHELLS, NFB], dtype='<i4').tofile(f)
        table.astype('<f8').tofile(f)
    print(f"\n[out] wrote {out} ({os.path.getsize(out)} bytes; "
          f"header 16 B + {NSHELLS}x{NFB}x8)")

    meta = dict(
        source_edd=a.edd, source_rvtj=a.rvtj, source_geom=a.geom,
        finish_rec=float(finish), cmfgen_ND=int(ND),
        cmfgen_nu_cov_hz=[float(nu_cov_min), float(nu_cov_max)],
        cmfgen_lam_cov_A=[float(CLIGHT_A_HZ/nu_cov_max), float(CLIGHT_A_HZ/nu_cov_min)],
        grid=dict(nshells=NSHELLS, nfb=NFB, nu_min=NLTE_NU_MIN, nu_max=NLTE_NU_MAX,
                  d_log_nu=dln),
        shell_velocities_kms=[float(x) for x in vmids],
        hold_outermost_beyond_kms=float(Vasc[-1]),
        shells_held=[int(i) for i in np.where(vmids > Vasc[-1])[0]],
        sanity=dict(
            extract_fuv_s0=float(ef_s0_fuv), extract_fuv_s8=float(ef_s8_fuv),
            extract_fuv_decline_dex=float(fuv_decl),
            extract_euv_s0=float(ef_s0_euv), extract_euv_s8=float(ef_s8_euv),
            extract_euv_decline_dex=float(euv_decl),
            table_fuv_s0=float(tb_s0_fuv), table_fuv_s8=float(tb_s8_fuv),
            table_fuv_decline_dex=float(tb_fuv_decl),
            n_fuv_bins=n_fuv_bins, n_euv_bins=n_euv_bins,
            nonzero_cells=int((table>0).sum()),
            interp_fallback_fills=n_interp_fallback),
        magic=MAGIC, version=VERSION,
        note="#33 gradient-transplant: inject CMFGEN J_nu(v) into Lumina Gph "
             "integral only (LUMINA_GPH_JTABLE). Diagnostic, not a production fix.")
    jout = os.path.splitext(out)[0] + '.json'
    with open(jout, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[out] wrote {jout}")

if __name__ == '__main__':
    main()
