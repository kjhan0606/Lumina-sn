#!/usr/bin/env python3
"""extract_jnu.py -- TASK A deliverable: J_nu(918-1290A) at the Lumina forming-shell
velocities from a CMFGEN EDDFACTOR (which stores J_nu directly; comp_j_blank.f:249/412/808).

Reader validated end-to-end (subagent aba40a30): EDDFACTOR = direct-access unformatted,
RECL/ND from EDDFACTOR_INFO; each record = J(1:ND) then FL (freq in 1e15 Hz); J data begins
at Fortran record 15 (0-based 14); rec5 (0-based 4) = FINISH_REC (nonzero=complete).
lam_A = 2997.92458 / FL_1e15. Sanity: INT J dnu -> sigma*T^4/pi at depth.
"""
import sys, argparse
import numpy as np

CLIGHT_A = 2997.92458  # lam_A = CLIGHT_A / FL(1e15 Hz)
TARGET_V = [4264, 5720, 7176, 7904, 8632, 9360, 10088, 10816, 11544]  # Lumina s0..s10 km/s
BAND_LO, BAND_HI = 918.0, 1290.0

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
    FL = data[good, ND]                                   # 1e15 Hz, descending
    lam = CLIGHT_A / FL
    return J, FL, lam, ND, finish, int((~good).sum()), data.shape[0]

def parse_rvtj_block(text, label, ND):
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == label:
            vals = []
            j = i + 1
            while j < len(lines) and len(vals) < ND:
                toks = lines[j].split()
                try:
                    vals += [float(t) for t in toks]
                except ValueError:
                    break
                j += 1
            return np.array(vals[:ND])
    raise KeyError(label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edd', required=True)
    ap.add_argument('--rvtj', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    J, FL, lam, ND, finish, nbad, ntot = read_eddfactor(a.edd)
    print(f"[edd] {a.edd}: ND={ND} nfreq_good={J.shape[0]} nbad={nbad}/{ntot} FINISH_REC={finish}")
    if not np.isfinite(finish) or finish == 0:
        print("  WARNING: FINISH_REC==0 or NaN -> field may be incomplete/mid-write.")

    rt = open(a.rvtj).read()
    V = parse_rvtj_block(rt, 'Velocity (km/s)', ND)       # descending with depth idx
    T = parse_rvtj_block(rt, 'Temperature (10^4K)', ND) * 1e4
    print(f"[rvtj] V[0..-1]={V[0]:.0f}..{V[-1]:.0f} km/s  T[0..-1]={T[0]:.0f}..{T[-1]:.0f} K")

    # sanity: INT J dnu vs sigma T^4/pi at innermost depth
    order = np.argsort(FL)
    nu_hz = FL[order] * 1e15
    Jin = J[order, ND - 1]
    intJ = np.trapz(Jin, nu_hz)
    sigSB = 5.670374419e-5
    print(f"[sanity] innermost depth: INT J dnu={intJ:.3e}  sigma*T^4/pi={sigSB*T[-1]**4/np.pi:.3e}  ratio={intJ/(sigSB*T[-1]**4/np.pi):.3f}")

    # band mask
    bm = (lam >= BAND_LO) & (lam <= BAND_HI)
    lam_b = lam[bm]; Jb = J[bm]                            # [nband, ND]
    ob = np.argsort(lam_b)
    lam_b = lam_b[ob]; Jb = Jb[ob]
    print(f"[band] {BAND_LO}-{BAND_HI} A: {lam_b.size} freq points")

    # depth order by velocity ascending for interpolation
    dv = np.argsort(V)                                    # ascending V
    Vasc = V[dv]

    rows = []; band_avg = {}
    for vt in TARGET_V:
        # per-wavelength interpolate log10(J) in velocity
        Jvt = np.empty(lam_b.size)
        for k in range(lam_b.size):
            yk = Jb[k, dv]
            yk = np.where(yk > 0, yk, np.nan)
            lyk = np.log10(yk)
            gk = np.isfinite(lyk)
            Jvt[k] = 10 ** np.interp(vt, Vasc[gk], lyk[gk])
        band_avg[vt] = np.exp(np.nanmean(np.log(Jvt)))    # geometric mean over band
        for k in range(lam_b.size):
            rows.append((vt, lam_b[k], Jvt[k]))

    with open(a.out, 'w') as f:
        f.write("velocity_km_s,wavelength_A,J_nu\n")
        for vt, lm, jv in rows:
            f.write(f"{vt},{lm:.4f},{jv:.6e}\n")
    print(f"[out] wrote {len(rows)} rows -> {a.out}")

    print("\n=== band-averaged (geometric mean) J_nu(918-1290A) per forming shell ===")
    print("  v(km/s)   shell   J_nu_bandavg")
    labels = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    for lb, vt in zip(labels, TARGET_V):
        print(f"  {vt:7d}   {lb:>4s}   {band_avg[vt]:.4e}")
    r = band_avg[TARGET_V[0]] / band_avg[TARGET_V[-1]]
    print(f"\n>>> J_nu(s0={TARGET_V[0]}) / J_nu(s10={TARGET_V[-1]}) = {r:.3f}  = {np.log10(r):+.3f} dex")
    print(f">>> (deep v~4-6k vs photosphere v~9-12k; positive dex = deep field HOTTER/BRIGHTER)")

if __name__ == '__main__':
    main()
