#!/usr/bin/env python3
"""Offline validation of the ORTHODOX conditioning recipe on the REAL raw NLTE
rate matrix (O II<->O III, cold Te, N=683), before wiring into the C solve.

Recipe under test (codex/TLUSTY-confirmed orthodox):
  1. departure-coeff transform  M = D^-1 A D,  D = diag(n*),  n* = TRUE
     cross-ion Saha-Boltzmann LTE pop  (NOT previous-iterate pops).
  2. zero-out genuinely DECOUPLED levels (n_i=0 identity row), NOT b=1 pin.
  3. (optional) final row-equilibration as numerical hygiene.

Inputs:
  bk2_raw_matrix.bin : [5 int hdr N,n_lo,Z,ion,shell][N*N f64 col-major A][N f64 b]
  bk2_nstar.bin      : [5 int hdr][3 f64 Te,n_e,chi][N int ionf][N f64 E][N f64 g][N f64 nstar]
"""
import numpy as np, struct, sys

def load_matrix(p):
    raw = open(p, 'rb').read()
    N, n_lo, Z, ion, shell = struct.unpack('<5i', raw[:20])
    o = 20
    A = np.frombuffer(raw[o:o+8*N*N], dtype='<f8').reshape(N, N, order='F').copy()
    b = np.frombuffer(raw[o+8*N*N:o+8*N*N+8*N], dtype='<f8').copy()
    return N, n_lo, A, b

def load_nstar(p):
    raw = open(p, 'rb').read()
    N, n_lo, Z, ion, shell = struct.unpack('<5i', raw[:20])
    Te, n_e, chi = struct.unpack('<3d', raw[20:44])
    o = 44
    ionf = np.frombuffer(raw[o:o+4*N], dtype='<i4').copy(); o += 4*N
    E = np.frombuffer(raw[o:o+8*N], dtype='<f8').copy(); o += 8*N
    g = np.frombuffer(raw[o:o+8*N], dtype='<f8').copy(); o += 8*N
    nst = np.frombuffer(raw[o:o+8*N], dtype='<f8').copy()
    return N, n_lo, Te, n_e, chi, ionf, E, g, nst

def cond(M):
    try: return np.linalg.cond(M)
    except Exception: return float('inf')

def solve_report(M, rhs, nst, tag):
    try:
        y = np.linalg.solve(M, rhs)          # departure coeffs b_k
        x = y * nst                          # populations
        neg = (x < 0).sum()
        print(f"  [{tag}] solve: b_k in [{y.min():.2e},{y.max():.2e}] "
              f"pops neg={neg}/{len(x)} ({100*neg/len(x):.1f}%) "
              f"x in [{x.min():.2e},{x.max():.2e}]")
        return y, x
    except Exception as e:
        print(f"  [{tag}] solve FAILED: {e}")
        return None, None

def main():
    Nm, nlo_m, A, b = load_matrix('bk2_raw_matrix.bin')
    Nn, nlo_n, Te, n_e, chi, ionf, E, g, nst = load_nstar('bk2_nstar.bin')
    assert Nm == Nn, f"N mismatch {Nm} vs {Nn}"
    N = Nm
    print(f"N={N} n_lo={nlo_m} Te={Te:.0f}K n_e={n_e:.3e} chi_lo={chi:.2f}eV")
    print(f"n* range: [{nst.min():.3e},{nst.max():.3e}]  span={nst.max()/nst.min():.2e} orders={np.log10(nst.max()/nst.min()):.0f}")
    print(f"raw matrix diag range [{np.abs(np.diag(A)).min():.2e},{np.abs(np.diag(A)).max():.2e}]")

    print(f"\n[0] RAW cond              = {cond(A):.3e}")

    # connectivity from raw off-diagonal magnitude
    off = np.abs(A - np.diag(np.diag(A)))
    rmax = off.max(axis=1)                    # strongest rate per row
    rmed = np.median(rmax[rmax > 0])

    # similarity transform M = D^-1 A D
    nst_s = np.where(nst > 0, nst, nst[nst > 0].min()*1e-3)
    D = nst_s
    M = (A * D[None, :]) / D[:, None]
    rhs = b / D
    print(f"[1] +departure transform  = {cond(M):.3e}   (D=Saha-Boltzmann n*)")
    solve_report(M, rhs, nst_s, "transform-only")

    # zero-out DECOUPLED levels: weak connectivity AND negligible n*
    for floor in (1e-30, 1e-20, 1e-15, 1e-10):
        dead = (rmax < 1e-6 * rmed)           # decoupled by rates
        tiny = (nst < floor * nst.max())      # negligible LTE pop
        zero = dead | tiny
        keep = ~zero
        Mz = M.copy(); rz = rhs.copy()
        for i in np.where(zero)[0]:
            Mz[i, :] = 0.0; Mz[i, i] = 1.0; rz[i] = 0.0
        # remove zeroed levels from any conservation row (col coeff -> 0)
        # conservation rows are the dense rows (>50% nonzero)
        nnz = (np.abs(M) > 0).sum(axis=1)
        cons_rows = np.where(nnz > 0.5*N)[0]
        for cr in cons_rows:
            if cr not in np.where(zero)[0]:
                Mz[cr, zero] = 0.0
        c1 = cond(Mz)
        # + row equilibration (hygiene)
        scale = np.abs(Mz).max(axis=1); scale[scale == 0] = 1.0
        Me = Mz / scale[:, None]; re = rz / scale
        c2 = cond(Me)
        print(f"[2] +zero-out floor={floor:.0e}: keep {keep.sum()}/{N}  "
              f"cond={c1:.3e}  +row-equil={c2:.3e}")
        if floor == 1e-15:
            solve_report(Me, re, nst_s, f"zero+equil floor1e-15")
    print("\nVERDICT: orthodox recipe works iff cond drops to <~1e10 AND pops >0.")

if __name__ == '__main__':
    main()
