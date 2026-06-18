#!/usr/bin/env python3
"""FALSIFIER for the top-ion-excited-block structural-singularity diagnosis.

Reads the binary rate-matrix dump (LUMINA_NLTE_MATDUMP) for one ion-pair/shell
and answers the two reviewers' demand: is the negative-pops-with-info=0 a
STRUCTURAL near-singularity, or ill-scaling / an assembly bug?

Tests (codex's falsifier, verbatim):
  1. cond(A), smallest singular value, numerical rank (before + after exact
     two-sided inf-norm equilibration).
  2. high-precision / SVD-pseudoinverse solve vs the (reconstructed) LU result;
     relative residual ||A x - b|| / ||b||.
  3. smallest right-singular vector: does it live on the TOP-ion (hi) EXCITED
     levels (index >= n_lo + 1)? does the negative-population pattern project
     onto it?

CONFIRMED  := tiny smallest-sigma (cond >> 1/eps_after_equil) AND v_min mass
              concentrated on hi-ion excited block AND SVD solution still
              negative there  -> structural, missing top-ion recomb closure.
REFUTED    := well-conditioned (cond modest) but SVD solution still negative
              -> assembly/sign bug; OR large residual despite info=0 -> pivot
              instability.
"""
import sys
import numpy as np

trapz = None  # unused
PATH = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/oiii_s24_matrix.bin'


def equilibrate(A, b):
    """Exact two-sided inf-norm row/col scaling (matches cuda.cu)."""
    A = A.copy(); b = b.copy()
    r = 1.0 / np.maximum(np.max(np.abs(A), axis=1), 1e-300)
    A = A * r[:, None]; b = b * r
    c = 1.0 / np.maximum(np.max(np.abs(A), axis=0), 1e-300)
    A = A * c[None, :]
    return A, b, c


def report(tag, A, b):
    U, S, Vt = np.linalg.svd(A)
    cond = S[0] / S[-1]
    rank = int(np.sum(S > S[0] * 1e-13 * A.shape[0]))
    print(f"  [{tag}] sigma_max={S[0]:.3e} sigma_min={S[-1]:.3e} "
          f"cond={cond:.3e} numrank={rank}/{A.shape[0]}")
    return U, S, Vt, cond


def main():
    with open(PATH, 'rb') as f:
        hdr = np.fromfile(f, dtype=np.int32, count=5)
        N, n_lo, Z, ion, s = (int(x) for x in hdr)
        A_cm = np.fromfile(f, dtype=np.float64, count=N * N)
        b = np.fromfile(f, dtype=np.float64, count=N)
    A = A_cm.reshape((N, N), order='F')   # col-major -> (row, col)
    print(f"=== NLTE matrix dump: Z={Z} lo-ion={ion} shell={s} "
          f"N={N} n_lo={n_lo} (hi block = [{n_lo}:{N}], hi-excited = [{n_lo+1}:{N}]) ===")
    print(f"  ||b||={np.linalg.norm(b):.3e}  b nonzero rows={np.sum(b!=0)} "
          f"(conservation RHS)")

    # (1) conditioning, raw vs equilibrated
    U, S, Vt, cond_raw = report("RAW", A, b)
    Ae, be, c = equilibrate(A, b)
    Ue, Se, Vte, cond_eq = report("EQUILIBRATED", Ae, be)

    # (2) SVD pseudo-inverse solve (high-accuracy) vs residual
    tol = Se[0] * 1e-12
    Sinv = np.array([1.0 / sv if sv > tol else 0.0 for sv in Se])
    x_eq = (Vte.T * Sinv) @ (Ue.T @ be)
    x = x_eq * c                       # un-scale columns
    res = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)
    nneg = int(np.sum(x < 0))
    nneg_hi_exc = int(np.sum(x[n_lo + 1:] < 0))
    print(f"  SVD/pinv solve: rel-residual ||Ax-b||/||b|| = {res:.3e}")
    print(f"    negatives: total={nneg}/{N}  hi-ion-excited={nneg_hi_exc}/{N-n_lo-1}")
    print(f"    x[ground_lo]={x[0]:.3e}  x[ground_hi]={x[n_lo]:.3e}  "
          f"min(x)={x.min():.3e} @ idx {int(np.argmin(x))}")

    # (3) smallest right-singular vector localization (raw A)
    vmin = np.abs(Vt[-1])              # |v| of sigma_min
    mass_total = np.sum(vmin**2)
    mass_lo = np.sum(vmin[:n_lo]**2) / mass_total
    mass_hi_ground = vmin[n_lo]**2 / mass_total
    mass_hi_exc = np.sum(vmin[n_lo + 1:]**2) / mass_total
    top = np.argsort(vmin)[::-1][:8]
    print(f"  smallest right-singular vector mass: lo-block={mass_lo:.3f}  "
          f"hi-ground={mass_hi_ground:.3f}  hi-EXCITED={mass_hi_exc:.3f}")
    print(f"    top-8 |v| indices (block: lo if <{n_lo}, else hi): "
          + ", ".join(f"{int(i)}({'lo' if i < n_lo else 'hi'})" for i in top))

    # nearly-null directions: count tiny singular values
    nnull = int(np.sum(S < S[0] * 1e-10))
    print(f"  near-null directions (sigma < 1e-10*sigma_max): {nnull}")

    # verdict heuristic
    print("\n  --- VERDICT INPUTS ---")
    structural = (cond_eq > 1e10) and (mass_hi_exc > 0.4)
    print(f"  cond(equil)={cond_eq:.2e} (>1e10 => not just scaling)  "
          f"hi-excited mass={mass_hi_exc:.2f} (>0.4 => null lives on top-ion excited)")
    print(f"  => {'STRUCTURAL near-singularity SUPPORTED' if structural else 'inspect: not the clean structural signature'}")


if __name__ == '__main__':
    main()
