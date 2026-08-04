/* lumina_cmf_solve.cu — GPU port of the comoving-frame formal solver cmf_solve_J
 * (src/lumina_cmfgen.c). Gate LUMINA_CMF_SOLVE_GPU (0=CPU/OMP, 1=GPU, 2=A/B check).
 *
 * The producer (cmfgen_fine_jbar / cmfgen_fine_emergent) is the FULL-WAVELENGTH
 * bottleneck: the fine mesh has NB~500k frequency bins and the CPU solver's
 * frequency loop is SEQUENTIAL (the blue->red comoving advection couples bin b
 * to bin b+1) with only the NP~70-80 rays OpenMP-parallel -> far too little
 * parallelism at NB=500k.
 *
 * PARALLELIZATION. The radial short-characteristic for a given (bin b, ray k) is
 * independent of all other rays AND of all other bins EXCEPT through (i) the ALI
 * scattering source S=S_fixed+(chi_es/chi_tot)*J (already lagged on J across ALI
 * iters in the CPU code) and (ii) the blue->red advection, which reads bin (b+1)'s
 * per-segment specific intensity Iin_p/Iout_p[k,seg]. We LAG the advection across
 * ALI iters exactly like the source: bin b at ALI iter t reads bin (b+1)'s
 * per-segment intensities from iter t-1. Then ALL (b,k) become independent -> one
 * kernel of NB*NP threads per ALI iter (~500k*80 = 40M threads). The lagged scheme
 * has the SAME fixed point as the sequential sweep (at convergence lag==cur, so the
 * per-segment intensities are mutually consistent and satisfy the identical
 * transfer equation); it only changes the iteration path.
 *
 * MEASURED (LUMINA_CMF_SOLVE_GPU=2 self-check): with the advection OFF
 * (LUMINA_CMF_ALAM=0, bins independent) the GPU matches the CPU to ~7e-16 in the
 * SAME ALI count -> the full NB*NP-thread parallelism is a clean win. With the
 * advection ON (default a_lam>0) the Courant number is large (adv*ds~10^2 at the
 * producer's fine resolution -> I_b ~= I_b+1), so the lagged field advances only
 * ~1 bin per ALI iter: it reaches the SAME fixed point but needs O(NB) iters
 * (NB=400 -> converged at iter 409 to 5.8e-6; impractical at NB=500k vs the ~24
 * budget). So this kernel is the solver for the static/transport-only limit; the
 * advection-coupled producer keeps the sequential CPU sweep. FP64 throughout.
 *
 * Kernels per ALI iter:
 *   k_march : thread (b,k) -> inbound+outbound radial short-char, writes the
 *             per-segment post-march intensities cur_in/cur_out[b,k,seg]. Reads the
 *             previous iter's cur as the lagged advection upstream (lag_in/lag_out).
 *   k_jint  : thread (s,b) -> integrates J[s,b] = Int 0.5(I+ + I-) dmu over the rays
 *             crossing shell s (precomputed mu-sorted sample list), trapezoid with a
 *             mu=0 endpoint, exactly mirroring the CPU mu-quadrature.
 *   k_update: J=Jnew + global max-relative-change reduction (ALI break test).
 * Ping-pong cur<->lag between iters. Uploaded ONCE per call: chi_tot/chi_es/chi_abs/
 * S_fixed, the precomputed Bin[b]/adv[b]/advcoef[b], the ray tables and the per-shell
 * mu-sorted sample tables. J stays resident across ALI iters; downloaded at the end.
 *
 * Reuses the CUDA_CHECK / extern "C" conventions of lumina_nlte_gemm.cu. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

extern "C" {
#include "lumina.h"
}

#define CUDA_CHECK(call) do {                                    \
    cudaError_t err = (call);                                    \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        return -1;                                               \
    }                                                            \
} while(0)

/* atomic max on double via CAS (no native FP64 atomicMax) */
__device__ static double atomicMaxD(double *addr, double val) {
    unsigned long long *a = (unsigned long long *)addr;
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        if (cur >= val) break;
        old = atomicCAS(a, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(*a);
}

/* ---- radial short-char march: one thread per (bin b, ray k) ---------------- */
/* Mirrors the inbound/outbound passes of cmf_solve_J line-by-line. cur_in/cur_out
 * are written with the per-segment post-march intensity (= the CPU ImL/IpL samples
 * AND next iter's lagged advection upstream). lag_in/lag_out hold the previous
 * iter's cur (bin b reads index b+1). */
__global__ void k_march(int NS, int NB, int NP, int adv_split, double a_lam,
                         const double *chi_tot, const double *chi_es,
                         const double *chi_abs, const double *S_fixed,
                         const double *J, const double *Bin,
                         const double *adv_b, const double *advcoef_b,
                         const int *rn, const int *rsh, const double *rz,
                         const int *rcore, const double *rzin,
                         const double *lag_in, const double *lag_out,
                         double *cur_in, double *cur_out)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)NB * NP;
    if (tid >= tot) return;
    int b = (int)(tid / NP);
    int k = (int)(tid % NP);
    int n = rn[k];
    size_t kb  = (size_t)k * (NS + 1);
    size_t seg0 = (size_t)b * NP * (NS + 1) + kb;     /* base of this (b,k) segment row */
    if (n == 0) return;

    int    has_blue = (b < NB - 1);
    double adv      = has_blue ? adv_b[b]     : 0.0;  /* a_lam*lam[b]/Dlam   */
    double advcoef  = has_blue ? advcoef_b[b] : 0.0;  /* a_lam*lam[b+1]/Dlam */
    size_t blue0    = has_blue ? ((size_t)(b + 1) * NP * (NS + 1) + kb) : 0;

    /* ---- inbound (outer -> tangent/core) ---- */
    double I = 0.0;
    for (int i = 0; i < n; ++i) {
        int s = rsh[kb + i];
        double ds = (i + 1 < n) ? (rz[kb + i] - rz[kb + i + 1]) : (rz[kb + i] - rzin[k]);
        size_t idx = (size_t)s * NB + b;
        double chi = chi_tot[idx];
        double rr  = (chi > 0.0) ? chi_es[idx] / chi : 0.0;
        double S   = S_fixed[idx] + rr * J[idx];
        if (adv_split) {
            double chi_rad = chi + a_lam * 4.0;
            double dtau = chi_rad * ds, ex = exp(-dtau), e0, e1, wu, wd;
            if (dtau > 1e-4) { e0 = 1 - ex; e1 = dtau - e0; wu = e0 - e1 / dtau; wd = e1 / dtau; }
            else             { wu = 0.5 * dtau; wd = 0.5 * dtau; }
            double I_rad = I * ex + (wu + wd) * S;
            double bds = adv * ds;
            double Iblue = has_blue ? lag_in[blue0 + i] : I_rad;
            I = (I_rad + bds * Iblue) / (1.0 + bds);
        } else {
            double chih = chi + a_lam * 4.0 + adv;
            double Su = (S * chi + (has_blue ? advcoef * lag_in[blue0 + i] : 0.0))
                        / (chih > 0 ? chih : 1.0);
            double dtau = chih * ds, ex = exp(-dtau), e0, e1, wu, wd;
            if (dtau > 1e-4) { e0 = 1 - ex; e1 = dtau - e0; wu = e0 - e1 / dtau; wd = e1 / dtau; }
            else             { wu = 0.5 * dtau; wd = 0.5 * dtau; }
            I = I * ex + wu * Su + wd * Su;
        }
        cur_in[seg0 + i] = I;
    }
    if (rcore[k]) I = Bin[b];
    /* ---- outbound (tangent/core -> outer) ---- */
    for (int i = n - 1; i >= 0; --i) {
        int s = rsh[kb + i];
        double ds = (i + 1 < n) ? (rz[kb + i] - rz[kb + i + 1]) : (rz[kb + i] - rzin[k]);
        size_t idx = (size_t)s * NB + b;
        double chi = chi_tot[idx];
        double rr  = (chi > 0.0) ? chi_es[idx] / chi : 0.0;
        double S   = S_fixed[idx] + rr * J[idx];
        if (adv_split) {
            double chi_rad = chi + a_lam * 4.0;
            double dtau = chi_rad * ds, ex = exp(-dtau), e0, e1, wu, wd;
            if (dtau > 1e-4) { e0 = 1 - ex; e1 = dtau - e0; wu = e0 - e1 / dtau; wd = e1 / dtau; }
            else             { wu = 0.5 * dtau; wd = 0.5 * dtau; }
            double I_rad = I * ex + (wu + wd) * S;
            double bds = adv * ds;
            double Iblue = has_blue ? lag_out[blue0 + i] : I_rad;
            I = (I_rad + bds * Iblue) / (1.0 + bds);
        } else {
            double chih = chi + a_lam * 4.0 + adv;
            double Su = (S * chi + (has_blue ? advcoef * lag_out[blue0 + i] : 0.0))
                        / (chih > 0 ? chih : 1.0);
            double dtau = chih * ds, ex = exp(-dtau), e0, e1, wu, wd;
            if (dtau > 1e-4) { e0 = 1 - ex; e1 = dtau - e0; wu = e0 - e1 / dtau; wd = e1 / dtau; }
            else             { wu = 0.5 * dtau; wd = 0.5 * dtau; }
            I = I * ex + wu * Su + wd * Su;
        }
        cur_out[seg0 + i] = I;
    }
}

/* ---- J integration: one thread per (shell s, bin b) ------------------------ */
/* For shell s, the contributing rays + their mu = rz/rmid are fixed by geometry;
 * shell_k/shell_seg/shell_mu hold them PRE-SORTED ascending in mu (offsets
 * shell_off[s]..shell_off[s+1]). Trapezoid 0.5(I+ + I-) over mu with a mu=0
 * endpoint flat-extrapolated from the smallest-mu sample (CPU mu-quadrature). */
__global__ void k_jint(int NS, int NB, int NP,
                        const double *chi_tot, const double *chi_es,
                        const double *S_fixed, const double *J,
                        const int *shell_off, const int *shell_k,
                        const int *shell_seg, const double *shell_mu,
                        const double *cur_in, const double *cur_out,
                        double *Jnew)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)NS * NB;
    if (tid >= tot) return;
    int s = (int)(tid / NB);
    int b = (int)(tid % NB);
    size_t idx = (size_t)s * NB + b;
    int o0 = shell_off[s], o1 = shell_off[s + 1], c = o1 - o0;
    if (c < 1) {
        double chi = chi_tot[idx];
        double rr  = (chi > 0.0) ? chi_es[idx] / chi : 0.0;
        Jnew[idx] = S_fixed[idx] + rr * J[idx];
        return;
    }
    size_t bbase = (size_t)b * NP * (NS + 1);
    /* mu=0 endpoint = jv of the smallest-mu sample (o0) */
    {
        int kk = shell_k[o0], ii = shell_seg[o0];
        size_t sg = bbase + (size_t)kk * (NS + 1) + ii;
        double jvf = 0.5 * (cur_out[sg] + cur_in[sg]);
        double prev_mu = 0.0, prev_jv = jvf, Js = 0.0;
        for (int a = o0; a < o1; ++a) {
            int k2 = shell_k[a], i2 = shell_seg[a];
            double mu = shell_mu[a];
            size_t sg2 = bbase + (size_t)k2 * (NS + 1) + i2;
            double jv = 0.5 * (cur_out[sg2] + cur_in[sg2]);
            Js += 0.5 * (prev_jv + jv) * (mu - prev_mu);
            prev_mu = mu; prev_jv = jv;
        }
        Jnew[idx] = Js;
    }
}

/* ---- J update + max-relative-change reduction ------------------------------ */
__global__ void k_update(long n, const double *Jnew, double *J, double *d_maxrel)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    double jo = J[tid], jn = Jnew[tid];
    double d = fabs(jn - jo) / (fabs(jo) + 1e-30);
    J[tid] = jn;
    atomicMaxD(d_maxrel, d);
}

/* ============================================================================
 * Host entry. Returns 0 on success (-1 on CUDA/alloc failure -> caller falls back
 * to CPU). On success J[] holds the converged field; *iters_out = ALI iters used.
 * ============================================================================ */
extern "C" int cmf_solve_J_gpu(
    int NS, int NB, int NP, int adv_split, double a_lam,
    const double *chi_tot, const double *chi_es, const double *chi_abs,
    const double *S_fixed, double *J,                 /* J in/out [NS*NB] */
    const double *Bin, const double *adv_b, const double *advcoef_b,  /* [NB] */
    const int *rn, const int *rsh, const double *rz,
    const int *rcore, const double *rzin,
    const int *shell_off, const int *shell_k, const int *shell_seg,
    const double *shell_mu, int nsamp,
    int n_ali_iter, double tol, int *iters_out)
{
    size_t cell  = (size_t)NS * NB;
    size_t segsz = (size_t)NP * (NS + 1) * NB;        /* per per-segment buffer */
    size_t seg1  = (size_t)NP * (NS + 1);

    /* memory guard: 4 segment buffers dominate at NB~500k (each NP*(NS+1)*NB*8) */
    size_t need = (4 * segsz + 6 * cell) * sizeof(double);
    size_t freeB = 0, totB = 0;
    if (cudaMemGetInfo(&freeB, &totB) == cudaSuccess && need > freeB) {
        fprintf(stderr, "[cmf_gpu] need %.1f GB > free %.1f GB on device -> CPU fallback\n",
                need / 1e9, freeB / 1e9);
        return -1;
    }

    double *d_chi_tot, *d_chi_es, *d_chi_abs, *d_S_fixed, *d_J, *d_Jnew;
    double *d_Bin, *d_adv, *d_advc, *d_rz, *d_rzin, *d_shell_mu;
    int    *d_rn, *d_rsh, *d_rcore, *d_shell_off, *d_shell_k, *d_shell_seg;
    double *d_in0, *d_in1, *d_out0, *d_out1, *d_maxrel;

    CUDA_CHECK(cudaMalloc(&d_chi_tot, cell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_chi_es,  cell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_chi_abs, cell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_S_fixed, cell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_J,       cell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Jnew,    cell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Bin,  NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_adv,  NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_advc, NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rn,    NP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rsh,   seg1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rz,    seg1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rcore, NP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rzin,  NP * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_shell_off, (NS + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_shell_k,   nsamp * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_shell_seg, nsamp * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_shell_mu,  nsamp * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_in0,  segsz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_in1,  segsz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out0, segsz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out1, segsz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maxrel, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_chi_tot, chi_tot, cell * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chi_es,  chi_es,  cell * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chi_abs, chi_abs, cell * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S_fixed, S_fixed, cell * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_J,       J,       cell * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bin,  Bin,      NB * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adv,  adv_b,    NB * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_advc, advcoef_b,NB * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rn,    rn,    NP * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rsh,   rsh,   seg1 * sizeof(int),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rz,    rz,    seg1 * sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rcore, rcore, NP * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rzin,  rzin,  NP * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shell_off, shell_off, (NS + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shell_k,   shell_k,   nsamp * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shell_seg, shell_seg, nsamp * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shell_mu,  shell_mu,  nsamp * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_in0,  0, segsz * sizeof(double)));   /* lag=0 at iter 0 */
    CUDA_CHECK(cudaMemset(d_in1,  0, segsz * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_out0, 0, segsz * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_out1, 0, segsz * sizeof(double)));

    double *lag_in = d_in0, *cur_in = d_in1, *lag_out = d_out0, *cur_out = d_out1;
    int TPB = 256;
    long n_march = (long)NB * NP, n_jint = (long)NS * NB;
    int blk_march = (int)((n_march + TPB - 1) / TPB);
    int blk_jint  = (int)((n_jint  + TPB - 1) / TPB);
    int blk_upd   = (int)((cell    + TPB - 1) / TPB);

    int used = n_ali_iter;
    for (int it = 0; it < n_ali_iter; ++it) {
        k_march<<<blk_march, TPB>>>(NS, NB, NP, adv_split, a_lam,
            d_chi_tot, d_chi_es, d_chi_abs, d_S_fixed, d_J, d_Bin, d_adv, d_advc,
            d_rn, d_rsh, d_rz, d_rcore, d_rzin, lag_in, lag_out, cur_in, cur_out);
        k_jint<<<blk_jint, TPB>>>(NS, NB, NP, d_chi_tot, d_chi_es, d_S_fixed, d_J,
            d_shell_off, d_shell_k, d_shell_seg, d_shell_mu, cur_in, cur_out, d_Jnew);
        CUDA_CHECK(cudaMemset(d_maxrel, 0, sizeof(double)));
        k_update<<<blk_upd, TPB>>>(cell, d_Jnew, d_J, d_maxrel);
        double maxrel = 0.0;
        CUDA_CHECK(cudaMemcpy(&maxrel, d_maxrel, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());
        if (maxrel < tol && it > 0) { used = it + 1; break; }
        /* ping-pong: this iter's cur becomes next iter's lagged upstream */
        double *t1 = lag_in;  lag_in  = cur_in;  cur_in  = t1;
        double *t2 = lag_out; lag_out = cur_out; cur_out = t2;
    }
    if (iters_out) *iters_out = used;

    CUDA_CHECK(cudaMemcpy(J, d_J, cell * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_chi_tot); cudaFree(d_chi_es); cudaFree(d_chi_abs); cudaFree(d_S_fixed);
    cudaFree(d_J); cudaFree(d_Jnew); cudaFree(d_Bin); cudaFree(d_adv); cudaFree(d_advc);
    cudaFree(d_rn); cudaFree(d_rsh); cudaFree(d_rz); cudaFree(d_rcore); cudaFree(d_rzin);
    cudaFree(d_shell_off); cudaFree(d_shell_k); cudaFree(d_shell_seg); cudaFree(d_shell_mu);
    cudaFree(d_in0); cudaFree(d_in1); cudaFree(d_out0); cudaFree(d_out1); cudaFree(d_maxrel);
    return 0;
}
