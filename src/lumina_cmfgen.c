/* ============================================================ */
/* lumina_cmfgen.c                                             */
/*                                                              */
/* Pure-CMFGEN deterministic radiation field (see header).      */
/* Coexists with the MC path; nothing here is reachable unless   */
/* LUMINA_PURE_CMFGEN=1 dispatches cmfgen_run() from main.       */
/* ============================================================ */
#include "lumina_cmfgen.h"
#include "physics_comparison.h"
#include "line_jbar.h"
#include "line_net_rate.h"
#include "cmf_exact_sliding.h"
#ifdef LUMINA_HAS_CUDA_BF_GEMM
#include "cmf_exact_multigpu.h"
#endif
#ifndef CMF_MGPU_REPORT_MAX_DEVICES
#define CMF_MGPU_REPORT_MAX_DEVICES 32
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <time.h>

static double cmf_wall_seconds(void)
{
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

/* Fine exact-solver ownership.  Unset or zero keeps the serial CPU owner;
 * a positive value requests exactly that many visible CUDA devices.  Invalid
 * text is terminal instead of being silently interpreted by atoi(). */
static int cmf_fine_multigpu_device_request(int *requested)
{
    const char *text = getenv("LUMINA_CMF_FINE_MGPU_DEVICES");
    if (!requested) return -1;
    *requested = 0;
    if (!text) return 0;
    if (!*text) return -1;
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' || value < 0 || value > INT_MAX)
        return -1;
    *requested = (int)value;
    return 0;
}

/* Proof-only configuration.  Refinement applies residual+K(e) to an already
 * verified error supersolution; it never changes the converged physical J.
 * Invalid or excessive requests fail closed instead of wrapping size_t. */
static int cmf_fine_envelope_refinement_request(size_t *requested)
{
    const char *text = getenv("LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS");
    if (!requested) return -1;
    *requested = 8U;
    if (!text) return 0;
    if (!*text || *text == '-') return -1;
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' || value < 1U || value > 64U)
        return -1;
    *requested = (size_t)value;
    return 0;
}

static int cmf_optional_binary_env(const char *name, int *enabled)
{
    const char *text;
    if (!name || !enabled) return -1;
    text = getenv(name);
    *enabled = 0;
    if (!text) return 0;
    if (strcmp(text, "0") == 0) return 0;
    if (strcmp(text, "1") == 0) {
        *enabled = 1;
        return 0;
    }
    return -1;
}

/* Physical constants (cgs) — local copies; planck_bnu in plasma.c is static. */
#define CM_H      6.62607015e-27   /* erg s            */
#define CM_KB     1.380649e-16     /* erg/K            */
#define CM_C      2.99792458e10    /* cm/s             */
#define CM_SIGMA_T 6.6524587e-25   /* Thomson cm^2     */

/* GPU comoving-frame formal solver (lumina_cmf_solve.cu); nonzero is terminal
 * for the current attempt (A2-12 forbids same-attempt CPU substitution). */
int cmf_solve_J_gpu(int NS, int NB, int NP, int adv_split, double a_lam,
    const double *chi_tot, const double *chi_es, const double *chi_abs,
    const double *S_fixed, double *J,
    const double *Bin, const double *adv_b, const double *advcoef_b,
    const int *rn, const int *rsh, const double *rz,
    const int *rcore, const double *rzin,
    const int *shell_off, const int *shell_k, const int *shell_seg,
    const double *shell_mu, int nsamp,
    int n_ali_iter, double tol, int *iters_out);

/* The default clean CPU target does not link CUDA objects.  Keep the existing
 * runtime fallback contract explicit so the CPU executable links portably;
 * CUDA builds define LUMINA_HAS_CUDA_BF_GEMM and use the strong GPU symbols. */
#ifndef LUMINA_HAS_CUDA_BF_GEMM
int cmf_solve_J_gpu(int NS, int NB, int NP, int adv_split, double a_lam,
    const double *chi_tot, const double *chi_es, const double *chi_abs,
    const double *S_fixed, double *J,
    const double *Bin, const double *adv_b, const double *advcoef_b,
    const int *rn, const int *rsh, const double *rz,
    const int *rcore, const double *rzin,
    const int *shell_off, const int *shell_k, const int *shell_seg,
    const double *shell_mu, int nsamp,
    int n_ali_iter, double tol, int *iters_out) {
    (void)NS;(void)NB;(void)NP;(void)adv_split;(void)a_lam;
    (void)chi_tot;(void)chi_es;(void)chi_abs;(void)S_fixed;(void)J;
    (void)Bin;(void)adv_b;(void)advcoef_b;(void)rn;(void)rsh;(void)rz;
    (void)rcore;(void)rzin;(void)shell_off;(void)shell_k;(void)shell_seg;
    (void)shell_mu;(void)nsamp;(void)n_ali_iter;(void)tol;(void)iters_out;
    return -1;
}

int bf_gemm_compute_fine(BFOpacity *bf, AtomicData *atom, PlasmaState *plasma,
                         int n_shells, const double *nu_fine, int n_fine,
                         double nu_min, double d_log_nu, double *chi_out) {
    (void)bf;(void)atom;(void)plasma;(void)n_shells;(void)nu_fine;(void)n_fine;
    (void)nu_min;(void)d_log_nu;(void)chi_out;
    return -1;
}
#endif

static int cmf_dcmp(const void *a, const void *b) {
    double d = *(const double*)a - *(const double*)b;
    return (d > 0) - (d < 0);
}

static inline double cm_planck(double nu, double T) {
    if (!(nu > 0.0) || !isfinite(nu) || T < 0.0 || !isfinite(T)) return NAN;
    if (T == 0.0) return 0.0;
    double x = CM_H * nu / (CM_KB * T);
    double prefactor = (2.0 * CM_H * nu * nu * nu) / (CM_C * CM_C);
    if (x > 50.0) {
        /* Algebraically identical Wien form.  Any zero is then IEEE
         * underflow of the represented value, not an imposed floor. */
        double y = exp(-x);
        return prefactor * y / (1.0 - y);
    }
    return prefactor / expm1(x);
}

/* Inner blackbody amplitude dilution (LUMINA_INNER_BB_SCALE, default 1.0). Read
 * once, shared by the J solve AND the emergent spectrum writers so the inner BC
 * is consistent. Previously only cmfgen_solve_J applied it; the emergent writers
 * (cmfgen_write_spectrum, cmfgen_write_spectrum_obs) used the full T_inner BB, so
 * INNER_BB_SCALE=0 diluted the solved field but NOT the emergent spectrum. */
static double cmf_inner_bb_scale(void) {
    static int init = 0; static double s = 1.0;
    if (!init) { const char *e = getenv("LUMINA_INNER_BB_SCALE");
                 if (e) s = atof(e); init = 1; }
    return s;
}

/* ---- Wave-3.2 R7: deterministic little-endian frozen chi/eta dump. ---- */
typedef struct {
    uint32_t h[8];
    uint64_t bits;
    unsigned char block[64];
    size_t used;
} CMFSHA256;

static uint32_t cmf_rotr32(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32U - n));
}

static void cmf_sha256_transform(CMFSHA256 *s, const unsigned char block[64]) {
    static const uint32_t K[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,
        0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,
        0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,
        0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,
        0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,
        0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,
        0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,
        0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,
        0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[4*i] << 24) |
               ((uint32_t)block[4*i+1] << 16) |
               ((uint32_t)block[4*i+2] << 8) | block[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t a = w[i-15], b = w[i-2];
        uint32_t s0 = cmf_rotr32(a,7) ^ cmf_rotr32(a,18) ^ (a >> 3);
        uint32_t s1 = cmf_rotr32(b,17) ^ cmf_rotr32(b,19) ^ (b >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3];
    uint32_t e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1=cmf_rotr32(e,6)^cmf_rotr32(e,11)^cmf_rotr32(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        uint32_t t1=h+S1+ch+K[i]+w[i];
        uint32_t S0=cmf_rotr32(a,2)^cmf_rotr32(a,13)^cmf_rotr32(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;
    s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}

static void cmf_sha256_init(CMFSHA256 *s) {
    static const uint32_t H[8] = {
        0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
        0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U
    };
    memcpy(s->h, H, sizeof(H)); s->bits = 0; s->used = 0;
}

static void cmf_sha256_update(CMFSHA256 *s, const void *ptr, size_t n) {
    const unsigned char *p = (const unsigned char *)ptr;
    s->bits += (uint64_t)n * 8U;
    while (n) {
        size_t take = 64 - s->used;
        if (take > n) take = n;
        memcpy(s->block + s->used, p, take);
        s->used += take; p += take; n -= take;
        if (s->used == 64) {
            cmf_sha256_transform(s, s->block);
            s->used = 0;
        }
    }
}

static void cmf_sha256_final(CMFSHA256 *s, unsigned char out[32]) {
    uint64_t bits = s->bits;
    unsigned char one = 0x80, zero = 0;
    cmf_sha256_update(s, &one, 1);
    while (s->used != 56) cmf_sha256_update(s, &zero, 1);
    unsigned char len[8];
    for (int i = 0; i < 8; i++) len[7-i] = (unsigned char)(bits >> (8*i));
    cmf_sha256_update(s, len, 8);
    for (int i = 0; i < 8; i++) {
        out[4*i] = (unsigned char)(s->h[i] >> 24);
        out[4*i+1] = (unsigned char)(s->h[i] >> 16);
        out[4*i+2] = (unsigned char)(s->h[i] >> 8);
        out[4*i+3] = (unsigned char)s->h[i];
    }
}

int cmfgen_sha256_file(const char *path, char hex[65]) {
    if (!path || !*path || !hex) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    CMFSHA256 sha;
    unsigned char buf[65536], digest[32];
    cmf_sha256_init(&sha);
    for (;;) {
        size_t n = fread(buf, 1, sizeof(buf), fp);
        if (n) cmf_sha256_update(&sha, buf, n);
        if (n < sizeof(buf)) {
            if (ferror(fp)) { fclose(fp); return -1; }
            break;
        }
    }
    if (fclose(fp)) return -1;
    cmf_sha256_final(&sha, digest);
    for (int i=0;i<32;i++) snprintf(hex+2*i,3,"%02x",digest[i]);
    hex[64]='\0';
    return 0;
}

static int cmf_dump_bytes(FILE *fp, CMFSHA256 *sha,
                          const unsigned char *p, size_t n) {
    if (n && fwrite(p, 1, n, fp) != n) return -1;
    cmf_sha256_update(sha, p, n);
    return 0;
}
static int cmf_dump_u32(FILE *fp, CMFSHA256 *sha, uint32_t x) {
    unsigned char b[4] = {(unsigned char)x,(unsigned char)(x>>8),
                          (unsigned char)(x>>16),(unsigned char)(x>>24)};
    return cmf_dump_bytes(fp, sha, b, sizeof(b));
}
static int cmf_dump_u64(FILE *fp, CMFSHA256 *sha, uint64_t x) {
    unsigned char b[8];
    for (int i=0;i<8;i++) b[i]=(unsigned char)(x>>(8*i));
    return cmf_dump_bytes(fp, sha, b, sizeof(b));
}
static int cmf_dump_f64(FILE *fp, CMFSHA256 *sha, double x) {
    uint64_t u;
    memcpy(&u, &x, sizeof(u));
    return cmf_dump_u64(fp, sha, u);
}

int cmfgen_dump_frozen_chieta_lane(const CMFGENState *cs, const Geometry *geo,
                                   int iter, int field_generation,
                                   int post_damping, const char *path,
                                   const CMFGENChietaLaneMeta *meta) {
    enum { FLAG_POST_DAMP=1U, FLAG_COHERENT_FROZEN=2U,
           FLAG_FREQUENCY_DESCENDING=4U };
    if (!cs || !geo || !path || !*path || iter < 0 || field_generation < 0 ||
        (post_damping != 0 && post_damping != 1) ||
        cs->n_shells <= 0 || cs->n_bins <= 0 ||
        geo->n_shells != cs->n_shells || !(geo->time_explosion > 0.0) ||
        !isfinite(geo->time_explosion) || !geo->r_inner || !geo->r_outer ||
        !cs->nu || !cs->dnu || !cs->chi_tot || !cs->chi_es ||
        !cs->S_fixed || !cs->J || !cs->eta_total_audit) {
        fprintf(stderr, "[CMF-CHIETA][FAIL] invalid state/path/iteration\n");
        return -1;
    }
    if (meta && (!meta->lane || !*meta->lane ||
                 !meta->common_state_sha256 ||
                 strlen(meta->common_state_sha256) != 64 ||
                 !meta->coverage)) {
        fprintf(stderr,"[CMF-CHIETA][FAIL] invalid E4 lane metadata\n");
        return -1;
    }
    for (int s=0;s<geo->n_shells;s++) {
        if (!(geo->r_inner[s] > 0.0) ||
            !(geo->r_outer[s] > geo->r_inner[s]) ||
            !isfinite(geo->r_inner[s]) || !isfinite(geo->r_outer[s])) {
            fprintf(stderr,"[CMF-CHIETA][FAIL] invalid radius shell=%d\n",s);
            return -1;
        }
        if (s && fabs(geo->r_inner[s]-geo->r_outer[s-1]) >
                 1e-12*fmax(fabs(geo->r_inner[s]),fabs(geo->r_outer[s-1]))) {
            fprintf(stderr,"[CMF-CHIETA][FAIL] non-contiguous radius shell=%d\n",s);
            return -1;
        }
    }
    for (int b=0;b<cs->n_bins;b++) {
        if (!(cs->nu[b] > 0.0) || !(cs->dnu[b] > 0.0) ||
            !isfinite(cs->nu[b]) || !isfinite(cs->dnu[b]) ||
            (b && !(cs->nu[b] > cs->nu[b-1]))) {
            fprintf(stderr,"[CMF-CHIETA][FAIL] source frequency grid b=%d\n",b);
            return -1;
        }
    }
    size_t cells = (size_t)cs->n_shells * (size_t)cs->n_bins;
    if (cells / (size_t)cs->n_bins != (size_t)cs->n_shells) {
        fprintf(stderr,"[CMF-CHIETA][FAIL] dimension overflow\n");
        return -1;
    }
    double eta_decomposition_max_abs = 0.0;
    int eta_decomposition_bitwise = 1;
    for (size_t q=0;q<cells;q++) {
        if (!(cs->chi_tot[q] >= 0.0) || !(cs->chi_es[q] >= 0.0) ||
            !(cs->J[q] >= 0.0) || !(cs->eta_total_audit[q] >= 0.0) ||
            !isfinite(cs->chi_tot[q]) ||
            !isfinite(cs->chi_es[q]) || !isfinite(cs->S_fixed[q]) ||
            !isfinite(cs->J[q]) || !isfinite(cs->eta_total_audit[q]) ||
            !isfinite(cs->chi_tot[q] * cs->S_fixed[q]) ||
            !isfinite(cs->chi_es[q] * cs->J[q]) ||
            !isfinite(cs->chi_tot[q] * cs->S_fixed[q] +
                      cs->chi_es[q] * cs->J[q])) {
            fprintf(stderr,"[CMF-CHIETA][FAIL] invalid cell=%zu\n",q);
            return -1;
        }
        double split = cs->chi_tot[q] * cs->S_fixed[q] +
                       cs->chi_es[q] * cs->J[q];
        double delta = fabs(cs->eta_total_audit[q] - split);
        if (delta > eta_decomposition_max_abs)
            eta_decomposition_max_abs = delta;
        if (memcmp(&cs->eta_total_audit[q], &split, sizeof(double)) != 0)
            eta_decomposition_bitwise = 0;
    }

    char manifest[4096], quarantine[4096];
    int mn=snprintf(manifest,sizeof(manifest),"%s.manifest.json",path);
    int qn=snprintf(quarantine,sizeof(quarantine),"%s.quarantine",path);
    if (mn < 0 || (size_t)mn >= sizeof(manifest) ||
        qn < 0 || (size_t)qn >= sizeof(quarantine)) {
        fprintf(stderr,"[CMF-CHIETA][FAIL] sidecar/quarantine path too long\n");
        return -1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr,"[CMF-CHIETA][FAIL] open %s: %s\n",path,strerror(errno));
        return -1;
    }
    CMFSHA256 sha; cmf_sha256_init(&sha);
    int fail = 0;
    const unsigned char magic[8] = {'L','C','M','F','C','E','0','1'};
#define DW(call) do { if (!fail && (call)) fail=1; } while (0)
    DW(cmf_dump_bytes(fp,&sha,magic,sizeof(magic)));
    DW(cmf_dump_u32(fp,&sha,UINT32_C(0x01020304)));
    DW(cmf_dump_u32(fp,&sha,UINT32_C(1)));
    DW(cmf_dump_u64(fp,&sha,(uint64_t)cs->n_shells));
    DW(cmf_dump_u64(fp,&sha,(uint64_t)cs->n_bins));
    DW(cmf_dump_u64(fp,&sha,(uint64_t)iter));
    DW(cmf_dump_u64(fp,&sha,(uint64_t)field_generation));
    DW(cmf_dump_u32(fp,&sha,(post_damping ? FLAG_POST_DAMP : 0U)|
                              FLAG_COHERENT_FROZEN|FLAG_FREQUENCY_DESCENDING));
    DW(cmf_dump_u32(fp,&sha,0));
    DW(cmf_dump_f64(fp,&sha,geo->time_explosion));
    DW(cmf_dump_f64(fp,&sha,geo->r_inner[0]));
    for (int s=0;s<geo->n_shells;s++)
        DW(cmf_dump_f64(fp,&sha,geo->r_outer[s]));
    for (int b=cs->n_bins-1;b>=0;b--) DW(cmf_dump_f64(fp,&sha,cs->nu[b]));
    for (int b=cs->n_bins-1;b>=0;b--) DW(cmf_dump_f64(fp,&sha,cs->dnu[b]));
    for (int s=0;s<cs->n_shells;s++) for (int b=cs->n_bins-1;b>=0;b--)
        DW(cmf_dump_f64(fp,&sha,cs->chi_tot[(size_t)s*cs->n_bins+b]));
    for (int s=0;s<cs->n_shells;s++) for (int b=cs->n_bins-1;b>=0;b--)
        DW(cmf_dump_f64(fp,&sha,cs->chi_es[(size_t)s*cs->n_bins+b]));
    for (int s=0;s<cs->n_shells;s++) for (int b=cs->n_bins-1;b>=0;b--) {
        size_t q=(size_t)s*cs->n_bins+b;
        DW(cmf_dump_f64(fp,&sha,cs->chi_tot[q]*cs->S_fixed[q]));
    }
    for (int s=0;s<cs->n_shells;s++) for (int b=cs->n_bins-1;b>=0;b--) {
        size_t q=(size_t)s*cs->n_bins+b;
        DW(cmf_dump_f64(fp,&sha,cs->chi_es[q]*cs->J[q]));
    }
    for (int s=0;s<cs->n_shells;s++) for (int b=cs->n_bins-1;b>=0;b--) {
        size_t q=(size_t)s*cs->n_bins+b;
        DW(cmf_dump_f64(fp,&sha,cs->eta_total_audit[q]));
    }
    for (int s=0;s<cs->n_shells;s++) for (int b=cs->n_bins-1;b>=0;b--)
        DW(cmf_dump_f64(fp,&sha,cs->J[(size_t)s*cs->n_bins+b]));
#undef DW
    if (fclose(fp)) fail=1;
    if (fail) {
        fprintf(stderr,"[CMF-CHIETA][FAIL] write %s: %s\n",path,strerror(errno));
        if (rename(path, quarantine) != 0)
            fprintf(stderr,"[CMF-CHIETA][FAIL] quarantine %s: %s\n",
                    quarantine,strerror(errno));
        return -1;
    }
    unsigned char digest[32]; char hex[65];
    cmf_sha256_final(&sha,digest);
    for(int i=0;i<32;i++) snprintf(hex+2*i,3,"%02x",digest[i]);
    hex[64]='\0';
    FILE *mf=fopen(manifest,"w");
    if(!mf) {
        fprintf(stderr,"[CMF-CHIETA][FAIL] open %s: %s\n",manifest,strerror(errno));
        if (rename(path, quarantine) != 0)
            fprintf(stderr,"[CMF-CHIETA][FAIL] quarantine %s: %s\n",
                    quarantine,strerror(errno));
        return -1;
    }
    int sidecar_fail = fprintf(mf,"{\n  \"schema\": \"LCMFCE01-v1\",\n"
               "  \"sha256\": \"%s\",\n"
               "  \"iteration\": %d,\n  \"field_generation\": %d,\n"
               "  \"post_damping\": %s,\n"
               "  \"coherent_frozen\": true,\n"
               "  \"frequency_descending\": true,\n"
               "  \"eta_decomposition_bitwise\": %s,\n"
               "  \"eta_decomposition_max_abs\": %.17g",
               hex,iter,field_generation,post_damping ? "true" : "false",
               eta_decomposition_bitwise ? "true" : "false",
               eta_decomposition_max_abs) < 0;
    if (!sidecar_fail && meta) {
        const CMFGENEmissABStats *st = meta->coverage;
        int is_a = strcmp(meta->lane,"A-production") == 0;
        int is_b2 = strcmp(meta->lane,"B2-Aul-nu-retain-A-undefined") == 0;
        const char *formula = is_a
            ? "production-eps-times-source"
            : (is_b2
               ? "covered:hnu-over-4pi-times-Aul-times-n_upper-over-dnu;undefined:production-A-retained"
               : "hnu-over-4pi-times-Aul-times-n_upper-over-dnu");
        const char *undefined_policy = is_a ? "production-not-applicable"
            : (is_b2 ? "retain-production-A-explicit-controlled"
                     : "zero-undefined-fail-closed");
        sidecar_fail = fprintf(mf,
               ",\n  \"emiss_ab_lane\": \"%s\",\n"
               "  \"common_assembly_state_sha256\": \"%s\",\n"
               "  \"line_emissivity_formula\": \"%s\",\n"
               "  \"undefined_transition_policy\": \"%s\",\n"
               "  \"controlled_retention\": %s,\n"
               "  \"undefined_transition_list_suffix\": \".undefined.csv\",\n"
               "  \"coverage\": {\n"
               "    \"active_transition_count\": %llu,\n"
               "    \"defined_transition_count\": %llu,\n"
               "    \"undefined_transition_count\": %llu,\n"
               "    \"active_line_shell_count\": %llu,\n"
               "    \"defined_line_shell_count\": %llu,\n"
               "    \"undefined_line_shell_count\": %llu,\n"
               "    \"retained_transition_count\": %llu,\n"
               "    \"retained_line_shell_count\": %llu,\n"
               "    \"a_reference_line_power\": %.17g,\n"
               "    \"a_reference_covered_line_power\": %.17g,\n"
               "    \"a_reference_undefined_line_power\": %.17g,\n"
               "    \"a_reference_contribution_fraction\": %.17g,\n"
               "    \"a_reference_undefined_contribution_fraction\": %.17g,\n"
               "    \"a_reference_retained_line_power\": %.17g,\n"
               "    \"a_reference_retained_contribution_fraction\": %.17g\n"
               "  },\n"
               "  \"seeded_defect\": {\"line_id\": %d, \"shell\": %d, "
               "\"population_factor\": %.17g, \"hits\": %llu},\n"
               "  \"undefined_a_reference_diagnostic\": {\n"
               "    \"epoch\": \"pre-EPAY\",\n"
               "    \"quantity\": \"sum eta_A_undefined*dnu\",\n"
               "    \"units\": \"erg s^-1 cm^-3 sr^-1\",\n"
               "    \"n_bins\": %d,\n"
               "    \"n_shells\": %d,\n"
               "    \"by_band\": [",
               meta->lane,meta->common_state_sha256,formula,undefined_policy,
               is_b2 ? "true" : "false",
               (unsigned long long)st->active_transition_count,
               (unsigned long long)st->defined_transition_count,
               (unsigned long long)st->undefined_transition_count,
               (unsigned long long)st->active_line_shell_count,
               (unsigned long long)st->defined_line_shell_count,
               (unsigned long long)st->undefined_line_shell_count,
               (unsigned long long)(is_b2 ? st->retained_transition_count : 0),
               (unsigned long long)(is_b2 ? st->retained_line_shell_count : 0),
               st->a_reference_line_power,
               st->a_reference_covered_line_power,
               st->a_reference_undefined_line_power,
               st->a_reference_contribution_fraction,
               st->a_reference_undefined_contribution_fraction,
               is_b2 ? st->a_reference_retained_line_power : 0.0,
               is_b2 ? st->a_reference_retained_contribution_fraction : 0.0,
               st->seed_line,st->seed_shell,st->seed_factor,
               (unsigned long long)st->seed_hits,st->n_bins,st->n_shells) < 0;
        for (int b=0;b<st->n_bins && !sidecar_fail;b++)
            sidecar_fail=fprintf(mf,"%s%.17g",b ? "," : "",
                st->undefined_a_emissivity_by_band[b]) < 0;
        if (!sidecar_fail)
            sidecar_fail=fprintf(mf,"],\n    \"by_shell\": [") < 0;
        for (int s=0;s<st->n_shells && !sidecar_fail;s++)
            sidecar_fail=fprintf(mf,"%s%.17g",s ? "," : "",
                st->undefined_a_emissivity_by_shell[s]) < 0;
        if (!sidecar_fail)
            sidecar_fail=fprintf(mf,"]\n  }\n") < 0;
    } else if (!sidecar_fail) {
        sidecar_fail = fprintf(mf,"\n") < 0;
    }
    if (!sidecar_fail) sidecar_fail = fprintf(mf,"}\n") < 0;
    if(fclose(mf)) sidecar_fail = 1;
    if(sidecar_fail) {
        fprintf(stderr,"[CMF-CHIETA][FAIL] write/close %s\n",manifest);
        if (rename(path, quarantine) != 0)
            fprintf(stderr,"[CMF-CHIETA][FAIL] quarantine %s: %s\n",
                    quarantine,strerror(errno));
        return -1;
    }
    fprintf(stderr,"[CMF-CHIETA] wrote %s iter=%d generation=%d post_damp=%d "
            "sha256=%s\n", path,iter,field_generation,post_damping,hex);
    return 0;
}

int cmfgen_dump_frozen_chieta(const CMFGENState *cs, const Geometry *geo,
                              int iter, int field_generation,
                              int post_damping, const char *path) {
    return cmfgen_dump_frozen_chieta_lane(cs,geo,iter,field_generation,
                                          post_damping,path,NULL);
}

/* ============================================================================
 * [CMF-LINEPOP T2] population-native line dump.
 *
 * WHY.  The frozen chi/eta capture (LCMFCE01) is per (shell, coarse bin): it
 * records what the assembled line forest *summed to*, not what any individual
 * line contributed nor which populations produced it.  T2 ("replace only the
 * line assembly with population-native chi_l[n_l,n_u] and eta_l = A_ul n_u")
 * therefore had no instrument: the E4 B/B2 lanes can swap the emissivity but
 * chi stays bitwise identical, so the experiment is not a single-factor test.
 *
 * WHAT.  A READ-ONLY REPLAY of the assemble per-line loop, fired in the same
 * `it == LUMINA_CMF_FROZEN_CHIETA_ITER` block that writes the chi/eta capture,
 * i.e. the SAME GENERATION BY CONSTRUCTION.  Nothing in cmfgen_assemble_impl
 * is touched; between assemble and this call only cmfgen_solve_J and the J
 * damping run, and neither writes chi_line/chi_line_th/chi_abs/chi_tot.  The
 * replay re-derives chi_line from the same inputs in the same order and the
 * writer records whether it reproduced cs->chi_line BITWISE -- a round-trip
 * identity that makes "same generation" checkable, not asserted.
 *
 * THE EPAY DISPOSITION.  Per (shell, bin) the artifact also records whether
 * the line emissivity assembled here actually survives into S_fixed: under
 * LUMINA_CMF_EPAY>=2 the thin-bin source is REBUILT from chi_line_th*B(T_e)
 * plus the Milne bf shape, so eta_line is discarded there.  Without this
 * column an offline T2 can swap eta in cells where eta is thrown away and
 * conclude nothing.
 *
 * Gate LUMINA_CMF_LINEPOP_DUMP (path). Unset => not called, allocates nothing.
 * ==========================================================================*/

static void cmf_pack_u32(unsigned char *b, uint32_t x) {
    b[0]=(unsigned char)x; b[1]=(unsigned char)(x>>8);
    b[2]=(unsigned char)(x>>16); b[3]=(unsigned char)(x>>24);
}
static void cmf_pack_i32(unsigned char *b, int x) {
    cmf_pack_u32(b,(uint32_t)x);
}
static void cmf_pack_f64(unsigned char *b, double x) {
    uint64_t u; memcpy(&u,&x,sizeof(u));
    for (int i=0;i<8;i++) b[i]=(unsigned char)(u>>(8*i));
}

#define CMF_LINEPOP_ROW_BYTES  76
#define CMF_LINEPOP_LINE_BYTES 80

enum {
    CMF_LP_F_NLTE_ION     = 1U << 0, /* line is mapped to an NLTE ion        */
    CMF_LP_F_POPS_DEFINED = 1U << 1, /* both n_l and n_u are NLTE-solved     */
    CMF_LP_F_SL_POP       = 1U << 2, /* line_source_S > 0 (population-native)*/
    CMF_LP_F_SL_FALLBACK  = 1U << 3, /* S used by assemble was B_nu(T_e)     */
    CMF_LP_F_STIM_CLAMPED = 1U << 4, /* 1-(g_l n_u)/(g_u n_l) <= 0           */
    CMF_LP_F_TAU_ROUNDTRIP= 1U << 5  /* tau recomputed from pops matches     */
};

typedef struct {
    int shells[64], n_shells_sel;
    double lam_lo, lam_hi;
    long max_rows;
} CMFLinePopSel;

static int cmf_linepop_parse_sel(CMFLinePopSel *sel, int n_shells) {
    memset(sel, 0, sizeof(*sel));
    sel->lam_lo = 600.0; sel->lam_hi = 3000.0; sel->max_rows = 4000000L;
    const char *se = getenv("LUMINA_CMF_LINEPOP_SHELLS");
    if (!se || !*se) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] LUMINA_CMF_LINEPOP_SHELLS is "
                "required when the dump is armed (comma list, e.g. 8,16,45)\n");
        return -1;
    }
    char buf[512];
    if (snprintf(buf,sizeof(buf),"%s",se) >= (int)sizeof(buf)) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] shell list too long\n");
        return -1;
    }
    for (char *tok=strtok(buf,", \t"); tok; tok=strtok(NULL,", \t")) {
        char *end=NULL; long v=strtol(tok,&end,10);
        if (end==tok || *end || v < 0 || v >= n_shells) {
            fprintf(stderr,"[CMF-LINEPOP][FAIL] shell '%s' out of [0,%d)\n",
                    tok,n_shells);
            return -1;
        }
        if (sel->n_shells_sel >= (int)(sizeof(sel->shells)/sizeof(sel->shells[0]))) {
            fprintf(stderr,"[CMF-LINEPOP][FAIL] too many shells selected\n");
            return -1;
        }
        for (int i=0;i<sel->n_shells_sel;i++) if (sel->shells[i]==(int)v) {
            fprintf(stderr,"[CMF-LINEPOP][FAIL] duplicate shell %ld\n",v);
            return -1;
        }
        sel->shells[sel->n_shells_sel++]=(int)v;
    }
    if (!sel->n_shells_sel) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] empty shell selection\n");
        return -1;
    }
    const char *le = getenv("LUMINA_CMF_LINEPOP_LAM");
    if (le && *le) {
        double lo=0.0, hi=0.0;
        if (sscanf(le,"%lf,%lf",&lo,&hi)!=2 || !(lo > 0.0) || !(hi > lo)) {
            fprintf(stderr,"[CMF-LINEPOP][FAIL] LUMINA_CMF_LINEPOP_LAM must be "
                    "lo,hi in Angstrom with 0<lo<hi (got %s)\n",le);
            return -1;
        }
        sel->lam_lo=lo; sel->lam_hi=hi;
    }
    const char *me = getenv("LUMINA_CMF_LINEPOP_MAXROWS");
    if (me && *me) {
        char *end=NULL; long v=strtol(me,&end,10);
        if (end==me || *end || v <= 0) {
            fprintf(stderr,"[CMF-LINEPOP][FAIL] LUMINA_CMF_LINEPOP_MAXROWS "
                    "must be a positive integer (got %s)\n",me);
            return -1;
        }
        sel->max_rows=v;
    }
    return 0;
}

/* level_num -> global level index, per ion population.  Same construction the
 * E4 lane uses, extended to serve BOTH the lower and the upper level. */
typedef struct { int **by_number; int *max_number; int n_ion_pops;
                 int zi[100*100]; } CMFLevIdx;

static void cmf_levidx_free(CMFLevIdx *ix) {
    if (!ix->by_number) return;
    for (int i=0;i<ix->n_ion_pops;i++) free(ix->by_number[i]);
    free(ix->by_number); free(ix->max_number);
    ix->by_number=NULL; ix->max_number=NULL;
}

static int cmf_levidx_build(CMFLevIdx *ix, const AtomicData *atom) {
    memset(ix,0,sizeof(*ix));
    int np = atom->n_ion_pops;
    if (np <= 0 || !atom->level_offset || !atom->level_num) return -1;
    ix->n_ion_pops=np;
    for (int k=0;k<100*100;k++) ix->zi[k]=-1;
    if (atom->ion_pop_Z && atom->ion_pop_stage)
        for (int j=0;j<np;j++) {
            int Z=atom->ion_pop_Z[j], st=atom->ion_pop_stage[j];
            if (Z >= 0 && Z < 100 && st >= 0 && st < 100) ix->zi[Z*100+st]=j;
        }
    ix->by_number=(int **)calloc((size_t)np,sizeof(int *));
    ix->max_number=(int *)malloc((size_t)np*sizeof(int));
    if (!ix->by_number || !ix->max_number) { cmf_levidx_free(ix); return -1; }
    for (int i=0;i<np;i++) {
        ix->max_number[i]=-1;
        for (int g=atom->level_offset[i]; g<atom->level_offset[i+1]; g++)
            if (atom->level_num[g] > ix->max_number[i])
                ix->max_number[i]=atom->level_num[g];
        if (ix->max_number[i] < 0) continue;
        ix->by_number[i]=(int *)malloc((size_t)(ix->max_number[i]+1)*sizeof(int));
        if (!ix->by_number[i]) { cmf_levidx_free(ix); return -1; }
        for (int k=0;k<=ix->max_number[i];k++) ix->by_number[i][k]=-1;
        for (int g=atom->level_offset[i]; g<atom->level_offset[i+1]; g++) {
            int number=atom->level_num[g];
            if (number >= 0 && number <= ix->max_number[i])
                ix->by_number[i][number]=g;
        }
    }
    return 0;
}

static int cmf_levidx_lookup(const CMFLevIdx *ix, int ion_pop, int level_num) {
    if (ion_pop < 0 || ion_pop >= ix->n_ion_pops || !ix->by_number[ion_pop] ||
        level_num < 0 || level_num > ix->max_number[ion_pop]) return -1;
    return ix->by_number[ion_pop][level_num];
}

int cmfgen_dump_line_populations(const CMFGENState *cs, const Geometry *geo,
                                 const OpacityState *opac,
                                 const PlasmaState *plasma,
                                 const NLTEConfig *nlte, const AtomicData *atom,
                                 int iter, int field_generation,
                                 const char *path) {
    if (!cs || !geo || !opac || !plasma || !nlte || !atom || !path || !*path ||
        iter < 0 || field_generation < 0 || cs->n_shells <= 0 ||
        cs->n_bins <= 0 || geo->n_shells != cs->n_shells ||
        opac->n_shells != cs->n_shells || atom->n_lines != opac->n_lines ||
        !opac->tau_sobolev || !opac->line_list_nu || !cs->nu || !cs->dnu ||
        !cs->chi_line || !cs->chi_line_th || !cs->chi_abs || !cs->chi_tot ||
        !plasma->T_e || !atom->line_atomic_number ||
        !atom->line_ion_number || !atom->line_level_lower ||
        !atom->line_level_upper || !atom->level_g || !atom->level_num ||
        !nlte->global_to_nlte_level || !nlte->nlte_level_populations) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] invalid state/path/iteration\n");
        return -1;
    }
    /* The replay reproduces the PRODUCTION assemble only.  Refuse the two
     * variant assemblies rather than emit a mislabeled artifact. */
    if (cs->cont_only || cs->frozen_morph_eps >= 0.0) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] state is a cont_only/frozen-morph "
                "assemble; the replay would not be the production forest\n");
        return -1;
    }
    { const char *tf=getenv("LUMINA_CMF_EPAY_TAUEFF");
      if (tf && atof(tf) > 0.0) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] LUMINA_CMF_EPAY_TAUEFF>0: the EPAY "
                "shell gate is built from the PREVIOUS assemble's chi_abs/"
                "chi_tot and cannot be reproduced here\n");
        return -1;
      } }

    const int NB = cs->n_bins, NS = cs->n_shells, n_lines = opac->n_lines;
    CMFLinePopSel sel;
    if (cmf_linepop_parse_sel(&sel,NS) != 0) return -1;

    /* Same env reads as cmfgen_assemble_impl, same defaults. */
    int eps_phys=0, src_nlte=0;
    double eps_floor=1e-5, eps_cap=1.0;
    { const char *ep=getenv("LUMINA_CMFGEN_LINE_EPS_PHYS");
      if (ep && atoi(ep)) eps_phys=1;
      const char *ef=getenv("LUMINA_CMFGEN_EPS_FLOOR"); if (ef) eps_floor=atof(ef);
      const char *ec=getenv("LUMINA_CMFGEN_EPS_CAP");   if (ec) eps_cap=atof(ec);
      const char *sn=getenv("LUMINA_CMFGEN_SRC_NLTE");
      if (sn && atoi(sn)) src_nlte=1; }
    double line_eps=-1.0, line_gate=1.0, eps_uv=-1.0;
    { const char *le=getenv("LUMINA_CMFGEN_LINE_EPS"); if (le) line_eps=atof(le);
      const char *lg=getenv("LUMINA_CMFGEN_LINE_EPS_GATE"); if (lg) line_gate=atof(lg);
      const char *eu=getenv("LUMINA_CMFGEN_LINE_EPS_UV"); if (eu) eps_uv=atof(eu); }
    int epay=0, epay_smin=0; double epay_taubin=1.0, epay_hotf=1.5;
    { const char *e=getenv("LUMINA_CMF_EPAY"); epay=e?atoi(e):0;
      if (epay < 0) epay=0;
      const char *es=getenv("LUMINA_CMF_EPAY_SMIN"); if (es) epay_smin=atoi(es);
      const char *tb=getenv("LUMINA_CMF_EPAY_TAUBIN"); if (tb) epay_taubin=atof(tb);
      const char *hf=getenv("LUMINA_CMF_EPAY_HOTF"); if (hf) epay_hotf=atof(hf); }

    const double inv_ct = 1.0 / (CM_C * geo->time_explosion);
    const double nu_lo = CM_C / (sel.lam_hi * 1.0e-8);
    const double nu_hi = CM_C / (sel.lam_lo * 1.0e-8);

    CMFLevIdx ix;
    if (cmf_levidx_build(&ix,atom) != 0) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] level index build failed\n");
        return -1;
    }

    int    *line_slot = (int *)malloc((size_t)n_lines*sizeof(int));
    double *chi_rep   = (double *)calloc((size_t)sel.n_shells_sel*NB,sizeof(double));
    double *chith_rep = (double *)calloc((size_t)sel.n_shells_sel*NB,sizeof(double));
    double *eta_rep   = (double *)calloc((size_t)sel.n_shells_sel*NB,sizeof(double));
    unsigned char *disp = (unsigned char *)calloc((size_t)NS*NB,1);
    if (!line_slot || !chi_rep || !chith_rep || !eta_rep || !disp) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] host allocation failed\n");
        free(line_slot); free(chi_rep); free(chith_rep); free(eta_rep);
        free(disp); cmf_levidx_free(&ix); return -1;
    }
    for (int l=0;l<n_lines;l++) line_slot[l]=-1;

    /* ---- pass 1: exact assemble predicates; count rows, mark lines -------- */
    long n_rows=0; int n_lines_sel=0;
    for (int si=0; si<sel.n_shells_sel; si++) {
        int s = sel.shells[si];
        for (int l=0;l<n_lines;l++) {
            double tau = opac->tau_sobolev[(size_t)l*NS+s];
            if (opac->tau_validity &&
                opac->tau_validity[(size_t)l*NS+s] != A208_VALID &&
                opac->tau_validity[(size_t)l*NS+s] != A208_EXACT_ZERO) continue;
            if (tau == 0.0) continue;
            double nu_l = opac->line_list_nu[l];
            if (nu_l <= cs->nu_min || nu_l >= cs->nu_max) continue;
            int b = (int)floor(log(nu_l/cs->nu_min)/cs->d_log_nu);
            if (b < 0 || b >= NB) continue;
            if (nu_l < nu_lo || nu_l > nu_hi) continue;   /* selection only */
            n_rows++;
            if (line_slot[l] < 0) line_slot[l]=n_lines_sel++;
        }
    }
    if (n_rows > sel.max_rows) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] selection yields %ld rows "
                "(%.1f MiB) > LUMINA_CMF_LINEPOP_MAXROWS=%ld; narrow the "
                "shell/lambda selection instead of truncating\n",
                n_rows,(double)n_rows*CMF_LINEPOP_ROW_BYTES/1048576.0,
                sel.max_rows);
        free(line_slot); free(chi_rep); free(chith_rep); free(eta_rep);
        free(disp); cmf_levidx_free(&ix); return -1;
    }

    size_t row_bytes  = (size_t)n_rows*CMF_LINEPOP_ROW_BYTES;
    size_t line_bytes = (size_t)n_lines_sel*CMF_LINEPOP_LINE_BYTES;
    unsigned char *rows = (unsigned char *)malloc(row_bytes ? row_bytes : 1);
    unsigned char *lrec = (unsigned char *)calloc(line_bytes ? line_bytes : 1,1);
    if (!rows || !lrec) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] payload allocation failed "
                "(%.1f MiB)\n",(double)(row_bytes+line_bytes)/1048576.0);
        free(rows); free(lrec); free(line_slot); free(chi_rep); free(chith_rep);
        free(eta_rep); free(disp); cmf_levidx_free(&ix); return -1;
    }

    /* ---- pass 2: replay (accumulation order identical to assemble) -------- */
    unsigned char *rp = rows;
    long written=0;
    for (int si=0; si<sel.n_shells_sel; si++) {
        int s = sel.shells[si];
        double Te = plasma->T_e[s];
        double ne_s = plasma->n_electron ? plasma->n_electron[s]
                                         : opac->electron_density[s];
        for (int l=0;l<n_lines;l++) {
            double tau = opac->tau_sobolev[(size_t)l*NS+s];
            A208Validity tau_status = opac->tau_validity
                ? opac->tau_validity[(size_t)l*NS+s]
                : (isfinite(tau) ? (tau == 0.0 ? A208_EXACT_ZERO : A208_VALID)
                                 : A208_NONFINITE);
            if (tau_status != A208_VALID && tau_status != A208_EXACT_ZERO) continue;
            if (tau == 0.0) continue;
            double nu_l = opac->line_list_nu[l];
            if (nu_l <= cs->nu_min || nu_l >= cs->nu_max) continue;
            int b = (int)floor(log(nu_l/cs->nu_min)/cs->d_log_nu);
            if (b < 0 || b >= NB) continue;
            double frac = (tau > 1e-6) ? -expm1(-tau) : tau;
            double w = frac * nu_l * inv_ct / cs->dnu[b];
            double Sl_pop = opac->line_source_S
                          ? opac->line_source_S[(size_t)l*NS+s] : NAN;
            A208Validity source_status = opac->line_source_validity
                ? opac->line_source_validity[(size_t)l*NS+s] : A208_UNSAMPLED;
            double Sl = src_nlte ? Sl_pop : NAN;
            int sl_fallback = source_status != A208_VALID &&
                              source_status != A208_EXACT_ZERO;
            size_t ridx=(size_t)si*NB+b;
            chi_rep[ridx] += w;
            double el = 1.0, eta_l;
            if (eps_phys) {
                el = radeq_line_eps_phys(l, ne_s, Te, tau);
                if (el < 0.0) el = 1.0;
                if (el < eps_floor) el = eps_floor;
                if (el > eps_cap)   el = eps_cap;
                chith_rep[ridx] += w * el;
                eta_l = w * el * Sl;
            } else {
                eta_l = w * Sl;
            }
            eta_rep[ridx] += eta_l;
            if (nu_l < nu_lo || nu_l > nu_hi) continue;   /* not recorded */

            int slot = line_slot[l];
            int Z = atom->line_atomic_number[l];
            int ion = atom->line_ion_number[l];
            int ion_pop=(Z >= 0 && Z < 100 && ion >= 0 && ion < 100)
                      ? ix.zi[Z*100+ion] : -1;
            int glo = cmf_levidx_lookup(&ix,ion_pop,atom->line_level_lower[l]);
            int gup = cmf_levidx_lookup(&ix,ion_pop,atom->line_level_upper[l]);
            int nlo = (glo >= 0) ? nlte->global_to_nlte_level[glo] : -1;
            int nup = (gup >= 0) ? nlte->global_to_nlte_level[gup] : -1;
            double n_lower = (nlo >= 0)
                ? nlte->nlte_level_populations[(size_t)nlo*NS+s] : -1.0;
            double n_upper = (nup >= 0)
                ? nlte->nlte_level_populations[(size_t)nup*NS+s] : -1.0;
            int g_lo = (glo >= 0) ? atom->level_g[glo] : -1;
            int g_up = (gup >= 0) ? atom->level_g[gup] : -1;

            unsigned int flags=0;
            if (nlte->nlte_line_map && nlte->nlte_line_map[l] >= 0)
                flags |= CMF_LP_F_NLTE_ION;
            if (n_lower > 0.0 && n_upper > 0.0) flags |= CMF_LP_F_POPS_DEFINED;
            if (Sl_pop > 0.0) flags |= CMF_LP_F_SL_POP;
            if (sl_fallback) flags |= CMF_LP_F_SL_FALLBACK;
            /* population-native tau round trip: the SAME formula
             * nlte_update_tau_sobolev used to write opac->tau_sobolev. */
            double tau_pop = -1.0;
            if ((flags & CMF_LP_F_POPS_DEFINED) && g_lo > 0 && g_up > 0 &&
                atom->line_f_lu && atom->line_wavelength_cm) {
                A208ValueView roundtrip = a208_signed_sobolev(
                    SOBOLEV_COEFF, atom->line_f_lu[l],
                    atom->line_wavelength_cm[l], geo->time_explosion,
                    n_lower, n_upper, g_lo, g_up,
                    opac->tau_computed_generation);
                tau_pop = roundtrip.value;
                if (memcmp(&tau_pop,&tau,sizeof(double))==0)
                    flags |= CMF_LP_F_TAU_ROUNDTRIP;
            }
            if (slot >= 0) {
                unsigned char *lp = lrec + (size_t)slot*CMF_LINEPOP_LINE_BYTES;
                cmf_pack_u32(lp+0,(uint32_t)l);
                cmf_pack_u32(lp+4,(uint32_t)b);
                cmf_pack_i32(lp+8,Z);
                cmf_pack_i32(lp+12,ion);
                cmf_pack_i32(lp+16,g_lo);
                cmf_pack_i32(lp+20,g_up);
                cmf_pack_i32(lp+24,nlo);
                cmf_pack_i32(lp+28,nup);
                cmf_pack_f64(lp+32,nu_l);
                cmf_pack_f64(lp+40,atom->line_wavelength_cm
                                   ? atom->line_wavelength_cm[l] : -1.0);
                cmf_pack_f64(lp+48,atom->line_A_ul ? atom->line_A_ul[l] : -1.0);
                cmf_pack_f64(lp+56,atom->line_f_lu ? atom->line_f_lu[l] : -1.0);
                cmf_pack_f64(lp+64,(glo>=0 && atom->level_energy_eV)
                                   ? atom->level_energy_eV[glo] : -1.0);
                cmf_pack_f64(lp+72,(gup>=0 && atom->level_energy_eV)
                                   ? atom->level_energy_eV[gup] : -1.0);
            }
            cmf_pack_u32(rp+0,(uint32_t)(slot >= 0 ? slot : 0));
            cmf_pack_u32(rp+4,(uint32_t)si);
            cmf_pack_u32(rp+8,flags);
            cmf_pack_f64(rp+12,tau);
            cmf_pack_f64(rp+20,tau_pop);
            cmf_pack_f64(rp+28,n_lower);
            cmf_pack_f64(rp+36,n_upper);
            cmf_pack_f64(rp+44,Sl_pop);
            cmf_pack_f64(rp+52,Sl);
            cmf_pack_f64(rp+60,el);
            cmf_pack_f64(rp+68,w);
            rp += CMF_LINEPOP_ROW_BYTES;
            written++;
        }
    }
    if (written != n_rows) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] row count changed between passes "
                "(%ld vs %ld)\n",written,n_rows);
        free(rows); free(lrec); free(line_slot); free(chi_rep); free(chith_rep);
        free(eta_rep); free(disp); cmf_levidx_free(&ix); return -1;
    }

    /* ---- round-trip identity against the live assembled state ------------- */
    int chi_bitwise=1; double chi_max_abs=0.0;
    int chith_comparable = (eps_phys && line_eps <= 0.0 && eps_uv <= 0.0);
    int chith_bitwise=1; double chith_max_abs=0.0;
    for (int si=0; si<sel.n_shells_sel; si++) {
        int s=sel.shells[si];
        for (int b=0;b<NB;b++) {
            double a=chi_rep[(size_t)si*NB+b];
            double c=cs->chi_line[(size_t)s*NB+b];
            double d=fabs(a-c); if (d > chi_max_abs) chi_max_abs=d;
            if (memcmp(&a,&c,sizeof(double))!=0) chi_bitwise=0;
            if (chith_comparable) {
                double at=chith_rep[(size_t)si*NB+b];
                double ct=cs->chi_line_th[(size_t)s*NB+b];
                double dt=fabs(at-ct); if (dt > chith_max_abs) chith_max_abs=dt;
                if (memcmp(&at,&ct,sizeof(double))!=0) chith_bitwise=0;
            }
        }
    }

    /* ---- EPAY disposition: does eta_line reach S_fixed in this cell? ------ */
    long disp_n[4]={0,0,0,0};
    for (int s=0;s<NS;s++) {
        double dr_s = geo->r_outer[s] - geo->r_inner[s];
        int hot = 0; /* legacy scalar-temperature classifier removed */
        for (int b=0;b<NB;b++) {
            size_t idx=(size_t)s*NB+b;
            unsigned char d=0;   /* pass-1 legacy source: eta_line live */
            if (epay && s >= epay_smin) {
                int thick=((cs->chi_abs[idx]+cs->chi_line_th[idx])*dr_s
                           > epay_taubin);
                d = thick ? 1 : ((epay >= 2 && hot) ? 2 : 3);
            }
            disp[idx]=d; disp_n[d]++;
        }
    }

    /* ---- write LCMFLP01 --------------------------------------------------- */
    char manifest[4096], quarantine[4096];
    int mn=snprintf(manifest,sizeof(manifest),"%s.manifest.json",path);
    int qn=snprintf(quarantine,sizeof(quarantine),"%s.quarantine",path);
    if (mn < 0 || (size_t)mn >= sizeof(manifest) ||
        qn < 0 || (size_t)qn >= sizeof(quarantine)) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] sidecar path too long\n");
        free(rows); free(lrec); free(line_slot); free(chi_rep); free(chith_rep);
        free(eta_rep); free(disp); cmf_levidx_free(&ix); return -1;
    }
    FILE *fp=fopen(path,"wb");
    if (!fp) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] open %s: %s\n",path,strerror(errno));
        free(rows); free(lrec); free(line_slot); free(chi_rep); free(chith_rep);
        free(eta_rep); free(disp); cmf_levidx_free(&ix); return -1;
    }
    CMFSHA256 sha; cmf_sha256_init(&sha);
    int fail=0;
    const unsigned char magic[8]={'L','C','M','F','L','P','0','2'};
#define LW(call) do { if (!fail && (call)) fail=1; } while (0)
    LW(cmf_dump_bytes(fp,&sha,magic,sizeof(magic)));
    LW(cmf_dump_u32(fp,&sha,UINT32_C(0x01020304)));
    LW(cmf_dump_u32(fp,&sha,UINT32_C(2)));
    LW(cmf_dump_u64(fp,&sha,(uint64_t)iter));
    LW(cmf_dump_u64(fp,&sha,(uint64_t)field_generation));
    LW(cmf_dump_u32(fp,&sha,(uint32_t)NS));
    LW(cmf_dump_u32(fp,&sha,(uint32_t)NB));
    LW(cmf_dump_u32(fp,&sha,(uint32_t)sel.n_shells_sel));
    LW(cmf_dump_u32(fp,&sha,(uint32_t)n_lines_sel));
    LW(cmf_dump_u64(fp,&sha,(uint64_t)n_rows));
    LW(cmf_dump_f64(fp,&sha,geo->time_explosion));
    LW(cmf_dump_f64(fp,&sha,sel.lam_lo));
    LW(cmf_dump_f64(fp,&sha,sel.lam_hi));
    LW(cmf_dump_u32(fp,&sha,(uint32_t)eps_phys));
    LW(cmf_dump_u32(fp,&sha,(uint32_t)src_nlte));
    LW(cmf_dump_u32(fp,&sha,(uint32_t)epay));
    LW(cmf_dump_u32(fp,&sha,(uint32_t)epay_smin));
    LW(cmf_dump_f64(fp,&sha,epay_taubin));
    LW(cmf_dump_f64(fp,&sha,epay_hotf));
    LW(cmf_dump_f64(fp,&sha,eps_floor));
    LW(cmf_dump_f64(fp,&sha,eps_cap));
    LW(cmf_dump_f64(fp,&sha,line_eps));
    LW(cmf_dump_f64(fp,&sha,eps_uv));
    LW(cmf_dump_f64(fp,&sha,line_gate));
    for (int si=0;si<sel.n_shells_sel;si++)
        LW(cmf_dump_u32(fp,&sha,(uint32_t)sel.shells[si]));
    for (int si=0;si<sel.n_shells_sel;si++) {
        int s=sel.shells[si];
        LW(cmf_dump_f64(fp,&sha,plasma->T_e[s]));
        LW(cmf_dump_f64(fp,&sha,plasma->n_electron ? plasma->n_electron[s]
                                                   : opac->electron_density[s]));
        LW(cmf_dump_f64(fp,&sha,geo->r_outer[s]-geo->r_inner[s]));
    }
    for (int b=0;b<NB;b++) LW(cmf_dump_f64(fp,&sha,cs->nu[b]));
    for (int b=0;b<NB;b++) LW(cmf_dump_f64(fp,&sha,cs->dnu[b]));
    for (size_t q=0;q<(size_t)sel.n_shells_sel*NB;q++)
        LW(cmf_dump_f64(fp,&sha,chi_rep[q]));
    for (size_t q=0;q<(size_t)sel.n_shells_sel*NB;q++)
        LW(cmf_dump_f64(fp,&sha,chith_rep[q]));
    for (size_t q=0;q<(size_t)sel.n_shells_sel*NB;q++)
        LW(cmf_dump_f64(fp,&sha,eta_rep[q]));
    LW(cmf_dump_bytes(fp,&sha,disp,(size_t)NS*NB));
    LW(cmf_dump_bytes(fp,&sha,lrec,line_bytes));
    LW(cmf_dump_bytes(fp,&sha,rows,row_bytes));
#undef LW
    if (fclose(fp)) fail=1;
    free(rows); free(lrec); free(line_slot); cmf_levidx_free(&ix);
    if (fail) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] write %s: %s\n",path,strerror(errno));
        if (rename(path,quarantine)!=0)
            fprintf(stderr,"[CMF-LINEPOP][FAIL] quarantine %s: %s\n",
                    quarantine,strerror(errno));
        free(chi_rep); free(chith_rep); free(eta_rep); free(disp); return -1;
    }
    unsigned char digest[32]; char hex[65];
    cmf_sha256_final(&sha,digest);
    for (int i=0;i<32;i++) snprintf(hex+2*i,3,"%02x",digest[i]);
    hex[64]='\0';
    FILE *mf=fopen(manifest,"w");
    int sfail=(mf==NULL);
    if (!sfail) {
        sfail = fprintf(mf,"{\n  \"schema\": \"LCMFLP01-v1\",\n"
            "  \"sha256\": \"%s\",\n"
            "  \"iteration\": %d,\n  \"field_generation\": %d,\n"
            "  \"n_shells\": %d,\n  \"n_bins\": %d,\n"
            "  \"selected_shells\": %d,\n  \"selected_lines\": %d,\n"
            "  \"rows\": %ld,\n  \"row_bytes\": %d,\n"
            "  \"lambda_window_A\": [%.17g, %.17g],\n"
            "  \"tau_min\": 1e-12,\n"
            "  \"chi_line_roundtrip_bitwise\": %s,\n"
            "  \"chi_line_roundtrip_max_abs\": %.17g,\n"
            "  \"chi_line_th_comparable\": %s,\n"
            "  \"chi_line_th_roundtrip_bitwise\": %s,\n"
            "  \"chi_line_th_roundtrip_max_abs\": %.17g,\n"
            "  \"eta_line_epoch\": \"pre-EPAY, pre-split (assemble line loop)\",\n"
            "  \"epay_disposition_counts\": {\"legacy_source\": %ld, "
            "\"thick_exempt\": %ld, \"rate_shape_replaced\": %ld, "
            "\"scalar_rescaled\": %ld},\n"
            "  \"epay_scale_not_reproducible\": true,\n"
            "  \"gates\": {\"eps_phys\": %d, \"src_nlte\": %d, \"epay\": %d, "
            "\"epay_smin\": %d, \"epay_taubin\": %.17g, \"epay_hotf\": %.17g, "
            "\"line_eps\": %.17g, \"eps_uv\": %.17g},\n"
            "  \"clamp\": 0, \"fallback\": 0\n}\n",
            hex,iter,field_generation,NS,NB,sel.n_shells_sel,n_lines_sel,n_rows,
            CMF_LINEPOP_ROW_BYTES,sel.lam_lo,sel.lam_hi,
            chi_bitwise ? "true" : "false", chi_max_abs,
            chith_comparable ? "true" : "false",
            (chith_comparable && chith_bitwise) ? "true" : "false",
            chith_max_abs,
            disp_n[0],disp_n[1],disp_n[2],disp_n[3],
            eps_phys,src_nlte,epay,epay_smin,epay_taubin,epay_hotf,
            line_eps,eps_uv) < 0;
        if (fclose(mf)) sfail=1;
    }
    free(chi_rep); free(chith_rep); free(eta_rep); free(disp);
    if (sfail) {
        fprintf(stderr,"[CMF-LINEPOP][FAIL] write/close %s\n",manifest);
        if (rename(path,quarantine)!=0)
            fprintf(stderr,"[CMF-LINEPOP][FAIL] quarantine %s: %s\n",
                    quarantine,strerror(errno));
        return -1;
    }
    fprintf(stderr,"[CMF-LINEPOP] wrote %s iter=%d generation=%d shells=%d "
            "lines=%d rows=%ld (%.1f MiB) chi_line_bitwise=%d "
            "epay_rate_shape_cells=%ld sha256=%s\n",
            path,iter,field_generation,sel.n_shells_sel,n_lines_sel,n_rows,
            (double)(row_bytes+line_bytes)/1048576.0,chi_bitwise,disp_n[2],hex);
    return 0;
}

/* Stage 3.2 Rung 1 binary schema.  The row's final field is disposition. */
#define CMF_R1_ROW_BYTES 120U
#define CMF_R1_PRIMARY_DEFINED 0U
#define CMF_R1_PRIMARY_UNDEFINED_CHI_TOT_ZERO 1U
#define CMF_R1_EV_REACHED       0x01U
#define CMF_R1_EV_EPAY_ELIGIBLE 0x02U
#define CMF_R1_EV_BIN_THICK     0x04U
#define CMF_R1_EV_EPAY_GE2      0x08U
#define CMF_R1_EV_ACC_W_POS     0x10U
#define CMF_R1_EV_HOT           0x20U
#define CMF_R1_EV_BRANCH        0x40U
#define CMF_R1_EV_ASSEMBLED     0x80U

static int cmf_r1_path_exists(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    fclose(fp);
    return 1;
}

static int cmf_r1_secondary_values(int line, double ne, double Te, double tau,
                                   double eps_floor, double eps_cap,
                                   double *beta, double *eps0,
                                   double *eps_prime, double *eps_applied) {
    if (radeq_line_local_response(line,ne,Te,tau,beta,eps0) != 0)
        return -1;
    *eps_prime=radeq_line_eps_phys(line,ne,Te,tau);
    if (!isfinite(*eps_prime) || !(*eps_prime >= 0.0) || !(*eps_prime <= 1.0))
        return -1;
    *eps_applied=*eps_prime;
    /* Diagnostic replay of production's existing two comparisons, solely for
     * the secondary clamp ledger.  Primary rho never consumes these values. */
    if (*eps_applied < eps_floor) *eps_applied=eps_floor;
    if (*eps_applied > eps_cap)   *eps_applied=eps_cap;
    return isfinite(*eps_applied) ? 0 : -1;
}

int cmfgen_dump_stage32_rung1(const CMFGENState *cs, const Geometry *geo,
                              const OpacityState *opac,
                              const PlasmaState *plasma,
                              int iteration,
                              const char *path) {
    static const double lam_lo = 600.0, lam_hi = 3000.0;
    if (!cs || !geo || !opac || !plasma || !path || !*path || iteration < 0 ||
        cs->stage32_field_generation == 0 ||
        cs->stage32_lambda_generation != cs->stage32_field_generation ||
        cs->stage32_diag_failed || cs->n_shells <= 0 || cs->n_bins <= 0 ||
        geo->n_shells != cs->n_shells || opac->n_shells != cs->n_shells ||
        opac->n_lines <= 0 || !opac->line_list_nu || !opac->tau_sobolev ||
        (!plasma->n_electron && !opac->electron_density) ||
        !plasma->T_e || !cs->nu || !cs->dnu ||
        !cs->stage32_eta_pre_epay || !cs->stage32_boundary_eta ||
        !cs->stage32_line_eta || !cs->stage32_line_slot ||
        cs->stage32_line_slot_n != opac->n_lines ||
        !cs->stage32_epay_disposition || !cs->stage32_epay_evidence ||
        (cs->stage32_source_nlte && !opac->line_source_S) ||
        !(geo->time_explosion > 0.0) ||
        !isfinite(geo->time_explosion)) {
        fprintf(stderr,"[STAGE32-R1][FAIL] invalid state/path or chi/lambda generation mismatch: assembly=%llu lambda=%llu\n",
                (unsigned long long)(cs ? cs->stage32_field_generation : 0),
                (unsigned long long)(cs ? cs->stage32_lambda_generation : 0));
        return -1;
    }
    double eps_floor=1e-5, eps_cap=1.0;
    { const char *ef=getenv("LUMINA_CMFGEN_EPS_FLOOR");
      const char *ec=getenv("LUMINA_CMFGEN_EPS_CAP");
      if (ef) eps_floor=atof(ef);
      if (ec) eps_cap=atof(ec); }
    if (!isfinite(eps_floor) || !isfinite(eps_cap)) {
        fprintf(stderr,"[STAGE32-R1][FAIL] nonfinite production epsilon limits\n");
        return -1;
    }
    char outpath[4096], manifest[4096], quarantine[4096];
    int on=snprintf(outpath,sizeof(outpath),"%s.iter%03d",path,iteration);
    int mn=snprintf(manifest,sizeof(manifest),"%s.manifest.json",outpath);
    int qn=snprintf(quarantine,sizeof(quarantine),"%s.quarantine",outpath);
    if (on < 0 || mn < 0 || qn < 0 || (size_t)on >= sizeof(outpath) ||
        (size_t)mn >= sizeof(manifest) ||
        (size_t)qn >= sizeof(quarantine)) {
        fprintf(stderr,"[STAGE32-R1][FAIL] sidecar path too long\n");
        return -1;
    }
    if (cmf_r1_path_exists(outpath) || cmf_r1_path_exists(manifest)) {
        fprintf(stderr,"[STAGE32-R1][FAIL] refusing generation overwrite: %s\n",outpath);
        return -1;
    }

    const int NS=cs->n_shells, NB=cs->n_bins, NL=opac->n_lines;
    const double nu_lo=CM_C/(lam_hi*1.0e-8);
    const double nu_hi=CM_C/(lam_lo*1.0e-8);
    uint64_t n_rows=0, count[4]={0,0,0,0};
    uint64_t eps_applied_diff_rows=0, rho_undefined_rows=0;
    long double energy[4]={0.0L,0.0L,0.0L,0.0L};
    long double selected_energy=0.0L;

    /* Validation/count pass.  The per-line eta below is the same eta_l value
     * added to production eta_line.  It is neither inferred from cell opacity nor
     * apportioned from a bin total. */
    for (int s=0;s<NS;s++) {
        double ne=plasma->n_electron ? plasma->n_electron[s]
                                     : opac->electron_density[s];
        double Te=plasma->T_e[s];
        if (!(ne >= 0.0) || !(Te > 0.0) || !isfinite(ne) || !isfinite(Te)) {
            fprintf(stderr,"[STAGE32-R1][FAIL] invalid plasma shell=%d\n",s);
            return -1;
        }
        for (int l=0;l<NL;l++) {
            double tau=opac->tau_sobolev[(size_t)l*NS+s];
            double nu=opac->line_list_nu[l];
            if (!(nu >= nu_lo && nu <= nu_hi)) continue;
            if (!(tau > 0.0) || !isfinite(tau) || !isfinite(nu)) {
                fprintf(stderr,"[STAGE32-R1][FAIL] nonfinite line=%d shell=%d\n",l,s);
                return -1;
            }
            int b=(int)(log(nu/cs->nu_min)/cs->d_log_nu);
            if (b < 0 || b >= NB) {
                fprintf(stderr,"[STAGE32-R1][FAIL] UV line outside grid line=%d\n",l);
                return -1;
            }
            size_t q=(size_t)s*NB+b;
            int slot=cs->stage32_line_slot[l];
            if (slot < 0 || slot >= cs->stage32_selected_lines) {
                fprintf(stderr,"[STAGE32-R1][FAIL] selected-line map mismatch line=%d\n",l);
                return -1;
            }
            double beta=0.0, eps0=0.0, eps_prime=0.0, eps_applied=0.0;
            if (cmf_r1_secondary_values(l,ne,Te,tau,eps_floor,eps_cap,
                                        &beta,&eps0,&eps_prime,&eps_applied) != 0) {
                fprintf(stderr,"[STAGE32-R1][FAIL] secondary rates unavailable line=%d shell=%d\n",l,s);
                return -1;
            }
            double chi_es=cs->chi_es[q], chi_tot=cs->chi_tot[q];
            double lambda_star=cs->lambda_star[q];
            unsigned primary_status=(chi_tot == 0.0)
                ? CMF_R1_PRIMARY_UNDEFINED_CHI_TOT_ZERO
                : CMF_R1_PRIMARY_DEFINED;
            double rho=(primary_status == CMF_R1_PRIMARY_DEFINED)
                       ? (chi_es/chi_tot)*lambda_star : NAN;
            double e=cs->stage32_line_eta[(size_t)slot*NS+s]*cs->dnu[b];
            unsigned d=cs->stage32_epay_disposition[q];
            unsigned ev=cs->stage32_epay_evidence[q];
            /* lambda_star == 1.0 is a CORRECT binary64 value, not a defect: the
             * diagonal is 1 - escape, and in a cell thick enough that escape
             * underflows it rounds to exactly 1.  Rejecting it fails a valid
             * solution, which is the F3 defect class this rung already removed
             * once (run 189065 died here at iter 10 on lambda_star=1, rho=0.9658).
             * The closed interval is the contract; rho is bounded separately. */
            if (!(lambda_star >= 0.0) || !(lambda_star <= 1.0) ||
                !(chi_es >= 0.0) || !(chi_tot >= 0.0) ||
                (primary_status == CMF_R1_PRIMARY_DEFINED &&
                 (!(rho >= 0.0) || !isfinite(rho))) ||
                (primary_status == CMF_R1_PRIMARY_UNDEFINED_CHI_TOT_ZERO &&
                 !isnan(rho)) || !(e >= 0.0) || d > 3 ||
                !(ev & CMF_R1_EV_REACHED) || !(ev & CMF_R1_EV_BRANCH) ||
                (ev & CMF_R1_EV_ASSEMBLED) ||
                !isfinite(chi_es) || !isfinite(chi_tot) ||
                !isfinite(lambda_star) || !isfinite(e) ||
                !isfinite(cs->dnu[b]) || !(cs->dnu[b] > 0.0)) {
                fprintf(stderr,"[STAGE32-R1][FAIL] invalid production primary value line=%d shell=%d chi_es=%.17g chi_tot=%.17g lambda_star=%.17g rho=%.17g\n",
                        l,s,chi_es,chi_tot,lambda_star,rho);
                return -1;
            }
            if (eps_applied != eps_prime) eps_applied_diff_rows++;
            if (primary_status == CMF_R1_PRIMARY_UNDEFINED_CHI_TOT_ZERO)
                rho_undefined_rows++;
            n_rows++; count[d]++; energy[d]+=(long double)e;
            selected_energy+=(long double)e;
        }
    }
    if (!n_rows) {
        fprintf(stderr,"[STAGE32-R1][FAIL] no active UV line-shell rows\n");
        return -1;
    }

    int blo=(int)floor(log(nu_lo/cs->nu_min)/cs->d_log_nu);
    int bhi=(int)floor(log(nu_hi/cs->nu_min)/cs->d_log_nu);
    if (blo > bhi) { int t=blo; blo=bhi; bhi=t; }
    if (blo < 0 || bhi >= NB) {
        fprintf(stderr,"[STAGE32-R1][FAIL] census window outside grid\n");
        return -1;
    }
    long double authoritative=0.0L, boundary=0.0L;
    for (int s=0;s<NS;s++) for (int b=blo;b<=bhi;b++) {
        size_t q=(size_t)s*NB+b;
        double a=cs->stage32_eta_pre_epay[q]*cs->dnu[b];
        double x=cs->stage32_boundary_eta[q]*cs->dnu[b];
        if (!(a >= 0.0) || !(x >= 0.0) || !isfinite(a) || !isfinite(x)) {
            fprintf(stderr,"[STAGE32-R1][FAIL] invalid independent census cell\n");
            return -1;
        }
        authoritative+=(long double)a;
        boundary+=(long double)x;
    }
    long double closure=authoritative-selected_energy-boundary;

    FILE *fp=fopen(outpath,"wb");
    if (!fp) {
        fprintf(stderr,"[STAGE32-R1][FAIL] open %s: %s\n",outpath,strerror(errno));
        return -1;
    }
    CMFSHA256 sha; cmf_sha256_init(&sha); int fail=0;
    const unsigned char magic[8]={'L','C','M','F','R','1','0','1'};
#define R1W(call) do { if (!fail && (call)) fail=1; } while (0)
    R1W(cmf_dump_bytes(fp,&sha,magic,sizeof(magic)));
    R1W(cmf_dump_u32(fp,&sha,UINT32_C(0x01020304)));
    R1W(cmf_dump_u32(fp,&sha,UINT32_C(3)));
    R1W(cmf_dump_u64(fp,&sha,(uint64_t)iteration));
    R1W(cmf_dump_u64(fp,&sha,cs->stage32_field_generation));
    R1W(cmf_dump_u64(fp,&sha,cs->stage32_lambda_generation));
    R1W(cmf_dump_u32(fp,&sha,(uint32_t)NS));
    R1W(cmf_dump_u32(fp,&sha,(uint32_t)NB));
    R1W(cmf_dump_u32(fp,&sha,(uint32_t)NL));
    R1W(cmf_dump_u32(fp,&sha,0));
    R1W(cmf_dump_u64(fp,&sha,n_rows));
    R1W(cmf_dump_f64(fp,&sha,lam_lo));
    R1W(cmf_dump_f64(fp,&sha,lam_hi));
    R1W(cmf_dump_f64(fp,&sha,geo->time_explosion));
    R1W(cmf_dump_u32(fp,&sha,CMF_R1_ROW_BYTES));
    R1W(cmf_dump_u32(fp,&sha,0));
    uint64_t written=0;
    for (int s=0;s<NS;s++) {
        double ne=plasma->n_electron ? plasma->n_electron[s]
                                     : opac->electron_density[s];
        double Te=plasma->T_e[s];
        for (int l=0;l<NL;l++) {
            double tau=opac->tau_sobolev[(size_t)l*NS+s];
            double nu=opac->line_list_nu[l];
            if (!(nu >= nu_lo && nu <= nu_hi)) continue;
            int b=(int)(log(nu/cs->nu_min)/cs->d_log_nu);
            size_t q=(size_t)s*NB+b;
            double beta=0.0, eps0=0.0, eps_prime=0.0, eps_applied=0.0;
            if (cmf_r1_secondary_values(l,ne,Te,tau,eps_floor,eps_cap,
                                        &beta,&eps0,&eps_prime,&eps_applied) != 0) {
                fail=1; break;
            }
            int slot=cs->stage32_line_slot[l];
            double e=cs->stage32_line_eta[(size_t)slot*NS+s]*cs->dnu[b];
            double chi_es=cs->chi_es[q], chi_tot=cs->chi_tot[q];
            double lambda_star=cs->lambda_star[q];
            unsigned primary_status=(chi_tot == 0.0)
                ? CMF_R1_PRIMARY_UNDEFINED_CHI_TOT_ZERO
                : CMF_R1_PRIMARY_DEFINED;
            double rho=(primary_status == CMF_R1_PRIMARY_DEFINED)
                       ? (chi_es/chi_tot)*lambda_star : NAN;
            double Sl=cs->stage32_source_nlte
                    ? opac->line_source_S[(size_t)l*NS+s] : 0.0;
            if (Sl <= 0.0) Sl=cm_planck(nu,Te);
            unsigned ev=cs->stage32_epay_evidence[q];
            if (tau > 1e-12) ev|=CMF_R1_EV_ASSEMBLED;
            R1W(cmf_dump_u32(fp,&sha,(uint32_t)l));
            R1W(cmf_dump_u32(fp,&sha,(uint32_t)s));
            R1W(cmf_dump_u32(fp,&sha,(uint32_t)b));
            R1W(cmf_dump_u32(fp,&sha,primary_status));
            R1W(cmf_dump_f64(fp,&sha,CM_C/nu/1.0e-8));
            R1W(cmf_dump_f64(fp,&sha,tau));
            R1W(cmf_dump_f64(fp,&sha,beta));
            R1W(cmf_dump_f64(fp,&sha,eps0));
            R1W(cmf_dump_f64(fp,&sha,eps_prime));
            R1W(cmf_dump_f64(fp,&sha,eps_applied));
            R1W(cmf_dump_f64(fp,&sha,chi_es));
            R1W(cmf_dump_f64(fp,&sha,chi_tot));
            R1W(cmf_dump_f64(fp,&sha,lambda_star));
            R1W(cmf_dump_f64(fp,&sha,rho));
            R1W(cmf_dump_f64(fp,&sha,Sl));
            R1W(cmf_dump_f64(fp,&sha,e));
            R1W(cmf_dump_u32(fp,&sha,ev));
            R1W(cmf_dump_u32(fp,&sha,(uint32_t)cs->stage32_epay_disposition[q]));
            written++;
        }
        if (fail) break;
    }
#undef R1W
    if (fclose(fp)) fail=1;
    if (fail || written != n_rows) {
        fprintf(stderr,"[STAGE32-R1][FAIL] payload write/count (%llu/%llu)\n",
                (unsigned long long)written,(unsigned long long)n_rows);
        if (rename(outpath,quarantine) != 0)
            fprintf(stderr,"[STAGE32-R1][FAIL] quarantine %s: %s\n",
                    quarantine,strerror(errno));
        return -1;
    }
    unsigned char digest[32]; char hex[65]; cmf_sha256_final(&sha,digest);
    for (int i=0;i<32;i++) snprintf(hex+2*i,3,"%02x",digest[i]);
    hex[64]='\0';
    FILE *mf=fopen(manifest,"w");
    int sfail=(mf==NULL);
    long double etot=energy[0]+energy[1]+energy[2]+energy[3];
    if (!sfail) {
        sfail=fprintf(mf,
            "{\n  \"schema\": \"LCMFR101-v3\",\n  \"sha256\": \"%s\",\n"
            "  \"iteration\": %d,\n  \"field_generation\": %llu,\n"
            "  \"lambda_generation\": %llu,\n"
            "  \"rows\": %llu,\n  \"row_bytes\": %u,\n"
            "  \"lambda_window_A\": [600, 3000],\n"
            "  \"primary_definition\": \"(chi_es/chi_tot)*lambda_star from production arrays; undefined when chi_tot==0\",\n"
            "  \"generation_contract\": \"field_generation == lambda_generation\",\n"
            "  \"secondary_definition\": {\"beta\": \"(1-exp(-tau))/tau\", \"eps0_raw\": \"C/(C+A)\", \"eps_prime\": \"C/(C+A*beta)\", \"eps_applied\": \"eps_prime after production eps_floor then eps_cap\"},\n"
            "  \"eps_floor\": %.17g,\n  \"eps_cap\": %.17g,\n"
            "  \"eps_applied_diff_rows\": %llu,\n"
            "  \"rho_undefined_chi_tot_zero_rows\": %llu,\n"
            "  \"energy_definition\": \"production selected-line eta_l*dnu; tau numerator for tau<=1e-6\",\n"
            "  \"branch_evidence_definition\": \"branch-site bits: reached,eligible,thick,epay_ge2,acc_w_positive,hot,branch; row bit7=production-assembled\",\n"
            "  \"disposition_row_counts\": {\"legacy_source\": %llu, \"thick_exempt\": %llu, \"rate_shape_replaced\": %llu, \"scalar_rescaled\": %llu},\n"
            "  \"disposition_energy\": {\"legacy_source\": %.17Lg, \"thick_exempt\": %.17Lg, \"rate_shape_replaced\": %.17Lg, \"scalar_rescaled\": %.17Lg, \"total\": %.17Lg},\n"
            "  \"authoritative_pre_epay_window_energy\": %.17Lg,\n"
            "  \"boundary_nonselected_line_energy\": %.17Lg,\n"
            "  \"selected_row_energy\": %.17Lg,\n"
            "  \"closure_residual\": %.17Lg,\n"
            "  \"primary_clamp\": 0, \"primary_floor\": 0, \"primary_cap\": 0, \"primary_fallback\": 0\n}\n",
            hex,iteration,(unsigned long long)cs->stage32_field_generation,
            (unsigned long long)cs->stage32_lambda_generation,
            (unsigned long long)n_rows,
            CMF_R1_ROW_BYTES,
            eps_floor,eps_cap,
            (unsigned long long)eps_applied_diff_rows,
            (unsigned long long)rho_undefined_rows,
            (unsigned long long)count[0],(unsigned long long)count[1],
            (unsigned long long)count[2],(unsigned long long)count[3],
            energy[0],energy[1],energy[2],energy[3],etot,
            authoritative,boundary,selected_energy,closure) < 0;
        if (fclose(mf)) sfail=1;
    }
    if (sfail) {
        fprintf(stderr,"[STAGE32-R1][FAIL] manifest write/close\n");
        if (rename(outpath,quarantine) != 0)
            fprintf(stderr,"[STAGE32-R1][FAIL] quarantine %s: %s\n",
                    quarantine,strerror(errno));
        return -1;
    }
    fprintf(stderr,"[STAGE32-R1] wrote %s iter=%d generation=%llu rows=%llu "
            "rho_undefined=%llu eps_applied_diff=%llu rate_shape=%llu E_rate_shape=%.9Lg closure=%.9Lg sha256=%s\n",outpath,iteration,
            (unsigned long long)cs->stage32_field_generation,(unsigned long long)n_rows,
            (unsigned long long)rho_undefined_rows,
            (unsigned long long)eps_applied_diff_rows,
            (unsigned long long)count[2],energy[2],closure,hex);
    return 0;
}

int cmfgen_stage32_rung1_maybe_dump(const CMFGENState *cs,
                                    const Geometry *geo,
                                    const OpacityState *opac,
                                    const PlasmaState *plasma,
                                    int iteration,
                                    int n_iterations) {
    const char *path=getenv("LUMINA_STAGE32_RUNG1_DUMP");
    if (!path || !*path) return 0;
    const char *ie=getenv("LUMINA_STAGE32_RUNG1_ITER");
    char *end=NULL; long wanted=ie ? strtol(ie,&end,10) : -1;
    if (!ie || end==ie || *end || wanted < 0 || wanted >= n_iterations) {
        fprintf(stderr,"[STAGE32-R1][FAIL] LUMINA_STAGE32_RUNG1_ITER must be in [0,%d) (got %s)\n",
                n_iterations,ie ? ie : "<unset>");
        return -1;
    }
    if (iteration != wanted) return 0;
    return cmfgen_dump_stage32_rung1(cs,geo,opac,plasma,iteration,path);
}

static int cmf_r1_prepare_assembly(CMFGENState *cs, const OpacityState *opac) {
    if (!cs->stage32_eta_pre_epay) return 0;
    const int NS=cs->n_shells, NL=opac->n_lines;
    const double nu_lo=CM_C/(3000.0e-8), nu_hi=CM_C/(600.0e-8);
    if (!cs->stage32_line_slot) {
        int *slot=(int *)malloc((size_t)NL*sizeof(*slot));
        if (!slot) goto fail;
        int nsel=0;
        for (int l=0;l<NL;l++) {
            double nu=opac->line_list_nu[l];
            slot[l]=(isfinite(nu) && nu >= nu_lo && nu <= nu_hi) ? nsel++ : -1;
        }
        if (nsel <= 0 || (size_t)nsel > SIZE_MAX/(size_t)NS/sizeof(double)) {
            free(slot); goto fail;
        }
        double *line_eta=(double *)calloc((size_t)nsel*NS,sizeof(*line_eta));
        if (!line_eta) { free(slot); goto fail; }
        cs->stage32_line_slot=slot;
        cs->stage32_line_eta=line_eta;
        cs->stage32_line_slot_n=NL;
        cs->stage32_selected_lines=nsel;
    } else if (cs->stage32_line_slot_n != NL) {
        goto fail;
    }
    memset(cs->stage32_line_eta,0,
           (size_t)cs->stage32_selected_lines*NS*sizeof(double));
    memset(cs->stage32_boundary_eta,0,(size_t)NS*cs->n_bins*sizeof(double));
    if (cs->stage32_field_generation == UINT64_MAX) goto fail;
    cs->stage32_field_generation++;
    /* Assembly has replaced chi_es/chi_tot, so any diagonal left by an older
     * formal solve is deliberately invalid until cmfgen_solve_J completes. */
    cs->stage32_lambda_generation=0;
    return 0;
fail:
    cs->stage32_diag_failed=1;
    fprintf(stderr,"[STAGE32-R1][FAIL] diagnostic line snapshot allocation/lineage\n");
    return -1;
}

/* ------------------------------------------------------------ */
int cmfgen_init(CMFGENState *cs, const Geometry *geo)
{
    memset(cs, 0, sizeof(*cs));
    cs->n_shells = geo->n_shells;
    cs->n_bins   = NLTE_N_FREQ_BINS;
    cs->nu_min   = NLTE_NU_MIN;
    cs->nu_max   = NLTE_NU_MAX;
    cs->d_log_nu = log(cs->nu_max / cs->nu_min) / (double)cs->n_bins;

    int NB = cs->n_bins, NS = cs->n_shells;
    cs->nu          = malloc(sizeof(double) * NB);
    cs->dnu         = malloc(sizeof(double) * NB);
    cs->chi_es      = calloc((size_t)NS * NB, sizeof(double));
    cs->chi_abs     = calloc((size_t)NS * NB, sizeof(double));
    cs->chi_line    = calloc((size_t)NS * NB, sizeof(double));
    cs->chi_line_th = calloc((size_t)NS * NB, sizeof(double));
    cs->chi_line_cls= calloc((size_t)NS * NB, sizeof(double));
    cs->chi_tot     = calloc((size_t)NS * NB, sizeof(double));
    cs->S_fixed     = calloc((size_t)NS * NB, sizeof(double));
    cs->J           = calloc((size_t)NS * NB, sizeof(double));
    cs->eta_total_audit = calloc((size_t)NS * NB, sizeof(double));
    cs->lambda_star = calloc((size_t)NS * NB, sizeof(double));
    cs->t_color     = calloc((size_t)NS, sizeof(double));
    cs->tri_lo      = calloc((size_t)NS * NB, sizeof(double));
    cs->tri_up      = calloc((size_t)NS * NB, sizeof(double));
    cs->tri_r       = calloc((size_t)NS * NB, sizeof(double));
    { const char *r1=getenv("LUMINA_STAGE32_RUNG1_DUMP");
      if (r1 && *r1) {
          cs->stage32_eta_pre_epay=
              (double *)calloc((size_t)NS*NB,sizeof(double));
          cs->stage32_boundary_eta=
              (double *)calloc((size_t)NS*NB,sizeof(double));
          cs->stage32_epay_disposition=
              (unsigned char *)calloc((size_t)NS*NB,sizeof(unsigned char));
          cs->stage32_epay_evidence=
              (unsigned char *)calloc((size_t)NS*NB,sizeof(unsigned char));
          if (!cs->stage32_eta_pre_epay || !cs->stage32_boundary_eta ||
              !cs->stage32_epay_disposition || !cs->stage32_epay_evidence) {
              fprintf(stderr,"[STAGE32-R1][FAIL] diagnostic allocation failed\n");
              return -1;
          }
      } }
    if (!cs->nu || !cs->dnu || !cs->chi_es || !cs->chi_abs || !cs->chi_line ||
        !cs->chi_line_th || !cs->t_color ||
        !cs->tri_lo || !cs->tri_up || !cs->tri_r ||
        !cs->chi_tot || !cs->S_fixed || !cs->J || !cs->eta_total_audit ||
        !cs->lambda_star) {
        fprintf(stderr, "[CMFGEN] init alloc failed\n");
        return -1;
    }
    for (int b = 0; b < NB; ++b) {
        cs->nu[b]  = cs->nu_min * exp((b + 0.5) * cs->d_log_nu);
        cs->dnu[b] = cs->nu[b] * cs->d_log_nu;   /* log-grid bin width */
    }

    /* Tangent rays: one grazing each shell's r_outer, plus core rays packed
     * inside r_inner[0] for the diffusive inner boundary. */
    int n_core = 8;
    cs->n_rays = NS + n_core;
    cs->p_ray  = malloc(sizeof(double) * cs->n_rays);
    if (!cs->p_ray) { fprintf(stderr, "[CMFGEN] ray alloc failed\n"); return -1; }
    double r_in0 = geo->r_inner[0];
    for (int k = 0; k < n_core; ++k)            /* Gauss-like core spacing */
        cs->p_ray[k] = r_in0 * (k + 0.5) / (double)n_core;
    for (int s = 0; s < NS; ++s)
        cs->p_ray[n_core + s] = geo->r_outer[s];

    const char *d = getenv("LUMINA_RADEQ_DIAG");
    cs->diag = (d && atoi(d)) ? 1 : 0;
    cs->frozen_morph_eps = -1.0;   /* off until the post-convergence pass sets it */
    cs->cont_only = 0;             /* continuum-only J_inc pass off by default */
    return 0;
}

void cmfgen_free(CMFGENState *cs)
{
    if (!cs) return;
    free(cs->nu); free(cs->dnu); free(cs->chi_es); free(cs->chi_abs);
    free(cs->chi_line); free(cs->chi_line_th); free(cs->chi_line_cls);
    free(cs->chi_tot); free(cs->S_fixed); free(cs->J);
    free(cs->eta_total_audit);
    free(cs->stage32_eta_pre_epay);
    free(cs->stage32_boundary_eta);
    free(cs->stage32_line_eta);
    free(cs->stage32_line_slot);
    free(cs->stage32_epay_disposition);
    free(cs->stage32_epay_evidence);
    free(cs->lambda_star); free(cs->t_color);
    free(cs->tri_lo); free(cs->tri_up); free(cs->tri_r);
    free(cs->p_ray);
    memset(cs, 0, sizeof(*cs));
}

/* Stage 1 (transport-coupled T_e): radioactive deposition heating per shell
 * [erg/s/cm^3], registered before cmfgen_assemble so the formal-solve emissivity
 * can carry it (gate LUMINA_CMF_DEP_SOURCE). Setter avoids changing the assemble
 * signature (5 call sites). NULL => no injection (byte-identical). */
static const double *g_dep_heating = NULL;
static int g_dep_heating_n = 0;
void cmfgen_set_deposition(const double *heating_rate, int n_shells) {
    g_dep_heating = heating_rate; g_dep_heating_n = n_shells;
}

static void cmf_hash_u64(CMFSHA256 *sha, uint64_t value) {
    unsigned char b[8];
    for (int i=0;i<8;i++) b[i]=(unsigned char)(value>>(8*i));
    cmf_sha256_update(sha,b,sizeof(b));
}

static void cmf_hash_f64(CMFSHA256 *sha, double value) {
    uint64_t bits;
    memcpy(&bits,&value,sizeof(bits));
    cmf_hash_u64(sha,bits);
}

static void cmf_hash_f64_array(CMFSHA256 *sha, const double *values,
                               size_t count) {
    cmf_hash_u64(sha,values ? 1U : 0U);
    if (values) for (size_t i=0;i<count;i++) cmf_hash_f64(sha,values[i]);
}

static void cmf_hash_i32_array(CMFSHA256 *sha, const int *values,
                               size_t count) {
    cmf_hash_u64(sha,values ? 1U : 0U);
    if (values) for (size_t i=0;i<count;i++)
        cmf_hash_u64(sha,(uint64_t)(uint32_t)values[i]);
}

int cmfgen_emiss_ab_state_sha256(const CMFGENState *cs,
                                 const Geometry *geo,
                                 const OpacityState *opac,
                                 const BFOpacity *bf,
                                 const PlasmaState *plasma,
                                 const NLTEConfig *nlte,
                                 const AtomicData *atom,
                                 char out_hex[65]) {
    if (!cs || !geo || !opac || !plasma || !nlte || !atom || !out_hex ||
        cs->n_shells <= 0 || cs->n_bins <= 0 || opac->n_lines < 0 ||
        geo->n_shells != cs->n_shells || opac->n_shells != cs->n_shells ||
        atom->n_lines != opac->n_lines || !cs->J || !cs->nu || !cs->dnu ||
        !geo->r_inner || !geo->r_outer || !opac->line_list_nu ||
        !opac->tau_sobolev || !plasma->T_e) return -1;
    CMFSHA256 sha;
    cmf_sha256_init(&sha);
    static const char domain[]="LUMINA-E4-ASSEMBLY-INPUT-v1";
    cmf_sha256_update(&sha,domain,sizeof(domain)-1);
    size_t cells=(size_t)cs->n_shells*cs->n_bins;
    size_t line_cells=(size_t)opac->n_lines*cs->n_shells;
    cmf_hash_u64(&sha,(uint64_t)cs->n_shells);
    cmf_hash_u64(&sha,(uint64_t)cs->n_bins);
    cmf_hash_u64(&sha,(uint64_t)opac->n_lines);
    cmf_hash_u64(&sha,(uint64_t)atom->n_levels);
    cmf_hash_u64(&sha,(uint64_t)nlte->n_nlte_levels_total);
    cmf_hash_u64(&sha,(uint64_t)nlte->n_nlte_ions);
    cmf_hash_u64(&sha,(uint64_t)(cs->cont_only != 0));
    cmf_hash_f64(&sha,cs->frozen_morph_eps);
    cmf_hash_f64(&sha,geo->time_explosion);
    cmf_hash_f64_array(&sha,geo->r_inner,cs->n_shells);
    cmf_hash_f64_array(&sha,geo->r_outer,cs->n_shells);
    cmf_hash_f64_array(&sha,cs->nu,cs->n_bins);
    cmf_hash_f64_array(&sha,cs->dnu,cs->n_bins);
    cmf_hash_f64_array(&sha,cs->J,cells); /* lagged field consumed by EPAY */
    cmf_hash_f64_array(&sha,plasma->T_e,cs->n_shells);
    cmf_hash_f64_array(&sha,plasma->n_electron,cs->n_shells);
    cmf_hash_f64_array(&sha,opac->electron_density,cs->n_shells);
    cmf_hash_f64_array(&sha,opac->line_list_nu,opac->n_lines);
    cmf_hash_f64_array(&sha,opac->tau_sobolev,line_cells);
    cmf_hash_f64_array(&sha,opac->line_source_S,line_cells);
    cmf_hash_f64_array(&sha,bf ? bf->chi_bf : NULL,bf ? cells : 0);
    cmf_hash_f64_array(&sha,bf ? bf->eta_bf : NULL,bf ? cells : 0);
    cmf_hash_f64_array(&sha,nlte->nlte_level_populations,
                       (size_t)nlte->n_nlte_levels_total*cs->n_shells);
    cmf_hash_i32_array(&sha,nlte->nlte_ion_level_offset,
                       (size_t)nlte->n_nlte_ions+1);
    cmf_hash_i32_array(&sha,nlte->nlte_to_global_level,
                       nlte->n_nlte_levels_total);
    cmf_hash_i32_array(&sha,nlte->nlte_line_map,opac->n_lines);
    cmf_hash_i32_array(&sha,nlte->global_to_nlte_level,atom->n_levels);
    cmf_hash_f64_array(&sha,atom->line_A_ul,opac->n_lines);
    cmf_hash_i32_array(&sha,atom->line_atomic_number,opac->n_lines);
    cmf_hash_i32_array(&sha,atom->line_ion_number,opac->n_lines);
    cmf_hash_i32_array(&sha,atom->line_level_upper,opac->n_lines);
    cmf_hash_i32_array(&sha,atom->level_num,atom->n_levels);
    cmf_hash_i32_array(&sha,atom->level_Z,atom->n_levels);
    cmf_hash_i32_array(&sha,atom->level_ion,atom->n_levels);
    cmf_hash_u64(&sha,(uint64_t)g_dep_heating_n);
    cmf_hash_f64_array(&sha,g_dep_heating,
                       g_dep_heating && g_dep_heating_n == cs->n_shells
                       ? cs->n_shells : 0);
    unsigned char digest[32];
    cmf_sha256_final(&sha,digest);
    for(int i=0;i<32;i++) snprintf(out_hex+2*i,3,"%02x",digest[i]);
    out_hex[64]='\0';
    return 0;
}

void cmfgen_emiss_ab_stats_free(CMFGENEmissABStats *stats) {
    if (!stats) return;
    free(stats->undefined_reason);
    free(stats->undefined_shell_count);
    free(stats->undefined_a_emissivity_by_band);
    free(stats->undefined_a_emissivity_by_shell);
    memset(stats,0,sizeof(*stats));
}

static const char *cmf_emiss_undef_text(unsigned reason) {
    if (reason & CMF_EMISS_UNDEF_NO_NLTE_LINE) return "population_not_tracked";
    if (reason & CMF_EMISS_UNDEF_UPPER_LEVEL) return "upper_level_unmapped";
    if (reason & CMF_EMISS_UNDEF_A_UL) return "A_ul_missing_or_invalid";
    if (reason & CMF_EMISS_UNDEF_POPULATION) return "population_missing_or_invalid";
    return "defined";
}

int cmfgen_write_emiss_ab_undefined(const CMFGENEmissABStats *stats,
                                     const AtomicData *atom,
                                     const char *path) {
    if (!stats || !atom || !path || !*path || stats->n_lines != atom->n_lines ||
        !stats->undefined_reason || !stats->undefined_shell_count) return -1;
    FILE *fp=fopen(path,"w");
    if (!fp) return -1;
    int fail=fprintf(fp,"line_id,Z,ion,lower_level,upper_level,A_ul_s-1,"
                        "undefined_shell_cells,reason_mask,reason\n") < 0;
    for (int l=0;l<stats->n_lines && !fail;l++) {
        unsigned reason=stats->undefined_reason[l];
        if (!reason) continue;
        double A=atom->line_A_ul ? atom->line_A_ul[l] : NAN;
        fail=fprintf(fp,"%d,%d,%d,%d,%d,%.17g,%u,%u,%s\n",l,
            atom->line_atomic_number ? atom->line_atomic_number[l] : -1,
            atom->line_ion_number ? atom->line_ion_number[l] : -1,
            atom->line_level_lower ? atom->line_level_lower[l] : -1,
            atom->line_level_upper ? atom->line_level_upper[l] : -1,
            A,stats->undefined_shell_count[l],reason,
            cmf_emiss_undef_text(reason)) < 0;
    }
    if (fclose(fp)) fail=1;
    return fail ? -1 : 0;
}

typedef struct {
    const NLTEConfig *nlte;
    const AtomicData *atom;
    CMFGENEmissABStats *stats;
    int *upper_nlte;
    unsigned char *active_seen;
    int retain_undefined_a;
    double *a_total_cell;
    double *a_covered_cell;
    double *a_undefined_cell;
} CMFEmissBContext;

/* ------------------------------------------------------------ */
/* Assemble per (shell,bin): electron-scatter, thermal bf/ff absorption,
 * expansion line opacity, and the scattering-independent source S_fixed. */
static void cmfgen_assemble_impl(CMFGENState *cs, const Geometry *geo,
                     const OpacityState *opac, BFOpacity *bf,
                     const PlasmaState *plasma, CMFEmissBContext *emiss_b)
{
    int NB = cs->n_bins, NS = cs->n_shells;
    int n_lines = opac->n_lines;
    double t_exp = geo->time_explosion;
    double inv_ct = 1.0 / (CM_C * t_exp);   /* expansion-opacity prefactor */

    /* Stage 1: thermalised radioactive deposition into the transport source.
     * Inject eta_dep,nu = kappa*B_nu(T_e) so 4pi*Int eta_dep dnu = frac*H_gamma[s],
     * with kappa normalised on the ACTUAL bin grid (kappa = frac*H_gamma/
     * (4pi*Sum_b B_nu*dnu)) so the injected power equals frac*H_gamma exactly
     * (no analytic-sigma-T^4 grid-truncation error). Added to S_fixed as
     * eta_dep/chi_tot below. Gate LUMINA_CMF_DEP_SOURCE (default off => no-op). */
    static int dep_src = -1; static double dep_frac = 1.0;
    if (dep_src < 0) {
        const char *e = getenv("LUMINA_CMF_DEP_SOURCE");
        dep_src = (e && atoi(e)) ? 1 : 0;
        const char *fe = getenv("LUMINA_CMF_DEP_FRAC");
        if (fe) dep_frac = atof(fe);
    }

    memset(cs->chi_line, 0, sizeof(double) * (size_t)NS * NB);
    memset(cs->chi_line_th, 0, sizeof(double) * (size_t)NS * NB);
    memset(cs->chi_line_cls, 0, sizeof(double) * (size_t)NS * NB);
    if (cs->stage32_epay_disposition) {
        memset(cs->stage32_epay_disposition,0xff,(size_t)NS*NB);
        memset(cs->stage32_epay_evidence,0,(size_t)NS*NB);
    }
    /* line emissivity accumulator reuses chi_tot scratch before it is summed */
    double *eta_line = cs->chi_tot;

    /* PHYSICAL per-line destruction probability (LUMINA_CMFGEN_LINE_EPS_PHYS=1):
     * eps_l = C_ul/(C_ul + A_ul*beta_esc(tau_l)) per line — the measured FUV
     * blockers are ground/metastable resonance lines (eps_phys~1e-3..1e-2) that
     * the legacy path treats as FULLY thermal, pinning inner FUV J to the local
     * cold B(T_e). The thermal channel chi_line_th and its emissivity are
     * accumulated PER LINE (no bin-average eps mixing, codex review); the
     * scattering remainder joins chi_es in the combine loop. Knobs:
     * LUMINA_CMFGEN_EPS_FLOOR (1e-5), LUMINA_CMFGEN_EPS_CAP (1.0). */
    int eps_phys = 0;
    double eps_floor = 1e-5, eps_cap = 1.0;
    /* NLTE line-source consumption gate (LUMINA_CMFGEN_SRC_NLTE=1, default
     * OFF). The pop-ratio S_l data is correct physics (fluorescence) but is
     * EXPLOSIVE under the operator split: saturated FUV lines in cold gas
     * carry S_l up to ~1e16 x the local J (run 165510: J[mid,500] 1.8e-18 ->
     * 4.2e-2 at the first NLTE-fed iter), eta_line -> J -> T_e no-root pins
     * at 2*T_rad. Until the eps-weighted thermal channel / A4 in-Newton
     * source absorbs it, the B(T_e) fallback (the de-facto thermostat the
     * champion metrics rest on) stays the default. */
    int src_nlte = 0;
    { const char *sn = getenv("LUMINA_CMFGEN_SRC_NLTE");
      if (sn && atoi(sn)) src_nlte = 1; }
    { const char *ep = getenv("LUMINA_CMFGEN_LINE_EPS_PHYS");
      if (ep && atoi(ep)) eps_phys = 1;
      const char *ef = getenv("LUMINA_CMFGEN_EPS_FLOOR");
      if (ef) eps_floor = atof(ef);
      const char *ec = getenv("LUMINA_CMFGEN_EPS_CAP");
      if (ec) eps_cap = atof(ec); }
    /* TRANSFER-ONLY eps (LUMINA_CMFGEN_LINE_EPS_UV, e.g. 0.03 = branch-like
     * interim; codex ruling 2026-06-11): the scattering share enters ONLY the
     * formal solve (chi_es), while the RE/Newton closure keeps the FULL
     * chi_line with the cooling-only form chi_line*(min(J,B)-B) — the
     * operator-split T_e anchor (audited: nothing else owns the root at the
     * first NLTE iters) without FUV line pumping heating the gas. */
    double eps_uv = -1.0;
    { const char *eu = getenv("LUMINA_CMFGEN_LINE_EPS_UV");
      if (eu) eps_uv = atof(eu); }
    if (eps_phys) eps_uv = -1.0;              /* phys mode wins (experimental) */
    cs->chi_line_re = (eps_uv > 0.0) ? cs->chi_line : cs->chi_line_th;
    memset(eta_line, 0, sizeof(double) * (size_t)NS * NB);
    if (cmf_r1_prepare_assembly(cs,opac) == 0 && cs->stage32_eta_pre_epay)
        cs->stage32_source_nlte=src_nlte;

    /* Expansion (Sobolev-binned) line opacity + emissivity.
     *   chi_line[bin] = sum_{l in bin} (1-e^{-tau_l}) * nu_l/(c t_exp dnu_bin)
     *   eta_line[bin] = sum_l eta_l, including the production eps_phys factor
     * S_l = line_source_S if >0 else B_nu(T_e) (thermalised fallback). */
    for (int s = 0; s < NS; ++s) {
        double Te = plasma->T_e[s];
        double ne_s = plasma->n_electron ? plasma->n_electron[s]
                                         : opac->electron_density[s];
        for (int l = 0; l < n_lines; ++l) {
            double tau = opac->tau_sobolev[(size_t)l * NS + s];
            if (tau <= 1e-12) continue;
            double nu_l = opac->line_list_nu[l];
            if (nu_l <= cs->nu_min || nu_l >= cs->nu_max) continue;
            int b = (int)floor(log(nu_l / cs->nu_min) / cs->d_log_nu);
            if (b < 0 || b >= NB) continue;
            double frac = (tau > 1e-6) ? -expm1(-tau) : tau;   /* 1-e^{-tau} */
            double w = frac * nu_l * inv_ct / cs->dnu[b];      /* cm^-1 */
            double Sl = src_nlte ? opac->line_source_S[(size_t)l * NS + s]
                                 : 0.0;
            if (Sl <= 0.0) Sl = cm_planck(nu_l, Te);
            size_t idx = (size_t)s * NB + b;
            cs->chi_line[idx] += w;
            double eta_l = 0.0;
            if (!emiss_b && eps_phys) {
                double el = radeq_line_eps_phys(l, ne_s, Te, tau);
                if (el < 0.0) el = 1.0;        /* table not built: thermal */
                if (el < eps_floor) el = eps_floor;
                if (el > eps_cap)   el = eps_cap;
                cs->chi_line_th[idx] += w * el;
                eta_l = w * el * Sl;
                eta_line[idx] += eta_l;        /* thermal-channel eta */
                /* A4 SRC_BLEND closure weight: exact two-level+Sobolev gas
                 * coupling. Jbar_l=(1-beta)S_l+beta*J_bin with S=(1-eps)Jbar
                 * +eps*B gives net line heating chi_l*eps*beta/(eps+beta)
                 * *(J_bin-B): saturated lines (beta<<eps, trapped field
                 * relaxed to local B) drop out — the 1/eps over-count AND the
                 * hot-continuum-fed-to-trapped-lines defect both die here. */
                {
                    double be = (tau > 700.0) ? 1.0 / tau
                              : (tau > 1e-6) ? (1.0 - exp(-tau)) / tau : 1.0;
                    cs->chi_line_cls[idx] += w * (el * be) / (el + be - el * be + 1e-300);
                }
            } else if (!emiss_b) {
                eta_l = w * Sl;
                eta_line[idx] += eta_l;
            } else {
                /* E4 B lane: opacity/destruction bookkeeping is byte-for-byte
                 * the production calculation above; only the line emissivity
                 * accumulator changes.  The direct CMFGEN-style emissivity is
                 * projected as a delta line into the same coarse bin. */
                double el = 1.0;
                if (eps_phys) {
                    el = radeq_line_eps_phys(l, ne_s, Te, tau);
                    if (el < 0.0) el = 1.0;
                    if (el < eps_floor) el = eps_floor;
                    if (el > eps_cap) el = eps_cap;
                    cs->chi_line_th[idx] += w * el;
                    {
                        double be = (tau > 700.0) ? 1.0 / tau
                                  : (tau > 1e-6) ? (1.0 - exp(-tau)) / tau : 1.0;
                        cs->chi_line_cls[idx] += w * (el * be) /
                            (el + be - el * be + 1e-300);
                    }
                }
                CMFGENEmissABStats *st=emiss_b->stats;
                emiss_b->active_seen[l]=1;
                st->active_line_shell_count++;
                double eta_a = eps_phys ? w * el * Sl : w * Sl;
                emiss_b->a_total_cell[idx] += eta_a;
                unsigned reason=st->undefined_reason[l] &
                                ~CMF_EMISS_UNDEF_POPULATION;
                int upper=emiss_b->upper_nlte[l];
                if (upper < 0) {
                    if (!reason) reason=CMF_EMISS_UNDEF_UPPER_LEVEL;
                }
                double A=(emiss_b->atom->line_A_ul && l < emiss_b->atom->n_lines)
                         ? emiss_b->atom->line_A_ul[l] : NAN;
                if (!(A > 0.0) || !isfinite(A)) reason|=CMF_EMISS_UNDEF_A_UL;
                double n_upper=NAN;
                if (!reason) {
                    if (emiss_b->nlte->nlte_level_populations)
                        n_upper=emiss_b->nlte->nlte_level_populations[
                            (size_t)upper*NS+s];
                    if (!isfinite(n_upper) || n_upper < 0.0)
                        reason|=CMF_EMISS_UNDEF_POPULATION;
                }
                if (reason) {
                    st->undefined_reason[l]|=(unsigned char)reason;
                    st->undefined_shell_count[l]++;
                    st->undefined_line_shell_count++;
                    emiss_b->a_undefined_cell[idx] += eta_a;
                } else {
                    if (l == st->seed_line && s == st->seed_shell) {
                        n_upper *= st->seed_factor;
                        st->seed_hits++;
                    }
                    eta_line[idx] += (CM_H*nu_l/(4.0*M_PI_VAL))*A*n_upper /
                                     cs->dnu[b];
                    st->defined_line_shell_count++;
                    emiss_b->a_covered_cell[idx] += eta_a;
                }
            }
            if (!emiss_b && cs->stage32_line_slot && !cs->stage32_diag_failed) {
                int slot=cs->stage32_line_slot[l];
                if (slot >= 0)
                    cs->stage32_line_eta[(size_t)slot*NS+s] += eta_l;
                else {
                    const double r1_lo=CM_C/(3000.0e-8);
                    const double r1_hi=CM_C/(600.0e-8);
                    int blo=(int)floor(log(r1_lo/cs->nu_min)/cs->d_log_nu);
                    int bhi=(int)floor(log(r1_hi/cs->nu_min)/cs->d_log_nu);
                    if (b == blo || b == bhi)
                        cs->stage32_boundary_eta[idx] += eta_l;
                }
            }
        }
    }

    /* Line-forest scattering split (LUMINA_CMFGEN_LINE_EPS = eps in (0,1]):
     * two-level treatment of the binned forest, S_line=(1-eps)*J+eps*B. The
     * scattering remainder (1-eps)*chi_line joins chi_es so the ALI closure
     * transports the photospheric FUV color outward (root of the inner
     * Gamma(Mg I/Si I) 5-1000x deficit: J was thermalized to the LOCAL cold
     * B within <1 shell by the pure-absorption frozen-source treatment).
     * Applied only in bins with chi_line > GATE*chi_abs (line-dominated;
     * default gate 1.0). eps<=0 or unset -> byte-identical legacy path. */
    double line_eps = -1.0, line_gate = 1.0;
    { const char *le = getenv("LUMINA_CMFGEN_LINE_EPS");
      if (le) line_eps = atof(le);
      const char *lg = getenv("LUMINA_CMFGEN_LINE_EPS_GATE");
      if (lg) line_gate = atof(lg); }
    long n_split = 0;

    /* LUMINA_CMF_EPAY=1: energy-paid thermal emission (deterministic kpkt
     * mirror / per-shell transport radiative equilibrium). The thermal source
     * (chi_a*B + eta_line) is rescaled per shell so its frequency-integrated
     * power equals what the gas actually absorbs from the (lagged) field:
     *   scale_s = [Sum_b (chi_a+chi_line_th) J dnu] / [Sum_b (chi_a B+eta_ln) dnu]
     * The deposition injection kappa_dep*B is already exactly H_dep and stays
     * unscaled. Fixed point = radiative equilibrium (emit = abs + dep), the
     * same closure CMFGEN's global linearization and ARTIS's e-packet
     * bookkeeping enforce — a conservation constraint, not a tuning knob.
     * iter0 (J=0) => thermal dark start (physical: the ejecta history never
     * visits the hot state); the unpaid mutual-illumination bath (hot band
     * s36-40, ledger 1e6-1e7x) cannot ignite because a shell can never emit
     * more than it absorbed + deposition. */
    static int epay = -1;
    static int epay_smin = 0;   /* diagnostic: EPAY only for s >= smin */
    static double epay_tau = 2.0;
    static double epay_taubin = 1.0;   /* per-bin thick exemption threshold */
    if (epay < 0) { const char *e = getenv("LUMINA_CMF_EPAY");
                    epay = e ? atoi(e) : 0;
                    if (epay < 0) epay = 0;
                    const char *et = getenv("LUMINA_CMF_EPAY_TAU");
                    if (et) epay_tau = atof(et);
                    const char *es = getenv("LUMINA_CMF_EPAY_SMIN");
                    if (es) epay_smin = atoi(es);
                    const char *tb = getenv("LUMINA_CMF_EPAY_TAUBIN");
                    if (tb) epay_taubin = atof(tb);   /* <=0: no exemption */
                    if (epay) printf("[CMF-EPAY] energy-paid thermal emission ON"
                                     " (tau_es < %.2f only; thick=LTE legacy)\n",
                                     epay_tau); }
    double epay_scale_dbg[4] = {0,0,0,0};
    /* inward electron-scattering optical depth: EPAY applies only to thin
     * shells (tau_es < EPAY_TAU). At depth LTE holds, chi*B IS the honest
     * emission and the books close naturally — enforcing the lagged-J scale
     * there just recreates Lambda-iteration slowness (epay1: s0 stuck 0.68x). */
    double *epay_tau_arr = NULL;
    if (epay) {
        /* LUMINA_CMF_EPAY_TAUEFF (default 1e4): Planck-weighted effective
         * absorption depth sqrt(chi_abs*chi_tot)*dr, accumulated inward.
         * The toy06 profile has a 26x cliff exactly at the diffusive-core
         * boundary (s4: 2.6e4 -> s5: 1.0e3): the core keeps the legacy LTE
         * chi*B source (books close at ~1.1 naturally; EPAY's lagged-J scale
         * would Lambda-stall there), everything outside is EPAY-enforced.
         * Replaces the diagnostic shell-index gate EPAY_SMIN. */
        static double epay_taueff = 0.0;   /* 0 = disabled (gate by EPAY_SMIN;
            * the toy06 diffusive-core boundary s4/s5 is a verified 26x cliff
            * in offline emission-weighted tau_eff — scripts note in design
            * doc). The in-code coarse Planck-weighted variant under-measures
            * the core (misses the UV/line-dominated bins): epay9 regression
            * s0-4 -40%. Opt-in via LUMINA_CMF_EPAY_TAUEFF until the measure
            * is emission-weighted. */
        { static int te_once = 0;
          if (!te_once) { te_once = 1;
              const char *tf = getenv("LUMINA_CMF_EPAY_TAUEFF");
              if (tf) epay_taueff = atof(tf); } }
        epay_tau_arr = (double *)calloc(NS, sizeof(double));
        double acc_tau = 0.0;
        if (epay_taueff > 0.0)
        for (int s2 = NS - 1; s2 >= 0; --s2) {
            double Te2 = plasma->T_e[s2];
            double dr2 = geo->r_outer[s2] - geo->r_inner[s2];
            double wsum = 0.0, ca = 0.0, ct = 0.0;
            for (int b2 = 0; b2 < NB; b2 += 8) {   /* coarse Planck weights */
                size_t i2 = (size_t)s2 * NB + b2;
                double wgt = cm_planck(cs->nu[b2], Te2) * cs->dnu[b2];
                wsum += wgt;
                ca += wgt * cs->chi_abs[i2];
                ct += wgt * cs->chi_tot[i2];
            }
            if (wsum > 0.0) { ca /= wsum; ct /= wsum; }
            acc_tau += sqrt((ca > 0 ? ca : 0) * (ct > 0 ? ct : 0)) *
                       (dr2 > 0.0 ? dr2 : 0.0);
            epay_tau_arr[s2] = acc_tau;
        }
        if (epay_taueff > 0.0) epay_tau = epay_taueff;
        /* else: arr stays 0 < epay_tau — tau gate passes, EPAY_SMIN gates */
    }

    /* electron scattering + bf/ff thermal absorption + combine. */
    for (int s = 0; s < NS; ++s) {
        double Te  = plasma->T_e[s];
        double n_e = plasma->n_electron ? plasma->n_electron[s]
                                        : opac->electron_density[s];
        double chi_e = n_e * CM_SIGMA_T;
        /* Stage 1 deposition: kappa_dep = frac*H_gamma / (4pi*Sum_b B_nu*dnu). */
        double kappa_dep = 0.0;
        double acc_emit = 0.0, acc_abs = 0.0;   /* EPAY per-shell books */
        double acc_w = 0.0, acc_dep = 0.0;      /* EPAY=2 rate-shape norm */
        if (dep_src && g_dep_heating && s < g_dep_heating_n &&
            g_dep_heating[s] > 0.0 && Te > 0.0) {
            double bnorm = 0.0;
            for (int b = 0; b < NB; ++b) bnorm += cm_planck(cs->nu[b], Te) * cs->dnu[b];
            if (bnorm > 0.0)
                kappa_dep = dep_frac * g_dep_heating[s] / (4.0 * M_PI_VAL * bnorm);
        }
        for (int b = 0; b < NB; ++b) {
            size_t idx = (size_t)s * NB + b;
            double nu = cs->nu[b];
            double B  = cm_planck(nu, Te);
            double chi_bf = bf ? bf_get_chi(bf, s, nu) : 0.0;
            if (chi_bf < 0.0) chi_bf = 0.0;
            /* free-free (Kramers, gaunt~1): chi_ff = 3.69e8 Z^2 n_e n_i T^-1/2
             * nu^-3 (1-e^{-h nu/kT}); approximate n_i ~ n_e, Z^2~1. */
            double chi_ff = 0.0;
            if (Te > 0.0 && nu > 0.0) {
                double gaunt = 1.0;
                double stim  = -expm1(-CM_H * nu / (CM_KB * Te));
                chi_ff = 3.692e8 * gaunt * n_e * n_e /
                         (sqrt(Te) * nu * nu * nu) * stim;
                if (chi_ff < 0.0) chi_ff = 0.0;
            }
            double chi_a   = chi_bf + chi_ff;        /* thermal true abs */
            /* continuum-only J_inc pass: drop ALL line opacity + emissivity so
             * the solved field is the bare chi_es+bf/ff continuum (no forest
             * blanketing, not self-referential with the line source). */
            double chi_ln  = cs->cont_only ? 0.0 : cs->chi_line[idx];
            double eta_ln  = cs->cont_only ? 0.0 : eta_line[idx];  /* in scratch */
            double chi_t   = chi_e + chi_a + chi_ln;

            double chi_ln_th = chi_ln;               /* legacy: all thermal */
            double a_line_factor = cs->cont_only ? 0.0 : 1.0;
            if (cs->frozen_morph_eps >= 0.0 &&
                chi_ln > line_gate * chi_a) {
                /* frozen-plasma morphology pass: force the forest-dominated
                 * bins to scatter with destruction prob eps (0 = pure
                 * coherent). The scattering remainder (1-eps)*chi_line joins
                 * chi_es so the ALI solve carries the photospheric field out;
                 * the per-line scattering source S_l=(1-eps)*Jbar+eps*B is
                 * written into opacity->line_source_S AFTER this solve. */
                double fe = cs->frozen_morph_eps;
                chi_ln_th = fe * chi_ln;
                if (!emiss_b) eta_ln *= fe;
                a_line_factor *= fe;
                n_split++;
            } else if (eps_phys) {
                chi_ln_th = cs->chi_line_th[idx];    /* per-line accumulated */
                if (chi_ln_th > 0.0 || chi_ln > 0.0) n_split++;
            } else if (eps_uv > 0.0 && eps_uv <= 1.0 &&
                       chi_ln > line_gate * chi_a) {
                /* transfer-only split: S_fixed/transfer see the eps_uv share,
                 * the RE closure sees FULL chi_line via cs->chi_line_re. */
                chi_ln_th = eps_uv * chi_ln;
                if (!emiss_b) eta_ln *= eps_uv;
                a_line_factor *= eps_uv;
                n_split++;
            } else if (line_eps > 0.0 && line_eps <= 1.0 &&
                       chi_ln > line_gate * chi_a) {
                chi_ln_th = line_eps * chi_ln;
                if (!emiss_b)
                    eta_ln *= line_eps; /* A-lane thermal share only */
                a_line_factor *= line_eps;
                n_split++;
            }
            if (emiss_b) {
                CMFGENEmissABStats *st=emiss_b->stats;
                double cell_power_scale=a_line_factor*cs->dnu[b];
                double undefined_eta=a_line_factor*
                                     emiss_b->a_undefined_cell[idx];
                st->a_reference_line_power +=
                    emiss_b->a_total_cell[idx]*cell_power_scale;
                st->a_reference_covered_line_power +=
                    emiss_b->a_covered_cell[idx]*cell_power_scale;
                st->a_reference_undefined_line_power +=
                    undefined_eta*cs->dnu[b];
                st->undefined_a_emissivity_by_band[b] +=
                    undefined_eta*cs->dnu[b];
                st->undefined_a_emissivity_by_shell[s] +=
                    undefined_eta*cs->dnu[b];
                if (emiss_b->retain_undefined_a)
                    eta_ln += undefined_eta;
            }
            cs->chi_es[idx]      = chi_e + (chi_ln - chi_ln_th);
            cs->chi_abs[idx]     = chi_a;
            cs->chi_line_th[idx] = chi_ln_th;
            /* S_fixed = (chi_abs*B + thermal line emissivity)/chi_tot; the
             * scattering share enters the solve as r*J, NOT here (no double
             * count of the line source). */
            if (cs->stage32_eta_pre_epay)
                cs->stage32_eta_pre_epay[idx]=eta_ln;
            cs->S_fixed[idx] = (chi_t > 0.0)
                             ? (chi_a * B + eta_ln) / chi_t : 0.0;
            /* EPAY books: thermal emitted vs absorbed (lagged J), per shell.
             * Per-bin THICK exemption (Kirchhoff limit): a locally thick bin
             * (tau_bin = (chi_a+chi_l,th)*dr > EPAY_TAUBIN) reabsorbs its own
             * emission — it cannot export unpaid energy, and its source MUST
             * relax to B (detailed balance) or the diffusive field dies (the
             * measured s5 J(16eV) = B/340, x3e5 below pre-EPAY: Si/S/Co stuck
             * at stage II -> the 5700-6000A blanket). Thick bins keep the
             * legacy chi*B source and are excluded from the books; the unpaid
             * -lamp problem only ever lived in EXPORTING (thin) bins. */
            int bin_thick = 0;
            if (epay) {
                double dr_s = geo->r_outer[s] - geo->r_inner[s];
                bin_thick = ((chi_a + chi_ln_th) * dr_s > epay_taubin);
            }
            if (epay && !bin_thick) {
                acc_emit += (chi_a * B + eta_ln) * cs->dnu[b];
                acc_abs  += (chi_a + chi_ln_th) * cs->J[idx] * cs->dnu[b];
                if (epay >= 2) {
                    /* rate-side emission shape: Milne spontaneous recombination
                     * (bf_get_eta, LUMINA_CMF_BF_MILNE=2 builder) + collisional
                     * line emissivity chi_l,th*B (= n_l C_lu h nu by the
                     * two-level identity — NOT eta_ln, whose macroatom
                     * line_source_S garbage bins (S_l/B~1e48 saga) concentrate
                     * the whole paid budget into one scattering-trapped IR bin
                     * and close a gain>1 loop: the epay6 J=1e107 runaway). */
                    acc_w   += ((bf ? bf_get_eta(bf, s, nu) : 0.0)
                                + chi_ln_th * B) * cs->dnu[b];
                    acc_dep += kappa_dep * B * cs->dnu[b];
                }
            }
            /* Stage 1: add the thermalised deposition emissivity eta_dep=kappa*B
             * to the source (eta_dep/chi_tot). No-op when the gate is off.
             * Under EPAY this is deferred to the rescale pass below. */
            if ((!epay || s < epay_smin ||
                 (epay_tau_arr && epay_tau_arr[s] >= epay_tau)) &&
                kappa_dep > 0.0 && chi_t > 0.0)
                cs->S_fixed[idx] += kappa_dep * B / chi_t;
            /* chi_tot scratch now overwritten with the real total */
            cs->chi_tot[idx] = chi_t;
        }
        if (epay && s >= epay_smin && epay_tau_arr[s] < epay_tau) {
            double scale = (acc_emit > 0.0) ? acc_abs / acc_emit : 0.0;
            /* rate-shape only in the NLTE lamp regime T_e >> T_rad: there
             * chi*B(T_e) is the proven unpaid-lamp form and Milne+lines is
             * honest. Near LTE (T_e ~ T_rad) the two forms agree (b->1,
             * Milne->B) BUT the Milne shape inherits the valley's corrupted
             * n_+ (recombination edges overweighted, epay4: n_e +0.3 dex) —
             * so keep the legacy chi*B shape there. */
            static double epay_hotf = 1.5;
            { static int hf_once = 0;
              if (!hf_once) { hf_once = 1;
                  const char *hf = getenv("LUMINA_CMF_EPAY_HOTF");
                  if (hf) epay_hotf = atof(hf); } }
            int hot_regime = 0; /* scalar-temperature EPAY branch retired */
            double dr_s = geo->r_outer[s] - geo->r_inner[s];
            unsigned char r1ev=CMF_R1_EV_REACHED|CMF_R1_EV_EPAY_ELIGIBLE|
                               CMF_R1_EV_BRANCH;
            if (epay >= 2) r1ev|=CMF_R1_EV_EPAY_GE2;
            if (acc_w > 0.0) r1ev|=CMF_R1_EV_ACC_W_POS;
            if (hot_regime) r1ev|=CMF_R1_EV_HOT;
            if (epay >= 2 && acc_w > 0.0 && hot_regime) {
                /* kpkt mirror proper: paid power E_pay = absorbed + deposition,
                 * spectrum = normalized rate-side emissivity w(nu). Thick bins
                 * (Kirchhoff) keep the pass-1 legacy chi*B source untouched. */
                double E_pay = acc_abs + acc_dep;
                double wn = E_pay / acc_w;
                for (int b = 0; b < NB; ++b) {
                    size_t idx = (size_t)s * NB + b;
                    double chi_t = cs->chi_tot[idx];
                    if ((cs->chi_abs[idx] + cs->chi_line_th[idx]) * dr_s
                        > epay_taubin) {
                        if (cs->stage32_epay_disposition) {
                            cs->stage32_epay_disposition[idx]=1;
                            cs->stage32_epay_evidence[idx]=
                                (unsigned char)(r1ev|CMF_R1_EV_BIN_THICK);
                        }
                        if (kappa_dep > 0.0 && chi_t > 0.0)
                            cs->S_fixed[idx] += kappa_dep *
                                cm_planck(cs->nu[b], Te) / chi_t;
                        continue;   /* legacy Kirchhoff source */
                    }
                    if (cs->stage32_epay_disposition) {
                        cs->stage32_epay_disposition[idx]=2;
                        cs->stage32_epay_evidence[idx]=r1ev;
                    }
                    double w = (bf ? bf_get_eta(bf, s, cs->nu[b]) : 0.0)
                             + cs->chi_line_th[idx] * cm_planck(cs->nu[b], Te);
                    cs->S_fixed[idx] = (chi_t > 0.0) ? wn * w / chi_t : 0.0;
                }
                scale = wn;   /* debug: report the shape norm instead */
            } else {
            for (int b = 0; b < NB; ++b) {
                size_t idx = (size_t)s * NB + b;
                double chi_t = cs->chi_tot[idx];
                if ((cs->chi_abs[idx] + cs->chi_line_th[idx]) * dr_s
                    > epay_taubin) {
                    if (cs->stage32_epay_disposition) {
                        cs->stage32_epay_disposition[idx]=1;
                        cs->stage32_epay_evidence[idx]=
                            (unsigned char)(r1ev|CMF_R1_EV_BIN_THICK);
                    }
                } else {
                    if (cs->stage32_epay_disposition) {
                        cs->stage32_epay_disposition[idx]=3;
                        cs->stage32_epay_evidence[idx]=r1ev;
                    }
                    cs->S_fixed[idx] *= scale;
                }
                if (kappa_dep > 0.0 && chi_t > 0.0)
                    cs->S_fixed[idx] += kappa_dep * cm_planck(cs->nu[b], Te) / chi_t;
            }
            }
            if (s == 0) epay_scale_dbg[0] = scale;
            if (s == NS/2) epay_scale_dbg[1] = scale;
            if (s == 38 && s < NS) epay_scale_dbg[2] = scale;
            if (s == NS-1) epay_scale_dbg[3] = scale;
        } else if (cs->stage32_epay_disposition) {
            for (int b=0;b<NB;b++) {
                size_t idx=(size_t)s*NB+b;
                cs->stage32_epay_disposition[idx]=0;
                cs->stage32_epay_evidence[idx]=
                    CMF_R1_EV_REACHED|CMF_R1_EV_BRANCH;
            }
        }
    }
    if (epay)
        printf("[CMF-EPAY] scale s0=%.3e s%d=%.3e s38=%.3e s%d=%.3e\n",
               epay_scale_dbg[0], NS/2, epay_scale_dbg[1],
               epay_scale_dbg[2], NS-1, epay_scale_dbg[3]);
    free(epay_tau_arr);
    if ((line_eps > 0.0 || eps_phys || eps_uv > 0.0) && cs->diag) {
        static int eps_diag_once = 0;
        if (!eps_diag_once && n_split > 0) {   /* first assemble with lines */
            eps_diag_once = 1;
            double thsum = 0.0, lnsum = 0.0;
            for (size_t i = 0; i < (size_t)NS * NB; i++) {
                thsum += cs->chi_line_th[i]; lnsum += cs->chi_line[i];
            }
            printf("[CMFGEN-LINEEPS] %s eps=%.3f gate=%.2f split bins %ld/%ld "
                   "global chi_th/chi_line=%.4f\n",
                   eps_phys ? "PHYS" : (eps_uv > 0.0 ? "UV-TRANSFER" : "CONST"),
                   eps_uv > 0.0 ? eps_uv : line_eps, line_gate,
                   n_split, (long)NS * NB, lnsum > 0.0 ? thsum / lnsum : 1.0);
        }
    }
}

void cmfgen_assemble(CMFGENState *cs, const Geometry *geo,
                     const OpacityState *opac, BFOpacity *bf,
                     const PlasmaState *plasma) {
    cmfgen_assemble_impl(cs,geo,opac,bf,plasma,NULL);
}

int cmfgen_assemble_aulnu(CMFGENState *cs, const Geometry *geo,
                          const OpacityState *opac, BFOpacity *bf,
                          const PlasmaState *plasma, const NLTEConfig *nlte,
                          const AtomicData *atom, int seed_line,
                          int seed_shell, double seed_factor,
                          int retain_undefined_a,
                          CMFGENEmissABStats *stats) {
    double *a_total_cell=NULL, *a_covered_cell=NULL, *a_undefined_cell=NULL;
    if (!cs || !geo || !opac || !plasma || !nlte || !atom || !stats ||
        opac->n_lines < 0 || atom->n_lines != opac->n_lines ||
        geo->n_shells != cs->n_shells || opac->n_shells != cs->n_shells ||
        !atom->line_level_upper || !atom->level_num ||
        !nlte->nlte_to_global_level ||
        (retain_undefined_a != 0 && retain_undefined_a != 1) ||
        (seed_line >= 0 && (seed_line >= opac->n_lines || seed_shell < 0 ||
                            seed_shell >= cs->n_shells ||
                            !(seed_factor > 0.0) || !isfinite(seed_factor))))
        return -1;
    memset(stats,0,sizeof(*stats));
    stats->n_lines=opac->n_lines;
    stats->seed_line=seed_line;
    stats->seed_shell=seed_line >= 0 ? seed_shell : -1;
    stats->seed_factor=seed_line >= 0 ? seed_factor : 1.0;
    stats->retain_undefined_a=retain_undefined_a;
    stats->n_shells=cs->n_shells;
    stats->n_bins=cs->n_bins;
    size_t nl=(size_t)(opac->n_lines > 0 ? opac->n_lines : 1);
    stats->undefined_reason=(unsigned char *)calloc(nl,sizeof(unsigned char));
    stats->undefined_shell_count=(uint32_t *)calloc(nl,sizeof(uint32_t));
    stats->undefined_a_emissivity_by_band=(double *)calloc(
        (size_t)cs->n_bins,sizeof(double));
    stats->undefined_a_emissivity_by_shell=(double *)calloc(
        (size_t)cs->n_shells,sizeof(double));
    int *upper=(int *)malloc(nl*sizeof(int));
    unsigned char *active=(unsigned char *)calloc(nl,sizeof(unsigned char));
    if (!stats->undefined_reason || !stats->undefined_shell_count ||
        !stats->undefined_a_emissivity_by_band ||
        !stats->undefined_a_emissivity_by_shell ||
        !upper || !active) goto fail;
    size_t cells=(size_t)cs->n_shells*cs->n_bins;
    a_total_cell=(double *)calloc(cells,sizeof(double));
    a_covered_cell=(double *)calloc(cells,sizeof(double));
    a_undefined_cell=(double *)calloc(cells,sizeof(double));
    if (!a_total_cell || !a_covered_cell || !a_undefined_cell) goto fail;
    for (int l=0;l<opac->n_lines;l++) upper[l]=-1;

    int ni=nlte->n_nlte_ions;
    int **by_number=(int **)calloc((size_t)(ni > 0 ? ni : 1),sizeof(int *));
    int *max_number=(int *)malloc((size_t)(ni > 0 ? ni : 1)*sizeof(int));
    if (!by_number || !max_number) {
        free(by_number); free(max_number); goto fail;
    }
    for (int i=0;i<ni;i++) {
        max_number[i]=-1;
        for (int k=nlte->nlte_ion_level_offset[i];
             k<nlte->nlte_ion_level_offset[i+1];k++) {
            int gl=nlte->nlte_to_global_level[k];
            if (gl >= 0 && gl < atom->n_levels &&
                atom->level_num[gl] > max_number[i])
                max_number[i]=atom->level_num[gl];
        }
        if (max_number[i] >= 0) {
            by_number[i]=(int *)malloc((size_t)(max_number[i]+1)*sizeof(int));
            if (!by_number[i]) {
                for (int j=0;j<i;j++) free(by_number[j]);
                free(by_number); free(max_number); goto fail;
            }
            for (int k=0;k<=max_number[i];k++) by_number[i][k]=-1;
            for (int k=nlte->nlte_ion_level_offset[i];
                 k<nlte->nlte_ion_level_offset[i+1];k++) {
                int gl=nlte->nlte_to_global_level[k];
                if (gl >= 0 && gl < atom->n_levels) {
                    int number=atom->level_num[gl];
                    if (number >= 0 && number <= max_number[i])
                        by_number[i][number]=k;
                }
            }
        }
    }
    for (int l=0;l<opac->n_lines;l++) {
        if (!atom->line_A_ul || !(atom->line_A_ul[l] > 0.0) ||
            !isfinite(atom->line_A_ul[l]))
            stats->undefined_reason[l]|=CMF_EMISS_UNDEF_A_UL;
        int ion=nlte->nlte_line_map ? nlte->nlte_line_map[l] : -1;
        if (ion < 0 || ion >= ni) {
            stats->undefined_reason[l]|=CMF_EMISS_UNDEF_NO_NLTE_LINE;
            continue;
        }
        int number=atom->line_level_upper[l];
        if (number < 0 || number > max_number[ion] || !by_number[ion] ||
            by_number[ion][number] < 0) {
            stats->undefined_reason[l]|=CMF_EMISS_UNDEF_UPPER_LEVEL;
            continue;
        }
        upper[l]=by_number[ion][number];
    }
    for (int i=0;i<ni;i++) free(by_number[i]);
    free(by_number); free(max_number);

    {
        CMFEmissBContext ctx={nlte,atom,stats,upper,active,
            retain_undefined_a,a_total_cell,a_covered_cell,a_undefined_cell};
        cmfgen_assemble_impl(cs,geo,opac,bf,plasma,&ctx);
    }
    free(a_total_cell); free(a_covered_cell); free(a_undefined_cell);
    for (int l=0;l<opac->n_lines;l++) {
        if (!active[l]) {
            stats->undefined_reason[l]=0;
            stats->undefined_shell_count[l]=0;
            continue;
        }
        stats->active_transition_count++;
        if (stats->undefined_reason[l]) stats->undefined_transition_count++;
        else stats->defined_transition_count++;
    }
    stats->a_reference_contribution_fraction =
        stats->a_reference_line_power > 0.0
        ? stats->a_reference_covered_line_power /
          stats->a_reference_line_power : 0.0;
    stats->a_reference_undefined_contribution_fraction =
        stats->a_reference_line_power > 0.0
        ? stats->a_reference_undefined_line_power /
          stats->a_reference_line_power : 0.0;
    if (retain_undefined_a) {
        stats->retained_transition_count=stats->undefined_transition_count;
        stats->retained_line_shell_count=stats->undefined_line_shell_count;
        stats->a_reference_retained_line_power=
            stats->a_reference_undefined_line_power;
        stats->a_reference_retained_contribution_fraction=
            stats->a_reference_undefined_contribution_fraction;
    }
    free(upper); free(active);
    if (seed_line >= 0 && stats->seed_hits != 1) {
        fprintf(stderr,"[EMISS-AB][FAIL] seeded transition line=%d shell=%d "
                       "was hit %llu times (expected 1)\n",seed_line,seed_shell,
                       (unsigned long long)stats->seed_hits);
        return -1;
    }
    return 0;

fail:
    free(a_total_cell); free(a_covered_cell); free(a_undefined_cell);
    free(upper); free(active);
    cmfgen_emiss_ab_stats_free(stats);
    return -1;
}

/* ------------------------------------------------------------ */
/* One short-characteristics formal solution along all tangent rays for a
 * single frequency bin b, given the current total source S[shell]. Accumulates
 * J[shell] (angle-averaged) and the diagonal lambda_star[shell]. */
static void formal_solve_bin(CMFGENState *cs, const Geometry *geo,
                             int b, const double *S, double Bnu_inner,
                             double *Jb, double *Lstar,
                             double *Tlo, double *Tup)
{
    /* Tlo/Tup (optional, NULL to skip): tridiagonal Lambda off-diagonals —
     * Tlo[s] = nearest-INNER-neighbour coefficient L[s,s-1], Tup[s] = L[s,s+1].
     * Accumulated like Lacc (one-segment-attenuated upstream psi), normalized
     * by wacc. Used as the A4 ALI preconditioner (LUMINA_CMFGEN_LAMBDA_TRI). */
    int NS = cs->n_shells, NB = cs->n_bins;
    double *Jacc = calloc(NS, sizeof(double));
    double *wacc = calloc(NS, sizeof(double));
    double *Lacc = calloc(NS, sizeof(double));
    double *TloA = Tlo ? calloc(NS, sizeof(double)) : NULL;
    double *TupA = Tup ? calloc(NS, sizeof(double)) : NULL;
    if (!Jacc || !wacc || !Lacc || (Tlo && !TloA) || (Tup && !TupA)) {
        free(Jacc); free(wacc); free(Lacc); free(TloA); free(TupA); return;
    }

    /* shell-midpoint radii for the source grid */
    /* For each ray of impact parameter p, find shells with r_outer > p and
     * integrate inward (mu<0) then reflect/emit and integrate outward (mu>0). */
    for (int ray = 0; ray < cs->n_rays; ++ray) {
        double p = cs->p_ray[ray];
        /* collect intersected shells (outer->in), store z and shell idx */
        int    *sh = malloc(sizeof(int) * (NS + 1));
        double *z  = malloc(sizeof(double) * (NS + 1));
        int n = 0;
        for (int s = NS - 1; s >= 0; --s) {
            double ro = geo->r_outer[s];
            if (ro <= p) break;          /* inner shells don't reach this p */
            double rmid = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
            if (rmid <= p) rmid = p * 1.0000001;
            sh[n] = s;
            z[n]  = sqrt(rmid * rmid - p * p);
            ++n;
        }
        if (n == 0) { free(sh); free(z); continue; }
        int core = (p < geo->r_inner[0]) ? 1 : 0;
        /* core rays terminate at the core SURFACE z_core=sqrt(r_in0^2-p^2),
         * not at z=0: the innermost segment is z[n-1]-z_core, else dtau is
         * overestimated ~r_in/dr (100x) and the core B(T_inner) never leaks
         * into shell 0 (artificial seed-pin of inner T_e). */
        double z_core = 0.0;
        if (core) {
            double ri0 = geo->r_inner[0];
            z_core = sqrt(ri0 * ri0 - p * p);
            if (z_core > z[n - 1]) z_core = z[n - 1];
        }

        /* angular weight: this ray represents mu-interval around mu=z/r at the
         * outermost shell; use simple dp annulus weight 2 p dp / r^2 ~ dmu.
         * For an even-handed first solver we use uniform ray weight then
         * renormalise J by total weight per shell (wacc). */
        double ray_w = p;   /* proportional to annulus area element p dp */

        /* ----- inbound leg (mu<0): from outer boundary inward ----- */
        double I = 0.0;                       /* outer BC: no incoming */
        double psi_prev = 0.0;                /* upstream segment's psi */
        for (int i = 0; i < n; ++i) {
            int s = sh[i];
            double S_s = S[s];
            double ds = (i + 1 < n) ? fabs(z[i] - z[i + 1])
                                    : (z[i] - z_core);  /* to core surface/origin */
            double dtau = cs->chi_tot[(size_t)s * NB + b] * ds;
            if (dtau < 0.0) dtau = 0.0;
            double ex = exp(-dtau);
            double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5*dtau*dtau);
            I = I * ex + S_s * psi;
            /* inbound (mu<0) hemisphere sample; the matching outbound leg
             * supplies mu>0, so J = mean over both legs = mean intensity.
             * No extra 0.5: the leg average already gives (I_+ + I_-)/2. */
            Jacc[s] += ray_w * I;
            wacc[s] += ray_w;
            Lacc[s] += ray_w * psi;           /* diagonal local response */
            /* inbound upstream neighbour is the next-OUTER shell sh[i-1]=s+1 */
            if (TupA && i > 0) TupA[s] += ray_w * ex * psi_prev;
            psi_prev = psi;
        }
        double psi_turn = core ? 0.0 : psi_prev;  /* tangent carry (non-core) */
        /* ----- inner boundary ----- */
        if (core) I = Bnu_inner;              /* diffusive core emits B */
        /* (non-core grazing ray: I continues with whatever it accumulated) */
        /* ----- outbound leg (mu>0): from inner shell back out ----- */
        psi_prev = 0.0;
        for (int i = n - 1; i >= 0; --i) {
            int s = sh[i];
            double S_s = S[s];
            double ds = (i + 1 < n) ? fabs(z[i] - z[i + 1]) : (z[i] - z_core);
            double dtau = cs->chi_tot[(size_t)s * NB + b] * ds;
            if (dtau < 0.0) dtau = 0.0;
            double ex = exp(-dtau);
            double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5*dtau*dtau);
            I = I * ex + S_s * psi;            /* outbound (mu>0) hemisphere */
            Jacc[s] += ray_w * I;
            wacc[s] += ray_w;
            Lacc[s] += ray_w * psi;
            if (i == n - 1) {
                /* first outbound visit: upstream is the SAME shell's inbound
                 * segment (tangent turn) — a tridiag-LOCAL (diagonal) term;
                 * core rays carry B(T_inner) instead (inhomogeneous, in J_fs).
                 * Only counted in tridiag mode (keeps the legacy diagonal-ALI
                 * Lambda* byte-identical when the preconditioner is off). */
                if (TloA) Lacc[s] += ray_w * ex * psi_turn;
            } else if (TloA) {
                /* outbound upstream neighbour is the next-INNER shell s-1 */
                TloA[s] += ray_w * ex * psi_prev;
            }
            psi_prev = psi;
        }
        free(sh); free(z);
    }

    for (int s = 0; s < NS; ++s) {
        double iw = (wacc[s] > 0.0) ? 1.0 / wacc[s] : 0.0;
        Jb[s]    = Jacc[s] * iw;
        Lstar[s] = Lacc[s] * iw;
        if (Tlo) Tlo[s] = TloA ? TloA[s] * iw : 0.0;
        if (Tup) Tup[s] = TupA ? TupA[s] * iw : 0.0;
    }
    free(Jacc); free(wacc); free(Lacc); free(TloA); free(TupA);
}

/* ------------------------------------------------------------ */
void cmfgen_solve_J(CMFGENState *cs, const Geometry *geo, double T_inner,
                    int n_ali_iter)
{
    int NS = cs->n_shells, NB = cs->n_bins;
    double *S    = malloc(sizeof(double) * NS);
    double *Jb   = malloc(sizeof(double) * NS);
    double *Lst  = malloc(sizeof(double) * NS);
    if (!S || !Jb || !Lst) { free(S); free(Jb); free(Lst); return; }

    /* A4 tridiagonal-Lambda ALI preconditioner (LUMINA_CMFGEN_LAMBDA_TRI=1):
     * solve (I - Lambda_tri*R) J_new = J_fs - Lambda_tri*R*J_old per ALI pass
     * (Thomas, R=diag(r_s)). Needed once the line forest carries a near-unity
     * scattering albedo (physical eps ~1e-3): the pure diagonal closure
     * re-floors (the thin-UV pathology). Early-stop at max|dJ|/J < ALI_TOL. */
    int use_tri = 0;
    { const char *tr = getenv("LUMINA_CMFGEN_LAMBDA_TRI"); if (tr) use_tri = atoi(tr); }
    double ali_tol = 1e-3;
    { const char *tl = getenv("LUMINA_CMFGEN_ALI_TOL"); if (tl) ali_tol = atof(tl); }
    /* near-unity albedo (line-eps runs): per-pass change underestimates the
     * true error (spectral radius -> 1), so enforce a minimum pass count
     * before the early-stop is trusted (A4 Stage-1 sets 16+). */
    int ali_minit = 1;
    { const char *mi = getenv("LUMINA_CMFGEN_ALI_MINIT"); if (mi) ali_minit = atoi(mi); }
    double *Tlo = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *Tup = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *rA  = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *aA  = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *dA  = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *cA  = use_tri ? malloc(sizeof(double) * NS) : NULL;
    double *rhs = use_tri ? malloc(sizeof(double) * NS) : NULL;
    if (use_tri && (!Tlo || !Tup || !rA || !aA || !dA || !cA || !rhs)) {
        free(Tlo); free(Tup); free(rA); free(aA); free(dA); free(cA); free(rhs);
        Tlo = Tup = rA = aA = dA = cA = rhs = NULL;
        use_tri = 0;
    }

    /* optional single-cell ALI trace: LUMINA_CMFGEN_CELLDIAG="s,b" */
    int cd_s = -1, cd_b = -1;
    const char *cd = getenv("LUMINA_CMFGEN_CELLDIAG");
    if (cd && sscanf(cd, "%d,%d", &cd_s, &cd_b) != 2) { cd_s = cd_b = -1; }

    /* Inner-BB energy-balance scale (LUMINA_INNER_BB_SCALE, default 1.0): dilute
     * the inner blackbody source by this factor. When radioactive deposition is
     * ON, the inner-BB (central luminosity) + distributed deposition double-count
     * the energy and over-heat the plasma (drives T_e into the partial-ionization
     * max-line-opacity regime → MC packet trapping). Reducing the inner-BB lets
     * deposition supply the shell energy without the double-count (ARTIS uses
     * distributed deposition with NO inner blackbody). Keeps T_inner's COLOR;
     * only scales the amplitude (a dilution W_inner). */
    double inner_bb_scale = cmf_inner_bb_scale();
    for (int b = 0; b < NB; ++b) {
        double Bin = inner_bb_scale * cm_planck(cs->nu[b], T_inner);
        /* (tridiagonal-)ALI Lambda iteration for the scattering channel */
        for (int it = 0; it < n_ali_iter; ++it) {
            for (int s = 0; s < NS; ++s) {
                size_t idx = (size_t)s * NB + b;
                double r = (cs->chi_tot[idx] > 0.0)
                         ? cs->chi_es[idx] / cs->chi_tot[idx] : 0.0;
                if (rA) rA[s] = r;
                S[s] = cs->S_fixed[idx] + r * cs->J[idx];
            }
            formal_solve_bin(cs, geo, b, S, Bin, Jb, Lst, Tlo, Tup);
            double maxrel = 0.0;
            if (use_tri) {
                for (int s = 0; s < NS; ++s) {
                    size_t idx = (size_t)s * NB + b;
                    double Jo  = cs->J[idx];
                    double Jom = (s > 0)      ? cs->J[idx - NB] : 0.0;
                    double Jop = (s < NS - 1) ? cs->J[idx + NB] : 0.0;
                    aA[s] = (s > 0)      ? -Tlo[s] * rA[s - 1] : 0.0;
                    cA[s] = (s < NS - 1) ? -Tup[s] * rA[s + 1] : 0.0;
                    dA[s] = 1.0 - Lst[s] * rA[s];
                    if (dA[s] < 1e-10) dA[s] = 1e-10;
                    rhs[s] = Jb[s] - Lst[s] * rA[s] * Jo
                           - ((s > 0)      ? Tlo[s] * rA[s - 1] * Jom : 0.0)
                           - ((s < NS - 1) ? Tup[s] * rA[s + 1] * Jop : 0.0);
                }
                /* Thomas forward sweep */
                for (int s = 1; s < NS; ++s) {
                    double m = aA[s] / dA[s - 1];
                    dA[s]  -= m * cA[s - 1];
                    if (dA[s] < 1e-10) dA[s] = 1e-10;
                    rhs[s] -= m * rhs[s - 1];
                }
                double Jn = rhs[NS - 1] / dA[NS - 1];
                for (int s = NS - 1; s >= 0; --s) {
                    if (s < NS - 1) Jn = (rhs[s] - cA[s] * Jb[s + 1]) / dA[s];
                    /* NOTE: back-substitution must use the NEW J of s+1; reuse
                     * Jb[] as the solved-J scratch to avoid another array. */
                    size_t idx = (size_t)s * NB + b;
                    double Jold = cs->J[idx];
                    if (!isfinite(Jn) || Jn < 0.0) Jn = 0.0;  /* finite guard:
                        warm-started J makes any one-iter accident permanent */
                    double rel = (Jn > 1e-300) ? fabs(Jn - Jold) / Jn : 0.0;
                    if (rel > maxrel) maxrel = rel;
                    Jb[s] = Jn;                 /* solved J for s (scratch) */
                }
                for (int s = 0; s < NS; ++s) {
                    size_t idx = (size_t)s * NB + b;
                    cs->J[idx] = Jb[s];
                    /* Stage-4 deferred: keep the FORMAL diagonal for the
                     * Newton response (semantics unchanged). */
                    cs->lambda_star[idx] = Lst[s];
                    /* A4 Stage-2.5: persist the tridiagonal response
                     * coefficients (last ALI pass wins) so the global Newton
                     * can apply delta-J = (I - Lambda_tri R)^-1 Lambda_tri
                     * delta-S without re-running the formal solve. */
                    cs->tri_lo[idx] = Tlo[s];
                    cs->tri_up[idx] = Tup[s];
                    cs->tri_r[idx]  = rA[s];
                }
            } else {
                for (int s = 0; s < NS; ++s) {
                    size_t idx = (size_t)s * NB + b;
                    double r = (cs->chi_tot[idx] > 0.0)
                             ? cs->chi_es[idx] / cs->chi_tot[idx] : 0.0;
                    /* local ALI accel: J = (J_fs - L* r J_old)/(1 - L* r) */
                    double Ldiag = Lst[s];
                    double denom = 1.0 - Ldiag * r;
                    double Jnew = (denom > 1e-10)
                                ? (Jb[s] - Ldiag * r * cs->J[idx]) / denom
                                : Jb[s];
                    if (b == cd_b && s == cd_s) {
                        size_t i2 = (size_t)cd_s * NB + cd_b;
                        printf("[CMFGEN-CELL] s=%d b=%d ali=%d chi_es=%.3e chi_abs=%.3e "
                               "chi_line=%.3e r=%.4f Sfix=%.3e S=%.3e Lst=%.4e Jfs=%.3e "
                               "Jold=%.3e Jnew=%.3e\n", cd_s, cd_b, it, cs->chi_es[i2],
                               cs->chi_abs[i2], cs->chi_line[i2], r, cs->S_fixed[i2],
                               S[s], Ldiag, Jb[s], cs->J[idx], Jnew < 0 ? 0 : Jnew);
                    }
                    if (!isfinite(Jnew) || Jnew < 0.0) Jnew = 0.0;
                    cs->J[idx] = Jnew;
                    /* Persist the diagonal ∂J/∂S operator for the RADEQ/Newton T_e
                     * solve (Phase-1 faithful radiation response). Last ALI iter wins. */
                    cs->lambda_star[idx] = Lst[s];
                }
            }
            if (use_tri && maxrel < ali_tol && it >= ali_minit) break;
        }
    }
    free(S); free(Jb); free(Lst);
    free(Tlo); free(Tup); free(rA); free(aA); free(dA); free(cA); free(rhs);
    /* This is the only point that certifies lambda_star as the diagonal paired
     * with the current assembled chi_es/chi_tot generation.  Early allocation
     * failure and alternate solvers leave the lineage invalid and dumping fails. */
    if (cs->stage32_eta_pre_epay && cs->stage32_field_generation > 0)
        cs->stage32_lambda_generation=cs->stage32_field_generation;
    if (cd_s >= 0) fflush(stdout);
}

/* ------------------------------------------------------------ */
void cmfgen_window_color(CMFGENState *cs)
{
    int NS = cs->n_shells, NB = cs->n_bins;
    /* Two CONTINUUM windows (path-4 redo of the sealed 165485 anchor). The
     * old 4150-4300A blue band was line-REPROCESSED in the outer shells
     * (chi_line/chi_cont up to 8.6e6): anchoring T_e brightened the locally
     * re-emitted trough J -> hotter color -> runaway to 24kK. These bands are
     * chosen from the 165584 JDUMP chi map: worst chi_line/(chi_es+chi_abs)
     * over sh40-48 is <0.10 in BOTH, so the window J is TRANSPORTED photo-
     * spheric light, not local re-emission — the feedback loop gain is dead.
     * Measured on the converged champion field: color = 2471.6K, FLAT across
     * sh38-48 (gold tail T_e 2505-2560K; treadmill extracts 2254K = -10%).
     * Override via LUMINA_CMFGEN_COLOR_BANDS="l1,l2,l3,l4" (Angstrom). */
    double LB1 = 5619e-8, LB2 = 6083e-8, LR1 = 9000e-8, LR2 = 11000e-8;
    { const char *cb = getenv("LUMINA_CMFGEN_COLOR_BANDS");
      if (cb) { double a, b2, c2, e;
                if (sscanf(cb, "%lf,%lf,%lf,%lf", &a, &b2, &c2, &e) == 4) {
                    LB1 = a * 1e-8; LB2 = b2 * 1e-8;
                    LR1 = c2 * 1e-8; LR2 = e * 1e-8; } } }
    for (int s = 0; s < NS; ++s) {
        double Jb = 0.0, wb = 0.0, Jr = 0.0, wr = 0.0;
        int nb_ = 0, nr_ = 0;
        for (int b = 0; b < NB; ++b) {
            double lam = CM_C / cs->nu[b];
            int blue = (lam >= LB1 && lam <= LB2);
            int red  = (lam >= LR1 && lam <= LR2);
            if (!blue && !red) continue;
            size_t idx = (size_t)s * NB + b;
            double J = cs->J[idx];
            if (J <= 0.0) continue;
            if (blue) { Jb += J * cs->dnu[b]; wb += cs->dnu[b]; nb_++; }
            else      { Jr += J * cs->dnu[b]; wr += cs->dnu[b]; nr_++; }
        }
        cs->t_color[s] = -1.0;
        if (nb_ < 2 || nr_ < 2 || Jr <= 0.0 || wb <= 0.0 || wr <= 0.0) continue;
        double target = (Jb / wb) / (Jr / wr);
        if (!(target > 0.0)) continue;
        /* Planck band-ratio is monotonically increasing in T: bisect. */
        double nub = CM_C / (0.5 * (LB1 + LB2)), nur = CM_C / (0.5 * (LR1 + LR2));
        double Tlo = 800.0, Thi = 30000.0;
        double rlo = cm_planck(nub, Tlo) / cm_planck(nur, Tlo);
        double rhi = cm_planck(nub, Thi) / cm_planck(nur, Thi);
        if (target <= rlo || target >= rhi) continue;
        for (int it = 0; it < 60; ++it) {
            double Tm = 0.5 * (Tlo + Thi);
            double rm = cm_planck(nub, Tm) / cm_planck(nur, Tm);
            if (rm < target) Tlo = Tm; else Thi = Tm;
        }
        cs->t_color[s] = 0.5 * (Tlo + Thi);
    }
}

/* ------------------------------------------------------------ */
/* Thick/thin limit validation. For chosen shells prints, at 3 bins, the
 * inward radial optical depth tau_r, B(T_e), the (converged) total source S,
 * the solved J, and the ratios J/B (->1 thick, ->W thin) and J/S (->1 thick).
 * A correct solver must show J/B->1 and J/S->1 in the most opaque cell, and
 * J/B ~ W (geometric dilution) in the optically-thin outer. */
void cmfgen_validate(const CMFGENState *cs, const Geometry *geo,
                     const PlasmaState *plasma)
{
    int NS = cs->n_shells, NB = cs->n_bins;
    int bins[3] = { NB / 5, NB / 2, (4 * NB) / 5 };   /* blue, mid, red */
    int shells[4] = { 0, NS / 4, NS / 2, NS - 1 };

    printf("[CMFGEN-VALID] thick(J/B->1,J/S->1) / thin(J/B->W) limit check\n");
    for (int si = 0; si < 4; ++si) {
        int s = shells[si];
        double Te = plasma->T_e[s];
        double rmid = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
        /* geometric dilution W = 0.5(1 - sqrt(1 - (r_in0/r)^2)) */
        double x = geo->r_inner[0] / rmid;
        double W = (x < 1.0) ? 0.5 * (1.0 - sqrt(1.0 - x * x)) : 0.5;
        for (int bi = 0; bi < 3; ++bi) {
            int b = bins[bi];
            size_t idx = (size_t)s * NB + b;
            /* inward radial optical depth from outer boundary to shell s */
            double tau_r = 0.0;
            for (int q = NS - 1; q >= s; --q) {
                double dr = geo->r_outer[q] - geo->r_inner[q];
                tau_r += cs->chi_tot[(size_t)q * NB + b] * dr;
            }
            double B = cm_planck(cs->nu[b], Te);
            double Sloc = cs->S_fixed[idx]
                        + ((cs->chi_tot[idx] > 0.0)
                           ? cs->chi_es[idx] / cs->chi_tot[idx] : 0.0) * cs->J[idx];
            double J = cs->J[idx];
            printf("[CMFGEN-VALID] s=%2d b=%4d nu=%.3e Te=%.0f tau_r=%.3e "
                   "W=%.4f B=%.3e S=%.3e J=%.3e J/B=%.3f J/S=%.3f\n",
                   s, b, cs->nu[b], Te, tau_r, W, B, Sloc, J,
                   B > 0 ? J / B : 0.0, Sloc > 0 ? J / Sloc : 0.0);
        }
    }

    /* MISSING-TERM PROBE: volumetric line RADIATIVE heating 4π∫(χ_line·J − η_line)dν
     * [erg/s/cm^3], the term absent from radeq_net (which carries lines only as the
     * collisional-net cooling). η_line is reconstructed from the stored source:
     *   S_fixed = (χ_abs·B + η_line)/χ_tot  ⇒  η_line = S_fixed·χ_tot − χ_abs·B.
     * Compare net against H_photo/C_bb to see if it can anchor outer T_e. */
    int dsh[5] = { 0, NS / 2, (3 * NS) / 4, (7 * NS) / 8, NS - 1 };
    printf("[CMFGEN-LINEHEAT] 4pi*Int(chi_line*J - eta_line)dnu  (>0 = net heating)\n");
    for (int di = 0; di < 5; ++di) {
        int s = dsh[di];
        double Te = plasma->T_e[s];
        double abs_r = 0.0, emi_r = 0.0;
        for (int b = 0; b < NB; ++b) {
            size_t idx = (size_t)s * NB + b;
            double B = cm_planck(cs->nu[b], Te);
            double eta_ln = cs->S_fixed[idx] * cs->chi_tot[idx]
                          - cs->chi_abs[idx] * B;
            if (eta_ln < 0.0) eta_ln = 0.0;
            abs_r += cs->chi_line[idx] * cs->J[idx] * cs->dnu[b];
            emi_r += eta_ln * cs->dnu[b];
        }
        abs_r *= 4.0 * M_PI_VAL;  emi_r *= 4.0 * M_PI_VAL;
        printf("[CMFGEN-LINEHEAT] s=%2d Te=%.0f  H_line_abs=%.3e  emis_line=%.3e  "
               "net=%.3e\n", s, Te, abs_r, emi_r, abs_r - emi_r);
    }

    /* Full deterministic-J dump (LUMINA_CMFGEN_JDUMP=1): per shell x bin, the
     * solved J plus the opacity split and ALI diagonal, so the thin-UV floor
     * (Defect A) can be located offline against the sigma_bf edges. */
    const char *jd = getenv("LUMINA_CMFGEN_JDUMP");
    if (jd && atoi(jd)) {
        FILE *jf = fopen("lumina_cmfgen_jnu.csv", "w");
        if (jf) {
            fprintf(jf, "shell,bin,nu,J,chi_es,chi_abs,chi_line,chi_tot,"
                        "S_fixed,lambda_star\n");
            for (int s = 0; s < NS; ++s)
                for (int b = 0; b < NB; ++b) {
                    size_t idx = (size_t)s * NB + b;
                    fprintf(jf, "%d,%d,%.6e,%.6e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e\n",
                            s, b, cs->nu[b], cs->J[idx], cs->chi_es[idx],
                            cs->chi_abs[idx], cs->chi_line[idx],
                            cs->chi_tot[idx], cs->S_fixed[idx],
                            cs->lambda_star[idx]);
                }
            fclose(jf);
            printf("[CMFGEN-JDUMP] wrote lumina_cmfgen_jnu.csv (%d shells x %d bins)\n",
                   NS, NB);
        }
    }
    fflush(stdout);
}

/* ------------------------------------------------------------ */
/* Emergent observer-frame spectrum from the converged deterministic field.
 *
 * For each tangent ray of impact parameter p we propagate the formal solution
 * (same inbound + inner-BC + outbound legs as formal_solve_bin) and keep only
 * the surface intensity I+(p) at the end of the outbound leg. The emergent
 * monochromatic luminosity is the p-z surface integral
 *      L_nu = 8 pi^2 Int_0^Rmax I+(p) p dp .
 * The expansion (Sobolev-binned) line opacity already subsumes the comoving
 * d/dnu, so each bin is an observer-frame-broadened quasi-static slab and the
 * binned L_nu is the line-blanketed emergent SED at bin resolution. */
int cmfgen_write_spectrum(const CMFGENState *cs, const Geometry *geo,
                          double T_inner, const char *path)
{
    int NS = cs->n_shells, NB = cs->n_bins, NR = cs->n_rays;

    /* ---- precompute per-ray intersected shells + segment lengths (b-indep) ---- */
    int    *ray_n    = malloc(sizeof(int) * NR);
    int    *ray_core = malloc(sizeof(int) * NR);
    int    *seg_sh   = malloc(sizeof(int) * (size_t)NR * NS);
    double *seg_ds   = malloc(sizeof(double) * (size_t)NR * NS);
    double *S        = malloc(sizeof(double) * NS);
    double *Lnu      = calloc(NB, sizeof(double));
    if (!ray_n || !ray_core || !seg_sh || !seg_ds || !S || !Lnu) {
        free(ray_n); free(ray_core); free(seg_sh); free(seg_ds);
        free(S); free(Lnu);
        fprintf(stderr, "[CMFGEN] spectrum alloc failed\n");
        return -1;
    }

    for (int ray = 0; ray < NR; ++ray) {
        double p = cs->p_ray[ray];
        int *sh = &seg_sh[(size_t)ray * NS];
        double *zz = malloc(sizeof(double) * (NS + 1));
        int nshell = 0;
        for (int s = NS - 1; s >= 0; --s) {
            double ro = geo->r_outer[s];
            if (ro <= p) break;
            double rmid = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
            if (rmid <= p) rmid = p * 1.0000001;
            sh[nshell] = s;
            zz[nshell] = sqrt(rmid * rmid - p * p);
            ++nshell;
        }
        ray_n[ray]    = nshell;
        ray_core[ray] = (p < geo->r_inner[0]) ? 1 : 0;
        double z_core = 0.0;
        if (ray_core[ray] && nshell > 0) {
            double ri0 = geo->r_inner[0];
            z_core = sqrt(ri0 * ri0 - p * p);
            if (z_core > zz[nshell - 1]) z_core = zz[nshell - 1];
        }
        double *ds = &seg_ds[(size_t)ray * NS];
        for (int i = 0; i < nshell; ++i)
            ds[i] = (i + 1 < nshell) ? fabs(zz[i] - zz[i + 1])
                                     : (ray_core[ray] ? zz[i] - z_core
                                                      : fabs(zz[i]));
        free(zz);
    }

    /* ---- per-bin emergent flux ---- */
    for (int b = 0; b < NB; ++b) {
        double Bin = cmf_inner_bb_scale() * cm_planck(cs->nu[b], T_inner);
        for (int s = 0; s < NS; ++s) {
            size_t idx = (size_t)s * NB + b;
            double r = (cs->chi_tot[idx] > 0.0)
                     ? cs->chi_es[idx] / cs->chi_tot[idx] : 0.0;
            S[s] = cs->S_fixed[idx] + r * cs->J[idx];   /* converged source */
        }
        /* integrate I+(p) p dp over the (ascending-p) ray grid, trapezoid,
         * with f(0)=0 at the origin. */
        double integ = 0.0, p_prev = 0.0, f_prev = 0.0;
        for (int ray = 0; ray < NR; ++ray) {
            int n = ray_n[ray];
            if (n == 0) continue;
            const int *sh = &seg_sh[(size_t)ray * NS];
            const double *ds = &seg_ds[(size_t)ray * NS];
            double I = 0.0;                       /* outer BC: no incoming */
            for (int i = 0; i < n; ++i) {         /* inbound (mu<0) */
                double dtau = cs->chi_tot[(size_t)sh[i] * NB + b] * ds[i];
                if (dtau < 0.0) dtau = 0.0;
                double ex = exp(-dtau);
                double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5*dtau*dtau);
                I = I * ex + S[sh[i]] * psi;
            }
            if (ray_core[ray]) I = Bin;           /* diffusive core emits B */
            for (int i = n - 1; i >= 0; --i) {    /* outbound (mu>0) */
                double dtau = cs->chi_tot[(size_t)sh[i] * NB + b] * ds[i];
                if (dtau < 0.0) dtau = 0.0;
                double ex = exp(-dtau);
                double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5*dtau*dtau);
                I = I * ex + S[sh[i]] * psi;
            }
            double p = cs->p_ray[ray];
            double f = I * p;                      /* integrand I+(p) p */
            integ += 0.5 * (f_prev + f) * (p - p_prev);
            p_prev = p; f_prev = f;
        }
        Lnu[b] = 8.0 * M_PI_VAL * M_PI_VAL * integ;        /* erg/s/Hz */
    }

    /* ---- write ascending-wavelength CSV: L_lambda = L_nu * c/lambda^2 ---- */
    FILE *fp = fopen(path, "w");
    if (!fp) {
        free(ray_n); free(ray_core); free(seg_sh); free(seg_ds);
        free(S); free(Lnu);
        fprintf(stderr, "[CMFGEN] cannot open %s\n", path);
        return -1;
    }
    fprintf(fp, "wavelength_angstrom,flux\n");
    for (int b = NB - 1; b >= 0; --b) {            /* nu desc -> lambda asc */
        double lam_cm = CM_C / cs->nu[b];
        double lam_A  = lam_cm * 1.0e8;
        double L_lam  = Lnu[b] * CM_C / (lam_cm * lam_cm) * 1.0e-8; /* erg/s/A */
        fprintf(fp, "%.6f,%.6e\n", lam_A, L_lam);
    }
    fclose(fp);
    printf("Pure-CMFGEN emergent spectrum written to %s (%d bins)\n", path, NB);

    free(ray_n); free(ray_core); free(seg_sh); free(seg_ds);
    free(S); free(Lnu);
    return 0;
}

/* ------------------------------------------------------------ */
/* Linear interpolation of y over the ASCENDING nu grid (cs->nu[0]=nu_min ..
 * nu[NB-1]=nu_max, see cmfgen_create nu[b]=nu_min*exp((b+0.5)*d_log_nu)).
 * Returns 0 (vacuum) for nu_q outside (nu[0], nu[NB-1]). */
static double cmf_interp_nu_asc(const double *nu, const double *y, int NB,
                                double nu_q)
{
    if (nu_q <= nu[0] || nu_q >= nu[NB - 1]) return 0.0;
    int lo = 0, hi = NB - 1;                     /* nu[lo] <= nu_q < nu[hi] */
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (nu[mid] <= nu_q) lo = mid; else hi = mid;
    }
    double t = (nu_q - nu[lo]) / (nu[lo + 1] - nu[lo]); /* nu[lo+1] > nu[lo] */
    return y[lo] + t * (y[lo + 1] - y[lo]);
}

/* Observer-frame Doppler q = gamma(1 - mu*beta) at signed path coordinate z
 * along a ray of impact parameter p (mu = z/r, beta = r/(c t_exp)). */
static double cmf_q_at_z(double p, double z, double inv_ct)
{
    double r = sqrt(p * p + z * z);
    double beta = r * inv_ct;
    double mu = (r > 0.0) ? z / r : 0.0;
    return (1.0 - mu * beta) / sqrt(1.0 - beta * beta);
}

/* March one ray segment [z0,z1] (signed z increasing toward observer) through
 * shell s, sub-splitting so the comoving frequency q*nu_obs sweeps <=0.5 bin
 * per sub-step (resolves line P-Cygni; continuum unaffected). Returns updated I. */
static double cmf_obs_march(double I, double p, double z0, double z1, int s,
                            const double *nu, const double *chi_tot,
                            const double *Sbin, int NB, double nu_obs,
                            double inv_ct, double d_log_nu)
{
    double q0 = cmf_q_at_z(p, z0, inv_ct);
    double q1 = cmf_q_at_z(p, z1, inv_ct);
    int nsub = (int)ceil(fabs(log(q1 / q0)) / (0.5 * d_log_nu));
    if (nsub < 1)  nsub = 1;
    if (nsub > 256) nsub = 256;
    for (int m = 0; m < nsub; ++m) {
        double za = z0 + (z1 - z0) * ((double)m / nsub);
        double zb = z0 + (z1 - z0) * ((double)(m + 1) / nsub);
        double zm = 0.5 * (za + zb);
        double ds = fabs(zb - za);
        double r  = sqrt(p * p + zm * zm);
        double mu = (r > 0.0) ? zm / r : 0.0;
        double beta = r * inv_ct;
        double q = (1.0 - mu * beta) / sqrt(1.0 - beta * beta);
        double D = 1.0 / q;
        double nucmf = q * nu_obs;
        double chi0 = cmf_interp_nu_asc(nu, &chi_tot[(size_t)s * NB], NB, nucmf);
        double S0   = cmf_interp_nu_asc(nu, &Sbin[(size_t)s * NB],    NB, nucmf);
        if (chi0 < 0.0) chi0 = 0.0;
        double alpha = q * chi0;
        double dtau  = alpha * ds; if (dtau < 0.0) dtau = 0.0;
        double ex  = (dtau > 700.0) ? 0.0 : exp(-dtau);
        double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5 * dtau * dtau);
        double Sobs = (alpha > 0.0) ? (D * D * D) * S0 : 0.0;
        I = I * ex + Sobs * psi;
    }
    return I;
}

/* S4 line thermalisation share (LUMINA_CMF_LINE_SOB_EPS, default 0 = pure
 * resonance scatter S_l=Jbar). Set once per spectrum in cmfgen_write_spectrum_obs. */
static double g_sob_eps = 0.0;
static int    g_sob_noemit = 0;
static int    g_sob_diag = 0;
static int    g_sob_contonly = 0;
static double g_sob_taumin = 1e-6;  /* min Sobolev tau for a line to fire (diag) */
static int    g_pray_bin = -1;
static double g_sob_rphot  = 0.0;   /* photosphere radius (r_inner[0]) for W(r) */
static double g_sob_Tinner = 0.0;   /* photosphere T for the diluted backlight */
static int    g_sob_srcj   = 0;     /* 1 = legacy cs->J source (contaminated) */
static int    g_sob_jbardet = 0;    /* 1 = ENERGY-CONSERVING scatter source S_l=Jbar_l
                                     * (producer jbar_line_det). W*B backlight (below
                                     * incident I) destroys ~47% of the flux. */
static int    g_sob_sei_cmv = 1;    /* 1 = SEI Pass-1 accumulates COMOVING I*q^3 (correct);
                                     * 0 = old observer-frame I (the +180% double-beam bug) */
static int    g_sob_sei = 0;        /* 1 = 2-pass SEI: lines jump at TRUE tau_S sourced
                                     * from the Pass-1 beamed continuum mean (jc[s]),
                                     * conserving AND sharp P-Cygni (2026-06-26 fix). */
static int    g_sob_faithful = 0;   /* 1 = transport static chi_tot+Sbin (conserving),
                                     * no Sobolev jumps — the obs energy fix (2026-06-26). */
/* Unified-emergent: drive the resonance line source from the NLTE-solved
 * line_source_S (fluorescence) instead of the W*B scattering backlight.
 * g_sob_sl_ptr points at opac->line_source_S[l*NS+s]; clamp caps S_l<=clamp*B. */
static const double *g_sob_sl_ptr = NULL;   /* NULL = scattering (default) */
static double        g_sob_sl_clamp = 0.0;  /* 0 = off */

/* S4 — LINE-RESOLVED Sobolev observer march (gate LUMINA_CMF_OBS_SOBOLEV).
 * The binned expansion opacity defangs tau_Sobolev (frac=1-e^{-tau}->1, /dnu_bin
 * => integrated ray tau ~ O(1), no P-Cygni; codex 019ef207 + agent ac7845ec).
 * Here the CONTINUUM is transported from chi_es+chi_abs (NOT the binned
 * chi_line), and each line whose rest nu_l lies in the sub-step's swept comoving
 * interval applies its FULL Sobolev jump  I = I*e^{-tau_S} + D^3*S_l*(1-e^{-tau_S})
 * at the resonance — exactly the directional resonance the MC does. Requires
 * LINE_EPS off (so chi_es is pure electron, no double-count). */
static double cmf_obs_march_sob_jc(const double *jc,
                                double I, double p, double z0, double z1, int s,
                                const CMFGENState *cs, const OpacityState *opac,
                                double Te_s, double nu_obs, double inv_ct)
{
    int NB = cs->n_bins, NS = cs->n_shells, NL = opac->n_lines;
    double q0 = cmf_q_at_z(p, z0, inv_ct);
    double q1 = cmf_q_at_z(p, z1, inv_ct);
    /* Resolve the Sobolev resonance to a FINE comoving-velocity step. The bin
     * width 0.5*d_log_nu (~800 km/s) smears each line jump over ~800 km/s of
     * observer wavelength -> too-shallow trough. Resolve to ~30 km/s (env
     * LUMINA_CMF_OBS_DVRES, km/s) so the jump lands at the sharp resonance. */
    double dlq = 30.0 / 2.99792458e5;
    { const char *e = getenv("LUMINA_CMF_OBS_DVRES"); if (e) dlq = atof(e) / 2.99792458e5; }
    int nsub = (int)ceil(fabs(log(q1 / q0)) / dlq);
    if (nsub < 1)    nsub = 1;
    if (nsub > 4096) nsub = 4096;
    for (int m = 0; m < nsub; ++m) {
        double za = z0 + (z1 - z0) * ((double)m / nsub);
        double zb = z0 + (z1 - z0) * ((double)(m + 1) / nsub);
        double zm = 0.5 * (za + zb);
        double ds = fabs(zb - za);
        double r  = sqrt(p * p + zm * zm);
        double mu = (r > 0.0) ? zm / r : 0.0;
        double beta = r * inv_ct;
        double q = (1.0 - mu * beta) / sqrt(1.0 - beta * beta);
        double D = 1.0 / q;
        double nucmf = q * nu_obs;
        if (g_sob_faithful) {
            /* FAITHFUL observer-frame transform of the CONSERVING static extractor
             * (physics review 2026-06-26): transport the SAME fine-grid chi_tot +
             * full source Sbin=S_fixed+r·J the static uses (lines RESOLVED on the
             * fine grid -> no Sobolev-jump hack). Conserves by construction; the
             * D⁴ beaming + P-Cygni asymmetry emerge from the line-of-sight geometry.
             * dtau=q·chi_tot·ds, source D³·Sbin. No separate line loop. */
            double chit = cmf_interp_nu_asc(cs->nu, &cs->chi_tot[(size_t)s*NB], NB, nucmf);
            double ces  = cmf_interp_nu_asc(cs->nu, &cs->chi_es[(size_t)s*NB], NB, nucmf);
            double Sfx  = cmf_interp_nu_asc(cs->nu, &cs->S_fixed[(size_t)s*NB], NB, nucmf);
            double Jvf  = cmf_interp_nu_asc(cs->nu, &cs->J[(size_t)s*NB], NB, nucmf);
            if (chit < 0.0) chit = 0.0; if (ces < 0.0) ces = 0.0;
            double rf   = (chit > 0.0) ? ces/chit : 0.0;
            double Sbin = Sfx + rf * Jvf;
            double dt   = q * chit * ds; if (dt < 0.0) dt = 0.0;
            double exf  = (dt > 700.0) ? 0.0 : exp(-dt);
            double psf  = (dt > 1e-4) ? (1.0 - exf) : (dt - 0.5*dt*dt);
            I = I*exf + (D*D*D) * Sbin * psf;
            continue;
        }
        /* continuum (electron scatter + thermal bf/ff), NO binned line */
        double chi_es = cmf_interp_nu_asc(cs->nu, &cs->chi_es[(size_t)s * NB], NB, nucmf);
        double chi_ab = cmf_interp_nu_asc(cs->nu, &cs->chi_abs[(size_t)s * NB], NB, nucmf);
        if (chi_es < 0.0) chi_es = 0.0;
        if (chi_ab < 0.0) chi_ab = 0.0;
        double chi_c = chi_es + chi_ab;
        double Jv = cmf_interp_nu_asc(cs->nu, &cs->J[(size_t)s * NB], NB, nucmf);
        double B  = cm_planck(nucmf, Te_s);
        double S_c = (chi_c > 0.0) ? (chi_ab * B + chi_es * Jv) / chi_c : 0.0;
        double alpha = q * chi_c;
        double dtau  = alpha * ds; if (dtau < 0.0) dtau = 0.0;
        double ex  = (dtau > 700.0) ? 0.0 : exp(-dtau);
        double psi = (dtau > 1e-4) ? (1.0 - ex) : (dtau - 0.5 * dtau * dtau);
        I = I * ex + (D * D * D) * S_c * psi;
        if (g_sob_contonly) continue;   /* continuum-only diagnostic (no line jumps) */
        /* line resonances crossed in [nlo,nhi] of this sub-step's comoving sweep */
        double nu_a = cmf_q_at_z(p, za, inv_ct) * nu_obs;
        double nu_b = cmf_q_at_z(p, zb, inv_ct) * nu_obs;
        double nlo = (nu_a < nu_b) ? nu_a : nu_b;
        double nhi = (nu_a < nu_b) ? nu_b : nu_a;
        /* line_list_nu DESCENDING: first index with nu_l <= nhi */
        int lo = 0, hi = NL;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (opac->line_list_nu[mid] > nhi) lo = mid + 1; else hi = mid;
        }
        /* half-open (nlo, nhi]: a line on a sub-step boundary fires once (codex #3) */
        for (int l = lo; l < NL && opac->line_list_nu[l] > nlo; ++l) {
            double tauS = opac->tau_sobolev[(size_t)l * NS + s];
            if (!(tauS > g_sob_taumin)) continue;    /* skips NaN too (codex #6) */
            /* Resonance-line source: scattering S_l=Jbar (CLEAN continuum mean
             * intensity = diluted backlight, NOT thermal B which refills the
             * line, NOR cs->J which is contaminated by the binned line — codex
             * #1). Prefer the cont_only field opac->jbar_line (LUMINA_CMF_JINC_
             * CONT); fall back to the local binned J. eps blends a thermal B
             * share for collision-thermalised lines (default 0 = pure scatter). */
            /* Resonance-scatter source = J_bar = the INCIDENT continuum field,
             * i.e. the geometrically-DILUTED photospheric backlight W(r)*B(T_phot)
             * — NOT the local cs->J, which is contaminated by the line's own
             * trapped radiation (binned line traps in its bin -> J/B~0.64 vs the
             * 0.5 clean continuum, self-refilling the trough). W(r) is the point-
             * photosphere dilution. (LUMINA_CMF_OBS_SRCJ=1 reverts to cs->J.) */
            double Sl;
            if (g_sob_sei && jc) {
                /* 2-pass SEI: scatter the Pass-1 BEAMED continuum mean intensity
                 * jc[s] (carries the D⁴ beaming that comoving J̄_l lacked). Then the
                 * true-tau_S black line conserves: blue trough (deep absorption of
                 * beamed I) balanced by red emission D³·jc[s]. eps blends thermal B. */
                double jb = jc[s]; if (!(jb > 0.0) || !isfinite(jb)) jb = B;
                Sl = (1.0 - g_sob_eps) * jb + g_sob_eps * B;
            } else if (g_sob_sl_ptr) {
                /* UNIFIED: NLTE-solved line source (carries fluorescence once the
                 * UV pump populates the upper level above Boltzmann). Thermal
                 * fallback B(T_e) if unset/<=0; optional clamp vs garbage. */
                Sl = g_sob_sl_ptr[(size_t)l * NS + s];
                if (!(Sl > 0.0) || !isfinite(Sl)) Sl = B;
                if (g_sob_sl_clamp > 0.0 && B > 0.0 && Sl > g_sob_sl_clamp * B)
                    Sl = g_sob_sl_clamp * B;
            } else {
                double Jbar;
                if (g_sob_jbardet) {
                    /* ENERGY-CONSERVING: S_l = local line mean intensity. Sobolev
                     * scatter then conserves (absorbed=re-emitted, only redistributed).
                     * Strong lines: producer J̄_l (jbar_line_det); weak/out-of-window
                     * lines (sentinel<0): the local fine-field J (Jv) — BOTH are local
                     * mean intensities. (W*B external backlight was the −47% energy
                     * leak: it sits below the incident I.) */
                    double jl = (opac->jbar_line_det)
                              ? opac->jbar_line_det[(size_t)l * NS + s] : -1.0;
                    Jbar = (jl >= 0.0 && isfinite(jl)) ? jl : Jv;
                } else if (g_sob_srcj) {
                    Jbar = Jv;
                    if (opac->jbar_line) {
                        double jl = opac->jbar_line[(size_t)l * NS + s];
                        if (jl > 0.0 && isfinite(jl)) Jbar = jl;
                    }
                } else {
                    double Wd = 0.5;
                    if (g_sob_rphot > 0.0 && r > g_sob_rphot) {
                        double a = 1.0 - (g_sob_rphot * g_sob_rphot) / (r * r);
                        Wd = 0.5 * (1.0 - sqrt(a > 0.0 ? a : 0.0));
                    }
                    Jbar = Wd * cm_planck(nucmf, g_sob_Tinner);
                }
                Sl = (1.0 - g_sob_eps) * Jbar + g_sob_eps * B;
            }
            if (g_sob_diag && tauS > 1e4) {
                static int nd = 0;
                if (nd < 8) { nd++;
                    printf("[SOB-SRC] s=%d nu_l=%.4e tauS=%.2e Sl=%.4e B=%.4e "
                           "Sl/B=%.4f src=%s\n", s, opac->line_list_nu[l],
                           tauS, Sl, B, (B>0?Sl/B:0.0),
                           g_sob_sl_ptr ? "NLTE" : "scatter"); }
            }
            double exl = (tauS > 700.0) ? 0.0 : exp(-tauS);
            I = I * exl + (g_sob_noemit ? 0.0 : (D * D * D) * Sl) * (1.0 - exl);
        }
    }
    return I;
}

/* Back-compat wrapper: no SEI beamed-continuum column (jc=NULL). */
static inline double cmf_obs_march_sob(double I, double p, double z0, double z1, int s,
                                const CMFGENState *cs, const OpacityState *opac,
                                double Te_s, double nu_obs, double inv_ct)
{ return cmf_obs_march_sob_jc(NULL, I, p, z0, z1, s, cs, opac, Te_s, nu_obs, inv_ct); }

/* Observer-frame emergent spectrum (gate LUMINA_CMF_OBSERVER_FRAME=1).
 *
 * Same tangent-ray geometry as cmfgen_write_spectrum, but a SEPARATE formal
 * solve per observer frequency nu_obs: along each ray the comoving frequency
 * that contributes is nu_cmf(z) = q*nu_obs with q = gamma(1 - mu*beta),
 * beta = r/(c t_exp), mu = +-z/r (signed: inbound far side mu<0, outbound near
 * side mu>0). Material coefficients are evaluated at nu_cmf by interpolation;
 * the moving frame enters as alpha_obs = q*chi_cmf and S_obs = D^3*S_cmf
 * (D=1/q), and the diffusive core emits D_core^3 * B(q_core*nu_obs, T_inner).
 * beta->0 reproduces the comoving cmfgen_write_spectrum. (codex 019eefe6) */
int cmfgen_write_spectrum_obs(const CMFGENState *cs, const Geometry *geo,
                              double T_inner, const OpacityState *opac,
                              const double *Te, const char *path)
{
    int NS = cs->n_shells, NB = cs->n_bins, NR = cs->n_rays;
    double t_exp  = geo->time_explosion;
    double inv_ct = 1.0 / (CM_C * t_exp);
    /* S4: line-resolved Sobolev line synthesis (default ON when line data is
     * available). Replaces the defanged binned chi_line with full-tau_S
     * resonance jumps. Set LUMINA_CMF_OBS_SOBOLEV=0 for the legacy binned path. */
    int sob = (opac && opac->line_list_nu && opac->tau_sobolev && Te);
    { const char *e = getenv("LUMINA_CMF_OBS_SOBOLEV");
      if (e) sob = sob && atoi(e); }
    { const char *e = getenv("LUMINA_CMF_LINE_SOB_EPS");
      g_sob_eps = e ? atof(e) : 0.0;
      if (g_sob_eps < 0.0) g_sob_eps = 0.0;
      if (g_sob_eps > 1.0) g_sob_eps = 1.0; }
    { const char *e = getenv("LUMINA_CMF_OBS_NOEMIT"); g_sob_noemit = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_DIAG"); g_sob_diag = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_SRCJ"); g_sob_srcj = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_CONTONLY"); g_sob_contonly = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_TAUMIN"); g_sob_taumin = e ? atof(e) : 1e-6; }
    g_pray_bin = -1;
    { const char *e = getenv("LUMINA_CMF_OBS_PRAY_LAM");
      if (e) { double tl = atof(e), best = 1e99;
        for (int b = 0; b < NB; ++b) { double la = CM_C / cs->nu[b] * 1.0e8;
          if (fabs(la - tl) < best) { best = fabs(la - tl); g_pray_bin = b; } } } }
    g_sob_rphot  = geo->r_inner[0];
    g_sob_Tinner = T_inner;
    if (g_sob_diag && opac->tau_sobolev) {
        for (int l = 0; l < opac->n_lines; ++l) {
            double la = CM_C / opac->line_list_nu[l] * 1.0e8;
            if (la > 7800.0 && la < 9300.0 &&
                opac->tau_sobolev[(size_t)l * NS] > g_sob_taumin)
                printf("[THICKLINE] rest=%.3fA tau=%.4e\n", la,
                       opac->tau_sobolev[(size_t)l * NS]);
        }
    }

    /* per-(shell,bin) source S = S_fixed + (chi_es/chi_tot) J for interpolation.
     * Interpolate chi_tot and S directly (not eta=chi*S then /chi) to avoid an
     * off-node 0/0 ratio mismatch between independent interpolations. */
    double *Sbin = malloc(sizeof(double) * (size_t)NS * NB);
    double *Lnu  = calloc(NB, sizeof(double));
    if (!Sbin || !Lnu) {
        free(Sbin); free(Lnu);
        fprintf(stderr, "[CMFGEN] obs-spectrum alloc failed\n");
        return -1;
    }
    for (int s = 0; s < NS; ++s)
        for (int b = 0; b < NB; ++b) {
            size_t idx = (size_t)s * NB + b;
            double r = (cs->chi_tot[idx] > 0.0)
                     ? cs->chi_es[idx] / cs->chi_tot[idx] : 0.0;
            Sbin[idx] = cs->S_fixed[idx] + r * cs->J[idx];
        }

    /* Dense observer disk-ray grid: a sharp line P-Cygni needs the iso-velocity
     * annuli (mu*v(r) = const) resolved across the disk; the ~9-ray plasma grid
     * (cs->p_ray) is far too coarse and clips the trough. Uniform in p (=equal
     * area weight p dp). Continuum is unaffected (smooth). LUMINA_CMF_OBS_NRAY. */
    int NRO = 256; { const char *e = getenv("LUMINA_CMF_OBS_NRAY"); if (e) NRO = atoi(e); }
    double r_max_obs = geo->r_outer[NS - 1];
    double *p_obs = malloc(sizeof(double) * NRO);
    for (int kk = 0; kk < NRO; ++kk) p_obs[kk] = r_max_obs * (kk + 0.5) / (double)NRO;

    /* per-observer-frequency formal solve */
    for (int k = 0; k < NB; ++k) {
        double nu_obs = cs->nu[k];
        double integ = 0.0, p_prev = 0.0, f_prev = 0.0;

        for (int ray = 0; ray < NRO; ++ray) {
            double p = p_obs[ray];
            /* intersected shells, outer -> inner (descending r) */
            int    sh[256]; double rmid[256]; int nshell = 0;
            for (int s = NS - 1; s >= 0 && nshell < 256; --s) {
                double ro = geo->r_outer[s];
                if (ro <= p) break;
                double rm = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
                if (rm <= p) rm = p * 1.0000001;
                sh[nshell] = s; rmid[nshell] = rm; ++nshell;
            }
            if (nshell == 0) continue;
            int core = (p < geo->r_inner[0]) ? 1 : 0;

            /* segment z-extents (|z| at shell midpoints), outer->inner */
            double zabs[256];
            for (int i = 0; i < nshell; ++i) {
                double a = rmid[i] * rmid[i] - p * p;
                zabs[i] = (a > 0.0) ? sqrt(a) : 0.0;
            }
            double z_core = 0.0;
            if (core) {
                double ri0 = geo->r_inner[0];
                z_core = sqrt(ri0 * ri0 - p * p);
                if (nshell > 0 && z_core > zabs[nshell - 1]) z_core = zabs[nshell - 1];
            }

            double I = 0.0;   /* outer BC: no incoming radiation */

            /* ---- inbound (far side, z<0): outer -> inner, z increasing ---- */
            for (int i = 0; i < nshell; ++i) {
                if (sob) {
                    /* full shell radial extent [r_inner,r_outer] (NOT the midpoint
                     * — the midpoint collapses the shell to half-thickness and
                     * clips the high-z/high-blueshift resonance covering). */
                    double ro = geo->r_outer[sh[i]], ri = geo->r_inner[sh[i]];
                    double z_hi = (ro > p) ? sqrt(ro * ro - p * p) : 0.0;
                    double z_lo = (ri > p) ? sqrt(ri * ri - p * p) : 0.0;
                    I = cmf_obs_march_sob(I, p, -z_hi, -z_lo, sh[i], cs, opac,
                                          Te[sh[i]], nu_obs, inv_ct);
                } else {
                    double z_hi = zabs[i];                       /* outer edge */
                    double z_lo = (i + 1 < nshell) ? zabs[i + 1]
                                : (core ? z_core : 0.0);          /* inner edge */
                    I = cmf_obs_march(I, p, -z_hi, -z_lo, sh[i], cs->nu, cs->chi_tot,
                                      Sbin, NB, nu_obs, inv_ct, cs->d_log_nu);
                }
            }

            /* ---- diffusive core: D_core^3 * B(q_core*nu_obs, T_inner) ---- */
            if (core) {
                double ri0 = geo->r_inner[0];
                double mu_c = z_core / ri0;
                double beta_in = ri0 * inv_ct;
                double gam_in = 1.0 / sqrt(1.0 - beta_in * beta_in);
                double q_c = gam_in * (1.0 - mu_c * beta_in);
                double D_c = 1.0 / q_c;
                I = cmf_inner_bb_scale() * (D_c * D_c * D_c) *
                    cm_planck(q_c * nu_obs, T_inner);
            }

            /* ---- outbound (near side, z>0): inner -> outer, z increasing ---- */
            for (int i = nshell - 1; i >= 0; --i) {
                if (sob) {
                    double ro = geo->r_outer[sh[i]], ri = geo->r_inner[sh[i]];
                    double z_hi = (ro > p) ? sqrt(ro * ro - p * p) : 0.0;
                    double z_lo = (ri > p) ? sqrt(ri * ri - p * p) : 0.0;
                    I = cmf_obs_march_sob(I, p, +z_lo, +z_hi, sh[i], cs, opac,
                                          Te[sh[i]], nu_obs, inv_ct);
                } else {
                    double z_hi = zabs[i];                       /* outer edge */
                    double z_lo = (i + 1 < nshell) ? zabs[i + 1]
                                : (core ? z_core : 0.0);          /* inner edge */
                    I = cmf_obs_march(I, p, +z_lo, +z_hi, sh[i], cs->nu, cs->chi_tot,
                                      Sbin, NB, nu_obs, inv_ct, cs->d_log_nu);
                }
            }

            double f = I * p;
            if (g_pray_bin == k)
                printf("[PRAY] lam=%.0f ray=%d p=%.4e core=%d I=%.5e\n",
                       CM_C / cs->nu[k] * 1.0e8, ray, p,
                       (p < geo->r_inner[0]) ? 1 : 0, I);
            integ += 0.5 * (f_prev + f) * (p - p_prev);
            p_prev = p; f_prev = f;
        }
        Lnu[k] = 8.0 * M_PI_VAL * M_PI_VAL * integ;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) { free(Sbin); free(Lnu); free(p_obs);
        fprintf(stderr, "[CMFGEN] cannot open %s\n", path); return -1; }
    fprintf(fp, "wavelength_angstrom,flux\n");
    for (int b = NB - 1; b >= 0; --b) {
        double lam_cm = CM_C / cs->nu[b];
        double lam_A  = lam_cm * 1.0e8;
        double L_lam  = Lnu[b] * CM_C / (lam_cm * lam_cm) * 1.0e-8;
        fprintf(fp, "%.6f,%.6e\n", lam_A, L_lam);
    }
    fclose(fp);
    printf("Pure-CMFGEN OBSERVER-frame spectrum -> %s (%d bins, beta_in=%.4f, "
           "lines=%s)\n", path, NB, geo->r_inner[0] * inv_ct,
           sob ? "SOBOLEV-resolved" : "binned");

    free(Sbin); free(Lnu); free(p_obs);
    return 0;
}

/* ------------------------------------------------------------ */
void cmfgen_write_jnu(const CMFGENState *cs, NLTEConfig *nlte)
{
    if (!nlte || !nlte->J_nu) return;
    size_t n = (size_t)cs->n_shells * cs->n_bins;
    memcpy(nlte->J_nu, cs->J, sizeof(double) * n);
}

int cmfgen_commit_jnu(const CMFGENState *cs, NLTEConfig *nlte,
                      const Geometry *geo, const OpacityState *opac,
                      uint64_t generation)
{
    if (!cs || !nlte || !geo || !opac || !cs->J || cs->n_shells <= 0 ||
        cs->n_bins <= 0 || geo->n_shells != cs->n_shells ||
        opac->n_shells != cs->n_shells || !nlte->radiation_field.enabled) {
        fprintf(stderr,
                "[R6][BLOCKED] reason=DETERMINISTIC_OWNER_PRECONDITION "
                "cs=%p nlte=%p geo=%p opac=%p J=%p shells=%d bins=%d "
                "geo_shells=%d opac_shells=%d owner_enabled=%d "
                "request_generation=%llu\n",
                (const void *)cs, (void *)nlte, (const void *)geo,
                (const void *)opac, cs ? (void *)cs->J : NULL,
                cs ? cs->n_shells : -1, cs ? cs->n_bins : -1,
                geo ? geo->n_shells : -1, opac ? opac->n_shells : -1,
                nlte ? nlte->radiation_field.enabled : 0,
                (unsigned long long)generation);
        return -1;
    }

    const char *line_producer = NULL;
    if (opac->jbar_line_det_operator ==
            CMF_FINE_LINE_OPERATOR_INIT_SHARED_GAUSSIAN)
        line_producer = LUMINA_LINE_JBAR_DETERMINISTIC_PRODUCER;
    else if (opac->jbar_line_det_operator ==
                 CMF_FINE_LINE_OPERATOR_CMFGEN_NONOVERLAP_SOBOLEV)
        line_producer =
            LUMINA_LINE_JBAR_CMFGEN_NONOVERLAP_SOBOLEV_PRODUCER;
    else {
        fprintf(stderr,
                "[R6][BLOCKED] reason=DETERMINISTIC_LINE_OPERATOR_MISSING "
                "operator=%d\n", opac->jbar_line_det_operator);
        return -1;
    }

    const LineJbarQSet *qset = (const LineJbarQSet *)nlte->line_qset;
    const LineJbarESet *eset = (const LineJbarESet *)nlte->line_eset;
    if (!qset || qset->n_q == 0 || !qset->line_id || !qset->line_nu ||
        strlen(qset->q_set_hash) != 64 || strlen(qset->profile_hash) != 64 ||
        !eset || eset->n_q == 0 || !eset->line_id || !eset->line_nu ||
        strlen(eset->q_set_hash) != 64 || strlen(eset->profile_hash) != 64 ||
        !opac->jbar_line_det || !opac->line_list_nu) {
        fprintf(stderr,
                "[R6][BLOCKED] reason=DETERMINISTIC_LINE_SET_MISSING\n");
        return -1;
    }
    size_t first_missing_q = SIZE_MAX;
    LineJbarSubsetStatus subset_status = line_jbar_qset_subset_of_eset(
        qset, eset, &first_missing_q);
    if (subset_status != LINE_JBAR_SUBSET_OK) {
        fprintf(stderr,
                "[R6][BLOCKED] reason=QG_NOT_SUBSET_QE status=%d "
                "first_missing_q=%zu q_lines=%zu e_lines=%zu\n",
                (int)subset_status, first_missing_q, qset->n_q, eset->n_q);
        return -1;
    }
    if (qset->profile_id != LINE_JBAR_PROFILE_GAUSS_VD10 ||
        strcmp(qset->profile_hash, LINE_JBAR_PROFILE_SHA256) != 0 ||
        strcmp(qset->domain_contract_hash,
               LINE_JBAR_BB_DOMAIN_CONTRACT_SHA256) != 0 ||
        opac->jbar_line_det_vdoppler_cms != LINE_JBAR_VDOPPLER_CMS ||
        opac->jbar_line_det_ndoppler != LINE_JBAR_PROFILE_NDOPPLER) {
        fprintf(stderr,
                "[R6][BLOCKED] reason=PROFILE_MISMATCH "
                "q_profile=%llu profile_hash=%s domain_hash=%s "
                "det_vdop=%.17g det_ndop=%.17g\n",
                (unsigned long long)qset->profile_id,
                qset->profile_hash, qset->domain_contract_hash,
                opac->jbar_line_det_vdoppler_cms,
                opac->jbar_line_det_ndoppler);
        return -1;
    }
    if (!opac->jbar_line_det_exact_converged ||
        opac->jbar_line_det_exact_iterations < 2 ||
        opac->jbar_line_det_exact_iterations >
            opac->jbar_line_det_exact_iteration_cap ||
        !(opac->jbar_line_det_exact_tolerance > 0.0) ||
        !isfinite(opac->jbar_line_det_exact_tolerance) ||
        !(opac->jbar_line_det_exact_residual >= 0.0) ||
        !isfinite(opac->jbar_line_det_exact_residual) ||
        !(opac->jbar_line_det_exact_residual <
          opac->jbar_line_det_exact_tolerance) ||
        !(opac->jbar_line_det_exact_max_scattering_ratio >= 0.0) ||
        !(opac->jbar_line_det_exact_max_scattering_ratio < 1.0) ||
        !opac->jbar_line_det_error_upper ||
        !opac->jbar_line_det_error_envelope_verified ||
        opac->jbar_line_det_error_refinement_iterations == 0 ||
        !(opac->jbar_line_det_component_error_min >= 0.0) ||
        !isfinite(opac->jbar_line_det_component_error_min) ||
        !(opac->jbar_line_det_component_error_max >=
          opac->jbar_line_det_component_error_min) ||
        !isfinite(opac->jbar_line_det_component_error_max) ||
        !(opac->jbar_line_det_profile_error_min >= 0.0) ||
        !isfinite(opac->jbar_line_det_profile_error_min) ||
        !(opac->jbar_line_det_profile_error_max >=
          opac->jbar_line_det_profile_error_min) ||
        !isfinite(opac->jbar_line_det_profile_error_max) ||
        opac->jbar_line_det_grid_n_bins < 2 ||
        !(opac->jbar_line_det_grid_nu_min > 0.0) ||
        !(opac->jbar_line_det_grid_nu_max >
          opac->jbar_line_det_grid_nu_min)) {
        fprintf(stderr,
                "[R6][BLOCKED] reason=EXACT_SOLVER_QUALIFICATION "
                "converged=%d iterations=%d cap=%d residual=%.17g "
                "tolerance=%.17g absolute_error_bound=%.17g "
                "scattering_ratio_bound=%.17g fine_bins=%d "
                "component_envelope=%d refinements=%zu "
                "component_error=[%.17g,%.17g] profile_error=[%.17g,%.17g] "
                "fine_nu=[%.17g,%.17g]\n",
                opac->jbar_line_det_exact_converged,
                opac->jbar_line_det_exact_iterations,
                opac->jbar_line_det_exact_iteration_cap,
                opac->jbar_line_det_exact_residual,
                opac->jbar_line_det_exact_tolerance,
                opac->jbar_line_det_exact_absolute_error_bound,
                opac->jbar_line_det_exact_max_scattering_ratio,
                opac->jbar_line_det_grid_n_bins,
                opac->jbar_line_det_error_envelope_verified,
                opac->jbar_line_det_error_refinement_iterations,
                opac->jbar_line_det_component_error_min,
                opac->jbar_line_det_component_error_max,
                opac->jbar_line_det_profile_error_min,
                opac->jbar_line_det_profile_error_max,
                opac->jbar_line_det_grid_nu_min,
                opac->jbar_line_det_grid_nu_max);
        return -1;
    }

    size_t first_bad_e = SIZE_MAX;
    if (line_jbar_qset_profile_support_covered(
            eset, opac->jbar_line_det_grid_nu_min,
            opac->jbar_line_det_grid_nu_max, &first_bad_e) != 0) {
        if (first_bad_e < eset->n_q) {
            double profile_width = LINE_JBAR_PROFILE_NDOPPLER *
                                   LINE_JBAR_VDOPPLER_CMS / CM_C;
            double line_nu = eset->line_nu[first_bad_e];
            fprintf(stderr,
                    "[R6][BLOCKED] reason=ESET_PROFILE_SUPPORT_COVERAGE "
                    "e=%zu line_id=%d line_nu=%.17g support=[%.17g,%.17g] "
                    "fine=[%.17g,%.17g] domain_hash=%s\n",
                    first_bad_e, eset->line_id[first_bad_e], line_nu,
                    line_nu * (1.0 - profile_width),
                    line_nu * (1.0 + profile_width),
                    opac->jbar_line_det_grid_nu_min,
                    opac->jbar_line_det_grid_nu_max,
                    eset->domain_contract_hash);
        } else {
            fprintf(stderr,
                    "[R6][BLOCKED] reason=ESET_PROFILE_SUPPORT_PRECONDITION "
                    "e_lines=%zu fine=[%.17g,%.17g] domain_hash=%s\n",
                    eset->n_q, opac->jbar_line_det_grid_nu_min,
                    opac->jbar_line_det_grid_nu_max,
                    eset->domain_contract_hash);
        }
        return -1;
    }

    size_t bins = (size_t)cs->n_bins;
    size_t cells = (size_t)cs->n_shells * bins;
    if (eset->n_q > SIZE_MAX / (size_t)cs->n_shells) return -1;
    size_t line_cells = eset->n_q * (size_t)cs->n_shells;
    double *edges = (double *)malloc((bins + 1) * sizeof(double));
    RadiationFieldValidityState *validity =
        (RadiationFieldValidityState *)malloc(cells * sizeof(*validity));
    uint64_t *line_id = (uint64_t *)malloc(eset->n_q * sizeof(*line_id));
    uint64_t *rate_line_id =
        (uint64_t *)malloc(qset->n_q * sizeof(*rate_line_id));
    double *line_jbar = (double *)malloc(line_cells * sizeof(*line_jbar));
    double *line_error = (double *)malloc(line_cells * sizeof(*line_error));
    int32_t *line_validity =
        (int32_t *)malloc(line_cells * sizeof(*line_validity));
    if (!edges || !validity || !line_id || !rate_line_id || !line_jbar ||
        !line_error || !line_validity) {
        free(edges); free(validity); free(line_id); free(rate_line_id);
        free(line_jbar); free(line_error);
        free(line_validity);
        return -1;
    }
    for (size_t q = 0; q < qset->n_q; ++q)
        rate_line_id[q] = (uint64_t)qset->line_id[q];
    for (size_t b = 0; b <= bins; ++b)
        edges[b] = cs->nu_min * exp((double)b * cs->d_log_nu);
    edges[0] = cs->nu_min;
    edges[bins] = cs->nu_max;
    for (size_t i = 0; i < cells; ++i) {
        if (!isfinite(cs->J[i]) || cs->J[i] < 0.0) {
            fprintf(stderr,
                    "[R6][BLOCKED] reason=DETERMINISTIC_J_INVALID "
                    "cell=%zu value=%.17g generation=%llu\n",
                    i, cs->J[i], (unsigned long long)generation);
            free(edges); free(validity); free(line_id); free(rate_line_id);
            free(line_jbar); free(line_error);
            free(line_validity);
            return -1;
        }
        validity[i] = cs->J[i] == 0.0
            ? RADIATION_FIELD_EXACT_ZERO : RADIATION_FIELD_VALID;
    }

    size_t valid_lines = 0, partial_lines = 0, unsampled_lines = 0;
    size_t valid_cells = 0, exact_zero_cells = 0;
    for (size_t e = 0; e < eset->n_q; e++) {
        int lid = eset->line_id[e];
        if (lid < 0 || lid >= opac->n_lines ||
            !isfinite(eset->line_nu[e]) || eset->line_nu[e] <= 0.0 ||
            eset->line_nu[e] != opac->line_list_nu[lid]) {
            fprintf(stderr,
                    "[R6][BLOCKED] reason=ESET_LINE_IDENTITY_MISMATCH e=%zu "
                    "line_id=%d\n", e, lid);
            free(edges); free(validity); free(line_id); free(rate_line_id);
                free(line_jbar); free(line_error);
            free(line_validity);
            return -1;
        }
        line_id[e] = (uint64_t)lid;
        size_t covered_shells = 0;
        for (size_t s = 0; s < (size_t)cs->n_shells; s++) {
            size_t src = (size_t)lid * (size_t)cs->n_shells + s;
            size_t dst = e * (size_t)cs->n_shells + s;
            double raw = opac->jbar_line_det[src];
            double error_upper = opac->jbar_line_det_error_upper[src];
            if (raw == -1.0) {
                if (error_upper != -1.0) {
                    fprintf(stderr,
                            "[R6][BLOCKED] reason=LINE_ERROR_SENTINEL_MISMATCH "
                            "line_id=%d shell=%zu Jbar=%.17g error=%.17g\n",
                            lid, s, raw, error_upper);
                    free(edges); free(validity); free(line_id);
                    free(rate_line_id); free(line_jbar); free(line_error);
                    free(line_validity);
                    return -1;
                }
                line_jbar[dst] = 0.0;
                line_error[dst] = 0.0;
                line_validity[dst] = LINE_JBAR_UNSAMPLED;
            } else if (!isfinite(raw) || raw < 0.0 ||
                       !(error_upper >= 0.0) || !isfinite(error_upper)) {
                fprintf(stderr,
                        "[R6][BLOCKED] reason=INVALID_PRIVATE_SENTINEL "
                        "line_id=%d shell=%zu value=%.17g error=%.17g\n",
                        lid, s, raw, error_upper);
                free(edges); free(validity); free(line_id);
                free(rate_line_id); free(line_jbar); free(line_error);
                free(line_validity);
                return -1;
            } else {
                line_jbar[dst] = raw;
                line_error[dst] = error_upper;
                line_validity[dst] = raw == 0.0
                    ? LINE_JBAR_EXACT_ZERO : LINE_JBAR_VALID;
                covered_shells++;
                if (raw == 0.0) exact_zero_cells++; else valid_cells++;
            }
        }
        if (covered_shells == (size_t)cs->n_shells) valid_lines++;
        else if (covered_shells > 0) partial_lines++;
        else unsampled_lines++;
    }
    if (valid_lines != eset->n_q || partial_lines != 0 ||
        unsampled_lines != 0 || valid_cells + exact_zero_cells == 0) {
        fprintf(stderr,
                "[R6][BLOCKED] reason=DETERMINISTIC_LINE_JBAR_INCOMPLETE "
                "e_lines=%zu valid_lines=%zu partial_lines=%zu "
                "unsampled_lines=%zu valid_cells=%zu exact_zero_cells=%zu "
                "residual=%.17g tolerance=%.17g\n",
                eset->n_q, valid_lines, partial_lines, unsampled_lines,
                valid_cells, exact_zero_cells,
                opac->jbar_line_det_exact_residual,
                opac->jbar_line_det_exact_tolerance);
        free(edges); free(validity); free(line_id); free(rate_line_id);
        free(line_jbar); free(line_error);
        free(line_validity);
        return -1;
    }

    RadiationFieldCommitRequest request;
    memset(&request, 0, sizeof(request));
    request.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
    request.producer = "PURE_CMFGEN_COMOVING_CONSERVATIVE_REBIN";
    request.generation = generation;
    request.epoch = geo->time_explosion;
    request.n_shells = (size_t)cs->n_shells;
    request.v_inner = geo->v_inner;
    request.v_outer = geo->v_outer;
    request.source_n_bins = bins;
    request.source_frequency_bin_edges = edges;
    request.source_J_nu = cs->J;
    request.source_validity = validity;
    request.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
    request.line_n = eset->n_q;
    request.line_id = line_id;
    request.line_q_set_hash = eset->q_set_hash;
    request.line_set_kind = LINE_JBAR_SET_ENERGY_DOMAIN;
    request.line_rate_graph_n = qset->n_q;
    request.line_rate_graph_id = rate_line_id;
    request.line_rate_graph_hash = qset->q_set_hash;
    request.line_profile_id = eset->profile_id;
    request.line_profile_hash = eset->profile_hash;
    request.line_provenance_kind =
        RADIATION_FIELD_PROVENANCE_CMFGEN_LINE_PROFILE_INTEGRAL;
    request.line_producer = line_producer;
    request.line_jbar = line_jbar;
    request.line_error_upper = line_error;
    request.line_validity = line_validity;
    int rc = radiation_field_commit(&nlte->radiation_field, &request);
    free(edges); free(validity); free(line_id); free(rate_line_id);
    free(line_jbar); free(line_error);
    free(line_validity);
    if (rc != 0) {
        fprintf(stderr,
                "[R6][BLOCKED] reason=CANONICAL_COMMIT_REJECTED "
                "request_generation=%llu owner_required=%llu "
                "owner_computed=%llu owner_valid=%d\n",
                (unsigned long long)generation,
                (unsigned long long)nlte->radiation_field.field.generation.required_generation,
                (unsigned long long)nlte->radiation_field.field.generation.computed_generation,
                radiation_field_validate_owner(&nlte->radiation_field) == 0);
        return -1;
    }
    /* A2-05: the only replay-lane view refresh point. */
    nlte->radfield_view_status = radiation_field_read_view(
        &nlte->radiation_field, geo->time_explosion, (size_t)cs->n_shells,
        generation, &nlte->radfield_view);
    if (nlte->radfield_view_status != RADIATION_FIELD_VIEW_OK) return -1;
    nlte->line_view_status = radiation_field_line_jbar_rate_view(
        &nlte->radiation_field, geo->time_explosion, (size_t)cs->n_shells,
        generation, qset->q_set_hash, qset->profile_id, qset->profile_hash,
        &nlte->line_view);
    if (nlte->line_view_status != LINE_JBAR_VIEW_OK) {
        fprintf(stderr,
                "[R6][BLOCKED] reason=LINE_JBAR_VIEW status=%d\n",
                nlte->line_view_status);
        return -1;
    }
    nlte->line_energy_view_status = radiation_field_line_jbar_energy_view(
        &nlte->radiation_field, geo->time_explosion, (size_t)cs->n_shells,
        generation, eset->q_set_hash, eset->profile_id, eset->profile_hash,
        &nlte->line_energy_view);
    if (nlte->line_energy_view_status != LINE_JBAR_VIEW_OK) {
        fprintf(stderr,
                "[R6][BLOCKED] reason=LINE_JBAR_ENERGY_VIEW status=%d\n",
                nlte->line_energy_view_status);
        return -1;
    }
    fprintf(stderr,
            "[R6][LINE-IDENTITY] lane=DET generation=%llu "
            "q_lines=%zu q_set_hash=%s e_lines=%zu e_set_hash=%s "
            "domain_hash=%s profile_id=%llu profile_hash=%s "
            "exact_iterations=%d exact_cap=%d exact_residual=%.17g "
            "exact_tolerance=%.17g absolute_error_bound=%.17g "
            "scattering_ratio_bound=%.17g fine_bins=%d "
            "component_envelope=%d refinements=%zu "
            "component_error=[%.17g,%.17g] profile_error=[%.17g,%.17g] "
            "fine_nu=[%.17g,%.17g] "
            "statistic_kind=DETERMINISTIC provenance=%s\n",
            (unsigned long long)generation, qset->n_q, qset->q_set_hash,
            eset->n_q, eset->q_set_hash,
            qset->domain_contract_hash,
            (unsigned long long)qset->profile_id, qset->profile_hash,
            opac->jbar_line_det_exact_iterations,
            opac->jbar_line_det_exact_iteration_cap,
            opac->jbar_line_det_exact_residual,
            opac->jbar_line_det_exact_tolerance,
            opac->jbar_line_det_exact_absolute_error_bound,
            opac->jbar_line_det_exact_max_scattering_ratio,
            opac->jbar_line_det_grid_n_bins,
            opac->jbar_line_det_error_envelope_verified,
            opac->jbar_line_det_error_refinement_iterations,
            opac->jbar_line_det_component_error_min,
            opac->jbar_line_det_component_error_max,
            opac->jbar_line_det_profile_error_min,
            opac->jbar_line_det_profile_error_max,
            opac->jbar_line_det_grid_nu_min,
            opac->jbar_line_det_grid_nu_max,
            line_producer);
    fprintf(stderr,
            "[R6][LINE-COVERAGE] generation=%llu all_lines=%d q_lines=%zu "
            "e_lines=%zu "
            "valid_lines=%zu partial_lines=%zu unsampled_lines=%zu "
            "valid_pct_eset=%.6f valid_pct_all=%.6f "
            "valid_cells=%zu exact_zero_cells=%zu\n",
            (unsigned long long)generation, opac->n_lines, qset->n_q,
            eset->n_q, valid_lines, partial_lines, unsampled_lines,
            100.0 * (double)valid_lines / (double)eset->n_q,
            100.0 * (double)valid_lines / (double)opac->n_lines,
            valid_cells, exact_zero_cells);

    /* Temporary compatibility view: A2-05+ migrates consumers to the canonical
     * field.  This copy preserves pre-migration rate/opacity/emissivity output. */
    cmfgen_write_jnu(cs, nlte);
    return 0;
}

/* ------------------------------------------------------------ */
/* ============================================================ */
/* P1: line-resolved comoving-frame (CMF) J solver (gate LUMINA_CMF_LINERES=1).
 * Replaces the per-bin frequency-DECOUPLED formal_solve_bin with a single
 * frequency-COUPLED sweep over all bins (b descending = blue->red), using the
 * validated PHOENIX/Hauschildt-Baron conservative upwind (chih=chi+a_lam*(4+
 * lambda/Dlam)) + Olson-Kunasz linear SC + tangent-ray mu-quadrature. Operates
 * on the SAME 1000-bin grid as the binned solver; gate 1a (cont_only) must
 * reproduce cmfgen_solve_J to <0.5% L2. cmfgen_solve_J is the untouched fallback.
 *   LUMINA_CMF_ALAM=0 disables the homologous freq advection (static limit, the
 *   transport-only sub-gate); =1 (default) the full CMF coupling.
 * Source = S_fixed + (chi_es/chi_tot)*J (the ALI scattering source, same as the
 * binned solver). Validated standalone in lumina_cmf_selftest.c (gates 2a/4a/2c). */
static int cmf_solve_J(CMFGENState *cs, const Geometry *geo, double T_inner,
                       int n_ali_iter)
{
    int NS = cs->n_shells, NB = cs->n_bins;
    double t_exp = geo->time_explosion;
    double a_lam_on = 1.0;
    { const char *e = getenv("LUMINA_CMF_ALAM"); if (e) a_lam_on = atof(e); }
    double a_lam = a_lam_on / (t_exp * CM_C);
    /* ADV_SPLIT (LUMINA_CMF_ADV_SPLIT=1): flux-limited operator split for the
     * comoving blue->red advection (physics review 2026-06-26). The HB upwind
     * form lumps beta into the spatial dtau=(chi+beta)*ds; at this grid the
     * frequency-advection Courant number beta*ds~80 (a shell spans ~80 fine
     * bins of redshift) so e^{-80} annihilates the radially-transported
     * upstream intensity -> the optical e-scatter continuum collapses to
     * J/B~1e-4 at outer shells. Fix: radial short-char with the REAL opacity
     * (dtau=chi*ds, retains radial memory) + a SEPARATE capped advection pass
     * I += min(beta*ds,1)*(I_bluer - I). Smooth continuum (I_bluer~=I) -> ~0
     * correction -> radial W*B restored; lines still redistribute (capped).
     * Off (default) = legacy lumped HB form (byte-identical). */
    int adv_split = 0;
    { const char *e = getenv("LUMINA_CMF_ADV_SPLIT"); if (e) adv_split = atoi(e); }
    /* GPU formal solver (lumina_cmf_solve.cu). 0=CPU/OMP (byte-identical legacy),
     * 1=GPU (lagged-advection NB*NP kernel), 2=run BOTH and report max rel diff
     * in the converged J (self-check). The producer's fine grid (NB~500k) is the
     * driver of this path; the GPU lags the blue->red advection across ALI iters
     * to expose all (bin,ray) as independent threads -- see lumina_cmf_solve.cu. */
    int use_gpu = 0;
    { const char *e = getenv("LUMINA_CMF_SOLVE_GPU"); if (e) use_gpu = atoi(e); }

    double *rmid = malloc(NS * sizeof(double));
    double *lam  = malloc(NB * sizeof(double));
    for (int s = 0; s < NS; ++s) rmid[s] = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
    for (int b = 0; b < NB; ++b) lam[b] = CM_C / cs->nu[b];  /* nu asc -> lam desc */

    int NCORE = 16, NP = NS + NCORE;
    double *p = malloc(NP * sizeof(double));
    for (int k = 0; k < NCORE; ++k) p[k] = rmid[0] * k / (double)NCORE;
    for (int s = 0; s < NS; ++s) p[NCORE + s] = rmid[s];
    int    *rn   = calloc(NP, sizeof(int));
    int    *rsh  = malloc((size_t)NP * (NS + 1) * sizeof(int));
    double *rz   = malloc((size_t)NP * (NS + 1) * sizeof(double));
    int    *rcore= calloc(NP, sizeof(int));
    double *rzin = calloc(NP, sizeof(double));
    for (int k = 0; k < NP; ++k) {
        double pk = p[k]; int n = 0;
        for (int s = NS - 1; s >= 0; --s) {
            if (rmid[s] <= pk) break;
            rsh[(size_t)k*(NS+1)+n] = s; rz[(size_t)k*(NS+1)+n] = sqrt(rmid[s]*rmid[s]-pk*pk); ++n;
        }
        rn[k] = n; rcore[k] = (pk < rmid[0]); rzin[k] = rcore[k] ? sqrt(rmid[0]*rmid[0]-pk*pk) : 0.0;
    }
    double *Iin_p =calloc((size_t)NP*(NS+1),sizeof(double)),*Iout_p=calloc((size_t)NP*(NS+1),sizeof(double));
    double *Iin_c =calloc((size_t)NP*(NS+1),sizeof(double)),*Iout_c=calloc((size_t)NP*(NS+1),sizeof(double));
    double *muL=malloc((size_t)NS*NP*sizeof(double)),*IpL=malloc((size_t)NS*NP*sizeof(double)),*ImL=malloc((size_t)NS*NP*sizeof(double));
    int *cnt = malloc(NS * sizeof(int));
    double *S    = malloc((size_t)NS*NB*sizeof(double));
    double *Jnew = malloc((size_t)NS*NB*sizeof(double));

    /* --- GPU aux tables (built only when LUMINA_CMF_SOLVE_GPU>=1) --------------
     * Bin/adv/advcoef are the per-bin scalars the kernel needs; the per-shell
     * mu-sorted sample tables let the J-integration kernel gather each shell's ray
     * crossings without atomics (geometry fixes which rays cross a shell + their
     * mu = rz/rmid). */
    double *Bin_h=NULL,*adv_h=NULL,*advc_h=NULL,*Jin=NULL,*Jcpu=NULL;
    int *shell_off=NULL,*shell_k=NULL,*shell_seg=NULL; double *shell_mu=NULL;
    int nsamp=0, cpu_iters=n_ali_iter, gpu_iters=0, solve_rc=0;
    if (use_gpu >= 1) {
        Bin_h=malloc(NB*sizeof(double)); adv_h=malloc(NB*sizeof(double)); advc_h=malloc(NB*sizeof(double));
        for (int b=0;b<NB;++b){ Bin_h[b]=cm_planck(cs->nu[b],T_inner);
            if (b<NB-1){ double Dlam=lam[b]-lam[b+1]; adv_h[b]=a_lam*lam[b]/Dlam; advc_h[b]=a_lam*lam[b+1]/Dlam; }
            else { adv_h[b]=0.0; advc_h[b]=0.0; } }
        int *sc=calloc(NS,sizeof(int));
        for (int k=0;k<NP;++k){ size_t kb=(size_t)k*(NS+1); for (int i=0;i<rn[k];++i) sc[rsh[kb+i]]++; }
        shell_off=malloc((NS+1)*sizeof(int)); shell_off[0]=0;
        for (int s=0;s<NS;++s) shell_off[s+1]=shell_off[s]+sc[s];
        nsamp=shell_off[NS];
        shell_k=malloc((size_t)nsamp*sizeof(int)); shell_seg=malloc((size_t)nsamp*sizeof(int));
        shell_mu=malloc((size_t)nsamp*sizeof(double));
        int *fl=calloc(NS,sizeof(int));
        for (int k=0;k<NP;++k){ size_t kb=(size_t)k*(NS+1);
            for (int i=0;i<rn[k];++i){ int s=rsh[kb+i]; int pos=shell_off[s]+fl[s]++;
                shell_k[pos]=k; shell_seg[pos]=i; shell_mu[pos]=rz[kb+i]/rmid[s]; } }
        for (int s=0;s<NS;++s){ int o0=shell_off[s],o1=shell_off[s+1];   /* sort each shell ascending mu */
            for (int a=o0+1;a<o1;++a){ double mk=shell_mu[a]; int kk=shell_k[a],ss=shell_seg[a]; int q=a-1;
                while(q>=o0&&shell_mu[q]>mk){ shell_mu[q+1]=shell_mu[q]; shell_k[q+1]=shell_k[q]; shell_seg[q+1]=shell_seg[q]; --q; }
                shell_mu[q+1]=mk; shell_k[q+1]=kk; shell_seg[q+1]=ss; } }
        free(sc); free(fl);
        if (use_gpu==2){ Jin=malloc((size_t)NS*NB*sizeof(double)); memcpy(Jin,cs->J,(size_t)NS*NB*sizeof(double)); }
    }

    /* The lagged-advection GPU scheme is BYTE-EXACT to the CPU and converges in
     * the SAME ALI count ONLY when the blue->red frequency coupling is off
     * (a_lam==0, i.e. LUMINA_CMF_ALAM=0 -- the static/transport-only limit, which
     * is also what the fine emergent step uses). With a_lam>0 the coupling is
     * Courant-dominant (adv*ds~10^2 at the producer's fine resolution -> I_b~=I_b+1)
     * and lagging advances the field only ~1 bin/ALI-iter, so it reaches the same
     * fixed point but needs O(NB) iters (verified: NB=400 -> 409 iters; impractical
     * at NB=500k vs the ~24 budget). Warn loudly so GPU=1 is not used blindly with
     * advection on. */
    if (use_gpu >= 1 && a_lam != 0.0)
        fprintf(stderr, "[cmf_gpu] WARNING: LUMINA_CMF_ALAM!=0 (advection on): the lagged "
            "GPU solver needs O(NB) ALI iters to converge -> the field at n_ali=%d is "
            "likely NOT converged. Use LUMINA_CMF_ALAM=0 (static limit) for the GPU path.\n",
            n_ali_iter);

    /* use_gpu==1 is one GPU attempt.  Failure is terminal for that attempt:
     * never execute or publish the CPU solver as a replacement. */
    int run_cpu = (use_gpu != 1);
    if (use_gpu == 1) {
        int rc = cmf_solve_J_gpu(NS, NB, NP, adv_split, a_lam,
            cs->chi_tot, cs->chi_es, cs->chi_abs, cs->S_fixed, cs->J,
            Bin_h, adv_h, advc_h, rn, rsh, rz, rcore, rzin,
            shell_off, shell_k, shell_seg, shell_mu, nsamp, n_ali_iter, 1e-4, &gpu_iters);
        if (rc != 0) {
            fprintf(stderr, "[cmf_gpu] GPU solve failed rc=%d "
                    "BLOCKED_GPU_FALLBACK_FORBIDDEN fallback_attempts=1 "
                    "physical_launches=0\n", rc);
            solve_rc = -1;
            goto cmf_solve_cleanup;
        }
    }
    if (use_gpu == 2) {
        /* A/B ordering is GPU-first.  Therefore a GPU failure cannot execute,
         * much less publish, the CPU solver in the same attempt. */
        int rc = cmf_solve_J_gpu(NS, NB, NP, adv_split, a_lam,
            cs->chi_tot, cs->chi_es, cs->chi_abs, cs->S_fixed, cs->J,
            Bin_h, adv_h, advc_h, rn, rsh, rz, rcore, rzin,
            shell_off, shell_k, shell_seg, shell_mu, nsamp, n_ali_iter, 1e-4, &gpu_iters);
        if (rc != 0) {
            fprintf(stderr, "[cmf_gpu] SELF-CHECK GPU-first solve failed rc=%d "
                    "BLOCKED_GPU_FALLBACK_FORBIDDEN fallback_attempts=1 "
                    "physical_launches=0\n", rc);
            memcpy(cs->J, Jin, (size_t)NS*NB*sizeof(double));
            solve_rc = -1;
            goto cmf_solve_cleanup;
        }
        Jcpu = malloc((size_t)NS*NB*sizeof(double));
        if (!Jcpu) {
            memcpy(cs->J, Jin, (size_t)NS*NB*sizeof(double));
            solve_rc = -1;
            goto cmf_solve_cleanup;
        }
        memcpy(Jcpu, cs->J, (size_t)NS*NB*sizeof(double)); /* GPU result */
        memcpy(cs->J, Jin, (size_t)NS*NB*sizeof(double));  /* CPU same input */
    }

    if (run_cpu)
    for (int it = 0; it < n_ali_iter; ++it) {
        for (int s = 0; s < NS; ++s) for (int b = 0; b < NB; ++b) {
            size_t idx = (size_t)s*NB+b;
            double r = (cs->chi_tot[idx] > 0.0) ? cs->chi_es[idx]/cs->chi_tot[idx] : 0.0;
            S[idx] = cs->S_fixed[idx] + r * cs->J[idx];
        }
        memset(Iin_p, 0, (size_t)NP*(NS+1)*sizeof(double));
        memset(Iout_p,0, (size_t)NP*(NS+1)*sizeof(double));
        for (int b = NB - 1; b >= 0; --b) {                 /* bluest (high nu) first */
            double Dlam = (b < NB-1) ? (lam[b]-lam[b+1]) : 0.0;   /* >0 */
            double adv  = (b < NB-1) ? a_lam*(lam[b]/Dlam) : 0.0;
            double lam_b= (b < NB-1) ? lam[b+1] : lam[b];
            double Bin  = cm_planck(cs->nu[b], T_inner);
            memset(cnt, 0, NS*sizeof(int));
            /* The rays (k) are INDEPENDENT within a frequency (the blue->red
             * advection couples only across b, which is the outer sequential
             * loop). Parallelize over rays. The only shared write target is the
             * per-shell accumulator cnt[s]/muL/ImL/IpL: each (ray,shell) crossing
             * reserves a unique slot via an atomic capture on cnt[s] (in the
             * inbound pass), stored in the per-ray local cloc[] and reused in the
             * outbound pass. Slot order across rays becomes arbitrary, but the
             * per-shell J integration below sorts by mu, so J is order-invariant.
             * Iin_c/Iout_c writes are per-ray (kb range) -> no race. */
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
            #endif
            for (int k = 0; k < NP; ++k) {
                int n = rn[k]; if (n == 0) continue; size_t kb = (size_t)k*(NS+1);
                int cloc[300];   /* reserved slot per segment (NS+1 <= 300, cf. mu[300]) */
                double I = 0.0;
                for (int i = 0; i < n; ++i) {
                    int s = rsh[kb+i]; double ds = (i+1<n)?(rz[kb+i]-rz[kb+i+1]):(rz[kb+i]-rzin[k]);
                    size_t idx = (size_t)s*NB+b; double chi = cs->chi_tot[idx], chih = chi + a_lam*4.0 + adv;
                    if (adv_split) {
                        /* radial transport with REAL opacity (+tiny sphericity), then
                         * a capped frequency-advection pass (operator split). */
                        double chi_rad = chi + a_lam*4.0;
                        double Su = S[idx];
                        double dtau=chi_rad*ds, ex=exp(-dtau), e0,e1,wu,wd;
                        if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                        double I_rad = I*ex + (wu+wd)*Su;
                        /* implicit upwind frequency advection (stable at any Courant
                         * beta*ds): smooth continuum (Iblue~=I_rad) -> I_rad (W*B
                         * retained); line (Iblue low) -> propagates absorption. */
                        double bds = adv*ds;
                        double Iblue = (b<NB-1)?Iin_p[kb+i]:I_rad;
                        I = (I_rad + bds*Iblue)/(1.0 + bds); Iin_c[kb+i]=I;
                    } else {
                    double Su = (S[idx]*chi + ((b<NB-1)?a_lam*(lam_b/Dlam)*Iin_p[kb+i]:0.0))/(chih>0?chih:1.0);
                    double dtau=chih*ds, ex=exp(-dtau), e0,e1,wu,wd;
                    if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                    I = I*ex + wu*Su + wd*Su; Iin_c[kb+i]=I;
                    }
                    int c;
                    #ifdef _OPENMP
                    #pragma omp atomic capture
                    #endif
                    { c = cnt[s]; cnt[s] += 1; }
                    cloc[i] = c; muL[(size_t)s*NP+c]=rz[kb+i]/rmid[s]; ImL[(size_t)s*NP+c]=I;
                }
                if (rcore[k]) I = Bin;
                for (int i = n-1; i >= 0; --i) {
                    int s = rsh[kb+i]; double ds = (i+1<n)?(rz[kb+i]-rz[kb+i+1]):(rz[kb+i]-rzin[k]);
                    size_t idx = (size_t)s*NB+b; double chi = cs->chi_tot[idx], chih = chi + a_lam*4.0 + adv;
                    if (adv_split) {
                        double chi_rad = chi + a_lam*4.0;
                        double Su = S[idx];
                        double dtau=chi_rad*ds, ex=exp(-dtau), e0,e1,wu,wd;
                        if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                        double I_rad = I*ex + (wu+wd)*Su;
                        double bds = adv*ds;
                        double Iblue = (b<NB-1)?Iout_p[kb+i]:I_rad;
                        I = (I_rad + bds*Iblue)/(1.0 + bds); Iout_c[kb+i]=I;
                    } else {
                    double Su = (S[idx]*chi + ((b<NB-1)?a_lam*(lam_b/Dlam)*Iout_p[kb+i]:0.0))/(chih>0?chih:1.0);
                    double dtau=chih*ds, ex=exp(-dtau), e0,e1,wu,wd;
                    if(dtau>1e-4){e0=1-ex;e1=dtau-e0;wu=e0-e1/dtau;wd=e1/dtau;}else{wu=0.5*dtau;wd=0.5*dtau;}
                    I = I*ex + wu*Su + wd*Su; Iout_c[kb+i]=I;
                    }
                    IpL[(size_t)s*NP+cloc[i]]=I;
                }
            }
            for (int s = 0; s < NS; ++s) {
                int c = cnt[s]; size_t idx=(size_t)s*NB+b;
                if (c < 1) { Jnew[idx] = S[idx]; continue; }
                for (int a=1;a<c;++a){double mk=muL[(size_t)s*NP+a],ip=IpL[(size_t)s*NP+a],im=ImL[(size_t)s*NP+a];int q=a-1;
                    while(q>=0&&muL[(size_t)s*NP+q]>mk){muL[(size_t)s*NP+q+1]=muL[(size_t)s*NP+q];IpL[(size_t)s*NP+q+1]=IpL[(size_t)s*NP+q];ImL[(size_t)s*NP+q+1]=ImL[(size_t)s*NP+q];--q;}
                    muL[(size_t)s*NP+q+1]=mk;IpL[(size_t)s*NP+q+1]=ip;ImL[(size_t)s*NP+q+1]=im;}
                double mu[300],jv[300];int q=0; mu[q]=0;jv[q]=0.5*(IpL[(size_t)s*NP+0]+ImL[(size_t)s*NP+0]);q++;
                for(int a=0;a<c;++a){mu[q]=muL[(size_t)s*NP+a];jv[q]=0.5*(IpL[(size_t)s*NP+a]+ImL[(size_t)s*NP+a]);q++;}
                double Js=0; for(int a=0;a+1<q;++a)Js+=0.5*(jv[a]+jv[a+1])*(mu[a+1]-mu[a]); Jnew[idx]=Js;
            }
            { double *t1=Iin_p;Iin_p=Iin_c;Iin_c=t1; double *t2=Iout_p;Iout_p=Iout_c;Iout_c=t2; }
        }
        double maxrel = 0.0;
        for (size_t i = 0; i < (size_t)NS*NB; ++i) {
            double d = fabs(Jnew[i]-cs->J[i])/(fabs(cs->J[i])+1e-30); if (d>maxrel) maxrel=d;
            cs->J[i] = Jnew[i];
        }
        if (maxrel < 1e-4 && it > 0) { cpu_iters = it + 1; break; }
    }

    /* --- self-check (use_gpu==2): GPU already completed successfully before
     * the CPU solve. Compare the two results; keep CPU authoritative. */
    if (use_gpu == 2) {
        double maxrel=0.0, l2n=0.0, l2d=0.0; size_t worst=0;
        for (size_t i=0;i<(size_t)NS*NB;++i){ double dn=fabs(Jcpu[i]-cs->J[i]);
            double d=dn/(fabs(cs->J[i])+1e-30); if(d>maxrel){maxrel=d;worst=i;}
            l2n+=dn*dn; l2d+=cs->J[i]*cs->J[i]; }
        fprintf(stderr,
            "[cmf_gpu] SELF-CHECK NS=%d NB=%d NP=%d adv_split=%d: max rel diff(J_gpu vs J_cpu)=%.3e "
            "L2 rel=%.3e  (worst s=%d b=%d: cpu=%.4e gpu=%.4e)  ALI iters cpu=%d gpu=%d\n",
            NS, NB, NP, adv_split, maxrel, (l2d>0)?sqrt(l2n/l2d):0.0,
            (int)(worst/NB),(int)(worst%NB), cs->J[worst], Jcpu[worst], cpu_iters, gpu_iters);
    }
cmf_solve_cleanup:
    (void)cpu_iters;
    free(Bin_h);free(adv_h);free(advc_h);free(shell_off);free(shell_k);free(shell_seg);free(shell_mu);
    free(Jin);free(Jcpu);
    free(rmid);free(lam);free(p);free(rn);free(rsh);free(rz);free(rcore);free(rzin);
    free(Iin_p);free(Iout_p);free(Iin_c);free(Iout_c);free(muL);free(IpL);free(ImL);free(cnt);free(S);free(Jnew);
    return solve_rc;
}

/* ===================================================================
 * STAGE-1 PROOF (LUMINA_CMF_FINE_EMERGENT=1): frequency-resolved emergent
 * spectrum on the fine mesh. Clones the static (no-Doppler) per-frequency
 * formal integral of cmfgen_write_spectrum but reads the FINE field
 * (fs nu, chi, S_fixed, J) so line windows stay OPEN and the warm photosphere
 * color is transported outward (binned-J collapses to grey-thermal; fine field
 * carries J/B>1 at cold shells -- run 169643). Reuses the binned ray grid
 * (csb->p_ray/n_rays); geometry is frequency-independent. Valid only over the
 * producer window; writes lumina_spectrum_freqres.csv. Color (SED peak) is the
 * gate vs gold 6630A -- the no-Doppler approximation drops P-Cygni profiles but
 * preserves the source-function color, which is what we test. */
static int cmfgen_fine_emergent(const CMFGENState *fs, const CMFGENState *csb,
                                const Geometry *geo, double T_inner,
                                const char *path)
{
    int NS = fs->n_shells, NF = fs->n_bins, NR = csb->n_rays;
    if (NR <= 0 || !csb->p_ray) {
        fprintf(stderr, "[cmf_fine] emergent: no binned ray grid\n"); return -1;
    }
    int    *ray_n    = malloc(sizeof(int) * NR);
    int    *ray_core = malloc(sizeof(int) * NR);
    int    *seg_sh   = malloc(sizeof(int) * (size_t)NR * NS);
    double *seg_ds   = malloc(sizeof(double) * (size_t)NR * NS);
    double *S        = malloc(sizeof(double) * NS);
    double *Lnu      = calloc(NF, sizeof(double));
    if (!ray_n||!ray_core||!seg_sh||!seg_ds||!S||!Lnu) {
        free(ray_n);free(ray_core);free(seg_sh);free(seg_ds);free(S);free(Lnu);
        fprintf(stderr,"[cmf_fine] emergent alloc failed\n"); return -1;
    }
    /* per-ray intersected shells + segment lengths (same as write_spectrum) */
    for (int ray = 0; ray < NR; ++ray) {
        double p = csb->p_ray[ray];
        int *sh = &seg_sh[(size_t)ray * NS];
        double *zz = malloc(sizeof(double) * (NS + 1));
        int nshell = 0;
        for (int s = NS - 1; s >= 0; --s) {
            double ro = geo->r_outer[s];
            if (ro <= p) break;
            double rmid = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
            if (rmid <= p) rmid = p * 1.0000001;
            sh[nshell] = s;
            zz[nshell] = sqrt(rmid*rmid - p*p);
            ++nshell;
        }
        ray_n[ray]    = nshell;
        ray_core[ray] = (p < geo->r_inner[0]) ? 1 : 0;
        double z_core = 0.0;
        if (ray_core[ray] && nshell > 0) {
            double ri0 = geo->r_inner[0];
            z_core = sqrt(ri0*ri0 - p*p);
            if (z_core > zz[nshell-1]) z_core = zz[nshell-1];
        }
        double *ds = &seg_ds[(size_t)ray * NS];
        for (int i = 0; i < nshell; ++i)
            ds[i] = (i+1 < nshell) ? fabs(zz[i]-zz[i+1])
                                   : (ray_core[ray] ? zz[i]-z_core : fabs(zz[i]));
        free(zz);
    }
    /* per-fine-frequency emergent flux (static formal integral) */
    for (int b = 0; b < NF; ++b) {
        double Bin = cm_planck(fs->nu[b], T_inner);
        for (int s = 0; s < NS; ++s) {
            size_t idx = (size_t)s * NF + b;
            double r = (fs->chi_tot[idx] > 0.0) ? fs->chi_es[idx]/fs->chi_tot[idx] : 0.0;
            S[s] = fs->S_fixed[idx] + r * fs->J[idx];   /* converged fine source */
        }
        double integ = 0.0, p_prev = 0.0, f_prev = 0.0;
        for (int ray = 0; ray < NR; ++ray) {
            int n = ray_n[ray];
            if (n == 0) continue;
            const int *sh = &seg_sh[(size_t)ray * NS];
            const double *ds = &seg_ds[(size_t)ray * NS];
            double I = 0.0;
            for (int i = 0; i < n; ++i) {            /* inbound */
                double dtau = fs->chi_tot[(size_t)sh[i]*NF + b] * ds[i];
                if (dtau < 0.0) dtau = 0.0;
                double ex = exp(-dtau);
                double psi = (dtau > 1e-4) ? (1.0-ex) : (dtau - 0.5*dtau*dtau);
                I = I*ex + S[sh[i]]*psi;
            }
            if (ray_core[ray]) I = Bin;              /* diffusive core */
            for (int i = n-1; i >= 0; --i) {         /* outbound */
                double dtau = fs->chi_tot[(size_t)sh[i]*NF + b] * ds[i];
                if (dtau < 0.0) dtau = 0.0;
                double ex = exp(-dtau);
                double psi = (dtau > 1e-4) ? (1.0-ex) : (dtau - 0.5*dtau*dtau);
                I = I*ex + S[sh[i]]*psi;
            }
            double p = csb->p_ray[ray];
            double f = I * p;
            integ += 0.5 * (f_prev + f) * (p - p_prev);
            p_prev = p; f_prev = f;
        }
        Lnu[b] = 8.0 * M_PI_VAL * M_PI_VAL * integ;
    }
    FILE *fp = fopen(path, "w");
    if (!fp) { free(ray_n);free(ray_core);free(seg_sh);free(seg_ds);free(S);free(Lnu);
        fprintf(stderr,"[cmf_fine] cannot open %s\n",path); return -1; }
    fprintf(fp, "wavelength_angstrom,flux\n");
    for (int b = NF-1; b >= 0; --b) {                /* nu desc -> lambda asc */
        double lam_cm = CM_C / fs->nu[b];
        double lam_A  = lam_cm * 1.0e8;
        double L_lam  = Lnu[b] * CM_C / (lam_cm*lam_cm) * 1.0e-8;
        fprintf(fp, "%.6f,%.6e\n", lam_A, L_lam);
    }
    fclose(fp);
    fprintf(stderr, "[cmf_fine] freq-resolved emergent -> %s (%d fine freqs)\n", path, NF);
    free(ray_n);free(ray_core);free(seg_sh);free(seg_ds);free(S);free(Lnu);
    return 0;
}

/* ===================================================================
 * UNIFIED emergent (LUMINA_CMF_FINE_EMERGENT_OBS=1): the deterministic
 * production spectrum that combines all three physics on one integrator:
 *   - CONTINUUM color    : the FINE field fs (chi_es/chi_abs/J), interpolated at
 *                          the local comoving frequency (cures binned-J grey).
 *   - P-Cygni profiles   : observer-frame Doppler march (q = gamma(1-mu beta))
 *                          with full-tau_Sobolev line resonances (cmf_obs_march_sob).
 *   - FLUORESCENCE source: line resonances emit the NLTE-solved line_source_S
 *                          (g_sob_sl_ptr), not the W*B scattering backlight.
 * Output is a moderate observer grid (LUMINA_CMF_FINE_OBS_NOBS, default 3000)
 * over the fine window; continuum is interpolated on the fine grid so the grid
 * need not be fine. Reuses cmf_obs_march_sob (fs passed as the field state).
 * beta->0 / no lines reduces to the static cmfgen_fine_emergent color. */
static int cmfgen_fine_emergent_obs(const CMFGENState *fs, const Geometry *geo,
                                    double T_inner, const OpacityState *opac,
                                    const double *Te, const char *path)
{
    int NS = fs->n_shells, NF = fs->n_bins;
    double inv_ct = 1.0 / (CM_C * geo->time_explosion);
    if (!opac || !opac->line_list_nu || !opac->tau_sobolev || !Te) {
        fprintf(stderr, "[cmf_obs] missing line data for unified emergent\n");
        return -1;
    }
    /* line source: NLTE line_source_S (fluorescence) by default; or the W*B
     * diluted-photosphere SCATTERING source (LUMINA_CMF_FINE_OBS_SCATTER=1),
     * which is what produces classical P-Cygni (resonance scatter of the hot
     * backlight). Thermal NLTE S_l=B gives NO P-Cygni (absorption=emission
     * cancel), so the scatter source is the right vehicle until fluorescence
     * lifts S_l above Boltzmann. */
    int obs_scatter = 0;
    { const char *e = getenv("LUMINA_CMF_FINE_OBS_SCATTER"); if (e) obs_scatter = atoi(e); }
    g_sob_sl_ptr   = obs_scatter ? NULL : opac->line_source_S;
    g_sob_sl_clamp = 0.0;
    { const char *e = getenv("LUMINA_CMF_FINE_SL_CLAMP"); if (e) g_sob_sl_clamp = atof(e); }
    { const char *e = getenv("LUMINA_CMF_OBS_TAUMIN");    g_sob_taumin = e ? atof(e) : 1e-6; }
    g_sob_eps = 0.0; g_sob_noemit = 0; g_sob_srcj = 0; g_sob_contonly = 0;
    /* F1 fine-tune: line scatter/absorption blend S_l=(1-eps)*W*B + eps*B(Te).
     * eps=0 = pure scatter (fills troughs, right color); eps>0 adds the cold
     * local-thermal absorbing fraction that DEEPENS the P-Cygni troughs
     * (Ca II H&K / Fe II / O I) toward gold without the full-thermal reddening. */
    { const char *e = getenv("LUMINA_CMF_LINE_SOB_EPS");
      g_sob_eps = e ? atof(e) : 0.0;
      if (g_sob_eps < 0.0) g_sob_eps = 0.0; if (g_sob_eps > 1.0) g_sob_eps = 1.0; }
    { const char *e = getenv("LUMINA_CMF_OBS_CONTONLY"); g_sob_contonly = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_JBARDET"); g_sob_jbardet = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_FAITHFUL"); g_sob_faithful = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_SEI"); g_sob_sei = e ? atoi(e) : 0; }
    { const char *e = getenv("LUMINA_CMF_OBS_SEI_CMV"); g_sob_sei_cmv = e ? atoi(e) : 1; }
    { const char *e = getenv("LUMINA_CMF_OBS_SRCJ"); g_sob_srcj = e ? atoi(e) : 0; }
    g_sob_rphot = geo->r_inner[0]; g_sob_Tinner = T_inner;

    int NObs = 3000; { const char *e = getenv("LUMINA_CMF_FINE_OBS_NOBS"); if (e) NObs = atoi(e); }
    if (NObs < 2) NObs = 2;
    int NRO = 256; { const char *e = getenv("LUMINA_CMF_OBS_NRAY"); if (e) NRO = atoi(e); }
    double r_max = geo->r_outer[NS - 1], r_in0 = geo->r_inner[0];
    double *p_obs = malloc(sizeof(double) * NRO);
    double *nuo   = malloc(sizeof(double) * NObs);
    double *Lnu   = calloc(NObs, sizeof(double));
    if (!p_obs || !nuo || !Lnu) { free(p_obs); free(nuo); free(Lnu);
        fprintf(stderr, "[cmf_obs] alloc failed\n"); return -1; }
    for (int kk = 0; kk < NRO; ++kk) p_obs[kk] = r_max * (kk + 0.5) / (double)NRO;
    /* observer output grid: log-uniform over the fine window [nu_lo,nu_hi] */
    double nu_lo = fs->nu[0], nu_hi = fs->nu[NF - 1];
    double dln = log(nu_hi / nu_lo) / (double)(NObs - 1);
    for (int k = 0; k < NObs; ++k) nuo[k] = nu_lo * exp(dln * k);

    /* SEI Pass 1: continuum-only march -> beamed continuum mean Jbar_C[k,s]
     * (ray-ensemble angle average, carries the D⁴ beaming). Pass 2 (main loop)
     * sources the true-tau_S line jumps from it -> conserving + sharp P-Cygni. */
    double *Jbar_C = NULL;
    if (g_sob_sei) {
        Jbar_C = calloc((size_t)NObs * NS, sizeof(double));
        int *cnt = calloc((size_t)NObs * NS, sizeof(int));
        int save_co = g_sob_contonly; g_sob_contonly = 1;
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < NObs; ++k) {
            double nu_obs = nuo[k];
            for (int ray = 0; ray < NRO; ++ray) {
                double p = p_obs[ray]; int sh[256], nshell = 0;
                for (int s = NS-1; s >= 0 && nshell < 256; --s) { if (geo->r_outer[s] <= p) break; sh[nshell++] = s; }
                if (nshell == 0) continue;
                int core = (p < r_in0) ? 1 : 0; double I = 0.0;
                /* CMV (default): accumulate the COMOVING intensity I_cmf = I_obs*q^3
                 * so jc[s] is the comoving mean J_bar; Pass-2 then beams it ONCE via
                 * D^3 (the old code stored observer-frame I -> D^3 double-beamed it,
                 * the +180% SEI bug). LUMINA_CMF_OBS_SEI_CMV=0 reverts. */
                for (int i = 0; i < nshell; ++i) {
                    double ro=geo->r_outer[sh[i]],ri=geo->r_inner[sh[i]];
                    double zh=(ro>p)?sqrt(ro*ro-p*p):0.0, zl=(ri>p)?sqrt(ri*ri-p*p):0.0;
                    I = cmf_obs_march_sob(I,p,-zh,-zl,sh[i],fs,opac,Te[sh[i]],nu_obs,inv_ct);
                    double w=1.0; if (g_sob_sei_cmv) { double zm=-0.5*(zh+zl),rm=sqrt(p*p+zm*zm),
                        mu=(rm>0)?zm/rm:0.0,be=rm*inv_ct,ga=1.0/sqrt(1.0-be*be),q=ga*(1.0-mu*be); w=q*q*q; }
                    Jbar_C[(size_t)k*NS+sh[i]] += I*w; cnt[(size_t)k*NS+sh[i]]++;
                }
                if (core) { double zc=sqrt(r_in0*r_in0-p*p),mc=zc/r_in0,bi=r_in0*inv_ct,gi=1.0/sqrt(1.0-bi*bi);
                            double qc=gi*(1.0-mc*bi),Dc=1.0/qc; I=(Dc*Dc*Dc)*cm_planck(qc*nu_obs,T_inner); }
                for (int i = nshell-1; i >= 0; --i) {
                    double ro=geo->r_outer[sh[i]],ri=geo->r_inner[sh[i]];
                    double zh=(ro>p)?sqrt(ro*ro-p*p):0.0, zl=(ri>p)?sqrt(ri*ri-p*p):0.0;
                    I = cmf_obs_march_sob(I,p,+zl,+zh,sh[i],fs,opac,Te[sh[i]],nu_obs,inv_ct);
                    double w=1.0; if (g_sob_sei_cmv) { double zm=0.5*(zh+zl),rm=sqrt(p*p+zm*zm),
                        mu=(rm>0)?zm/rm:0.0,be=rm*inv_ct,ga=1.0/sqrt(1.0-be*be),q=ga*(1.0-mu*be); w=q*q*q; }
                    Jbar_C[(size_t)k*NS+sh[i]] += I*w; cnt[(size_t)k*NS+sh[i]]++;
                }
            }
        }
        g_sob_contonly = save_co;
        for (size_t i = 0; i < (size_t)NObs*NS; ++i) if (cnt[i] > 0) Jbar_C[i] /= cnt[i];
        free(cnt);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < NObs; ++k) {
        double nu_obs = nuo[k];
        const double *jc = (g_sob_sei && Jbar_C) ? &Jbar_C[(size_t)k * NS] : NULL;
        double integ = 0.0, p_prev = 0.0, f_prev = 0.0;
        for (int ray = 0; ray < NRO; ++ray) {
            double p = p_obs[ray];
            int sh[256], nshell = 0;
            for (int s = NS - 1; s >= 0 && nshell < 256; --s) {
                if (geo->r_outer[s] <= p) break;
                sh[nshell++] = s;
            }
            if (nshell == 0) continue;
            int core = (p < r_in0) ? 1 : 0;
            double I = 0.0;
            /* inbound (far side z<0): outer -> inner */
            for (int i = 0; i < nshell; ++i) {
                double ro = geo->r_outer[sh[i]], ri = geo->r_inner[sh[i]];
                double z_hi = (ro > p) ? sqrt(ro*ro - p*p) : 0.0;
                double z_lo = (ri > p) ? sqrt(ri*ri - p*p) : 0.0;
                I = cmf_obs_march_sob_jc(jc, I, p, -z_hi, -z_lo, sh[i], fs, opac,
                                      Te[sh[i]], nu_obs, inv_ct);
            }
            if (core) {                          /* diffusive core */
                double z_core = sqrt(r_in0*r_in0 - p*p);
                double mu_c = z_core / r_in0, beta_in = r_in0 * inv_ct;
                double gam_in = 1.0 / sqrt(1.0 - beta_in*beta_in);
                double q_c = gam_in * (1.0 - mu_c * beta_in), D_c = 1.0 / q_c;
                I = (D_c*D_c*D_c) * cm_planck(q_c * nu_obs, T_inner);
            }
            /* outbound (near side z>0): inner -> outer */
            for (int i = nshell - 1; i >= 0; --i) {
                double ro = geo->r_outer[sh[i]], ri = geo->r_inner[sh[i]];
                double z_hi = (ro > p) ? sqrt(ro*ro - p*p) : 0.0;
                double z_lo = (ri > p) ? sqrt(ri*ri - p*p) : 0.0;
                I = cmf_obs_march_sob_jc(jc, I, p, +z_lo, +z_hi, sh[i], fs, opac,
                                      Te[sh[i]], nu_obs, inv_ct);
            }
            double f = I * p;
            integ += 0.5 * (f_prev + f) * (p - p_prev);
            p_prev = p; f_prev = f;
        }
        Lnu[k] = 8.0 * M_PI_VAL * M_PI_VAL * integ;
    }
    g_sob_sl_ptr = NULL;   /* reset global */
    free(Jbar_C);

    FILE *fp = fopen(path, "w");
    if (!fp) { free(p_obs); free(nuo); free(Lnu);
        fprintf(stderr, "[cmf_obs] cannot open %s\n", path); return -1; }
    fprintf(fp, "wavelength_angstrom,flux\n");
    for (int k = NObs - 1; k >= 0; --k) {       /* nu desc -> lambda asc */
        double lam_cm = CM_C / nuo[k];
        double L_lam  = Lnu[k] * CM_C / (lam_cm*lam_cm) * 1.0e-8;
        fprintf(fp, "%.6f,%.6e\n", lam_cm * 1.0e8, L_lam);
    }
    fclose(fp);
    fprintf(stderr, "[cmf_obs] unified Doppler emergent -> %s (%d obs freqs, "
            "src=%s)\n", path, NObs, opac->line_source_S ? "NLTE-Sl" : "scatter");
    free(p_obs); free(nuo); free(Lnu);
    return 0;
}

/* ===================================================================
 * P7 PRODUCER: fine-grid line-resolved J_bar_l (gate II producer).
 *
 * Reuses the validated frequency-coupled formal kernel on a fine uniform-log
 * frequency mesh spanning a wavelength window (default 1000-4000 A, the
 * fluorescence pump region).  When the registered Q_E material context is
 * present, the initialization-reference operator deposits every line from the
 * CMFGEN direct-bracket material:
 *     Int chi_line dnu = tau_S * nu_l/(c t_exp)
 *     Int eta_line dnu = n_u A_ul h nu_l/(4 pi)
 * with the typed CMFGEN tau<-0.5 replacement.  R2 and later instead solve the
 * exact fine continuum only, then apply the same material line-by-line through
 * the CMFGEN non-overlap Sobolev EXPONX operator.  This preserves physical
 * mild negative tau without summing signed profiles into shared extinction.
 * The initialization-only shared operator remains explicit so R1 stays the
 * audited seed-predictor input; it is never selected by line sign.
 * The historical (1-exp(-tau_S)) expansion-opacity/source deposit remains only
 * for standalone legacy diagnostics lacking Q_E context.
 * Continuum (chi_es, chi_abs) is log-nu interpolated from the binned state.
 * Lines carry their lagged source S_l (line_source_S, fallback B(nu_l,Te))
 * in S_fixed; electron scattering is the ALI scattering channel (the outer
 * NLTE loop re-derives S_l from the J_bar this produces -- the lagged-J
 * scheme validated in gate 5d). After the solve, samples the continuum with
 * the registered profile and, in the production operator, computes
 *     J_bar_l = beta J_cont + (1-beta) S_l
 * into opac->jbar_line_det (private sentinel -1 outside the window; R6 maps
 * it to public UNSAMPLED). Standalone-validated: gates 3a/4b/4c/5*.
 *
 * Caller allocates opac->jbar_line_det[n_lines*n_shells]. */
/* bf + atom registered by cuda.cu so the producer can build the fine bf continuum
 * opacity (LUMINA_CMF_FINE_BF_OPAC). NULL → keep the interpolated continuum. */
static BFOpacity   *g_fine_bf   = NULL;
static AtomicData  *g_fine_atom = NULL;
static NLTEConfig  *g_fine_nlte = NULL;
static LuminaLineUpperPopulationFillFunction g_fine_upper_population_fill = NULL;
void cmfgen_fine_set_bf_atom(
        BFOpacity *bf, AtomicData *atom, NLTEConfig *nlte,
        LuminaLineUpperPopulationFillFunction upper_population_fill) {
    g_fine_bf = bf; g_fine_atom = atom; g_fine_nlte = nlte;
    g_fine_upper_population_fill = upper_population_fill;
}

static int cmf_fine_global_level(const AtomicData *atom, int line, int upper) {
    if (!atom || line < 0 || line >= atom->n_lines || !atom->level_Z ||
        !atom->level_ion || !atom->level_num || !atom->line_atomic_number ||
        !atom->line_ion_number || !atom->line_level_lower ||
        !atom->line_level_upper)
        return -1;
    int Z = atom->line_atomic_number[line];
    int ion = atom->line_ion_number[line];
    int level = upper ? atom->line_level_upper[line]
                      : atom->line_level_lower[line];
    for (int g = 0; g < atom->n_levels; ++g)
        if (atom->level_Z[g] == Z && atom->level_ion[g] == ion &&
            atom->level_num[g] == level)
            return g;
    return -1;
}

static int cmf_fine_line_material(
        const OpacityState *opac,const PlasmaState *plasma,
        int line,int shell,int n_shells,double nu,double tau,double dnu_d,
        double time_explosion,double legacy_chi0_pref,int parity,
        int legacy_source_nlte,double line_eps,double sl_clamp,
        const double *upper_population_cache,
        double *chi0,double *eta0,double *source_diag,
        int *srce_chk_applied,int *clamped){
    if(!opac||!plasma||!chi0||!eta0||!source_diag||
       !srce_chk_applied||!clamped||line<0||shell<0||
       shell>=n_shells||!(dnu_d>0.0))return-1;
    *chi0=0.0;*eta0=0.0;*source_diag=0.0;
    *srce_chk_applied=0;*clamped=0;
    double B=cm_planck(nu,plasma->T_e[shell]);
    if(parity){
        size_t cell=(size_t)line*(size_t)n_shells+(size_t)shell;
        if(!opac->tau_validity||
           (opac->tau_validity[cell]!=A208_VALID&&
            opac->tau_validity[cell]!=A208_EXACT_ZERO)||
           !g_fine_atom||!g_fine_nlte||!upper_population_cache)
            return-1;
        double nupper=upper_population_cache[cell];
        LineNetSobolevMaterial material;
        double Aul=g_fine_atom->line_A_ul?
            g_fine_atom->line_A_ul[line]:NAN;
        if(!isfinite(nupper)||nupper<0.0||line_net_sobolev_material(
               nupper,Aul,nu,tau,time_explosion,
               LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK,1,
               &material)!=0)return-1;
        *chi0=material.effective_integrated_opacity/
              (1.7724538509055160*dnu_d);
        *eta0=material.emission_per_sr/(1.7724538509055160*dnu_d);
        if (*chi0 != 0.0) {
            double ratio = *eta0 / *chi0;
            *source_diag = isfinite(ratio) ? ratio : 0.0;
        }
        *srce_chk_applied=material.srce_chk_applied;
        return 0;
    }
    if(tau<=1.0e-12)return 1;
    double fraction=tau>1.0e-6?-expm1(-tau):tau;
    *chi0=fraction*legacy_chi0_pref;
    double source=legacy_source_nlte&&opac->line_source_S?
        opac->line_source_S[(size_t)line*(size_t)n_shells+(size_t)shell]:0.0;
    if(source<=0.0)source=B;
    if(sl_clamp>0.0&&B>0.0&&source>sl_clamp*B){
        source=sl_clamp*B;*clamped=1;
    }
    double emit=line_eps<1.0?line_eps*B:source;
    *eta0=*chi0*emit;*source_diag=emit;
    return 0;
}

typedef enum {
    CMF_FINE_OWNER_CONFIG_OK = 0,
    CMF_FINE_OWNER_INVALID_DEVICE_REQUEST = 1,
    CMF_FINE_OWNER_CUDA_NOT_LINKED = 2
} CMFFineOwnerConfigStatus;

typedef struct {
    CMFFineOwnerConfigStatus config_status;
    int multigpu_requested;
    int multigpu_status;
    int devices_used;
    int visible_devices;
    int epoch_block_size;
    int epoch_batch_cardinality;
    int epoch_direct_replay_max_window;
    int weighted_partition;
    size_t max_device_allocated_bytes;
    size_t total_device_allocated_bytes;
    int device_partition_count;
    int device_ray_begin[CMF_MGPU_REPORT_MAX_DEVICES];
    int device_ray_end[CMF_MGPU_REPORT_MAX_DEVICES];
    size_t device_owned_segment_work[CMF_MGPU_REPORT_MAX_DEVICES];
    size_t device_computed_segment_work[CMF_MGPU_REPORT_MAX_DEVICES];
    size_t device_allocated_bytes[CMF_MGPU_REPORT_MAX_DEVICES];
    double initialization_seconds;
    double source_assembly_seconds;
    double host_to_device_seconds;
    double device_sweep_seconds;
    double device_to_host_seconds;
    double host_reduction_seconds;
    double convergence_check_seconds;
    double fixed_point_solve_seconds;
    double envelope_context_setup_seconds;
    double bounds_seconds;
    double envelope_residual_seconds;
    double envelope_verify_seconds;
    double envelope_refine_seconds;
    double publication_seconds;
    double cleanup_seconds;
    double total_seconds;
    int failure_phase;
    int failure_iteration;
    int failure_device_index;
    int failure_ray_begin;
    int failure_ray_end;
    int failure_segment_index;
    int failure_bin_index;
    double failure_value;
    CMFExactReport exact;
} CMFFineOwnerResult;

/* Sole serial/multi-GPU dispatch used by the production fine-grid owner and
 * its pre-model smoke.  No non-OK multi-GPU attempt falls back to the CPU. */
static int cmf_fine_exact_owner_solve(
    int n_shells, int n_bins, double dlognu, const double *nu,
    const double *r_inner, const double *r_outer, double time_explosion,
    double T_inner, double inner_boundary_scale,
    const double *chi_tot, const double *chi_es,
    const double *S_fixed, double *J, double *error_upper,
    size_t envelope_refinements, int iteration_cap, double tolerance,
    CMFFineOwnerResult *result)
{
    if (!result) return -1;
    memset(result, 0, sizeof(*result));
    result->exact.status = CMF_EXACT_INVALID_INPUT;
    result->exact.final_max_relative_change = INFINITY;
    result->exact.final_max_absolute_change = INFINITY;
    result->exact.max_scattering_ratio = INFINITY;
    result->exact.fixed_point_absolute_error_bound = INFINITY;
    result->exact.componentwise_residual_upper_max = INFINITY;
    result->exact.componentwise_error_upper_min = INFINITY;
    result->exact.componentwise_error_upper_max = INFINITY;
    result->failure_iteration = -1;
    result->failure_device_index = -1;
    result->failure_ray_begin = -1;
    result->failure_ray_end = -1;
    result->failure_segment_index = -1;
    result->failure_bin_index = -1;
    result->failure_value = NAN;
    int requested_devices = 0;
    if (cmf_fine_multigpu_device_request(&requested_devices) != 0) {
        result->config_status = CMF_FINE_OWNER_INVALID_DEVICE_REQUEST;
        return -1;
    }
    result->multigpu_requested = requested_devices;
    if (requested_devices == 0) {
        CMFExactStatus status = cmf_exact_characteristic_solve_with_envelope(
            n_shells, n_bins, dlognu, nu, r_inner, r_outer,
            time_explosion, T_inner, inner_boundary_scale,
            chi_tot, chi_es, S_fixed, J, error_upper,
            envelope_refinements, iteration_cap, tolerance,
            CMF_EXACT_MODE_POSITIVE_SLIDING, &result->exact);
        return status == CMF_EXACT_OK ? 0 : -1;
    }
#ifdef LUMINA_HAS_CUDA_BF_GEMM
    const CMFMultiGPUEpochSchedule schedule = {128, 64, 32};
    CMFMultiGPUReport report;
    CMFMultiGPUStatus status =
        cmf_exact_multigpu_positive_solve_envelope_epoch_partitioned(
            n_shells, n_bins, dlognu, nu, r_inner, r_outer,
            time_explosion, T_inner, inner_boundary_scale,
            chi_tot, chi_es, S_fixed, J, error_upper,
            envelope_refinements,
            CMF_MGPU_PARTITION_WEIGHTED_SEGMENTS, &schedule,
            requested_devices, iteration_cap, tolerance, &report);
    result->multigpu_status = (int)status;
    result->devices_used = report.devices_used;
    result->visible_devices = report.visible_devices;
    result->epoch_block_size = report.epoch_block_size;
    result->epoch_batch_cardinality = report.epoch_batch_cardinality;
    result->epoch_direct_replay_max_window =
        report.epoch_direct_replay_max_window;
    result->weighted_partition = report.weighted_contiguous_ray_partition;
    result->max_device_allocated_bytes = report.max_device_allocated_bytes;
    result->total_device_allocated_bytes = report.total_device_allocated_bytes;
    result->device_partition_count = report.device_partition_count;
    for (int device = 0; device < report.device_partition_count; ++device) {
        result->device_ray_begin[device] = report.device_ray_begin[device];
        result->device_ray_end[device] = report.device_ray_end[device];
        result->device_owned_segment_work[device] =
            report.device_owned_segment_work[device];
        result->device_computed_segment_work[device] =
            report.device_computed_segment_work[device];
        result->device_allocated_bytes[device] =
            report.device_allocated_bytes[device];
    }
    result->initialization_seconds = report.initialization_seconds;
    result->source_assembly_seconds = report.source_assembly_seconds;
    result->host_to_device_seconds = report.host_to_device_seconds;
    result->device_sweep_seconds = report.device_sweep_seconds;
    result->device_to_host_seconds = report.device_to_host_seconds;
    result->host_reduction_seconds = report.host_reduction_seconds;
    result->convergence_check_seconds = report.convergence_check_seconds;
    result->fixed_point_solve_seconds = report.fixed_point_solve_seconds;
    result->envelope_context_setup_seconds =
        report.envelope_context_setup_seconds;
    result->bounds_seconds = report.bounds_seconds;
    result->envelope_residual_seconds = report.envelope_residual_seconds;
    result->envelope_verify_seconds = report.envelope_verify_seconds;
    result->envelope_refine_seconds = report.envelope_refine_seconds;
    result->publication_seconds = report.publication_seconds;
    result->cleanup_seconds = report.cleanup_seconds;
    result->total_seconds = report.total_seconds;
    result->failure_phase = report.failure_phase;
    result->failure_iteration = report.failure_iteration;
    result->failure_device_index = report.failure_device_index;
    result->failure_ray_begin = report.failure_ray_begin;
    result->failure_ray_end = report.failure_ray_end;
    result->failure_segment_index = report.failure_segment_index;
    result->failure_bin_index = report.failure_bin_index;
    result->failure_value = report.failure_nearest;
    result->exact.status = status == CMF_MGPU_OK ?
        CMF_EXACT_OK : CMF_EXACT_NONFINITE;
    result->exact.mode = CMF_EXACT_MODE_POSITIVE_SLIDING;
    result->exact.iterations_used = report.iterations_used;
    result->exact.iteration_cap = report.iteration_cap;
    result->exact.tolerance = report.tolerance;
    result->exact.final_max_relative_change =
        report.final_max_relative_change;
    result->exact.final_max_absolute_change =
        report.final_max_absolute_change;
    result->exact.max_scattering_ratio = report.max_scattering_ratio;
    result->exact.fixed_point_absolute_error_bound =
        report.fixed_point_absolute_error_bound;
    result->exact.max_characteristic_drift_bins =
        report.max_characteristic_drift_bins;
    result->exact.n_rays = report.n_rays;
    result->exact.componentwise_error_envelope_verified =
        report.componentwise_error_envelope_verified;
    result->exact.componentwise_error_seed_attempts =
        report.componentwise_error_seed_attempts;
    result->exact.componentwise_error_refinement_iterations =
        report.componentwise_error_refinement_iterations;
    result->exact.componentwise_residual_upper_max =
        report.componentwise_residual_upper_max;
    result->exact.componentwise_error_upper_min =
        report.componentwise_error_upper_min;
    result->exact.componentwise_error_upper_max =
        report.componentwise_error_upper_max;
    return status == CMF_MGPU_OK ? 0 : -1;
#else
    result->config_status = CMF_FINE_OWNER_CUDA_NOT_LINKED;
    return -1;
#endif
}

int cmf_exact_owner_selftest(void)
{
#ifndef LUMINA_HAS_CUDA_BF_GEMM
    fprintf(stderr, "CMF_EXACT_OWNER_SELFTEST_BLOCKED CUDA_NOT_LINKED\n");
    return 77;
#else
    enum { NS = 3, NB = 96, CELLS = NS * NB };
    const double dlognu = 1.0e-3;
    const double time_explosion = 1.0e6;
    double r_inner[NS] = {3.0e14, 4.0e14, 5.0e14};
    double r_outer[NS] = {4.0e14, 5.0e14, 6.0e14};
    double nu[NB], chi_tot[CELLS], chi_es[CELLS], fixed[CELLS];
    double initial[CELLS], cpu_J[CELLS], gpu_J[CELLS];
    double cpu_error[CELLS], gpu_error[CELLS];
    for (int b = 0; b < NB; ++b)
        nu[b] = 1.0e15 * exp(((double)b + 0.5) * dlognu);
    for (int s = 0; s < NS; ++s) {
        for (int b = 0; b < NB; ++b) {
            size_t cell = (size_t)s * NB + (size_t)b;
            double ripple = 1.0 + 0.35 * sin(0.17 * b + 0.4 * s);
            chi_tot[cell] = (2.0 + s) * 1.0e-15 * ripple;
            chi_es[cell] = 0.27 * chi_tot[cell];
            fixed[cell] = (1.0 + 0.1 * s) * 1.0e-7 *
                          (1.0 + 0.2 * cos(0.11 * b));
            initial[cell] = cpu_J[cell] = gpu_J[cell] = 0.8e-7;
            cpu_error[cell] = gpu_error[cell] = -1.0;
        }
    }
    CMFExactReport cpu_report;
    CMFExactStatus cpu_status =
        cmf_exact_characteristic_solve_with_envelope(
            NS, NB, dlognu, nu, r_inner, r_outer, time_explosion,
            10000.0, 1.0, chi_tot, chi_es, fixed, cpu_J, cpu_error,
            8U, 120, 1.0e-13, CMF_EXACT_MODE_POSITIVE_SLIDING,
            &cpu_report);
    CMFFineOwnerResult gpu_result;
    int gpu_rc = cmf_fine_exact_owner_solve(
        NS, NB, dlognu, nu, r_inner, r_outer, time_explosion,
        10000.0, 1.0, chi_tot, chi_es, fixed, gpu_J, gpu_error,
        8U, 120, 1.0e-13, &gpu_result);
    if (cpu_status != CMF_EXACT_OK || gpu_rc != 0 ||
        gpu_result.multigpu_requested <= 0 ||
        gpu_result.devices_used != gpu_result.multigpu_requested ||
        gpu_result.exact.status != CMF_EXACT_OK ||
        !gpu_result.exact.componentwise_error_envelope_verified ||
        gpu_result.exact.componentwise_error_refinement_iterations != 8U ||
        !(gpu_result.exact.max_scattering_ratio >= 0.0) ||
        !(gpu_result.exact.max_scattering_ratio < 1.0) ||
        !isfinite(gpu_result.exact.fixed_point_absolute_error_bound)) {
        fprintf(stderr,
                "CMF_EXACT_OWNER_SELFTEST_FAIL owner cpu_status=%s "
                "gpu_status=%s devices=%d/%d envelope=%d refinements=%zu\n",
                cmf_exact_status_name(cpu_status),
                cmf_multigpu_status_name(
                    (CMFMultiGPUStatus)gpu_result.multigpu_status),
                gpu_result.devices_used, gpu_result.multigpu_requested,
                gpu_result.exact.componentwise_error_envelope_verified,
                gpu_result.exact.componentwise_error_refinement_iterations);
        return 1;
    }
    double max_relative_J = 0.0, max_relative_error = 0.0;
    double max_enclosure_overlap_ratio = 0.0;
    double finite_min = INFINITY, finite_max = 0.0;
    for (size_t cell = 0; cell < (size_t)CELLS; ++cell) {
        if (!(gpu_J[cell] >= 0.0) || !isfinite(gpu_J[cell]) ||
            !(gpu_error[cell] >= 0.0) || !isfinite(gpu_error[cell])) {
            fprintf(stderr,
                    "CMF_EXACT_OWNER_SELFTEST_FAIL nonfinite cell=%zu "
                    "J=%.17g error=%.17g\n",
                    cell, gpu_J[cell], gpu_error[cell]);
            return 1;
        }
        double rel_J = fabs(gpu_J[cell] - cpu_J[cell]) /
                       (fabs(cpu_J[cell]) + 1.0e-300);
        double rel_error = fabs(gpu_error[cell] - cpu_error[cell]) /
                           (fabs(cpu_error[cell]) + 1.0e-300);
        long double distance = fabsl(
            (long double)gpu_J[cell] - (long double)cpu_J[cell]);
        long double combined_envelope =
            (long double)gpu_error[cell] + (long double)cpu_error[cell];
        if (!(distance <= combined_envelope)) {
            fprintf(stderr,
                    "CMF_EXACT_OWNER_SELFTEST_FAIL disjoint_envelope cell=%zu "
                    "distance=%.21Lg cpu_error=%.17g gpu_error=%.17g\n",
                    cell, distance, cpu_error[cell], gpu_error[cell]);
            return 1;
        }
        double overlap_ratio = combined_envelope > 0.0L ?
            (double)(distance / combined_envelope) : 0.0;
        if (overlap_ratio > max_enclosure_overlap_ratio)
            max_enclosure_overlap_ratio = overlap_ratio;
        if (rel_J > max_relative_J) max_relative_J = rel_J;
        if (rel_error > max_relative_error) max_relative_error = rel_error;
        if (gpu_J[cell] < finite_min) finite_min = gpu_J[cell];
        if (gpu_J[cell] > finite_max) finite_max = gpu_J[cell];
    }
    if (!(max_relative_J <= 1.0e-12)) {
        fprintf(stderr,
                "CMF_EXACT_OWNER_SELFTEST_FAIL cpu_gpu max_rel_J=%.17g\n",
                max_relative_J);
        return 1;
    }

    double bad_chi[CELLS], rejected_J[CELLS], rejected_error[CELLS];
    double rejected_J_before[CELLS], rejected_error_before[CELLS];
    memcpy(bad_chi, chi_tot, sizeof(bad_chi));
    memcpy(rejected_J, initial, sizeof(rejected_J));
    for (size_t cell = 0; cell < (size_t)CELLS; ++cell)
        rejected_error[cell] = 5.0 + (double)cell;
    memcpy(rejected_J_before, rejected_J, sizeof(rejected_J));
    memcpy(rejected_error_before, rejected_error, sizeof(rejected_error));
    bad_chi[3] = -1.0;
    CMFFineOwnerResult rejected_result;
    int rejected_rc = cmf_fine_exact_owner_solve(
        NS, NB, dlognu, nu, r_inner, r_outer, time_explosion,
        10000.0, 1.0, bad_chi, chi_es, fixed,
        rejected_J, rejected_error, 8U, 120, 1.0e-13,
        &rejected_result);
    if (rejected_rc == 0 ||
        rejected_result.multigpu_status != CMF_MGPU_NONFINITE ||
        memcmp(rejected_J, rejected_J_before, sizeof(rejected_J)) != 0 ||
        memcmp(rejected_error, rejected_error_before,
               sizeof(rejected_error)) != 0) {
        fprintf(stderr,
                "CMF_EXACT_OWNER_SELFTEST_FAIL transactional status=%s "
                "J_preserved=%d error_preserved=%d\n",
                cmf_multigpu_status_name(
                    (CMFMultiGPUStatus)rejected_result.multigpu_status),
                memcmp(rejected_J, rejected_J_before,
                       sizeof(rejected_J)) == 0,
                memcmp(rejected_error, rejected_error_before,
                       sizeof(rejected_error)) == 0);
        return 1;
    }

    const char *saved_request = getenv("LUMINA_CMF_FINE_MGPU_DEVICES");
    char saved_copy[64];
    if (!saved_request || strlen(saved_request) >= sizeof(saved_copy)) {
        fprintf(stderr,
                "CMF_EXACT_OWNER_SELFTEST_FAIL missing_device_request\n");
        return 1;
    }
    strcpy(saved_copy, saved_request);
    setenv("LUMINA_CMF_FINE_MGPU_DEVICES", "invalid", 1);
    memcpy(rejected_J, initial, sizeof(rejected_J));
    memcpy(rejected_J_before, rejected_J, sizeof(rejected_J));
    CMFFineOwnerResult invalid_result;
    int invalid_rc = cmf_fine_exact_owner_solve(
        NS, NB, dlognu, nu, r_inner, r_outer, time_explosion,
        10000.0, 1.0, chi_tot, chi_es, fixed,
        rejected_J, rejected_error, 8U, 120, 1.0e-13,
        &invalid_result);
    setenv("LUMINA_CMF_FINE_MGPU_DEVICES", saved_copy, 1);
    if (invalid_rc == 0 ||
        invalid_result.config_status !=
            CMF_FINE_OWNER_INVALID_DEVICE_REQUEST ||
        memcmp(rejected_J, rejected_J_before, sizeof(rejected_J)) != 0) {
        fprintf(stderr,
                "CMF_EXACT_OWNER_SELFTEST_FAIL invalid_config rc=%d "
                "config=%d J_preserved=%d\n",
                invalid_rc, (int)invalid_result.config_status,
                memcmp(rejected_J, rejected_J_before,
                       sizeof(rejected_J)) == 0);
        return 1;
    }
    printf("CMF_EXACT_OWNER_SELFTEST PASS devices=%d visible=%d "
           "finite_J=[%.17g,%.17g] max_rel_cpu_gpu_J=%.17g "
           "max_rel_cpu_gpu_error_width=%.17g "
           "max_enclosure_overlap_ratio=%.17g "
           "component_error=[%.17g,%.17g] "
           "transactional_negative=PASS invalid_config=PASS "
           "floor=0 clamp=0 jitter=0 repair=0\n",
           gpu_result.devices_used, gpu_result.visible_devices,
           finite_min, finite_max, max_relative_J, max_relative_error,
           max_enclosure_overlap_ratio,
           gpu_result.exact.componentwise_error_upper_min,
           gpu_result.exact.componentwise_error_upper_max);
    return 0;
#endif
}

int cmfgen_fine_jbar(CMFGENState *csb, const Geometry *geo,
                     OpacityState *opac, double T_inner, PlasmaState *plasma,
                     CMFFineLineOperator line_operator)
{
    int NS = csb->n_shells, NL = opac->n_lines;
    opac->jbar_line_det_vdoppler_cms = 0.0;
    opac->jbar_line_det_ndoppler = 0.0;
    opac->jbar_line_det_operator = 0;
    opac->jbar_line_det_exact_converged = 0;
    opac->jbar_line_det_exact_iterations = 0;
    opac->jbar_line_det_exact_iteration_cap = 0;
    opac->jbar_line_det_exact_residual = INFINITY;
    opac->jbar_line_det_exact_tolerance = 0.0;
    opac->jbar_line_det_exact_absolute_error_bound = INFINITY;
    opac->jbar_line_det_exact_max_scattering_ratio = INFINITY;
    opac->jbar_line_det_error_envelope_verified = 0;
    opac->jbar_line_det_error_refinement_iterations = 0;
    opac->jbar_line_det_component_error_min = INFINITY;
    opac->jbar_line_det_component_error_max = INFINITY;
    opac->jbar_line_det_profile_error_min = INFINITY;
    opac->jbar_line_det_profile_error_max = INFINITY;
    opac->jbar_line_det_grid_n_bins = 0;
    opac->jbar_line_det_grid_nu_min = 0.0;
    opac->jbar_line_det_grid_nu_max = 0.0;
    opac->jbar_line_det_continuum_captured = 0;
    free(opac->jbar_line_det_continuum);
    free(opac->jbar_line_det_continuum_error_upper);
    opac->jbar_line_det_continuum = NULL;
    opac->jbar_line_det_continuum_error_upper = NULL;
    if (NL <= 0 || !opac->jbar_line_det || !geo || !plasma ||
        (line_operator != CMF_FINE_LINE_OPERATOR_INIT_SHARED_GAUSSIAN &&
         line_operator !=
             CMF_FINE_LINE_OPERATOR_CMFGEN_NONOVERLAP_SOBOLEV))
        return -1;
    double t_exp = geo->time_explosion;
    int diag = 0; { const char *e=getenv("LUMINA_CMF_FINE_DIAG"); if(e) diag=atoi(e); }
    int independent_capture_requested = 0;
    if (cmf_optional_binary_env("LUMINA_A210_INDEPENDENT_CAPTURE",
                                &independent_capture_requested) != 0) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=INVALID_INDEPENDENT_CAPTURE_REQUEST "
                "value=%s expected=0_or_1\n",
                getenv("LUMINA_A210_INDEPENDENT_CAPTURE") ?
                    getenv("LUMINA_A210_INDEPENDENT_CAPTURE") : "(unset)");
        return -1;
    }

    /* --- window + resolution (env-tunable) --- */
    double lam_lo = LINE_JBAR_BB_LAMBDA_MIN_ANGSTROM;
    double lam_hi = LINE_JBAR_BB_LAMBDA_MAX_ANGSTROM; /* line-centre domain, A */
    { const char *e=getenv("LUMINA_CMF_FINE_LAMLO"); if(e) lam_lo=atof(e); }
    { const char *e=getenv("LUMINA_CMF_FINE_LAMHI"); if(e) lam_hi=atof(e); }
    double vdop = 1.0e6;                        /* cm/s, Doppler width  */
    { const char *e=getenv("LUMINA_CMF_FINE_VDOP"); if(e) vdop=atof(e); }
    double ppd = 12.0;                          /* fine points / vdop   */
    { const char *e=getenv("LUMINA_CMF_FINE_PPD"); if(e) ppd=atof(e); }
    int n_ali = 64; { const char *e=getenv("LUMINA_CMF_FINE_ALI"); if(e) n_ali=atoi(e); }
    double solve_tol = 1.0e-8;
    { const char *e=getenv("LUMINA_CMF_FINE_TOL"); if(e) solve_tol=atof(e); }
    size_t envelope_refinements = 8U;
    if (cmf_fine_envelope_refinement_request(&envelope_refinements) != 0) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=INVALID_ENVELOPE_REFINEMENTS "
                "value=%s allowed=1:64 physical_values_modified=0\n",
                getenv("LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS") ?
                    getenv("LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS") :
                    "(unset)");
        return -1;
    }
    /* Stage-2-pre stabilization: clamp lagged super-thermal S_l in the line
     * emissivity deposit. The cold-shell NLTE matrix is ill-conditioned ->
     * S_l/B can be huge (numerical artifact, rates are DB-correct); the closed
     * producer<->NLTE loop amplifies it (Jbar/B 454,659 at iter5,7 in 169651).
     * Clamp S_l <= sl_clamp*B(Te) (orthodox LTE@Te-floor analog). 0 = off. */
    double sl_clamp = 0.0; { const char *e=getenv("LUMINA_CMF_FINE_SL_CLAMP"); if(e) sl_clamp=atof(e); }
    /* Line-forest SCATTERING source (LUMINA_CMF_FINE_LINE_EPS = eps in (0,1]):
     * the ROOT CAUSE of the absent fluorescence (2026-06-26, triple-verified) is
     * that the producer emits the in-window forest as THERMAL emitters (eta =
     * chi_line*B(Te)), so the UV pump field thermalises to B(Te) in the green-
     * emitting shells (measured J_bar/B = 1.001) -> no super-thermal UV to pump.
     * With eps<1 the line source becomes S_l=(1-eps)*Jbar + eps*B: the thermal
     * fraction eps*chi_line emits B, the scattering remainder (1-eps)*chi_line
     * joins chi_es so the ALI carries the diluted-but-HOT photospheric UV field
     * outward (W*B(T_phot) > cold B(T_e)) -> super-thermal pump -> fluorescence.
     * eps=1.0 (default) = legacy pure-thermal (byte-identical). FROZEN-PLASMA
     * staged use: converge thermal first, then enable as a perturbation. */
    double line_eps = 1.0;
    { const char *e=getenv("LUMINA_CMF_FINE_LINE_EPS"); if(e) line_eps=atof(e); }

    if (!(lam_lo > 0.0) || !(lam_hi > lam_lo) || !(vdop > 0.0) ||
        !(ppd > 0.0) || n_ali < 2 || !(solve_tol > 0.0) ||
        !isfinite(lam_lo) || !isfinite(lam_hi) || !isfinite(vdop) ||
        !isfinite(ppd) || !isfinite(solve_tol) || !isfinite(sl_clamp) ||
        sl_clamp < 0.0 || !isfinite(line_eps) || line_eps < 0.0 ||
        line_eps > 1.0) {
        fprintf(stderr, "[cmf_fine][BLOCKED] reason=INVALID_SOLVE_CONFIG\n");
        return -1;
    }
    /* Only the final non-overlap Sobolev producer is consumed by the Stage-4
     * saturation rows.  The initialization producer remains byte-for-byte
     * free of the extra diagnostic solve. */
    int independent_capture = independent_capture_requested &&
        line_operator == CMF_FINE_LINE_OPERATOR_CMFGEN_NONOVERLAP_SOBOLEV;
    if (independent_capture && line_eps != 1.0) {
        /* The independent solve is intentionally line-free.  When eps<1 the
         * production assembly folds a line-scattering term into chi_es; using
         * subtraction here would manufacture a large-number cancellation and
         * would not be an independent physical continuum. */
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=INDEPENDENT_CAPTURE_REQUIRES_"
                "PURE_CONTINUUM_SCATTERING line_eps=%.17g "
                "physical_values_modified=0 floor=0 cap=0 clamp=0 jitter=0 repair=0\n",
                line_eps);
        return -1;
    }
    if (independent_capture) {
        const char *photoion = getenv("LUMINA_CMF_FINE_PHOTOION");
        if (photoion && strcmp(photoion, "0") != 0) {
            /* The fine-photoion lane transfers ownership of fs.J to the
             * coupled BF integrator after this routine.  Refuse the probe
             * rather than accidentally transferring the line-free field. */
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=INDEPENDENT_CAPTURE_"
                    "INCOMPATIBLE_FINE_PHOTOION value=%s physical_values_modified=0 "
                    "floor=0 cap=0 clamp=0 jitter=0 repair=0\n", photoion);
            return -1;
        }
    }
    double dlognu = (vdop / CM_C) / ppd;
    double profile_width = LINE_JBAR_PROFILE_NDOPPLER * vdop / CM_C;
    if (!(profile_width > 0.0) || !(profile_width < 1.0)) {
        fprintf(stderr, "[cmf_fine][BLOCKED] reason=INVALID_PROFILE_WIDTH value=%.17g\n",
                profile_width);
        return -1;
    }
    double line_nu_lo = CM_C / (lam_hi * 1.0e-8);
    double line_nu_hi = CM_C / (lam_lo * 1.0e-8);
    double support_nu_lo = line_nu_lo * (1.0 - profile_width);
    double support_nu_hi = line_nu_hi * (1.0 + profile_width);
    /* Redward bins cannot affect Q_g (the characteristic is blue->red), but
     * every Q profile wing must fit.  Blueward radiation is causal upstream:
     * use the canonical union-owner edge as the reservoir, then prove below
     * that it exceeds the maximum geometry-specific upstream requirement. */
    double reservoir_nu_hi = LUMINA_RADFIELD_NU_MAX_HZ;
    if (reservoir_nu_hi < support_nu_hi) reservoir_nu_hi = support_nu_hi;
    double max_upstream_log_shift = 0.0;
    {
        double rmid[256];
        if (NS > (int)(sizeof(rmid) / sizeof(rmid[0])) ||
            !(geo->time_explosion > 0.0)) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=INVALID_RESERVOIR_GEOMETRY "
                    "shells=%d t_exp=%.17g\n", NS, geo->time_explosion);
            return -1;
        }
        for (int s = 0; s < NS; ++s)
            rmid[s] = 0.5 * (geo->r_inner[s] + geo->r_outer[s]);
        for (int k = 0; k < NS + 16; ++k) {
            double impact = k < 16
                ? rmid[0] * (double)k / 16.0 : rmid[k - 16];
            double z_outer = 0.0;
            for (int s = NS - 1; s >= 0; --s) {
                if (rmid[s] <= impact) break;
                if (z_outer == 0.0)
                    z_outer = sqrt(rmid[s] * rmid[s] - impact * impact);
            }
            double z_inner = impact < rmid[0]
                ? sqrt(rmid[0] * rmid[0] - impact * impact) : 0.0;
            double shift = (z_outer - z_inner) /
                           (geo->time_explosion * CM_C);
            if (shift > max_upstream_log_shift) max_upstream_log_shift = shift;
        }
    }
    double required_upstream_nu = support_nu_hi * exp(max_upstream_log_shift);
    if (!isfinite(required_upstream_nu) || required_upstream_nu > reservoir_nu_hi) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=BLUE_RESERVOIR_COVERAGE "
                "support_nu_hi=%.17g max_log_shift=%.17g required=%.17g "
                "available=%.17g canonical_edge_hash=%s\n",
                support_nu_hi, max_upstream_log_shift, required_upstream_nu,
                reservoir_nu_hi, LUMINA_RADFIELD_EDGE_SHA256);
        return -1;
    }
    double nu_lo = support_nu_lo * exp(-0.5 * dlognu);
    double requested_nu_hi = reservoir_nu_hi * exp(0.5 * dlognu);
    double span = log(requested_nu_hi / nu_lo) / dlognu;
    if (!(span > 0.0) || !isfinite(span) || span > (double)INT32_MAX - 2.0) {
        fprintf(stderr, "[cmf_fine][BLOCKED] reason=INVALID_GRID_SPAN span=%.17g\n",
                span);
        return -1;
    }
    int NF = (int)ceil(span);
    if (NF < 2) return -1;
    double nu_hi = nu_lo * exp((double)NF * dlognu);

    /* default ALL lines to the sentinel (out-of-window fall back) */
    if ((size_t)NL > SIZE_MAX / (size_t)NS ||
        (size_t)NL * (size_t)NS > SIZE_MAX / sizeof(double)) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=LINE_ERROR_SIZE_OVERFLOW "
                "lines=%d shells=%d\n", NL, NS);
        return -1;
    }
    size_t line_cells = (size_t)NL * (size_t)NS;
    if (!opac->jbar_line_det_error_upper)
        opac->jbar_line_det_error_upper =
            (double *)malloc(line_cells * sizeof(double));
    if (!opac->jbar_line_det_error_upper) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=LINE_ERROR_ALLOCATION "
                "cells=%zu bytes=%zu\n",
                line_cells, line_cells * sizeof(double));
        return -1;
    }
    for (size_t i = 0; i < line_cells; ++i) {
        opac->jbar_line_det[i] = -1.0;
        opac->jbar_line_det_error_upper[i] = -1.0;
    }

    if (diag) fprintf(stderr,
        "[cmf_fine] BB centers %.0f-%.0f A support_red=%.6f A "
        "reservoir_blue=%.6f A vdop=%.1f km/s ppd=%.0f NF=%d (%.1fM cells)\n",
        lam_lo, lam_hi, CM_C/support_nu_lo*1.0e8,
        CM_C/reservoir_nu_hi*1.0e8, vdop/1e5, ppd, NF,
        (double)NF*NS/1e6);

    /* --- fine state: only the fields cmf_solve_J reads, plus chi_line/dnu --- */
    CMFGENState fs; memset(&fs, 0, sizeof fs);
    fs.n_shells = NS; fs.n_bins = NF;
    fs.nu_min = nu_lo; fs.nu_max = nu_hi; fs.d_log_nu = dlognu;
    fs.nu       = malloc((size_t)NF * sizeof(double));
    fs.dnu      = malloc((size_t)NF * sizeof(double));
    fs.chi_es   = calloc((size_t)NS * NF, sizeof(double));
    fs.chi_abs  = calloc((size_t)NS * NF, sizeof(double));
    fs.chi_line = calloc((size_t)NS * NF, sizeof(double));
    fs.chi_tot  = calloc((size_t)NS * NF, sizeof(double));
    fs.S_fixed  = calloc((size_t)NS * NF, sizeof(double));
    fs.J        = calloc((size_t)NS * NF, sizeof(double));
    double *eta = calloc((size_t)NS * NF, sizeof(double));   /* line emissivity */
    if (!fs.nu||!fs.dnu||!fs.chi_es||!fs.chi_abs||!fs.chi_line||!fs.chi_tot||
        !fs.S_fixed||!fs.J||!eta) {
        fprintf(stderr, "[cmf_fine] alloc failed (NF=%d NS=%d)\n", NF, NS);
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);free(fs.chi_line);
        free(fs.chi_tot);free(fs.S_fixed);free(fs.J);free(eta); return -1;
    }
    for (int i = 0; i < NF; ++i) {
        fs.nu[i]  = nu_lo * exp((i + 0.5) * dlognu);
        fs.dnu[i] = fs.nu[i] * dlognu;
    }

    /* --- continuum: log-nu interpolate chi_es/chi_abs from the binned state --- */
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < NS; ++s) {
        for (int i = 0; i < NF; ++i) {
            double x = log(fs.nu[i] / csb->nu_min) / csb->d_log_nu - 0.5;
            int b = (int)floor(x); double f = x - b;
            if (b < 0) { b = 0; f = 0.0; }
            if (b >= csb->n_bins - 1) { b = csb->n_bins - 2; f = 1.0; }
            size_t ia = (size_t)s*csb->n_bins + b, ib = ia + 1;
            fs.chi_es [(size_t)s*NF+i] = (1-f)*csb->chi_es [ia] + f*csb->chi_es [ib];
            fs.chi_abs[(size_t)s*NF+i] = (1-f)*csb->chi_abs[ia] + f*csb->chi_abs[ib];
        }
    }

    /* --- CMFGEN-method fix (LUMINA_CMF_FINE_BF_OPAC): replace the smeared interpolated
     * bf part of the continuum with the fine-ν bf opacity (sharp edges at the exact
     * thresholds), so the solved field develops the across-edge frequency structure the
     * binned/interpolated continuum averages away. chi_abs_fine = interp_chi_abs
     * − bf_get_chi(binned bf @ fine ν, the smeared part) + chi_bf_fine(sharp). ff kept. */
    {
        static int fbf = -1;
        if (fbf < 0) { const char *e = getenv("LUMINA_CMF_FINE_BF_OPAC"); fbf = (e && atoi(e)) ? 1 : 0; }
        if (fbf && g_fine_bf && g_fine_atom) {
            double *chi_bf_fine = (double *)malloc((size_t)NS * NF * sizeof(double));
            if (chi_bf_fine && bf_gemm_compute_fine(g_fine_bf, g_fine_atom, plasma, NS,
                    fs.nu, NF, g_fine_bf->nu_min, g_fine_bf->d_log_nu, chi_bf_fine) == 0) {
                double sum_old = 0.0, sum_new = 0.0;
                size_t first_invalid = SIZE_MAX;
                #pragma omp parallel for schedule(static) \
                    reduction(+:sum_old,sum_new) reduction(min:first_invalid)
                for (int s = 0; s < NS; ++s)
                    for (int i = 0; i < NF; ++i) {
                        size_t k = (size_t)s*NF + i;
                        double smeared_bf = bf_get_chi(g_fine_bf, s, fs.nu[i]);
                        double newabs = fs.chi_abs[k] - smeared_bf + chi_bf_fine[k];
                        if (!isfinite(newabs) || newabs < 0.0)
                            if (k < first_invalid) first_invalid = k;
                        sum_old += fs.chi_abs[k]; sum_new += newabs;
                        fs.chi_abs[k] = newabs;
                    }
                if (first_invalid != SIZE_MAX) {
                    int bad_shell = (int)(first_invalid / (size_t)NF);
                    int bad_bin = (int)(first_invalid % (size_t)NF);
                    double smeared = bf_get_chi(
                        g_fine_bf, bad_shell, fs.nu[bad_bin]);
                    double sharp = chi_bf_fine[first_invalid];
                    double assembled = fs.chi_abs[first_invalid];
                    double interpolated = assembled + smeared - sharp;
                    fprintf(stderr,
                            "[FINE-BF-OPAC][BLOCKED] reason=NEGATIVE_OR_"
                            "NONFINITE_TRUE_ABSORPTION shell=%d bin=%d nu=%.17g "
                            "interpolated=%.17g smeared_bf=%.17g sharp_bf=%.17g "
                            "assembled=%.17g floor=0 clamp=0 fallback=0\n",
                            bad_shell,bad_bin,fs.nu[bad_bin],interpolated,
                            smeared,sharp,assembled);
                    free(chi_bf_fine);
                    free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
                    free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
                    free(eta);
                    return -1;
                }
                fprintf(stderr, "[FINE-BF-OPAC] sharp-edge bf continuum applied "
                        "(NS=%d NF=%d, Σchi_abs %.3e -> %.3e)\n", NS, NF, sum_old, sum_new);
                /* DIAG: for the outer shell, show how much the bf sharpening changed
                 * chi_abs locally AND chi_abs vs scattering — disambiguates "swamped"
                 * (chi_abs<<chi_es) vs "too weak" (chi_bf_fine~smeared). One-shot. */
                static int bfdiag = -1;
                if (bfdiag < 0) { const char *e=getenv("LUMINA_CMF_FINE_BF_DIAG"); bfdiag=(e&&atoi(e))?1:0; }
                if (bfdiag) {
                    int sd = NS-1;   /* outer shell */
                    fprintf(stderr, "[FINE-BF-DIAG] shell %d (outer): lam_A  chi_es  smeared_bf  fine_bf  fine/smeared\n", sd);
                    for (int j = 1; j <= 12; ++j) {
                        int i = (int)((double)j/13.0 * NF);
                        double lamA = 2.99792458e18 / fs.nu[i];
                        double ce = fs.chi_es[(size_t)sd*NF+i];
                        double sb = bf_get_chi(g_fine_bf, sd, fs.nu[i]);
                        double ff = chi_bf_fine[(size_t)sd*NF+i];
                        fprintf(stderr, "[FINE-BF-DIAG]  %.0f  %.3e  %.3e  %.3e  %.2f\n",
                                lamA, ce, sb, ff, sb>0?ff/sb:0.0);
                    }
                }
            } else {
                fprintf(stderr,
                        "[FINE-BF-OPAC][BLOCKED] requested sharp-edge "
                        "continuum compute failed\n");
                free(chi_bf_fine);
                free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
                free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
                free(eta);
                return -1;
            }
            free(chi_bf_fine);
        }
    }

    /* --- deposit line profiles (chi_line) + emissivity (eta) --- */
    /* CONTONLY falsifier (2026-06-25): skip the line deposit entirely so the
     * emergent is pure continuum (S_fixed = chi_abs*B/(chi_es+chi_abs)).
     * A/B vs full isolates whether the grn/nir gap is continuum color
     * (blanketing) or thermal line re-emission (-> needs fluorescence). */
    int fine_contonly = 0;
    { const char *e = getenv("LUMINA_CMF_FINE_CONTONLY"); if (e && atoi(e)) fine_contonly = 1; }
    const double SQRTPI = 1.7724538509055160;
    double chi0_pref = 1.0 / (SQRTPI * vdop * t_exp);   /* chi0 = (1-e^-tau)*pref */
    int src_nlte = (opac->line_source_S != NULL);
    long n_inwin = 0, n_clamped = 0;
    long srce_chk_cells = 0;
    int material_error = 0;
    uint64_t signed_cells = 0, exact_zero_tau_cells = 0;
    uint64_t raw_negative_cells = 0, mild_negative_cells = 0;
    uint64_t srce_chk_expected_cells = 0;
    double max_slb = 0.0;   /* max S_l/B(Te) over deposited lines (diagnostic) */
    /* Strong-line threshold: skip lines whose Sobolev tau is below fine_taumin in
     * every shell. The dense UV pump forest (1000-3000A) is otherwise intractable
     * to deposit on the fine mesh; for the fluorescence pump only the dominant
     * tau_S lines matter (design option-c). Default 1e-12 = deposit all (legacy).
     * Set LUMINA_CMF_FINE_TAUMIN ~0.1-1 for a tractable UV pump producer. */
    double fine_taumin = 1e-12;
    { const char *e = getenv("LUMINA_CMF_FINE_TAUMIN"); if (e) fine_taumin = atof(e); }
    long n_skip_weak = 0;
    int line_net_parity = g_fine_atom && g_fine_nlte &&
        g_fine_upper_population_fill && g_fine_nlte->line_eset;
    int sobolev_operator = line_net_parity &&
        line_operator == CMF_FINE_LINE_OPERATOR_CMFGEN_NONOVERLAP_SOBOLEV;
    if (line_operator ==
            CMF_FINE_LINE_OPERATOR_CMFGEN_NONOVERLAP_SOBOLEV &&
        !line_net_parity) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=SOBOLEV_OPERATOR_WITHOUT_PARITY "
                "operator=%d action=TERMINATE\n", (int)line_operator);
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta);return -1;
    }
    if (line_net_parity &&
        (fine_contonly || line_eps != 1.0 || sl_clamp != 0.0 ||
         fine_taumin > 1.0e-12)) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=LINE_NET_PARITY_CONFIG "
                "contonly=%d line_eps=%.17g sl_clamp=%.17g taumin=%.17g "
                "required=full_raw_material_SRCE_CHK\n",
                fine_contonly,line_eps,sl_clamp,fine_taumin);
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta);return -1;
    }
    double *upper_population_cache = NULL;
    if (line_net_parity) {
        if ((size_t)NL > SIZE_MAX / (size_t)NS ||
            (size_t)NL * (size_t)NS > SIZE_MAX / sizeof(double)) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=UPPER_POPULATION_SHAPE\n");
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta);return -1;
        }
        size_t population_cells = (size_t)NL * (size_t)NS;
        upper_population_cache = malloc(
            population_cells * sizeof(*upper_population_cache));
        if (!upper_population_cache ||
            g_fine_upper_population_fill(
                g_fine_atom, plasma, g_fine_nlte, (size_t)NL, (size_t)NS,
                upper_population_cache) != 0) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=UPPER_POPULATION_BULK_BUILD\n");
            free(upper_population_cache);
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta);return -1;
        }
        fprintf(stderr,
                "[cmf_fine][LINE-MATERIAL] mode=%s "
                "upper_population_cells=%zu bytes=%zu\n",
                sobolev_operator ? "CMFGEN_NONOVERLAP_SOBOLEV" :
                                   "INIT_SHARED_GAUSSIAN",
                population_cells,
                population_cells * sizeof(*upper_population_cache));
    }
    /* OMP enabler: the per-line deposit accumulates into chi_line[s,:]/eta[s,:], so
     * parallelising over LINES would race. Instead precompute the per-line in-window
     * + weak-skip flag SERIALLY (cheap, read-only), then parallelise the deposit over
     * SHELLS (each thread owns one shell's chi_line/eta -> no race). Behaviour is
     * identical to the old line-serial loop (same deposit, same diagnostics). */
    char *line_use = NULL;
    if (!fine_contonly) {
        line_use = (char *)malloc((size_t)NL);
        if (!line_use) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=LINE_SELECTION_ALLOCATION\n");
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta);free(upper_population_cache);
            return -1;
        }
        for (int l = 0; l < NL; ++l) {
            double nu_l = opac->line_list_nu[l];
            char use = (nu_l >= line_nu_lo && nu_l <= line_nu_hi) ? 1 : 0;
            if (use) {
                ++n_inwin;
                if (fine_taumin > 1e-12) {        /* skip line if weak in ALL shells */
                    double tmax = 0.0;
                    for (int s = 0; s < NS; ++s) {
                        double t = opac->tau_sobolev[(size_t)l * NS + s];
                        if (t > tmax) tmax = t;
                    }
                    if (tmax < fine_taumin) { ++n_skip_weak; use = 0; }
                }
            }
            line_use[l] = use;
        }
    }
    if (line_net_parity) {
        double minimum_tau = 0.0;
        #pragma omp parallel for schedule(static) collapse(2) \
            reduction(+:signed_cells,exact_zero_tau_cells,raw_negative_cells, \
                        mild_negative_cells,srce_chk_expected_cells, \
                        srce_chk_cells) reduction(|:material_error) \
            reduction(min:minimum_tau)
        for (int l = 0; l < NL; ++l) {
            for (int s = 0; s < NS; ++s) {
                if (!line_use[l]) continue;
                size_t cell = (size_t)l * (size_t)NS + (size_t)s;
                double tau = opac->tau_sobolev[cell];
                int validity = opac->tau_validity[cell];
                if (validity == A208_EXACT_ZERO) {
                    ++exact_zero_tau_cells;
                } else if (validity == A208_VALID && isfinite(tau)) {
                    ++signed_cells;
                    if (tau < minimum_tau) minimum_tau = tau;
                    if (tau < 0.0) {
                        ++raw_negative_cells;
                        if (tau < -0.5) ++srce_chk_expected_cells;
                        else ++mild_negative_cells;
                    }
                } else {
                    material_error = 1;
                    continue;
                }
                double nupper = upper_population_cache[cell];
                double Aul = g_fine_atom->line_A_ul ?
                    g_fine_atom->line_A_ul[l] : NAN;
                LineNetSobolevMaterial material;
                if (!isfinite(nupper) || nupper < 0.0 ||
                    line_net_sobolev_material(
                        nupper, Aul, opac->line_list_nu[l], tau, t_exp,
                        LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK, 1,
                        &material) != 0) {
                    material_error = 1;
                    continue;
                }
                srce_chk_cells += material.srce_chk_applied;
            }
        }
        size_t minimum_cell = SIZE_MAX;
        if (raw_negative_cells != 0) {
            #pragma omp parallel for schedule(static) collapse(2) \
                reduction(min:minimum_cell)
            for (int l = 0; l < NL; ++l) {
                for (int s = 0; s < NS; ++s) {
                    size_t cell = (size_t)l * (size_t)NS + (size_t)s;
                    if (line_use[l] && opac->tau_sobolev[cell] == minimum_tau &&
                        cell < minimum_cell)
                        minimum_cell = cell;
                }
            }
        }
        int minimum_line = minimum_cell == SIZE_MAX ? -1
                         : (int)(minimum_cell / (size_t)NS);
        int minimum_shell = minimum_cell == SIZE_MAX ? -1
                          : (int)(minimum_cell % (size_t)NS);
        double n_upper = minimum_cell == SIZE_MAX ? NAN
                       : upper_population_cache[minimum_cell];
        int lower_global = minimum_line < 0 ? -1
                         : cmf_fine_global_level(g_fine_atom, minimum_line, 0);
        int upper_global = minimum_line < 0 ? -1
                         : cmf_fine_global_level(g_fine_atom, minimum_line, 1);
        double stimulated_upper = NAN, population_difference = NAN;
        double reconstructed_lower = NAN;
        if (minimum_line >= 0 && lower_global >= 0 && upper_global >= 0 &&
            g_fine_atom->level_g && g_fine_atom->line_f_lu &&
            g_fine_atom->line_wavelength_cm) {
            double g_lower = g_fine_atom->level_g[lower_global];
            double g_upper = g_fine_atom->level_g[upper_global];
            double denominator = SOBOLEV_COEFF *
                g_fine_atom->line_f_lu[minimum_line] *
                g_fine_atom->line_wavelength_cm[minimum_line] * t_exp;
            if (g_lower > 0.0 && g_upper > 0.0 && denominator != 0.0 &&
                isfinite(denominator)) {
                stimulated_upper = (g_lower / g_upper) * n_upper;
                population_difference = minimum_tau / denominator;
                reconstructed_lower = stimulated_upper + population_difference;
            }
        }
        fprintf(stderr,
                "[cmf_fine][SIGNED-MATERIAL-CENSUS] line_shells=%llu "
                "exact_zero_tau=%llu raw_negative=%llu mild_negative=%llu "
                "srce_chk=%llu minimum_line=%d minimum_shell=%d "
                "minimum_tau=%.17g n_upper=%.17g stimulated_upper=%.17g "
                "population_difference_from_tau=%.17g "
                "reconstructed_n_lower=%.17g raw_preserved=1 floor=0 "
                "clamp=0 jitter=0\n",
                (unsigned long long)signed_cells,
                (unsigned long long)exact_zero_tau_cells,
                (unsigned long long)raw_negative_cells,
                (unsigned long long)mild_negative_cells,
                (unsigned long long)srce_chk_expected_cells,
                minimum_line,minimum_shell,minimum_tau,n_upper,
                stimulated_upper,population_difference,reconstructed_lower);
        if (material_error ||
            (uint64_t)srce_chk_cells != srce_chk_expected_cells) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=SIGNED_MATERIAL_CENSUS_MISMATCH "
                    "srce_chk_expected=%llu srce_chk_material=%ld "
                    "material_error=%d raw_preserved=1 floor=0 clamp=0 "
                    "jitter=0 repair=0 action=TERMINATE\n",
                    (unsigned long long)srce_chk_expected_cells,
                    srce_chk_cells,material_error);
            free(line_use);free(upper_population_cache);
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta);return -1;
        }
        fprintf(stderr,
                "[cmf_fine][SIGNED-MATERIAL-POLICY] operator=%s "
                "srce_chk_expected=%llu srce_chk_material=%ld "
                "raw_preserved=1 floor=0 clamp=0 jitter=0 repair=0\n",
                sobolev_operator ? "CMFGEN_NONOVERLAP_SOBOLEV" :
                                   "INIT_SHARED_GAUSSIAN",
                (unsigned long long)srce_chk_expected_cells,
                srce_chk_cells);
        /* The census count is independent evidence.  Runtime line-operator
         * accounting below starts from zero and must reproduce it. */
        srce_chk_cells = 0;
    }
    /* [OWNER] load-balance redesign (user #1, 2026-07-06): the legacy loop is
     * shell-parallel (width 50, inner-shell tail-dominated -> ~18 cores).
     * Owner-computes: fine-grid CHUNKS own their bins; each (s,chunk) task
     * binary-searches the DESCENDING line_list_nu for lines whose +-(4 sigma
     * + guard) deposit window reaches its bins. Every (s,i) bin is owned by
     * exactly one task and contributing lines are processed in the same l
     * order as the legacy loop => bit-identical accumulation.
     * LUMINA_FINE_OWNER=0 restores the legacy loop; non-monotone line list
     * also falls back. */
    int fine_owner = 1;
    { const char *e = getenv("LUMINA_FINE_OWNER"); if (e) fine_owner = atoi(e); }
    if (fine_owner) {
        static int nu_desc_ok = -1;
        if (nu_desc_ok < 0) {
            nu_desc_ok = 1;
            for (int l = 1; l < NL; ++l)
                if (opac->line_list_nu[l] > opac->line_list_nu[l-1]) { nu_desc_ok = 0; break; }
            if (!nu_desc_ok)
                printf("[FINE-OWNER] line_list_nu not descending -> legacy loop\n");
        }
        if (!nu_desc_ok) fine_owner = 0;
    }
    if (fine_owner && !fine_contonly && !sobolev_operator) {
        int nch = 512;
        { const char *e = getenv("LUMINA_FINE_NCHUNK"); if (e) nch = atoi(e); }
        if (nch < 1) nch = 1; if (nch > NF) nch = NF;
        int chunk_sz = (NF + nch - 1) / nch;
        double marg = 8.0 * vdop / CM_C + 4.0 * dlognu;   /* window guard */
        #pragma omp parallel for schedule(dynamic) collapse(2) \
                reduction(max:max_slb) reduction(+:n_clamped,srce_chk_cells) \
                reduction(|:material_error)
        for (int s = 0; s < NS; ++s) {
            for (int ch = 0; ch < nch; ++ch) {
                const int ICLO = ch * chunk_sz;
                int ichi_t = ICLO + chunk_sz - 1;
                if (ichi_t >= NF) ichi_t = NF - 1;
                const int ICHI = ichi_t;
                if (ICLO > ICHI) continue;
                double nu_min = fs.nu[ICLO] * (1.0 - marg);
                double nu_max = fs.nu[ICHI] * (1.0 + marg);
                /* descending list: la = first idx with nu <= nu_max,
                 *                  lb = first idx with nu <  nu_min */
                int la, lb;
                { int lo = 0, hi = NL;
                  while (lo < hi) { int mid = lo + (hi - lo) / 2;
                      if (opac->line_list_nu[mid] > nu_max) lo = mid + 1; else hi = mid; }
                  la = lo; lo = la; hi = NL;
                  while (lo < hi) { int mid = lo + (hi - lo) / 2;
                      if (opac->line_list_nu[mid] >= nu_min) lo = mid + 1; else hi = mid; }
                  lb = lo; }
                for (int l = la; l < lb; ++l) {
            if (!line_use[l]) continue;
            double nu_l = opac->line_list_nu[l];
            double tau = opac->tau_sobolev[(size_t)l * NS + s];
            double dnuD = nu_l * vdop / CM_C;
            double xc = log(nu_l / nu_lo) / dlognu - 0.5;
            int half  = (int)ceil(4.0 * dnuD / (nu_l * dlognu)) + 1;
            int ic = (int)floor(xc + 0.5);
            int i0 = ic - half, i1 = ic + half;
            if (i0 < ICLO) i0 = ICLO; if (i1 > ICHI) i1 = ICHI;
            if (i0 > i1) continue;
            double chi0 = 0.0, eta0 = 0.0, source_diag = 0.0;
            int srce_chk = 0, clamped = 0;
            int material_rc = cmf_fine_line_material(
                opac, plasma, l, s, NS, nu_l, tau, dnuD, t_exp,
                chi0_pref, line_net_parity, src_nlte, line_eps, sl_clamp,
                upper_population_cache,
                &chi0, &eta0, &source_diag, &srce_chk, &clamped);
            if (material_rc > 0) continue;
            if (material_rc < 0) { material_error = 1; continue; }
            n_clamped += clamped;
            /* A profile may touch two owner chunks. Count its material
             * identity once, in the unique chunk containing its centre. */
            if (srce_chk && ic >= ICLO && ic <= ICHI) ++srce_chk_cells;
            double Bl = cm_planck(nu_l, plasma->T_e[s]);
            if (Bl > 0.0 && source_diag > 0.0) {
                double rb = source_diag / Bl;
                if (rb > max_slb) max_slb = rb;
            }
            for (int i = i0; i <= i1; ++i) {
                double xv = (fs.nu[i] - nu_l) / dnuD;
                if (fabs(xv) > LINE_JBAR_PROFILE_NDOPPLER) continue;
                double p  = exp(-xv * xv);
                double cl = chi0 * p;
                fs.chi_line[(size_t)s*NF+i] += cl;
                eta        [(size_t)s*NF+i] += eta0 * p;
            }
                }
            }
        }
    } else if (!fine_contonly && !sobolev_operator) {
    #pragma omp parallel for schedule(dynamic) reduction(max:max_slb) \
            reduction(+:n_clamped,srce_chk_cells) reduction(|:material_error)
    for (int s = 0; s < NS; ++s) {
        const int ICLO = 0, ICHI = NF - 1;
        for (int l = 0; l < NL; ++l) {
            if (!line_use[l]) continue;
            double nu_l = opac->line_list_nu[l];
            double tau = opac->tau_sobolev[(size_t)l * NS + s];
            double dnuD = nu_l * vdop / CM_C;
            double xc = log(nu_l / nu_lo) / dlognu - 0.5;
            int half  = (int)ceil(4.0 * dnuD / (nu_l * dlognu)) + 1;
            int ic = (int)floor(xc + 0.5);
            int i0 = ic - half, i1 = ic + half;
            if (i0 < ICLO) i0 = ICLO; if (i1 > ICHI) i1 = ICHI;
            if (i0 > i1) continue;
            double chi0 = 0.0, eta0 = 0.0, source_diag = 0.0;
            int srce_chk = 0, clamped = 0;
            int material_rc = cmf_fine_line_material(
                opac, plasma, l, s, NS, nu_l, tau, dnuD, t_exp,
                chi0_pref, line_net_parity, src_nlte, line_eps, sl_clamp,
                upper_population_cache,
                &chi0, &eta0, &source_diag, &srce_chk, &clamped);
            if (material_rc > 0) continue;
            if (material_rc < 0) { material_error = 1; continue; }
            n_clamped += clamped;
            srce_chk_cells += srce_chk;
            double Bl = cm_planck(nu_l, plasma->T_e[s]);
            if (Bl > 0.0 && source_diag > 0.0) {
                double rb = source_diag / Bl;
                if (rb > max_slb) max_slb = rb;
            }
            for (int i = i0; i <= i1; ++i) {
                double xv = (fs.nu[i] - nu_l) / dnuD;
                if (fabs(xv) > LINE_JBAR_PROFILE_NDOPPLER) continue;
                double p  = exp(-xv * xv);
                double cl = chi0 * p;
                fs.chi_line[(size_t)s*NF+i] += cl;
                eta        [(size_t)s*NF+i] += eta0 * p;
            }
        }
    }
    }
    free(line_use);
    free(upper_population_cache);
    if (material_error) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=LINE_NET_MATERIAL_INVALID "
                "mode=%s\n", line_net_parity ? "CMFGEN_DIRECT" : "LEGACY");
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta);
        return -1;
    }

    /* --- assemble chi_tot + S_fixed.  eps<1: fold (1-eps)*chi_line into chi_es
     * (ALI scattering albedo) so the line forest scatters the photospheric UV
     * field instead of thermalising it; eta already holds only eps*chi_line*B.
     * Total opacity is preserved: chi_es+(1-eps)cl + chi_abs + eps*cl = orig. */
    size_t first_assembly_invalid = SIZE_MAX;
    #pragma omp parallel for schedule(static) reduction(min:first_assembly_invalid)
    for (int s = 0; s < NS; ++s) {
        double Te = plasma->T_e[s];
        for (int i = 0; i < NF; ++i) {
            size_t idx = (size_t)s*NF+i;
            if (line_eps < 1.0) fs.chi_es[idx] += (1.0 - line_eps) * fs.chi_line[idx];
            double chi_ln_th = (line_eps < 1.0) ? (line_eps * fs.chi_line[idx]) : fs.chi_line[idx];
            double ct = fs.chi_es[idx] + fs.chi_abs[idx] + chi_ln_th;
            fs.chi_tot[idx] = ct;
            double Bnu = cm_planck(fs.nu[i], Te);
            double numerator = fs.chi_abs[idx]*Bnu + eta[idx];
            if (isfinite(ct) && isfinite(numerator) && ct > 0.0 &&
                numerator >= 0.0)
                fs.S_fixed[idx] = numerator / ct;
            else if (ct == 0.0 && numerator == 0.0)
                fs.S_fixed[idx] = 0.0; /* algebraic exact-zero provenance */
            else {
                fs.S_fixed[idx] = NAN;
                if (idx < first_assembly_invalid)
                    first_assembly_invalid = idx;
            }
            fs.J[idx] = Bnu;                              /* warm ALI start */
        }
    }
    if (first_assembly_invalid != SIZE_MAX) {
        size_t bad_shell = first_assembly_invalid / (size_t)NF;
        size_t bad_bin = first_assembly_invalid % (size_t)NF;
        double Bnu = cm_planck(fs.nu[bad_bin], plasma->T_e[bad_shell]);
        double numerator = fs.chi_abs[first_assembly_invalid] * Bnu +
                           eta[first_assembly_invalid];
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=NEGATIVE_OR_NONFINITE_TOTAL_SOURCE "
                "shell=%zu bin=%zu nu=%.17g chi_es=%.17g chi_abs=%.17g "
                "chi_line=%.17g chi_tot=%.17g eta_line=%.17g "
                "source_numerator=%.17g floor=0 clamp=0 fallback=0\n",
                bad_shell,bad_bin,fs.nu[bad_bin],
                fs.chi_es[first_assembly_invalid],
                fs.chi_abs[first_assembly_invalid],
                fs.chi_line[first_assembly_invalid],
                fs.chi_tot[first_assembly_invalid],
                eta[first_assembly_invalid],numerator);
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta);
        return -1;
    }

    if (diag && sobolev_operator) {
        fprintf(stderr,
                "[cmf_fine][LINE-OPERATOR] mode=CMFGEN_NONOVERLAP_SOBOLEV "
                "fine_exact_field=CONTINUUM_ONLY shared_line_deposit_cells=0 "
                "signed_cells=%llu exact_zero_tau=%llu raw_negative=%llu "
                "mild_negative=%llu srce_chk_expected=%llu raw_preserved=1 "
                "floor=0 clamp=0 jitter=0 repair=0\n",
                (unsigned long long)signed_cells,
                (unsigned long long)exact_zero_tau_cells,
                (unsigned long long)raw_negative_cells,
                (unsigned long long)mild_negative_cells,
                (unsigned long long)srce_chk_expected_cells);
    } else if (diag) {   /* shared-profile initialization tie-back */
        int st = NS/2; double got=0.0, exp_=0.0;
        for (int i = 0; i < NF; ++i) got += fs.chi_line[(size_t)st*NF+i]*fs.dnu[i];
        for (int l = 0; l < NL; ++l) { double nu_l=opac->line_list_nu[l];
            if (nu_l < line_nu_lo || nu_l > line_nu_hi) continue;
            double tau=opac->tau_sobolev[(size_t)l*NS+st];
            if (line_net_parity) {
                LineNetSobolevMaterial material;
                if (line_net_sobolev_material(
                        0.0, 0.0, nu_l, tau, t_exp,
                        LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK, 1,
                        &material) == 0)
                    exp_ += material.effective_integrated_opacity;
            } else {
                double frac=(tau>1e-6)?-expm1(-tau):(tau>0?tau:0);
                exp_ += frac*nu_l/(CM_C*t_exp);
            } }
        fprintf(stderr,
            "[cmf_fine] S_l deposit: max S_l/B=%.3e  clamped=%ld/%ld lines (sl_clamp=%.1f)  "
            "skipped weak(tau<%.2g)=%ld  line_eps=%.3g%s\n",
            max_slb, n_clamped, n_inwin, sl_clamp, fine_taumin, n_skip_weak,
            line_eps, (line_eps < 1.0) ? " [SCATTERING pump]" :
            (line_net_parity ? " [INIT shared CMFGEN material]" : " [thermal]"));
        fprintf(stderr,
            "[cmf_fine] lines in window=%ld  tie-back shell %d: "
            "Int chi_line dnu=%.4e  material expect=%.4e  ratio=%.4f "
            "mode=%s srce_chk_line_shells=%ld\n",
            n_inwin, st, got, exp_, (exp_!=0.0)?got/exp_:0.0,
            line_net_parity ? "INIT_SHARED_GAUSSIAN" : "LEGACY_EXPANSION",
            srce_chk_cells);
    }

    /* --- exact frequency-coupled solve on true drifting characteristics --- */
    size_t fine_cells = (size_t)NS * (size_t)NF;
    int ab_enabled = 0;
    int external_fixture_enabled = 0;
    int ab_requested_devices = 0;
    if (cmf_optional_binary_env("LUMINA_CMF_FINE_MGPU_AB", &ab_enabled) != 0) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=INVALID_MGPU_AB_REQUEST "
                "value=%s expected=0_or_1\n",
                getenv("LUMINA_CMF_FINE_MGPU_AB") ?
                    getenv("LUMINA_CMF_FINE_MGPU_AB") : "(unset)");
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta);
        return -1;
    }
    if (cmf_optional_binary_env(
            "LUMINA_CMF_FINE_EXTERNAL_FIXTURE",
            &external_fixture_enabled) != 0) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=INVALID_EXTERNAL_FIXTURE_REQUEST "
                "value=%s expected=0_or_1\n",
                getenv("LUMINA_CMF_FINE_EXTERNAL_FIXTURE") ?
                    getenv("LUMINA_CMF_FINE_EXTERNAL_FIXTURE") : "(unset)");
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta);
        return -1;
    }
    if (ab_enabled &&
        (cmf_fine_multigpu_device_request(&ab_requested_devices) != 0 ||
         ab_requested_devices <= 0)) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=MGPU_AB_REQUIRES_POSITIVE_DEVICES "
                "value=%s\n",
                getenv("LUMINA_CMF_FINE_MGPU_DEVICES") ?
                    getenv("LUMINA_CMF_FINE_MGPU_DEVICES") : "(unset)");
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta);
        return -1;
    }
#ifndef LUMINA_HAS_CUDA_BF_GEMM
    if (ab_enabled) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=MGPU_AB_REQUESTED_IN_CPU_BUILD "
                "devices=%d\n", ab_requested_devices);
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta);
        return -1;
    }
#endif
    double *fine_error_upper =
        (double *)malloc(fine_cells * sizeof(*fine_error_upper));
    if (!fine_error_upper) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=COMPONENT_ERROR_ALLOCATION "
                "cells=%zu bytes=%zu\n",
                fine_cells, fine_cells * sizeof(*fine_error_upper));
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta);
        return -1;
    }
    double *ab_cpu_J = NULL;
    double *ab_cpu_error_upper = NULL;
    CMFExactReport ab_cpu_report;
    CMFExactStatus ab_cpu_status = CMF_EXACT_INVALID_INPUT;
    double ab_cpu_seconds = 0.0;
    if (ab_enabled) {
        ab_cpu_J = (double *)malloc(fine_cells * sizeof(*ab_cpu_J));
        ab_cpu_error_upper =
            (double *)malloc(fine_cells * sizeof(*ab_cpu_error_upper));
        if (!ab_cpu_J || !ab_cpu_error_upper) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=MGPU_AB_CPU_ALLOCATION "
                    "cells=%zu bytes_per_array=%zu\n",
                    fine_cells, fine_cells * sizeof(*ab_cpu_J));
            free(ab_cpu_J); free(ab_cpu_error_upper);
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta); free(fine_error_upper);
            return -1;
        }
        memcpy(ab_cpu_J, fs.J, fine_cells * sizeof(*ab_cpu_J));
        double ab_cpu_start = cmf_wall_seconds();
        ab_cpu_status = cmf_exact_characteristic_solve_with_envelope(
            NS, NF, dlognu, fs.nu, geo->r_inner, geo->r_outer,
            geo->time_explosion, T_inner, cmf_inner_bb_scale(),
            fs.chi_tot, fs.chi_es, fs.S_fixed,
            ab_cpu_J, ab_cpu_error_upper, envelope_refinements,
            n_ali, solve_tol, CMF_EXACT_MODE_POSITIVE_SLIDING,
            &ab_cpu_report);
        ab_cpu_seconds = cmf_wall_seconds() - ab_cpu_start;
        if (ab_cpu_status != CMF_EXACT_OK) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=MGPU_AB_CPU_BASELINE "
                    "status=%s iterations=%d residual=%.17g "
                    "absolute_change=%.17g component_envelope=%d "
                    "floor=0 clamp=0 jitter=0 fallback=0\n",
                    cmf_exact_status_name(ab_cpu_status),
                    ab_cpu_report.iterations_used,
                    ab_cpu_report.final_max_relative_change,
                    ab_cpu_report.final_max_absolute_change,
                    ab_cpu_report.componentwise_error_envelope_verified);
            free(ab_cpu_J); free(ab_cpu_error_upper);
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta); free(fine_error_upper);
            return -1;
        }
    }
    CMFFineOwnerResult owner_result;
    double owner_start = cmf_wall_seconds();
    int owner_rc = cmf_fine_exact_owner_solve(
        NS, NF, dlognu, fs.nu, geo->r_inner, geo->r_outer,
        geo->time_explosion, T_inner, cmf_inner_bb_scale(),
        fs.chi_tot, fs.chi_es, fs.S_fixed, fs.J, fine_error_upper,
        envelope_refinements, n_ali, solve_tol, &owner_result);
    double owner_seconds = cmf_wall_seconds() - owner_start;
    CMFExactReport exact_report = owner_result.exact;
    if (owner_result.config_status ==
            CMF_FINE_OWNER_INVALID_DEVICE_REQUEST) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=INVALID_MGPU_DEVICE_REQUEST "
                "value=%s\n",
                getenv("LUMINA_CMF_FINE_MGPU_DEVICES") ?
                    getenv("LUMINA_CMF_FINE_MGPU_DEVICES") : "(unset)");
    } else if (owner_result.config_status ==
                   CMF_FINE_OWNER_CUDA_NOT_LINKED) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=MULTIGPU_REQUESTED_IN_CPU_BUILD "
                "devices=%d\n", owner_result.multigpu_requested);
    } else if (owner_result.multigpu_requested > 0) {
#ifdef LUMINA_HAS_CUDA_BF_GEMM
        fprintf(stderr,
                "[cmf_fine][EXACT-MULTIGPU-EPOCH] status=%s devices=%d/%d "
                "iterations=%d cap=%d residual=%.17g tolerance=%.17g "
                "absolute_change=%.17g scattering_ratio_bound=%.17g "
                "absolute_error_bound=%.17g component_envelope=%d "
                "seed_attempts=%zu refinements=%zu "
                "component_residual_upper=%.17g "
                "component_error=[%.17g,%.17g] max_drift_bins=%.17g "
                "schedule=%d/%d/%d partition_weighted=%d "
                "max_device_bytes=%zu total_device_bytes=%zu "
                "failure_phase=%d failure_iteration=%d failure_device=%d "
                "failure_ray=[%d,%d) failure_segment=%d failure_bin=%d "
                "failure_value=%.17g floor=0 clamp=0 jitter=0 "
                "domain_hash=%s canonical_edge_hash=%s\n",
                cmf_multigpu_status_name(
                    (CMFMultiGPUStatus)owner_result.multigpu_status),
                owner_result.devices_used, owner_result.visible_devices,
                exact_report.iterations_used, exact_report.iteration_cap,
                exact_report.final_max_relative_change,
                exact_report.tolerance,
                exact_report.final_max_absolute_change,
                exact_report.max_scattering_ratio,
                exact_report.fixed_point_absolute_error_bound,
                exact_report.componentwise_error_envelope_verified,
                exact_report.componentwise_error_seed_attempts,
                exact_report.componentwise_error_refinement_iterations,
                exact_report.componentwise_residual_upper_max,
                exact_report.componentwise_error_upper_min,
                exact_report.componentwise_error_upper_max,
                exact_report.max_characteristic_drift_bins,
                owner_result.epoch_block_size,
                owner_result.epoch_batch_cardinality,
                owner_result.epoch_direct_replay_max_window,
                owner_result.weighted_partition,
                owner_result.max_device_allocated_bytes,
                owner_result.total_device_allocated_bytes,
                owner_result.failure_phase,
                owner_result.failure_iteration,
                owner_result.failure_device_index,
                owner_result.failure_ray_begin,
                owner_result.failure_ray_end,
                owner_result.failure_segment_index,
                owner_result.failure_bin_index,
                owner_result.failure_value,
                LINE_JBAR_BB_DOMAIN_CONTRACT_SHA256,
                LUMINA_RADFIELD_EDGE_SHA256);
        fprintf(stderr,
                "[cmf_fine][EXACT-MULTIGPU-TIMING] devices=%d "
                "initialization_s=%.9f fixed_point_s=%.9f "
                "source_assembly_s=%.9f h2d_s=%.9f "
                "device_sweep_s=%.9f d2h_s=%.9f host_reduction_s=%.9f "
                "convergence_check_s=%.9f envelope_context_setup_s=%.9f "
                "bounds_s=%.9f envelope_residual_s=%.9f "
                "envelope_verify_s=%.9f envelope_refine_s=%.9f "
                "publication_s=%.9f cleanup_s=%.9f "
                "reported_total_s=%.9f caller_total_s=%.9f "
                "convergence_denominator_floor=0\n",
                owner_result.devices_used,
                owner_result.initialization_seconds,
                owner_result.fixed_point_solve_seconds,
                owner_result.source_assembly_seconds,
                owner_result.host_to_device_seconds,
                owner_result.device_sweep_seconds,
                owner_result.device_to_host_seconds,
                owner_result.host_reduction_seconds,
                owner_result.convergence_check_seconds,
                owner_result.envelope_context_setup_seconds,
                owner_result.bounds_seconds,
                owner_result.envelope_residual_seconds,
                owner_result.envelope_verify_seconds,
                owner_result.envelope_refine_seconds,
                owner_result.publication_seconds,
                owner_result.cleanup_seconds,
                owner_result.total_seconds, owner_seconds);
        for (int device = 0;
             device < owner_result.device_partition_count; ++device) {
            fprintf(stderr,
                    "[cmf_fine][EXACT-MULTIGPU-DEVICE] index=%d "
                    "rays=[%d,%d) owned_segment_work=%zu "
                    "computed_segment_work=%zu allocated_bytes=%zu\n",
                    device, owner_result.device_ray_begin[device],
                    owner_result.device_ray_end[device],
                    owner_result.device_owned_segment_work[device],
                    owner_result.device_computed_segment_work[device],
                    owner_result.device_allocated_bytes[device]);
        }
#endif
    } else {
        fprintf(stderr,
                "[cmf_fine][EXACT-POSITIVE-SLIDING] status=%s iterations=%d cap=%d "
                "residual=%.17g tolerance=%.17g absolute_change=%.17g "
                "scattering_ratio_bound=%.17g absolute_error_bound=%.17g "
                "component_envelope=%d seed_attempts=%zu refinements=%zu "
                "component_residual_upper=%.17g component_error=[%.17g,%.17g] "
                "max_drift_bins=%.17g "
                "negative_recurrence=%llu first_negative=%.17g "
                "domain_hash=%s canonical_edge_hash=%s\n",
                cmf_exact_status_name(exact_report.status),
                exact_report.iterations_used, exact_report.iteration_cap,
                exact_report.final_max_relative_change,
                exact_report.tolerance,
                exact_report.final_max_absolute_change,
                exact_report.max_scattering_ratio,
                exact_report.fixed_point_absolute_error_bound,
                exact_report.componentwise_error_envelope_verified,
                exact_report.componentwise_error_seed_attempts,
                exact_report.componentwise_error_refinement_iterations,
                exact_report.componentwise_residual_upper_max,
                exact_report.componentwise_error_upper_min,
                exact_report.componentwise_error_upper_max,
                exact_report.max_characteristic_drift_bins,
                (unsigned long long)exact_report.negative_recurrence_count,
                exact_report.first_negative_recurrence,
                LINE_JBAR_BB_DOMAIN_CONTRACT_SHA256,
                LUMINA_RADFIELD_EDGE_SHA256);
    }
    if (owner_rc != 0) {
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta); free(fine_error_upper);
        free(ab_cpu_J); free(ab_cpu_error_upper);
        return -1;
    }
    if (ab_enabled) {
        double ab_compare_start = cmf_wall_seconds();
        size_t ab_bad_cell = SIZE_MAX;
        const char *ab_bad_reason = NULL;
        double max_relative_J = 0.0;
        double max_relative_error_width = 0.0;
        double max_distance_over_combined_envelope = 0.0;
        double cpu_J_min = INFINITY, cpu_J_max = -INFINITY;
        double gpu_J_min = INFINITY, gpu_J_max = -INFINITY;
        for (size_t cell = 0; cell < fine_cells; ++cell) {
            double cpu_J = ab_cpu_J[cell];
            double gpu_J = fs.J[cell];
            double cpu_error = ab_cpu_error_upper[cell];
            double gpu_error = fine_error_upper[cell];
            if (!(cpu_J >= 0.0) || !isfinite(cpu_J) ||
                !(gpu_J >= 0.0) || !isfinite(gpu_J) ||
                !(cpu_error >= 0.0) || !isfinite(cpu_error) ||
                !(gpu_error >= 0.0) || !isfinite(gpu_error)) {
                ab_bad_cell = cell;
                ab_bad_reason = "NEGATIVE_OR_NONFINITE_AB_VALUE";
                break;
            }
            long double distance = fabsl(
                (long double)gpu_J - (long double)cpu_J);
            long double combined_envelope =
                (long double)gpu_error + (long double)cpu_error;
            if (!(distance <= combined_envelope)) {
                ab_bad_cell = cell;
                ab_bad_reason = "DISJOINT_CPU_GPU_ENVELOPES";
                break;
            }
            double J_scale = fabs(cpu_J) > fabs(gpu_J) ?
                fabs(cpu_J) : fabs(gpu_J);
            double error_scale = cpu_error > gpu_error ?
                cpu_error : gpu_error;
            double relative_J = J_scale > 0.0 ?
                (double)(distance / (long double)J_scale) : 0.0;
            double relative_error_width = error_scale > 0.0 ?
                fabs(gpu_error - cpu_error) / error_scale : 0.0;
            double distance_over_envelope = combined_envelope > 0.0L ?
                (double)(distance / combined_envelope) : 0.0;
            if (relative_J > max_relative_J)
                max_relative_J = relative_J;
            if (relative_error_width > max_relative_error_width)
                max_relative_error_width = relative_error_width;
            if (distance_over_envelope >
                max_distance_over_combined_envelope)
                max_distance_over_combined_envelope =
                    distance_over_envelope;
            if (cpu_J < cpu_J_min) cpu_J_min = cpu_J;
            if (cpu_J > cpu_J_max) cpu_J_max = cpu_J;
            if (gpu_J < gpu_J_min) gpu_J_min = gpu_J;
            if (gpu_J > gpu_J_max) gpu_J_max = gpu_J;
        }
        if (ab_bad_cell != SIZE_MAX) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=%s cell=%zu shell=%zu bin=%zu "
                    "cpu_J=%.17g gpu_J=%.17g cpu_error=%.17g "
                    "gpu_error=%.17g floor=0 clamp=0 jitter=0 fallback=0\n",
                    ab_bad_reason, ab_bad_cell,
                    ab_bad_cell / (size_t)NF,
                    ab_bad_cell % (size_t)NF,
                    ab_cpu_J[ab_bad_cell], fs.J[ab_bad_cell],
                    ab_cpu_error_upper[ab_bad_cell],
                    fine_error_upper[ab_bad_cell]);
            free(ab_cpu_J); free(ab_cpu_error_upper);
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta); free(fine_error_upper);
            return -1;
        }
        if (!(max_relative_J <= 1.0e-12)) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=MGPU_AB_J_MISMATCH "
                    "max_relative_J=%.17g tolerance=1e-12 "
                    "floor=0 clamp=0 jitter=0 fallback=0\n",
                    max_relative_J);
            free(ab_cpu_J); free(ab_cpu_error_upper);
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta); free(fine_error_upper);
            return -1;
        }
        double ab_comparison_seconds = cmf_wall_seconds() - ab_compare_start;
        fprintf(stderr,
                "[cmf_fine][EXACT-MULTIGPU-AB] PASS cells=%zu devices=%d "
                "cpu_iterations=%d gpu_iterations=%d "
                "finite_cpu_J=[%.17g,%.17g] "
                "finite_gpu_J=[%.17g,%.17g] max_relative_J=%.17g "
                "max_relative_error_width=%.17g "
                "max_distance_over_combined_envelope=%.17g "
                "cpu_component_error=[%.17g,%.17g] "
                "gpu_component_error=[%.17g,%.17g] "
                "cpu_baseline_s=%.9f gpu_owner_s=%.9f comparison_s=%.9f "
                "floor=0 clamp=0 jitter=0 repair=0\n",
                fine_cells, owner_result.devices_used,
                ab_cpu_report.iterations_used, exact_report.iterations_used,
                cpu_J_min, cpu_J_max, gpu_J_min, gpu_J_max,
                max_relative_J, max_relative_error_width,
                max_distance_over_combined_envelope,
                ab_cpu_report.componentwise_error_upper_min,
                ab_cpu_report.componentwise_error_upper_max,
                exact_report.componentwise_error_upper_min,
                exact_report.componentwise_error_upper_max,
                ab_cpu_seconds, owner_seconds, ab_comparison_seconds);
        free(ab_cpu_J); free(ab_cpu_error_upper);
    }

    /* Read-only external-code fixture.  Each row is the production exact J_nu
     * at an actual fine-bin centre and the shell midpoint velocity.  An
     * independent CMFGEN reader can therefore evaluate the same scalar J_nu
     * definition at exactly these coordinates without reusing Lumina's
     * transport implementation. */
    if (external_fixture_enabled) {
        static const double target_lambda_A[] = {
            350.0, 600.0, 1000.0, 1500.0,
            2500.0, 5000.0, 10000.0, 15000.0
        };
        FILE *fixture = fopen("lumina_cmf_fine_external_fixture.csv", "w");
        int fixture_failed = fixture == NULL;
        if (fixture) {
            fprintf(fixture,
                    "shell,v_mid_km_s,target_lambda_A,actual_lambda_A,"
                    "nu_hz,J_nu\n");
            for (int shell = 0; shell < NS && !fixture_failed; ++shell) {
                double v_mid = 0.5 *
                    (geo->v_inner[shell] + geo->v_outer[shell]) / 1.0e5;
                if (!isfinite(v_mid)) {
                    fixture_failed = 1;
                    break;
                }
                for (size_t point = 0;
                     point < sizeof(target_lambda_A) /
                                 sizeof(target_lambda_A[0]); ++point) {
                    double target_nu = CM_C /
                        (target_lambda_A[point] * 1.0e-8);
                    double position = log(target_nu / nu_lo) / dlognu - 0.5;
                    long nearest = lround(position);
                    if (nearest < 0 || nearest >= NF) {
                        fixture_failed = 1;
                        break;
                    }
                    size_t cell = (size_t)shell * (size_t)NF +
                                  (size_t)nearest;
                    double value = fs.J[cell];
                    double actual_lambda = CM_C / fs.nu[nearest] * 1.0e8;
                    if (!(value >= 0.0) || !isfinite(value) ||
                        !(actual_lambda > 0.0) || !isfinite(actual_lambda)) {
                        fixture_failed = 1;
                        break;
                    }
                    fprintf(fixture,
                            "%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                            shell, v_mid, target_lambda_A[point],
                            actual_lambda, fs.nu[nearest], value);
                }
            }
            if (fclose(fixture) != 0) fixture_failed = 1;
        }
        if (fixture_failed) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] "
                    "reason=EXTERNAL_FIXTURE_PUBLICATION_FAILED "
                    "floor=0 clamp=0 jitter=0 repair=0\n");
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta); free(fine_error_upper);
            return -1;
        }
        fprintf(stderr,
                "[cmf_fine][EXTERNAL-JNU-FIXTURE] PASS rows=%zu shells=%d "
                "points_per_shell=%zu file=lumina_cmf_fine_external_fixture.csv "
                "quantity=J_nu units=erg_s-1_cm-2_Hz-1_sr-1 "
                "coordinate=actual_fine_bin_center floor=0 clamp=0 jitter=0 "
                "repair=0\n",
                (size_t)NS * sizeof(target_lambda_A) /
                    sizeof(target_lambda_A[0]),
                NS, sizeof(target_lambda_A) / sizeof(target_lambda_A[0]));
    }

    /* --- DIAGNOSTIC: J/B(lambda, shell) map (LUMINA_CMF_FINE_JMAP=1).
     * Locates WHERE (wavelength, shell) the fine field is super-thermal -> the
     * only place a fluorescence pump can come from (physics-agent 2026-06-26).
     * read-only, no feedback. Coarse 100A bins, all shells. */
    { const char *e = getenv("LUMINA_CMF_FINE_JMAP");
      if (e && atoi(e)) {
        FILE *fp = fopen("lumina_fine_jmap.csv", "w");
        if (fp) {
            fprintf(fp, "shell,lambda_A,Te,J_over_B\n");
            double binA = 100.0;   /* 100 Angstrom bins */
            for (int s = 0; s < NS; ++s) {
                double Te = plasma->T_e[s];
                double lam0 = lam_lo, lam;
                for (lam = lam0; lam < lam_hi; lam += binA) {
                    double lam_c = lam + 0.5*binA;
                    double nu_c = CM_C / (lam_c * 1.0e-8);
                    /* nearest fine bin */
                    double xx = log(nu_c / nu_lo) / dlognu - 0.5;
                    int bi = (int)floor(xx + 0.5);
                    if (bi < 0 || bi >= NF) continue;
                    double Jv = fs.J[(size_t)s*NF+bi];
                    double Bv = cm_planck(fs.nu[bi], Te);
                    fprintf(fp, "%d,%.1f,%.0f,%.4e\n", s, lam_c, Te,
                            (Bv > 0.0) ? Jv/Bv : 0.0);
                }
            }
            fclose(fp);
            if (diag) fprintf(stderr, "[cmf_fine] wrote lumina_fine_jmap.csv (J/B vs lambda,shell)\n");
        }
      }
    }

    /* --- Stage-1 PROOF: frequency-resolved emergent from the fine field --- */
    { const char *e = getenv("LUMINA_CMF_FINE_EMERGENT");
      if (e && atoi(e))
          cmfgen_fine_emergent(&fs, csb, geo, T_inner, "lumina_spectrum_freqres.csv"); }

    /* --- UNIFIED Doppler emergent: P-Cygni + fine color + NLTE fluorescence --- */
    { const char *e = getenv("LUMINA_CMF_FINE_EMERGENT_OBS");
      if (e && atoi(e))
          cmfgen_fine_emergent_obs(&fs, geo, T_inner, opac, plasma->T_e,
                                   "lumina_spectrum_freqres_obs.csv"); }

    /* --- extract J_bar_l and certified delta-J_bar from the same profile --- */
    int profile_failed = 0;
    int first_profile_bad_line = INT_MAX;
    int first_profile_bad_status = LINE_JBAR_PROFILE_OK;
    double profile_error_min = INFINITY, profile_error_max = 0.0;
    double sobolev_beta_min = INFINITY, sobolev_beta_max = 0.0;
    unsigned long long sobolev_jbar_cells = 0;
    unsigned long long sobolev_srce_chk_cells = 0;
    double *sobolev_upper_population_cache = NULL;
    if (sobolev_operator) {
        size_t population_cells = (size_t)NL * (size_t)NS;
        sobolev_upper_population_cache = malloc(
            population_cells * sizeof(*sobolev_upper_population_cache));
        if (!sobolev_upper_population_cache ||
            g_fine_upper_population_fill(
                g_fine_atom, plasma, g_fine_nlte, (size_t)NL, (size_t)NS,
                sobolev_upper_population_cache) != 0) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] "
                    "reason=SOBOLEV_UPPER_POPULATION_BULK_BUILD "
                    "floor=0 clamp=0 jitter=0 repair=0 action=TERMINATE\n");
            free(sobolev_upper_population_cache);
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta); free(fine_error_upper);
            return -1;
        }
    }
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64) reduction(|:profile_failed) \
        reduction(min:profile_error_min,sobolev_beta_min) \
        reduction(max:profile_error_max,sobolev_beta_max) \
        reduction(+:sobolev_jbar_cells,sobolev_srce_chk_cells)
    #endif
    for (int l = 0; l < NL; ++l) {
        double nu_l = opac->line_list_nu[l];
        if (nu_l < line_nu_lo || nu_l > line_nu_hi) continue;
        double line_value[256], line_error[256];
        LineJbarProfileReport profile_report;
        LineJbarProfileStatus profile_status =
            line_jbar_gaussian_discrete_shells(
                (size_t)NS, (size_t)NF, fs.nu, fs.dnu, fs.J,
                fine_error_upper, nu_l, vdop,
                LINE_JBAR_PROFILE_NDOPPLER,
                line_value, line_error, &profile_report);
        if (profile_status != LINE_JBAR_PROFILE_OK) {
            profile_failed = 1;
            #ifdef _OPENMP
            #pragma omp critical(cmf_profile_failure)
            #endif
            {
                if (l < first_profile_bad_line) {
                    first_profile_bad_line = l;
                    first_profile_bad_status = (int)profile_status;
                }
            }
            continue;
        }
        for (int s = 0; s < NS; ++s) {
            size_t cell = (size_t)l * (size_t)NS + (size_t)s;
            if (sobolev_operator) {
                int validity = opac->tau_validity[cell];
                double tau = opac->tau_sobolev[cell];
                double nupper = sobolev_upper_population_cache[cell];
                double Aul = g_fine_atom->line_A_ul ?
                    g_fine_atom->line_A_ul[l] : NAN;
                LineNetSobolevMaterial material;
                LineNetSobolevRadiation radiation;
                if ((validity != A208_VALID &&
                     validity != A208_EXACT_ZERO) ||
                    !isfinite(tau) || !isfinite(nupper) || nupper < 0.0 ||
                    line_net_sobolev_material(
                        nupper, Aul, nu_l, tau, t_exp,
                        LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK, 1,
                        &material) != 0 ||
                    line_net_sobolev_radiation(
                        &material, line_value[s], line_error[s], nu_l,
                        t_exp, &radiation) != 0) {
                    profile_failed = 1;
                    #ifdef _OPENMP
                    #pragma omp critical(cmf_profile_failure)
                    #endif
                    {
                        if (l < first_profile_bad_line) {
                            first_profile_bad_line = l;
                            first_profile_bad_status = -10;
                        }
                    }
                    continue;
                }
                opac->jbar_line_det[cell] = radiation.jbar;
                opac->jbar_line_det_error_upper[cell] =
                    radiation.jbar_absolute_uncertainty;
                ++sobolev_jbar_cells;
                sobolev_srce_chk_cells += material.srce_chk_applied;
                if (radiation.beta < sobolev_beta_min)
                    sobolev_beta_min = radiation.beta;
                if (radiation.beta > sobolev_beta_max)
                    sobolev_beta_max = radiation.beta;
                if (radiation.jbar_absolute_uncertainty < profile_error_min)
                    profile_error_min =
                        radiation.jbar_absolute_uncertainty;
                if (radiation.jbar_absolute_uncertainty > profile_error_max)
                    profile_error_max =
                        radiation.jbar_absolute_uncertainty;
            } else {
                opac->jbar_line_det[cell] = line_value[s];
                opac->jbar_line_det_error_upper[cell] = line_error[s];
            }
        }
        if (!sobolev_operator) {
            if (profile_report.error_upper_min < profile_error_min)
                profile_error_min = profile_report.error_upper_min;
            if (profile_report.error_upper_max > profile_error_max)
                profile_error_max = profile_report.error_upper_max;
        }
    }
    free(sobolev_upper_population_cache);
    if (sobolev_operator &&
        (sobolev_srce_chk_cells != srce_chk_expected_cells ||
         sobolev_jbar_cells != signed_cells + exact_zero_tau_cells)) {
        profile_failed = 1;
        if (first_profile_bad_line == INT_MAX) {
            first_profile_bad_line = -1;
            first_profile_bad_status = -11;
        }
    }
    if (profile_failed || !isfinite(profile_error_min) ||
        !isfinite(profile_error_max) ||
        (sobolev_operator &&
         (!isfinite(sobolev_beta_min) || !isfinite(sobolev_beta_max) ||
          sobolev_beta_min <= 0.0 || sobolev_beta_max <= 0.0))) {
        fprintf(stderr,
                "[cmf_fine][BLOCKED] reason=LINE_OPERATOR_ERROR_ENVELOPE "
                "operator=%s first_line=%d status=%d "
                "jbar_cells=%llu expected_cells=%llu "
                "srce_chk_applied=%llu srce_chk_expected=%llu "
                "component_error=[%.17g,%.17g] floor=0 clamp=0 "
                "jitter=0 repair=0\n",
                sobolev_operator ? "CMFGEN_NONOVERLAP_SOBOLEV" :
                                   "INIT_SHARED_GAUSSIAN",
                first_profile_bad_line, first_profile_bad_status,
                sobolev_jbar_cells,
                (unsigned long long)(signed_cells + exact_zero_tau_cells),
                sobolev_srce_chk_cells,
                (unsigned long long)srce_chk_expected_cells,
                exact_report.componentwise_error_upper_min,
                exact_report.componentwise_error_upper_max);
        free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
        free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
        free(eta); free(fine_error_upper);
        return -1;
    }
    /* Stage-4 independent probe.  The production line field above has already
     * been profiled and stored.  In this opt-in diagnostic lane reuse the
     * fine-grid buffers for a second exact solve with the line deposit omitted;
     * no production rate or publication array is changed. */
    if (independent_capture) {
        size_t first_cont_bad = SIZE_MAX;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(min:first_cont_bad)
        #endif
        for (size_t cell = 0; cell < fine_cells; ++cell) {
            int shell = (int)(cell / (size_t)NF);
            int bin = (int)(cell % (size_t)NF);
            double ce = fs.chi_es[cell];
            double ca = fs.chi_abs[cell];
            double ct = ce + ca;
            double Bnu = cm_planck(fs.nu[bin], plasma->T_e[shell]);
            if (!(ce >= 0.0) || !isfinite(ce) || !(ca >= 0.0) ||
                !isfinite(ca) || !(ct >= 0.0) || !isfinite(ct) ||
                !(Bnu >= 0.0) || !isfinite(Bnu)) {
                if (cell < first_cont_bad) first_cont_bad = cell;
                continue;
            }
            fs.chi_tot[cell] = ct;
            double numerator = ca * Bnu;
            if (!isfinite(numerator) || numerator < 0.0) {
                if (cell < first_cont_bad) first_cont_bad = cell;
            } else if (ct > 0.0) {
                fs.S_fixed[cell] = numerator / ct;
                if (!isfinite(fs.S_fixed[cell]))
                    if (cell < first_cont_bad) first_cont_bad = cell;
            } else if (numerator == 0.0) {
                fs.S_fixed[cell] = 0.0;
            } else {
                if (cell < first_cont_bad) first_cont_bad = cell;
            }
            fs.J[cell] = Bnu;
        }
        if (first_cont_bad != SIZE_MAX) {
            size_t bad_shell = first_cont_bad / (size_t)NF;
            size_t bad_bin = first_cont_bad % (size_t)NF;
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=INDEPENDENT_CONTINUUM_"
                    "ASSEMBLY shell=%zu bin=%zu nu=%.17g chi_es=%.17g "
                    "chi_abs=%.17g chi_tot=%.17g floor=0 cap=0 clamp=0 "
                    "jitter=0 repair=0\n", bad_shell, bad_bin,
                    fs.nu[bad_bin], fs.chi_es[first_cont_bad],
                    fs.chi_abs[first_cont_bad], fs.chi_tot[first_cont_bad]);
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta); free(fine_error_upper);
            return -1;
        }
        CMFFineOwnerResult continuum_result;
        double continuum_start = cmf_wall_seconds();
        int continuum_rc = cmf_fine_exact_owner_solve(
            NS, NF, dlognu, fs.nu, geo->r_inner, geo->r_outer,
            geo->time_explosion, T_inner, cmf_inner_bb_scale(),
            fs.chi_tot, fs.chi_es, fs.S_fixed, fs.J, fine_error_upper,
            envelope_refinements, n_ali, solve_tol, &continuum_result);
        double continuum_seconds = cmf_wall_seconds() - continuum_start;
        if (continuum_rc != 0) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=INDEPENDENT_CONTINUUM_"
                    "SOLVE status=%s seconds=%.9f floor=0 cap=0 clamp=0 "
                    "jitter=0 repair=0\n",
                    cmf_exact_status_name(continuum_result.exact.status),
                    continuum_seconds);
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta); free(fine_error_upper);
            return -1;
        }
        opac->jbar_line_det_continuum =
            (double *)malloc(line_cells * sizeof(double));
        opac->jbar_line_det_continuum_error_upper =
            (double *)malloc(line_cells * sizeof(double));
        if (!opac->jbar_line_det_continuum ||
            !opac->jbar_line_det_continuum_error_upper) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=INDEPENDENT_CONTINUUM_"
                    "LINE_ALLOCATION cells=%zu floor=0 cap=0 clamp=0 "
                    "jitter=0 repair=0\n", line_cells);
            free(opac->jbar_line_det_continuum);
            free(opac->jbar_line_det_continuum_error_upper);
            opac->jbar_line_det_continuum = NULL;
            opac->jbar_line_det_continuum_error_upper = NULL;
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta); free(fine_error_upper);
            return -1;
        }
        for (size_t cell = 0; cell < line_cells; ++cell) {
            opac->jbar_line_det_continuum[cell] = -1.0;
            opac->jbar_line_det_continuum_error_upper[cell] = -1.0;
        }
        int continuum_profile_failed = 0;
        unsigned long long continuum_jbar_cells = 0;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) reduction(|:continuum_profile_failed) \
            reduction(+:continuum_jbar_cells)
        #endif
        for (int l = 0; l < NL; ++l) {
            double nu_l = opac->line_list_nu[l];
            if (nu_l < line_nu_lo || nu_l > line_nu_hi) continue;
            double line_value[256], line_error[256];
            LineJbarProfileReport cont_profile_report;
            LineJbarProfileStatus cont_profile_status =
                line_jbar_gaussian_discrete_shells(
                    (size_t)NS, (size_t)NF, fs.nu, fs.dnu, fs.J,
                    fine_error_upper, nu_l, vdop,
                    LINE_JBAR_PROFILE_NDOPPLER, line_value, line_error,
                    &cont_profile_report);
            if (cont_profile_status != LINE_JBAR_PROFILE_OK) {
                continuum_profile_failed = 1;
                continue;
            }
            for (int s = 0; s < NS; ++s) {
                size_t cell = (size_t)l * (size_t)NS + (size_t)s;
                if (!(line_value[s] >= 0.0) || !isfinite(line_value[s]) ||
                    !(line_error[s] >= 0.0) || !isfinite(line_error[s])) {
                    continuum_profile_failed = 1;
                    continue;
                }
                opac->jbar_line_det_continuum[cell] = line_value[s];
                opac->jbar_line_det_continuum_error_upper[cell] = line_error[s];
                ++continuum_jbar_cells;
            }
        }
        if (continuum_profile_failed || continuum_jbar_cells == 0) {
            fprintf(stderr,
                    "[cmf_fine][BLOCKED] reason=INDEPENDENT_CONTINUUM_"
                    "PROFILE cells=%llu failed=%d floor=0 cap=0 clamp=0 "
                    "jitter=0 repair=0\n", continuum_jbar_cells,
                    continuum_profile_failed);
            free(opac->jbar_line_det_continuum);
            free(opac->jbar_line_det_continuum_error_upper);
            opac->jbar_line_det_continuum = NULL;
            opac->jbar_line_det_continuum_error_upper = NULL;
            free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);
            free(fs.chi_line);free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
            free(eta); free(fine_error_upper);
            return -1;
        }
        opac->jbar_line_det_continuum_captured = 1;
        fprintf(stderr,
                "[cmf_fine][INDEPENDENT-JCONT] PASS cells=%llu "
                "operator=LINE_FREE_CONTINUUM exact_status=%s iterations=%d "
                "residual=%.17g tolerance=%.17g error_envelope=%d "
                "refinements=%zu seconds=%.9f source=chi_abs*B/(chi_es+chi_abs) "
                "line_material_reused=0 physical_values_modified=0 floor=0 "
                "cap=0 clamp=0 jitter=0 repair=0\n",
                continuum_jbar_cells,
                cmf_exact_status_name(continuum_result.exact.status),
                continuum_result.exact.iterations_used,
                continuum_result.exact.final_max_relative_change,
                continuum_result.exact.tolerance,
                continuum_result.exact.componentwise_error_envelope_verified,
                continuum_result.exact.componentwise_error_refinement_iterations,
                continuum_seconds);
    }
    if (sobolev_operator) {
        srce_chk_cells = (long)sobolev_srce_chk_cells;
        fprintf(stderr,
                "[cmf_fine][SOBOLEV-LINE-OPERATOR] status=PASS "
                "mode=CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0 "
                "continuum_sampling=GAUSSIAN_PROFILE "
                "jbar_cells=%llu raw_negative=%llu mild_negative=%llu "
                "srce_chk_expected=%llu srce_chk_applied=%llu "
                "beta_min=%.17g beta_max=%.17g all_jbar_finite=1 "
                "raw_preserved=1 floor=0 cap=0 clamp=0 jitter=0 repair=0\n",
                sobolev_jbar_cells,
                (unsigned long long)raw_negative_cells,
                (unsigned long long)mild_negative_cells,
                (unsigned long long)srce_chk_expected_cells,
                sobolev_srce_chk_cells,sobolev_beta_min,sobolev_beta_max);
    }
    /* R6 identity evidence.  Publication accepts this field only when these
     * actual producer parameters equal the frozen MC profile definition. */
    opac->jbar_line_det_vdoppler_cms = vdop;
    opac->jbar_line_det_ndoppler = 4.0;

    if (diag) {   /* J_bar_l sanity + S_l/B (b_k proxy) per-iter at warm+cold shells.
                   * Loop over {0(inner warm), 6, NS/2(cold)} so an A/B + iteration
                   * trace shows whether the fluorescence converges (warm) vs the
                   * cold near-singular tail blows up. Also report S_l/J_bar (>>1 =
                   * multi-level fluorescence emission beyond the local field). */
        int diag_shells[3] = { 0, 6, NS/2 };
        for (int di = 0; di < 3; ++di) {
        int st = diag_shells[di]; if (st >= NS) continue; double Te = plasma->T_e[st];
        long nf=0; double jmin=1e300, jmax=-1e300, rsum=0.0; long rn2=0;
        double slsum=0.0; long sln=0, sl_hot=0; double sjmax=0.0; long sj_hi=0;
        double *slbuf = (double*)malloc((size_t)NL * sizeof(double));
        for (int l = 0; l < NL; ++l) {
            double v = opac->jbar_line_det[(size_t)l*NS+st];
            if (v < 0.0) continue;
            ++nf; if (v<jmin) jmin=v; if (v>jmax) jmax=v;
            double B = cm_planck(opac->line_list_nu[l], Te);
            if (B>0) { rsum += v/B; ++rn2;
                double Sl = opac->line_source_S ? opac->line_source_S[(size_t)l*NS+st] : 0.0;
                if (Sl > 0.0) { double r=Sl/B; slsum+=r;
                    if (slbuf) slbuf[sln]=r; ++sln; if (r>10.0) ++sl_hot;
                    double sj = Sl/v; if (sj>sjmax) sjmax=sj; if (sj>3.0) ++sj_hi; } }
        }
        double med=0.0, p90=0.0;
        if (slbuf && sln>0) {   /* sort for median / p90 */
            qsort(slbuf, sln, sizeof(double), cmf_dcmp);
            med = slbuf[sln/2]; p90 = slbuf[(long)(0.9*sln)];
        }
        free(slbuf);
        fprintf(stderr, "[cmf_fine] shell %d Te=%.0f: filled=%ld  Jbar_l in "
            "[%.3e,%.3e]  mean Jbar/B=%.3f | in-window S_l/B: n=%ld mean=%.3e "
            "MEDIAN=%.3f p90=%.3f hot(>10)=%ld(%.1f%%) | S_l/Jbar max=%.2e fluor(>3)=%ld\n",
            st, Te, nf, jmin, jmax,
            (rn2>0)?rsum/rn2:0.0, sln, (sln>0)?slsum/sln:0.0,
            med, p90, sl_hot, (sln>0)?100.0*sl_hot/sln:0.0, sjmax, sj_hi);
        }
    }

    /* Codex test E (LUMINA_CMF_FINE_LINEDUMP=1): per in-window line at mid shell,
     * dump J_fine (line-resolved jbar_line_det) vs J_binned (1000-bin csb->J at the
     * line) vs S_l/B. If high S_l/B correlates with J_binned/J_fine >> 1, the
     * super-thermal is a binned-J contrast-collapse ARTIFACT; if J_binned ~ J_fine,
     * the fluorescence is robust physics independent of the binning. */
    /* SHELL COVERAGE (2026-07-26): the dump used to be hard-wired to the mid
     * shell (NS/2), so the deterministic fine J_bar_l could not be inspected
     * anywhere else -- in particular not in the photospheric shells where the
     * suspected ~10^3x det-jbar collapse lives. LUMINA_CMF_FINE_LINEDUMP_SHELL
     * = <int> selects the shell; unset keeps the historical NS/2 AND the
     * historical file name, so existing consumers are unaffected. When a shell
     * is named explicitly the output goes to cmf_fine_linedump_s<N>.csv instead,
     * so a targeted dump can never be mistaken for the legacy mid-shell one.
     * line_id (the opacity line-list index -- the key that joins this dump to
     * lumina_events_lines.bin, jbar_line[] and tau_sobolev[]) is now the first
     * column; it was missing, which forced consumers to re-derive it from
     * lambda. */
    /* TAU AT THE CONSUMPTION POINT (2026-07-28). Two additions, both forced by
     * the parity36 A/B:
     *  (1) tau_sob column. The line source S_l only means something PAIRED with
     *      the tau the same solver uses: the Sobolev tau carries
     *      stim_corr = d/(1+d) while S_l = (2hv^3/c^2)/d, so S_l*tau is finite
     *      (∝ n_upper) however small d gets. A mismatched pair manufactures
     *      energy without bound — that is what LUMINA_SL_WRITE_SKIPZ did
     *      (FORMAL-CONS 3.484 -> 5973). There was NO observer for tau here, and
     *      the one instrument that looked like it could stand in — beta in
     *      lumina_jbar_dump.csv — cannot: compute_plasma_state rewrites
     *      tau_sobolev from the nebular ion pops at the top of every iteration
     *      and the rate assembly (where that beta is recorded) runs BEFORE the
     *      NLTE tau write inside nlte_solve_all_gpu, so the dumped beta is the
     *      NEBULAR tau in every arm (measured: 89% of Si III betas byte-identical
     *      between LUMINA_NLTE_SKIP_Z=14 and SKIP_Z-empty runs). Read here, the
     *      tau is the one this formal solve actually consumes, with S_l beside it.
     *  (2) shell LIST. The 1000-1300A pathology sits in s40-49, and one shell per
     *      run cost a whole run per shell.
     * LUMINA_CMF_FINE_LINEDUMP_SHELL now takes a comma list ("8,45,49"); each
     * shell gets its own cmf_fine_linedump_s<N>.csv. Unset keeps NS/2 and the
     * historical cmf_fine_linedump.csv name. */
    { const char *e = getenv("LUMINA_CMF_FINE_LINEDUMP");
      if (e && atoi(e)) {
        int shells[64]; int n_sh = 0; int st_explicit = 0;
        { const char *se = getenv("LUMINA_CMF_FINE_LINEDUMP_SHELL");
          if (se && *se) {
            char buf[256]; strncpy(buf, se, sizeof buf - 1); buf[sizeof buf - 1] = 0;
            for (char *tok = strtok(buf, ", \t"); tok && n_sh < 64;
                 tok = strtok(NULL, ", \t")) {
                int sv = atoi(tok);
                if (sv >= 0 && sv < NS) { shells[n_sh++] = sv; st_explicit = 1; }
                else fprintf(stderr, "[cmf_fine] LINEDUMP: shell %s out of range "
                             "[0,%d] — skipped\n", tok, NS-1);
            }
          } }
        if (n_sh == 0) { shells[0] = NS/2; n_sh = 1; }   /* legacy default */
        for (int is = 0; is < n_sh; ++is) {
            int st = shells[is];
            double Te = plasma->T_e[st];
            char dpath[256];
            if (st_explicit) snprintf(dpath, sizeof dpath, "cmf_fine_linedump_s%d.csv", st);
            else             snprintf(dpath, sizeof dpath, "cmf_fine_linedump.csv");
            FILE *df = fopen(dpath, "w");
            if (!df) {
                fprintf(stderr, "[cmf_fine] LINEDUMP: cannot open %s\n", dpath);
                continue;
            }
            fprintf(df, "line_id,shell,lambda_A,nu,J_fine,J_binned,S_l,B,SoverB,"
                        "Jbin_over_Jfine,tau_sob,Sl_times_esc\n");
            for (int l = 0; l < NL; ++l) {
                double jf = opac->jbar_line_det[(size_t)l*NS+st];
                if (jf < 0.0) continue;
                double nu_l = opac->line_list_nu[l];
                int b = (int)floor(log(nu_l/csb->nu_min)/csb->d_log_nu);
                double jb = (b>=0 && b<csb->n_bins) ? csb->J[(size_t)st*csb->n_bins+b] : -1.0;
                double B = cm_planck(nu_l, Te);
                double Sl = opac->line_source_S ? opac->line_source_S[(size_t)l*NS+st] : 0.0;
                double tau = opac->tau_sobolev ? opac->tau_sobolev[(size_t)l*NS+st] : -1.0;
                /* S_l*(1-e^-tau): what this line can put into the emergent beam
                 * from this shell. Bounded for a MATCHED pair however small d is;
                 * unbounded when S_l and tau come from different populations. */
                double esc = (tau > 0.0) ? -expm1(-tau) : 0.0;
                fprintf(df, "%d,%d,%.3f,%.6e,%.6e,%.6e,%.6e,%.6e,%.4e,%.4e,%.6e,%.6e\n",
                    l, st, CM_C/nu_l*1e8, nu_l, jf, jb, Sl, B,
                    (B>0)?Sl/B:0.0, (jf>0&&jb>0)?jb/jf:0.0, tau, Sl*esc);
            }
            fclose(df);
            fprintf(stderr, "[cmf_fine] LINEDUMP wrote %s (shell %d, Te=%.0f K, "
                            "in-window lines only, tau at the consumption point)\n",
                            dpath, st, Te);
        }
      }
    }

    /* FINE-ν PHOTOION (LUMINA_CMF_FINE_PHOTOION): retain the local fine-ν field
     * fs.J + fs.nu (transfer ownership to OpacityState; otherwise freed below) and
     * register it so coupled_photoion_rate_jnu integrates bf rates on the fine grid
     * instead of the binned J. */
    { static int fph = -1;
      if (fph < 0) { const char *e=getenv("LUMINA_CMF_FINE_PHOTOION"); fph=(e&&atoi(e))?1:0; }
      if (fph) {
          free(opac->jnu_fine); free(opac->nu_fine);
          opac->jnu_fine = fs.J;  fs.J  = NULL;   /* transfer (skip free below) */
          opac->nu_fine  = fs.nu; fs.nu = NULL;
          opac->n_fine = NF; opac->nu_lo_fine = nu_lo; opac->dlognu_fine = dlognu;
          coupled_set_fine_jnu(opac->jnu_fine, opac->nu_fine, NF,
                               nu_lo, dlognu, NS);
      }
    }

    free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);free(fs.chi_line);
    free(fs.chi_tot);free(fs.S_fixed);free(fs.J);free(eta);
    free(fine_error_upper);
    opac->jbar_line_det_exact_converged = 1;
    opac->jbar_line_det_exact_iterations = exact_report.iterations_used;
    opac->jbar_line_det_exact_iteration_cap = exact_report.iteration_cap;
    opac->jbar_line_det_exact_residual =
        exact_report.final_max_relative_change;
    opac->jbar_line_det_exact_tolerance = exact_report.tolerance;
    opac->jbar_line_det_exact_absolute_error_bound =
        exact_report.fixed_point_absolute_error_bound;
    opac->jbar_line_det_exact_max_scattering_ratio =
        exact_report.max_scattering_ratio;
    opac->jbar_line_det_error_envelope_verified =
        exact_report.componentwise_error_envelope_verified;
    opac->jbar_line_det_error_refinement_iterations =
        exact_report.componentwise_error_refinement_iterations;
    opac->jbar_line_det_component_error_min =
        exact_report.componentwise_error_upper_min;
    opac->jbar_line_det_component_error_max =
        exact_report.componentwise_error_upper_max;
    opac->jbar_line_det_profile_error_min = profile_error_min;
    opac->jbar_line_det_profile_error_max = profile_error_max;
    opac->jbar_line_det_grid_n_bins = NF;
    opac->jbar_line_det_grid_nu_min = nu_lo * exp(0.5 * dlognu);
    opac->jbar_line_det_grid_nu_max =
        nu_lo * exp(((double)NF - 0.5) * dlognu);
    opac->jbar_line_det_operator = (int)line_operator;
    return 0;
}

/* ===================================================================
 * CONFIRMATION-LADDER T1: controlled single-line P-Cygni self-test.
 * Synthetic homologous ejecta + BB core + ONE pure-scatter resonance line.
 * Feeds the SAME cmfgen_fine_emergent_obs (whatever obs path the env selects:
 * scatter / SEI / faithful) and writes the profile. KNOWN ANSWER:
 *   - pure scatter ⇒ net equivalent width = 0 (exact photon conservation)
 *   - Castor/SEI P-Cygni shape (blue absorption, red emission)
 *   - tau_S→∞ ⇒ blue-edge flux → 0
 * Env: LUMINA_CMF_OBS_SELFTEST=1; LUMINA_CMF_SELFTEST_TAUS=tau_S (default 100). */
/* ===================================================================
 * CONFIRMATION-LADDER Stage N: NLTE level populations / line source S_l.
 *   N1 (two-level atom): analytic S_l = (J̄ + εB)/(1+ε), ε=(C_ul/A_ul)(1−e^{−hν/kT}).
 *       Solve the rate equation, verify num==ana AND limits (ε→0:S_l→J̄; ε→∞:S_l→B).
 *   N4 (three-level UV pump): ground(0)/mid(1)/upper(2). UV resonance pump 0→2 (2500A),
 *       fluorescence decay 2→1 (5000A), optical 1→0 (5000A). Feed the UV pump field
 *       either FREQ-RESOLVED (sees hot photospheric W·B(T_rad)) or BINNED (collapses to
 *       local-thermal B(T_e), the binned-J grey defect). KNOWN ANSWER: freq-resolved
 *       pumps b_2>1 → elevated optical S_l(2→1) = fluorescence; binned gives b_2≈1, none.
 *       This is the controlled capstone of the fluorescence saga. Env LUMINA_CMF_NLTE_SELFTEST=n1|n4. */
static void cmf_nlte_solve3(const double E[3], const double g[3],
                            const double A[3][3], const double Jbar[3][3],
                            double qcoef, double ne, double Te, double n_out[3])
{
    const double H=6.62607015e-27, C=2.99792458e10, K=1.380649e-16;
    double R[3][3]; for (int i=0;i<3;++i) for (int j=0;j<3;++j) R[i][j]=0.0;
    for (int lo=0; lo<3; ++lo) for (int up=lo+1; up<3; ++up) {
        if (A[up][lo]<=0.0) continue;
        double dE=E[up]-E[lo], nu=dE/H;
        double Bul=A[up][lo]*C*C/(2.0*H*nu*nu*nu);
        double Blu=(g[up]/g[lo])*Bul;
        double qul=qcoef*ne;                          /* de-excit rate s^-1 */
        double qlu=qul*(g[up]/g[lo])*exp(-dE/(K*Te));
        double J=Jbar[lo][up];
        R[up][lo]=A[up][lo]+Bul*J+qul;                /* up->lo (down) */
        R[lo][up]=Blu*J+qlu;                          /* lo->up (up)   */
    }
    /* statistical equilibrium M n = 0, replace row 0 with normalization */
    double M[3][3];
    for (int i=0;i<3;++i){ double out=0; for(int j=0;j<3;++j) if(j!=i) out+=R[i][j];
        for(int j=0;j<3;++j) M[i][j]=(i==j)?-out:R[j][i]; }
    for (int j=0;j<3;++j) M[0][j]=1.0; double rhs[3]={1.0,0.0,0.0};
    /* 3x3 Gaussian elimination */
    for (int c=0;c<3;++c){ int p=c; for(int r=c+1;r<3;++r) if(fabs(M[r][c])>fabs(M[p][c])) p=r;
        for(int j=0;j<3;++j){double t=M[c][j];M[c][j]=M[p][j];M[p][j]=t;} double t=rhs[c];rhs[c]=rhs[p];rhs[p]=t;
        for(int r=0;r<3;++r) if(r!=c){double f=M[r][c]/M[c][c]; for(int j=0;j<3;++j) M[r][j]-=f*M[c][j]; rhs[r]-=f*rhs[c];}}
    for (int i=0;i<3;++i) n_out[i]=rhs[i]/M[i][i];
}

/* ===================================================================
 * CONFIRMATION-LADDER Stage P: plasma (ionization + energy balance).
 *   P1 (Saha LTE limit): 2-ion ground system + e-. Collisional ion/3-body recomb
 *       (detailed-balance → Saha) + radiative recomb + photoionization (non-thermal
 *       field). KNOWN: ne→∞ ⇒ (n_{i+1}·n_e/n_i) → S_Saha(Te) DESPITE the photoion field.
 *   P2 (gray radiative equilibrium): J_ν = W·B_ν(T_rad). KNOWN: ∫(J−B(Te))dν=0 ⇒
 *       Te = W^{1/4}·T_rad. Tests the energy-balance root-finder. */
void cmf_plasma_selftest(const char *mode)
{
    const double H=6.62607015e-27, C=2.99792458e10, K=1.380649e-16;
    const double ME=9.1093837e-28, EV=1.602176634e-12, PI=3.14159265358979;
    if (!mode || strcmp(mode,"p1")==0) {
        double chi=13.6*EV, gi=9, gip1=4, Te=8000.0, Trad=10000.0, Wd=0.5;
        double S=2.0*(gip1/gi)*pow(2*PI*ME*K*Te/(H*H),1.5)*exp(-chi/(K*Te)); /* Saha RHS cm^-3 */
        double Cion=1e-8*sqrt(Te)*exp(-chi/(K*Te));            /* collisional ion coeff */
        double alpha=2e-13*pow(Te/1e4,-0.7);                   /* radiative recomb coeff */
        double Gamma=Wd*1e-10*exp(-chi/(K*Trad));              /* photoion rate (hot field, toy) */
        printf("=== CONFIRMATION-LADDER P1 (Saha LTE limit), Te=%.0f Trad=%.0f, photoion field ON ===\n",Te,Trad);
        printf("  %-10s %-13s %-13s %-10s\n","ne","(n+·ne/n)","S_Saha(Te)","ratio");
        for (double ne=1e6; ne<=1e16; ne*=100){
            double x=(Cion*ne+Gamma)/(alpha*ne + Cion*ne*ne/S);  /* n_{i+1}/n_i */
            printf("  %-10.0e %-13.4e %-13.4e %-10.4f\n",ne,x*ne,S,x*ne/S);
        }
        printf("  KNOWN: ne→∞ ⇒ (n+·ne/n)→S_Saha (ratio→1) DESPITE photoion; low ne ratio>1 (field over-ionizes, NLTE).\n");
        printf("  ⟹ P1 PASS if ratio→1 at high ne (ionization rate network has correct Saha/LTE limit).\n");
    }
    if (!mode || strcmp(mode,"p2")==0) {
        double Trad=10000.0;
        int NF=400; double nu_lo=C/(30000e-8), nu_hi=C/(1000e-8), dln=log(nu_hi/nu_lo)/NF;
        printf("=== CONFIRMATION-LADDER P2 (gray radiative equilibrium), Trad=%.0f ===\n",Trad);
        printf("  %-7s %-11s %-13s %-9s\n","W","Te(solve)","Te=W^.25·Tr","ratio");
        double Wlist[4]={1.0,0.5,0.25,0.1};
        for (int w=0; w<4; ++w){ double W=Wlist[w];
            double lo=1000, hi=30000;
            for (int it=0; it<60; ++it){ double Te=0.5*(lo+hi), r=0;
                for (int i=0;i<NF;++i){ double nu=nu_lo*exp((i+0.5)*dln), dnu=nu*dln;
                    r += (W*cm_planck(nu,Trad)-cm_planck(nu,Te))*dnu; }
                if (r>0) lo=Te; else hi=Te; }
            double Te=0.5*(lo+hi), Tek=pow(W,0.25)*Trad;
            printf("  %-7.2f %-11.1f %-13.1f %-9.4f\n",W,Te,Tek,Te/Tek);
        }
        printf("  KNOWN: gray radeq ∫(W·B(Trad)−B(Te))dν=0 ⇒ Te=W^{1/4}·Trad. ratio→1 ⇒ energy-balance root-finder correct.\n");
    }
}

void cmf_nlte_selftest(const char *mode)
{
    const double H=6.62607015e-27, C=2.99792458e10, K=1.380649e-16;
    double Te=8000.0, Trad=10000.0, W=0.5;
    if (!mode || strcmp(mode,"n1")==0) {
        double gl=2,gu=4, lam=5000e-8, nu=C/lam, dE=H*nu, Aul=1e8;
        double twohnu3c2=2*H*nu*nu*nu/(C*C), Bnu=cm_planck(nu,Te), Jbar=W*Bnu;
        printf("=== CONFIRMATION-LADDER N1 (two-level S_l), Te=%.0f lam=5000A, Jbar=W·B=%.2fB ===\n",Te,W);
        printf("  %-9s %-10s %-11s %-11s %-9s\n","ne","eps","S_l/B(num)","S_l/B(ana)","match");
        for (double ne=1e5; ne<=1e13; ne*=100){
            double Bul=Aul*C*C/(2*H*nu*nu*nu), Blu=(gu/gl)*Bul;
            double qul=1e-7*ne, qlu=qul*(gu/gl)*exp(-dE/(K*Te));
            double Rlu=Blu*Jbar+qlu, Rul=Aul+Bul*Jbar+qul, ratio=Rlu/Rul; /* nu/nl */
            double Sl=twohnu3c2/((gu/gl)/ratio-1.0);
            double eps=(qul/Aul)*(1-exp(-dE/(K*Te))), Sl_ana=(Jbar+eps*Bnu)/(1+eps);
            printf("  %-9.0e %-10.3e %-11.4f %-11.4f %-9s\n",ne,eps,Sl/Bnu,Sl_ana/Bnu,
                   (fabs(Sl-Sl_ana)/Sl_ana<1e-6)?"OK":"FAIL");
        }
        printf("  KNOWN: ε→0 S_l→Jbar(0.50B); ε→∞ S_l→B(1.0); num==ana ⇒ 2-level machinery PASS\n");
    }
    if (!mode || strcmp(mode,"n4")==0) {
        /* 3 levels: 0 ground, 1 mid(5000A from g), 2 upper(2500A from g) */
        double lam10=5000e-8, lam02=2500e-8;
        double E[3]={0.0, H*C/lam10, H*C/lam02}, g[3]={2,4,2};
        double lam21=H*C/(E[2]-E[1]);  /* fluorescence line 2->1 */
        double A[3][3]={{0}}; A[2][0]=1e8; A[2][1]=1e7; A[1][0]=1e7;  /* UV pump / fluor / optical */
        double nu02=(E[2]-E[0])/H, nu21=(E[2]-E[1])/H, nu10=(E[1]-E[0])/H;
        double ne=1e8, qcoef=1e-8;   /* nebular: weak collisions */
        printf("=== CONFIRMATION-LADDER N4 (3-level UV pump→fluorescence), Te=%.0f Trad=%.0f ===\n",Te,Trad);
        printf("  pump 0→2=%.0fA, fluor 2→1=%.0fA, optical 1→0=%.0fA; ne=%.0e\n",lam02*1e8,lam21*1e8,lam10*1e8,ne);
        for (int cs=0; cs<2; ++cs){
            int freq=(cs==0);
            double Jbar[3][3]={{0}};
            /* UV pump 0↔2: FREQ sees hot photospheric W·B(Trad); BINNED collapses to local B(Te) */
            Jbar[0][2]=Jbar[2][0]= freq ? W*cm_planck(nu02,Trad) : cm_planck(nu02,Te);
            Jbar[1][2]=Jbar[2][1]= W*cm_planck(nu21,Te);     /* optical: both ~W·B(Te) */
            Jbar[0][1]=Jbar[1][0]= W*cm_planck(nu10,Te);
            double n[3]; cmf_nlte_solve3(E,g,A,Jbar,qcoef,ne,Te,n);
            /* LTE pops at Te for departure b_i = n_i/n_i^LTE (normalized) */
            double zl=0; double nlte_lte[3]; for(int i=0;i<3;++i){nlte_lte[i]=g[i]*exp(-E[i]/(K*Te)); zl+=nlte_lte[i];}
            for(int i=0;i<3;++i) nlte_lte[i]/=zl;
            double ntot=n[0]+n[1]+n[2];
            double b2=(n[2]/ntot)/nlte_lte[2], b1=(n[1]/ntot)/nlte_lte[1];
            /* S_l of fluorescence line 2→1 */
            double twohnu3c2=2*H*nu21*nu21*nu21/(C*C);
            double Sl21=twohnu3c2/((g[2]/g[1])*(n[1]/n[2])-1.0);
            double B21=cm_planck(nu21,Te);
            printf("  [%s] UV J̄_02/B(Te)=%6.1f  b_2=%7.3f  b_1=%6.3f  S_l(2→1)/B(Te)=%7.3f %s\n",
                   freq?"FREQ ":"BIN  ", Jbar[0][2]/cm_planck(nu02,Te), b2, b1, Sl21/B21,
                   freq?"":" (binned baseline)");
        }
        printf("  KNOWN: FREQ pump (UV J̄≫B) → b_2>1 + S_l(2→1)>1 = FLUORESCENCE; BIN → b_2≈1, S_l≈1 (none).\n");
        printf("  ⟹ if FREQ shows fluorescence and BIN does not, binned-J STRUCTURALLY cannot pump (saga capstone).\n");
    }
    if (!mode || strcmp(mode,"n2")==0) {
        /* LTE limit: high collisions must thermalize pops → Boltzmann@Te DESPITE a
         * non-thermal (10×B UV) field. Same 3-level atom as N4, ramp n_e. */
        double lam10=5000e-8, lam02=2500e-8;
        double E[3]={0.0, H*C/lam10, H*C/lam02}, g[3]={2,4,2};
        double A[3][3]={{0}}; A[2][0]=1e8; A[2][1]=1e7; A[1][0]=1e7;
        double nu21=(E[2]-E[1])/H;
        double Jbar[3][3]={{0}};
        Jbar[0][2]=Jbar[2][0]=10.0*cm_planck((E[2]-E[0])/H,Te);  /* hot UV pump (non-thermal) */
        Jbar[1][2]=Jbar[2][1]=W*cm_planck(nu21,Te);
        Jbar[0][1]=Jbar[1][0]=W*cm_planck((E[1]-E[0])/H,Te);
        printf("=== CONFIRMATION-LADDER N2 (LTE limit: high collisions→Boltzmann), Te=%.0f, field NON-thermal(UV 10×B) ===\n",Te);
        printf("  %-10s %-9s %-9s %-9s %-12s\n","ne","b_0","b_1","b_2","S_l(2→1)/B");
        for (double ne=1e8; ne<=1e18; ne*=1000){
            double n[3]; cmf_nlte_solve3(E,g,A,Jbar,1e-7,ne,Te,n);
            double zl=0,nl[3]; for(int i=0;i<3;++i){nl[i]=g[i]*exp(-E[i]/(K*Te));zl+=nl[i];} for(int i=0;i<3;++i)nl[i]/=zl;
            double ntot=n[0]+n[1]+n[2];
            double twohnu3c2=2*H*nu21*nu21*nu21/(C*C);
            double Sl21=twohnu3c2/((g[2]/g[1])*(n[1]/n[2])-1.0), B21=cm_planck(nu21,Te);
            printf("  %-10.0e %-9.3f %-9.3f %-9.3f %-12.3f\n",ne,(n[0]/ntot)/nl[0],(n[1]/ntot)/nl[1],(n[2]/ntot)/nl[2],Sl21/B21);
        }
        printf("  KNOWN: ne→∞ (collisions dominate) ⇒ b_i→1 ALL levels (Boltzmann@Te), S_l→1, DESPITE the 10×B UV pump.\n");
        printf("  ⟹ N2 PASS if b_i→1 at high ne (rate matrix has correct LTE limit; collisions wash out non-thermal field).\n");
    }
    if (!mode || strcmp(mode,"n5")==0 || strcmp(mode,"cond")==0) {
        /* CONDITIONING (ARTIS recipe, nltepop.cc:733): an NLTE-like detailed-balance
         * rate matrix whose equilibrium x_true spans many orders → cond≫1/eps → raw LU
         * garbage. ARTIS fix = iterative ROW-COL equilibration (f=√(colL2/rowL2), 10×)
         * before LU. KNOWN ANSWER = x_true (Boltzmann). Raw vs equilibrated vs x_true. */
        enum { NN=10 };
        printf("=== CONFIRMATION-LADDER N5 (NLTE matrix conditioning, ARTIS row-col equilibration) ===\n");
        printf("  detailed-balance rate matrix, equilibrium x_true=Boltzmann; raw LU vs ARTIS-equilibrated LU vs x_true\n");
        printf("  %-8s %-12s %-13s %-13s %-6s\n","decade","span","RAW err","EQ(ARTIS) err","PASS");
        double decades[5]={28,56,84,140,200};   /* x_true span in orders of magnitude */
        for (int sp=0; sp<5; ++sp){
        double perlev = decades[sp]*log(10.0)/(NN-1);   /* exp(-perlev*i) gives the target span */
        double xt[NN]; for(int i=0;i<NN;++i) xt[i]=exp(-perlev*i);
        double M0[NN*NN]; for(int i=0;i<NN*NN;++i) M0[i]=0.0;
        for(int i=0;i<NN;++i){ double dout=0;
            for(int j=0;j<NN;++j) if(j!=i){
                double Rji=(j<i)?1.0:xt[i]/xt[j];   /* rate j→i: up=1, down=xt[i]/xt[j] */
                M0[i*NN+j]=Rji;
                double Rik=(i<j)?1.0:xt[j]/xt[i];   /* rate i→j (for diagonal) */
                dout+=Rik;
            }
            M0[i*NN+i]=-dout;
        }
        double sumxt=0; for(int i=0;i<NN;++i) sumxt+=xt[i];
        for(int j=0;j<NN;++j) M0[0*NN+j]=1.0;       /* row 0 = normalization Σx=sumxt */
        /* local LU solve (partial pivot) on row-major A[NN*NN], b[NN] -> x[NN] */
        #define LUSOLVE(Asrc,bsrc,xout) do{ double A[NN*NN],b[NN]; \
            for(int _i=0;_i<NN*NN;_i++)A[_i]=(Asrc)[_i]; for(int _i=0;_i<NN;_i++)b[_i]=(bsrc)[_i]; \
            for(int c=0;c<NN;c++){ int p=c; for(int r=c+1;r<NN;r++) if(fabs(A[r*NN+c])>fabs(A[p*NN+c]))p=r; \
              for(int k=0;k<NN;k++){double t=A[c*NN+k];A[c*NN+k]=A[p*NN+k];A[p*NN+k]=t;} double tb=b[c];b[c]=b[p];b[p]=tb; \
              for(int r=0;r<NN;r++) if(r!=c){double f=A[r*NN+c]/A[c*NN+c]; for(int k=0;k<NN;k++)A[r*NN+k]-=f*A[c*NN+k]; b[r]-=f*b[c];}} \
            for(int _i=0;_i<NN;_i++)(xout)[_i]=b[_i]/A[_i*NN+_i]; }while(0)
        double bvec[NN]={0}; bvec[0]=sumxt;
        double xraw[NN]; LUSOLVE(M0,bvec,xraw);
        /* ARTIS iterative row-col equilibration */
        double Meq[NN*NN],beq[NN],cscale[NN]; for(int i=0;i<NN*NN;i++)Meq[i]=M0[i];
        for(int i=0;i<NN;i++){beq[i]=bvec[i];cscale[i]=1.0;}
        for(int it=0;it<10;it++){ int changed=0;
            for(int i=0;i<NN;i++){ double rn=0,cn=0;
                for(int j=0;j<NN;j++){rn+=Meq[i*NN+j]*Meq[i*NN+j]; cn+=Meq[j*NN+i]*Meq[j*NN+i];}
                rn=sqrt(rn);cn=sqrt(cn); if(rn==0||cn==0)continue;
                double f=sqrt(cn/rn); if(fabs(f-1.0)<1e-3)continue; changed=1;
                for(int j=0;j<NN;j++)Meq[i*NN+j]*=f; beq[i]*=f;          /* row i ×f */
                for(int j=0;j<NN;j++)Meq[j*NN+i]/=f; cscale[i]/=f;       /* col i ÷f */
            }
            if(!changed)break;
        }
        double yeq[NN],xeq[NN]; LUSOLVE(Meq,beq,yeq); for(int i=0;i<NN;i++)xeq[i]=cscale[i]*yeq[i];
        #undef LUSOLVE
        double eraw=0,eeq=0; for(int i=0;i<NN;i++){
            eraw=fmax(eraw,fabs(xraw[i]-xt[i])/fabs(xt[i])); eeq=fmax(eeq,fabs(xeq[i]-xt[i])/fabs(xt[i])); }
        printf("  %-8.0f %.0e..%-6.0e %-13.3e %-13.3e %-6s\n", decades[sp], xt[0], xt[NN-1], eraw, eeq,
               (eeq < 1e-3 && eeq < eraw*0.01) ? "EQ✓" : (eraw<1e-3?"both ok":"—"));
        }
        printf("  KNOWN: equilibrated recovers x_true (err≪1) even as span→200 orders; raw LU degrades.\n");
        printf("  ⟹ N5 PASS = ARTIS row-col equilibration holds where raw LU fails (the conditioning fix).\n");
    }
    if (!mode || strcmp(mode,"n6")==0) {
        /* SUPER-LEVELS (ARTIS recipe, nltepop.cc:1411 superlevel_boltzmann): solve an
         * N-level atom (a) FULL explicit vs (b) K explicit + high levels lumped into ONE
         * Boltzmann(T_exc) super-level (s_renorm-weighted rates). KNOWN ANSWER = full
         * solution. Super-level should recover the low-level pops AND cut the matrix
         * dimension N→K+1. Validates the span-reduction conditioning fix before plasma.c. */
        enum { NL=24 };
        const double EV=1.602176634e-12;
        int Kx=6;  /* explicit low levels 0..Kx-1; Kx..NL-1 → super-level */
        double Eg[NL], gg[NL];
        for(int i=0;i<NL;++i){ Eg[i]=i*0.40*EV; gg[i]=2.0*i+1.0; }   /* ladder, g=2i+1 */
        double ne=1e9, qc=1e-8, Te_=8000.0, Texc=8000.0;
        /* radiative+collisional rates between all pairs (toy: A∝1/(E_u-E_l)^3-ish, here A=1e7) */
        /* --- build full NL×NL statistical-equilibrium matrix --- */
        double Jbar=0.5;  /* W; field = W·B(nu,Trad=10000) per pair */
        double Tr=10000.0;
        #define RATE_UP(lo,hi)  ( atom_blu(lo,hi)*Jbar*cm_planck((Eg[hi]-Eg[lo])/H,Tr) + qc*ne*(gg[hi]/gg[lo])*exp(-(Eg[hi]-Eg[lo])/(K*Te_)) )
        #define RATE_DN(lo,hi)  ( atom_aul(lo,hi) + atom_blu(lo,hi)*(gg[lo]/gg[hi])*Jbar*cm_planck((Eg[hi]-Eg[lo])/H,Tr) + qc*ne )
        /* helper lambdas via macros referencing A_ul=1e7, B_lu from A */
        double Aul=1e7;
        #define atom_aul(lo,hi) (Aul)
        #define atom_blu(lo,hi) (Aul*C*C/(2*H*pow((Eg[hi]-Eg[lo])/H,3.0))*(gg[hi]/gg[lo]))
        /* FULL solve */
        static double Mf[NL*NL]; for(int i=0;i<NL*NL;++i)Mf[i]=0;
        for(int lo=0;lo<NL;++lo)for(int hi=lo+1;hi<NL;++hi){
            double ru=RATE_UP(lo,hi), rd=RATE_DN(lo,hi);
            Mf[hi*NL+lo]+=ru; Mf[lo*NL+hi]+=rd;          /* into hi from lo; into lo from hi */
            Mf[lo*NL+lo]-=ru; Mf[hi*NL+hi]-=rd;          /* out of lo (up), out of hi (down) */
        }
        for(int j=0;j<NL;++j)Mf[0*NL+j]=1.0;             /* row0 = normalization */
        #define SOLVEGEN(N,Asrc,bsrc,xout) do{ static double A[NL*NL],b[NL]; \
            for(int _i=0;_i<(N)*(N);_i++)A[_i]=(Asrc)[_i]; for(int _i=0;_i<(N);_i++)b[_i]=(bsrc)[_i]; \
            for(int c=0;c<(N);c++){int p=c;for(int r=c+1;r<(N);r++)if(fabs(A[r*(N)+c])>fabs(A[p*(N)+c]))p=r; \
              for(int k=0;k<(N);k++){double t=A[c*(N)+k];A[c*(N)+k]=A[p*(N)+k];A[p*(N)+k]=t;}double tb=b[c];b[c]=b[p];b[p]=tb; \
              for(int r=0;r<(N);r++)if(r!=c){double f=A[r*(N)+c]/A[c*(N)+c];for(int k=0;k<(N);k++)A[r*(N)+k]-=f*A[c*(N)+k];b[r]-=f*b[c];}} \
            for(int _i=0;_i<(N);_i++)(xout)[_i]=b[_i]/A[_i*(N)+_i]; }while(0)
        double bf[NL]={0}; bf[0]=1.0; double nf[NL]; SOLVEGEN(NL,Mf,bf,nf);
        /* --- SUPER-LEVEL solve: dim = Kx+1 (levels 0..Kx-1 explicit, index Kx = super) --- */
        double zsl=0, sb[NL]; for(int i=Kx;i<NL;++i){ sb[i]=gg[i]*exp(-(Eg[i]-Eg[Kx])/(K*Texc)); zsl+=sb[i]; }
        double sren[NL]; for(int i=Kx;i<NL;++i) sren[i]=sb[i]/zsl;     /* Boltzmann fraction within SL */
        int DS=Kx+1; static double Ms[NL*NL]; for(int i=0;i<DS*DS;++i)Ms[i]=0;
        #define IDX(i) ((i)<Kx?(i):Kx)                    /* map physical level → super index */
        for(int lo=0;lo<NL;++lo)for(int hi=lo+1;hi<NL;++hi){
            double ru=RATE_UP(lo,hi), rd=RATE_DN(lo,hi);
            double wlo=(lo<Kx)?1.0:sren[lo], whi=(hi<Kx)?1.0:sren[hi];  /* SL members enter ×Boltzmann frac */
            int a=IDX(lo), c=IDX(hi);
            Ms[c*DS+a]+=ru*wlo; Ms[a*DS+c]+=rd*whi;
            Ms[a*DS+a]-=ru*wlo; Ms[c*DS+c]-=rd*whi;
        }
        for(int j=0;j<DS;++j)Ms[0*DS+j]=1.0; double bs[NL]={0}; bs[0]=1.0; double ns[NL]; SOLVEGEN(DS,Ms,bs,ns);
        #undef SOLVEGEN
        #undef RATE_UP
        #undef RATE_DN
        #undef atom_aul
        #undef atom_blu
        #undef IDX
        /* compare low-level pops (normalize full so Σ=1; super already Σ=1 over DS) */
        double sumf=0; for(int i=0;i<NL;++i)sumf+=nf[i]; for(int i=0;i<NL;++i)nf[i]/=sumf;
        double slfull=0; for(int i=Kx;i<NL;++i)slfull+=nf[i];
        printf("=== CONFIRMATION-LADDER N6 (super-levels, ARTIS recipe), NL=%d, K_explicit=%d → dim %d→%d ===\n",NL,Kx,NL,DS);
        printf("  %-5s %-13s %-13s %-8s\n","lev","n_full","n_super","rel.err");
        double maxe=0; for(int i=0;i<Kx;++i){ double e=fabs(ns[i]-nf[i])/fmax(nf[i],1e-30); maxe=fmax(maxe,e);
            printf("  %-5d %-13.4e %-13.4e %-8.1e\n",i,nf[i],ns[i],e); }
        double esl=fabs(ns[Kx]-slfull)/fmax(slfull,1e-30);
        printf("  SL    %-13.4e %-13.4e %-8.1e (super-level total vs Σ full high)\n",slfull,ns[Kx],esl);
        printf("  max low-level rel.err = %.2e\n",maxe);
        printf("  KNOWN: super-level (Boltzmann high-E lump) recovers explicit low-level pops; dim %d→%d.\n",NL,DS);
        printf("  ⟹ N6 PASS if low-level err≪1 (super-level approx valid) → span-reduction conditioning fix works.\n");
    }
}

/* ===================================================================
 * CONFIRMATION-LADDER Stage F: cmf_solve_J (comoving J̄ formal solve) on
 * known-answer cases. Env LUMINA_CMF_FSOLVE_SELFTEST = "absorb"|"dilute".
 *   F1 absorb: isothermal, pure absorption (chi_abs only), thick, S_fixed=B(T)
 *              → J → B(T) in the interior (thermalization). Known: J/B → 1.
 *   F3 dilute: thin e-scatter halo (tau_es~0.3) + BB core, chi_abs=0, S_fixed=0
 *              → J → W(r)·B(T_inner) at outer shells. Known: J/(W·B) → 1. */
void cmf_fsolve_selftest(const char *mode)
{
    int NS=40, NF=200; { const char *e=getenv("LUMINA_CMF_SELFTEST_NS"); if(e) NS=atoi(e); }
    double t_exp=86400.0, Tinner=10000.0;
    double vph=5.0e8, vmax=2.5e9;
    int absorb = (mode && strcmp(mode,"absorb")==0);
    Geometry geo; memset(&geo,0,sizeof geo); geo.n_shells=NS; geo.time_explosion=t_exp;
    geo.r_inner=malloc(NS*sizeof(double)); geo.r_outer=malloc(NS*sizeof(double));
    geo.v_inner=malloc(NS*sizeof(double)); geo.v_outer=malloc(NS*sizeof(double));
    for (int s=0;s<NS;++s){ double v0=vph+(vmax-vph)*s/(double)NS,v1=vph+(vmax-vph)*(s+1)/(double)NS;
        geo.v_inner[s]=v0;geo.v_outer[s]=v1;geo.r_inner[s]=v0*t_exp;geo.r_outer[s]=v1*t_exp; }
    double nu_min=CM_C/(6000e-8), nu_max=CM_C/(4000e-8), dlognu=log(nu_max/nu_min)/NF;
    CMFGENState fs; memset(&fs,0,sizeof fs);
    fs.n_shells=NS;fs.n_bins=NF;fs.nu_min=nu_min;fs.nu_max=nu_max;fs.d_log_nu=dlognu;
    fs.nu=malloc(NF*sizeof(double));fs.dnu=malloc(NF*sizeof(double));
    fs.chi_es=calloc((size_t)NS*NF,sizeof(double));fs.chi_abs=calloc((size_t)NS*NF,sizeof(double));
    fs.chi_line=calloc((size_t)NS*NF,sizeof(double));fs.chi_tot=calloc((size_t)NS*NF,sizeof(double));
    fs.S_fixed=calloc((size_t)NS*NF,sizeof(double));fs.J=calloc((size_t)NS*NF,sizeof(double));
    for (int i=0;i<NF;++i){ fs.nu[i]=nu_min*exp((i+0.5)*dlognu); fs.dnu[i]=fs.nu[i]*dlognu; }
    double dr=geo.r_outer[NS-1]-geo.r_inner[0];
    for (int s=0;s<NS;++s) for (int i=0;i<NF;++i){ size_t idx=(size_t)s*NF+i;
        double B=cm_planck(fs.nu[i],Tinner);
        if (absorb){ double chia=100.0/dr; fs.chi_abs[idx]=chia; fs.chi_tot[idx]=chia;
                     fs.S_fixed[idx]=B; fs.J[idx]=0.5*B; }            /* thick, S=B → J→B */
        else { double taues=0.3; { const char *e=getenv("LUMINA_CMF_FSOLVE_TAUES"); if(e) taues=atof(e); }
               double ces=taues/dr; fs.chi_es[idx]=ces; fs.chi_tot[idx]=ces;
               fs.S_fixed[idx]=0.0; fs.J[idx]=0.5*B; }                /* thin scatter+core */
    }
    int n_ali=24; { const char *e=getenv("LUMINA_CMFGEN_ALI_ITER"); if(e) n_ali=atoi(e); }
    cmf_solve_J(&fs,&geo,Tinner,n_ali);
    int ic=NF/2;  /* mid frequency */
    printf("=== CONFIRMATION-LADDER F (%s), cmf_solve_J ===\n", absorb?"F1 absorb J→B":"F3 dilute J→W·B");
    for (int s=0;s<NS;s+=(absorb?NS/8:NS/8)){
        double B=cm_planck(fs.nu[ic],Tinner), J=fs.J[(size_t)s*NF+ic];
        double r=0.5*(geo.r_inner[s]+geo.r_outer[s]);
        double a=1.0-(geo.r_inner[0]*geo.r_inner[0])/(r*r), W=0.5*(1.0-sqrt(a>0?a:0));
        if (absorb) printf("  s=%2d  J/B=%.4f  (known: →1.0 thick interior)\n", s, J/B);
        else        printf("  s=%2d  J/(W·B)=%.4f  W=%.4f  (known: →1.0 dilution)\n", s, (W>0)?J/(W*B):0.0, W);
    }
    free(geo.r_inner);free(geo.r_outer);free(geo.v_inner);free(geo.v_outer);
    free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);free(fs.chi_line);
    free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
}

/* ============================================================ */
/* GPU cmf_solve_J self-check (model-free; dispatched pre-model from main under
 * env LUMINA_CMF_SOLVE_SELFTEST). Builds a representative fine grid -- a thin
 * e-scatter halo + warm BB core (continuum) plus, unless mode=="cont", a strong
 * Gaussian SCATTERING line deposited into chi_es (the producer's UV-pump pattern,
 * which stresses the blue->red advection across the line). Then calls cmf_solve_J,
 * which -- with LUMINA_CMF_SOLVE_GPU=2 -- runs BOTH the CPU and GPU solvers from
 * the same input and prints the max relative J difference + ALI iter counts.
 *   LUMINA_CMF_SOLVE_SELFTEST_NS / _NF / _TAUS tune the grid. */
void cmf_solve_gpu_selftest(const char *mode)
{
    int NS=40, NF=400;
    { const char *e=getenv("LUMINA_CMF_SOLVE_SELFTEST_NS"); if(e) NS=atoi(e); }
    { const char *e=getenv("LUMINA_CMF_SOLVE_SELFTEST_NF"); if(e) NF=atoi(e); }
    int with_line = !(mode && strcmp(mode,"cont")==0);
    double tauS=50.0; { const char *e=getenv("LUMINA_CMF_SOLVE_SELFTEST_TAUS"); if(e) tauS=atof(e); }
    double t_exp=86400.0, Tinner=10000.0, vph=5.0e8, vmax=2.5e9;
    Geometry geo; memset(&geo,0,sizeof geo); geo.n_shells=NS; geo.time_explosion=t_exp;
    geo.r_inner=malloc(NS*sizeof(double)); geo.r_outer=malloc(NS*sizeof(double));
    geo.v_inner=malloc(NS*sizeof(double)); geo.v_outer=malloc(NS*sizeof(double));
    for (int s=0;s<NS;++s){ double v0=vph+(vmax-vph)*s/(double)NS,v1=vph+(vmax-vph)*(s+1)/(double)NS;
        geo.v_inner[s]=v0;geo.v_outer[s]=v1;geo.r_inner[s]=v0*t_exp;geo.r_outer[s]=v1*t_exp; }
    /* fine uniform-log mesh over 4500-5500 A (line at 5000 A) */
    double nu_min=CM_C/(5500e-8), nu_max=CM_C/(4500e-8), dlognu=log(nu_max/nu_min)/NF;
    double nu_l=CM_C/(5000e-8), vdop=1.0e6;
    CMFGENState fs; memset(&fs,0,sizeof fs);
    fs.n_shells=NS;fs.n_bins=NF;fs.nu_min=nu_min;fs.nu_max=nu_max;fs.d_log_nu=dlognu;
    fs.nu=malloc(NF*sizeof(double));fs.dnu=malloc(NF*sizeof(double));
    fs.chi_es=calloc((size_t)NS*NF,sizeof(double));fs.chi_abs=calloc((size_t)NS*NF,sizeof(double));
    fs.chi_line=calloc((size_t)NS*NF,sizeof(double));fs.chi_tot=calloc((size_t)NS*NF,sizeof(double));
    fs.S_fixed=calloc((size_t)NS*NF,sizeof(double));fs.J=calloc((size_t)NS*NF,sizeof(double));
    for (int i=0;i<NF;++i){ fs.nu[i]=nu_min*exp((i+0.5)*dlognu); fs.dnu[i]=fs.nu[i]*dlognu; }
    double dr=geo.r_outer[NS-1]-geo.r_inner[0];
    double ces=0.3/dr, cab=0.05/dr;     /* tau_es~0.3 halo + weak true absorption */
    const double SQRTPI=1.7724538509055160;
    double dnuD=nu_l*vdop/CM_C, chi0=( (tauS>1e-6)?-expm1(-tauS):tauS )/(SQRTPI*vdop*t_exp);
    for (int s=0;s<NS;++s) for (int i=0;i<NF;++i){ size_t idx=(size_t)s*NF+i;
        double B=cm_planck(fs.nu[i],Tinner);
        double cl=0.0;
        if (with_line){ double xv=(fs.nu[i]-nu_l)/dnuD; cl=chi0*exp(-xv*xv); }
        fs.chi_es[idx]=ces+cl;                       /* line scatters (folded into chi_es) */
        fs.chi_abs[idx]=cab; fs.chi_line[idx]=0.0;
        fs.chi_tot[idx]=fs.chi_es[idx]+fs.chi_abs[idx];
        fs.S_fixed[idx]=(fs.chi_tot[idx]>0)?(cab*B)/fs.chi_tot[idx]:0.0;
        fs.J[idx]=0.5*B; }
    int n_ali=80; { const char *e=getenv("LUMINA_CMFGEN_ALI_ITER"); if(e) n_ali=atoi(e); }
    fprintf(stderr,"[cmf_gpu] selftest grid: NS=%d NF=%d line=%s tauS=%.1f n_ali=%d\n",
            NS,NF,with_line?"on":"off",tauS,n_ali);
    cmf_solve_J(&fs,&geo,Tinner,n_ali);   /* honours LUMINA_CMF_SOLVE_GPU (set =2 for A/B) */
    free(geo.r_inner);free(geo.r_outer);free(geo.v_inner);free(geo.v_outer);
    free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);free(fs.chi_line);
    free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
}

/* ... T1 single-line obs P-Cygni self-test ... */
void cmf_obs_selftest(void)
{
    int NS = 40; { const char *e=getenv("LUMINA_CMF_SELFTEST_NS"); if(e) NS=atoi(e); }
    double t_exp = 86400.0;                 /* 1 day */
    double vph = 5.0e8, vmax = 2.5e9;       /* 5000, 25000 km/s [cm/s] */
    { const char *e=getenv("LUMINA_CMF_SELFTEST_VSCALE"); if(e){ double v=atof(e); vph*=v; vmax*=v; } }
    double Tinner = 10000.0;
    double tauS = 100.0; { const char *e=getenv("LUMINA_CMF_SELFTEST_TAUS"); if(e) tauS=atof(e); }

    Geometry geo; memset(&geo,0,sizeof geo);
    geo.n_shells=NS; geo.time_explosion=t_exp;
    geo.r_inner=malloc(NS*sizeof(double)); geo.r_outer=malloc(NS*sizeof(double));
    geo.v_inner=malloc(NS*sizeof(double)); geo.v_outer=malloc(NS*sizeof(double));
    for (int s=0;s<NS;++s){
        double v0=vph+(vmax-vph)*s/(double)NS, v1=vph+(vmax-vph)*(s+1)/(double)NS;
        geo.v_inner[s]=v0; geo.v_outer[s]=v1; geo.r_inner[s]=v0*t_exp; geo.r_outer[s]=v1*t_exp;
    }
    double lam_l=5000e-8, nu_l=CM_C/lam_l;
    double nu_min=CM_C/(5500e-8), nu_max=CM_C/(4500e-8);
    double vdop=1.0e6, dlognu=(vdop/CM_C)/4.0;
    int NF=(int)(log(nu_max/nu_min)/dlognu)+1;
    CMFGENState fs; memset(&fs,0,sizeof fs);
    fs.n_shells=NS; fs.n_bins=NF; fs.nu_min=nu_min; fs.nu_max=nu_max; fs.d_log_nu=dlognu;
    fs.nu=malloc(NF*sizeof(double)); fs.dnu=malloc(NF*sizeof(double));
    fs.chi_es=calloc((size_t)NS*NF,sizeof(double)); fs.chi_abs=calloc((size_t)NS*NF,sizeof(double));
    fs.chi_line=calloc((size_t)NS*NF,sizeof(double)); fs.chi_tot=calloc((size_t)NS*NF,sizeof(double));
    fs.S_fixed=calloc((size_t)NS*NF,sizeof(double)); fs.J=calloc((size_t)NS*NF,sizeof(double));
    for (int i=0;i<NF;++i){ fs.nu[i]=nu_min*exp((i+0.5)*dlognu); fs.dnu[i]=fs.nu[i]*dlognu; }
    double dr=geo.r_outer[NS-1]-geo.r_inner[0], chi_es=0.1/dr;   /* thin halo tau_es~0.1 */
    for (int s=0;s<NS;++s){
        double r=0.5*(geo.r_inner[s]+geo.r_outer[s]);
        double a=1.0-(geo.r_inner[0]*geo.r_inner[0])/(r*r); double W=0.5*(1.0-sqrt(a>0?a:0));
        for (int i=0;i<NF;++i){ size_t idx=(size_t)s*NF+i;
            fs.chi_es[idx]=chi_es; fs.chi_tot[idx]=chi_es;
            fs.J[idx]=W*cm_planck(fs.nu[i],Tinner); }
    }
    OpacityState opac; memset(&opac,0,sizeof opac);
    opac.n_lines=1; opac.line_list_nu=malloc(sizeof(double)); opac.line_list_nu[0]=nu_l;
    opac.tau_sobolev=malloc((size_t)NS*sizeof(double));
    for (int s=0;s<NS;++s) opac.tau_sobolev[s]=tauS;
    double *Te=malloc(NS*sizeof(double)); for(int s=0;s<NS;++s) Te[s]=8000.0;
    fprintf(stderr,"[OBS-SELFTEST] single line 5000A tauS=%.1f, NS=%d NF=%d, beta=%.3f-%.3f\n",
            tauS,NS,NF,vph/CM_C,vmax/CM_C);
    /* CONFIRMATION-LADDER T1b: self-consistent comoving J̄ source (physics-agent fix).
     * Deposit the line into chi_es (pure scatter), solve cmf_solve_J for the
     * line-coupled J̄ field (line scattering raises the halo J̄ that fixed W·B
     * misses → the -68A static leak), extract J̄_l, feed the obs-march via
     * jbar_line_det (g_sob_jbardet). Gate LUMINA_CMF_SELFTEST_JBAR=1. */
    if (getenv("LUMINA_CMF_SELFTEST_JBAR") && atoi(getenv("LUMINA_CMF_SELFTEST_JBAR"))) {
        const double SQRTPI=1.7724538509055160;
        double dnuD=nu_l*vdop/CM_C, chi0_pref=1.0/(SQRTPI*vdop*t_exp);
        double frac=(tauS>1e-6)?-expm1(-tauS):tauS, chi0=frac*chi0_pref;
        double *cl_save=calloc((size_t)NS*NF,sizeof(double));
        for (int s=0;s<NS;++s) for (int i=0;i<NF;++i){
            double xv=(fs.nu[i]-nu_l)/dnuD, cl=chi0*exp(-xv*xv);
            size_t idx=(size_t)s*NF+i; cl_save[idx]=cl;
            fs.chi_es[idx]+=cl; fs.chi_tot[idx]=fs.chi_es[idx]+fs.chi_abs[idx];
            fs.J[idx]=cm_planck(fs.nu[i],Tinner)*0.5;  /* warm start */
        }
        int n_ali=24; { const char *e=getenv("LUMINA_CMFGEN_ALI_ITER"); if(e) n_ali=atoi(e); }
        cmf_solve_J(&fs,&geo,Tinner,n_ali);            /* self-consistent line-coupled J̄ */
        opac.jbar_line_det=malloc((size_t)NS*sizeof(double));
        for (int s=0;s<NS;++s){ double num=0,den=0;
            for (int i=0;i<NF;++i){ double xv=(fs.nu[i]-nu_l)/dnuD, phi=exp(-xv*xv);
                num+=phi*fs.J[(size_t)s*NF+i]; den+=phi; }
            opac.jbar_line_det[s]=(den>0)?num/den:-1.0; }
        for (int s=0;s<NS;++s) for (int i=0;i<NF;++i){    /* restore continuum chi_es for obs */
            size_t idx=(size_t)s*NF+i; fs.chi_es[idx]-=cl_save[idx];
            fs.chi_tot[idx]=fs.chi_es[idx]+fs.chi_abs[idx]; }
        free(cl_save);
        setenv("LUMINA_CMF_OBS_JBARDET","1",1);          /* obs uses J̄_l source */
        fprintf(stderr,"[OBS-SELFTEST] JBAR mode: self-consistent J̄_l[sh0]=%.3e [shNS/2]=%.3e (W*B sh0~%.3e)\n",
                opac.jbar_line_det[0],opac.jbar_line_det[NS/2],0.5*cm_planck(nu_l,Tinner));
    }
    cmfgen_fine_emergent_obs(&fs,&geo,Tinner,&opac,Te,"lumina_obs_selftest.csv");
    fprintf(stderr,"[OBS-SELFTEST] wrote lumina_obs_selftest.csv (analytic: net EW=0 for pure scatter)\n");
    free(geo.r_inner);free(geo.r_outer);free(geo.v_inner);free(geo.v_outer);
    free(fs.nu);free(fs.dnu);free(fs.chi_es);free(fs.chi_abs);free(fs.chi_line);
    free(fs.chi_tot);free(fs.S_fixed);free(fs.J);
    free(opac.line_list_nu);free(opac.tau_sobolev);free(Te);
    if (opac.jbar_line_det) free(opac.jbar_line_det);
}

int cmfgen_run(Geometry *geo, OpacityState *opac, BFOpacity *bf,
               PlasmaState *plasma, NLTEConfig *nlte, AtomicData *atom,
               GammaDeposition *gamma, double T_inner, int n_iter)
{
    if (getenv("LUMINA_CMF_OBS_SELFTEST")) { cmf_obs_selftest(); return 0; }
    int validation_exit_after_r6 = 0;
    if (cmf_optional_binary_env(
            "LUMINA_VALIDATION_EXIT_AFTER_R6",
            &validation_exit_after_r6) != 0) {
        fprintf(stderr,
                "[VALIDATION][BLOCKED] reason=INVALID_EXIT_AFTER_R6 "
                "value=%s expected=0_or_1\n",
                getenv("LUMINA_VALIDATION_EXIT_AFTER_R6") ?
                    getenv("LUMINA_VALIDATION_EXIT_AFTER_R6") : "(unset)");
        return -1;
    }
    if (validation_exit_after_r6) {
        int ab_enabled = 0;
        int requested_devices = 0;
        if (n_iter != 1 ||
            cmf_optional_binary_env(
                "LUMINA_CMF_FINE_MGPU_AB", &ab_enabled) != 0 ||
            !ab_enabled ||
            cmf_fine_multigpu_device_request(&requested_devices) != 0 ||
            requested_devices <= 0) {
            fprintf(stderr,
                    "[VALIDATION][BLOCKED] "
                    "reason=EXIT_AFTER_R6_REQUIRES_SINGLE_ITER_MGPU_AB "
                    "n_iter=%d ab=%s devices=%s\n",
                    n_iter,
                    getenv("LUMINA_CMF_FINE_MGPU_AB") ?
                        getenv("LUMINA_CMF_FINE_MGPU_AB") : "(unset)",
                    getenv("LUMINA_CMF_FINE_MGPU_DEVICES") ?
                        getenv("LUMINA_CMF_FINE_MGPU_DEVICES") : "(unset)");
            return -1;
        }
    }
    /* Signed line opacity is resolved by the line-resolved producer: CMFGEN
     * SRCE_CHK replaces tau<-0.5 while the mild interval [-0.5,0) remains
     * signed.  Do not reject or clamp it at this outer dispatcher. */
    if (bf && bf->chi_bf) {
        size_t n=(size_t)bf->n_shells*bf->n_freq_bins;
        for(size_t k=0;k<n;k++) if(bf->chi_bf[k]<0.0) {
            a208_counters()->blocked_negative_formal++;
            fprintf(stderr,"[A2-08][BLOCKED] consumer=F01-G23 reason="
                    "BLOCKED_NEGATIVE_OPACITY_SEMANTICS bf_identity=%zu rc=3\n",k);
            return 3;
        }
    }
    CMFGENState cs;
    if (cmfgen_init(&cs, geo) != 0) {
        fprintf(stderr, "[CMFGEN][FATAL] cmfgen_init failed\n");
        return -1;
    }

    /* P1 gate 1a: LUMINA_CMF_CONTONLY=1 forces continuum-only assemble (line
     * opacity zeroed) so the CMF J-producer can be compared to cmfgen_solve_J
     * on identical continuum opacity. */
    int cmf_lineres = 0;
    { const char *e = getenv("LUMINA_CMF_LINERES"); if (e) cmf_lineres = atoi(e); }
    { const char *e = getenv("LUMINA_CMF_CONTONLY"); if (e && atoi(e)) cs.cont_only = 1; }

    const char *ali_env = getenv("LUMINA_CMFGEN_ALI_ITER");
    int n_ali = ali_env ? atoi(ali_env) : 8;
    if (n_ali < 1) n_ali = 1;

    printf("[CMFGEN] pure deterministic radiation driver: %d shells, %d bins, "
           "%d rays, %d outer iters, %d ALI/iter, t_exp=%.4e s\n",
           cs.n_shells, cs.n_bins, cs.n_rays, n_iter, n_ali, geo->time_explosion);

    double t_exp = geo->time_explosion;
    /* A2-INIT (2026-08-16): pass 0 is the separately-labelled initialization
     * pass — R1 from the bootstrap material, then exactly one fixed-seed-Te
     * NLTE material predictor commit (INIT_SEED_MATERIAL_PREDICTOR).  It
     * consumes none of the user's n_iter logical iterations and produces no
     * physics-comparison snapshot.  Ordinary logical iterations keep their
     * 0..n_iter-1 numbering.  The radiation generation is a monotonic commit
     * counter (owner contract: previous+1), not iter+1: the init pass owns
     * generation 1 and logical iteration k owns generation k+2. */
    uint64_t rad_generation = 0;
    for (int pass = 0; pass < (n_iter > 0 ? n_iter + 1 : 0); ++pass) {
        const int init_pass = (pass == 0);
        const int iter = pass - 1;
        if (nlte) nlte->current_iter = init_pass ? 0 : iter;

        /* refresh bf opacity for current ionization/T_e */
        if (bf) compute_bf_opacity(bf, atom, plasma, cs.n_shells);

        cmfgen_assemble(&cs, geo, opac, bf, plasma);
        if (cmf_lineres) {
            if (cmf_solve_J(&cs, geo, T_inner, n_ali) != 0) {
                fprintf(stderr, "[CMFGEN][FATAL] GPU CMF lifecycle failure; no CPU publication\n");
                cmfgen_free(&cs);
                return -1;
            }
        } else {
            cmfgen_solve_J(&cs, geo, T_inner, n_ali);/* binned (champion) */
        }
        if (!init_pass &&
            cmfgen_stage32_rung1_maybe_dump(&cs,geo,opac,plasma,
                                             iter,n_iter) != 0) {
            /* ★침묵 금지(2026-08-07) */
            fprintf(stderr, "[CMFGEN][FATAL] stage32 rung1 dump failed iter=%d\n", iter);
            cmfgen_free(&cs);
            return -1;
        }
        cmfgen_window_color(&cs);
        radeq_set_tail_color(cs.t_color, cs.n_shells);
        radeq_set_tri_response(cs.tri_lo, cs.tri_up, cs.tri_r,
                               cs.n_shells, cs.n_bins);
        if (cs.diag && iter == n_iter - 1)
            cmfgen_validate(&cs, geo, plasma);

        radeq_set_line_re_source(cs.chi_line_re, cs.chi_abs, cs.chi_tot,
                                 cs.S_fixed, cs.J, cs.nu, cs.dnu,
                                 cs.lambda_star, plasma->T_e,
                                 cs.chi_line, cs.chi_line_cls,
                                 cs.n_shells, cs.n_bins);

        /* R6 contract: deterministic line-Jbar is not optional configuration.
         * Produce it before the commit so continuum and selective line views
         * become visible at one generation and one choke point. */
        if (opac->n_lines <= 0 || !opac->tau_sobolev) {
            fprintf(stderr,
                    "[R6][BLOCKED] reason=DETERMINISTIC_LINE_INPUT_MISSING "
                    "iter=%d\n", iter);
            cmfgen_free(&cs);
            return -1;
        }
        if (!opac->jbar_line_det)
            opac->jbar_line_det = (double *)malloc(
                (size_t)opac->n_lines * (size_t)cs.n_shells * sizeof(double));
        if (!opac->jbar_line_det) {
            fprintf(stderr,
                    "[R6][FATAL] reason=DETERMINISTIC_LINE_JBAR_ALLOCATION "
                    "iter=%d\n", iter);
            cmfgen_free(&cs);
            return -1;
        }
        CMFFineLineOperator line_operator = init_pass
            ? CMF_FINE_LINE_OPERATOR_INIT_SHARED_GAUSSIAN
            : CMF_FINE_LINE_OPERATOR_CMFGEN_NONOVERLAP_SOBOLEV;
        if (cmfgen_fine_jbar(&cs, geo, opac, T_inner, plasma,
                             line_operator) != 0) {
            fprintf(stderr,
                    "[R6][BLOCKED] reason=DETERMINISTIC_LINE_JBAR_PRODUCER "
                    "iter=%d\n", iter);
            cmfgen_free(&cs);
            return -1;
        }

        a208_counters()->replay_line_blocks_attempted++;
        ++rad_generation;
        if (cmfgen_commit_jnu(&cs, nlte, geo, opac, rad_generation) != 0) {
            fprintf(stderr, "[RADIATION-FIELD][FATAL] pure-CMFGEN commit failed iter=%d\n",
                    iter);
            cmfgen_free(&cs);
            return -1;
        }
        a208_counters()->replay_line_blocks_committed++;

        if (validation_exit_after_r6) {
            fprintf(stderr,
                    "[VALIDATION][EXIT-AFTER-R6] PASS iter=%d generation=%llu "
                    "downstream_r7=NOT_RUN a2_10=NOT_RUN spectra=NOT_WRITTEN\n",
                    iter, (unsigned long long)rad_generation);
            cmfgen_free(&cs);
            return 0;
        }

        /* R7 phase: commit/view -> gamma publication -> A2-08/A2-09/A2-10.
         * The publication is once per physical explosion epoch, before any
         * material mutation. */
        if (gamma && gamma->generation == 0) {
            int gamma_rc = gamma_deposition_publish(
                gamma, GAMMA_PROVENANCE_INTERNAL_BATEMAN, t_exp,
                atom, plasma, geo, NULL);
            if (gamma_rc != 0) {
                fprintf(stderr,
                        "[GAMMA][FATAL] lane=DET iter=%d rc=%d\n",
                        iter, gamma_rc);
                cmfgen_free(&cs);
                return gamma_rc;
            }
        }

        if (init_pass) {
            /* Exactly-once, unconditional: NLTE material predictor at the
             * published seed Te.  Commits population m->m+1 while the public
             * Te bytes/generation/publication stay byte-identical; R2 (the
             * next pass's exact publication) is then computed from P2 before
             * the first A2-10 root.  Failure terminates — no LTE/Saha or
             * old-material fallback exists past this point. */
            int seed_rc = lumina_init_seed_material_predictor(
                opac, bf, atom, plasma, nlte, gamma, t_exp, cs.n_shells);
            if (seed_rc != 0) {
                fprintf(stderr,
                        "[A2-INIT][FATAL] lane=DET "
                        "event=SEED_MATERIAL_PREDICTOR rc=%d\n", seed_rc);
                cmfgen_free(&cs);
                return seed_rc;
            }
            continue;
        }

        {
            int r7_rc = lumina_r7_publish_and_solve_te(
                opac, bf, atom, plasma, nlte, gamma,
                t_exp, cs.n_shells, 1, "DET", iter);
            if (r7_rc != 0) {
                fprintf(stderr,
                        "[R7][FATAL] lane=DET iter=%d rc=%d\n",
                        iter, r7_rc);
                cmfgen_free(&cs);
                return r7_rc;
            }
            PhysicsComparisonStatus dump_status =
                physics_comparison_dump_if_requested(
                    "DET",iter,geo,atom,plasma,opac,nlte);
            if (dump_status != PHYSICS_COMPARISON_OK &&
                dump_status != PHYSICS_COMPARISON_NOT_REQUESTED) {
                fprintf(stderr,
                        "[PHYSICS-COMPARISON][FATAL] lane=DET iter=%d status=%s\n",
                        iter,physics_comparison_status_name(dump_status));
                cmfgen_free(&cs);
                return -1;
            }
        }

        /* CORRECTED FALSIFIER (LUMINA_CMF_JINC_CONT=1, codex 2026-06-22): compute the
         * CLEAN external continuum field J_inc (cont_only solve, ALL line opacity zeroed
         * => no line self-emission, no super-thermal contamination, bounded) and sample
         * it per line into opac->jbar_line so the mode-3 bb-rate hook (LUMINA_NLTE_JBAR_
         * POPS=3) pumps R_lu=B_lu*beta_l*J_inc_cont. Decisive: max(S_l/B)<=1e6 => mode-3
         * FORM is correct (the sealed MC explosion was the contaminated input, NOT the
         * form); max(S_l/B)>1e10 => form unstable => full MALI needed. cs.J + opacity are
         * saved/restored so continuum rates, line-RE, and the spectrum are unaffected. */
        {
            static int jinc_cont = -1;
            if (jinc_cont < 0) { const char *e = getenv("LUMINA_CMF_JINC_CONT");
                jinc_cont = (e && atoi(e)) ? 1 : 0; }
            if (jinc_cont && opac->tau_sobolev && opac->line_list_nu) {
                size_t NS = cs.n_shells, NB = cs.n_bins; int NL = opac->n_lines;
                if (!opac->jbar_line)  opac->jbar_line  = (double*)calloc((size_t)NL*NS, sizeof(double));
                if (!opac->jbar_count) opac->jbar_count = (int*)   calloc((size_t)NL*NS, sizeof(int));
                static double *Jsave = NULL;
                if (!Jsave) Jsave = (double*)malloc(NS*NB*sizeof(double));
                memcpy(Jsave, cs.J, NS*NB*sizeof(double));     /* save full J */
                cs.cont_only = 1;
                cmfgen_assemble(&cs, geo, opac, bf, plasma);   /* line opacity zeroed */
                cmfgen_solve_J(&cs, geo, T_inner, n_ali);      /* cs.J = J_inc_cont (continuum) */
                for (size_t s = 0; s < NS; ++s)
                    for (int l = 0; l < NL; ++l) {
                        size_t k = (size_t)l*NS + s;
                        double tau = opac->tau_sobolev[k];
                        double nu_l = opac->line_list_nu[l];
                        if (tau <= 1e-12 || nu_l <= cs.nu_min || nu_l >= cs.nu_max) {
                            opac->jbar_count[k] = 0; continue; }
                        int b = (int)floor(log(nu_l / cs.nu_min) / cs.d_log_nu);
                        if (b < 0 || b >= (int)NB) { opac->jbar_count[k] = 0; continue; }
                        opac->jbar_line[k]  = cs.J[s*NB + (size_t)b]; /* clean external J_inc */
                        opac->jbar_count[k] = 1000;                   /* pass use_jbar guard */
                    }
                /* CROSS-LINE OVERLAP (4a, LUMINA_CMF_OVERLAP=1, codex 2026-06-22): per shell,
                 * blue->red sweep carrying the redshifted ESCAPING emission of bluer lines into
                 * redder lines' incident field = the UV->optical fluorescence carrier (form
                 * validated by self-test 4a, +175%). J_emit = f_out*beta_l*S_l_lag (escaping
                 * part, NOT the trapped (1-beta)S_l); J_overlap attenuates by exp(-dtau_cont)
                 * over the frequency gap (Sobolev dr_res=(dnu/nu)*c*t_exp). cs.chi_es/chi_abs
                 * are continuum-only here (cont_only state). Lagged: S_l from previous-pass pops.
                 * Order: use J_overlap for line l, THEN add l's emission (no self-pump). */
                {
                    static int ovl = -1; static double fout = 0.5;
                    if (ovl < 0) { const char *e = getenv("LUMINA_CMF_OVERLAP"); ovl = (e&&atoi(e))?1:0;
                        const char *fo = getenv("LUMINA_OVERLAP_FOUT"); if (fo) fout = atof(fo); }
                    if (ovl) {
                        double t_exp_l = geo->time_explosion;
                        for (size_t s = 0; s < NS; ++s) {
                            double Jov = 0.0, rmax = 1.0;
                            for (int l = 0; l < NL; ++l) {   /* line_list_nu DESCENDING = blue->red */
                                double nu_l = opac->line_list_nu[l];
                                if (nu_l <= cs.nu_min || nu_l >= cs.nu_max) continue;
                                int b = (int)floor(log(nu_l/cs.nu_min)/cs.d_log_nu);
                                if (b < 0 || b >= (int)NB) continue;
                                size_t k = (size_t)l*NS + s;
                                double tau = opac->tau_sobolev[k];
                                if (tau > 1e-12) {
                                    double Jcont = opac->jbar_line[k];          /* continuum J_inc */
                                    opac->jbar_line[k] = Jcont + Jov;           /* + overlap incident */
                                    opac->jbar_count[k] = 1000;
                                    if (Jcont > 0.0 && (Jcont+Jov)/Jcont > rmax) rmax = (Jcont+Jov)/Jcont;
                                    double beta = (tau > 1e-6) ? -expm1(-tau)/tau : 1.0;
                                    double S = opac->line_source_S[k];
                                    if (S > 0.0) Jov += fout * beta * S;        /* escaping emission */
                                }
                                if (l < NL-1) {                                  /* attenuate over gap */
                                    double dnu = nu_l - opac->line_list_nu[l+1];
                                    if (dnu > 0.0) {
                                        double chic = cs.chi_es[s*NB+(size_t)b] + cs.chi_abs[s*NB+(size_t)b];
                                        Jov *= exp(-chic * (dnu/nu_l) * CM_C * t_exp_l);
                                    }
                                }
                            }
                            if (rmax > 1e3)
                                printf("  [OVERLAP] shell %zu max(J_inc/J_cont)=%.1e (runaway watch)\n", s, rmax);
                        }
                    }
                }
                cs.cont_only = 0;
                cmfgen_assemble(&cs, geo, opac, bf, plasma);   /* restore full opacity */
                memcpy(cs.J, Jsave, NS*NB*sizeof(double));     /* restore full J */
            }
        }
        /* R7/A2-10 has already committed Te, ne, populations, BF, EW tau,
         * A208 and A209 as one material generation.  A second population or
         * NLTE solve here would split that transaction.  Optional J_inc/line
         * overlap values produced above are deliberately lagged inputs for
         * the next outer iteration's private candidate bundle. */
        if (atom->partition_stamp.te_generation !=
                plasma->te_publication.committed_te_generation ||
            strcmp(atom->partition_stamp.te_manifest_sha256,
                   plasma->te_publication.te_manifest_sha256) != 0) {
            fprintf(stderr,
                    "[A2-10][FATAL] committed CMF Te/population stamp mismatch "
                    "iter=%d\n",
                    iter);
            return -1;
        }

        if (cs.diag) {
            int mid = cs.n_shells / 2;
            printf("[CMFGEN] iter %2d: T_e[0]=%.0fK T_e[%d]=%.0fK T_e[%d]=%.0fK "
                   "J[mid,bin500]=%.3e\n",
                   iter, plasma->T_e[0], mid, plasma->T_e[mid],
                   cs.n_shells - 1, plasma->T_e[cs.n_shells - 1],
                   cs.J[(size_t)mid * cs.n_bins + 500]);
        }
    }

    /* Default frame = observer (all comparisons to gold are observer-frame);
     * set LUMINA_CMF_OBSERVER_FRAME=0 for the legacy comoving spectrum. */
    const char *obs_env = getenv("LUMINA_CMF_OBSERVER_FRAME");
    int obs_frame = obs_env ? atoi(obs_env) : 1;
    if (obs_frame) {
        cmfgen_write_spectrum_obs(&cs, geo, T_inner, opac, plasma->T_e,
                                  "lumina_spectrum.csv");
        cmfgen_write_spectrum(&cs, geo, T_inner, "lumina_spectrum_comoving.csv");
    } else {
        cmfgen_write_spectrum(&cs, geo, T_inner, "lumina_spectrum.csv");
    }
    cmfgen_free(&cs);
    return 0;
}
