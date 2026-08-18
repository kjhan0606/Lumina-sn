#define _GNU_SOURCE
#include "line_jbar.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define C_CGS 2.99792458e10

static int proof_multiply_up(double a, double b, double *upper)
{
    if (!upper || !(a >= 0.0) || !(b >= 0.0) ||
        !isfinite(a) || !isfinite(b)) return -1;
    if (a == 0.0 || b == 0.0) {
        *upper = 0.0;
        return 0;
    }
    double product = a * b;
    if (!isfinite(product)) return -1;
    *upper = product == 0.0 ? nextafter(0.0, INFINITY)
                            : nextafter(product, INFINITY);
    return isfinite(*upper) ? 0 : -1;
}

static int proof_add_up(double a, double b, double *upper)
{
    if (!upper || !(a >= 0.0) || !(b >= 0.0) ||
        !isfinite(a) || !isfinite(b)) return -1;
    double sum = a + b;
    if (!isfinite(sum)) return -1;
    *upper = (a == 0.0 || b == 0.0) ? sum : nextafter(sum, INFINITY);
    return isfinite(*upper) ? 0 : -1;
}

static int proof_add_down(double a, double b, double *lower)
{
    if (!lower || !(a >= 0.0) || !(b >= 0.0) ||
        !isfinite(a) || !isfinite(b)) return -1;
    double sum = a + b;
    if (!isfinite(sum)) return -1;
    *lower = (a == 0.0 || b == 0.0) ? sum : nextafter(sum, 0.0);
    return 0;
}

static int proof_divide_up(double numerator, double denominator,
                           double *upper)
{
    if (!upper || !(numerator >= 0.0) || !(denominator > 0.0) ||
        !isfinite(numerator) || !isfinite(denominator)) return -1;
    if (numerator == 0.0) {
        *upper = 0.0;
        return 0;
    }
    double quotient = numerator / denominator;
    if (!isfinite(quotient)) return -1;
    *upper = quotient == 0.0 ? nextafter(0.0, INFINITY)
                             : nextafter(quotient, INFINITY);
    return isfinite(*upper) ? 0 : -1;
}

LineJbarProfileStatus line_jbar_gaussian_discrete_shells(
    size_t n_shells, size_t n_bins,
    const double *nu, const double *dnu,
    const double *value, const double *value_error_upper,
    double line_nu, double vdoppler_cms, double ndoppler,
    double *value_average, double *error_average_upper,
    LineJbarProfileReport *report)
{
    LineJbarProfileReport local;
    memset(&local, 0, sizeof(local));
    local.first_bin = SIZE_MAX;
    local.last_bin = SIZE_MAX;
    local.error_upper_min = INFINITY;
    if (report) *report = local;
    if (n_shells == 0 || n_bins < 2 || !nu || !dnu || !value ||
        !value_error_upper || !value_average || !error_average_upper ||
        !(line_nu > 0.0) || !(vdoppler_cms > 0.0) ||
        !(ndoppler > 0.0) || !isfinite(line_nu) ||
        !isfinite(vdoppler_cms) || !isfinite(ndoppler) ||
        n_shells > SIZE_MAX / n_bins)
        return LINE_JBAR_PROFILE_INVALID_INPUT;
    if (!(nu[0] > 0.0) || !(nu[n_bins - 1] > nu[0]) ||
        !isfinite(nu[0]) || !isfinite(nu[n_bins - 1]))
        return LINE_JBAR_PROFILE_INVALID_INPUT;

    double dnu_doppler = line_nu * vdoppler_cms / C_CGS;
    double fractional_width = ndoppler * vdoppler_cms / C_CGS;
    double support_lo = line_nu * (1.0 - fractional_width);
    double support_hi = line_nu * (1.0 + fractional_width);
    if (!(dnu_doppler > 0.0) || !(support_lo > 0.0) ||
        !(support_hi > support_lo) || !isfinite(support_hi))
        return LINE_JBAR_PROFILE_INVALID_INPUT;
    if (support_lo < nu[0] || support_hi > nu[n_bins - 1])
        return LINE_JBAR_PROFILE_UNCOVERED;

    size_t lo = 0, hi = n_bins;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2U;
        if (nu[mid] < support_lo) lo = mid + 1U;
        else hi = mid;
    }
    size_t first = lo;
    lo = first; hi = n_bins;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2U;
        if (nu[mid] <= support_hi) lo = mid + 1U;
        else hi = mid;
    }
    size_t last_exclusive = lo;
    /* The authoritative membership is |x|<=ndoppler below.  Include one
     * adjacent centre on each side so an independently rounded support-edge
     * product cannot omit a boundary bin. */
    if (first > 0) --first;
    if (last_exclusive < n_bins) ++last_exclusive;
    if (first >= last_exclusive) return LINE_JBAR_PROFILE_UNCOVERED;
    for (size_t s = 0; s < n_shells; ++s) {
        value_average[s] = 0.0;
        error_average_upper[s] = 0.0;
    }

    double denominator = 0.0;
    double denominator_lower = 0.0;
    size_t used = 0;
    for (size_t i = first; i < last_exclusive; ++i) {
        if (!(nu[i] > 0.0) || !(dnu[i] > 0.0) ||
            !isfinite(nu[i]) || !isfinite(dnu[i]) ||
            (i > first && !(nu[i] > nu[i - 1])))
            return LINE_JBAR_PROFILE_INVALID_INPUT;
        double x = (nu[i] - line_nu) / dnu_doppler;
        if (!isfinite(x) || fabs(x) > ndoppler) continue;
        double weight = exp(-x * x) * dnu[i];
        if (!(weight > 0.0) || !isfinite(weight))
            return LINE_JBAR_PROFILE_NONFINITE;
        double next_denominator = denominator + weight;
        double next_denominator_lower;
        if (!isfinite(next_denominator) ||
            proof_add_down(denominator_lower, weight,
                           &next_denominator_lower) != 0)
            return LINE_JBAR_PROFILE_NONFINITE;
        denominator = next_denominator;
        denominator_lower = next_denominator_lower;
        ++used;
        for (size_t s = 0; s < n_shells; ++s) {
            size_t cell = s * n_bins + i;
            double physical = value[cell];
            double error = value_error_upper[cell];
            if (!(physical >= 0.0) || !(error >= 0.0) ||
                !isfinite(physical) || !isfinite(error))
                return LINE_JBAR_PROFILE_NONFINITE;
            double contribution = weight * physical;
            double error_product, error_sum;
            if (!isfinite(contribution) ||
                proof_multiply_up(weight, error, &error_product) != 0 ||
                proof_add_up(error_average_upper[s], error_product,
                             &error_sum) != 0)
                return LINE_JBAR_PROFILE_NONFINITE;
            value_average[s] += contribution;
            if (!isfinite(value_average[s]))
                return LINE_JBAR_PROFILE_NONFINITE;
            error_average_upper[s] = error_sum;
        }
    }
    if (used == 0 || !(denominator > 0.0) ||
        !(denominator_lower > 0.0) || !isfinite(denominator))
        return LINE_JBAR_PROFILE_UNCOVERED;
    double error_min = INFINITY, error_max = 0.0;
    for (size_t s = 0; s < n_shells; ++s) {
        value_average[s] /= denominator;
        if (!(value_average[s] >= 0.0) || !isfinite(value_average[s]) ||
            proof_divide_up(error_average_upper[s], denominator_lower,
                            &error_average_upper[s]) != 0)
            return LINE_JBAR_PROFILE_NONFINITE;
        if (error_average_upper[s] < error_min)
            error_min = error_average_upper[s];
        if (error_average_upper[s] > error_max)
            error_max = error_average_upper[s];
    }
    local.first_bin = first;
    local.last_bin = last_exclusive - 1U;
    local.contributing_bins = used;
    local.weight_sum = denominator;
    local.weight_sum_lower = denominator_lower;
    local.error_upper_min = error_min;
    local.error_upper_max = error_max;
    if (report) *report = local;
    return LINE_JBAR_PROFILE_OK;
}

/* ---- tiny SHA-256 (FIPS 180-4), enough for Q-set/profile binding ---- */
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t n; } Sha256;
static const uint32_t K256[64] = {
 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define ROR(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void sha256_init(Sha256 *s){
    static const uint32_t h0[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                 0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(s->h,h0,sizeof h0); s->len=0; s->n=0;
}
static void sha256_block(Sha256 *s,const uint8_t *p){
    uint32_t w[64],a,b,c,d,e,f,g,h;
    for(int i=0;i<16;i++) w[i]=(uint32_t)p[4*i]<<24|(uint32_t)p[4*i+1]<<16|
                               (uint32_t)p[4*i+2]<<8|p[4*i+3];
    for(int i=16;i<64;i++){
        uint32_t s0=ROR(w[i-15],7)^ROR(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1=ROR(w[i-2],17)^ROR(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    a=s->h[0];b=s->h[1];c=s->h[2];d=s->h[3];e=s->h[4];f=s->h[5];g=s->h[6];h=s->h[7];
    for(int i=0;i<64;i++){
        uint32_t S1=ROR(e,6)^ROR(e,11)^ROR(e,25), ch=(e&f)^((~e)&g);
        uint32_t t1=h+S1+ch+K256[i]+w[i];
        uint32_t S0=ROR(a,2)^ROR(a,13)^ROR(a,22), mj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+mj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
static void sha256_update(Sha256 *s,const void *data,size_t n){
    const uint8_t *p=data; s->len+=n;
    while(n){ size_t k=64-s->n; if(k>n)k=n; memcpy(s->buf+s->n,p,k);
        s->n+=k;p+=k;n-=k; if(s->n==64){sha256_block(s,s->buf);s->n=0;} }
}
static void sha256_hex(Sha256 *s,char out[65]){
    uint64_t bits=s->len*8; uint8_t pad=0x80;
    sha256_update(s,&pad,1); uint8_t z=0;
    while(s->n!=56) sha256_update(s,&z,1);
    uint8_t lenb[8]; for(int i=0;i<8;i++) lenb[i]=(uint8_t)(bits>>(56-8*i));
    sha256_update(s,lenb,8);
    for(int i=0;i<8;i++) sprintf(out+8*i,"%08x",s->h[i]);
    out[64]=0;
}

/* ---- Q_g / Q_E immutable membership ---- */
static int cmp_by_nu(const void *xa, const void *xb, void *ctx)
{
    const double *nu = ctx;
    size_t a = *(const size_t *)xa, b = *(const size_t *)xb;
    return nu[a] < nu[b] ? -1 : nu[a] > nu[b] ? 1 : (a < b ? -1 : a > b);
}

int line_jbar_bb_domain_mask_build(uint8_t *mask, int n_lines,
                                   const double *line_nu_all,
                                   const int *nlte_line_map,
                                   size_t *inside_enabled,
                                   size_t *outside_enabled)
{
    size_t inside = 0, outside = 0;
    if (!mask || n_lines <= 0 || !line_nu_all || !nlte_line_map) return -1;
    for (int l = 0; l < n_lines; ++l) {
        mask[l] = 0;
        if (nlte_line_map[l] < 0) continue;
        if (!(line_nu_all[l] > 0.0) || !isfinite(line_nu_all[l])) return -1;
        if (line_jbar_frequency_in_bb_domain(line_nu_all[l])) {
            mask[l] = 1;
            ++inside;
        } else {
            ++outside;
        }
    }
    if (inside_enabled) *inside_enabled = inside;
    if (outside_enabled) *outside_enabled = outside;
    return inside ? 0 : -1;
}

int line_jbar_qset_profile_support_covered(const LineJbarQSet *q,
                                           double grid_nu_min,
                                           double grid_nu_max,
                                           size_t *first_bad_q)
{
    if (first_bad_q) *first_bad_q = SIZE_MAX;
    if (!q || q->n_q == 0 || !q->line_nu ||
        !(grid_nu_min > 0.0) || !(grid_nu_max > grid_nu_min) ||
        !isfinite(grid_nu_min) || !isfinite(grid_nu_max)) return -1;
    double width = LINE_JBAR_PROFILE_NDOPPLER *
                   LINE_JBAR_VDOPPLER_CMS / C_CGS;
    for (size_t i = 0; i < q->n_q; ++i) {
        double nu = q->line_nu[i];
        if (!line_jbar_frequency_in_bb_domain(nu) ||
            nu * (1.0 - width) < grid_nu_min ||
            nu * (1.0 + width) > grid_nu_max) {
            if (first_bad_q) *first_bad_q = i;
            return -1;
        }
    }
    return 0;
}

int line_jbar_qset_build(LineJbarQSet *q, int n_lines,
                         const double *line_nu_all, const int *nlte_line_map,
                         const uint8_t *bb_in_domain)
{
    if (!q || n_lines <= 0 || !line_nu_all || !nlte_line_map) return -1;
    memset(q, 0, sizeof(*q));
    size_t n = 0;
    for (int l = 0; l < n_lines; l++)
        if (nlte_line_map[l] >= 0 && (!bb_in_domain || bb_in_domain[l]))
            n++;
    if (n == 0) return -1;
    q->n_q = n;
    q->line_id = malloc(n * sizeof(int));
    q->line_nu = malloc(n * sizeof(double));
    q->by_nu = malloc(n * sizeof(size_t));
    if (!q->line_id || !q->line_nu || !q->by_nu) { line_jbar_qset_free(q); return -1; }
    size_t k = 0;
    for (int l = 0; l < n_lines; l++)
        if (nlte_line_map[l] >= 0 && (!bb_in_domain || bb_in_domain[l])) {
            if (!(line_nu_all[l] > 0.0) || !isfinite(line_nu_all[l])) {
                line_jbar_qset_free(q); return -1;
            }
            q->line_id[k] = l; q->line_nu[k] = line_nu_all[l]; k++;
        }
    for (size_t i = 0; i < n; i++) q->by_nu[i] = i;
    qsort_r(q->by_nu, n, sizeof(size_t), cmp_by_nu, q->line_nu);
    Sha256 s; sha256_init(&s);
    if (bb_in_domain) {
        const char *domain_tag = "BB_IN_DOMAIN|";
        memcpy(q->domain_contract_hash,
               LINE_JBAR_BB_DOMAIN_CONTRACT_SHA256, 65);
        sha256_update(&s, domain_tag, strlen(domain_tag));
        sha256_update(&s, q->domain_contract_hash, 64);
    }
    sha256_update(&s, q->line_id, n * sizeof(int));
    sha256_hex(&s, q->q_set_hash);
    q->profile_id = LINE_JBAR_PROFILE_GAUSS_VD10;
    memcpy(q->profile_hash, LINE_JBAR_PROFILE_SHA256, 65);
    q->set_kind = LINE_JBAR_SET_RATE_GRAPH;
    return 0;
}

int line_jbar_eset_build(LineJbarESet *e, int n_lines,
                         const double *line_nu_all)
{
    if (!e || n_lines <= 0 || !line_nu_all) return -1;
    memset(e, 0, sizeof(*e));
    size_t n = 0;
    for (int l = 0; l < n_lines; ++l) {
        if (!(line_nu_all[l] > 0.0) || !isfinite(line_nu_all[l])) return -1;
        if (line_jbar_frequency_in_bb_domain(line_nu_all[l])) ++n;
    }
    if (n == 0) return -1;
    e->n_q = n;
    e->line_id = malloc(n * sizeof(int));
    e->line_nu = malloc(n * sizeof(double));
    e->by_nu = malloc(n * sizeof(size_t));
    if (!e->line_id || !e->line_nu || !e->by_nu) {
        line_jbar_qset_free(e);
        return -1;
    }
    size_t k = 0;
    for (int l = 0; l < n_lines; ++l) {
        if (!line_jbar_frequency_in_bb_domain(line_nu_all[l])) continue;
        e->line_id[k] = l;
        e->line_nu[k] = line_nu_all[l];
        e->by_nu[k] = k;
        ++k;
    }
    qsort_r(e->by_nu, n, sizeof(size_t), cmp_by_nu, e->line_nu);
    memcpy(e->domain_contract_hash,
           LINE_JBAR_BB_DOMAIN_CONTRACT_SHA256, 65);
    Sha256 s;
    sha256_init(&s);
    const char *domain_tag = "BB_ENERGY_DOMAIN|";
    sha256_update(&s, domain_tag, strlen(domain_tag));
    sha256_update(&s, e->domain_contract_hash, 64);
    sha256_update(&s, e->line_id, n * sizeof(int));
    sha256_hex(&s, e->q_set_hash);
    e->profile_id = LINE_JBAR_PROFILE_GAUSS_VD10;
    memcpy(e->profile_hash, LINE_JBAR_PROFILE_SHA256, 65);
    e->set_kind = LINE_JBAR_SET_ENERGY_DOMAIN;
    return 0;
}

static int line_jbar_set_hash_matches(const LineJbarQSet *set,
                                      const char *tag)
{
    if (!set || !tag || !set->line_id || set->n_q == 0 ||
        strlen(set->q_set_hash) != 64 ||
        strcmp(set->domain_contract_hash,
               LINE_JBAR_BB_DOMAIN_CONTRACT_SHA256) != 0)
        return 0;
    Sha256 s;
    char actual[65];
    sha256_init(&s);
    sha256_update(&s, tag, strlen(tag));
    sha256_update(&s, set->domain_contract_hash, 64);
    sha256_update(&s, set->line_id, set->n_q * sizeof(int));
    sha256_hex(&s, actual);
    return strcmp(actual, set->q_set_hash) == 0;
}

LineJbarSubsetStatus line_jbar_qset_subset_of_eset(
        const LineJbarQSet *q, const LineJbarESet *e,
        size_t *first_missing_q)
{
    if (first_missing_q) *first_missing_q = SIZE_MAX;
    if (!q || !e || q->set_kind != LINE_JBAR_SET_RATE_GRAPH ||
        e->set_kind != LINE_JBAR_SET_ENERGY_DOMAIN ||
        !q->line_id || !q->line_nu || !e->line_id || !e->line_nu ||
        q->n_q == 0 || e->n_q == 0)
        return LINE_JBAR_SUBSET_INVALID;
    if (strcmp(q->domain_contract_hash, e->domain_contract_hash) != 0 ||
        q->profile_id != e->profile_id ||
        strcmp(q->profile_hash, e->profile_hash) != 0)
        return LINE_JBAR_SUBSET_IDENTITY_MISMATCH;
    if (!line_jbar_set_hash_matches(q, "BB_IN_DOMAIN|") ||
        !line_jbar_set_hash_matches(e, "BB_ENERGY_DOMAIN|"))
        return LINE_JBAR_SUBSET_HASH_MISMATCH;

    for (size_t i = 0; i < q->n_q; ++i)
        if (q->line_id[i] < 0 ||
            (i && q->line_id[i] <= q->line_id[i - 1]))
            return LINE_JBAR_SUBSET_INVALID;
    for (size_t i = 0; i < e->n_q; ++i)
        if (e->line_id[i] < 0 ||
            (i && e->line_id[i] <= e->line_id[i - 1]))
            return LINE_JBAR_SUBSET_INVALID;

    size_t ie = 0;
    for (size_t iq = 0; iq < q->n_q; ++iq) {
        while (ie < e->n_q && e->line_id[ie] < q->line_id[iq]) ++ie;
        if (ie == e->n_q || e->line_id[ie] != q->line_id[iq]) {
            if (first_missing_q) *first_missing_q = iq;
            return LINE_JBAR_SUBSET_MISSING_LINE;
        }
        if (e->line_nu[ie] != q->line_nu[iq]) {
            if (first_missing_q) *first_missing_q = iq;
            return LINE_JBAR_SUBSET_FREQUENCY_MISMATCH;
        }
    }
    return LINE_JBAR_SUBSET_OK;
}

void line_jbar_qset_free(LineJbarQSet *q)
{
    if (!q) return;
    free(q->line_id); free(q->line_nu); free(q->by_nu);
    memset(q, 0, sizeof(*q));
}

/* ---- accumulator / partial ---- */
int line_jbar_accumulator_init(LineJbarAccumulator *a, size_t n_q, size_t n_shells)
{
    if (!a || !n_q || !n_shells || n_shells > 255) return -1;
    if (n_q > SIZE_MAX / n_shells) return -1;
    size_t cells = n_q * n_shells;
    if (cells > SIZE_MAX / sizeof(double) ||
        cells > SIZE_MAX / sizeof(uint64_t))
        return -1;
    a->n_q = n_q; a->n_shells = n_shells; a->error_latch = 0;
    a->sum = calloc(cells, sizeof(double));
    a->sumsq = calloc(cells, sizeof(double));
    a->count = calloc(cells, sizeof(uint64_t));
    if (!a->sum || !a->sumsq || !a->count) { line_jbar_accumulator_free(a); return -1; }
    return 0;
}
void line_jbar_accumulator_free(LineJbarAccumulator *a)
{
    if (!a) return;
    free(a->sum); free(a->sumsq); free(a->count);
    memset(a, 0, sizeof(*a));
}
int line_jbar_partial_init(LineJbarPacketPartial *p)
{
    if (!p) return -1;
    p->capacity = 1024; p->used = 0;
    p->key = calloc(p->capacity, sizeof(uint64_t));
    p->value = calloc(p->capacity, sizeof(double));
    p->touched = malloc(p->capacity * sizeof(uint64_t));
    return (p->key && p->value && p->touched) ? 0 : -1;
}
void line_jbar_partial_free(LineJbarPacketPartial *p)
{
    if (!p) return;
    free(p->key); free(p->value); free(p->touched);
    memset(p, 0, sizeof(*p));
}
static int partial_grow(LineJbarPacketPartial *p)
{
    size_t nc = p->capacity * 2;
    uint64_t *nk = calloc(nc, sizeof(uint64_t));
    double *nv = calloc(nc, sizeof(double));
    uint64_t *nt = malloc(nc * sizeof(uint64_t));
    if (!nk || !nv || !nt) { free(nk); free(nv); free(nt); return -1; }
    for (size_t i = 0; i < p->used; i++) {
        uint64_t key = p->touched[i];
        size_t h = (key * 0x9e3779b97f4a7c15ull) & (p->capacity - 1);
        while (p->key[h] != key + 1) h = (h + 1) & (p->capacity - 1);
        size_t h2 = (key * 0x9e3779b97f4a7c15ull) & (nc - 1);
        while (nk[h2]) h2 = (h2 + 1) & (nc - 1);
        nk[h2] = key + 1; nv[h2] = p->value[h]; nt[i] = key;
    }
    free(p->key); free(p->value); free(p->touched);
    p->key = nk; p->value = nv; p->touched = nt; p->capacity = nc;
    return 0;
}
static int partial_add(LineJbarPacketPartial *p, uint64_t key, double w)
{
    if (p->used * 2 >= p->capacity && partial_grow(p) != 0) return -1;
    size_t h = (key * 0x9e3779b97f4a7c15ull) & (p->capacity - 1);
    while (p->key[h] && p->key[h] != key + 1) h = (h + 1) & (p->capacity - 1);
    if (!p->key[h]) { p->key[h] = key + 1; p->value[h] = 0.0;
                      p->touched[p->used++] = key; }
    p->value[h] += w;
    return 0;
}

/* ---- registered profile: normalized Gaussian, closed-form segment integral.
 * phi(nu) = exp(-x^2) / (sqrt(pi)*erf(4)*dnu_D), x=(nu-nu_l)/dnu_D, |x|<=4;
 * truncation renormalized so Integral phi dnu = 1 over the support.
 * Integral_0^1 (e0+de*t) phi(nu0+dnu*t) L dt has erf/exp closed form. */
double line_jbar_segment_phi_integral(double nu_line, double nu0, double nu1,
                                      double e0, double e1, double length)
{
    double dD = nu_line * (LINE_JBAR_VDOPPLER_CMS / C_CGS);
    double norm = sqrt(M_PI) * erf(LINE_JBAR_PROFILE_NDOPPLER) * dD;
    double x0 = (nu0 - nu_line) / dD, x1 = (nu1 - nu_line) / dD;
    double N = LINE_JBAR_PROFILE_NDOPPLER;
    if (x0 == x1) {                          /* static segment: phi * mean(e) */
        if (fabs(x0) > N) return 0.0;
        return length * 0.5 * (e0 + e1) * exp(-x0 * x0) / norm;
    }
    /* clip [x0,x1] (either order) to [-N, N] in t-space */
    double xa = x0 < x1 ? x0 : x1, xb = x0 < x1 ? x1 : x0;
    if (xb < -N || xa > N) return 0.0;
    double ca = xa < -N ? -N : xa, cb = xb > N ? N : xb;
    /* t at clipped x (t along original direction x0->x1) */
    double ta = (ca - x0) / (x1 - x0), tb = (cb - x0) / (x1 - x0);
    if (ta > tb) { double tmp = ta; ta = tb; tb = tmp; }
    /* Integral phi dt = (erf(xB)-erf(xA)) / ((x1-x0)*...) with sign; and
     * Integral t*phi dt via the Gaussian first moment. */
    double xA = x0 + (x1 - x0) * ta, xB = x0 + (x1 - x0) * tb;
    double inv = 1.0 / (x1 - x0);
    double I0 = 0.5 * sqrt(M_PI) * (erf(xB) - erf(xA)) * inv;      /* ∫e^{-x²}dt */
    double I1x = -0.5 * (exp(-xB * xB) - exp(-xA * xA)) * inv;     /* ∫x e^{-x²}dt */
    /* t = (x - x0)/(x1-x0)  =>  ∫t e^{-x²} dt = (I1x - x0*I0) * inv */
    double I1 = (I1x - x0 * I0) * inv;
    double de = e1 - e0;
    return length * (e0 * I0 + de * I1) / norm;
}

int line_jbar_segment_add(const LineJbarQSet *q, LineJbarPacketPartial *p,
                          int shell, double nu0, double nu1,
                          double e0, double e1, double length)
{
    if (!q || !p || shell < 0 || shell > 255 ||
        !(nu0 > 0.0) || !(nu1 > 0.0) || !isfinite(e0) || !isfinite(e1) ||
        !(length >= 0.0) || !isfinite(length))
        return -1;
    double lo = nu0 < nu1 ? nu0 : nu1, hi = nu0 < nu1 ? nu1 : nu0;
    /* widen by max profile half-width (relative: 4*v_D/c) */
    double wrel = LINE_JBAR_PROFILE_NDOPPLER * LINE_JBAR_VDOPPLER_CMS / C_CGS;
    double qlo = lo * (1.0 - wrel), qhi = hi * (1.0 + wrel);
    /* binary search ascending line_nu permutation */
    size_t a = 0, b = q->n_q;
    while (a < b) { size_t m = (a + b) / 2;
        if (q->line_nu[q->by_nu[m]] < qlo) a = m + 1; else b = m; }
    for (size_t i = a; i < q->n_q; i++) {
        size_t idx = q->by_nu[i];
        double nul = q->line_nu[idx];
        if (nul > qhi) break;
        double w = line_jbar_segment_phi_integral(nul, nu0, nu1, e0, e1, length);
        if (!isfinite(w)) return -1;
        if (w != 0.0 &&
            partial_add(p, ((uint64_t)idx << 8) | (uint64_t)shell, w) != 0)
            return -1;
    }
    return 0;
}

int line_jbar_packet_flush(LineJbarAccumulator *a, LineJbarPacketPartial *p)
{
    if (!a || !p) return -1;
    int rc = 0;
    for (size_t i = 0; i < p->used; i++) {
        uint64_t key = p->touched[i];
        size_t h = (key * 0x9e3779b97f4a7c15ull) & (p->capacity - 1);
        while (p->key[h] != key + 1) h = (h + 1) & (p->capacity - 1);
        double y = p->value[h];
        size_t idx = (size_t)(key >> 8);
        size_t shell = (size_t)(key & 0xff);
        if (idx >= a->n_q || shell >= a->n_shells) { rc = -1; break; }
        size_t cell = idx * a->n_shells + shell;
#ifdef _OPENMP
#pragma omp atomic
#endif
        a->sum[cell] += y;
#ifdef _OPENMP
#pragma omp atomic
#endif
        a->sumsq[cell] += y * y;
#ifdef _OPENMP
#pragma omp atomic
#endif
        a->count[cell] += 1;
        p->key[h] = 0;
    }
    if (rc != 0) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
        a->error_latch = 1;
    }
    p->used = 0;
    return rc;
}
