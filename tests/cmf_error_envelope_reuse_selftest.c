/* 이전 동결 구현과 상계/판정/보고서가 같은지 검사한다. 생산 J는 입력하지 않는다. */
#include "cmf_error_envelope.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern CMFEnvelopeStatus baseline_cmf_error_envelope_refine(
    size_t, const double *, double *, size_t, CMFEnvelopeApplyUpper,
    void *, CMFEnvelopeReport *);

typedef struct {
    double matrix[16];
    size_t calls;
    int failure;
} Operator;

static int apply(const double *x, double *y, size_t n, void *opaque)
{
    Operator *op = opaque;
    if (n != 4) return -1;
    op->calls++;
    if (op->failure && x[0] < 2.0) {
        if (op->failure == 1) return -1;
        for (size_t i = 0; i < n; i++) y[i] = 0;
        y[0] = op->failure == 2 ? NAN : -1;
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        long double sum = 0;
        for (size_t j = 0; j < n; j++) sum += (long double)op->matrix[i*n+j]*x[j];
        y[i] = (double)sum;
        if ((long double)y[i] < sum) y[i] = nextafter(y[i], INFINITY);
    }
    return 0;
}

static int same_double(double a, double b)
{
    return (isnan(a) && isnan(b)) || memcmp(&a, &b, sizeof(a)) == 0;
}

static int same_report(const CMFEnvelopeReport *a, const CMFEnvelopeReport *b)
{
    return a->status == b->status && a->first_bad_component == b->first_bad_component &&
        a->iterations_completed == b->iterations_completed &&
        same_double(a->candidate,b->candidate) &&
        same_double(a->required_upper,b->required_upper) &&
        same_double(a->minimum_margin,b->minimum_margin);
}

int main(void)
{
    const size_t steps[] = {0,1,2,8,24,42,64};
    uint32_t rng = 107;
    size_t cases = 0;
    for (int fixture = 0; fixture < 100; fixture++) {
        Operator original = {{0},0,0};
        for (int i = 0; i < 16; i++) {
            rng = rng*1664525U+1013904223U;
            original.matrix[i] = (double)(rng%16)/128;
        }
        for (size_t k = 0; k < sizeof(steps)/sizeof(*steps); k++) {
            const double r[4] = {0.5,0.25,0.125,0.0625};
            double old[4] = {4,4,4,4}, fresh[4] = {4,4,4,4};
            Operator before = original, after = original;
            CMFEnvelopeReport a,b;
            CMFEnvelopeStatus sa = baseline_cmf_error_envelope_refine(
                4,r,old,steps[k],apply,&before,&a);
            CMFEnvelopeStatus sb = cmf_error_envelope_refine(
                4,r,fresh,steps[k],apply,&after,&b);
            if (sa != CMF_ENVELOPE_OK || sa != sb || memcmp(old,fresh,sizeof(old)) ||
                !same_report(&a,&b) || before.calls != 2*steps[k]+1 ||
                after.calls != steps[k]+1) {
                fprintf(stderr,"REUSE_FAIL fixture=%d steps=%zu old=%d new=%d calls=%zu/%zu\n",
                    fixture,steps[k],sa,sb,before.calls,after.calls);
                return 1;
            }
            cases++;
        }
    }
    /* 다음 후보의 연산 실패/NaN/음수는 게시 전에 거부되어야 한다. */
    for (int fail = 1; fail <= 3; fail++) {
        Operator before = {{0},0,fail}, after = {{0},0,fail};
        for (int i=0;i<4;i++) before.matrix[5*i]=after.matrix[5*i]=0.25;
        const double r[4] = {0.5,0.5,0.5,0.5};
        const double seed[4] = {4,4,4,4};
        double old[4],fresh[4];memcpy(old,seed,sizeof(old));memcpy(fresh,seed,sizeof(fresh));
        CMFEnvelopeReport a,b;
        CMFEnvelopeStatus sa=baseline_cmf_error_envelope_refine(4,r,old,42,apply,&before,&a);
        CMFEnvelopeStatus sb=cmf_error_envelope_refine(4,r,fresh,42,apply,&after,&b);
        CMFEnvelopeStatus expected=fail==1?CMF_ENVELOPE_OPERATOR_FAILED:
            (fail==2?CMF_ENVELOPE_NONFINITE:CMF_ENVELOPE_OPERATOR_NEGATIVE);
        if(sa!=expected||sa!=sb||memcmp(old,fresh,sizeof(old))||
           memcmp(seed,fresh,sizeof(seed))||!same_report(&a,&b))return 2;
        cases++;
    }
    puts("CMF_ENVELOPE_REUSE_PASS cases=703 byte_identical_bounds=1 identical_reports=1 calls_at42=85_to43 physical_J_changed=0");
    return cases==703?0:3;
}
