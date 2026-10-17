#include "cmf_error_envelope.h"

#include <math.h>
#include <stdio.h>

static int failures;
#define CHECK(condition, label) do { if (!(condition)) { \
    fprintf(stderr, "CMF_ERROR_ENVELOPE_FAIL %s line=%d\n", \
            label, __LINE__); \
    ++failures; \
} } while (0)

typedef struct {
    double k00, k01, k10, k11;
} Matrix2;

static int matrix2_apply_upper(const double *input, double *output,
                               size_t n, void *context)
{
    const Matrix2 *m = (const Matrix2 *)context;
    if (!input || !output || n != 2 || !m) return -1;
    /* All production known-answer coefficients and inputs here are powers of
     * two, so these products/sums are exact binary64 upper bounds. */
    output[0] = m->k00 * input[0] + m->k01 * input[1];
    output[1] = m->k10 * input[0] + m->k11 * input[1];
    return 0;
}

static int failing_apply(const double *input, double *output,
                         size_t n, void *context)
{
    (void)input; (void)output; (void)n; (void)context;
    return -1;
}

int main(void)
{
    const Matrix2 positive = {0.25, 0.25, 0.0, 0.0};
    const double residual[2] = {0.5, 1.0};
    CMFEnvelopeReport report;

    /* (I-K)^-1 residual = [1,1].  A verified global seed refines from above
     * without changing the fixed point or invoking a sign tolerance. */
    double seed[2] = {2.0, 2.0};
    CHECK(cmf_error_envelope_verify(
              2, residual, seed, matrix2_apply_upper,
              (void *)&positive, &report) == CMF_ENVELOPE_OK,
          "global-seed-verified");
    CHECK(cmf_error_envelope_refine(
              2, residual, seed, 24, matrix2_apply_upper,
              (void *)&positive, &report) == CMF_ENVELOPE_OK,
          "refine-from-above");
    CHECK(seed[0] >= 1.0 && seed[1] == 1.0 && seed[0] < 1.000000000001,
          "componentwise-known-answer");

    /* The verifier has teeth: the exact envelope, shrunk by one binary64
     * step, must fail rather than becoming a tunable tolerance. */
    double exact[2] = {1.0, 1.0};
    CHECK(cmf_error_envelope_verify(
              2, residual, exact, matrix2_apply_upper,
              (void *)&positive, &report) == CMF_ENVELOPE_OK,
          "exact-envelope-pass");
    exact[0] = nextafter(exact[0], 0.0);
    CHECK(cmf_error_envelope_verify(
              2, residual, exact, matrix2_apply_upper,
              (void *)&positive, &report) ==
              CMF_ENVELOPE_NOT_SUPERSOLUTION &&
          report.first_bad_component == 0,
          "shrunk-envelope-fail");

    /* A zero-iteration lower candidate is not silently accepted. */
    double zero_iteration[2] = {residual[0], residual[1]};
    CHECK(cmf_error_envelope_verify(
              2, residual, zero_iteration, matrix2_apply_upper,
              (void *)&positive, &report) ==
              CMF_ENVELOPE_NOT_SUPERSOLUTION,
          "zero-iteration-fail");

    /* The tempting local diagonal formula misses K01*e1: [2/3,1] is below
     * the true first-component error 1 and must be rejected. */
    double local_diagonal[2] = {2.0 / 3.0, 1.0};
    CHECK(cmf_error_envelope_verify(
              2, residual, local_diagonal, matrix2_apply_upper,
              (void *)&positive, &report) ==
              CMF_ENVELOPE_NOT_SUPERSOLUTION &&
          report.first_bad_component == 0,
          "local-diagonal-counterexample");

    /* Rounding-direction boundary: round-to-nearest loses +2^-54 next to 1
     * and would falsely accept u=1.  The verifier's outward TwoSum addition
     * must retain that positive remainder and reject the candidate. */
    const Matrix2 rounding = {0x1p-54, 0.0, 0.0, 0.0};
    const double rounding_residual[2] = {1.0, 0.0};
    double rounding_candidate[2] = {1.0, 0.0};
    CHECK(rounding_residual[0] +
              rounding.k00 * rounding_candidate[0] == 1.0,
          "round-to-nearest-witness");
    CHECK(cmf_error_envelope_verify(
              2, rounding_residual, rounding_candidate,
              matrix2_apply_upper, (void *)&rounding, &report) ==
              CMF_ENVELOPE_NOT_SUPERSOLUTION &&
          report.first_bad_component == 0 &&
          report.required_upper == nextafter(1.0, INFINITY),
          "outward-addition-rejects-false-pass");

    const Matrix2 negative = {0.25, -0.5, 0.0, 0.0};
    /* 행 합 노름>=1은 성분별 상계의 부재를 뜻하지 않는다.
     * K^2=0, ||K||inf=2인 양의 연산자에서 (I-K)^-1*[1,1]=[3,1].
     * 실제 GPU 실패 원인을 단정하는 시험이 아니라 seed 선택의 반례다. */
    const Matrix2 noncontractive_norm = {0.0, 2.0, 0.0, 0.0};
    const double unit_residual[2] = {1.0, 1.0};
    double shaped[2] = {3.0, 1.0};
    CHECK(cmf_error_envelope_verify(
              2, unit_residual, shaped, matrix2_apply_upper,
              (void *)&noncontractive_norm, &report) == CMF_ENVELOPE_OK,
          "noncontractive-infinity-norm-valid-envelope");
    CHECK(cmf_error_envelope_refine(
              2, unit_residual, shaped, 2, matrix2_apply_upper,
              (void *)&noncontractive_norm, &report) == CMF_ENVELOPE_OK &&
          shaped[0] == 3.0 && shaped[1] == 1.0,
          "noncontractive-infinity-norm-exact-envelope-preserved");
    shaped[0] = nextafter(3.0, 0.0);
    CHECK(cmf_error_envelope_verify(
              2, unit_residual, shaped, matrix2_apply_upper,
              (void *)&noncontractive_norm, &report) == CMF_ENVELOPE_NOT_SUPERSOLUTION,
          "noncontractive-infinity-norm-shrunk-envelope-rejected");
    /* 실질적인 불안정성은 거부한다. K00=1과 양의 잔차에는 유한 상계가 없다. */
    const Matrix2 no_envelope = {1.0, 0.0, 0.0, 0.0};
    shaped[0] = 4.0;
    CHECK(cmf_error_envelope_verify(
              2, unit_residual, shaped, matrix2_apply_upper,
              (void *)&no_envelope, &report) == CMF_ENVELOPE_NOT_SUPERSOLUTION,
          "unit-feedback-positive-residual-rejected");
    /* 같은 비정규 행렬에서 가중 seed 구성과 기존 검증을 연결한다. */
    double kones[2] = {2.0, 0.0}, weighted_seed[2], q = INFINITY;
    size_t weight_steps = 0;
    CHECK(cmf_error_envelope_weighted_seed(
              2, unit_residual, kones, 4, matrix2_apply_upper,
              (void *)&noncontractive_norm, weighted_seed, &q, &weight_steps) == CMF_ENVELOPE_OK &&
          q > 0.0 && q < 1.0 && weight_steps == 1,
          "weighted-seed-contraction-proved");
    CHECK(cmf_error_envelope_verify(2, unit_residual, weighted_seed,
              matrix2_apply_upper, (void *)&noncontractive_norm, &report) == CMF_ENVELOPE_OK,
          "weighted-seed-componentwise-verified");
    CHECK(cmf_error_envelope_refine(2, unit_residual, weighted_seed, 2,
              matrix2_apply_upper, (void *)&noncontractive_norm, &report) == CMF_ENVELOPE_OK &&
          weighted_seed[0] == 3.0 && weighted_seed[1] == 1.0,
          "weighted-seed-exact-answer-after-refinement");
    CHECK(cmf_error_envelope_weighted_seed(
              2, unit_residual, kones, 4, matrix2_apply_upper,
              (void *)&noncontractive_norm, kones, &q, NULL) == CMF_ENVELOPE_OK,
          "weighted-seed-kones-alias-supported");
    CHECK(cmf_error_envelope_weighted_seed(
              2, unit_residual, kones, 4, matrix2_apply_upper,
              (void *)&no_envelope, weighted_seed, &q, &weight_steps) == CMF_ENVELOPE_NOT_SUPERSOLUTION &&
          q >= 1.0 && weight_steps == 4,
          "weighted-seed-unit-feedback-rejected");
    CHECK(cmf_error_envelope_weighted_seed(
              2, unit_residual, kones, 4, failing_apply,
              NULL, weighted_seed, &q, NULL) == CMF_ENVELOPE_OPERATOR_FAILED,
          "weighted-seed-failed-operator-rejected");
    const Matrix2 all_negative = {-1.0, 0.0, 0.0, 0.0};
    CHECK(cmf_error_envelope_weighted_seed(
              2, unit_residual, kones, 4, matrix2_apply_upper,
              (void *)&all_negative, weighted_seed, &q, NULL) == CMF_ENVELOPE_OPERATOR_NEGATIVE,
          "weighted-seed-negative-operator-rejected");
    const double zero_residual[2] = {0.0, 0.0};
    CHECK(cmf_error_envelope_weighted_seed(
              2, zero_residual, kones, 4, matrix2_apply_upper,
              (void *)&noncontractive_norm, weighted_seed, &q, NULL) == CMF_ENVELOPE_OK &&
          weighted_seed[0] == 0.0 && weighted_seed[1] == 0.0,
          "weighted-seed-zero-residual-exact");
    double negative_candidate[2] = {4.0, 4.0};
    CHECK(cmf_error_envelope_verify(
              2, residual, negative_candidate, matrix2_apply_upper,
              (void *)&negative, &report) ==
              CMF_ENVELOPE_OPERATOR_NEGATIVE,
          "negative-operator-fail");
    CHECK(cmf_error_envelope_verify(
              2, residual, negative_candidate, failing_apply,
              NULL, &report) == CMF_ENVELOPE_OPERATOR_FAILED,
          "operator-failure-preserved");

    if (failures) return 1;
    printf("CMF_ERROR_ENVELOPE_SELFTEST PASS known=[%.17g,%.17g] "
           "floor=0 clamp=0 jitter=0\n", seed[0], seed[1]);
    return 0;
}
