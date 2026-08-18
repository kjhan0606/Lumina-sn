#ifndef LUMINA_CMF_ERROR_ENVELOPE_H
#define LUMINA_CMF_ERROR_ENVELOPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CMF_ENVELOPE_OK = 0,
    CMF_ENVELOPE_INVALID_INPUT,
    CMF_ENVELOPE_ALLOCATION_FAILED,
    CMF_ENVELOPE_OPERATOR_FAILED,
    CMF_ENVELOPE_OPERATOR_NEGATIVE,
    CMF_ENVELOPE_NOT_SUPERSOLUTION,
    CMF_ENVELOPE_NOT_MONOTONE,
    CMF_ENVELOPE_NONFINITE
} CMFEnvelopeStatus;

/* The callback must evaluate the same nonnegative floating operator K used by
 * the production fixed-point solve and return a componentwise upper bound on
 * K*input.  The verifier checks finite/nonnegative output, but the callback's
 * outward-rounding implementation remains part of its producer contract. */
typedef int (*CMFEnvelopeApplyUpper)(const double *input,
                                    double *upper_output,
                                    size_t n,
                                    void *context);

typedef struct {
    CMFEnvelopeStatus status;
    size_t first_bad_component;
    size_t iterations_completed;
    double candidate;
    double required_upper;
    double minimum_margin;
} CMFEnvelopeReport;

/* Prove candidate >= residual_upper + K*candidate componentwise.  A PASS is
 * an a-posteriori proof; no convergence or tolerance claim is involved. */
CMFEnvelopeStatus cmf_error_envelope_verify(
    size_t n, const double *residual_upper, const double *candidate,
    CMFEnvelopeApplyUpper apply_upper, void *context,
    CMFEnvelopeReport *report);

/* Tighten an already verified supersolution from above.  Each proposed step
 * is residual_upper + K*current, must be componentwise non-increasing, and is
 * independently re-verified before it replaces current.  Physical J and the
 * signed line values are not inputs and cannot be changed by this routine. */
CMFEnvelopeStatus cmf_error_envelope_refine(
    size_t n, const double *residual_upper, double *candidate,
    size_t iterations, CMFEnvelopeApplyUpper apply_upper, void *context,
    CMFEnvelopeReport *report);

const char *cmf_error_envelope_status_name(CMFEnvelopeStatus status);

#ifdef __cplusplus
}
#endif

#endif
