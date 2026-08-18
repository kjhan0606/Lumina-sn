#include "cmf_error_envelope.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void report_init(CMFEnvelopeReport *report)
{
    if (!report) return;
    memset(report, 0, sizeof(*report));
    report->status = CMF_ENVELOPE_INVALID_INPUT;
    report->first_bad_component = SIZE_MAX;
    report->candidate = NAN;
    report->required_upper = NAN;
    report->minimum_margin = INFINITY;
}

const char *cmf_error_envelope_status_name(CMFEnvelopeStatus status)
{
    switch (status) {
    case CMF_ENVELOPE_OK: return "OK";
    case CMF_ENVELOPE_INVALID_INPUT: return "INVALID_INPUT";
    case CMF_ENVELOPE_ALLOCATION_FAILED: return "ALLOCATION_FAILED";
    case CMF_ENVELOPE_OPERATOR_FAILED: return "OPERATOR_FAILED";
    case CMF_ENVELOPE_OPERATOR_NEGATIVE: return "OPERATOR_NEGATIVE";
    case CMF_ENVELOPE_NOT_SUPERSOLUTION: return "NOT_SUPERSOLUTION";
    case CMF_ENVELOPE_NOT_MONOTONE: return "NOT_MONOTONE";
    case CMF_ENVELOPE_NONFINITE: return "NONFINITE";
    default: return "UNKNOWN";
    }
}

/* Error-free TwoSum decomposition under the default round-to-nearest mode.
 * For nonnegative finite inputs, incrementing only when the exact residual is
 * positive returns the least adjacent binary64 value known to be no smaller
 * than the exact sum.  This changes an error bound, never a physical field. */
static int add_nonnegative_up(double a, double b, double *upper)
{
    if (!upper || !(a >= 0.0) || !(b >= 0.0) ||
        !isfinite(a) || !isfinite(b)) return -1;
    double sum = a + b;
    if (!isfinite(sum)) return -1;
    double b_virtual = sum - a;
    double error = (a - (sum - b_virtual)) + (b - b_virtual);
    if (error > 0.0) sum = nextafter(sum, INFINITY);
    if (!isfinite(sum)) return -1;
    *upper = sum;
    return 0;
}

static CMFEnvelopeStatus validate_vectors(size_t n,
                                          const double *residual_upper,
                                          const double *candidate,
                                          CMFEnvelopeReport *report)
{
    if (n == 0 || !residual_upper || !candidate) {
        if (report) report->status = CMF_ENVELOPE_INVALID_INPUT;
        return CMF_ENVELOPE_INVALID_INPUT;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!(residual_upper[i] >= 0.0) || !(candidate[i] >= 0.0) ||
            !isfinite(residual_upper[i]) || !isfinite(candidate[i])) {
            if (report) {
                report->status = CMF_ENVELOPE_NONFINITE;
                report->first_bad_component = i;
                report->candidate = candidate[i];
                report->required_upper = residual_upper[i];
            }
            return CMF_ENVELOPE_NONFINITE;
        }
    }
    return CMF_ENVELOPE_OK;
}

CMFEnvelopeStatus cmf_error_envelope_verify(
    size_t n, const double *residual_upper, const double *candidate,
    CMFEnvelopeApplyUpper apply_upper, void *context,
    CMFEnvelopeReport *report)
{
    report_init(report);
    if (!apply_upper) return CMF_ENVELOPE_INVALID_INPUT;
    CMFEnvelopeStatus status =
        validate_vectors(n, residual_upper, candidate, report);
    if (status != CMF_ENVELOPE_OK) return status;
    if (n > SIZE_MAX / sizeof(double)) {
        if (report) report->status = CMF_ENVELOPE_ALLOCATION_FAILED;
        return CMF_ENVELOPE_ALLOCATION_FAILED;
    }
    double *ku = (double *)malloc(n * sizeof(*ku));
    if (!ku) {
        if (report) report->status = CMF_ENVELOPE_ALLOCATION_FAILED;
        return CMF_ENVELOPE_ALLOCATION_FAILED;
    }
    if (apply_upper(candidate, ku, n, context) != 0) {
        free(ku);
        if (report) report->status = CMF_ENVELOPE_OPERATOR_FAILED;
        return CMF_ENVELOPE_OPERATOR_FAILED;
    }
    double minimum_margin = INFINITY;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(ku[i])) {
            status = CMF_ENVELOPE_NONFINITE;
        } else if (ku[i] < 0.0) {
            status = CMF_ENVELOPE_OPERATOR_NEGATIVE;
        } else {
            double required;
            if (add_nonnegative_up(residual_upper[i], ku[i],
                                   &required) != 0) {
                status = CMF_ENVELOPE_NONFINITE;
            } else {
                double margin = candidate[i] - required;
                if (margin < minimum_margin) minimum_margin = margin;
                if (candidate[i] >= required) continue;
                status = CMF_ENVELOPE_NOT_SUPERSOLUTION;
                if (report) report->required_upper = required;
            }
        }
        if (report) {
            report->status = status;
            report->first_bad_component = i;
            report->candidate = candidate[i];
            if (status != CMF_ENVELOPE_NOT_SUPERSOLUTION)
                report->required_upper = ku[i];
            report->minimum_margin = minimum_margin;
        }
        free(ku);
        return status;
    }
    free(ku);
    if (report) {
        report->status = CMF_ENVELOPE_OK;
        report->minimum_margin = minimum_margin;
    }
    return CMF_ENVELOPE_OK;
}

CMFEnvelopeStatus cmf_error_envelope_refine(
    size_t n, const double *residual_upper, double *candidate,
    size_t iterations, CMFEnvelopeApplyUpper apply_upper, void *context,
    CMFEnvelopeReport *report)
{
    report_init(report);
    if (!apply_upper) return CMF_ENVELOPE_INVALID_INPUT;
    CMFEnvelopeReport verify_report;
    CMFEnvelopeStatus status = cmf_error_envelope_verify(
        n, residual_upper, candidate, apply_upper, context, &verify_report);
    if (status != CMF_ENVELOPE_OK) {
        if (report) *report = verify_report;
        return status;
    }
    if (n > SIZE_MAX / (2U * sizeof(double))) {
        if (report) report->status = CMF_ENVELOPE_ALLOCATION_FAILED;
        return CMF_ENVELOPE_ALLOCATION_FAILED;
    }
    double *ku = (double *)malloc(2U * n * sizeof(*ku));
    if (!ku) {
        if (report) report->status = CMF_ENVELOPE_ALLOCATION_FAILED;
        return CMF_ENVELOPE_ALLOCATION_FAILED;
    }
    double *next = ku + n;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        if (apply_upper(candidate, ku, n, context) != 0) {
            status = CMF_ENVELOPE_OPERATOR_FAILED;
            break;
        }
        for (size_t i = 0; i < n; ++i) {
            if (!isfinite(ku[i])) {
                status = CMF_ENVELOPE_NONFINITE;
                if (report) report->first_bad_component = i;
                break;
            }
            if (ku[i] < 0.0) {
                status = CMF_ENVELOPE_OPERATOR_NEGATIVE;
                if (report) report->first_bad_component = i;
                break;
            }
            if (add_nonnegative_up(residual_upper[i], ku[i], &next[i]) != 0) {
                status = CMF_ENVELOPE_NONFINITE;
                if (report) report->first_bad_component = i;
                break;
            }
            if (next[i] > candidate[i]) {
                status = CMF_ENVELOPE_NOT_MONOTONE;
                if (report) {
                    report->first_bad_component = i;
                    report->candidate = candidate[i];
                    report->required_upper = next[i];
                }
                break;
            }
        }
        if (status != CMF_ENVELOPE_OK) break;
        status = cmf_error_envelope_verify(
            n, residual_upper, next, apply_upper, context, &verify_report);
        if (status != CMF_ENVELOPE_OK) {
            if (report) *report = verify_report;
            break;
        }
        memcpy(candidate, next, n * sizeof(*candidate));
        if (report) report->iterations_completed = iteration + 1U;
    }
    free(ku);
    if (report) {
        report->status = status;
        if (status == CMF_ENVELOPE_OK)
            report->minimum_margin = verify_report.minimum_margin;
    }
    return status;
}
