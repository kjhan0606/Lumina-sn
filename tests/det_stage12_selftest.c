#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "radeq_publication.h"
#include "lumina.h"

static int expect_ok(const char *name, const char *reason)
{
    if (reason) {
        fprintf(stderr, "FAIL %s: unexpected reason=%s\n", name, reason);
        return 0;
    }
    return 1;
}

static int expect_reason(
    const char *name,
    const char *actual,
    const char *expected)
{
    if (!actual || strcmp(actual, expected) != 0) {
        fprintf(stderr,
                "FAIL %s: got=%s expected=%s\n",
                name,
                actual ? actual : "(null)",
                expected);
        return 0;
    }
    return 1;
}

static ElectronTemperaturePublication fixed_publication(size_t shells)
{
    ElectronTemperaturePublication te = {0};
    static const char hash[] =
        "0123456789abcdef0123456789abcdef"
        "0123456789abcdef0123456789abcdef";

    te.te_lane = A210_TE_LANE_FIXED_T;
    memcpy(te.te_profile_sha256, hash, sizeof(hash));
    te.pinned_shells = shells;
    te.re_root_required = 0;
    return te;
}

/* Exercise the production loader through a real pathname, while keeping each
 * profile ephemeral and removing it after the assertion.  The range check is
 * the production caller's next fail-closed step after a successful load. */
static const char *load_profile_text(const char *text, int check_domain)
{
    char path[] = "/tmp/lumina-fixed-te-selftest-XXXXXX";
    int fd = mkstemp(path);
    FILE *stream;
    double *profile = NULL;
    char hash[65] = {0};
    double minimum = 0.0, maximum = 0.0;
    const char *reason;

    if (fd < 0)
        return "SELFTEST_PROFILE_TEMPFILE_FAILED";
    stream = fdopen(fd, "w");
    if (!stream) {
        close(fd);
        unlink(path);
        return "SELFTEST_PROFILE_TEMPFILE_FAILED";
    }
    if (fputs(text, stream) == EOF || fclose(stream) != 0) {
        unlink(path);
        return "SELFTEST_PROFILE_TEMPFILE_FAILED";
    }

    reason = a210_fixed_te_profile_load(
        path, 2, &profile, hash, &minimum, &maximum);
    if (!reason && check_domain)
        reason = a210_fixed_te_profile_validate(
            profile, 2, 2, A210_PRODUCTION_TE_MIN_K,
            A210_PRODUCTION_TE_MAX_K);
    free(profile);
    unlink(path);
    return reason;
}

int main(void)
{
    const double te_min = 3500.0;
    const double te_max = 140000.0;
    const double physical[2] = {10000.0, 12000.0};
    int positive = 0;

    /* DET-SPRIM NC-C1..C3: every injected failure has a named observation,
     * followed by the same production validator passing after removal. */
    {
        int defined = 0, srce_chk = 0, exact_zero = 0, enabled = 0;
        const char *reason = a210_sproducer_raw_decode(
            1, 0, 2.0, 0.25, 0, &defined, &srce_chk, &exact_zero);
        if (reason || defined) {
            fprintf(stderr, "FAIL NC-C1: stride mismatch did not fail closed\n");
            return 1;
        }
        printf("NC-C1 inject=STRIDE_MISMATCH status=FAIL "
               "reason=SPRODUCER_RAW_UNAVAILABLE\n");
        reason = a210_sproducer_raw_decode(
            1, 1, 2.0, 0.25, 0, &defined, &srce_chk, &exact_zero);
        if (reason || !defined) {
            fprintf(stderr, "FAIL NC-C1: removal did not restore fields\n");
            return 1;
        }
        printf("NC-C1 remove=STRIDE_MATCH status=PASS\n");

        reason = a210_sproducer_capture_request_parse("2", &enabled);
        if (!reason || strcmp(reason, "INVALID_SPRODUCER_CAPTURE_REQUEST") != 0) {
            fprintf(stderr, "FAIL NC-C2: capture value 2 was accepted\n");
            return 1;
        }
        printf("NC-C2 inject=SPRODUCER_CAPTURE_2 status=FAIL reason=%s\n",
               reason);
        reason = a210_sproducer_capture_request_parse("1", &enabled);
        if (reason || enabled != 1) {
            fprintf(stderr, "FAIL NC-C2: capture value 1 was rejected\n");
            return 1;
        }
        printf("NC-C2 remove=SPRODUCER_CAPTURE_1 status=PASS\n");

        reason = a210_sproducer_raw_decode(
            1, 1, 2.0, 0.25, 4, &defined, &srce_chk, &exact_zero);
        if (!reason || strcmp(reason, "INVALID_SPRODUCER_PROVENANCE") != 0) {
            fprintf(stderr, "FAIL NC-C3: provenance bit pollution was accepted\n");
            return 1;
        }
        printf("NC-C3 inject=PROVENANCE_BIT2 status=FAIL reason=%s\n", reason);
        reason = a210_sproducer_raw_decode(
            1, 1, 2.0, 0.25, 3, &defined, &srce_chk, &exact_zero);
        if (reason || !defined || !srce_chk || !exact_zero) {
            fprintf(stderr, "FAIL NC-C3: valid provenance was rejected\n");
            return 1;
        }
        printf("NC-C3 remove=PROVENANCE_BITS_0_1 status=PASS\n");

        reason = a210_sproducer_raw_decode(
            1, 1, -1.0, -1.0, UINT8_MAX,
            &defined, &srce_chk, &exact_zero);
        if (reason || defined) {
            fprintf(stderr, "FAIL DET-SPRIM sentinel was not unavailable\n");
            return 1;
        }
        reason = a210_sproducer_raw_decode(
            1, 1, 0.0, -0.5, 0, &defined, &srce_chk, &exact_zero);
        if (reason || !defined) {
            fprintf(stderr, "FAIL DET-SPRIM legitimate boundary hit sentinel\n");
            return 1;
        }
        printf("DET-SPRIM-SENTINEL status=PASS eta_min=0 tau_eff_min=-0.5 "
               "double_sentinel=-1 provenance_sentinel=255 collision=0\n");
        ++positive;
    }

    /* NL1: missing publication field; positive control first. */
    {
        ElectronTemperaturePublication te = fixed_publication(2);

        if (!expect_ok(
                "NL1-positive",
                a210_temperature_publication_validate(&te, 2)))
            return 1;
        ++positive;

        te.re_root_required = 1;
        if (!expect_reason(
                "NL1-negative",
                a210_temperature_publication_validate(&te, 2),
                "RADEQ_FIXED_T_PUBLICATION_INCOMPLETE"))
            return 1;
    }

    /* NL2: shell count mismatch. */
    {
        if (!expect_ok(
                "NL2-positive",
                a210_fixed_te_profile_validate(
                    physical, 2, 2, te_min, te_max)))
            return 1;
        ++positive;

        if (!expect_reason(
                "NL2-negative",
                a210_fixed_te_profile_validate(
                    physical, 1, 2, te_min, te_max),
                "RADEQ_FIXED_T_SHELL_COUNT_MISMATCH"))
            return 1;
    }

    /* NL3: negative, zero, and non-finite values. */
    {
        const double negative[2] = {-1.0, 12000.0};
        const double zero[2] = {0.0, 12000.0};
        const double nonfinite[2] = {NAN, INFINITY};

        if (!expect_ok(
                "NL3-positive",
                a210_fixed_te_profile_validate(
                    physical, 2, 2, te_min, te_max)))
            return 1;
        ++positive;

        if (!expect_reason(
                "NL3-negative-negative",
                a210_fixed_te_profile_validate(
                    negative, 2, 2, te_min, te_max),
                "RADEQ_FIXED_T_NONPHYSICAL_PROFILE"))
            return 1;

        if (!expect_reason(
                "NL3-negative-zero",
                a210_fixed_te_profile_validate(
                    zero, 2, 2, te_min, te_max),
                "RADEQ_FIXED_T_NONPHYSICAL_PROFILE"))
            return 1;

        if (!expect_reason(
                "NL3-negative-nonfinite",
                a210_fixed_te_profile_validate(
                    nonfinite, 2, 2, te_min, te_max),
                "RADEQ_FIXED_T_NONPHYSICAL_PROFILE"))
            return 1;
    }

    /* NL4: fixed-T fields must not leak into FREE_T. */
    {
        ElectronTemperaturePublication te = {0};

        te.te_lane = A210_TE_LANE_FREE_T;
        te.re_root_required = 1;

        if (!expect_ok(
                "NL4-positive",
                a210_temperature_publication_validate(&te, 2)))
            return 1;

        if (a210_te_manifest_has_fixed_fields(&te)) {
            fprintf(stderr, "FAIL NL4-positive: fixed fields emitted\n");
            return 1;
        }
        ++positive;

        te.pinned_shells = 2;
        if (!expect_reason(
                "NL4-negative",
                a210_temperature_publication_validate(&te, 2),
                "RADEQ_FREE_T_PUBLICATION_LEAK"))
            return 1;
    }

    /* NL5: domain rejection, with an in-domain positive control. */
    {
        const double out_of_domain[2] = {te_min - 1.0, te_max};

        if (!expect_ok(
                "NL5-positive",
                a210_fixed_te_profile_validate(
                    physical, 2, 2, te_min, te_max)))
            return 1;
        ++positive;

        if (!expect_reason(
                "NL5-negative",
                a210_fixed_te_profile_validate(
                    out_of_domain, 2, 2, te_min, te_max),
                "RADEQ_FIXED_T_PROFILE_OUT_OF_DOMAIN"))
            return 1;
    }

    /* NL2/NL3 production coverage: these cases go through the real loader.
     * Shell topology and nonphysical tokens must be rejected by its named
     * reasons; range rejection is the caller's immediate validation step. */
    {
        if (!expect_ok(
                "NL2-loader-positive",
                load_profile_text("0 10000\n1 12000\n", 1)))
            return 1;
        ++positive;

        if (!expect_reason(
                "NL2-loader-missing-shell",
                load_profile_text("0 10000\n", 0),
                "RADEQ_FIXED_T_SHELL_COUNT_MISMATCH"))
            return 1;
        if (!expect_reason(
                "NL2-loader-duplicate-shell",
                load_profile_text("0 10000\n0 12000\n1 11000\n", 0),
                "RADEQ_FIXED_T_SHELL_COUNT_MISMATCH"))
            return 1;
        if (!expect_reason(
                "NL2-loader-excess-shell",
                load_profile_text("0 10000\n1 12000\n2 13000\n", 0),
                "RADEQ_FIXED_T_SHELL_COUNT_MISMATCH"))
            return 1;
        if (!expect_reason(
                "NL2-loader-out-of-domain",
                load_profile_text("0 3499\n1 12000\n", 1),
                "RADEQ_FIXED_T_PROFILE_OUT_OF_DOMAIN"))
            return 1;

        if (!expect_reason(
                "NL3-loader-negative",
                load_profile_text("0 -1\n1 12000\n", 0),
                "RADEQ_FIXED_T_NONPHYSICAL_PROFILE"))
            return 1;
        if (!expect_reason(
                "NL3-loader-zero",
                load_profile_text("0 0\n1 12000\n", 0),
                "RADEQ_FIXED_T_NONPHYSICAL_PROFILE"))
            return 1;
        if (!expect_reason(
                "NL3-loader-nan",
                load_profile_text("0 nan\n1 12000\n", 0),
                "RADEQ_FIXED_T_NONPHYSICAL_PROFILE"))
            return 1;
        if (!expect_reason(
                "NL3-loader-inf",
                load_profile_text("0 inf\n1 12000\n", 0),
                "RADEQ_FIXED_T_NONPHYSICAL_PROFILE"))
            return 1;
        if (!expect_reason(
                "NL3-loader-parse",
                load_profile_text("0 not-a-temperature\n1 12000\n", 0),
                "RADEQ_FIXED_T_PROFILE_PARSE_FAILED"))
            return 1;
    }

    printf("PASS det_stage12 NL1..NL5 production_loader positive=%d\n",
           positive);
    return 0;
}
