#define _POSIX_C_SOURCE 200809L

#include "lumina.h"
#include "radiation_field.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int failures;
#define CHECK(c, l) do { if (!(c)) { \
    fprintf(stderr, "GATE_RECOVERY_FAIL %s line=%d\n", l, __LINE__); \
    failures++; } } while (0)

#define QH "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
#define PH "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

int main(void)
{
    const double EPOCH = 1683072.0;

    /* NC2: publication at the recorded epoch is usable, but a different
     * expected epoch is stale.  The external lane keeps this test independent
     * of AtomicData/PlasmaState/Geometry construction. */
    {
        GammaDeposition gd;
        double external_heating[2] = {2.0, 0.0};

        gamma_deposition_init(&gd, 2);
        CHECK(gamma_deposition_publish(
                  &gd, GAMMA_PROVENANCE_EXTERNAL_FILE, EPOCH,
                  NULL, NULL, NULL, external_heating) == 0,
              "NC2-publish");
        CHECK(gamma_deposition_require(&gd, EPOCH) ==
                  GAMMA_PUBLICATION_OK,
              "NC2-positive-current-epoch");
        CHECK(gamma_deposition_require(&gd, EPOCH + 1.0) ==
                  GAMMA_PUBLICATION_STALE_EPOCH,
              "NC2-stale-epoch");
        gamma_deposition_free(&gd);
    }

    /* NC4: same-epoch publication is closed exactly once.  Capture the
     * implementation diagnostic so the reason is checked, not inferred only
     * from rc=4. */
    {
        GammaDeposition gd;
        double external_heating[2] = {3.0, 0.0};
        double external_heating_next[2] = {4.0, 0.0};
        double saved_heating[2];
        double saved_nonthermal[2];
        char diagnostic[1024] = {0};
        int same_epoch_rc = -1;

        /* These are real, zero-initialized objects; the guard returns first,
         * so the internal producer does not touch their fields. */
        AtomicData atom;
        PlasmaState plasma;
        Geometry geo;
        memset(&atom, 0, sizeof atom);
        memset(&plasma, 0, sizeof plasma);
        memset(&geo, 0, sizeof geo);

        gamma_deposition_init(&gd, 2);
        CHECK(gamma_deposition_publish(
                  &gd, GAMMA_PROVENANCE_EXTERNAL_FILE, EPOCH,
                  NULL, NULL, NULL, external_heating) == 0,
              "NC4-initial-external-publish");
        memcpy(saved_heating, gd.heating_rate, sizeof(saved_heating));
        memcpy(saved_nonthermal, gd.nonthermal_ioniz_rate,
               sizeof(saved_nonthermal));

        FILE *capture = tmpfile();
        if (capture == NULL) {
            CHECK(0, "NC4-stderr-capture-open");
            same_epoch_rc = gamma_deposition_publish(
                &gd, GAMMA_PROVENANCE_INTERNAL_BATEMAN, EPOCH,
                &atom, &plasma, &geo, NULL);
        } else {
            int stderr_fd = fileno(stderr);
            int saved_stderr_fd = dup(stderr_fd);
            int redirected = 0;

            if (saved_stderr_fd >= 0 &&
                dup2(fileno(capture), stderr_fd) >= 0) {
                redirected = 1;
            } else {
                CHECK(0, "NC4-stderr-redirect");
            }

            same_epoch_rc = gamma_deposition_publish(
                &gd, GAMMA_PROVENANCE_INTERNAL_BATEMAN, EPOCH,
                &atom, &plasma, &geo, NULL);
            fflush(stderr);

            if (redirected) {
                CHECK(dup2(saved_stderr_fd, stderr_fd) >= 0,
                      "NC4-stderr-restore");
            }
            if (saved_stderr_fd >= 0)
                close(saved_stderr_fd);

            rewind(capture);
            size_t diagnostic_size = fread(
                diagnostic, 1, sizeof(diagnostic) - 1, capture);
            diagnostic[diagnostic_size] = '\0';
            fclose(capture);
        }

        CHECK(same_epoch_rc == 4, "NC4-same-epoch-rc4");
        CHECK(strstr(diagnostic, "GAMMA_DOUBLE_PUBLISH") != NULL,
              "NC4-stderr-reason");
        CHECK(memcmp(gd.heating_rate, saved_heating,
                     sizeof(saved_heating)) == 0,
              "NC4-heating-array-preserved");
        CHECK(memcmp(gd.nonthermal_ioniz_rate, saved_nonthermal,
                     sizeof(saved_nonthermal)) == 0,
              "NC4-nonthermal-array-preserved");

        /* Positive contrast: a new epoch is a legitimate re-publication and
         * must not be confused with the same-epoch double-publish guard. */
        CHECK(gamma_deposition_publish(
                  &gd, GAMMA_PROVENANCE_EXTERNAL_FILE, EPOCH + 1.0,
                  NULL, NULL, NULL, external_heating_next) == 0,
              "NC4-positive-new-epoch-republish");
        CHECK(gamma_deposition_require(&gd, EPOCH + 1.0) ==
                  GAMMA_PUBLICATION_OK,
              "NC4-positive-new-epoch-require");
        gamma_deposition_free(&gd);
    }

    /* R6 owner construction follows a2_06_dual_commit_selftest.c.  One
     * normal MC commit supplies the positive control for both line-view
     * injections and gives the deterministic injection a published source
     * field to replay. */
    {
        const size_t NS = 1;
        const size_t NQ = 2;
        const double v_inner[1] = {1.0e8};
        const double v_outer[1] = {2.0e8};
        const double volume[1] = {3.0e40};
        const uint64_t line_id[2] = {11, 42};
        const double line_sum[2] = {2.0, 0.0};
        const double line_sumsq[2] = {1.5, 0.0};
        const uint64_t line_count[2] = {3, 2};
        const uint64_t line_n_packets = 100;

        RadiationFieldOwner owner;
        CHECK(radiation_field_owner_init(&owner, NS) == 0,
              "R6-owner-init");
        CHECK(radiation_field_begin_mc(
                  &owner, v_inner, v_outer, NS, EPOCH, 1) == 0,
              "R6-begin-positive");
        double nu_mid = sqrt(owner.field.frequency_bin_edges.values[2000] *
                             owner.field.frequency_bin_edges.values[2001]);
        CHECK(radiation_field_accumulator_add(
                  &owner.accumulator, 0, nu_mid, 5.0) == 0,
              "R6-acc-positive");

        RadiationFieldCommitRequest request;
        memset(&request, 0, sizeof(request));
        request.provenance_kind = RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH;
        request.producer = "GATE_RECOVERY_SELFTEST_MC";
        request.generation = 1;
        request.epoch = EPOCH;
        request.n_shells = NS;
        request.v_inner = v_inner;
        request.v_outer = v_outer;
        request.source_n_bins = LUMINA_RADFIELD_N_BINS;
        request.statistic_kind = RADIATION_FIELD_ESTIMATOR_COUNT;
        request.source_count = owner.accumulator.contribution_count;
        request.raw_path_length = owner.accumulator.raw_path_length;
        request.volume = volume;
        request.time_simulation = 2.0;
        request.line_n = NQ;
        request.line_id = line_id;
        request.line_q_set_hash = QH;
        request.line_profile_id = 1;
        request.line_profile_hash = PH;
        request.line_sum = line_sum;
        request.line_sumsq = line_sumsq;
        request.line_count = line_count;
        request.line_n_packets = line_n_packets;
        CHECK(radiation_field_commit(&owner, &request) == 0,
              "N6-positive-commit");

        LineJbarView view;
        CHECK(radiation_field_line_jbar_view(
                  &owner, EPOCH, NS, 1, QH, 1, PH, &view) ==
                  LINE_JBAR_VIEW_OK,
              "N6-positive-view");
        LineJbarValue value;
        CHECK(line_jbar_lookup(&view, 0, 11, &value) == 0 &&
                  value.validity == LINE_JBAR_VALID && value.jbar > 0.0,
              "N6-positive-consumer");

        /* N6-2: q-set identity differs by exactly one hexadecimal character.
         * This is intentionally not PH, unlike the old coarse negative test. */
        char qhash_one_char_bad[sizeof(QH)];
        memcpy(qhash_one_char_bad, QH, sizeof(qhash_one_char_bad));
        qhash_one_char_bad[0] = 'b';
        size_t qhash_differences = 0;
        for (size_t i = 0; QH[i] != '\0'; ++i)
            if (QH[i] != qhash_one_char_bad[i])
                ++qhash_differences;
        CHECK(qhash_differences == 1, "N6-2-exactly-one-character");
        CHECK(radiation_field_line_jbar_view(
                  &owner, EPOCH, NS, 1, qhash_one_char_bad, 1, PH, &view) ==
                  LINE_JBAR_VIEW_QHASH,
              "N6-2-qhash-rejected");

        /* N6-3: replay a source field while presenting one sentinel line as
         * VALID with Jbar=-1.  The value is not clamped or replaced here;
         * only the exposed consumer result decides the verdict. */
        CHECK(radiation_field_begin_mc(
                  &owner, v_inner, v_outer, NS, EPOCH, 2) == 0,
              "N6-3-begin");
        CHECK(radiation_field_accumulator_add(
                  &owner.accumulator, 0, nu_mid, 7.0) == 0,
              "N6-3-acc");

        const double spoofed_jbar[2] = {3.0, -1.0};
        const double spoofed_error[2] = {0.0, 0.0};
        const int32_t spoofed_validity[2] = {
            LINE_JBAR_VALID, LINE_JBAR_VALID
        };
        double spoofed_source_frequency_bin_edges[
            LUMINA_RADFIELD_N_BINS + 1];
        double spoofed_source_J_nu[LUMINA_RADFIELD_N_BINS];
        RadiationFieldValidityState spoofed_source_validity[
            LUMINA_RADFIELD_N_BINS];
        memcpy(spoofed_source_frequency_bin_edges,
               owner.field.frequency_bin_edges.values,
               sizeof(spoofed_source_frequency_bin_edges));
        memcpy(spoofed_source_J_nu, owner.field.J_nu.values,
               sizeof(spoofed_source_J_nu));
        memcpy(spoofed_source_validity, owner.field.validity.values,
               sizeof(spoofed_source_validity));

        RadiationFieldCommitRequest spoofed;
        memset(&spoofed, 0, sizeof(spoofed));
        spoofed.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
        spoofed.producer = "GATE_RECOVERY_SELFTEST_SENTINEL";
        spoofed.generation = 2;
        spoofed.epoch = EPOCH;
        spoofed.n_shells = NS;
        spoofed.v_inner = v_inner;
        spoofed.v_outer = v_outer;
        spoofed.source_n_bins = LUMINA_RADFIELD_N_BINS;
        spoofed.source_frequency_bin_edges = spoofed_source_frequency_bin_edges;
        spoofed.source_J_nu = spoofed_source_J_nu;
        spoofed.source_validity = spoofed_source_validity;
        spoofed.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
        spoofed.line_n = NQ;
        spoofed.line_id = line_id;
        spoofed.line_q_set_hash = QH;
        spoofed.line_profile_id = 1;
        spoofed.line_profile_hash = PH;
        spoofed.line_provenance_kind =
            RADIATION_FIELD_PROVENANCE_CMFGEN_LINE_PROFILE_INTEGRAL;
        spoofed.line_producer = LUMINA_LINE_JBAR_DETERMINISTIC_PRODUCER;
        spoofed.line_jbar = spoofed_jbar;
        spoofed.line_error_upper = spoofed_error;
        spoofed.line_validity = spoofed_validity;

        int spoofed_commit_attempted = 0;
        int spoofed_commit_rc;
        spoofed_commit_attempted = 1;
        spoofed_commit_rc = radiation_field_commit(&owner, &spoofed);
        int spoofed_view_rc = LINE_JBAR_VIEW_STALE_GENERATION;
        int spoofed_lookup_rc = -1;
        LineJbarValue spoofed_value = {0};
        if (spoofed_commit_rc == 0) {
            spoofed_view_rc = radiation_field_line_jbar_view(
                &owner, EPOCH, NS, 2, QH, 1, PH, &view);
            if (spoofed_view_rc == LINE_JBAR_VIEW_OK)
                spoofed_lookup_rc = line_jbar_lookup(
                    &view, 0, 42, &spoofed_value);
        }

        const char *spoofed_rejection_stage = "none";
        if (spoofed_commit_rc != 0)
            spoofed_rejection_stage = "commit";
        else if (spoofed_view_rc != LINE_JBAR_VIEW_OK)
            spoofed_rejection_stage = "view";
        else if (spoofed_lookup_rc != 0)
            spoofed_rejection_stage = "lookup";
        else if (spoofed_value.validity != LINE_JBAR_VALID ||
                 !(spoofed_value.jbar < 0.0))
            spoofed_rejection_stage = "value";

        fprintf(stderr,
                "N6-3 trace commit_rc=%d view_rc=%d lookup_rc=%d "
                "validity=%d jbar=%.17g rejected_at=%s "
                "injection_commit_attempted=%d\n",
                spoofed_commit_rc, spoofed_view_rc, spoofed_lookup_rc,
                (int)spoofed_value.validity, spoofed_value.jbar,
                spoofed_rejection_stage, spoofed_commit_attempted);
        CHECK(spoofed_commit_attempted,
              "N6-3-injection-commit-request-called");
        CHECK(strcmp(spoofed_rejection_stage, "none") != 0,
              "N6-3-negative-jbar-rejected");

        radiation_field_owner_free(&owner);
    }

    if (failures) {
        fprintf(stderr, "FAIL gate_recovery failures=%d\n", failures);
        return 1;
    }
    printf("PASS gate_recovery NC2 NC4 N6-2 N6-3 positive_controls=4\n");
    return 0;
}
