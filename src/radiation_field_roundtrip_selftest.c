/* Standalone option-B source -> canonical commit -> NLTE consumer round trip.
 * Build with radiation_field.c and -lm; no model deck or runtime knob enters. */
#include "lumina.h"
#include "seed_capability.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* radiation_field.c references the production capability hook.  This fixture
 * never installs a capability, so the standalone link supplies an inert stub. */
SeedStatus seed_capability_revoke_on_first_commit(SeedCapability *cap)
{
    return cap ? SEED_ERR_WRONG_STATE : SEED_ERR_NULL;
}

static double source_edge(size_t b)
{
    if (b == 0) return NLTE_NU_MIN;
    if (b == NLTE_N_FREQ_BINS) return NLTE_NU_MAX;
    return NLTE_NU_MIN * exp((double)b *
        log(NLTE_NU_MAX / NLTE_NU_MIN) / (double)NLTE_N_FREQ_BINS);
}

static int rebin_to_source(const RadiationFieldView *view,
                           const double *edges, double *out)
{
    for (size_t b = 0; b < NLTE_N_FREQ_BINS; ++b) {
        double lo = edges[b], hi = edges[b + 1];
        double integral = 0.0, covered = 0.0;
        for (size_t q = 0; q < view->n_bins; ++q) {
            double a = fmax(lo, view->frequency_bin_edges[q]);
            double z = fmin(hi, view->frequency_bin_edges[q + 1]);
            if (z <= a) continue;
            if (view->validity[q] != RADIATION_FIELD_VALID &&
                view->validity[q] != RADIATION_FIELD_EXACT_ZERO)
                return -1;
            integral += view->J_nu[q] * (z - a);
            covered += z - a;
        }
        if (fabs(covered - (hi - lo)) > 32.0 * DBL_EPSILON * (hi - lo))
            return -1;
        out[b] = integral / (hi - lo);
    }
    return 0;
}

int main(void)
{
    double *edges = malloc((NLTE_N_FREQ_BINS + 1) * sizeof(*edges));
    double *source = malloc(NLTE_N_FREQ_BINS * sizeof(*source));
    double *roundtrip = malloc(NLTE_N_FREQ_BINS * sizeof(*roundtrip));
    RadiationFieldValidityState *validity =
        malloc(NLTE_N_FREQ_BINS * sizeof(*validity));
    if (!edges || !source || !roundtrip || !validity) {
        fprintf(stderr, "[GRID-ROUNDTRIP][FAIL] reason=OOM\n");
        free(edges); free(source); free(roundtrip); free(validity);
        return EXIT_FAILURE;
    }
    for (size_t b = 0; b <= NLTE_N_FREQ_BINS; ++b) edges[b] = source_edge(b);
    for (size_t b = 0; b < NLTE_N_FREQ_BINS; ++b) {
        source[b] = 1.0 + (double)(b % 37) / 64.0;
        validity[b] = RADIATION_FIELD_VALID;
    }

    /* NB3: a legitimately narrowed source/consumer pair is still contained;
     * the predicate must not demand unrelated coverage outside that domain. */
    const double narrow_edges[2] = { edges[1], edges[NLTE_N_FREQ_BINS - 1] };
    GridContainmentStatus narrow_status = grid_containment_check(
        narrow_edges, 1, narrow_edges, 1, 0, NULL);
    const double shortened_producer[2] = { edges[1], edges[NLTE_N_FREQ_BINS] };
    const double full_consumer[2] = { edges[0], edges[NLTE_N_FREQ_BINS] };
    GridContainmentStatus short_status = grid_containment_check(
        shortened_producer, 1, full_consumer, 1, 0, NULL);
    if (narrow_status != GRID_CONTAINMENT_OK ||
        short_status != GRID_CONTAINMENT_LOW_SHORTFALL) {
        fprintf(stderr,
                "[GRID-ROUNDTRIP][FAIL] reason=CONTAINMENT_PREDICATE "
                "narrow_status=%d short_status=%d\n",
                (int)narrow_status, (int)short_status);
        free(edges); free(source); free(roundtrip); free(validity);
        return EXIT_FAILURE;
    }

    RadiationFieldOwner owner;
    if (radiation_field_owner_init(&owner, 1) != 0) {
        fprintf(stderr, "[GRID-ROUNDTRIP][FAIL] reason=GRID_ALIGNMENT_VIOLATION\n");
        free(edges); free(source); free(roundtrip); free(validity);
        return EXIT_FAILURE;
    }
    double v_inner[1] = { 1.0 };
    double v_outer[1] = { 2.0 };
    RadiationFieldCommitRequest request;
    memset(&request, 0, sizeof(request));
    request.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
    request.producer = "GRID_ROUNDTRIP_KNOWN_J";
    request.generation = 1;
    request.epoch = 1.0;
    request.n_shells = 1;
    request.v_inner = v_inner;
    request.v_outer = v_outer;
    request.source_n_bins = NLTE_N_FREQ_BINS;
    request.source_frequency_bin_edges = edges;
    request.source_J_nu = source;
    request.source_validity = validity;
    request.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
    if (radiation_field_commit(&owner, &request) != 0) {
        fprintf(stderr, "[GRID-ROUNDTRIP][FAIL] reason=COMMIT_REJECTED\n");
        radiation_field_owner_free(&owner);
        free(edges); free(source); free(roundtrip); free(validity);
        return EXIT_FAILURE;
    }

    RadiationFieldView view;
    if (radiation_field_read_view(&owner, 1.0, 1, 1, &view) !=
            RADIATION_FIELD_VIEW_OK ||
        rebin_to_source(&view, edges, roundtrip) != 0) {
        fprintf(stderr, "[GRID-ROUNDTRIP][FAIL] reason=REBIN_INVALID\n");
        radiation_field_owner_free(&owner);
        free(edges); free(source); free(roundtrip); free(validity);
        return EXIT_FAILURE;
    }

    /* NB4: structural absence is never a numeric zero. */
    size_t first_consumer_bin = (size_t)(-LUMINA_RADFIELD_J_LO);
    owner.field.validity.values[first_consumer_bin] =
        RADIATION_FIELD_OUT_OF_GRID;
    int oog_rejected = rebin_to_source(&view, edges, roundtrip) != 0;
    owner.field.validity.values[first_consumer_bin] = RADIATION_FIELD_VALID;
    if (!oog_rejected) {
        fprintf(stderr, "[GRID-ROUNDTRIP][FAIL] reason=NB4_OUT_OF_GRID_AS_ZERO\n");
        radiation_field_owner_free(&owner);
        free(edges); free(source); free(roundtrip); free(validity);
        return EXIT_FAILURE;
    }

    double max_abs = 0.0, max_rel = 0.0;
    size_t worst = 0;
    for (size_t b = 0; b < NLTE_N_FREQ_BINS; ++b) {
        double abs_error = fabs(roundtrip[b] - source[b]);
        double rel_error = abs_error / source[b];
        if (rel_error > max_rel) {
            max_abs = abs_error;
            max_rel = rel_error;
            worst = b;
        }
    }
    int pass = max_rel <= 16.0 * DBL_EPSILON;
    fprintf(stderr,
            "[GRID-ROUNDTRIP][%s] K=%d source_bins=%d canonical_bins=%d "
            "anchor=%d canonical_range=[%.17g,%.17g] "
            "max_abs=%.17g max_rel=%.17g worst_bin=%zu "
            "NB3_narrow=PASS NB4_oog_rejected=PASS shortfall_control=PASS\n",
            pass ? "PASS" : "FAIL", LUMINA_RADFIELD_REFINEMENT_K,
            NLTE_N_FREQ_BINS, LUMINA_RADFIELD_N_BINS,
            -LUMINA_RADFIELD_J_LO, view.frequency_bin_edges[0],
            view.frequency_bin_edges[view.n_bins], max_abs, max_rel, worst);

    radiation_field_owner_free(&owner);
    free(edges); free(source); free(roundtrip); free(validity);
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
