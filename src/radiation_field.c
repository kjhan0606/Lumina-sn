/* strdup 은 C11 표준이 아니다. feature test 없이 쓰면 암시적 선언(int 반환)이 되어
 * LP64 에서 포인터가 절단된다 — 실제 잠재 결함이라 여기서 닫는다. */
#define _POSIX_C_SOURCE 200809L
#include "lumina.h"
#include "seed_capability.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double radiation_field_canonical_frequency_edge(size_t b)
{
    if (b > (size_t)LUMINA_RADFIELD_N_BINS) return NAN;
    int j = LUMINA_RADFIELD_J_LO + (int)b;
    int k = LUMINA_RADFIELD_REFINEMENT_K;
    if (j >= 0 && j <= k * NLTE_N_FREQ_BINS && j % k == 0) {
        int bf_edge = j / k;
        if (bf_edge == 0) return NLTE_NU_MIN;
        if (bf_edge == NLTE_N_FREQ_BINS) return NLTE_NU_MAX;
        return NLTE_NU_MIN * exp((double)bf_edge *
            log(NLTE_NU_MAX / NLTE_NU_MIN) / (double)NLTE_N_FREQ_BINS);
    }
    return NLTE_NU_MIN * exp((double)j * LUMINA_RADFIELD_DLOG);
}

static int radiation_field_alignment_contract_ok(const double *edges)
{
    const size_t lo = (size_t)(-LUMINA_RADFIELD_J_LO);
    const size_t hi = lo + (size_t)LUMINA_RADFIELD_REFINEMENT_K *
                           (size_t)NLTE_N_FREQ_BINS;
    double dlog = LUMINA_RADFIELD_DLOG;
    if (!edges || !(dlog > 0.0) || !isfinite(dlog) ||
        lo >= (size_t)LUMINA_RADFIELD_N_BINS ||
        hi > (size_t)LUMINA_RADFIELD_N_BINS ||
        edges[lo] != NLTE_NU_MIN || edges[hi] != NLTE_NU_MAX) {
        fprintf(stderr,
                "[RADIATION-FIELD][FATAL] reason=GRID_ALIGNMENT_VIOLATION "
                "K=%d j_lo=%d j_hi=%d bf_bins=%d anchor_lo=%zu "
                "anchor_hi=%zu\n",
                LUMINA_RADFIELD_REFINEMENT_K, LUMINA_RADFIELD_J_LO,
                LUMINA_RADFIELD_J_HI, NLTE_N_FREQ_BINS, lo, hi);
        return 0;
    }
    for (int bf = 0; bf <= NLTE_N_FREQ_BINS; ++bf) {
        size_t at = lo + (size_t)LUMINA_RADFIELD_REFINEMENT_K * (size_t)bf;
        double expected = bf == 0 ? NLTE_NU_MIN :
            (bf == NLTE_N_FREQ_BINS ? NLTE_NU_MAX :
             NLTE_NU_MIN * exp((double)bf *
                 log(NLTE_NU_MAX / NLTE_NU_MIN) /
                 (double)NLTE_N_FREQ_BINS));
        if (edges[at] != expected) {
            fprintf(stderr,
                    "[RADIATION-FIELD][FATAL] reason=GRID_ALIGNMENT_VIOLATION "
                    "bf_edge=%d canonical_edge=%zu got=%.17g expected=%.17g\n",
                    bf, at, edges[at], expected);
            return 0;
        }
    }
    return 1;
}

static size_t radiation_field_cell_count(size_t n_shells)
{
    return n_shells * (size_t)LUMINA_RADFIELD_N_BINS;
}

static void radiation_field_mark_all(RadiationField *field,
                                     RadiationFieldValidityState state)
{
    size_t cells = field->validity.n_shells * field->validity.n_bins;
    for (size_t i = 0; i < cells; ++i)
        field->validity.values[i] = state;
}

static void line_jbar_cache_release(LineJbarCache *cache)
{
    free(cache->shell_id);
    free(cache->line_id);
    free(cache->profile_id);
    free(cache->profile_hash);
    free(cache->jbar_value);
    free(cache->validity);
    free(cache->sample_count);
    free(cache->variance_or_standard_error);
    free((char *)cache->q_set_hash);
    memset(cache, 0, sizeof(*cache));
}

GridContainmentStatus grid_containment_check(
    const double *producer_edges, size_t producer_n_bins,
    const double *consumer_edges, size_t consumer_n_bins,
    size_t margin_bins, GridContainmentResult *out)
{
    GridContainmentResult result;
    memset(&result, 0, sizeof(result));
    result.status = GRID_CONTAINMENT_INVALID_GRID;
    if (!producer_edges || !consumer_edges || producer_n_bins == 0 ||
        consumer_n_bins == 0 || margin_bins > producer_n_bins / 2)
        goto done;
    for (size_t i = 0; i <= producer_n_bins; ++i)
        if (!isfinite(producer_edges[i]) || producer_edges[i] <= 0.0 ||
            (i > 0 && producer_edges[i] <= producer_edges[i - 1]))
            goto done;
    for (size_t i = 0; i <= consumer_n_bins; ++i)
        if (!isfinite(consumer_edges[i]) || consumer_edges[i] <= 0.0 ||
            (i > 0 && consumer_edges[i] <= consumer_edges[i - 1]))
            goto done;

    result.producer_min = producer_edges[margin_bins];
    result.producer_max = producer_edges[producer_n_bins - margin_bins];
    result.consumer_min = consumer_edges[0];
    result.consumer_max = consumer_edges[consumer_n_bins];
    int low = result.producer_min > result.consumer_min;
    int high = result.producer_max < result.consumer_max;
    result.low_shortfall_hz = low
        ? result.producer_min - result.consumer_min : 0.0;
    result.high_shortfall_hz = high
        ? result.consumer_max - result.producer_max : 0.0;
    result.status = low && high ? GRID_CONTAINMENT_BOTH_SHORTFALL :
        (low ? GRID_CONTAINMENT_LOW_SHORTFALL :
         (high ? GRID_CONTAINMENT_HIGH_SHORTFALL : GRID_CONTAINMENT_OK));
done:
    if (out) *out = result;
    return result.status;
}

int radiation_field_owner_init(RadiationFieldOwner *shadow, size_t n_shells)
{
    if (!shadow || n_shells == 0) return -1;
    memset(shadow, 0, sizeof(*shadow));

    RadiationField *field = &shadow->field;
    size_t cells = radiation_field_cell_count(n_shells);
    field->shell_boundaries.count = n_shells + 1;
    field->shell_boundaries.coordinate_units = "cm s^-1";
    field->shell_boundaries.values = (double *)calloc(n_shells + 1, sizeof(double));
    field->frequency_bin_edges.count = LUMINA_RADFIELD_N_BINS + 1;
    field->frequency_bin_edges.coordinate_units = "Hz";
    field->frequency_bin_edges.values = (double *)malloc(
        (LUMINA_RADFIELD_N_BINS + 1) * sizeof(double));
    field->J_nu.n_shells = n_shells;
    field->J_nu.n_bins = LUMINA_RADFIELD_N_BINS;
    field->J_nu.values = (double *)calloc(cells, sizeof(double));
    field->units = RADIATION_FIELD_UNITS_ERG_S_NEG1_CM_NEG2_HZ_NEG1_SR_NEG1;
    field->frame = RADIATION_FIELD_FRAME_SHELL_COMOVING;
    field->generation.required_generation = 0;
    field->generation.computed_generation = 0;
    field->provenance.kind = RADIATION_FIELD_PROVENANCE_NONE;
    field->provenance.producer = "UNCOMPUTED_SHADOW";
    field->provenance.frequency_union_sha256 = LUMINA_RADFIELD_UNION_SHA256;
    field->provenance.frequency_edge_sha256 = LUMINA_RADFIELD_EDGE_SHA256;
    field->validity.n_shells = n_shells;
    field->validity.n_bins = LUMINA_RADFIELD_N_BINS;
    field->validity.values = (RadiationFieldValidityState *)malloc(
        cells * sizeof(RadiationFieldValidityState));
    field->estimator_count_or_variance.kind = RADIATION_FIELD_ESTIMATOR_COUNT;
    field->estimator_count_or_variance.n_shells = n_shells;
    field->estimator_count_or_variance.n_bins = LUMINA_RADFIELD_N_BINS;
    field->estimator_count_or_variance.count = (uint64_t *)calloc(cells, sizeof(uint64_t));
    field->estimator_count_or_variance.variance = NULL;

    shadow->accumulator.n_shells = n_shells;
    shadow->accumulator.n_bins = LUMINA_RADFIELD_N_BINS;
    shadow->accumulator.raw_path_length = (double *)calloc(cells, sizeof(double));
    shadow->accumulator.contribution_count = (uint64_t *)calloc(cells, sizeof(uint64_t));

    if (!field->shell_boundaries.values || !field->frequency_bin_edges.values ||
        !field->J_nu.values || !field->validity.values ||
        !field->estimator_count_or_variance.count ||
        !shadow->accumulator.raw_path_length ||
        !shadow->accumulator.contribution_count) {
        radiation_field_owner_free(shadow);
        return -1;
    }

    for (size_t b = 0; b <= LUMINA_RADFIELD_N_BINS; ++b)
        field->frequency_bin_edges.values[b] =
            radiation_field_canonical_frequency_edge(b);
    if (!radiation_field_alignment_contract_ok(
            field->frequency_bin_edges.values)) {
        radiation_field_owner_free(shadow);
        return -1;
    }
    radiation_field_mark_all(field, RADIATION_FIELD_STALE);
    shadow->enabled = 1;
    return 0;
}

void radiation_field_owner_free(RadiationFieldOwner *shadow)
{
    if (!shadow) return;
    free(shadow->field.shell_boundaries.values);
    free(shadow->field.frequency_bin_edges.values);
    free(shadow->field.J_nu.values);
    free(shadow->field.validity.values);
    free(shadow->field.estimator_count_or_variance.count);
    free(shadow->field.estimator_count_or_variance.variance);
    free(shadow->accumulator.raw_path_length);
    free(shadow->accumulator.contribution_count);
    line_jbar_cache_release(&shadow->line_jbar_cache);
    free(shadow->line_ids_compact);
    free(shadow->line_profile_hash_storage);
    free(shadow->line_rate_graph_ids_compact);
    free(shadow->line_rate_graph_cache_index);
    free(shadow->line_rate_graph_hash_storage);
    memset(shadow, 0, sizeof(*shadow));
}

int radiation_field_begin_mc(RadiationFieldOwner *shadow,
                                    const double *v_inner,
                                    const double *v_outer,
                                    size_t n_shells, double epoch,
                                    uint64_t required_generation)
{
    if (!shadow || !shadow->enabled) return -1;
    if (!v_inner || !v_outer || n_shells != shadow->field.J_nu.n_shells ||
        required_generation == 0 || !isfinite(epoch) || epoch <= 0.0)
        return -1;
    if (required_generation !=
        shadow->field.generation.computed_generation + 1)
        return -1;

    for (size_t s = 0; s < n_shells; ++s) {
        if (!isfinite(v_inner[s]) || !isfinite(v_outer[s]) ||
            v_inner[s] >= v_outer[s]) return -1;
        if (s > 0 && v_inner[s] != v_outer[s - 1]) return -1;
    }
    memset(shadow->accumulator.raw_path_length, 0,
           radiation_field_cell_count(n_shells) * sizeof(double));
    memset(shadow->accumulator.contribution_count, 0,
           radiation_field_cell_count(n_shells) * sizeof(uint64_t));
    shadow->accumulator.out_of_grid_contribution_count = 0;
    return 0;
}

RadiationFieldAccumulator *radiation_field_accumulator_create(size_t n_shells)
{
    if (n_shells == 0) return NULL;
    RadiationFieldAccumulator *accumulator =
        (RadiationFieldAccumulator *)calloc(1, sizeof(*accumulator));
    if (!accumulator) return NULL;
    size_t cells = radiation_field_cell_count(n_shells);
    accumulator->n_shells = n_shells;
    accumulator->n_bins = LUMINA_RADFIELD_N_BINS;
    accumulator->raw_path_length = (double *)calloc(cells, sizeof(double));
    accumulator->contribution_count = (uint64_t *)calloc(cells, sizeof(uint64_t));
    if (!accumulator->raw_path_length || !accumulator->contribution_count) {
        radiation_field_accumulator_free(accumulator);
        return NULL;
    }
    return accumulator;
}

void radiation_field_accumulator_free(RadiationFieldAccumulator *accumulator)
{
    if (!accumulator) return;
    free(accumulator->raw_path_length);
    free(accumulator->contribution_count);
    free(accumulator);
}

int radiation_field_accumulator_add(RadiationFieldAccumulator *accumulator,
                                    size_t shell, double comoving_nu,
                                    double path_length_measure)
{
    if (!accumulator || shell >= accumulator->n_shells ||
        !isfinite(comoving_nu) || !isfinite(path_length_measure) ||
        path_length_measure < 0.0)
        return -1;
    if (comoving_nu < LUMINA_RADFIELD_NU_MIN_HZ ||
        comoving_nu > LUMINA_RADFIELD_NU_MAX_HZ) {
        accumulator->out_of_grid_contribution_count++;
        return 1;
    }
    double dlog = LUMINA_RADFIELD_DLOG;
    size_t bin = comoving_nu == LUMINA_RADFIELD_NU_MAX_HZ
        ? LUMINA_RADFIELD_N_BINS - 1
        : (size_t)(log(comoving_nu / LUMINA_RADFIELD_NU_MIN_HZ) / dlog);
    if (bin >= LUMINA_RADFIELD_N_BINS) return -1;
    size_t index = shell * (size_t)LUMINA_RADFIELD_N_BINS + bin;
    accumulator->raw_path_length[index] += path_length_measure;
    accumulator->contribution_count[index]++;
    return 0;
}

int radiation_field_accumulator_reduce(RadiationFieldAccumulator *destination,
                                       const RadiationFieldAccumulator *source)
{
    if (!destination || !source || destination->n_shells != source->n_shells ||
        destination->n_bins != source->n_bins) return -1;
    size_t cells = destination->n_shells * destination->n_bins;
    for (size_t i = 0; i < cells; ++i) {
        destination->raw_path_length[i] += source->raw_path_length[i];
        destination->contribution_count[i] += source->contribution_count[i];
    }
    destination->out_of_grid_contribution_count +=
        source->out_of_grid_contribution_count;
    return 0;
}

int radiation_field_validate_owner(const RadiationFieldOwner *shadow)
{
    if (!shadow || !shadow->enabled) return 0;
    const RadiationField *field = &shadow->field;
    if (field->units != RADIATION_FIELD_UNITS_ERG_S_NEG1_CM_NEG2_HZ_NEG1_SR_NEG1 ||
        field->frame != RADIATION_FIELD_FRAME_SHELL_COMOVING ||
        field->generation.required_generation == 0 ||
        field->generation.computed_generation != field->generation.required_generation ||
        field->J_nu.n_bins != LUMINA_RADFIELD_N_BINS ||
        strcmp(field->provenance.frequency_union_sha256,
               LUMINA_RADFIELD_UNION_SHA256) != 0 ||
        strcmp(field->provenance.frequency_edge_sha256,
               LUMINA_RADFIELD_EDGE_SHA256) != 0)
        return -1;

    size_t cells = field->J_nu.n_shells * field->J_nu.n_bins;
    for (size_t i = 0; i < cells; ++i) {
        double value = field->J_nu.values[i];
        uint64_t count = field->estimator_count_or_variance.count[i];
        RadiationFieldValidityState validity = field->validity.values[i];
        if (!isfinite(value) || value < 0.0) return -1;
        if (validity == RADIATION_FIELD_UNSAMPLED) {
            if (count != 0 || value != 0.0) return -1;
        } else if (validity == RADIATION_FIELD_EXACT_ZERO) {
            if (value != 0.0 ||
                (field->estimator_count_or_variance.kind ==
                     RADIATION_FIELD_ESTIMATOR_COUNT && count == 0)) return -1;
        } else if (validity == RADIATION_FIELD_VALID) {
            if (value <= 0.0 ||
                (field->estimator_count_or_variance.kind ==
                     RADIATION_FIELD_ESTIMATOR_COUNT && count == 0)) return -1;
        } else if (validity == RADIATION_FIELD_OUT_OF_GRID) {
            if (count != 0 || value != 0.0) return -1;
        } else {
            return -1;
        }
        if (field->estimator_count_or_variance.kind ==
                RADIATION_FIELD_DETERMINISTIC && count != 0)
            return -1;
    }
    const LineJbarCache *cache = &shadow->line_jbar_cache;
    if (cache->generation.computed_generation ==
            field->generation.computed_generation) {
        if ((cache->set_kind != LINE_JBAR_SET_RATE_GRAPH &&
             cache->set_kind != LINE_JBAR_SET_ENERGY_DOMAIN) ||
            cache->generation.required_generation !=
                field->generation.required_generation ||
            !cache->q_set_hash || strlen(cache->q_set_hash) != 64 ||
            !shadow->line_ids_compact || shadow->line_n_compact == 0 ||
            !shadow->line_rate_graph_ids_compact ||
            shadow->line_rate_graph_n_compact == 0 ||
            !shadow->line_rate_graph_cache_index ||
            !shadow->line_rate_graph_hash_storage ||
            strlen(shadow->line_rate_graph_hash_storage) != 64 ||
            !shadow->line_profile_hash_storage ||
            shadow->line_profile_id == 0 || !cache->jbar_value ||
            !cache->validity || !cache->sample_count ||
            !cache->variance_or_standard_error ||
            shadow->line_n_compact > SIZE_MAX / field->J_nu.n_shells ||
            cache->entry_count != shadow->line_n_compact *
                                  field->J_nu.n_shells)
            return -1;
        for (size_t i = 0; i < shadow->line_n_compact; ++i)
            if (i && shadow->line_ids_compact[i] <=
                     shadow->line_ids_compact[i - 1])
                return -1;
        for (size_t i = 0; i < shadow->line_rate_graph_n_compact; ++i) {
            size_t at = shadow->line_rate_graph_cache_index[i];
            if ((i && shadow->line_rate_graph_ids_compact[i] <=
                      shadow->line_rate_graph_ids_compact[i - 1]) ||
                at >= shadow->line_n_compact ||
                shadow->line_ids_compact[at] !=
                    shadow->line_rate_graph_ids_compact[i])
                return -1;
            if (cache->set_kind == LINE_JBAR_SET_RATE_GRAPH && at != i)
                return -1;
        }
        if (cache->set_kind == LINE_JBAR_SET_RATE_GRAPH &&
            (shadow->line_rate_graph_n_compact != shadow->line_n_compact ||
             strcmp(shadow->line_rate_graph_hash_storage,
                    cache->q_set_hash) != 0))
            return -1;
    }
    return 0;
}

static int radiation_field_request_geometry_ok(
    const RadiationFieldCommitRequest *request)
{
    if (!request->v_inner || !request->v_outer ||
        !isfinite(request->epoch) || request->epoch <= 0.0)
        return 0;
    for (size_t s = 0; s < request->n_shells; ++s) {
        if (!isfinite(request->v_inner[s]) ||
            !isfinite(request->v_outer[s]) ||
            request->v_inner[s] >= request->v_outer[s] ||
            (s > 0 && request->v_inner[s] != request->v_outer[s - 1]))
            return 0;
    }
    return 1;
}

static int radiation_field_prepare_mc(
    const RadiationFieldCommitRequest *request,
    const RadiationField *field, double *values,
    RadiationFieldValidityState *validity, uint64_t *count)
{
    if (!request->raw_path_length || !request->source_count ||
        !request->volume || !isfinite(request->time_simulation) ||
        request->time_simulation <= 0.0 ||
        request->source_frequency_bin_edges || request->source_J_nu ||
        request->source_validity || request->source_variance ||
        request->source_n_bins != LUMINA_RADFIELD_N_BINS ||
        request->statistic_kind != RADIATION_FIELD_ESTIMATOR_COUNT)
        return -1;
    for (size_t s = 0; s < request->n_shells; ++s) {
        if (!isfinite(request->volume[s]) || request->volume[s] <= 0.0)
            return -1;
        for (size_t b = 0; b < LUMINA_RADFIELD_N_BINS; ++b) {
            size_t index = s * (size_t)LUMINA_RADFIELD_N_BINS + b;
            uint64_t samples = request->source_count[index];
            double raw = request->raw_path_length[index];
            double delta_nu = field->frequency_bin_edges.values[b + 1] -
                              field->frequency_bin_edges.values[b];
            count[index] = samples;
            if (samples == 0) {
                values[index] = 0.0;
                validity[index] = RADIATION_FIELD_UNSAMPLED;
            } else if (raw == 0.0) {
                values[index] = 0.0;
                validity[index] = RADIATION_FIELD_EXACT_ZERO;
            } else if (raw > 0.0 && isfinite(raw) && delta_nu > 0.0) {
                values[index] = raw /
                    (4.0 * M_PI * request->volume[s] *
                     request->time_simulation * delta_nu);
                validity[index] = RADIATION_FIELD_VALID;
            } else {
                return -1;
            }
        }
    }
    return 0;
}

static int radiation_field_prepare_deterministic(
    const RadiationFieldCommitRequest *request,
    const RadiationField *field, double *values,
    RadiationFieldValidityState *validity, uint64_t *count)
{
    if (!request->source_frequency_bin_edges || !request->source_J_nu ||
        !request->source_validity || request->source_n_bins == 0 ||
        request->raw_path_length || request->volume ||
        request->time_simulation != 0.0 || request->source_count ||
        request->statistic_kind != RADIATION_FIELD_DETERMINISTIC)
        return -1;
    const double *source_edges = request->source_frequency_bin_edges;
    for (size_t b = 0; b <= request->source_n_bins; ++b)
        if (!isfinite(source_edges[b]) || source_edges[b] <= 0.0 ||
            (b > 0 && source_edges[b] <= source_edges[b - 1])) return -1;

    size_t first = 0;
    for (size_t b = 0; b < LUMINA_RADFIELD_N_BINS; ++b) {
        double target_lo = field->frequency_bin_edges.values[b];
        double target_hi = field->frequency_bin_edges.values[b + 1];
        if (target_lo < source_edges[0] ||
            target_hi > source_edges[request->source_n_bins]) {
            for (size_t s = 0; s < request->n_shells; ++s) {
                size_t out = s * (size_t)LUMINA_RADFIELD_N_BINS + b;
                values[out] = 0.0;
                validity[out] = RADIATION_FIELD_OUT_OF_GRID;
                count[out] = 0;
            }
            continue;
        }
        while (first + 1 < request->source_n_bins &&
               source_edges[first + 1] <= target_lo) first++;
        for (size_t s = 0; s < request->n_shells; ++s) {
            double integral = 0.0, covered = 0.0;
            int has_positive = 0, gap_unsampled = 0, gap_out_of_grid = 0;
            size_t k = first;
            while (k < request->source_n_bins && source_edges[k] < target_hi) {
                double lo = source_edges[k] > target_lo ? source_edges[k] : target_lo;
                double hi = source_edges[k + 1] < target_hi
                    ? source_edges[k + 1] : target_hi;
                if (hi > lo) {
                    size_t in = s * request->source_n_bins + k;
                    double source = request->source_J_nu[in];
                    RadiationFieldValidityState state = request->source_validity[in];
                    if (!isfinite(source) || source < 0.0) return -1;
                    if (state == RADIATION_FIELD_VALID && source > 0.0) {
                        integral += source * (hi - lo);
                        has_positive = 1;
                    } else if (state == RADIATION_FIELD_EXACT_ZERO && source == 0.0) {
                        /* An exact zero contributes a measured zero integral. */
                    } else if (state == RADIATION_FIELD_OUT_OF_GRID) {
                        /* 2026-08-06 driver fix: a producer-declared structural
                         * absence must survive the commit as OUT_OF_GRID.  The
                         * previous single `unavailable` flag collapsed it to
                         * UNSAMPLED (measured on real EDDFACTOR input: 15,268
                         * cells state 4 -> 3), erasing the section-9 four-state
                         * distinction that section-13 path 12 depends on. */
                        gap_out_of_grid = 1;
                    } else {
                        gap_unsampled = 1;
                    }
                    covered += hi - lo;
                }
                k++;
            }
            size_t out = s * (size_t)LUMINA_RADFIELD_N_BINS + b;
            double width = target_hi - target_lo;
            if (fabs(covered - width) > 32.0 *
                    2.2204460492503131e-16 * width) {
                /* Coverage shortfall means the target bin extends beyond the
                 * source grid: structural, not statistical. */
                gap_out_of_grid = 1;
            }
            if (gap_unsampled) {
                values[out] = 0.0;
                validity[out] = RADIATION_FIELD_UNSAMPLED;
            } else if (gap_out_of_grid) {
                values[out] = 0.0;
                validity[out] = RADIATION_FIELD_OUT_OF_GRID;
            } else if (has_positive) {
                values[out] = integral / width;
                validity[out] = RADIATION_FIELD_VALID;
            } else {
                values[out] = 0.0;
                validity[out] = RADIATION_FIELD_EXACT_ZERO;
            }
            count[out] = 0;
        }
    }
    return 0;
}

static int radiation_field_candidate_ok(
    const double *values, const RadiationFieldValidityState *validity,
    const uint64_t *count, size_t cells,
    RadiationFieldEstimatorStatisticKind statistic_kind)
{
    if (statistic_kind != RADIATION_FIELD_ESTIMATOR_COUNT &&
        statistic_kind != RADIATION_FIELD_DETERMINISTIC)
        return 0;
    for (size_t i = 0; i < cells; ++i) {
        if (!isfinite(values[i]) || values[i] < 0.0) return 0;
        if (validity[i] == RADIATION_FIELD_VALID) {
            if (values[i] <= 0.0 ||
                (statistic_kind == RADIATION_FIELD_ESTIMATOR_COUNT &&
                 count[i] == 0)) return 0;
        } else if (validity[i] == RADIATION_FIELD_EXACT_ZERO) {
            if (values[i] != 0.0 ||
                (statistic_kind == RADIATION_FIELD_ESTIMATOR_COUNT &&
                 count[i] == 0)) return 0;
        } else if (validity[i] == RADIATION_FIELD_UNSAMPLED ||
                   validity[i] == RADIATION_FIELD_OUT_OF_GRID) {
            if (values[i] != 0.0 || count[i] != 0) return 0;
        } else {
            return 0;
        }
        if (statistic_kind == RADIATION_FIELD_DETERMINISTIC && count[i] != 0)
            return 0;
    }
    return 1;
}

/* A2-06: validate + stage the selective line-Jbar candidate.  Runs BEFORE any
 * public mutation so a line failure aborts the whole commit atomically.
 * MC form: value = sum/(4pi V_s dt); se = norm*sqrt(N*s^2),
 * s^2 = (sumsq - sum^2/N)/(N-1), packet population incl. zero contributors. */
static int radiation_field_prepare_line(
    const RadiationFieldCommitRequest *request,
    double *value, LineJbarValidityState *validity, uint64_t *count,
    double *se)
{
    size_t n = request->line_n * request->n_shells;
    int mc = request->statistic_kind == RADIATION_FIELD_ESTIMATOR_COUNT;
    RadiationFieldProvenanceKind line_provenance =
        request->line_provenance_kind != RADIATION_FIELD_PROVENANCE_NONE
        ? request->line_provenance_kind : request->provenance_kind;
    const char *line_producer = request->line_producer
        ? request->line_producer : request->producer;
    if (!request->line_id || !request->line_q_set_hash ||
        strlen(request->line_q_set_hash) != 64 ||
        !request->line_profile_hash || strlen(request->line_profile_hash) != 64 ||
        request->line_profile_id == 0)
        return -1;
    if (mc) {
        if (line_provenance != RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH ||
            !line_producer)
            return -1;
    } else if (request->statistic_kind == RADIATION_FIELD_DETERMINISTIC) {
        if (line_provenance !=
                RADIATION_FIELD_PROVENANCE_CMFGEN_LINE_PROFILE_INTEGRAL ||
            !line_producer ||
            (strcmp(line_producer,
                    LUMINA_LINE_JBAR_DETERMINISTIC_PRODUCER) != 0 &&
             strcmp(line_producer,
                    LUMINA_LINE_JBAR_CMFGEN_NONOVERLAP_SOBOLEV_PRODUCER) != 0))
            return -1;
    } else {
        return -1;
    }
    for (size_t i = 1; i < request->line_n; i++)
        if (request->line_id[i] <= request->line_id[i - 1]) return -1;
    if (mc) {
        if (!request->line_sum || !request->line_sumsq || !request->line_count ||
            request->line_n_packets < 2 || !request->volume)
            return -1;
        double N = (double)request->line_n_packets;
        for (size_t q = 0; q < request->line_n; q++)
            for (size_t s = 0; s < request->n_shells; s++) {
                size_t i = q * request->n_shells + s;
                double sum = request->line_sum[i];
                double sq = request->line_sumsq[i];
                uint64_t ct = request->line_count[i];
                if (!isfinite(sum) || !isfinite(sq) || sum < 0.0 || sq < 0.0)
                    return -1;
                double norm = 1.0 / (4.0 * M_PI * request->volume[s] *
                                     request->time_simulation);
                count[i] = ct;
                if (ct == 0) {
                    if (sum != 0.0) return -1;
                    value[i] = 0.0; se[i] = 0.0;
                    validity[i] = LINE_JBAR_UNSAMPLED;
                } else if (sum == 0.0) {
                    value[i] = 0.0; se[i] = 0.0;
                    validity[i] = LINE_JBAR_EXACT_ZERO;
                } else {
                    /* Preserve the signed cancellation result.  A negative
                     * variance numerator is not rounded into an apparently
                     * certain measurement: it invalidates the whole atomic
                     * publication, leaving the previous generation intact. */
                    double variance_numerator = fma(-sum / N, sum, sq);
                    if (!isfinite(variance_numerator) ||
                        variance_numerator < 0.0) {
                        fprintf(stderr,
                                "[LINE_JBAR][BLOCKED] reason=NEGATIVE_MC_"
                                "VARIANCE_NUMERATOR q=%zu shell=%zu "
                                "sum=%.17g sumsq=%.17g packets=%llu "
                                "numerator=%.17g\n",
                                q, s, sum, sq,
                                (unsigned long long)request->line_n_packets,
                                variance_numerator);
                        return -1;
                    }
                    if (variance_numerator == 0.0)
                        variance_numerator = 0.0; /* canonical +0, no floor */
                    double s2 = variance_numerator / (N - 1.0);
                    value[i] = norm * sum;
                    se[i] = norm * sqrt(N * s2);
                    validity[i] = LINE_JBAR_VALID;
                }
            }
    } else {
        if (!request->line_jbar || !request->line_validity) return -1;
        for (size_t i = 0; i < n; i++) {
            int32_t v = request->line_validity[i];
            if (v != LINE_JBAR_VALID && v != LINE_JBAR_EXACT_ZERO &&
                v != LINE_JBAR_UNSAMPLED && v != LINE_JBAR_OUT_OF_BB_DOMAIN)
                return -1;
            if (!isfinite(request->line_jbar[i]) || request->line_jbar[i] < 0.0) {
                fprintf(stderr,
                        "[LINE_JBAR][BLOCKED] reason=NEGATIVE_OR_NONFINITE_JBAR "
                        "cell=%zu validity=%d value=%.17g\n",
                        i, (int)v, request->line_jbar[i]);
                return -1;
            }
            if ((v == LINE_JBAR_VALID && request->line_jbar[i] <= 0.0) ||
                (v != LINE_JBAR_VALID && request->line_jbar[i] != 0.0))
                return -1;
            value[i] = request->line_jbar[i];
            validity[i] = (LineJbarValidityState)v;
            count[i] = 0;
            if (request->line_error_upper) {
                double error_upper = request->line_error_upper[i];
                if (!(error_upper >= 0.0) || !isfinite(error_upper) ||
                    (v != LINE_JBAR_VALID &&
                     v != LINE_JBAR_EXACT_ZERO && error_upper != 0.0))
                    return -1;
                se[i] = error_upper;
            } else {
                se[i] = 0.0;
            }
        }
    }
    return 0;
}

static LineJbarSetKind radiation_field_request_line_set_kind(
        const RadiationFieldCommitRequest *request)
{
    return request->line_set_kind == LINE_JBAR_SET_UNSPECIFIED
         ? LINE_JBAR_SET_RATE_GRAPH : request->line_set_kind;
}

/* Stage Q_g identity and its sparse map into the numeric cache.  This runs
 * before public mutation and owns no numeric Jbar copy. */
static int radiation_field_prepare_line_membership(
        const RadiationFieldCommitRequest *request,
        uint64_t **rate_ids_out, size_t **cache_index_out,
        size_t *rate_n_out, char **rate_hash_out)
{
    if (!request || !rate_ids_out || !cache_index_out || !rate_n_out ||
        !rate_hash_out || request->line_n == 0 || !request->line_id ||
        !request->line_q_set_hash)
        return -1;
    *rate_ids_out = NULL;
    *cache_index_out = NULL;
    *rate_n_out = 0;
    *rate_hash_out = NULL;

    LineJbarSetKind kind = radiation_field_request_line_set_kind(request);
    const uint64_t *source_ids = NULL;
    const char *source_hash = NULL;
    size_t source_n = 0;
    if (kind == LINE_JBAR_SET_RATE_GRAPH) {
        if (request->line_rate_graph_n != 0 || request->line_rate_graph_id ||
            request->line_rate_graph_hash)
            return -1;
        source_ids = request->line_id;
        source_n = request->line_n;
        source_hash = request->line_q_set_hash;
    } else if (kind == LINE_JBAR_SET_ENERGY_DOMAIN) {
        if (request->line_rate_graph_n == 0 ||
            request->line_rate_graph_n > request->line_n ||
            !request->line_rate_graph_id ||
            !request->line_rate_graph_hash ||
            strlen(request->line_rate_graph_hash) != 64)
            return -1;
        source_ids = request->line_rate_graph_id;
        source_n = request->line_rate_graph_n;
        source_hash = request->line_rate_graph_hash;
    } else {
        return -1;
    }

    uint64_t *rate_ids = malloc(source_n * sizeof(*rate_ids));
    size_t *cache_index = malloc(source_n * sizeof(*cache_index));
    char *rate_hash = strdup(source_hash);
    if (!rate_ids || !cache_index || !rate_hash) {
        free(rate_ids); free(cache_index); free(rate_hash);
        return -1;
    }
    memcpy(rate_ids, source_ids, source_n * sizeof(*rate_ids));
    size_t cache_at = 0;
    for (size_t q = 0; q < source_n; ++q) {
        if (q && source_ids[q] <= source_ids[q - 1]) {
            free(rate_ids); free(cache_index); free(rate_hash);
            return -1;
        }
        while (cache_at < request->line_n &&
               request->line_id[cache_at] < source_ids[q])
            ++cache_at;
        if (cache_at == request->line_n ||
            request->line_id[cache_at] != source_ids[q]) {
            free(rate_ids); free(cache_index); free(rate_hash);
            return -1;
        }
        cache_index[q] = cache_at;
    }
    *rate_ids_out = rate_ids;
    *cache_index_out = cache_index;
    *rate_n_out = source_n;
    *rate_hash_out = rate_hash;
    return 0;
}

int radiation_field_commit(RadiationFieldOwner *shadow,
                           const RadiationFieldCommitRequest *request)
{
    if (!shadow || !shadow->enabled || !request) return -1;
    RadiationField *field = &shadow->field;
    if (request->n_shells != field->J_nu.n_shells ||
        request->generation == 0 ||
        request->generation != field->generation.computed_generation + 1 ||
        !radiation_field_request_geometry_ok(request) ||
        !request->producer ||
        (request->provenance_kind != RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH &&
         request->provenance_kind != RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY))
        return -1;

    /* Publication-time ownership of the source -> NLTE consumer relation.
     * The MC source is already the canonical grid; deterministic replay names
     * its actual source grid.  The wider canonical tails may remain explicitly
     * OUT_OF_GRID, but every downstream NLTE/BF bin must be fully covered. */
    const double consumer_edges[2] = { NLTE_NU_MIN, NLTE_NU_MAX };
    const double *producer_edges =
        request->provenance_kind == RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH
        ? field->frequency_bin_edges.values
        : request->source_frequency_bin_edges;
    size_t producer_n_bins =
        request->provenance_kind == RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH
        ? field->frequency_bin_edges.count - 1
        : request->source_n_bins;
    GridContainmentResult containment;
    GridContainmentStatus containment_status = grid_containment_check(
        producer_edges, producer_n_bins, consumer_edges, 1, 0, &containment);
    if (containment_status != GRID_CONTAINMENT_OK) {
        fprintf(stderr,
                "[RADIATION-FIELD][BLOCKED] "
                "reason=GRID_CONTAINMENT_VIOLATION status=%d producer=%s "
                "producer_range=[%.17g,%.17g] consumer_range=[%.17g,%.17g] "
                "low_shortfall_hz=%.17g high_shortfall_hz=%.17g\n",
                (int)containment_status, request->producer,
                containment.producer_min, containment.producer_max,
                containment.consumer_min, containment.consumer_max,
                containment.low_shortfall_hz, containment.high_shortfall_hz);
        return -1;
    }

    size_t cells = radiation_field_cell_count(request->n_shells);
    double *values = (double *)calloc(cells, sizeof(double));
    RadiationFieldValidityState *validity =
        (RadiationFieldValidityState *)malloc(cells * sizeof(*validity));
    uint64_t *count = (uint64_t *)calloc(cells, sizeof(uint64_t));
    if (!values || !validity || !count) {
        free(values); free(validity); free(count);
        return -1;
    }
    int rc = request->provenance_kind == RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH
        ? radiation_field_prepare_mc(request, field, values, validity, count)
        : radiation_field_prepare_deterministic(request, field, values, validity, count);
    if (rc != 0) {
        free(values); free(validity); free(count);
        return -1;
    }
    if (!radiation_field_candidate_ok(values, validity, count, cells,
                                      request->statistic_kind)) {
        free(values); free(validity); free(count);
        return -1;
    }

    /* A2-06: line candidate staged and validated BEFORE any public mutation.
     * An accumulation latch or a bad line block aborts the whole commit. */
    double *lv = NULL, *lse = NULL;
    LineJbarValidityState *lval = NULL;
    uint64_t *lct = NULL;
    if (request->line_n > 0 &&
        request->n_shells > SIZE_MAX / request->line_n) {
        free(values); free(validity); free(count);
        return -1;
    }
    size_t lcells = request->line_n * request->n_shells;
    if (request->line_error_latch) {
        free(values); free(validity); free(count);
        return -1;
    }
    if (request->line_n > 0) {
        lv = malloc(lcells * sizeof(double));
        lse = malloc(lcells * sizeof(double));
        lval = malloc(lcells * sizeof(*lval));
        lct = malloc(lcells * sizeof(uint64_t));
        if (!lv || !lse || !lval || !lct ||
            radiation_field_prepare_line(request, lv, lval, lct, lse) != 0) {
            free(values); free(validity); free(count);
            free(lv); free(lse); free(lval); free(lct);
            return -1;
        }
    }

    /* R6: allocate and fill the complete line cache, identity strings and
     * compact index before touching either public view.  Allocation failure
     * therefore leaves the previous continuum and line generations intact. */
    LineJbarCache staged_line;
    memset(&staged_line, 0, sizeof(staged_line));
    uint64_t *staged_line_ids = NULL;
    char *staged_profile_hash = NULL;
    uint64_t *staged_rate_ids = NULL;
    size_t *staged_rate_cache_index = NULL;
    size_t staged_rate_n = 0;
    char *staged_rate_hash = NULL;
    if (request->line_n > 0) {
        RadiationFieldProvenanceKind line_provenance =
            request->line_provenance_kind != RADIATION_FIELD_PROVENANCE_NONE
            ? request->line_provenance_kind : request->provenance_kind;
        const char *line_producer = request->line_producer
            ? request->line_producer : request->producer;
        if (radiation_field_prepare_line_membership(
                request, &staged_rate_ids, &staged_rate_cache_index,
                &staged_rate_n, &staged_rate_hash) != 0) {
            free(values); free(validity); free(count);
            free(lv); free(lse); free(lval); free(lct);
            return -1;
        }
        staged_line.shell_id = malloc(lcells * sizeof(uint64_t));
        staged_line.line_id = malloc(lcells * sizeof(uint64_t));
        staged_line.profile_id = malloc(lcells * sizeof(uint64_t));
        staged_line.profile_hash = malloc(lcells * sizeof(const char *));
        staged_line.jbar_value = malloc(lcells * sizeof(double));
        staged_line.validity = malloc(lcells * sizeof(LineJbarValidityState));
        staged_line.sample_count = malloc(lcells * sizeof(uint64_t));
        staged_line.variance_or_standard_error = malloc(lcells * sizeof(double));
        staged_line.q_set_hash = strdup(request->line_q_set_hash);
        staged_profile_hash = strdup(request->line_profile_hash);
        staged_line_ids = malloc(request->line_n * sizeof(uint64_t));
        if (!staged_line.shell_id || !staged_line.line_id ||
            !staged_line.profile_id || !staged_line.profile_hash ||
            !staged_line.jbar_value || !staged_line.validity ||
            !staged_line.sample_count ||
            !staged_line.variance_or_standard_error ||
            !staged_line.q_set_hash || !staged_profile_hash ||
            !staged_line_ids) {
            line_jbar_cache_release(&staged_line);
            free(staged_profile_hash); free(staged_line_ids);
            free(staged_rate_ids); free(staged_rate_cache_index);
            free(staged_rate_hash);
            free(values); free(validity); free(count);
            free(lv); free(lse); free(lval); free(lct);
            return -1;
        }
        staged_line.entry_count = lcells;
        staged_line.set_kind = radiation_field_request_line_set_kind(request);
        staged_line.generation.required_generation = request->generation;
        staged_line.generation.computed_generation = request->generation;
        staged_line.statistic_kind = request->statistic_kind;
        staged_line.units = field->units;
        staged_line.frame = field->frame;
        staged_line.provenance.kind = line_provenance;
        staged_line.provenance.producer = line_producer;
        memcpy(staged_line_ids, request->line_id,
               request->line_n * sizeof(uint64_t));
        for (size_t q = 0; q < request->line_n; q++)
            for (size_t s = 0; s < request->n_shells; s++) {
                size_t i = q * request->n_shells + s;
                staged_line.shell_id[i] = (uint64_t)s;
                staged_line.line_id[i] = request->line_id[q];
                staged_line.profile_id[i] = request->line_profile_id;
                staged_line.profile_hash[i] = staged_profile_hash;
                staged_line.jbar_value[i] = lv[i];
                staged_line.validity[i] = lval[i];
                staged_line.sample_count[i] = lct[i];
                staged_line.variance_or_standard_error[i] = lse[i];
            }
    }
    free(lv); free(lse); free(lval); free(lct);

    uint64_t total = 0;
    if (request->statistic_kind == RADIATION_FIELD_ESTIMATOR_COUNT)
        for (size_t i = 0; i < cells; ++i) total += count[i];

    /* Atomic publication: generation, validity and public statistics change in
     * this choke point only, after the complete candidate has validated. */
    for (size_t s = 0; s < request->n_shells; ++s)
        field->shell_boundaries.values[s] = request->v_inner[s];
    field->shell_boundaries.values[request->n_shells] =
        request->v_outer[request->n_shells - 1];
    memcpy(field->J_nu.values, values, cells * sizeof(double));
    memcpy(field->validity.values, validity, cells * sizeof(*validity));
    memcpy(field->estimator_count_or_variance.count, count,
           cells * sizeof(uint64_t));
    field->estimator_count_or_variance.kind = request->statistic_kind;
    field->epoch = request->epoch;
    field->frame = RADIATION_FIELD_FRAME_SHELL_COMOVING;
    field->provenance.kind = request->provenance_kind;
    field->provenance.producer = request->producer;
    field->provenance.raw_ledger_sha256 = request->raw_ledger_sha256;
    field->provenance.contribution_count = request->statistic_kind ==
        RADIATION_FIELD_ESTIMATOR_COUNT ? total : request->contribution_count;
    field->provenance.out_of_grid_contribution_count =
        request->out_of_grid_contribution_count;
    field->generation.required_generation = request->generation;
    field->generation.computed_generation = request->generation;
    /* A2-16: the scalar seed's capability dies here — revoke BEFORE any caller
     * can observe the published generation, so no post-commit read can slip
     * through.  The owner's seed payload is zeroed by its owner afterwards. */
    if (shadow->seed_capability)
        (void)seed_capability_revoke_on_first_commit(
            (SeedCapability *)shadow->seed_capability);
    if (request->line_n > 0) {
        line_jbar_cache_release(&shadow->line_jbar_cache);
        shadow->line_jbar_cache = staged_line;
        free((char *)shadow->line_profile_hash_storage);
        shadow->line_profile_hash_storage = staged_profile_hash;
        shadow->line_profile_id = request->line_profile_id;
        shadow->line_n_compact = request->line_n;
        free(shadow->line_ids_compact);
        shadow->line_ids_compact = staged_line_ids;
        free(shadow->line_rate_graph_ids_compact);
        free(shadow->line_rate_graph_cache_index);
        free(shadow->line_rate_graph_hash_storage);
        shadow->line_rate_graph_ids_compact = staged_rate_ids;
        shadow->line_rate_graph_cache_index = staged_rate_cache_index;
        shadow->line_rate_graph_n_compact = staged_rate_n;
        shadow->line_rate_graph_hash_storage = staged_rate_hash;
    } else {
        shadow->line_jbar_cache.generation.required_generation =
            request->generation;
        shadow->line_jbar_cache.generation.computed_generation = 0;
        shadow->line_jbar_cache.provenance.kind = RADIATION_FIELD_PROVENANCE_NONE;
    }

    free(values); free(validity); free(count);
    if (radiation_field_validate_owner(shadow) != 0) return -1;
    return radiation_field_dump_if_requested(shadow);
}

int radiation_field_dump_if_requested(const RadiationFieldOwner *shadow)
{
    if (!shadow || !shadow->enabled) return 0;
    const char *path = getenv("LUMINA_RADFIELD_COMMIT_DUMP");
    if (!path || !*path) path = getenv("LUMINA_RADFIELD_SHADOW_DUMP");
    if (!path || !*path) return 0;
    FILE *stream = fopen(path, "w");
    if (!stream) {
        fprintf(stderr, "[RADIATION-FIELD][FATAL] cannot open %s: %s\n",
                path, strerror(errno));
        return -1;
    }
    const RadiationField *field = &shadow->field;
    fprintf(stream, "#schema=lumina-radiation-field-commit-v2\n");
    fprintf(stream, "#units=erg s^-1 cm^-2 Hz^-1 sr^-1\n");
    fprintf(stream, "#frame=shell-comoving\n");
    fprintf(stream, "#epoch=%.17g\n", field->epoch);
    fprintf(stream, "#required_generation=%llu\n",
            (unsigned long long)field->generation.required_generation);
    fprintf(stream, "#computed_generation=%llu\n",
            (unsigned long long)field->generation.computed_generation);
    fprintf(stream, "#union_sha256=%s\n", field->provenance.frequency_union_sha256);
    fprintf(stream, "#edge_sha256=%s\n", field->provenance.frequency_edge_sha256);
    fprintf(stream, "#out_of_grid_contributions=%llu\n",
            (unsigned long long)field->provenance.out_of_grid_contribution_count);
    fprintf(stream, "shell,bin,nu_lo_hz,nu_hi_hz,J_nu,validity,contribution_count\n");
    for (size_t s = 0; s < field->J_nu.n_shells; ++s) {
        for (size_t b = 0; b < field->J_nu.n_bins; ++b) {
            size_t index = s * field->J_nu.n_bins + b;
            fprintf(stream, "%zu,%zu,%.17g,%.17g,%.17g,%d,%llu\n",
                    s, b, field->frequency_bin_edges.values[b],
                    field->frequency_bin_edges.values[b + 1],
                    field->J_nu.values[index], (int)field->validity.values[index],
                    (unsigned long long)
                        field->estimator_count_or_variance.count[index]);
        }
    }
    if (fclose(stream) != 0) return -1;
    return 0;
}

int radiation_field_read_view(const RadiationFieldOwner *owner,
                              double expected_epoch,
                              size_t expected_n_shells,
                              uint64_t expected_generation,
                              RadiationFieldView *out)
{
    /* A2-05 R5: every check below is a distinct falsifiable success condition;
     * see SPEC_A2_05_V2.md.  radiation_field_validate_owner() cannot serve
     * here because it accepts NULL/disabled owners by design. */
    if (!out) return RADIATION_FIELD_VIEW_GRID;
    /* Full invalidation on every failure path: no stale pointer, count or
     * generation from a previous successful view may survive in *out. */
    memset(out, 0, sizeof(*out));
    if (!owner || !owner->enabled)
        return RADIATION_FIELD_VIEW_DISABLED;
    const RadiationField *field = &owner->field;
    if (field->units != RADIATION_FIELD_UNITS_ERG_S_NEG1_CM_NEG2_HZ_NEG1_SR_NEG1 ||
        field->frame != RADIATION_FIELD_FRAME_SHELL_COMOVING)
        return RADIATION_FIELD_VIEW_UNITS_FRAME;
    if (!(field->epoch == expected_epoch) ||
        field->J_nu.n_shells != expected_n_shells ||
        field->validity.n_shells != expected_n_shells)
        return RADIATION_FIELD_VIEW_EPOCH_SHELLS;
    if (expected_generation == 0 ||
        field->generation.required_generation != expected_generation ||
        field->generation.computed_generation != expected_generation)
        return RADIATION_FIELD_VIEW_STALE_GENERATION;
    if (field->J_nu.n_bins != LUMINA_RADFIELD_N_BINS ||
        field->validity.n_bins != LUMINA_RADFIELD_N_BINS ||
        field->frequency_bin_edges.count != LUMINA_RADFIELD_N_BINS + 1 ||
        !field->frequency_bin_edges.values || !field->J_nu.values ||
        !field->validity.values || !field->estimator_count_or_variance.count)
        return RADIATION_FIELD_VIEW_GRID;
    /* Canonical-grid identity: every edge is recomputed from the NLTE-derived
     * option-B expression owner-init uses, including exact K-boundaries. */
    {
        const double *edges = field->frequency_bin_edges.values;
        if (edges[0] != LUMINA_RADFIELD_NU_MIN_HZ ||
            edges[LUMINA_RADFIELD_N_BINS] != LUMINA_RADFIELD_NU_MAX_HZ)
            return RADIATION_FIELD_VIEW_GRID;
        for (size_t b = 1; b < LUMINA_RADFIELD_N_BINS; ++b) {
            if (edges[b] != radiation_field_canonical_frequency_edge(b))
                return RADIATION_FIELD_VIEW_GRID;
            if (!(edges[b] > edges[b - 1]))
                return RADIATION_FIELD_VIEW_GRID;
        }
        if (!(edges[LUMINA_RADFIELD_N_BINS] >
              edges[LUMINA_RADFIELD_N_BINS - 1]))
            return RADIATION_FIELD_VIEW_GRID;
    }
    out->n_shells = expected_n_shells;
    out->n_bins = LUMINA_RADFIELD_N_BINS;
    out->frequency_bin_edges = field->frequency_bin_edges.values;
    out->J_nu = field->J_nu.values;
    out->validity = field->validity.values;
    out->count = field->estimator_count_or_variance.count;
    out->generation = expected_generation;
    return RADIATION_FIELD_VIEW_OK;
}

static LineJbarViewStatus radiation_field_line_jbar_view_base(
        const RadiationFieldOwner *owner, double expected_epoch,
        size_t expected_n_shells, uint64_t expected_generation,
        uint64_t expected_profile_id, const char *expected_profile_hash,
        LineJbarView *out)
{
    if (!out) return LINE_JBAR_VIEW_PROFILE;
    memset(out, 0, sizeof(*out));
    if (!owner || !owner->enabled) return LINE_JBAR_VIEW_DISABLED;
    const LineJbarCache *cache = &owner->line_jbar_cache;
    if (cache->units != RADIATION_FIELD_UNITS_ERG_S_NEG1_CM_NEG2_HZ_NEG1_SR_NEG1 ||
        cache->frame != RADIATION_FIELD_FRAME_SHELL_COMOVING)
        return LINE_JBAR_VIEW_UNITS_FRAME;
    if (!(owner->field.epoch == expected_epoch) ||
        owner->field.J_nu.n_shells != expected_n_shells)
        return LINE_JBAR_VIEW_EPOCH_SHELLS;
    if (expected_generation == 0 ||
        cache->generation.required_generation != expected_generation ||
        cache->generation.computed_generation != expected_generation ||
        owner->field.generation.computed_generation != expected_generation)
        return LINE_JBAR_VIEW_STALE_GENERATION;
    if (expected_profile_id == 0 ||
        owner->line_profile_id != expected_profile_id ||
        !expected_profile_hash || !owner->line_profile_hash_storage ||
        strcmp(owner->line_profile_hash_storage, expected_profile_hash) != 0 ||
        (cache->statistic_kind != RADIATION_FIELD_ESTIMATOR_COUNT &&
         cache->statistic_kind != RADIATION_FIELD_DETERMINISTIC) ||
        !owner->line_ids_compact || owner->line_n_compact == 0 ||
        !cache->jbar_value || !cache->validity || !cache->sample_count ||
        !cache->variance_or_standard_error)
        return LINE_JBAR_VIEW_PROFILE;
    if ((cache->set_kind != LINE_JBAR_SET_RATE_GRAPH &&
         cache->set_kind != LINE_JBAR_SET_ENERGY_DOMAIN) ||
        !cache->q_set_hash || strlen(cache->q_set_hash) != 64 ||
        owner->line_n_compact > SIZE_MAX / expected_n_shells ||
        cache->entry_count != owner->line_n_compact * expected_n_shells)
        return LINE_JBAR_VIEW_SET_KIND;
    out->cache_n_lines = owner->line_n_compact;
    out->n_shells = expected_n_shells;
    out->jbar = cache->jbar_value;
    out->validity = cache->validity;
    out->count = cache->sample_count;
    out->se = cache->variance_or_standard_error;
    out->cache_set_hash = cache->q_set_hash;
    out->rate_graph_hash = owner->line_rate_graph_hash_storage;
    out->cache_set_kind = cache->set_kind;
    out->statistic_kind = cache->statistic_kind;
    out->generation = expected_generation;
    return LINE_JBAR_VIEW_OK;
}

int radiation_field_line_jbar_rate_view(const RadiationFieldOwner *owner,
                                        double expected_epoch,
                                        size_t expected_n_shells,
                                        uint64_t expected_generation,
                                        const char *expected_q_set_hash,
                                        uint64_t expected_profile_id,
                                        const char *expected_profile_hash,
                                        LineJbarView *out)
{
    LineJbarViewStatus status = radiation_field_line_jbar_view_base(
        owner, expected_epoch, expected_n_shells, expected_generation,
        expected_profile_id, expected_profile_hash, out);
    if (status != LINE_JBAR_VIEW_OK) return status;
    if (!expected_q_set_hash || !owner->line_rate_graph_hash_storage ||
        strcmp(owner->line_rate_graph_hash_storage,
               expected_q_set_hash) != 0) {
        memset(out, 0, sizeof(*out));
        fprintf(stderr,
                "[LINE_JBAR_VIEW][BLOCKED] reason=QHASH_MISMATCH\n");
        return LINE_JBAR_VIEW_QHASH;
    }
    if (!owner->line_rate_graph_ids_compact ||
        !owner->line_rate_graph_cache_index ||
        owner->line_rate_graph_n_compact == 0) {
        memset(out, 0, sizeof(*out));
        return LINE_JBAR_VIEW_SUBSET;
    }
    out->n_lines = owner->line_rate_graph_n_compact;
    out->line_id = owner->line_rate_graph_ids_compact;
    out->cache_index = out->cache_set_kind == LINE_JBAR_SET_RATE_GRAPH
                     ? NULL : owner->line_rate_graph_cache_index;
    return LINE_JBAR_VIEW_OK;
}

int radiation_field_line_jbar_energy_view(const RadiationFieldOwner *owner,
                                          double expected_epoch,
                                          size_t expected_n_shells,
                                          uint64_t expected_generation,
                                          const char *expected_e_set_hash,
                                          uint64_t expected_profile_id,
                                          const char *expected_profile_hash,
                                          LineJbarView *out)
{
    LineJbarViewStatus status = radiation_field_line_jbar_view_base(
        owner, expected_epoch, expected_n_shells, expected_generation,
        expected_profile_id, expected_profile_hash, out);
    if (status != LINE_JBAR_VIEW_OK) return status;
    if (out->cache_set_kind != LINE_JBAR_SET_ENERGY_DOMAIN) {
        memset(out, 0, sizeof(*out));
        return LINE_JBAR_VIEW_SET_KIND;
    }
    if (!expected_e_set_hash || !out->cache_set_hash ||
        strcmp(out->cache_set_hash, expected_e_set_hash) != 0) {
        memset(out, 0, sizeof(*out));
        fprintf(stderr,
                "[LINE_JBAR_VIEW][BLOCKED] reason=EHASH_MISMATCH\n");
        return LINE_JBAR_VIEW_QHASH;
    }
    out->n_lines = out->cache_n_lines;
    out->line_id = owner->line_ids_compact;
    out->cache_index = NULL;
    return LINE_JBAR_VIEW_OK;
}

/* Compatibility entry point: population/rate consumers request Q_g. */
int radiation_field_line_jbar_view(const RadiationFieldOwner *owner,
                                   double expected_epoch,
                                   size_t expected_n_shells,
                                   uint64_t expected_generation,
                                   const char *expected_q_set_hash,
                                   uint64_t expected_profile_id,
                                   const char *expected_profile_hash,
                                   LineJbarView *out)
{
    return radiation_field_line_jbar_rate_view(
        owner, expected_epoch, expected_n_shells, expected_generation,
        expected_q_set_hash, expected_profile_id, expected_profile_hash, out);
}

static int line_jbar_lookup_resolved(const LineJbarView *view, size_t shell,
                                     size_t requested_index, uint64_t line_id,
                                     LineJbarValue *out)
{
    if (!view || !out || !view->line_id || !view->jbar || !view->validity ||
        !view->count || !view->se || view->n_lines == 0 ||
        view->cache_n_lines == 0 || shell >= view->n_shells)
        return -1;
    memset(out, 0, sizeof(*out));
    if (requested_index >= view->n_lines ||
        view->line_id[requested_index] != line_id)
        return -2;                       /* MISS: distinct error, no value */
    size_t cache_line = view->cache_index
                      ? view->cache_index[requested_index]
                      : requested_index;
    if (cache_line >= view->cache_n_lines) return -3;
    size_t i = cache_line * view->n_shells + shell;
    if ((view->statistic_kind != RADIATION_FIELD_ESTIMATOR_COUNT &&
         view->statistic_kind != RADIATION_FIELD_DETERMINISTIC) ||
        (view->statistic_kind == RADIATION_FIELD_DETERMINISTIC &&
         view->count[i] != 0) ||
        !(view->se[i] >= 0.0) || !isfinite(view->se[i]) ||
        !isfinite(view->jbar[i]) || view->jbar[i] < 0.0 ||
        (view->validity[i] == LINE_JBAR_VALID && view->jbar[i] <= 0.0) ||
        (view->validity[i] != LINE_JBAR_VALID && view->jbar[i] != 0.0)) {
        fprintf(stderr,
                "[LINE_JBAR_LOOKUP][BLOCKED] reason=NEGATIVE_OR_INVALID_JBAR "
                "line_id=%llu shell=%zu validity=%d value=%.17g "
                "statistic_kind=%d\n",
                (unsigned long long)line_id, shell,
                (int)view->validity[i], view->jbar[i],
                (int)view->statistic_kind);
        return -3;
    }
    out->jbar = view->jbar[i];
    out->validity = view->validity[i];
    out->count = view->count[i];
    out->se = view->se[i];
    out->statistic_kind = view->statistic_kind;
    return 0;
}

int line_jbar_lookup_index(const LineJbarView *view, size_t shell,
                           size_t requested_index, uint64_t line_id,
                           LineJbarValue *out)
{
    return line_jbar_lookup_resolved(
        view, shell, requested_index, line_id, out);
}

int line_jbar_lookup(const LineJbarView *view, size_t shell,
                     uint64_t line_id, LineJbarValue *out)
{
    if (!view || !out || !view->line_id || !view->jbar || !view->validity ||
        !view->count || !view->se || view->n_lines == 0 ||
        view->cache_n_lines == 0 || shell >= view->n_shells)
        return -1;
    size_t a = 0, b = view->n_lines;
    while (a < b) {
        size_t m = (a + b) / 2;
        if (view->line_id[m] < line_id) a = m + 1; else b = m;
    }
    return line_jbar_lookup_resolved(view, shell, a, line_id, out);
}
