/* strdup 은 C11 표준이 아니다. feature test 없이 쓰면 암시적 선언(int 반환)이 되어
 * LP64 에서 포인터가 절단된다 — 실제 잠재 결함이라 여기서 닫는다. */
#define _POSIX_C_SOURCE 200809L
#include "radiation_field.h"
#include "seed_capability.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    double dlog = log(LUMINA_RADFIELD_NU_MAX_HZ / LUMINA_RADFIELD_NU_MIN_HZ) /
                  (double)LUMINA_RADFIELD_N_BINS;
    for (size_t b = 0; b <= LUMINA_RADFIELD_N_BINS; ++b)
        field->frequency_bin_edges.values[b] =
            LUMINA_RADFIELD_NU_MIN_HZ * exp((double)b * dlog);
    field->frequency_bin_edges.values[0] = LUMINA_RADFIELD_NU_MIN_HZ;
    field->frequency_bin_edges.values[LUMINA_RADFIELD_N_BINS] =
        LUMINA_RADFIELD_NU_MAX_HZ;
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
    free(shadow->line_jbar_cache.shell_id);
    free(shadow->line_jbar_cache.line_id);
    free(shadow->line_jbar_cache.profile_id);
    free(shadow->line_jbar_cache.profile_hash);
    free(shadow->line_jbar_cache.jbar_value);
    free(shadow->line_jbar_cache.validity);
    free(shadow->line_jbar_cache.sample_count);
    free(shadow->line_jbar_cache.variance_or_standard_error);
    free((char *)shadow->line_jbar_cache.q_set_hash);
    free(shadow->line_ids_compact);
    free(shadow->line_profile_hash_storage);
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
    double dlog = log(LUMINA_RADFIELD_NU_MAX_HZ / LUMINA_RADFIELD_NU_MIN_HZ) /
                  (double)LUMINA_RADFIELD_N_BINS;
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
    int mc = request->provenance_kind == RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH;
    if (!request->line_id || !request->line_q_set_hash ||
        strlen(request->line_q_set_hash) != 64 ||
        !request->line_profile_hash || request->line_profile_id == 0)
        return -1;
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
                    double s2 = (sq - sum * sum / N) / (N - 1.0);
                    if (s2 < 0.0) s2 = 0.0;
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
            if (!isfinite(request->line_jbar[i]) || request->line_jbar[i] < 0.0)
                return -1;
            if (v != LINE_JBAR_VALID && request->line_jbar[i] != 0.0) return -1;
            value[i] = request->line_jbar[i];
            validity[i] = (LineJbarValidityState)v;
            count[i] = 0; se[i] = 0.0;
        }
    }
    return 0;
}

static int radiation_field_line_cache_reserve(LineJbarCache *cache, size_t n)
{
    if (cache->entry_count == n && cache->jbar_value) return 0;
    free(cache->shell_id); free(cache->line_id); free(cache->profile_id);
    free(cache->profile_hash); free(cache->jbar_value); free(cache->validity);
    free(cache->sample_count); free(cache->variance_or_standard_error);
    cache->shell_id = malloc(n * sizeof(uint64_t));
    cache->line_id = malloc(n * sizeof(uint64_t));
    cache->profile_id = malloc(n * sizeof(uint64_t));
    cache->profile_hash = malloc(n * sizeof(const char *));
    cache->jbar_value = malloc(n * sizeof(double));
    cache->validity = malloc(n * sizeof(LineJbarValidityState));
    cache->sample_count = malloc(n * sizeof(uint64_t));
    cache->variance_or_standard_error = malloc(n * sizeof(double));
    cache->entry_count = n;
    return (cache->shell_id && cache->line_id && cache->profile_id &&
            cache->profile_hash && cache->jbar_value && cache->validity &&
            cache->sample_count && cache->variance_or_standard_error) ? 0 : -1;
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
    shadow->line_jbar_cache.generation.required_generation = request->generation;
    shadow->line_jbar_cache.units = field->units;
    shadow->line_jbar_cache.frame = field->frame;
    if (request->line_n > 0) {
        LineJbarCache *cache = &shadow->line_jbar_cache;
        if (radiation_field_line_cache_reserve(cache, lcells) != 0) {
            /* reserve failed AFTER J publish would break atomicity -- so the
             * reserve is the only allocation here and it precedes every cache
             * write; on failure the cache stays computed=0 (stale) while the
             * J field has already published.  Prevent that: reserve BEFORE
             * publish would be better, but reserve cannot fail after the
             * first iteration (same size).  Treat failure as fatal. */
            free(lv); free(lse); free(lval); free(lct);
            free(values); free(validity); free(count);
            return -1;
        }
        for (size_t q = 0; q < request->line_n; q++)
            for (size_t s = 0; s < request->n_shells; s++) {
                size_t i = q * request->n_shells + s;
                cache->shell_id[i] = (uint64_t)s;
                cache->line_id[i] = request->line_id[q];
                cache->profile_id[i] = request->line_profile_id;
                cache->profile_hash[i] = cache->q_set_hash; /* placeholder, set below */
                cache->jbar_value[i] = lv[i];
                cache->validity[i] = lval[i];
                cache->sample_count[i] = lct[i];
                cache->variance_or_standard_error[i] = lse[i];
            }
        free((char *)cache->q_set_hash);
        cache->q_set_hash = strdup(request->line_q_set_hash);
        free((char *)shadow->line_profile_hash_storage);
        shadow->line_profile_hash_storage = strdup(request->line_profile_hash);
        shadow->line_profile_id = request->line_profile_id;
        for (size_t i = 0; i < lcells; i++)
            cache->profile_hash[i] = shadow->line_profile_hash_storage;
        shadow->line_n_compact = request->line_n;
        free(shadow->line_ids_compact);
        shadow->line_ids_compact = malloc(request->line_n * sizeof(uint64_t));
        if (shadow->line_ids_compact)
            memcpy(shadow->line_ids_compact, request->line_id,
                   request->line_n * sizeof(uint64_t));
        cache->provenance.kind = request->provenance_kind;
        cache->provenance.producer = request->producer;
        cache->generation.computed_generation = request->generation;
    } else {
        shadow->line_jbar_cache.generation.computed_generation = 0;
        shadow->line_jbar_cache.provenance.kind = RADIATION_FIELD_PROVENANCE_NONE;
    }
    free(lv); free(lse); free(lval); free(lct);

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
    /* Canonical-grid identity (R5 edge shape): every edge is recomputed from
     * the A2-02 authority by the SAME expression owner-init uses, so any
     * interior tampering fails bit-exactly, not just monotonically.  One pass
     * of 4001 exp() per view refresh (once per commit) is negligible. */
    {
        const double *edges = field->frequency_bin_edges.values;
        if (edges[0] != LUMINA_RADFIELD_NU_MIN_HZ ||
            edges[LUMINA_RADFIELD_N_BINS] != LUMINA_RADFIELD_NU_MAX_HZ)
            return RADIATION_FIELD_VIEW_GRID;
        double dlog = log(LUMINA_RADFIELD_NU_MAX_HZ /
                          LUMINA_RADFIELD_NU_MIN_HZ) /
                      (double)LUMINA_RADFIELD_N_BINS;
        for (size_t b = 1; b < LUMINA_RADFIELD_N_BINS; ++b) {
            if (edges[b] != LUMINA_RADFIELD_NU_MIN_HZ * exp((double)b * dlog))
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

int radiation_field_line_jbar_view(const RadiationFieldOwner *owner,
                                   double expected_epoch,
                                   size_t expected_n_shells,
                                   uint64_t expected_generation,
                                   const char *expected_q_set_hash,
                                   uint64_t expected_profile_id,
                                   LineJbarView *out)
{
    if (!out) return LINE_JBAR_VIEW_QHASH;
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
    if (!expected_q_set_hash || !cache->q_set_hash ||
        strcmp(cache->q_set_hash, expected_q_set_hash) != 0)
        return LINE_JBAR_VIEW_QHASH;
    if (expected_profile_id == 0 ||
        owner->line_profile_id != expected_profile_id ||
        !owner->line_ids_compact || owner->line_n_compact == 0 ||
        !cache->jbar_value || !cache->validity || !cache->sample_count ||
        !cache->variance_or_standard_error)
        return LINE_JBAR_VIEW_PROFILE;
    out->n_lines = owner->line_n_compact;
    out->n_shells = expected_n_shells;
    out->line_id = owner->line_ids_compact;
    out->jbar = cache->jbar_value;
    out->validity = cache->validity;
    out->count = cache->sample_count;
    out->se = cache->variance_or_standard_error;
    out->generation = expected_generation;
    return LINE_JBAR_VIEW_OK;
}

int line_jbar_lookup(const LineJbarView *view, size_t shell,
                     uint64_t line_id, LineJbarValue *out)
{
    if (!view || !out || !view->jbar || shell >= view->n_shells) return -1;
    memset(out, 0, sizeof(*out));
    size_t a = 0, b = view->n_lines;
    while (a < b) {
        size_t m = (a + b) / 2;
        if (view->line_id[m] < line_id) a = m + 1; else b = m;
    }
    if (a >= view->n_lines || view->line_id[a] != line_id)
        return -2;                       /* MISS: distinct error, no value */
    size_t i = a * view->n_shells + shell;
    out->jbar = view->jbar[i];
    out->validity = view->validity[i];
    out->count = view->count[i];
    out->se = view->se[i];
    return 0;
}
