#include "radiation_field.h"

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

int radiation_field_shadow_gate_enabled(void)
{
    const char *value = getenv("LUMINA_RADFIELD_SHADOW");
    return value && atoi(value) != 0;
}

static void radiation_field_mark_all(RadiationField *field,
                                     RadiationFieldValidityState state)
{
    size_t cells = field->validity.n_shells * field->validity.n_bins;
    for (size_t i = 0; i < cells; ++i)
        field->validity.values[i] = state;
}

int radiation_field_shadow_init(RadiationFieldShadow *shadow, size_t n_shells)
{
    if (!shadow || n_shells == 0) return -1;
    memset(shadow, 0, sizeof(*shadow));
    if (!radiation_field_shadow_gate_enabled()) return 0;

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
    field->generation.required_generation = 1;
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
        radiation_field_shadow_free(shadow);
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

void radiation_field_shadow_free(RadiationFieldShadow *shadow)
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
    memset(shadow, 0, sizeof(*shadow));
}

int radiation_field_shadow_begin_mc(RadiationFieldShadow *shadow,
                                    const double *v_inner,
                                    const double *v_outer,
                                    size_t n_shells, double epoch,
                                    uint64_t required_generation)
{
    if (!shadow || !shadow->enabled) return 0;
    RadiationField *field = &shadow->field;
    if (!v_inner || !v_outer || n_shells != field->J_nu.n_shells ||
        required_generation == 0 || !isfinite(epoch) || epoch <= 0.0)
        return -1;
    if (field->generation.computed_generation != 0 &&
        required_generation != field->generation.computed_generation + 1)
        return -1;

    for (size_t s = 0; s < n_shells; ++s) {
        if (!isfinite(v_inner[s]) || !isfinite(v_outer[s]) ||
            v_inner[s] >= v_outer[s]) return -1;
        if (s > 0 && v_inner[s] != v_outer[s - 1]) return -1;
        field->shell_boundaries.values[s] = v_inner[s];
    }
    field->shell_boundaries.values[n_shells] = v_outer[n_shells - 1];
    field->epoch = epoch;
    field->frame = RADIATION_FIELD_FRAME_SHELL_COMOVING;
    field->generation.required_generation = required_generation;
    field->provenance.kind = RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH;
    field->provenance.producer = "CPU_MC_COMOVING_PATH_LENGTH_BIN_AVERAGE";
    field->provenance.raw_ledger_sha256 = NULL;
    field->provenance.contribution_count = 0;
    field->provenance.out_of_grid_contribution_count = 0;
    memset(field->J_nu.values, 0,
           radiation_field_cell_count(n_shells) * sizeof(double));
    memset(field->estimator_count_or_variance.count, 0,
           radiation_field_cell_count(n_shells) * sizeof(uint64_t));
    memset(shadow->accumulator.raw_path_length, 0,
           radiation_field_cell_count(n_shells) * sizeof(double));
    memset(shadow->accumulator.contribution_count, 0,
           radiation_field_cell_count(n_shells) * sizeof(uint64_t));
    shadow->accumulator.out_of_grid_contribution_count = 0;
    radiation_field_mark_all(field, RADIATION_FIELD_STALE);

    /* The A2-06 cache cannot outlive or lead its owner generation. */
    shadow->line_jbar_cache.generation.required_generation = required_generation;
    shadow->line_jbar_cache.generation.computed_generation = 0;
    shadow->line_jbar_cache.units = field->units;
    shadow->line_jbar_cache.frame = field->frame;
    shadow->line_jbar_cache.provenance.kind = RADIATION_FIELD_PROVENANCE_NONE;
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

int radiation_field_shadow_validate_owner(const RadiationFieldShadow *shadow)
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
            if (count == 0 || value != 0.0) return -1;
        } else if (validity == RADIATION_FIELD_VALID) {
            if (count == 0 || value <= 0.0) return -1;
        } else {
            return -1;
        }
    }
    return 0;
}

int radiation_field_shadow_commit_mc(RadiationFieldShadow *shadow,
                                     const double *volume,
                                     size_t n_shells,
                                     double time_simulation)
{
    if (!shadow || !shadow->enabled) return 0;
    RadiationField *field = &shadow->field;
    if (!volume || n_shells != field->J_nu.n_shells ||
        !isfinite(time_simulation) || time_simulation <= 0.0 ||
        field->frame != RADIATION_FIELD_FRAME_SHELL_COMOVING)
        return -1;

    uint64_t total = 0;
    for (size_t s = 0; s < n_shells; ++s) {
        if (!isfinite(volume[s]) || volume[s] <= 0.0) return -1;
        for (size_t b = 0; b < LUMINA_RADFIELD_N_BINS; ++b) {
            size_t index = s * (size_t)LUMINA_RADFIELD_N_BINS + b;
            uint64_t count = shadow->accumulator.contribution_count[index];
            double raw = shadow->accumulator.raw_path_length[index];
            double delta_nu = field->frequency_bin_edges.values[b + 1] -
                              field->frequency_bin_edges.values[b];
            field->estimator_count_or_variance.count[index] = count;
            total += count;
            if (count == 0) {
                field->J_nu.values[index] = 0.0;
                field->validity.values[index] = RADIATION_FIELD_UNSAMPLED;
            } else if (raw == 0.0) {
                field->J_nu.values[index] = 0.0;
                field->validity.values[index] = RADIATION_FIELD_EXACT_ZERO;
            } else if (raw > 0.0 && isfinite(raw) && delta_nu > 0.0) {
                /* A path-length integral divided by Delta-nu is the bin average;
                 * no bin-center sample enters this expression. */
                field->J_nu.values[index] = raw /
                    (4.0 * M_PI * volume[s] * time_simulation * delta_nu);
                field->validity.values[index] = RADIATION_FIELD_VALID;
            } else {
                return -1;
            }
        }
    }
    field->provenance.contribution_count = total;
    field->provenance.out_of_grid_contribution_count =
        shadow->accumulator.out_of_grid_contribution_count;
    field->generation.computed_generation =
        field->generation.required_generation;
    if (radiation_field_shadow_validate_owner(shadow) != 0) {
        field->generation.computed_generation = 0;
        radiation_field_mark_all(field, RADIATION_FIELD_STALE);
        return -1;
    }
    return radiation_field_shadow_dump_if_requested(shadow);
}

int radiation_field_shadow_dump_if_requested(const RadiationFieldShadow *shadow)
{
    if (!shadow || !shadow->enabled) return 0;
    const char *path = getenv("LUMINA_RADFIELD_SHADOW_DUMP");
    if (!path || !*path) return 0;
    FILE *stream = fopen(path, "w");
    if (!stream) {
        fprintf(stderr, "[RADFIELD-SHADOW][FATAL] cannot open %s: %s\n",
                path, strerror(errno));
        return -1;
    }
    const RadiationField *field = &shadow->field;
    fprintf(stream, "#schema=lumina-radiation-field-shadow-v1\n");
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
