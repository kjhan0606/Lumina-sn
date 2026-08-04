#include "lumina_cmf_field.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define H_PLANCK 6.62607015e-27
#define K_BOLTZMANN 1.380649e-16
#define C_LIGHT 2.99792458e10

typedef struct {
    double temperature;
    double scale;
} BlackbodyBoundary;

typedef struct {
    char sha256[65];
    uint64_t iteration;
    uint64_t field_generation;
} SidecarContract;

static int parse_size_nonzero(const char *text, size_t *value)
{
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0u ||
        parsed > SIZE_MAX) return 0;
    *value = (size_t)parsed;
    return 1;
}

static int parse_size_zero_ok(const char *text, size_t *value)
{
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > SIZE_MAX)
        return 0;
    *value = (size_t)parsed;
    return 1;
}

static int parse_positive_double(const char *text, double *value)
{
    char *end = NULL;
    errno = 0;
    *value = strtod(text, &end);
    return errno == 0 && end != text && *end == '\0' &&
           isfinite(*value) && *value > 0.0;
}

static double blackbody(void *opaque, double p_cm, double mu, double nu_hz)
{
    const BlackbodyBoundary *boundary = (const BlackbodyBoundary *)opaque;
    const double x = H_PLANCK * nu_hz / (K_BOLTZMANN * boundary->temperature);
    double value;
    (void)p_cm;
    (void)mu;
    if (!(x > 0.0) || x >= 700.0) return 0.0;
    value = (2.0 * H_PLANCK * nu_hz * nu_hz * nu_hz / (C_LIGHT * C_LIGHT)) /
            expm1(x);
    return boundary->scale * value;
}

static const char *unique_key(const char *json, const char *key)
{
    const char *first = strstr(json, key);
    if (first == NULL || strstr(first + strlen(key), key) != NULL) return NULL;
    first += strlen(key);
    while (isspace((unsigned char)*first)) ++first;
    if (*first++ != ':') return NULL;
    while (isspace((unsigned char)*first)) ++first;
    return first;
}

static int json_string(const char *json, const char *key,
                       char *value, size_t capacity)
{
    const char *at = unique_key(json, key);
    const char *end;
    size_t length;
    if (at == NULL || *at++ != '"') return 0;
    end = strchr(at, '"');
    if (end == NULL) return 0;
    length = (size_t)(end - at);
    if (length + 1u > capacity) return 0;
    memcpy(value, at, length);
    value[length] = '\0';
    return 1;
}

static int json_u64(const char *json, const char *key, uint64_t *value)
{
    const char *at = unique_key(json, key);
    char *end = NULL;
    unsigned long long parsed;
    if (at == NULL) return 0;
    errno = 0;
    parsed = strtoull(at, &end, 10);
    if (errno != 0 || end == at) return 0;
    while (isspace((unsigned char)*end)) ++end;
    if (*end != ',' && *end != '}') return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int json_bool(const char *json, const char *key, int *value)
{
    const char *at = unique_key(json, key);
    const char *end;
    if (at == NULL) return 0;
    if (strncmp(at, "true", 4u) == 0) {
        end = at + 4u;
        while (isspace((unsigned char)*end)) ++end;
        if (*end != ',' && *end != '}') return 0;
        *value = 1;
        return 1;
    }
    if (strncmp(at, "false", 5u) == 0) {
        end = at + 5u;
        while (isspace((unsigned char)*end)) ++end;
        if (*end != ',' && *end != '}') return 0;
        *value = 0;
        return 1;
    }
    return 0;
}

static int json_double(const char *json, const char *key, double *value)
{
    const char *at = unique_key(json, key);
    char *end = NULL;
    if (at == NULL) return 0;
    errno = 0;
    *value = strtod(at, &end);
    if (errno != 0 || end == at || !isfinite(*value)) return 0;
    while (isspace((unsigned char)*end)) ++end;
    return *end == ',' || *end == '}';
}

static int read_contract_sidecar(const char *path, SidecarContract *contract)
{
    FILE *stream = NULL;
    char *json = NULL;
    char schema[32];
    long length;
    int post_damping, coherent_frozen, frequency_descending;
    int eta_bitwise;
    double eta_max_abs;
    size_t i;
    int ok = 0;
    stream = fopen(path, "rb");
    if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) goto done;
    length = ftell(stream);
    /* E5 sidecars carry the pre-EPAY by-band/by-shell diagnostic arrays and
     * are therefore larger than the original 16 KiB contract envelope. */
    if (length <= 0 || length > 1048576 || fseek(stream, 0, SEEK_SET) != 0)
        goto done;
    json = (char *)calloc((size_t)length + 1u, 1u);
    if (json == NULL || fread(json, 1u, (size_t)length, stream) != (size_t)length)
        goto done;
    if (memchr(json, '\0', (size_t)length) != NULL) goto done;
    if (!json_string(json, "\"schema\"", schema, sizeof(schema)) ||
        strcmp(schema, "LCMFCE01-v1") != 0 ||
        !json_string(json, "\"sha256\"", contract->sha256,
                     sizeof(contract->sha256)) ||
        !json_u64(json, "\"iteration\"", &contract->iteration) ||
        !json_u64(json, "\"field_generation\"", &contract->field_generation) ||
        !json_bool(json, "\"post_damping\"", &post_damping) ||
        !json_bool(json, "\"coherent_frozen\"", &coherent_frozen) ||
        !json_bool(json, "\"frequency_descending\"", &frequency_descending) ||
        !json_bool(json, "\"eta_decomposition_bitwise\"", &eta_bitwise) ||
        !json_double(json, "\"eta_decomposition_max_abs\"", &eta_max_abs))
        goto done;
    if (!post_damping || !coherent_frozen || !frequency_descending ||
        !eta_bitwise || eta_max_abs != 0.0 || contract->iteration != 10u ||
        contract->field_generation != 10u || strlen(contract->sha256) != 64u)
        goto done;
    for (i = 0u; i < 64u; ++i) {
        if (!isxdigit((unsigned char)contract->sha256[i])) goto done;
        contract->sha256[i] = (char)tolower((unsigned char)contract->sha256[i]);
    }
    ok = 1;
done:
    if (stream != NULL) fclose(stream);
    free(json);
    return ok;
}

static int load_frozen_json_sidecar(const char *binary_path,
                                    const char *sidecar_path,
                                    LCMFFrozenField *field,
                                    SidecarContract *contract,
                                    LCMFError *error)
{
    char temporary[] = "/tmp/stage31_manifest_XXXXXX";
    char line[80];
    int descriptor = -1;
    int length;
    int status = LCMF_ESCHEMA;
    if (!read_contract_sidecar(sidecar_path, contract)) {
        (void)fprintf(stderr, "sidecar failed closed contract validation: %s\n",
                      sidecar_path);
        return LCMF_ESCHEMA;
    }
    descriptor = mkstemp(temporary);
    if (descriptor < 0) return LCMF_EIO;
    length = snprintf(line, sizeof(line), "sha256=%s\n", contract->sha256);
    if (length <= 0 || (size_t)length >= sizeof(line) ||
        write(descriptor, line, (size_t)length) != length) {
        (void)close(descriptor);
        (void)unlink(temporary);
        return LCMF_EIO;
    }
    if (close(descriptor) != 0) {
        (void)unlink(temporary);
        return LCMF_EIO;
    }
    descriptor = -1;
    status = lumina_cmf_frozen_load(binary_path, temporary, field, error);
    (void)unlink(temporary);
    return status;
}

int main(int argc, char **argv)
{
    const char *binary_path;
    const char *sidecar_path;
    const char *output_path;
    size_t shell, n_mu, k;
    double temperature, scale, radius, velocity;
    LCMFFrozenField frozen;
    SidecarContract contract;
    BlackbodyBoundary boundary;
    LCMFInput input;
    LCMFOptions options;
    LCMFResult result;
    LCMFError load_error;
    FILE *output = NULL;
    int status;
    memset(&frozen, 0, sizeof(frozen));
    memset(&contract, 0, sizeof(contract));
    memset(&result, 0, sizeof(result));
    memset(&load_error, 0, sizeof(load_error));
    if (argc != 8 || !parse_size_zero_ok(argv[3], &shell) ||
        !parse_size_nonzero(argv[4], &n_mu) ||
        !parse_positive_double(argv[5], &temperature) ||
        !parse_positive_double(argv[6], &scale)) {
        (void)fprintf(stderr,
                      "usage: %s FROZEN SIDECAR SHELL NMU T_INNER_K BB_SCALE OUTPUT.tsv\n",
                      argv[0]);
        return 2;
    }
    binary_path = argv[1];
    sidecar_path = argv[2];
    output_path = argv[7];
    status = load_frozen_json_sidecar(binary_path, sidecar_path, &frozen,
                                      &contract, &load_error);
    if (status != LCMF_OK) {
        (void)fprintf(stderr, "frozen load failed: %s: %s\n",
                      lumina_cmf_status_string(status), load_error.message);
        return 1;
    }
    if (shell >= frozen.nr || frozen.nr != 50u || frozen.nnu != 1000u ||
        frozen.iteration != 10u || frozen.field_generation != 10u ||
        (frozen.flags & (LCMF_FROZEN_POST_DAMP |
                         LCMF_FROZEN_COHERENT |
                         LCMF_FROZEN_FREQUENCY_DESCENDING)) !=
                        (LCMF_FROZEN_POST_DAMP |
                         LCMF_FROZEN_COHERENT |
                         LCMF_FROZEN_FREQUENCY_DESCENDING)) {
        (void)fprintf(stderr, "frozen payload violates the parity bench contract\n");
        lumina_cmf_frozen_free(&frozen);
        return 1;
    }
    radius = 0.5 * (frozen.r_edge[shell] + frozen.r_edge[shell + 1u]);
    velocity = radius / frozen.t_exp_s / 1.0e5;
    boundary.temperature = temperature;
    boundary.scale = scale;
    memset(&input, 0, sizeof(input));
    memset(&options, 0, sizeof(options));
    input.nr = (size_t)frozen.nr;
    input.nnu = (size_t)frozen.nnu;
    input.r_edge = frozen.r_edge;
    input.nu = frozen.nu;
    input.chi_total = frozen.chi_total;
    input.eta_fixed = frozen.eta_total;
    input.chi_coherent = NULL;
    input.t_exp_s = frozen.t_exp_s;
    input.inner_bc = LCMF_BC_IRRADIATION;
    input.scatter_mode = LCMF_SCAT_NONE;
    input.inner_irradiation = blackbody;
    input.boundary_ctx = &boundary;
    options.n_mu = n_mu;
    options.n_r_eval = 1u;
    options.r_eval = &radius;
    options.max_source_iter = 1u;
    options.source_rtol = 1.0e-12;
    options.frequency_advection = 1;
    status = lumina_cmf_field_solve(&input, &options, &result);
    if (status != LCMF_OK) {
        (void)fprintf(stderr,
                      "deterministic solve failed: %s: %s radial=%zu frequency=%zu ray=%zu segment=%zu substep=%zu value=%.17g interval=[%.17g,%.17g] scale=%.17g h=%.17g B_trunc=%.17g\n",
                      lumina_cmf_status_string(status), result.error.message,
                      result.error.radial_index, result.error.frequency_index,
                      result.error.ray_index, result.error.segment_index,
                      result.error.substep_index, result.error.value,
                      result.error.interval_lower, result.error.interval_upper,
                      result.error.term_previous, result.error.term_previous2,
                      result.error.theoretical_limit);
        lumina_cmf_result_free(&result);
        lumina_cmf_frozen_free(&frozen);
        return 1;
    }
    output = fopen(output_path, "wb");
    if (output == NULL) {
        lumina_cmf_result_free(&result);
        lumina_cmf_frozen_free(&frozen);
        return 1;
    }
    (void)fprintf(output,
                  "# schema=stage31-cmf-field-v1 shell=%zu nmu=%zu nr=%" PRIu64 " nnu=%" PRIu64 " iteration=%" PRIu64 " generation=%" PRIu64 " post_damp=1 t_exp_s=%.17g radius_cm=%.17g velocity_kms=%.17g T_inner_K=%.17g bb_scale=%.17g transport_residual=%.17g source_residual=%.17g source_iterations=%zu clamp=%" PRIu64 " bdf_eta_negative=%" PRIu64 " solution_negative_excess=%" PRIu64 " solution_subtruncation=%" PRIu64 " solution_sign_indeterminate_subtruncation=%" PRIu64 " solution_roundoff_enclosure_restart=%" PRIu64 " solution_subtruncation_min=%.17g solution_subtruncation_min_frequency=%zu solution_subtruncation_min_ray=%zu solution_subtruncation_min_segment=%zu solution_subtruncation_min_substep=%zu solution_subtruncation_first_value=%.17g solution_subtruncation_first_bound=%.17g solution_subtruncation_first_h=%.17g solution_subtruncation_first_scale=%.17g solution_subtruncation_first_frequency=%zu solution_subtruncation_first_ray=%zu solution_subtruncation_first_segment=%zu solution_subtruncation_first_substep=%zu sign_uncertain=%" PRIu64 " nonfinite=%" PRIu64 " sha256=%s\n",
                  shell, n_mu, frozen.nr, frozen.nnu, frozen.iteration,
                  frozen.field_generation, frozen.t_exp_s, radius, velocity,
                  temperature, scale, result.transport_resid_linf,
                  result.source_resid_linf, result.source_iterations,
                  result.clamp_count, result.bdf_eta_negative_count,
                  result.solution_negative_excess_count,
                  result.solution_subtruncation_count,
                  result.solution_sign_indeterminate_subtruncation_count,
                  result.solution_roundoff_enclosure_restart_count,
                  result.solution_subtruncation_min,
                  result.solution_subtruncation_min_location.frequency_index,
                  result.solution_subtruncation_min_location.ray_index,
                  result.solution_subtruncation_min_location.segment_index,
                  result.solution_subtruncation_min_location.substep_index,
                  result.solution_subtruncation_first.value,
                  result.solution_subtruncation_first_bound,
                  result.solution_subtruncation_first_h,
                  result.solution_subtruncation_first_scale,
                  result.solution_subtruncation_first.frequency_index,
                  result.solution_subtruncation_first.ray_index,
                  result.solution_subtruncation_first.segment_index,
                  result.solution_subtruncation_first.substep_index,
                  result.sign_uncertain_count, result.nonfinite_count,
                  contract.sha256);
    (void)fprintf(output, "k\tnu_hz\tdnu_hz\tJ_det\tJ_producer\n");
    for (k = 0u; k < (size_t)frozen.nnu; ++k) {
        const size_t q = shell * (size_t)frozen.nnu + k;
        (void)fprintf(output, "%zu\t%.17g\t%.17g\t%.17g\t%.17g\n",
                      k, frozen.nu[k], frozen.dnu[k], result.J[k],
                      frozen.J_producer[q]);
    }
    if (fclose(output) != 0) status = LCMF_EIO;
    lumina_cmf_result_free(&result);
    lumina_cmf_frozen_free(&frozen);
    return status == LCMF_OK ? 0 : 1;
}
