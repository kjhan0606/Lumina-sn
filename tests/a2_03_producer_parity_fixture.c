#include "lumina.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    RadiationFieldShadow shadow;
    if (radiation_field_owner_init(&shadow, 1) != 0) return 2;
    const double v_inner[1] = {1.0e8}, v_outer[1] = {2.0e8};
    if (radiation_field_begin_mc(&shadow, v_inner, v_outer, 1,
                                        86400.0, 1) != 0) return 2;

    Estimators estimator;
    memset(&estimator, 0, sizeof(estimator));
    estimator.n_shells = 1;
    estimator.j_estimator = (double *)calloc(1, sizeof(double));
    estimator.nu_bar_estimator = (double *)calloc(1, sizeof(double));
    estimator.nlte_n_freq_bins = NLTE_N_FREQ_BINS;
    estimator.nlte_nu_min = NLTE_NU_MIN;
    estimator.nlte_d_log_nu =
        log(NLTE_NU_MAX / NLTE_NU_MIN) / (double)NLTE_N_FREQ_BINS;
    estimator.j_nu_estimator = (double *)calloc(
        NLTE_N_FREQ_BINS, sizeof(double));
    if (!estimator.j_estimator || !estimator.nu_bar_estimator ||
        !estimator.j_nu_estimator) return 2;
    if (shadow.enabled) {
        estimator.radiation_field_accumulator =
            radiation_field_accumulator_create(1);
        if (!estimator.radiation_field_accumulator) return 2;
    }

    RPacket packet;
    memset(&packet, 0, sizeof(packet));
    packet.current_shell_id = 0;
    update_base_estimators(&packet, 3.0, &estimator, 5.0e14, 2.0);
    update_base_estimators(&packet, 7.0, &estimator, 8.0e14, 4.0);
    update_base_estimators(&packet, 11.0, &estimator, 8.0e14, 5.0);

    if (shadow.enabled && radiation_field_accumulator_reduce(
            &shadow.accumulator,
            estimator.radiation_field_accumulator) != 0) return 2;
    const double volume[1] = {9.0};
    RadiationFieldCommitRequest request;
    memset(&request, 0, sizeof(request));
    request.provenance_kind = RADIATION_FIELD_PROVENANCE_MC_PATH_LENGTH;
    request.producer = "A2_03_REGRESSION_MC";
    request.generation = 1;
    request.epoch = 86400.0;
    request.n_shells = 1;
    request.v_inner = v_inner;
    request.v_outer = v_outer;
    request.source_n_bins = LUMINA_RADFIELD_N_BINS;
    request.statistic_kind = RADIATION_FIELD_ESTIMATOR_COUNT;
    request.source_count = shadow.accumulator.contribution_count;
    request.raw_path_length = shadow.accumulator.raw_path_length;
    request.volume = volume;
    request.time_simulation = 13.0;
    request.out_of_grid_contribution_count =
        shadow.accumulator.out_of_grid_contribution_count;
    if (radiation_field_commit(&shadow, &request) != 0)
        return 2;

    FILE *stream = fopen("legacy.bin", "wb");
    if (!stream) return 2;
    if (fwrite(estimator.j_estimator, sizeof(double), 1, stream) != 1 ||
        fwrite(estimator.nu_bar_estimator, sizeof(double), 1, stream) != 1 ||
        fwrite(estimator.j_nu_estimator, sizeof(double), NLTE_N_FREQ_BINS,
               stream) != NLTE_N_FREQ_BINS || fclose(stream) != 0)
        return 2;

    radiation_field_accumulator_free(
        estimator.radiation_field_accumulator);
    free(estimator.j_estimator);
    free(estimator.nu_bar_estimator);
    free(estimator.j_nu_estimator);
    radiation_field_owner_free(&shadow);
    return 0;
}
