#include "radiation_field.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_exact(FILE *stream, void *data, size_t size)
{
    return fread(data, 1, size, stream) == size ? 0 : -1;
}

static int write_exact(FILE *stream, const void *data, size_t size)
{
    return fwrite(data, 1, size, stream) == size ? 0 : -1;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s INPUT OUTPUT\n", argv[0]);
        return 2;
    }
    FILE *input = fopen(argv[1], "rb");
    if (!input) return 2;
    char magic[8];
    uint64_t n_shells_u64, generation;
    double epoch;
    if (read_exact(input, magic, sizeof(magic)) ||
        memcmp(magic, "A204IN01", 8) != 0 ||
        read_exact(input, &n_shells_u64, sizeof(n_shells_u64)) ||
        read_exact(input, &generation, sizeof(generation)) ||
        read_exact(input, &epoch, sizeof(epoch)) ||
        n_shells_u64 == 0 || n_shells_u64 > 1000) {
        fclose(input);
        return 2;
    }
    size_t n_shells = (size_t)n_shells_u64;
    size_t cells = n_shells * (size_t)LUMINA_RADFIELD_N_BINS;
    double *v_inner = (double *)malloc(n_shells * sizeof(double));
    double *v_outer = (double *)malloc(n_shells * sizeof(double));
    double *values = (double *)malloc(cells * sizeof(double));
    RadiationFieldValidityState *validity =
        (RadiationFieldValidityState *)malloc(cells * sizeof(*validity));
    if (!v_inner || !v_outer || !values || !validity ||
        read_exact(input, v_inner, n_shells * sizeof(double)) ||
        read_exact(input, v_outer, n_shells * sizeof(double)) ||
        read_exact(input, values, cells * sizeof(double)) ||
        read_exact(input, validity, cells * sizeof(*validity)) ||
        fgetc(input) != EOF || fclose(input) != 0) {
        free(v_inner); free(v_outer); free(values); free(validity);
        return 2;
    }

    RadiationFieldShadow owner;
    if (radiation_field_owner_init(&owner, n_shells) != 0) return 2;
    RadiationFieldCommitRequest request;
    memset(&request, 0, sizeof(request));
    request.provenance_kind = RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY;
    request.producer = "CMFGEN_EDDFACTOR_L0_REPLAY";
    request.generation = generation;
    request.epoch = epoch;
    request.n_shells = n_shells;
    request.v_inner = v_inner;
    request.v_outer = v_outer;
    request.source_n_bins = LUMINA_RADFIELD_N_BINS;
    request.source_frequency_bin_edges = owner.field.frequency_bin_edges.values;
    request.source_J_nu = values;
    request.source_validity = validity;
    request.statistic_kind = RADIATION_FIELD_DETERMINISTIC;
    if (radiation_field_commit(&owner, &request) != 0) {
        radiation_field_owner_free(&owner);
        free(v_inner); free(v_outer); free(values); free(validity);
        return 1;
    }

    FILE *output = fopen(argv[2], "wb");
    uint64_t committed = owner.field.generation.computed_generation;
    int rc = !output ||
        write_exact(output, "A204OUT1", 8) ||
        write_exact(output, &n_shells_u64, sizeof(n_shells_u64)) ||
        write_exact(output, &committed, sizeof(committed)) ||
        write_exact(output, owner.field.J_nu.values, cells * sizeof(double)) ||
        write_exact(output, owner.field.validity.values,
                    cells * sizeof(*owner.field.validity.values)) ||
        fclose(output) != 0;
    radiation_field_owner_free(&owner);
    free(v_inner); free(v_outer); free(values); free(validity);
    if (rc) return 2;
    printf("A2_04_REPLAY_COMMIT PASS shells=%zu bins=%d generation=%llu\n",
           n_shells, LUMINA_RADFIELD_N_BINS,
           (unsigned long long)committed);
    return 0;
}
