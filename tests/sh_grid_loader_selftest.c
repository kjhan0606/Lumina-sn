#include "lumina.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int write_asset(const char *path, double nu_min, int trailing)
{
    FILE *stream = fopen(path, "wb");
    uint32_t magic = 0x434D4644u, version = 1u;
    int32_t n_levels = 1, n_freq = NLTE_N_FREQ_BINS;
    int8_t flag = 1;
    unsigned char pad[7] = {0};
    double sigma = 1.0;
    if (!stream) return -1;
    if (fwrite(&magic, sizeof(magic), 1, stream) != 1 ||
        fwrite(&version, sizeof(version), 1, stream) != 1 ||
        fwrite(&n_levels, sizeof(n_levels), 1, stream) != 1 ||
        fwrite(&n_freq, sizeof(n_freq), 1, stream) != 1 ||
        fwrite(&nu_min, sizeof(nu_min), 1, stream) != 1 ||
        fwrite(&(double){NLTE_NU_MAX}, sizeof(double), 1, stream) != 1 ||
        fwrite(&flag, sizeof(flag), 1, stream) != 1 ||
        fwrite(pad, sizeof(pad), 1, stream) != 1) {
        fclose(stream);
        return -1;
    }
    for (int b = 0; b < NLTE_N_FREQ_BINS; ++b)
        if (fwrite(&sigma, sizeof(sigma), 1, stream) != 1) {
            fclose(stream);
            return -1;
        }
    if (trailing) fputc(0, stream);
    return fclose(stream);
}

static void clear_asset(AtomicData *atom)
{
    free(atom->cmfgen_has_sigma);
    free(atom->cmfgen_sigma_bf);
    atom->cmfgen_has_sigma = NULL;
    atom->cmfgen_sigma_bf = NULL;
    atom->cmfgen_loaded = 0;
}

int main(void)
{
    char path[256];
    AtomicData atom;
    memset(&atom, 0, sizeof(atom));
    atom.n_levels = 1;
    snprintf(path, sizeof(path), "/tmp/lumina-sh-grid-loader-%ld.bin",
             (long)getpid());

    if (write_asset(path, NLTE_NU_MIN, 0) != 0 ||
        load_cmfgen_sigma_bf(&atom, path) != 0 || !atom.cmfgen_loaded ||
        atom.cmfgen_n_freq_bins != NLTE_N_FREQ_BINS ||
        atom.cmfgen_nu_min != NLTE_NU_MIN || atom.cmfgen_nu_max != NLTE_NU_MAX) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL positive\n");
        remove(path);
        clear_asset(&atom);
        return 1;
    }
    clear_asset(&atom);

    if (write_asset(path, 1.5e14, 0) != 0 ||
        load_cmfgen_sigma_bf(&atom, path) == 0 || atom.cmfgen_loaded) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL stale-range\n");
        remove(path);
        clear_asset(&atom);
        return 1;
    }
    if (write_asset(path, NLTE_NU_MIN, 1) != 0 ||
        load_cmfgen_sigma_bf(&atom, path) == 0 || atom.cmfgen_loaded) {
        fprintf(stderr, "SH_GRID_LOADER_SELFTEST_FAIL trailing-byte\n");
        remove(path);
        clear_asset(&atom);
        return 1;
    }
    remove(path);
    clear_asset(&atom);
    printf("SH_GRID_LOADER_SELFTEST PASS bins=%d stale_range=REJECTED "
           "trailing_byte=REJECTED\n", NLTE_N_FREQ_BINS);
    return 0;
}
