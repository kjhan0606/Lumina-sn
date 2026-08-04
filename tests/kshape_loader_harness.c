#include "lumina.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s DECK\n", argv[0]);
        return 2;
    }
    Geometry geo;
    OpacityState opacity;
    PlasmaState plasma;
    MCConfig config;
    memset(&geo, 0, sizeof(geo));
    memset(&opacity, 0, sizeof(opacity));
    memset(&plasma, 0, sizeof(plasma));
    memset(&config, 0, sizeof(config));
    if (load_tardis_reference_data(argv[1], &geo, &opacity, &plasma, &config) != 0)
        return 1;
    printf("KSHAPE_LOADER_PASS rows=%d cols=%d tau_seed=%.17g "
           "computed_generation=%llu required_generation=%llu\n",
           opacity.n_lines, opacity.n_shells, opacity.tau_sobolev[0],
           (unsigned long long)opacity.tau_computed_generation,
           (unsigned long long)opacity.tau_required_generation);
    free_geometry(&geo);
    free_opacity_state(&opacity);
    free_plasma_state(&plasma);
    return 0;
}
