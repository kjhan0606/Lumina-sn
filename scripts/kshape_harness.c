/* CPU-only K-SHAPE / K-FRESH harness (driver-built measurement instrument).
 *
 * The order-D harness calls load_atomic_data(), which does NOT contain the
 * K-SHAPE contract validator.  That validator lives in
 * load_tardis_reference_data() (lumina_atomic.c), together with the
 * tau_sobolev.npy and transition_probabilities.npy loads.  A negative-control
 * battery run against the wrong entry point reports PASS for every injected
 * defect, which is how this file came to exist.
 *
 *   usage: kshape_harness DECK_DIR
 *   exit 0 on successful load, nonzero on any FATAL contract violation.
 */
#include <stdio.h>
#include <string.h>
#include "lumina.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s DECK_DIR\n", argv[0]);
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

    int rc = load_tardis_reference_data(argv[1], &geo, &opacity, &plasma, &config);
    printf("[KSHAPE_HARNESS] load_tardis_reference_data rc=%d n_shells=%d n_lines=%d\n",
           rc, geo.n_shells, opacity.n_lines);
    return rc == 0 ? 0 : 1;
}
