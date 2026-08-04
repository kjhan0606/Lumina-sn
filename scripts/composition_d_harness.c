/* CPU-only order-D harness.  The parser under test is load_atomic_data() from
 * src/lumina_atomic.c; no composition logic is reproduced here. */
#include "lumina.h"

/* load_atomic_data() only queries this optional gate after composition passes.
 * Keeping it off makes the harness independent of plasma/CUDA subsystems. */
int nlte_element_wide_enabled(void) {
    return 0;
}

int main(int argc, char **argv) {
    AtomicData atom;
    char *end = NULL;
    long n_shells;
    int rc;

    if (argc != 3) {
        fprintf(stderr, "usage: %s DECK_DIR N_SHELLS\n", argv[0]);
        return 2;
    }
    n_shells = strtol(argv[2], &end, 10);
    if (!end || *end != '\0' || n_shells <= 0 || n_shells > INT32_MAX) {
        fprintf(stderr, "invalid N_SHELLS: %s\n", argv[2]);
        return 2;
    }

    memset(&atom, 0, sizeof(atom));
    rc = load_atomic_data(&atom, argv[1], (int)n_shells);
    free_atomic_data(&atom);
    printf("[COMPOSITION_D_HARNESS] load_atomic_data rc=%d\n", rc);
    return rc == 0 ? 0 : 1;
}
