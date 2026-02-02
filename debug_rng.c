/**
 * LUMINA-SN Debug RNG Override
 * debug_rng.c - Implementation of RNG injection for TARDIS validation
 */

#include "debug_rng.h"

/* Global RNG pool and index - actual definitions */
double g_debug_rng_pool[DEBUG_RNG_POOL_SIZE];
int g_debug_rng_index = 0;
int g_debug_rng_loaded = 0;
int g_debug_rng_count = 0;

int debug_rng_load(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[DEBUG_RNG] Error: Cannot open %s\n", filename);
        return -1;
    }

    g_debug_rng_count = 0;
    g_debug_rng_index = 0;
    char line[256];

    while (fgets(line, sizeof(line), fp) && g_debug_rng_count < DEBUG_RNG_POOL_SIZE) {
        /* Skip comments and empty lines */
        if (line[0] == '#' || line[0] == '\n') continue;

        double val;
        if (sscanf(line, "%lf", &val) == 1) {
            g_debug_rng_pool[g_debug_rng_count++] = val;
        }
    }

    fclose(fp);
    g_debug_rng_loaded = 1;

    fprintf(stderr, "[DEBUG_RNG] Loaded %d values from %s\n",
            g_debug_rng_count, filename);

    return g_debug_rng_count;
}

double debug_rng_next(void) {
    if (!g_debug_rng_loaded) {
        fprintf(stderr, "[DEBUG_RNG] Error: RNG pool not loaded! Call debug_rng_load() first.\n");
        exit(1);
    }

    if (g_debug_rng_index >= g_debug_rng_count) {
        fprintf(stderr, "[DEBUG_RNG] Error: RNG pool exhausted at index %d\n",
                g_debug_rng_index);
        exit(1);
    }

    double val = g_debug_rng_pool[g_debug_rng_index];

    #ifdef DEBUG_RNG_VERBOSE
    fprintf(stderr, "[DEBUG_RNG] [%d] = %.18e\n", g_debug_rng_index, val);
    #endif

    g_debug_rng_index++;
    return val;
}

void debug_rng_reset(void) {
    g_debug_rng_index = 0;
    fprintf(stderr, "[DEBUG_RNG] Reset index to 0\n");
}

int debug_rng_get_index(void) {
    return g_debug_rng_index;
}

void debug_rng_skip(int n) {
    if (g_debug_rng_index + n > g_debug_rng_count) {
        fprintf(stderr, "[DEBUG_RNG] Warning: Skip %d would exceed pool size %d, capping at max\n",
                n, g_debug_rng_count);
        g_debug_rng_index = g_debug_rng_count;
    } else {
        fprintf(stderr, "[DEBUG_RNG] Skipping %d values (index %d -> %d)\n",
                n, g_debug_rng_index, g_debug_rng_index + n);
        g_debug_rng_index += n;
    }
}
