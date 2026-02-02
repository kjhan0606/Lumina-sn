/**
 * LUMINA-SN Debug RNG Override
 * debug_rng.h - Read pre-generated random numbers from file
 *
 * Purpose: Synchronize RNG with TARDIS (Python) for validation.
 * TARDIS uses NumPy's Mersenne Twister; LUMINA uses Xorshift64*.
 * This module injects the exact Python RNG stream into C.
 *
 * Usage:
 *   1. Generate RNG stream: python generate_rng_stream.py
 *   2. Compile with debug_rng.c linked
 *   3. Run: ./test_transport --trace --seed 23111963 --n_packets 1 --inject-rng tardis_rng_stream.txt
 */

#ifndef DEBUG_RNG_H
#define DEBUG_RNG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Maximum RNG pool size */
#define DEBUG_RNG_POOL_SIZE 10000

/* Global RNG pool and index - EXTERN declarations (defined in debug_rng.c) */
extern double g_debug_rng_pool[DEBUG_RNG_POOL_SIZE];
extern int g_debug_rng_index;
extern int g_debug_rng_loaded;
extern int g_debug_rng_count;

/**
 * Load RNG stream from file.
 * Call once at program start.
 *
 * @param filename Path to RNG stream file (one double per line)
 * @return Number of values loaded, or -1 on error
 */
int debug_rng_load(const char *filename);

/**
 * Get next random number from injected pool.
 * Replaces rng_uniform() when injected RNG is enabled.
 *
 * @return Random number from pool, or exits if exhausted
 */
double debug_rng_next(void);

/**
 * Reset RNG index to beginning.
 * Call between packets if needed.
 */
void debug_rng_reset(void);

/**
 * Get current RNG index (for debugging).
 */
int debug_rng_get_index(void);

/**
 * Skip/burn N random numbers (advance index without using values).
 * Useful for aligning with TARDIS which may consume RNG during init.
 *
 * @param n Number of RNG values to skip
 */
void debug_rng_skip(int n);

#endif /* DEBUG_RNG_H */
