/**
 * LUMINA-SN Validation Framework
 * validation.h - Structures and functions for comparing C vs Python outputs
 *
 * Strategy: Dump packet state after each step to binary file.
 * Compare with equivalent Python output for numerical accuracy.
 */

#ifndef VALIDATION_H
#define VALIDATION_H

#include <stdint.h>
#include <stdio.h>
#include "rpacket.h"

/* ============================================================================
 * PACKET STATE SNAPSHOT - Matches Python debug output structure
 * ============================================================================ */
typedef struct {
    int64_t step_number;      /* Which step in the transport loop */
    double r;                 /* Position after step */
    double mu;                /* Direction after step */
    double nu;                /* Frequency after step */
    double energy;            /* Energy after step */
    int64_t shell_id;         /* Current shell */
    int64_t status;           /* Packet status */
    int64_t interaction_type; /* What interaction occurred */
    double distance;          /* Distance traveled in this step */
} PacketSnapshot;

/* ============================================================================
 * VALIDATION TRACE - Full history of one packet
 * ============================================================================ */
typedef struct {
    int64_t packet_index;
    int64_t n_snapshots;
    int64_t capacity;
    PacketSnapshot *snapshots;
} ValidationTrace;

/* --- Trace Management --- */

/**
 * Create a new validation trace with given capacity
 */
ValidationTrace *validation_trace_create(int64_t packet_index, int64_t capacity);

/**
 * Free a validation trace
 */
void validation_trace_free(ValidationTrace *trace);

/**
 * Record current packet state
 */
void validation_trace_record(ValidationTrace *trace, const RPacket *pkt,
                             InteractionType itype, double distance);

/**
 * Write trace to binary file (for comparison with Python)
 */
int validation_trace_write_binary(const ValidationTrace *trace, const char *filename);

/**
 * Write trace to human-readable CSV
 */
int validation_trace_write_csv(const ValidationTrace *trace, const char *filename);

/* ============================================================================
 * COMPARISON FUNCTIONS
 * ============================================================================ */

/**
 * Compare two traces, return number of mismatches
 *
 * @param trace_c     C implementation trace
 * @param trace_py    Python implementation trace (loaded from file)
 * @param tolerance   Relative tolerance for floating point comparison
 *
 * @return Number of steps with discrepancies (0 = perfect match)
 */
int64_t validation_compare_traces(const ValidationTrace *trace_c,
                                   const ValidationTrace *trace_py,
                                   double tolerance);

/**
 * Load a trace from binary file (Python output format)
 */
ValidationTrace *validation_trace_load_binary(const char *filename);

/* ============================================================================
 * SINGLE-PACKET LOOP WITH TRACING
 * ============================================================================ */

/**
 * Same as single_packet_loop but records state after each step
 */
void single_packet_loop_traced(RPacket *pkt, const NumbaModel *model,
                               const NumbaPlasma *plasma,
                               const MonteCarloConfig *config,
                               Estimators *estimators,
                               ValidationTrace *trace);

#endif /* VALIDATION_H */
