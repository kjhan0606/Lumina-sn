#ifndef LUMINA_JNU_SEED_H
#define LUMINA_JNU_SEED_H

#include "radiation_field.h"
#include "seed_capability.h"

#include <stddef.h>
#include <stdint.h>

#define LUMINA_JNU_SEED_MAGIC "LUMINAJNUSEED1\0"
#define LUMINA_JNU_SEED_VERSION 1U
#define LUMINA_JNU_SEED_SHAPE_SHA256 LUMINA_RADFIELD_UNION_SHA256

typedef enum {
    JNU_SEED_OK = 0,
    JNU_SEED_IO_OR_SCHEMA = 2,
    JNU_SEED_BLOCKED_INCOMPLETE_COVERAGE = 3,
    JNU_SEED_FORBIDDEN_FALLBACK = 5
} JnuSeedStatus;

/* Fixed-width on-disk prefix. Arrays follow in this exact order:
 * shell_id[n_shells], shell_edges[n_shells+1], frequency_edges[n_bins+1],
 * validity[n_shells*n_bins], J_nu[n_shells*n_bins]. All numbers are native
 * little-endian IEEE-754/uint64; endian_tag makes cross-endian input fail. */
typedef struct {
    char magic[16];
    uint32_t version;
    uint32_t endian_tag;
    uint64_t n_shells;
    uint64_t n_bins;
    uint32_t units;
    uint32_t frame;
    uint32_t provenance;
    uint32_t reserved;
    double epoch;
    char shape_sha256[65];
    char edge_sha256[65];
    char source_payload_sha256[65];
    char source_geometry_sha256[65];
} JnuSeedDiskHeader;

typedef struct {
    uint64_t seed_files_opened;
    uint64_t seed_cells_loaded;
    uint64_t seed_invalid_cells;
    uint64_t shape_hash_failures;
    uint64_t edge_hash_failures;
    uint64_t shell_identity_failures;
    uint64_t coverage_failures_s44_s49;
    uint64_t hold_attempts;
    uint64_t extrapolation_attempts;
    uint64_t neighbor_copy_attempts;
    uint64_t zero_fill_attempts;
    uint64_t seed_fallback_attempts;
    uint64_t partial_seed_publish_attempts;
} JnuSeedCounters;

/* Load a native seed atomically into generation zero of an initialized owner.
 * expected_shell_edges and expected_shell_ids are the A2-02 geometry identity.
 * No owner mutation occurs unless every required cell is valid. */
JnuSeedStatus jnu_seed_load_native(const char *path,
                                   const double *expected_shell_edges,
                                   const uint64_t *expected_shell_ids,
                                   size_t expected_n_shells,
                                   double expected_epoch,
                                   RadiationFieldOwner *owner,
                                   SeedCapability *capability,
                                   JnuSeedCounters *counters,
                                   char manifest_sha256[65]);

int jnu_seed_sha256_file(const char *path, char out[65]);
void jnu_seed_report(const JnuSeedCounters *counters);

#endif
