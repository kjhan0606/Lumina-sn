#ifndef LUMINA_SEED_CAPABILITY_H
#define LUMINA_SEED_CAPABILITY_H

/* A2-16 (SPEC_A2_13_18_V1 §3): the scalar (W, T_rad) seed is legal ONLY while
 * building the generation-0 bootstrap field.  The capability is revoked the
 * instant the first RadiationField commit succeeds; any later read fails
 * regardless of value.
 *
 *   SEED_UNOPENED -> SEED_G0_ACTIVE -> FIRST_FIELD_COMMITTED -> SEED_REVOKED
 *
 *   seed_read_allowed iff
 *     required_generation == 0 AND computed_generation == 0
 *     AND no field commit has succeeded
 *     AND the capability epoch/hash matches the loader manifest
 *
 * Runtime never parses legacy (W, T_rad) columns as a physics input: the
 * offline converter serializes them into an explicit J_nu array with
 * provenance DILUTE_PLANCK_LEGACY_APPROXIMATION.  This header owns only the
 * temporal capability; the seed payload lives with the RadiationField owner.
 */

#include <stddef.h>
#include <stdint.h>

typedef enum {
    SEED_UNOPENED = 0,
    SEED_G0_ACTIVE = 1,
    SEED_FIRST_FIELD_COMMITTED = 2,
    SEED_REVOKED = 3
} SeedCapabilityState;

typedef enum {
    SEED_OK = 0,
    SEED_ERR_NULL = -1,
    SEED_ERR_WRONG_STATE = -2,       /* read attempted outside G0_ACTIVE */
    SEED_ERR_GENERATION = -3,        /* required/computed generation != 0 */
    SEED_ERR_COMMITTED = -4,         /* a field commit already succeeded */
    SEED_ERR_MANIFEST = -5,          /* epoch/hash mismatch with loader */
    SEED_ERR_REVOKED = -6,           /* post-revocation read */
    SEED_ERR_OBSOLETE_OPTION = -7    /* legacy scalar env still set */
} SeedStatus;

typedef struct {
    SeedCapabilityState state;
    double   epoch;                  /* must match the loader manifest */
    char     manifest_sha256[65];    /* seed payload identity */
    uint64_t reads_allowed;          /* successful G0 seed reads */
    uint64_t reads_denied;           /* denied attempts, by any reason */
    uint64_t post_g0_read_attempts;  /* the N16-2/3 negative-control counter */
    uint64_t revocations;
    uint64_t hold_attempts;          /* seed reuse across generations (N16-4) */
} SeedCapability;

/* Open the capability for generation 0.  Fails if a commit already happened. */
SeedStatus seed_capability_open(SeedCapability *cap, double epoch,
                                const char *manifest_sha256);

/* Gate every seed read.  required/computed generation come from the field
 * owner so the check cannot drift from the published state. */
SeedStatus seed_capability_check_read(SeedCapability *cap,
                                      uint64_t required_generation,
                                      uint64_t computed_generation,
                                      double epoch,
                                      const char *manifest_sha256);

/* Called by the commit choke point the moment the first commit succeeds:
 * revoke FIRST, then the caller zeroes/frees the seed payload.  Idempotent. */
SeedStatus seed_capability_revoke_on_first_commit(SeedCapability *cap);

/* Production must not silently ignore legacy scalar knobs (spec §3.1). */
SeedStatus seed_capability_reject_obsolete_options(void);

const char *seed_capability_state_name(SeedCapabilityState s);

/* One line, printed once at teardown, mirroring the A2-05..12 counter style. */
void seed_capability_report(const SeedCapability *cap);

#endif
