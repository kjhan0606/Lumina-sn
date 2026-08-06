#include "seed_capability.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *seed_capability_state_name(SeedCapabilityState s)
{
    switch (s) {
    case SEED_UNOPENED: return "SEED_UNOPENED";
    case SEED_G0_ACTIVE: return "SEED_G0_ACTIVE";
    case SEED_FIRST_FIELD_COMMITTED: return "SEED_FIRST_FIELD_COMMITTED";
    case SEED_REVOKED: return "SEED_REVOKED";
    }
    return "SEED_INVALID";
}

SeedStatus seed_capability_open(SeedCapability *cap, double epoch,
                                const char *manifest_sha256)
{
    if (!cap || !manifest_sha256 || strlen(manifest_sha256) != 64)
        return SEED_ERR_NULL;
    /* Re-opening after a commit is the seed-hold defect (N16-4). */
    if (cap->state == SEED_FIRST_FIELD_COMMITTED ||
        cap->state == SEED_REVOKED) {
        cap->hold_attempts++;
        return SEED_ERR_COMMITTED;
    }
    cap->state = SEED_G0_ACTIVE;
    cap->epoch = epoch;
    memcpy(cap->manifest_sha256, manifest_sha256, 64);
    cap->manifest_sha256[64] = '\0';
    return SEED_OK;
}

SeedStatus seed_capability_check_read(SeedCapability *cap,
                                      uint64_t required_generation,
                                      uint64_t computed_generation,
                                      double epoch,
                                      const char *manifest_sha256)
{
    if (!cap) return SEED_ERR_NULL;
    /* Order matters: the post-revocation counter must fire even when the
     * generation argument would also have failed, so the negative controls
     * can attribute the denial to exactly one cause. */
    if (cap->state == SEED_REVOKED ||
        cap->state == SEED_FIRST_FIELD_COMMITTED) {
        cap->reads_denied++;
        cap->post_g0_read_attempts++;
        return SEED_ERR_REVOKED;
    }
    if (cap->state != SEED_G0_ACTIVE) {
        cap->reads_denied++;
        return SEED_ERR_WRONG_STATE;
    }
    if (required_generation != 0 || computed_generation != 0) {
        cap->reads_denied++;
        cap->post_g0_read_attempts++;
        return SEED_ERR_GENERATION;
    }
    if (!(cap->epoch == epoch) || !manifest_sha256 ||
        strncmp(cap->manifest_sha256, manifest_sha256, 64) != 0) {
        cap->reads_denied++;
        return SEED_ERR_MANIFEST;
    }
    cap->reads_allowed++;
    return SEED_OK;
}

SeedStatus seed_capability_revoke_on_first_commit(SeedCapability *cap)
{
    if (!cap) return SEED_ERR_NULL;
    if (cap->state == SEED_REVOKED) return SEED_OK;   /* idempotent */
    cap->state = SEED_REVOKED;
    cap->revocations++;
    return SEED_OK;
}

SeedStatus seed_capability_reject_obsolete_options(void)
{
    /* SPEC §3.1: setting these in production is an obsolete-option error, not
     * a silently ignored knob. */
    static const char *obsolete[] = {"LUMINA_TE_TRAD_RATIO",
                                     "LUMINA_TRAD_COLOR_FIX"};
    for (size_t i = 0; i < sizeof(obsolete) / sizeof(obsolete[0]); ++i) {
        const char *v = getenv(obsolete[i]);
        if (v && *v) {
            fprintf(stderr,
                    "[A2-16][FATAL] %s is obsolete: the scalar seed is limited "
                    "to generation 0 and cannot be retuned at runtime\n",
                    obsolete[i]);
            return SEED_ERR_OBSOLETE_OPTION;
        }
    }
    return SEED_OK;
}

void seed_capability_report(const SeedCapability *cap)
{
    if (!cap) return;
    printf("[A2-16][SEED] state=%s reads_allowed=%llu reads_denied=%llu "
           "post_g0_read_attempts=%llu revocations=%llu hold_attempts=%llu\n",
           seed_capability_state_name(cap->state),
           (unsigned long long)cap->reads_allowed,
           (unsigned long long)cap->reads_denied,
           (unsigned long long)cap->post_g0_read_attempts,
           (unsigned long long)cap->revocations,
           (unsigned long long)cap->hold_attempts);
    fflush(stdout);
}
