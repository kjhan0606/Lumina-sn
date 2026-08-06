#include "seed_capability.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(c, l) do { if (!(c)) { \
    fprintf(stderr, "A2_16_SEED_FAIL %s line=%d\n", l, __LINE__); failures++; } } while (0)

#define H1 "1111111111111111111111111111111111111111111111111111111111111111"
#define H2 "2222222222222222222222222222222222222222222222222222222222222222"

int main(void)
{
    const double EPOCH = 1683072.0;
    SeedCapability cap; memset(&cap, 0, sizeof(cap));

    /* --- state machine happy path --- */
    CHECK(cap.state == SEED_UNOPENED, "init-state");
    /* reading before open is a wrong-state denial, not a value */
    CHECK(seed_capability_check_read(&cap, 0, 0, EPOCH, H1) ==
          SEED_ERR_WRONG_STATE, "read-before-open");
    CHECK(seed_capability_open(&cap, EPOCH, H1) == SEED_OK, "open");
    CHECK(cap.state == SEED_G0_ACTIVE, "g0-active");
    CHECK(seed_capability_check_read(&cap, 0, 0, EPOCH, H1) == SEED_OK, "g0-read");
    CHECK(cap.reads_allowed == 1, "reads-allowed");

    /* --- N16-2/3: post-G0 read of T_rad / W must fail --- */
    /* (a) generation advanced but capability not yet revoked */
    CHECK(seed_capability_check_read(&cap, 1, 1, EPOCH, H1) ==
          SEED_ERR_GENERATION, "N16-2-generation");
    CHECK(cap.post_g0_read_attempts == 1, "post-g0-counter");

    /* --- manifest binding --- */
    CHECK(seed_capability_check_read(&cap, 0, 0, EPOCH, H2) ==
          SEED_ERR_MANIFEST, "manifest-hash");
    CHECK(seed_capability_check_read(&cap, 0, 0, EPOCH + 1.0, H1) ==
          SEED_ERR_MANIFEST, "manifest-epoch");

    /* --- N16-1: revoke on first commit, then every read fails --- */
    CHECK(seed_capability_revoke_on_first_commit(&cap) == SEED_OK, "revoke");
    CHECK(cap.state == SEED_REVOKED, "revoked-state");
    CHECK(seed_capability_revoke_on_first_commit(&cap) == SEED_OK, "revoke-idem");
    CHECK(cap.revocations == 1, "revocation-count-idempotent");
    /* even a perfectly-formed generation-0 read is refused after revocation */
    CHECK(seed_capability_check_read(&cap, 0, 0, EPOCH, H1) ==
          SEED_ERR_REVOKED, "N16-1-post-revoke");
    CHECK(cap.post_g0_read_attempts == 2, "post-g0-counter-2");
    CHECK(cap.reads_allowed == 1, "no-extra-allowed");

    /* --- N16-4: seed hold (re-open after commit) must fail and be counted --- */
    CHECK(seed_capability_open(&cap, EPOCH, H1) == SEED_ERR_COMMITTED, "N16-4-hold");
    CHECK(cap.hold_attempts == 1, "hold-counter");
    CHECK(cap.state == SEED_REVOKED, "hold-does-not-reopen");

    /* --- argument contract --- */
    CHECK(seed_capability_check_read(NULL, 0, 0, EPOCH, H1) == SEED_ERR_NULL, "null-cap");
    { SeedCapability c2; memset(&c2, 0, sizeof(c2));
      CHECK(seed_capability_open(&c2, EPOCH, "short") == SEED_ERR_NULL, "short-hash"); }

    /* --- N16-5: obsolete scalar options are an error, not ignored --- */
    unsetenv("LUMINA_TE_TRAD_RATIO"); unsetenv("LUMINA_TRAD_COLOR_FIX");
    CHECK(seed_capability_reject_obsolete_options() == SEED_OK, "no-obsolete");
    setenv("LUMINA_TE_TRAD_RATIO", "0.7", 1);
    CHECK(seed_capability_reject_obsolete_options() == SEED_ERR_OBSOLETE_OPTION,
          "N16-5-ratio");
    unsetenv("LUMINA_TE_TRAD_RATIO");
    setenv("LUMINA_TRAD_COLOR_FIX", "1", 1);
    CHECK(seed_capability_reject_obsolete_options() == SEED_ERR_OBSOLETE_OPTION,
          "N16-5-colorfix");
    unsetenv("LUMINA_TRAD_COLOR_FIX");

    if (failures) {
        fprintf(stderr, "A2_16_SEED_CAPABILITY_SELFTEST FAIL failures=%d\n", failures);
        return 1;
    }
    printf("A2_16_SEED_CAPABILITY_SELFTEST PASS "
           "states=4 negatives=N16-1,2,3,4,5\n");
    return 0;
}
