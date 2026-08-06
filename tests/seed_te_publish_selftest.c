/* 음성대조 배터리 — lumina_publish_seed_te 가 **주입된 결함으로 FAIL 을 시연**해야
 * PASS 자격을 얻는다(캠페인 규약).
 *
 * 덱을 건드리지 않는다(덱 정본 불변) — 함수를 직접 부른다.
 * 정본 docs/RUNG_SEED_TE_PUBLICATION.md 의 게이트 G2·G4·G5.
 *
 * rc=0 전부 기대대로 · rc=1 하나라도 어긋남.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/lumina.h"

#define NS 4

static int fails;

static void check(const char *name, int got, int want) {
    int ok = (want == 0) ? (got == 0) : (got != 0);
    printf("  %-46s rc=%-3d expect=%-9s %s\n", name, got,
           want == 0 ? "accept" : "REJECT", ok ? "PASS" : "**FAIL**");
    if (!ok) fails++;
}

static void fresh(PlasmaState *p, double *te) {
    memset(p, 0, sizeof(*p));
    p->n_shells = NS;
    p->T_e = te;
    for (int i = 0; i < NS; i++) te[i] = 8000.0 + 100.0 * i;
}

int main(void) {
    PlasmaState p;
    double te[NS];

    puts("=== G-positive: 정상 seed 는 1세대로 발행된다 ===");
    fresh(&p, te);
    check("valid seed", lumina_publish_seed_te(&p, "selftest"), 0);
    printf("     T_e_generation=%llu committed=%llu manifest=%.12s\n",
           (unsigned long long)p.T_e_generation,
           (unsigned long long)p.te_publication.committed_te_generation,
           p.te_publication.te_manifest_sha256);
    if (p.T_e_generation != 1) { puts("  **FAIL** generation != 1"); fails++; }
    if (p.te_publication.committed_te_generation != 1) {
        puts("  **FAIL** committed != 1"); fails++;
    }
    if (p.te_publication.te_manifest_sha256[0] == '\0') {
        puts("  **FAIL** manifest 비어 있음"); fails++;
    }

    puts("\n=== G4: 두 번째 발행은 거부 (부트스트랩은 1회) ===");
    check("double publish", lumina_publish_seed_te(&p, "selftest"), 1);

    puts("\n=== G2: NaN 주입 → fail-closed (클램프 금지) ===");
    fresh(&p, te);
    te[2] = NAN;
    check("NaN in seed T_e", lumina_publish_seed_te(&p, "selftest"), 1);
    if (p.T_e_generation != 0) { puts("  **FAIL** 거부인데 세대가 올랐다"); fails++; }
    if (isnan(te[2]) == 0) { puts("  **FAIL** 거부인데 seed 가 수정됐다"); fails++; }

    puts("\n=== G5: 비양수·퇴화 입력 → 거부 ===");
    fresh(&p, te); te[0] = 0.0;
    check("T_e == 0", lumina_publish_seed_te(&p, "selftest"), 1);
    fresh(&p, te); te[3] = -1.0;
    check("T_e < 0", lumina_publish_seed_te(&p, "selftest"), 1);
    fresh(&p, te); te[1] = INFINITY;
    check("T_e == inf", lumina_publish_seed_te(&p, "selftest"), 1);
    fresh(&p, te); p.n_shells = 0;
    check("n_shells == 0", lumina_publish_seed_te(&p, "selftest"), 1);
    fresh(&p, te); p.T_e = NULL;
    check("T_e == NULL", lumina_publish_seed_te(&p, "selftest"), 1);
    check("plasma == NULL", lumina_publish_seed_te(NULL, "selftest"), 1);

    printf("\nSEED_TE_PUBLISH fails=%d verdict=%s\n", fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
