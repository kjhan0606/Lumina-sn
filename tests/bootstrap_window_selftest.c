/* L1-1 음성대조 배터리 — 부트스트랩 창이 **주입된 결함으로 FAIL 을 시연**해야 PASS 자격.
 *
 * 사전등록 docs/RUNG_L1_1_BOOTSTRAP_SUPPLIER.md 의 게이트 중 창 자체에 걸리는 것:
 *   G3 창이 닫히면 공급하지 않는다 (반복 >=1 fail-closed 의 구성적 근거)
 *   G5 재진입 거부 (BOOTSTRAP_REENTRY) — 런당 1회
 *
 * ★Fable G 보강: 창을 끄는 **노브를 만들지 않는다**.  따라서 G2(공급자 강제 실패)는
 * env 가 아니라 이 하네스처럼 **API 를 직접 부르는 테스트 전용 경로**로만 시연한다.
 *
 * 덱을 건드리지 않는다(덱 정본 불변).  rc=0 전부 기대대로 · rc=1 하나라도 어긋남.
 */
#include <stdio.h>

#include "../src/lumina.h"

static int fails;

static void check(const char *name, int got, int want) {
    int ok = (got == want);
    printf("  %-52s got=%-3d want=%-3d %s\n", name, got, want, ok ? "PASS" : "**FAIL**");
    if (!ok) fails++;
}

int main(void) {
    puts("=== 초기 상태: 창은 닫혀 있다 ===");
    check("lumina_bootstrap_active() before open", lumina_bootstrap_active(), 0);

    puts("\n=== 열기 ===");
    check("window_open rc", lumina_bootstrap_window_open("selftest"), 0);
    check("active while open", lumina_bootstrap_active(), 1);

    puts("\n=== G5 음성: 재진입은 거부 (런당 1회) ===");
    check("second open rc (BOOTSTRAP_REENTRY)",
          lumina_bootstrap_window_open("selftest-again") != 0, 1);

    puts("\n=== G3 음성: 닫으면 공급하지 않는다 ===");
    lumina_bootstrap_window_close();
    check("active after close", lumina_bootstrap_active(), 0);

    puts("\n=== 닫은 뒤 다시 열 수 없다 (래치) ===");
    check("reopen rc (must be refused)",
          lumina_bootstrap_window_open("selftest-reopen") != 0, 1);
    check("still inactive", lumina_bootstrap_active(), 0);

    puts("\n=== 닫기는 멱등 ===");
    lumina_bootstrap_window_close();
    check("active after double close", lumina_bootstrap_active(), 0);

    printf("\nBOOTSTRAP_WINDOW fails=%d verdict=%s\n", fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
