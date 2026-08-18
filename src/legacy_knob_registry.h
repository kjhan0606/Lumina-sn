#ifndef LUMINA_LEGACY_KNOB_REGISTRY_H
#define LUMINA_LEGACY_KNOB_REGISTRY_H

/* C9 수리 — legacy env 노브의 **단일 선언처**.
 *
 * 계측·배선 부채 census C9(Fable L3 Q1-2): 같은 성격의 계약이 서로 다른 원소로
 * 3사본 존재했다.
 *
 *   src/lumina_plasma.c      population/numeric-repair lists 22종 강제(FATAL)
 *   src/lumina_element_wide.c ew_guard_config_count()       16종  관측만
 *   src/seed_capability.c    obsolete[]                      2종  강제(FATAL)
 *
 * 겹침은 5종뿐이었다.  그 결과:
 *   - **8종이 관측되는데 강제되지 않는다**  (관측 목록에만 있음)
 *   - **5종이 강제되는데 관측되지 않는다**  (강제 목록에만 있음 → 진단이 과소보고)
 *   - 어느 목록도 legacy 노브 전집을 모른다
 *
 * ★이 헤더는 **처분을 바꾸지 않는다.**  현재 각 노브가 실제로 받고 있는 처분을
 * 그대로 옮겨 적었을 뿐이다.  병합하면서 강제 범위를 넓히면 조용한 동작 변경이 되고,
 * 그것은 이 저장소에서 가장 비싼 종류의 사고다.  처분 변경은 별건으로,
 * 기대 변경집합 사전등록과 함께 한다.
 *
 * 이 헤더의 값은 목록을 합치는 데 있지 않다 — **불일치를 보이게 만드는 데** 있다.
 * `disposition` 열이 있으므로 "관측만 되는 8종"이 사고가 아니라 **결정**으로 보인다.
 * 결정이 잘못됐다면 그것은 이제 눈에 띄는 결정이다.
 */

/* 처분 */
#define LK_ENFORCE_FATAL  1   /* 설정되면 런이 죽는다 */
#define LK_OBSERVE_ONLY   2   /* 세기만 한다.  아무것도 막지 않는다 (C8 fail-open) */

/* X(env 이름, 처분, 강제 사이트, 관측 사이트)
 *
 * 강제 사이트 표기:  P=lumina_plasma.c nlte_solve_all  S=seed_capability.c
 *                    A=lumina_atomic.c:840-870 (A2-17 폐기)  -=없음
 * 관측 사이트 표기:  E=lumina_element_wide.c ew_guard_config_count  -=없음
 *
 * ⚠ 강제 P 는 `enable_nlte && iter >= nlte_start_iter` 안에서만 검사된다
 *   (lumina_main.c:656-663).  비-NLTE 런에서는 죽지도 효과도 없다 = C10 silent no-op.
 */
#define LUMINA_LEGACY_KNOBS(X)                                                 \
    /* --- 강제 + 관측 (겹침 5종) --- */                                       \
    X("LUMINA_NLTE_FORCE_LTE_LEVELS", LK_ENFORCE_FATAL, "P", "E")              \
    X("LUMINA_NLTE_LTE_REPAIR",       LK_ENFORCE_FATAL, "P", "E")              \
    X("LUMINA_NLTE_FLOOR_MODE",       LK_ENFORCE_FATAL, "P", "E")              \
    X("LUMINA_NLTE_FLOOR_REG",        LK_ENFORCE_FATAL, "P", "E")              \
    X("LUMINA_NLTE_BK_PARTIAL",       LK_ENFORCE_FATAL, "P", "E")              \
    X("LUMINA_NLTE_COLL_FLOOR",       LK_ENFORCE_FATAL, "P", "E")              \
    X("LUMINA_NLTE_LTE_FLOOR",        LK_ENFORCE_FATAL, "P", "E")              \
    X("LUMINA_NLTE_BK_CEIL",          LK_ENFORCE_FATAL, "P", "E")              \
    X("LUMINA_DR_FLOOR_CMS",          LK_ENFORCE_FATAL, "P", "E")              \
    X("LUMINA_STAGE4_BK_CAP",         LK_ENFORCE_FATAL, "P", "E")              \
    /* --- 강제만 (관측 목록에 없다 → element-wide 진단이 과소보고) --- */      \
    X("LUMINA_TOPSTAGE_THERMALIZE",   LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_NLTE_BF_JEQB",          LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_C2_MATRIX_BF",          LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_NLTE_JEQB",             LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_NLTE_FALLBACK_TE",      LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_FROZENIN",              LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_NLTE_FLOOR_BKMAX",      LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_NLTE_INV_CEIL",         LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_HRESP_CLAMP",           LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_TE_STEP_CLAMP",         LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_J_CAP_FACTOR",          LK_ENFORCE_FATAL, "P", "-")              \
    X("LUMINA_J_FLOOR_FACTOR",        LK_ENFORCE_FATAL, "P", "-")              \
    /* --- 폐기 스칼라 (A2-16; 값을 보지 않는다 — 항등값 1.0 도 FATAL) --- */   \
    X("LUMINA_TE_TRAD_RATIO",         LK_ENFORCE_FATAL, "S", "-")              \
    X("LUMINA_TRAD_COLOR_FIX",        LK_ENFORCE_FATAL, "S", "-")              \
    /* --- A2-17 폐기 스칼라 (src/lumina_atomic.c:840-870, 사이트 표기 A) ---
     * ★2026-08-07 T3 제출 실패로 발견됐다. C9 레지스트리 첫 작성이 이 **네 번째
     * 강제 사이트를 통째로 놓쳤고**, 그래서 레지스트리를 읽는 T3 프리플라이트도
     * LUMINA_CMF_EPAY_HOTF 를 통과시켰다 — 계측이 자기 구멍으로 런을 죽였다.
     * 하드 거부 env 는 12종이 아니라 **22종**이다. */                        \
    X("LUMINA_BSRC_WFLOOR",           LK_ENFORCE_FATAL, "A", "-")              \
    X("LUMINA_CMF_EPAY",              LK_ENFORCE_FATAL, "A", "-")              \
    X("LUMINA_CMF_EPAY_HOTF",         LK_ENFORCE_FATAL, "A", "-")              \
    X("LUMINA_FIXED_TRAD_PROFILE",    LK_ENFORCE_FATAL, "A", "-")              \
    X("LUMINA_F_COLL_BOOST",          LK_ENFORCE_FATAL, "A", "-")              \
    X("LUMINA_KPEMISS_BSRC_TAU",      LK_ENFORCE_FATAL, "A", "-")              \
    X("LUMINA_OUTER_TE_DAMP_FACTOR",  LK_ENFORCE_FATAL, "A", "-")              \
    X("LUMINA_OUTER_TE_DAMP_SMIN",    LK_ENFORCE_FATAL, "A", "-")              \
    X("LUMINA_VALIDATE_PLASMA",       LK_ENFORCE_FATAL, "A", "-")              \
    X("LUMINA_W_CAP",                 LK_ENFORCE_FATAL, "A", "-")              \
    /* --- 관측만. 구조/소유권 선택이며 수치 repair가 아니다. --- */            \
    X("LUMINA_NLTE_ION_LOCK",         LK_OBSERVE_ONLY,  "-", "E")              \
    X("LUMINA_NLTE_METASTABLE_COLL",  LK_OBSERVE_ONLY,  "-", "E")              \
    X("LUMINA_NLTE_PER_ION_RESCALE",  LK_OBSERVE_ONLY,  "-", "E")              \
    X("LUMINA_NLTE_STAGE4",           LK_OBSERVE_ONLY,  "-", "E")              \
    X("LUMINA_SUPER_CUTOFF",          LK_OBSERVE_ONLY,  "-", "E")              \
    X("LUMINA_TIMEDEP_ION",           LK_OBSERVE_ONLY,  "-", "E")

#endif /* LUMINA_LEGACY_KNOB_REGISTRY_H */
