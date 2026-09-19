# 판독 — 오프라인 감쇠-맵 판별: CONT 초열 폭주는 A(감쇠) 불가, B(완전 선형화) 필요 (2026-09-19, 런 0회)

운전자(Claude) 판독. user 「(1)으로 가 … 오프라인 판별 돌려 … A,B 모두 구현」. 정찰(general-purpose)이 Lumina 루프(lagged)·감쇠 부재·CMFGEN 선형화를 대조; 운전자가 CONT 궤적으로 판별.

## 1. 정찰 확정분 (file:line)
- **Lumina 외부 루프 = operator-split/lagged**(`lumina_cmfgen.c` cmfgen_run 8001-8169): (a)이전 인구로 소스 조립 → (b)J 해 → (c)J 커밋 → (d)lagged J 로 인구/T_e 갱신. **∂J/∂N 항 없음**.
- **CONT 런에 이온화 감쇠 0**: SIMUL ion_damp(`lumina_plasma.c:13361`)는 `LUMINA_RADEQ_SIMUL` 경로뿐인데 CONT 는 미설정(SIMUL 마커 0). TE_STEP_CLAMP 도 off. 인구 커밋은 관성 없는 memcpy 덮어쓰기(`:22236`/`:22240`) ⟹ 구조적 과잉진동 취약. **⟹ 폭주 궤적은 「감쇠 없는 맵」이지 감쇠 실패 증거 아님.**
- CMFGEN 은 완전 선형화(`do_var_cont.f:5` dJ/dN → BA)+보정 상한(`solve_for_pops.f:132`)을 **둘 다** 씀.

## 2. 판별 (CONT 셸20 궤적, under-relaxation 정리)
committed 맵(r=f_III/f_II): iter0 r=127.2(f_II .0078) → iter1 r=231.6(f_II .0043). base(DR-off) r=8.8. CMFGEN 표적 r~0.41(f_II .71).
- **맵이 r 을 더 큰 r 로 보낸다**: M_r(127)=232 > 127. 소스 S/B 가속(pass1-2 ~0.9 → pass3-4 12.8/9.6, 초선형).
- **under-relaxation 정리(강건, 선형 가정 불요)**: `r_{n+1}=(1-w)·127+w·232=127+104w > 127` ∀w∈(0,1]. **감쇠는 늦출 뿐 M_r(r)>r 를 못 되돌린다.** 닿을 안정 고정점 없음.
- ⟹ **A(감쇠) 불가·B(완전 선형화) 필요.** 감쇠는 스텝만 재조정; 선형화는 dJ/dN 결합으로 **맵 자체**를 바꿔 일관 상태(유효 이득<1)에 착지.
- ⚠캐비앳: 2 반복 외삽. A 의 짧은 확인 런(w≈0.25, 4-6 반복)이 최종 확정(감쇠해도 r 상승 지속 ⟹ B 확정).

## 3. 수리안 (user 「A,B 모두 구현」)
- **A(감쇠)**: `lumina_plasma.c:22236`/`:22240` 인구 커밋에 `w·new+(1-w)·old`(env `LUMINA_POP_DAMP`, 기본 1.0=byte-동일). 클램프 아님(고정점 불변). 선례 `:13378`. 역할 = 판정의 최종 확인 + CMFGEN 식 강건 층.
- **B(완전 선형화, tensor-core)**: ∂J/∂N 을 인구 야코비안(BA)에 접어 J·N 결합 뉴턴 해. Lumina 이온 해는 스칼라 쌍 비(`parity_rate_se_ratio_checked` 2720-2804)라 행렬화가 아키텍처 변경. J 생산기(`cmf_exact_multigpu.cu`)와 인구 솔버가 단방향 인터페이스 — 결합 필요. **tensor-core GEMM(`lumina_nlte_gemm.cu`) 위에 BA 조립·해**.

## 4. 다음
A 계약 먼저(Fable 기안 — 값싼 확인+강건 층) → 짧은 확인 런 → B(대) 설계·계약. src 편집 1개 규칙: DR-on 385315 폐합 후.
