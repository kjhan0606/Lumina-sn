# 형광/재분배 전선 — post-EPAY 설계 (2026-07-05)

## 출발 상태 (epay7/10 챔피언, 커밋 afd9493)

- plasma: T_e %RMS 20.4%, 전셸 0.79-1.56×CMFGEN, 에너지 장부 닫힘
- formal 스펙트럼: corr 0.663, **UV(2500-3500Å) 53.5% vs CMFGEN 23.3%** — 마지막 대형 격차
- UV 과잉의 기원 (진단 완결): s0-8 방출 97% + T_inner(10020K) 백라이트 피크 2892Å.
  장부 정상 → unpaid 아님. **내부 UV가 광학으로 재분배(형광)되지 않고 탈출하는 문제.**
- CMFGEN의 답: IGE forest가 UV를 흡수 → macro-atom 캐스케이드가 광학으로 재방출
  (주파수-특정 UV→광학 형광; NOBLANK/thermal-all 실험 계보로 양성증거 확립, 2026-06-19)

## 기존 기계 (조사 2026-07-05)

| 기계 | 위치 | 상태 |
|---|---|---|
| MC macroatom+kpkt (형광 완비) | cuda.cu THEN_MC 경로 | 가동; frozen pops로 분기 재구축 |
| MC downbranch (1-step 형광) | lumina_main.c LINE_DOWNBRANCH | 가동 |
| FLUOR-ORACLE (S_l 창 부스터) | plasma.c:9728 | 진단용 (DDC15 시대) |
| CMF line-resolved J̄ producer | cmfgen_fine_jbar (사다리 23/41) | 검증완료; **consumer(II-2) 미구현** |
| selftest 판정 | cmf_selftest:2650 | "binned J̄로는 b_2≈1=형광 없음; line-resolved J̄ 필요" |

## 아키텍처 후보

**A. 하이브리드 파이프라인 (기록된 project_spectrum_pipeline_fluorescence 그대로):**
결정론(EPAY) plasma 수렴 → THEN_MC=1 → frozen pops 위 MC macroatom emergent.
형광은 macroatom이 자동 제공. 리스크: T_inner 컨트롤러 표류 기록(blue-tilt), MC 노이즈.
**falsifier = epay11 (실행중): 챔피언 plasma 위 MC 스펙트럼의 UV 분율/corr.**

**B. 결정론 완전판 (multi-week):** CMF line-resolved J̄ consumer(II-2) → UV 공진 펌핑
b_u>1 → NLTE 캐스케이드 line_source_S → formal. orthodox하지만 사다리 18단계 잔여.

**C. 중간: binned downbranch 재분배 in formal (EPAY-양식):** forest 산란몫의 일부를
흡수-재분배 채널로 — 셸별 형광 스펙트럼 F_s(ν)를 macroatom 전이표에서 구축, 지불
전력을 F_s로 재방출. EPAY w(ν) 패턴의 확장. 리스크: complete-redistribution 근사.

## 결정 게이트

epay11 판정: (i) UV 53.5→얼마? (ii) corr? (iii) 광학 밴드(green 7.7%→21.9% 목표) —
**A가 CMFGEN급이면 파이프라인 확정(과학목표 즉시 접근), 아니면 결함 분해 후 B/C.**
판정 주의: MC csv 직접 판독(THEN_MC=1이므로 유효), T_inner 표류 각주 확인.

## Acceptance (P-Cygni 설계기록 계승, toy06 기준으로 번역)

UV≤30%, 광학 밴드 CMFGEN ±30%, corr≥0.75, plasma 무회귀(THEN_MC는 plasma 동결이라 자동).
