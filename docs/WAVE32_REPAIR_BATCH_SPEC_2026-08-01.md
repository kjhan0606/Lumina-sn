# Wave-3.2 수리 배치 명세 (운전석 작성, 2026-08-01 새벽)

근거: docs/WAVE3_TRIAD_COMPARISON_2026-07-31.md §5-7 (Opus 감사 preflight 판정 + 수리 목록). 파일 소유권: **`src/lumina_plasma.c` + `src/lumina_element_wide.c`(+Makefile)** — Stage 3.1(greenfield)과 병행 규칙은 docs/STAGE3_CMF_FIELD_CHARTER_2026-08-01.md §1.
공정: Codex A 구현 → B 테스트 / C 독립 리뷰(상호 열람 금지) → 운전석 통합. 신규 모델 런 0 — 전 판정은 오프라인 oracle 리플레이.

## R1 [필수·차단기] D1 — shadow/off-target 불변식 복원

- 결함: EW 무장 시 33-슬롯 레이아웃이 Fe IV(슬롯6)/S IV(슬롯9)를 line map에 편입시키나 어떤 레인도 그 인구를 쓰지 않아, `nlte_update_tau_sobolev`가 4336개 Fe IV 선의 τ를 전 셸 1e-100으로 덮음(`lumina_plasma.c:16894-16908`). 스펙 §6.2.2/§6.2.5 위반.
- **수리 계약(기전 아닌 불변식으로 명세)**: ①무장+COMMIT=0 → 전 모델 산출(τ·인구·pair 해)이 비무장과 **byte-동일** ②COMMIT=1 → 파일럿 (원소,셸)의 산출만 변경, 그 외 전부 byte-동일.
- 금지: 영-인구 τ-스킵 같은 증상 가드(clamp 규율 — 마스킹층 신설 금지). 정공법 방향: 권위 레인이 실제로 풀지 않는 (이온,셸)은 line map/τ 오버라이드 대상에서 제외(SKIP_Z의 `skip_tau` 기전 유사 구조 허용).
- B 판정: armed(COMMIT=0) vs unarmed 오프라인 리플레이 전 산출 byte-diff = 0 (s0·s8).

## R2 [falsifier 분기] D8 — super_mode/레이아웃 강제 해제

- 결함: 무장이 `super_mode`를 전 원소·전 셸에 강제(`lumina_plasma.c:14087-14090`) — 리플레이 내 pair 기준선≠생산 기준선 위험.
- 분기(docs/CODEX_D8_PAIR_BASELINE_FALSIFIER_2026-07-31.md 착지 대기):
  - falsifier "armed=unarmed byte-ident" → R1 계약에 이미 포섭되므로 별도 수리 불요(라이브 우려로만 원장 유지).
  - falsifier "차이 실재" → pair 레인 산술이 무장과 무관하게 생산 기본값을 따르도록 분리 + **B2 improvement 분모 재산정**(오염 dex 보고).
- 어느 분기든 R1의 byte-불변식 테스트가 최종 심판.

## R3 D4 — EW bf 장 소스를 기준선과 동일 조건부로

- 결함: EW는 `bf_rate_estimator`를 무조건 소비(`lumina_element_wide.c:324-333`), 기준선은 `(artis_parity||C2_MATRIX_BF)&&!BF_JEQB` 조건부(`lumina_plasma.c:15628-15642`) — 비-parity 환경에서 두 레인이 다른 장을 봄, `BF_JEQB` falsifier가 EW에서 무음.
- 수리: 조건 판정을 **공유 헬퍼 1개**로 추출해 양 레인이 동일 호출(복붙 금지 — 조건 분기 재발 방지). Γ 삼중대조가 A=B로 현 산술을 인증했으므로 parity 환경 산출은 불변이어야 함.
- B 판정: parity 환경 오프라인 리플레이에서 수리 전후 EW provenance byte-동일 + `BF_JEQB=1` 설정 시 EW rad_ion 경로가 pref·J로 전환됨을 카운터로 실증.

## R4 단방향 DR IV→III — 세부균형 감사 후 처분

- 결함: EW 레인이 `R_dr`를 IV바닥→III바닥으로만 주입(`lumina_plasma.c:16051-16055`), 역 autoionization 항 부재. 16-pair 레인에는 없던 신규 비대칭. 부수: DR floor(`LUMINA_DR_FLOOR_CMS`)가 `!ew_capture` 조건으로 capture 경로만 우회 — 레인 간 불일치.
- 절차(수리 전 감사 — 추정 금지): CMFGEN DIE 관례(/gpfs/kjhan/cmfgen_src/cur_cmf/ 의 DR 처리)에서 역과정 처리 실물 확인 → ①역항 필요 시 detailed-balance 짝 구현 ②CMFGEN이 net-rate 관례면 그 관례를 정확 복제 ③불명이면 UNRESOLVED로 항목 동결(원장 기재, 임의 구현 금지).
- 방향 참고: 현 신호(과이온)와 반대 방향이므로 판정 뒤집기용이 아님 — 내용 정확성 항목.

## R5 계기 정직성 — D6/D7/D3

- D6: 채널 열합 게이트가 동일 문장쌍 `+r/−r`의 항등식(공허) → 독립 누산기로 실제 조립 행렬 열합을 대조하는 실검사로 교체.
- D7: manifest의 `*_calls,0`·`hot_cold_seed` 하드코딩 리터럴 → 실카운터 배선.
- D3: σ row 부재 준위(실측 Fe II 122개)의 연속 기여 무음 삭제 + coverage 100% 허위 → **기준선(pair 레인)과 동일한 Kramers 폴백을 EW에도 적용**(내용 동등) + 폴백/삭제 카운터를 manifest에 노출. 폴백 적용에 따른 Γ 변화는 측정·기재(튜닝 판단 금지).

## R6 s0 Fe II–V 창 확장 (스펙 §1.3.2 기요구)

- 선행 실측: Fe V의 데이터 가용성(levels/σ_bf/ma_rr targets/이온화에너지) 전수 확인. 가용 시 창을 II–V로 확장(indexer·보존행·경계 게이트가 V를 포함).
- 데이터 불충분 시: V를 완전 NLTE stage 대신 **명시적 경계-질량 stage**(플럭스 폐합 기록 포함)로 처리하는 안을 설계 대안으로 제출하고 운전석 검수 대기(무단 구현 금지).
- **사전등록 기대(재정규화 실측 기반, 드라마화 금지)**: s0 Fe IV 과충전 1.0111→~1.000, 성분별 d_k(elem)<d_k(pair) 전항(II/III/IV) 성립. s8은 유의 변화 없음 기대(진범=장 내용 — 기대와 다르면 그 자체가 신규 신호로 기재).

## R7 [조건부] Stage 3.1 입력 덤프 확장

- docs/CODEX_STAGE31_CMF_DESIGN_SPEC(착지 대기)이 동결 χ,η 복원 불가 갭을 보고하면, 최소 덤프 확장(χ,η per-bin writer)을 본 배치에 계기 항목으로 편입. 착지 전에는 자리만.

## 배치 공통 acceptance

1. R1 byte-불변식 2종 PASS (armed/COMMIT=0 vs unarmed; COMMIT=1 off-target).
2. parity 환경 회귀: 수리 전후 기존 산출(EW provenance·pair 해) 변경은 **의도된 항목(R5 폴백·R6 창)만** — 각 변화는 카운터·dex로 계량 보고.
3. 신규 floor/cap/clamp 0 (발견 시 C가 FAIL).
4. s0 재측정(II–V): 사전등록 기대 대조표. s8 재측정: 무변화 확인(변하면 기재).
5. C 독립 리뷰: diff 전체 + R4 감사 결론의 CMFGEN 원전 인용 검증.
6. **음성 대조 의무(08-01 user 문답 후 신설, B 단계 집행)**: R5로 수리·신설되는 모든 게이트/카운터는 **주입 결함(seeded defect)으로 FAIL을 시연**해야 PASS 자격 획득 — 시연 없는 게이트의 PASS는 무효. (형식적 검사 재발 방지 — 사례 19-후속 envcheck 규율의 전면 일반화. D6이 바로 "FAIL 시연 불가능한 게이트"의 실물이었다.)

## 발주 시점

- R1·R3·R4·R5는 즉시 발주 가능. R2는 falsifier 착지 후 분기 확정. R6은 선행 실측 포함 즉시. R7은 Stage 3.1 설계 착지 후.
- 실행: falsifier(bbbduq038) 착지 → R2 분기 확정 → **단일 Codex A 발주(전 항목 한 배치)** → B/C 병렬 → 운전석 통합 → 오프라인 재측정. Stage 3.1 구현은 본 배치가 B/C로 넘어간 뒤 병행 개시(차터 §3).
