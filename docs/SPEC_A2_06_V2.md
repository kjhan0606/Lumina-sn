# A2-06 구현 명세 v2 — CPU bound-bound rate: J̄ selective estimator + LineJbarCache 이관

저작·구현 운전석(개정 8) · 1차 검수 Codex BLOCK 7건(`scratchpad/a2_06_codex_review.txt`)
전부 계약화 · 기준 HEAD=d8b9870 (A2-05 폐합). V1 은 폐기 — **V1 의 "정본 4000빈 J_ν
프로파일 재적분" 노선은 상위 격자 개정 계약 위반**이었다
(`docs/ORDER_A2_GRID_AMENDMENT_BY_CODEX.md` §3·§5.2; 실측: 선폭 ±4·v_D=10km/s ≈ 빈폭의
1/9.6). 본 명세의 규범은 개정 §5.2 교체 문구 그대로다.

## 1. 단일 계약

CPU 생산 물리의 bound-bound 복사율이 `jbar_line`/`jblue_line`/셸 스칼라 (W,T_rad) 대신
**정본 generation 에 원자 결박된 `LineJbarCache` 의 checked view** 를 소비한다.
J̄ 의 물리 정의 `J̄=∫φJ_ν dν` 는 유지하되, production MC 에서는 전역 J_ν 빈 재적분이
아니라 **같은 raw path-length measure 에서 φ-가중 선택적 estimator** 로 직접 누적한다.
`R_lu=B_lu·Ĵ̄` · `R_ul^stim=B_ul·Ĵ̄` · `R_ul^sp=A_ul` 분리 유지(net 단독 비교 금지).

## 2. 1차 검수 BLOCKER 의 계약화 (B1~B7)

### B1 — 생산 노선 = selective estimator (coarse 재적분 금지)

- transport 가 세그먼트를 전역 accumulator 에 넣는 **같은 지점**에서, `Q_g` 에 속한
  (line, shell) 에 대해 φ 가중 세그먼트 적분을 누적:
  `Ĵ̄_lu,s = 1/(4π V_s Δt) Σ_seg ∫ ε φ_lu(ν'(ℓ)) dℓ` — frame·시간·체적·4π 정규화는
  전역 J_ν estimator 와 동일 계약. 세그먼트 내 ν' 변화는 등록된 세그먼트 적분 규칙
  (A2-02C `_accumulate_hist_chunk` 와 동형의 선형 경로 적분)으로, line-center point
  sample 금지.
- φ = 등록 프로파일(`A2_02C_LINE_CENSUS.json`: Gaussian, v_D=10 km/s, ±4 Doppler,
  정규화 ∫φdν=1), `profile_id`/`profile_hash` 결박. 셸 의존성 없음(공이동 v_D 상수)을
  명세 사실로 기록.
- 전역 성긴 빈 `ΣφJΔν` 를 production J̄ 로 쓰는 경로는 **작성 자체 금지**.

### B2 — LineJbarCache 완성 (A2-03 스키마의 생산자·view)

- `Q_g`: generation g 의 enabled bound-bound rate graph 가 요청하는 `BB_IN_DOMAIN`
  (line, shell) 집합. **누적 전 고정 + q_set_hash**. 선택 기준=rate graph 도달성만
  (A_ul/gf/population/직전 estimator 크기로 솎기 금지).
- accumulator: (Q_g line × shell) 별 raw 가중합 + sample_count (+variance 가능하면).
  OMP thread-local → reduce (전역 accumulator 와 동일 패턴).
- **원자 commit**: `radiation_field_commit` 확장 — 전역 J_ν 와 line cache 가 같은 raw
  ledger·같은 generation 으로 **한 commit** 에서 publish (부분 commit 금지; 둘 중
  하나 실패 = 전체 실패). validity {MEASURED(=LINE_JBAR_VALID), EXACT_ZERO,
  UNSAMPLED, OUT_OF_BB_DOMAIN, STALE}.
- **checked view**: `radiation_field_line_jbar_view(owner, expected_epoch,
  expected_n_shells, expected_generation, …)` + 조회
  `(generation, shell, line_id, profile_id)` → 값+validity+count. 실패·miss·
  UNSAMPLED active·profile mismatch = 구별 오류 (0·coarse·이전 세대 fallback 금지 —
  개정 §3.2). pure-CMFGEN replay 는 provenance `CMFGEN_REPLAY` 로 같은 view 를 생산
  가능(통계 수렴 증거로는 불인정).

### B3 — 소비지점 전수 (census 16행 + 1차 검수 전수 9지점 + 잔류 허용목록)

이관 대상 (전부 R_lu/R_ul^stim/R_ul^sp 분리 산식으로 교체):
- macro-atom 상향률: `lumina_plasma.c:4633`(jbar_line_det/jbar_line/coarse 선택),
  `:4661`(jblue_line), `:4731`(census 밖 W/T_rad B_lu−B_ul 구성)
- ETLA: `:10827`(simul) · `:12182`(RADEQ) · `:13823`(coupled RADEQ)
- NLTE rate matrix: `:15238`(dilute W/T_rad rate source) · `:15292`(jbar_line 직접) ·
  `:15361`(R_absorb/R_stim/R_spont 분기)
- census A2-06 REPLACE_SCALAR_RATE_READ 16행 (`docs/A2_01_DISPOSITION_LEDGER.md`;
  줄번호는 bafd2bb 시점) — 위 코드 지점들과 행 단위 대응표를 ADDENDUM 으로.
- `jbar_line`/`jblue_line` 의 rate-소스 지위 제거(생산·진단·비교 shadow 로 강등).
잔류 허용목록 (이관 아님, 카운터/귀속 명기):
- `lumina_cmfgen.c:3145` 부근 formal/source 소비 2곳 = A2-08(emissivity/source) 귀속
- census 진단 3행(:13920/:13940/:14080) = 정본-세대 결박 진단 파생
- GPU 경로 전부 = A2-12/13. JEQB 류 falsifier 장치 보존.
- transition 연결은 원소·이온·level label·에너지·g 로 (§6.2; 주파수 근접·index 단독
  금지). 런타임 구조체에 configuration label 부재가 확인됨(`lumina.h:373`) —
  crosswalk 는 (Z, ion, level energy, g) 4-튜플 + 로드 시 원본 라벨 해시로.

### B4 — 존재 문서만 인용

필독 = `docs/SPEC_A2_05_V2.md` · `validation/a2_05/A2_05_CLOSURE.md` ·
`docs/ORDER_A2_GRID_AMENDMENT_BY_CODEX.md` §3·§5.2 · `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md`
§7·§13. (V1 의 CODEX_IMPL_A2_05.md 인용은 오기 — 그 파일은 없다.)

### B5 — validity·fallback 계약

- UNSAMPLED active line → 그 (line, shell) rate 항 BLOCK + 카운터
  (`bb_view_blocked_{unsampled,oog,stale,profile}` / `bb_view_rate_terms`,
  `nlte_free` 종료 보고 라인 — A2-05 패턴). 값 대입(0 포함) 금지, abort 금지.
- EXACT_ZERO 는 전 구간 관측·0 확정일 때만(A2-05 6항 선례).
- STALE(세대 불일치) > UNSAMPLED > OUT_OF_BB_DOMAIN 우선순위.

### B6 — 게이트 사전등록 (개정 §3.3 정합 3종 + §3.5 음성대조 + A_ul crosswalk)

1. **동일 measure·동일 commit**: 전역 J_ν 와 Ĵ̄ 가 같은 raw ledger hash·generation —
   commit 시 hash 기록, 게이트가 대조.
2. **canonical projection closure**: canonical 빈마다 상수인 제어 프로파일 φ^(N)
   사전등록 → direct estimator vs Σ_b J_b∫_bφ^(N)dν, 한계 최대 1%·중앙값 0.2%
   (배선 시험 — 좁은 선 근사 시험 아님).
3. **fine diagnostic closure**: A2-02C 세그먼트 ledger 로 만든 fine 진단 히스토그램
   (해상도 수렴 먼저 증명) vs direct estimator, 한계 동일 1%/0.2%. 실행층: A2-02C
   고정 seed 캡처 replay (lageunha, A2-05 CHAIN 인프라 재사용).
4. **음성대조 (개정 §3.5 4종)**: (a) cache generation 을 J_ν−1 로 → 조회·commit FAIL
   (b) line ID/profile hash 교환 → 결박 검사 FAIL (c) 4π/V_s/Δt/frame 누락 fixture →
   projection closure FAIL (d) UNSAMPLED→0 치환 또는 coarse fallback 주입 → FAIL.
   각각 기대 FAIL metric·runner rc 규약은 A2-05 형식.
5. **A_ul crosswalk ≤ 1e-10** (스냅샷 무관 — 지금 판정): matching universe =
   (Z, ion, E_level, g) 4-튜플 + CMFGEN 원자자료(osc) 대조, 0-값 처리·coverage 보고,
   f_cov 분모 = truth-측(CMFGEN A_ul·통계가중 기여) — A2-05 의 truth-측 원칙.
6. **L-1bb 최종 판정 = `BLOCKED_MISSING_RATE_EXPORT`** (NETRATE/TOTRATE 부재 실측;
   `A2_00_OPHYS_PROFILE.json` 에 요건 등재 확인됨). PASS 조작 금지. wiring replay
   (CMFGEN_REPLAY provenance) 로 배선만 검증·기록.
7. MC lane CI 자격: count/variance 기반 §6.3 (CI 반폭 ≤ 한계/3) + truth-측 f_cov —
   A2-05 CHAIN 의 자격=CI∧f_cov≥0.999 형식.

### B7 — 회귀 (전판 명시)

배터리 36케이스 · L-0 replay(음성대조 포함) · **L-1bf(A2-05) 게이트+selftest** ·
A2-03/04 selftest · `make lumina` 실빌드. 신규: line-estimator selftest(합성 세그먼트
해석해 — 상수 ε·선형 ν' 세그먼트의 φ 적분 닫힌형 대조) + cache view 오류코드 전수.

## 3. 구현 순서 (운전석)

1. `Q_g` 구축(rate graph 도달성 산출 + hash) + LineJbar accumulator (thread-local
   +reduce; transport 세그먼트 훅은 전역 accumulator_add 인접 단일 지점)
2. `radiation_field_commit` dual-view 원자화 + validity + line view API + selftest
3. 소비 이관 (B3 전수) + census ADDENDUM diff
4. 게이트 스크립트(B6: 정합 3종·음성대조 4종·A_ul crosswalk·wiring replay·BLOCKED 기록)
5. 회귀 전판 → Codex diff 검수 → 커밋

## 4. 제약

src-편집 1태스크 규율 · CPU 만 · population/이온화 솔버 불변(A2-07) · 덱·/gpfs 불변 ·
로그인 노드 연산 금지(빌드 예외) · 대형 replay=lageunha. 크기 정당화: 마이크로 픽스처
(합성 세그먼트)+선별 회귀, 전량 회귀는 마디 아님.
