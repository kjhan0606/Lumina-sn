# 단 사전등록 — DET-SPROD: 생산자 선 물질 소스 직접 캡처 (2026-08-18)

계약 1개: **Sobolev 생산자가 Jbar 조립에 실제 사용한 선 물질 소스 S_producer 를
read-only 로 캡처해 A2-10 Stage-4 row 에 싣는다.** 물리 변조 0.
근거: Stage-4 판정(docs/VERDICT_A210_STAGE4_JO_2026-08-18.md) 감리가 지정한 다음 단 —
V2("B(T_e) 대용 일치")를 생산자 실측으로 확정/기각한다.

## 구성 사실 (사전 확정, src/line_net_rate.c:167-)
생산식은 `jbar = continuum_term + local_emission_term`
(continuum_term=β·J̄_cont, local_emission_term=η_per_sr·(ct/ν)·(1−β)/τ_eff).
**v2 (코드검사 1차 반영, 2026-08-18)**: S=η/χ_eff 재유도는 χ_eff=τν/(ct) 대수 가정을
끼므로 기각(검사 지적). 대신 **radiation 구조체의 두 가수를 그대로 캡처**한다 —
S2 가 순수 덧셈 항등이 되어 가정 0. S_producer 는 오프라인에서
local_emission_term/(1−β) 로 유도(S3 전용, (1−β)=0 행 제외).

## 노브
`LUMINA_A210_SPRODUCER_CAPTURE` ∈ {0,1} strict (그 외 값 = BLOCKED, fail-closed).
Sobolev 생산자에서만 유효. 기본 0 = 기존 경로 byte 불변.

## 기대 변경집합
- `src/lumina.h`: OpacityState 에 `line_producer_continuum_term`·
  `line_producer_local_emission_term`(double[NL·NS], 센티널 −1),
  `line_producer_terms_captured`·`line_producer_terms_n_shells`(int) 추가 —
  n_shells 는 stride 봉인용(소비자가 불일치 시 fail-closed; 검사 지적 반영).
  할당 전 크기 overflow 가드(검사 지적 반영)
- `src/lumina_cmfgen.c`: 리셋(4943 인접)·env 파싱(4955 인접)·Sobolev 루프 충전(6318 인접)·PASS 마커
- `src/lumina_plasma.c`: `a210_line_saturation_add` 에 s_producer 인자·row 말미
  `S_producer= producer_source_defined=` 추가(기존 필드 순서 불변)·호출부
- `src/lumina_atomic.c`: free_opacity_state 에 free 1줄
- `src/env_universe.h`: scripts/derive_env_universe.py 재생성 (손편집 금지)

## 게이트
- **S0 빌드**: `make OMP=1 lumina` + `make cuda` 두 타깃 모두 (빌드 게이트, sha·mtime 확인)
- **S1 음성대조**: (a) 노브 미설정 시 A2-10 관련 selftest 출력 byte-불변(패치 전후 대조),
  (b) 주입 결함 FAIL 시연: `=2` → BLOCKED 종료 실측
- **S2 장부 항등** (판정런): IV(target_ion=3) 1,282행에서
  `max |Jbar/(producer_continuum_term+producer_local_emission_term) − 1| ≤ 4e-16`
  (덧셈 1회 상대오차 ≤2 ulp) 사전등록. 위반 행 ≥1 ⟹ PASS 아님 —
  생산-소비 view/세대 결함 후보로 별도 단 개설.
  UNAVAILABLE(producer_terms_defined=0) 행이 1,282 중 1개라도 있으면 S2 판정 불가로 기재
  (stride 불일치도 이 경로로 외부 가시화 — 재검사 v2 지적 반영).
  **0분모 규칙(재검사 v2 지적)**: 두 가수의 합이 0인 행은 비율 대신
  `Jbar == 0` 정확 항등을 요구하고, 해당 행 수를 보고서에 명기.
- **S2b 교차검증**: `|producer_continuum_term/(β·J_cont_capture) − 1| ≤ 1e-12`
  (`β·J_cont_capture > 0` 행 한정 — 0인 행은 `producer_continuum_term == 0` 정확
  항등을 요구하고 제외 수를 명기; 재검사 v2 지적 반영) — 캡처 J_cont 와 생산
  profile 값의 bit-path 동일 주장을 실측으로 대체
- **S3 분모 T_e 확정(2026-08-18 실측)**: seed T_e 는 덱 `T_inner_K` 를 전 셸에 균일 복사한다 —
  `src/lumina_atomic.c:1060` (`opacity->t_electrons[i]=config->T_inner`) →
  `src/lumina_main.c:164` (`plasma.T_e[i]=opacity.t_electrons[i]`), 덱 `config.json:T_inner_K=10020.0`.
  ⟹ **T_e[0]=10020.0 K 정확**(로그의 `Te=10020` 은 5자리 표시일 뿐이며, ±0.5 K 모호성이
  남았다면 Wien 영역 x≈7 에서 B 에 3.6e-4 오차 = 판정 신호와 같은 크기였다).
- **S3 물리 귀속**: `[local_emission_term/(1−β)]/B(T_e=10020.0)` 분포가 기존 실측6
  (중앙 0.9992·q10 0.7131)과 일치 ⟹ q10 꼬리 = 생산자 NLTE S 의 B(T_e) 이탈로 귀속.
  불일치 ⟹ 미결 기재(튜닝 금지). (1−β)=0 또는 UNAVAILABLE 행은 S3 에서 제외하고
  제외 수를 보고서에 명기
- **S4 (=구 2번 단, III 음성대조)**: 같은 빌드로 target_ion=2 런 —
  III 에서도 S2 항등 성립 + Stage-4 비교의 안정성 표 생성

## 판정 절차 (개정13, 2026-08-18 저녁)
사전등록·코드검수·판정·감리=**Fable** / 코딩=**Codex** / 빌드·실행·게이트·대장·커밋=**운전석**.
판정과 감리는 같은 Fable 컨텍스트에서 하지 않는다(감리=fresh 컨텍스트).
※본 단의 구현·검수는 개정13 발효 전 개정12 체제(코딩=운전석·검수=Codex 3회전 승인)로 완료됐다.

## 판정기
`scripts/a210_sproducer_ledger_check.py` (S2/S2b/S3 집행, fail-closed) +
`tests/a2_10_sproducer_ledger_selftest.py` (양성 1 + 주입 결함 음성대조 6, PASS 실측).

## 런 계획
syn101 A100 수동(tripwire, 빈 pair) 또는 slurm. 기존 stage bundle 재사용 +
`LUMINA_A210_INDEPENDENT_CAPTURE=1`·`LUMINA_A210_SPRODUCER_CAPTURE=1`.
구판 대기 4건(313386/87/89/94, stage4=0·캡처 無)은 신규 제출 시점에 취소(대체 관계 명시).
