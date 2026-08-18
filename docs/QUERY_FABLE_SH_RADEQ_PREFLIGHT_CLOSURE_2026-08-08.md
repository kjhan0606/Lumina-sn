# Fable 중요 통합 재심 — SH-RADEQ 구현 `REVISE` 폐합

역할: 물리/구조 감리자. 사소한 코드 스타일이 아니라, 직전 판정에서 flight 전 필수로
지정한 불변식 1–3과 등록 의무 4–5가 닫혔는지만 판정하라. 코드를 작성하지 말라.

## 직전 판정

session `c208bf1a-7779-4bb1-ba0a-ff86197f9849`, model `claude-fable-5`:

1. raw tau writer census + 소비 양끝 generation bracket + mutation 음성대조 필요
2. writer/reader NLTE tau authority predicate 단일화 필요
3. bulk tau/A2-09 LTE n_u 루틴 공유, LTE/NLTE branch 검사 필요
4. signed negative tau 지수 증폭 및 CMFGEN 비교 사전등록 필요
5. 707개 sub-nu_min BF edge의 SH-GRID 소비 계약 사전등록 필요

직전 마지막 줄은 `IMPLEMENTATION_VERDICT = REVISE`였다.

## 반영 증거

### 1. Raw tau slab 계약

- production writer는 `compute_tau_sobolev`, `nlte_update_tau_sobolev`,
  `apply_overlap_corrections` 3개로 census됐다.
- 각 함수는 첫 write 전 `tau_sobolev_require_refresh`, 마지막 write 후
  `tau_sobolev_mark_computed`를 자체 호출한다.
- CUDA NLTE solve의 중복 writer를 제거하고 공용 host writer를 호출한다.
- post-NLTE diagnostic의 generation 밖 NaN/Inf→0 변이를 제거했다. 이제 nonfinite가
  있으면 `[TAU-DIAG][FATAL]`로 종료한다.
- A2-09는 소비 시작과 종료에 raw required/computed, A2-08 tau, atom/A2-08
  population, plasma/A2-08 T_e, NLTE population, epoch 등식 전체를 검사한다.
- τ 값을 바꾸고 정상 writer처럼 required/computed/A2-08 tau generation을 모두 올린
  음성대조가 `EMISS_STALE_OPACITY`로 차단된다.
- static census: writers=3, CUDA writers=0. 음성대조 4/4 PASS.

### 2. NLTE authority 단일화

- writer와 reader 모두 `nlte_tau_line_authority`로 line/ion/level mapping,
  pair-owned/candidate-only, `NLTE_SKIP_Z`를 얻는다.
- shell별 element-wide commit은 `nlte_tau_line_shell_authorized` 하나만 사용한다.
- reader의 NLTE/LTE 선택은 같은 결과를 소비하는 `nlte_tau_line_uses_nlte`다.
- 정적 음성대조가 공유 predicate 제거/우회를 검출한다.

### 3. LTE/NLTE population

- `population_line_level_number_density`가 공용 routine이다.
- bulk tau writer는 LTE lower/upper 모두 이 routine을 호출한다.
- A2-09도 같은 routine에 `POP_LINE_VIEW_LTE_TE` 또는
  `POP_LINE_VIEW_NLTE_COMMITTED`를 넘긴다.
- 자가검사는 LTE branch, committed NLTE branch, 음수 NLTE population 거부를 덮는다.
- 직접 eta와 chi*S의 정상 cell closure는 `<=1e-12`다.

### 4. Signed tau 등록

- `tau=-a`에서 `beta=(exp(a)-1)/a`의 지수 성장을 명시했다.
- clamp/floor가 없고, beta/eta nonfinite는 `EMISS_DIRECT_LINE_INVALID`로 candidate 전체를
  abort한다.
- `tau=-0.25` analytic 대조와 `tau=-800` nonfinite abort 자가검사가 통과했다.
- CMFGEN의 동일 inversion 처리는 아직 증거가 없으므로 동등하다고 선언하지 않았다.
  동일 snapshot의 inverted line-cell census 및 구간별 처리 비교를 사전등록했다.

### 5. SH-GRID 등록

- 기존 dlog와 `3e16 Hz` 상한을 보존하고 1000→1178 bins,
  새 하한 `5.84127859196e13 Hz`로 확장하도록 사전등록했다.
- 707개 edge는 새 domain에서 기존 threshold classifier+CMFGEN sigma로 정상 적분한다.
- 모든 producer/consumer manifest와 GPU upload를 새 canonical edges에서 재생성하며
  padding/첫-bin 대입을 금지한다.
- 실제 census는 707개 default-active, CMFGEN sigma 707, Kramers 0,
  최저 edge `5.84852771e13 Hz`, 의도된 `REOPEN_SH_GRID rc=3`이다.
- 같은 low-frequency band의 level별 BF rate/emissivity CMFGEN 대조를 preregister했다.

## 검증

- CPU, OpenMP, full CUDA sm_80 link: rc 0
- A2-07/08/09/10 selftests PASS; A2-10 기본 계약은
  `L6=BLOCKED_INCOMPLETE_ADIABATIC`
- source static/negative controls 8/8 PASS
- writer census negative controls 4/4 PASS
- event-measure static/negative controls PASS
- Makefile header drift PASS, `git diff --check` PASS

## 판정 경계

이번 질의는 직전 `REVISE` 구현조건의 폐합만 묻는다. 완전 CMFGEN 단열항과 atomic trial
transaction은 아직 없고 모델 flight도 하지 않았다. 그러므로 구현 폐합이 ACCEPT여도
전체 SH-RADEQ flight는 `BLOCKED_INCOMPLETE_ADIABATIC`일 수 있다.

다음 두 줄로 끝내라.

```text
IMPLEMENTATION_CLOSURE = ACCEPT|REVISE
FLIGHT_STATE = BLOCKED_INCOMPLETE_ADIABATIC|READY
```
