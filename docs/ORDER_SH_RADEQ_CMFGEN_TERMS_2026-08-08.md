# SH-RADEQ CMFGEN 항 정의 대조 및 작업 순서 — 2026-08-08

상태: **진행 중 / 폐합 아님**. 이 문서는 solver 변경 발주서가 아니라, 현재
`RADEQ_NO_BRACKET`을 항 정의와 생산자 상태로 먼저 분해하는 사전 작업 순서다.

## 1. 지금 확정된 사실

### 1.1 기준 CMFGEN 실행이 실제로 사용한 온도 조건

- [실측] 기준 덱은 `T [FIX_T]`, `F [DO_DDT]`, `T [INC_RAD_DECAYS]`,
  `F [TRT_NON_TE]`, `T [INC_AD]`다
  (`/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/VADAT:119,622,675,679-680`).
- [실측] 덱에는 `USE_EHB`와 `COMP_EHB`가 없다. CMFGEN 기본값은 둘 다 false다
  (`new_main/mod_subs/rd_control_variables.f:979-984`).
- [실측] 온도 행렬은 `USE_EHB=true`일 때만 `BA_T_EHB/STEQ_T_EHB`를 선택하고,
  아니면 `BA_T/STEQ_T`를 선택한다
  (`new_main/subs/generate_full_matrix_v3.f:267-276`).
- [실측] 따라서 이 실행은 EHB가 아니라 적분형 복사평형을 계산했지만, `FIX_T=T`라
  그 잔차로 온도를 갱신하지 않았다. `STEQ_VALS`도 EHB를
  `not computed`로 기록한다.

결론: 현재 스냅숏은 항 정의를 조사하는 조건부 증거일 수는 있지만
`T_e` root truth는 아니다. 최종 L-6은 기존 사전등록대로 수렴한 `FIX_T=F` 실행을
기다린다.

### 1.2 CMFGEN 적분형 RE의 실제 식

- [실측] 주파수마다
  `STEQ_T += FQW * (CHI_NOSCAT*RJ - ETA_NOSCAT)`를 적분한다
  (`new_main/cmfgen_sub.f:2305-2321`).
- [실측] 선은 같은 `CHI/ETA`에 `CHIL_MAT*profile`과 `ETAL_MAT*profile`로 들어간다
  (`new_main/cmfgen_sub.f:2194-2207`).
- [실측] 선 방출 생산자는 source division이 아니라
  `ETAL_MAT = const * nu * A_ul * population_upper`다
  (`new_main/cmfgen_sub.f:3477-3489`).
- [실측] `FIX_T`와 무관하게 방사성 가열은 RE에 더해지며, RE의 역사적 단위에 맞춰
  `10^10/(4*pi)`로 스케일된다
  (`new_main/subs/eval_rad_decay_v1.f:29-38`).
- [실측] 단열항은 원자+전자 병진 에너지, 전자분율 구배, 여기·전리 내부에너지
  구배를 포함하고 `STEQ_T -= WORK`로 들어간다
  (`new_main/subs/eval_adiabatic_v3.f:114-139,219-250`).

### 1.3 EHB는 별개의 항 분해다

- [실측] CMFGEN `BFCR`은 `(nu-edge)/nu`의 초과 에너지 가중을 쓴다
  (`subs/prrr_sl_v6.f:102-130`).
- [실측] 충돌 여기·전리의 전자 에너지 교환은 `STEQ_T_EHB`에 별도로 합산된다
  (`new_main/subs/steq_multi_v10.f:186-222`).
- [실측] `COMP_EHB=T, USE_EHB=F`이면 EHB를 온도 제약으로 선택하지 않고도
  점검용으로 계산할 수 있는 구조다
  (`rd_control_variables.f:979-984`, `generate_full_matrix_v3.f:270-276`).

따라서 적분형 RE의 `chi*J-eta` 항과 EHB의 photoelectron/collisional 항을 한 장부에서
이름만 바꾸어 섞으면 안 된다. 두 표현의 등가는 통계평형·내부에너지 항까지 닫혔을 때
검증할 대상이지 선험적 가정이 아니다.

## 2. Lumina에서 확정된 첫 결함과 수리

- [실측] DET 로그 shell 0은 `line_abs=1.049004e-01`, `line_emit=0`이다
  (`validation/radeq/nbdiag.log:164-176`).
- [실측] 종전 A2-09 생산자는 unusable `line_source_validity`를 만나면
  `blocked_source++` 후 `continue`하고, 마지막에 빈 `eta_bb`를
  `EMISS_EXACT_ZERO`로 바꾸어 커밋했다.
- [판정] 이는 `docs/SPEC_A2_09_10_V1.md:249-288,693-700`의
  “source undefined와 exact zero 분리, blocked가 있으면 publish 0” 계약을 위반한다.
- [조치 1] 종전 `chi*line_source_S` 경로의 zero laundering을 fail-closed로 바꿨다.
- [조치 2] Fable 판정 뒤 생산식을
  `n_u A_ul h nu beta_esc(tau)/(4 pi dnu)` 직접형으로 교체했다.
  `line_source_S`와 `line_source_validity`는 A2-09 생산자가 더 이상 읽지 않는다.
  `tau=0, n_u>0`은 `beta=1`의 유한 방출이며 exact zero가 아니다.
- [조치 3] A2-08 publication의 population/tau/T_e/epoch generation과 A2-09 입력을
  대조하고, 실제 tau writer와 동일하게 NLTE 또는 LTE@T_e 상준위 population을 선택한다.

### 2.1 함께 확인된 line-view 소유권 부채

- [실측] `a208_publish_cpu_opacity`는 `a208_publication_init(..., n_lines=0, ...)`을
  호출한다. 주석은 약 1.25억 line-shell slab의 두 번째 복사를 피하기 위해서라고 적는다.
- [실측] 따라서 committed `CpuOpacityPublication`에는 명세에 있는
  `tau_sobolev/tau_validity/line_source_S/line_source_validity`가 없다.
- [실측] A2-09는 committed opacity generation을 확인한 뒤에도 line 항만은 다시
  가변 `OpacityState`의 네 raw 배열에서 읽는다.
- [판정] 호출 순서상 A2-08 직후 A2-09가 실행된다는 사실은 그 배열을 immutable
  generation-bound view로 만들지 않는다. 이는 `SPEC_A2_08_V2:104-131` 및
  `SPEC_A2_09_10_V1:127-146`과 맞지 않는다.
- [조치 경계] 즉시 대용량 slab 복사를 추가하지 않는다. Fable의 line-emission 식 판정에
  따라 필요한 정본 입력이 달라진다. 직접 `n_u A_ul` 식이 정본이면 source quotient slab
  자체를 A2-09 입력에서 제거할 수 있고, `beta_esc`가 필요하면 같은 generation의 tau
  또는 그 직접 계산 입력만 compact immutable view로 게시하면 된다.

## 3. 아직 맞지 않는 항 정의

| 항 | CMFGEN | 현 Lumina | 상태 |
|---|---|---|---|
| line emission | upper population과 `A_ul`의 직접 생산자 | **직접식 구현 완료**, tau=0 극한 자가검사 포함 | 실제 덱 flight 대기 |
| line input ownership | 행렬 내 동일 population/line data | raw tau writer 3개 census, 소비 양끝 generation bracket, 공유 NLTE authority/LTE population | Fable 조건부 허용 조건 충족; compact copy는 이번 rung에 강제하지 않음 |
| BF heating/cooling | RE는 전 광자에너지 `chi*J-eta`; EHB `BFCR`은 초과 에너지 | 현 생산 잔차는 signed net `chi_bf*J` 전량과 `eta_bf`를 사용 | Fable: RE가 producer, EHB는 독립 진단 |
| trial dependence | CMFGEN 행렬은 population/opacity/emissivity와 T 변화를 결합 | 현 A2-10은 committed eta에 `sqrt(T_ref/T_trial)`만 적용 | 항 정의 폐합 후 atomic trial transaction 필요 |
| adiabatic | 원자+전자 및 내부에너지/구배 | 전자 병진항은 진단값으로만 보존, status=`A210_INCOMPLETE` | fixed/free-T 모두 `RADEQ_INCOMPLETE_ADIABATIC`으로 구현 차단 |
| gamma | local deposited heating, RE 단위 환산 | 양의 체적 가열 | 단위·shell별 수치 대조 대기 |
| nonthermal | 기준 덱 OFF | exact zero | 일치 |

## 4. 순서가 있는 실행 계획

1. **SH-RADEQ-0 — zero laundering 차단**
   - 정적 빌드와 A2-09 selftest를 통과시킨다.
   - 다음 DET/MC 실행은 `line_emit=0` root까지 진행하는 대신, 실제 source 미발행
     히스토그램을 A2-09에서 fail-closed로 내야 한다.
2. **SH-RADEQ-1 — source 생산자 위치 측정**
   - 첫 block line/shell, tau/source status별 수, NLTE population generation을 기록한다.
   - 첫 반복에서 NLTE source가 아직 없다는 가설과 cancellation-singular 비중을 분리한다.
   - `OpacityState` raw line slab을 committed view라고 부르지 않는다. raw 배열의
     mutation generation과 A2-08/A2-09 호출 토큰을 함께 기록한다.
3. **SH-RADEQ-2 — Fable 판정**
   - **완료: `REVISE`**. 정본은
     `docs/FABLE_VERDICT_SH_RADEQ_2026-08-08.md`다.
4. **SH-RADEQ-3 — 선 방출 생산자 구현 — 코드 완료, flight 대기**
   - 선택된 식을 upper-level population, `A_ul`, `nu`, 동일 Sobolev operator에서 직접
     계산한다. `chi*S`와 정상 비특이 cell에서 수치 동등성을 selftest하고,
   cancellation cell은 직접식만 유한하게 남는지 음성 대조한다.
   - 선택된 식에 필요한 최소 line 입력만 immutable generation-bound view로 만들고,
     A2-09의 raw `OpacityState.line_source_*` read를 0으로 만든다.
5. **SH-RADEQ-4 — 방정식별 장부 분리 — schema 완료**
   - `RE_INTEGRAL`과 `EHB_THERMAL`을 provenance/status에서 구별한다. 온도 publication은
     `RE_INTEGRAL`만 허용하고 EHB ledger를 producer로 주입하면 거부한다.
   - CMFGEN `COMP_EHB=T, USE_EHB=F` 진단 복제 실행을 운전석이 별도 경로에서 수행해
     EHB 항을 얻는다. `/gpfs` 정본은 수정하지 않는다.
6. **SH-RADEQ-5 — 완전 단열항 + trial transaction**
   - trial `T_e -> population/n_e -> opacity/emissivity -> residual`을 비공개 버퍼에서
     전량 재평가한다. 모든 shell 성공 전 public generation은 0개다.
7. **SH-RADEQ-6 — root와 두 팔**
   - 같은 방정식 schema, atomic/population generation, J/line view를 사용한 DET/MC를
     각각 실행한다. `NO_BRACKET`은 1-5가 폐합된 뒤에만 물리적 no-root 후보가 된다.

## 5. 합격 조건

- undefined/missing/cancellation이 `EXACT_ZERO`로 바뀐 횟수 0.
- A2-09 blocked source가 있으면 emissivity commit 0, A2-10 trial 0.
- 정상 line cell에서 직접 `n_u A_ul` 식과 `chi_eff*S` 식의 상대 closure `<=1e-12`.
- RE와 EHB 장부 provenance가 서로 다르고 혼합 합산 0.
- trial마다 population/opacity/emissivity generation이 같은 token에 묶임.
- 최종 물리 gate는 `FIX_T=F`, `temperature_solved=true`, normalized heat residual
  `<=1e-3`인 CMFGEN truth가 오기 전까지 계속 BLOCKED.

## 6. 2026-08-08 로컬 검증 (Fable 판정 반영 후)

이 검증은 로그인 노드의 정적·빌드 확인이며 DET/MC 모델 실행 또는 물리 폐합이 아니다.

- 전체 CPU 소스를 별도 `/tmp/lumina_sh_radeq_check`로 컴파일: rc 0,
  SHA-256 `5999bb9b0fb1c6dcf4a53cb93c77e2180a9b0e42cd12985ead70b8d1514587e5`.
  기존 경고는 남았지만 이번 변경의 컴파일/링크 오류는 없었다.
- 같은 소스를 `-fopenmp`로 별도 `/tmp/lumina_sh_radeq_check_omp`에 재컴파일: rc 0,
  SHA-256 `6c45c87bcc21115203efd999f6b8675cc8c22aeff720c62bbc72dabe1c90c565`.
- `make selftest-sh-radeq-source`:
  - `[SH-RADEQ-0][STATIC][PASS]`
  - `[SH-RADEQ-0][NEGATIVE-CONTROL][PASS] injections=5 detected=5`
- 부정대조 8개는 차단 조건/return, 직접식 호출, 양끝 generation bracket,
  공유 NLTE authority/LTE population, `tau=0 -> beta=1` 극한을 각각 훼손하며
  작업트리는 바꾸지 않는다.
- `make selftest-tau-writer-census`: production writer 3개, CUDA writer 0개,
  require/mark 및 미등록 writer 음성대조 4/4 PASS.
- `make selftest_a2_09_emissivity`: rc 0. tau=0, 작은 양의 tau, 음의 tau,
  `n_u=0`, 음수 population을 독립 검사했다.
- `make selftest_a2_10_radeq`: rc 0. `RE_INTEGRAL` producer만 publication 가능,
  EHB producer 주입 거부, 불완전 단열항 publication 거부를 검사했다.
- Makefile header drift: `declared=22 included=22 missing=0 stale=0 verdict=PASS`.
- `git diff --check`: PASS.

따라서 SH-RADEQ-3의 직접 선 방출과 SH-RADEQ-4의 식 provenance는 코드/자가검사 단계가
완료됐다. 실제 DET/MC flight와 완전 CMFGEN 단열항은 남았다. 현 생산 solve는
`RADEQ_INCOMPLETE_ADIABATIC`으로 멈추며 T_e 세대를 발행하지 않는다.

## 7. Fable 판정 이후 구현 경계

`docs/FABLE_VERDICT_SH_RADEQ_2026-08-08.md`의 판정과 정정이 도착했다.

1. 온도 producer는 `RE_INTEGRAL`; `EHB_THERMAL`은 independent diagnostic이다.
2. 선 방출은 `n_u A_ul h nu beta_esc/(4 pi dnu)` 직접식이다. `tau -> 0`이면
   `beta_esc -> 1`이며 `n_u>0` 방출을 exact zero로 만들지 않는다.
3. `n_u`, `A_ul`, 동일 signed Sobolev `tau`는 같은 immutable generation-bound view다.
4. `FIX_T=T`도 `INC_AD=T`이면 diagnostic `STEQ_T`에 단열항이 포함된다. 따라서 fixed-T
   대조와 free-T solve 모두 현 전자 병진항만으로는 PASS 자격이 없다.

이 문서는 아직 단 폐합 승인이 아니라 `REVISE` 작업서다. SH-RADEQ-3/4 코드 구현은
끝났지만, SH-RADEQ-5의 완전 단열항·trial transaction과 운전석 flight가 남았다.

## 8. 구현 재심 `REVISE` 반영

정본 응답은 `docs/FABLE_REVIEW_SH_RADEQ_IMPLEMENTATION_2026-08-08.md`다.

- raw τ writer를 `compute_tau_sobolev`, `nlte_update_tau_sobolev`,
  `apply_overlap_corrections` 3개로 폐합했다. CUDA solve는 공용 host writer를 호출한다.
- A2-09는 raw τ 소비 전후 동일 generation view를 검사한다. 소비 중 τ와 세 토큰을
  함께 바꾸는 음성대조도 candidate publication을 차단한다.
- writer/reader는 `nlte_tau_line_authority`와 shell 권한 함수를 공유한다.
- bulk τ와 A2-09 LTE branch는 `population_line_level_number_density`를 공유하고,
  LTE/NLTE 두 branch를 자가검사한다.
- signed τ의 지수 증폭과 nonfinite 오류면은
  `docs/SIGNED_TAU_EMISSIVITY_REGISTER_2026-08-08.md`에 등록했다.
- 707개 sub-min BF edge의 SH-GRID 소비 계약은
  `docs/SH_GRID_REOPEN_CONTRACT_2026-08-08.md`에 등록했다.

이 반영은 Fable의 flight 전 필수 1–3과 등록 의무 4–5를 대상으로 한다. 완전 CMFGEN
단열항과 trial transaction은 여전히 다음 구조 단계이며, 이들이 없으므로 현 flight는
기본 경로에서 `RADEQ_INCOMPLETE_ADIABATIC`으로 멈춰야 한다.
