# 판독 — A210-Z2A R-III / R-IV (운전석 계측, 2026-08-28)

계약 = `docs/RUNG_A210_Z2A_DIAG_ISOLATION_2026-08-26.md.rev1`. 판독자 = 운전석.
★판정 아님 — 계약 판정식 적용은 판정자 몫. 아래는 실측과 명령뿐이다.

## 0. 런 착지

| | R-III | R-IV |
|---|---|---|
| jobid / state | 325913 COMPLETED 0:0 04:28:51 syn102 | 325914 COMPLETED 0:0 04:28:12 syn102 |
| runroot | `…/a210_z2a_diag_isolation_a100x2_k36/riii_20260827T233659Z_ion2` | `…/riv_20260827T233659Z_ion3` |
| `model.rc` | 0 | 0 |
| target_ion (env) | 2 (`ion_label=3`) | 3 (`ion_label=4`) |
| probe report | `status=PASS branch=ZI-A blocked_count=0` | `status=PASS branch=ZI-A blocked_count=0` |
| hazard_rows | 364,488 (Z26 66,714 / Z27 265,084 / Z28 32,690) | **0** |
| stage4_rows | 5,375 | 594 |
| `line_262210_present` | true | — |
| cap/clamp/floor/jitter/repair | 전건 0 · `physical_values_modified=false` | 동일 |

두 런의 env 실질 차이 = `LUMINA_A210_LINE_SATURATION_TARGET_ION` 2→3 **하나**
(나머지 차이는 각자 런루트를 가리키는 경로 8건). `diff` 로 확인.

## 1. G5 (R-III) — 집행 결과

명령: `python3 -B scripts/a210_sproducer_ledger_check.py --te-k 10020 --output <out> <RIII>/stderr.log`

| 등록 요구 | 실측 |
|---|---|
| 완주 | `model.rc=0` |
| SUMMARY 2줄 (target_ion=2) | 2줄 (stderr 행 840·190088) |
| Stage-4 row(producer 필드) ≥1 | ROW 마커 5,375 · 원장 `row_count=5264` · producer 필드 실존(`producer_continuum_term`·`producer_local_emission_term`·`producer_terms_defined`) |
| **S2** (`Jbar == continuum+local_emission`, tol 4e-16) | **`pass=true`, `max_relative_deviation=0.0`, `zero_sum_rows=0`, `producer_terms_undefined_rows=0`** |
| ion=2 hazard census 보고 | 364,488 (위 표) |

★**도구의 종합 `verdict=FAIL` 은 S2 가 아니라 S2b 가 낸 것**:
`S2b: pass=false, max_relative_deviation=0.4236527496808573, missing_j_cont_rows=389, tolerance=1e-12`.
`S3: evaluated_rows=4875, excluded_one_minus_beta_zero=389`.
S2b 는 G5 등록 판정식이 아니다. (대장 기재: S2b FAIL → DET-BETAPROD 미구현.)
★대장 캐비앳 승계: S2 의 bit-항등은 **구성상 참**일 수 있어 물리 검증이 아니다.

**NC-G5 (음성대조) — 시연 성립**: 스크래치 사본의 ROW 3행에 `producer_terms_defined=1→0` 주입
→ `verdict=UNDECIDABLE_PRODUCER_TERMS_MISSING`, **보고서에 `S2` 키 자체가 부재**(S2 계산 전
fail-closed). 성한 입력은 위와 같이 S2 pass.

## 2. G6 (R-IV) — 집행 결과

명령: `python3 -B scripts/byte_parity_compare.py --value-file …(계약 §4 G6 목록 9건) --output <out> <L6> <RIV>`
좌(봉인 L6) = `/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe`
⚠**이 L6 경로는 운전석이 탐색으로 고른 것**이다 — 계약 문면에 절대경로 등록이 없어 grep 0건.
판정자는 이 선택의 정당성부터 검사하라.

| 파일 | L 크기 | R 크기 | 판정 |
|---|---|---|---|
| `work/lumina_spectrum.csv` | 30,637 | 30,637 | **IDENTICAL** (sha256 일치) |
| `work/lumina_spectrum_comoving.csv` | 30,637 | 30,637 | **IDENTICAL** |
| `work/lumina_plasma_state.csv` | 1,457 | 1,457 | **IDENTICAL** |
| `work/physics_DET_iter0000/{manifest.json,shell.csv,spectral.csv}` | — | — | **양쪽 부재** |
| `work/physics_DET_iter0001/{manifest.json,shell.csv,spectral.csv}` | — | — | **양쪽 부재** |

⟹ 도구 `verdict=FAIL_CLOSED`, `compared_files=3/9`. 양쪽 `work/` 실물은
`lumina_plasma_state.csv`·`lumina_spectrum_comoving.csv`·`lumina_spectrum.csv`·`physics_comparison`
넷뿐이다. **선언 6건이 두 런 모두에 없다 — 대칭적 부재.**

SUMMARY 토큰: `zero_opacity_emitting_rows=0` ∧ `sprobe_undefined_rows=0` (양 반복).

**NC-G6 (음성대조) — 시연 성립**: 스크래치 사본 3파일. 무변조 → `PASS`. `lumina_plasma_state.csv`
offset 728 `'+'→'9'` **정확히 1바이트** 변조 → `FAIL`, 불일치 파일을 정확히 지목.
원본 두 개는 대조 후에도 서로 byte-동일(무접촉 확인).

## 3. 반복 0→1 물질 응답 (관측 — 게이트 아님)

`te_lane=FIXED_T`, `pinned_shells=50`, 두 반복 모두 shell T_e = 10,020 K.
반복 경계: `[R7][PHASE] lane=DET iter=0`(행 432) / `iter=1`(행 189680),
`te_generation=1->2` / `2->3`.

**전역 `LINE-ION-OWNER-SUMMARY` (셸 0-3)**

| shell | n_e Δ | signed_rate iter0 → iter1 | Δ |
|---|---|---|---|
| 0 | +0.196% | 3.46702707782643841628 → 3.16092073845693864167 | −8.829% |
| 1 | +0.196% | 2.71213932543303387775 → 2.45896948195986000279 | −9.335% |
| 2 | +0.196% | 2.118102202294158019 → 1.9084453048639471762 | −9.898% |
| 3 | +0.196% | 1.65410794745966632476 → 1.47970314773787959683 | −10.544% |

`signed_grouping_delta` 최대 **2.03e-16** (독립 계산 순서 두 개의 일치 = 잡음 바닥).

★**이 전역 잣대는 `TARGET_ION` 을 보지 않는다** — R-III·R-IV 두 팔에서 위 여섯 필드가
**문자열까지 동일**(`T_e_K`·`n_e_cm3`·`line_order_signed_rate`·`line_order_absolute_sum`·
`ion_records`·`eligible_cells`). ⟹ 위 기울기는 **측정 하나이며 R-IV 가 확증하지 않는다.**
⟹ 부수 소득: 4.5시간 GPU 런 두 개가 21자릿수까지 같은 전역 상태 = **결정론 확인**.
⟹ ★운전석 대조축 설계 오류 1건 (「이온을 바꿔도 기울기가 유지되는가」를 이온을 안 보는
잣대에 걸었다). 기재한다.

**이온을 보는 잣대 — A2-10 per-ion (iter0 → iter1)**

| | ion III (R-III) | ion IV (R-IV) |
|---|---|---|
| candidate_rows | 624,860 → 624,860 | 211,887 → 211,887 |
| selected_rows | 2,955 → 2,420 (**−18.1%**) | 297 → 297 (**변화 0**) |
| total_scaled_emission | 53.2279335647573468536 → 52.8790664054608953656 (−0.655%) | 1.44174e-07 → 1.43942e-07 (−0.16%) |
| Z별 selected | Z26 993→847 · Z27 1870→1490 · Z28 92→83 | Z26 156→156 · Z27 20→20 · Z28 121→121 |
| Z별 emission Δ | −0.84% · −0.57% · −0.62% | −0.19% · −0.154% · −0.195% |
| zero_opacity_emitting_rows | 182,244 (반복당, 불변) | 0 |
| sprobe_undefined_rows | 182,244 (반복당, 불변) | 0 |

★`blocked_count=0` 은 `INDEPENDENT_SPROBE_UNDEFINED` **조건의 소멸이 아니라 차단하지 않음**을
뜻한다(ion III 에 반복당 182,244행 존속).
★ion IV selected 합 = 156+20+121 = **297**. 대장의 DET-A207-WIRING 이 「297선이 전부 ion IV」로
귀속한 수와 일치 — **동일 집합인지는 미확인. 판정자는 이를 사실로 받지 마라.**

## 4. 운전석이 하지 않은 것

- 계약 §4 의 선행 게이트 Z2 제3원·Z3·Z4 자체의 재집행 (이번 집행은 G5·G6 과 그 NC 뿐)
- d-proxy / J̄ 구동자 분리 분석 (A2-10 은 `interpretation=DIAGNOSTIC_ONLY`)
- 297 집합의 DET-A207-WIRING 297 과의 동일성 확인
- 봉인 L6 선택의 계약 근거 확인 (위 ⚠)
