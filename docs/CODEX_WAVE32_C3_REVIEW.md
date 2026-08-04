# Codex C3 독립 소스 리뷰

## 최종 판정

- 패치 자체: rung1~3은 PASS, rung4~7은 FAIL.
- C2 차단 5건은 패치 의도상 모두 다뤄졌지만, 최종 확인 시점인 2026-08-01 03:45:52 KST의 현 소스에는 A3 누적 구현이 반영되어 있지 않습니다. 따라서 현 소스 기준 C2 차단 해소는 **0/5**입니다.
- rung7은 실제 행렬 미지수와 IV↔V 조립 자체는 대체로 올바르지만, 음수 fail-closed와 독립 보존행 감사가 성립하지 않아 전체 FAIL입니다.
- B3 산출물은 열람하지 않았습니다. 읽기 전용 조건에 따라 빌드·테스트·파일 변경은 수행하지 않았습니다.

검토 도중 03:40:55 KST에 관련 소스가 외부에서 변경되어 A3 구현이 사라졌습니다. 아래 “패치 판정”과 “현 소스 상태”는 의도적으로 분리했습니다.

| 항목 | 패치 판정 | 현 소스 |
|---|---|---|
| rung1 projection builder | **PASS** | FAIL—미적용 |
| rung2 실패 전파 | **PASS** | FAIL—미적용 |
| rung3 helper/GPU telemetry | **PASS** | FAIL—미적용 |
| rung4 실소유 카운터 | **FAIL** | FAIL |
| rung5 R7 writer | **FAIL** | FAIL—미적용 |
| rung6 within-SL OOM | **FAIL** | FAIL—미적용 |
| rung7 Fe V boundary mass | **FAIL** | FAIL—미적용 |
| clamp/floor/cap | **PASS** | 신규 값 변조 없음 |
| D6형 항등식 재발 | **FAIL** | rung7 감사 구조 결함 |

## 1. rung1 — [PASS]

패치는 투영 구축을 `nlte_build_projection()` 하나로 모읍니다. 함수 정의는 [patch:17](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung1_projection_builder.patch:17), 일반 `nlte_init()`의 실호출은 [patch:164](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung1_projection_builder.patch:164), private lane 실호출은 [patch:372](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung1_projection_builder.patch:372)입니다. 기존 private builder 본문도 삭제합니다.

`n_lines` 불일치에는 실제 assert와 오류 반환이 모두 있습니다.

> `assert(atom->n_lines == opacity->n_lines);`  
> `return -1;`

근거: [patch:33](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung1_projection_builder.patch:33).

단, 현 소스는 다시 두 builder를 갖습니다. 일반 투영은 [lumina_plasma.c:14057](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14057), private 복제는 [lumina_element_wide.c:561](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:561)에 있으며 `n_lines` assert도 없습니다.

## 2. rung2 — [PASS]

패치에서 두 `(void)nlte_element_wide_run_labeled(...)` 폐기는 모두 제거됩니다. S와 Fe 호출 각각 `<0`을 검사하고, 어느 하나라도 실패하면 상태 복원 후 `run_cell()`이 `-1`을 반환합니다. 근거: [patch:9](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung2_harness_failure_status.patch:9)-[27](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung2_harness_failure_status.patch:27).

그러나 현 소스는 두 반환값을 다시 폐기합니다: [bench_frozen_oracle.c:623](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:623)-[628](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:628).

## 3. rung3 — [PASS]

Top-stage는 공유 helper를 두 곳에서 실호출합니다.

- field 종류 선택: [patch:42](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch:42)
- bin별 `J_selected`: [patch:52](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch:52)
- estimator 선택도 `artis_parity_enabled()` 복제가 아니라 helper 반환값을 사용: [patch:62](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch:62)-[69](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch:69)

GPU bypass 카운터는 frozen 조건 밖에서, helper가 bypass를 결정한 실제 지점에서 OpenMP atomic으로 증가합니다: [patch:21](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch:21)-[34](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch:34). 값은 production manifest에 기록됩니다: [patch:86](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch:86)-[96](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung3_topstage_bf_gpu_telemetry.patch:96).

현 소스는 top-stage에서 `J_nu`와 ARTIS 조건을 직접 사용하고([lumina_plasma.c:15987](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15987)-[15999](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15999)), helper는 bypass를 반환만 할 뿐 생산 카운터를 증가시키지 않습니다([lumina_plasma.c:393](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:393)-[414](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:414)).

## 4. rung4 — [FAIL]

긍정적인 부분은 분명합니다. 세 note 함수는 가짜 hook이 아니라 실제 소유 분기에 배치되어 있습니다.

- top-stage 진입: [lumina_plasma.c:15894](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15894)-[15897](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15897)
- per-ion pin 행 교체: [lumina_plasma.c:16476](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16476)-[16494](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16494)
- 실제 save/restore: [lumina_plasma.c:17190](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17190)-[17209](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17209)

fixture도 직접 note hook을 호출하지 않고 실제 `nlte_solve_all()`로 들어갑니다: [patch:78](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung4_runtime_counter_owners.patch:78)-[99](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung4_runtime_counter_owners.patch:99).

하지만 production 카운터 증가는 plain `unsigned long++`입니다: [lumina_element_wide.c:230](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:230)-[245](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:245). `per_ion_pin`과 `topstage` 호출은 OpenMP shell 병렬 영역에서 발생하므로 경쟁 상태와 누락 증가가 가능합니다. rung3의 GPU 카운터와 달리 atomic 보호가 없습니다. 단일-shell fixture 결과가 production 누계의 정직성을 보장하지 못합니다.

또한 현 소스에는 patch가 추가하는 최종 publisher([patch:32](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung4_runtime_counter_owners.patch:32)-[61](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung4_runtime_counter_owners.patch:61))가 없습니다.

## 5. rung5 — [FAIL]

세부 판정은 다음과 같습니다.

- η 실측: PASS. 별도 snapshot과 재계산값의 max-abs 및 bitwise 비교가 실제 loop에서 수행됩니다: [patch:42](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:42)-[66](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:66). 기존 `true/0.0` 상수는 실측 변수로 대체됩니다: [patch:117](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:117)-[131](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:131).
- quarantine: PASS. payload/sidecar write·open·close 실패에서 payload를 `.quarantine`으로 rename합니다: [patch:90](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:90)-[138](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:138).
- seeded defects: PASS. bad η는 snapshot 한 원소에 실제 `+1.0`을 주입합니다([patch:257](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:257)-[266](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:266)). Matrix fixture는 ledger를 유지한 채 diagonal만 `-4→-3`으로 훼손합니다([patch:373](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:373)-[383](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:383)). 둘 다 항등형이 아닙니다.
- iter 계약 고정: **FAIL**. 코드는 `expected == wanted`만 요구합니다: [patch:192](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:192)-[205](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung5_r7_honest_writer.patch:205). 따라서 `wanted=7, expected=7`도 허용됩니다. env 파일이 10을 제시하는 것은 launcher 기본값이지 소스 계약 고정이 아닙니다.

현 소스는 η를 writer 안에서 다시 합성하고 `true/0.0`을 상수 기록합니다: [lumina_cmfgen.c:296](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:296)-[332](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:332).

## 6. rung6 — [FAIL]

checked helper 자체는 OOM에서 `-1`을 반환하고([patch:60](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung6_within_sl_oom.patch:60)-[69](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung6_within_sl_oom.patch:69)), private view는 실패 시 전체 view를 해제합니다([patch:119](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung6_within_sl_oom.patch:119)-[124](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung6_within_sl_oom.patch:124)). 직접적인 `Zsl` 누수는 없습니다.

하지만 CPU와 GPU 최상위 solve는 모두 `void`이고, OOM에서 단순 `return;` 합니다: [patch:88](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung6_within_sl_oom.patch:88)-[98](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung6_within_sl_oom.patch:98), [patch:141](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung6_within_sl_oom.patch:141)-[150](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung6_within_sl_oom.patch:150). 호출자에게 오류 상태가 전달되지 않아 “오류 반환으로 폐합”되지 않습니다. legacy wrapper도 반환값을 명시적으로 폐기합니다: [patch:80](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung6_within_sl_oom.patch:80)-[85](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung6_within_sl_oom.patch:85).

현 소스는 `malloc` 결과를 검사하지 않고 즉시 `Zsl`을 씁니다: [lumina_plasma.c:17030](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17030)-[17038](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17038).

## 7. rung7 M_V 물리 — [FAIL]

세부적으로는 상당 부분 정방향입니다.

- 실제 행렬 미지수: PASS. `N`에 scalar 하나를 추가하고 `m_index=N-1`로 둡니다: [patch:307](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:307)-[317](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:317). solve 후 별도 재정규화 통으로 채우는 코드가 아닙니다.
- 부호·위치: PASS. `EW(target,source)+=rate`, `EW(source,source)-=rate`이고([patch:158](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:158)-[168](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:168)), forward는 M 행/IV 열, reverse는 IV 행/M 열입니다: [patch:262](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:262)-[273](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:273).
- q projection: PASS. `g exp(-ΔE/kT)` 정규화 외에 조정 knob·floor·cap은 없고, 합/min/max/checksum을 기록합니다: [patch:100](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:100)-[144](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:144). q는 설계대로 reverse V→IV 분배에만 들어갑니다.
- 보존행 수식: PASS. 모든 II/III/IV/M 열을 1로 교체하고 RHS를 `n_elem-M_outside`로 둡니다: [patch:340](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:340)-[349](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:349).
- V→VI 부재: PASS. `absent_declared_truncation`으로 명시됩니다: [patch:468](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:468)-[480](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:480).
- DR 비복제: PASS. boundary assembler는 rad/coll bf만 조립하며 DR/autoion은 `not_replicated_no_producer`로 기록합니다.
- 음수/비유한 fail-closed: **FAIL**. 비유한은 차단하지만 해의 음수는 `x/n_Fe < -1e-14`일 때만 세므로 작은 음수를 승인합니다: [patch:364](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:364)-[369](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:369). C 설계의 “음수 0건” 기준과 불일치합니다.
- 독립 보존행 감사: **FAIL**. `boundary.row_residual`은 실제 `Anorm` 보존행과 `b[0]`을 다시 읽지 않고 이미 계산한 `fabs(sumx-active_mass)`를 그대로 반복합니다: [patch:376](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:376)-[383](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:383). 보존행 계수를 seed로 손상해도 이 지표는 이를 독립 검출하지 못합니다.
- flux 감사도 최종 `Araw/Anorm`이 아니라 조립 중간 `plane[c]`를 읽습니다: [patch:387](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:387)-[405](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:405). plane→최종 matrix 합성 손상은 감사 범위 밖입니다.
- boundary fixture는 q 배열과 일반 2×2 debit/target만 손상합니다. 보존행, 실제 boundary route 누락, q가 들어간 rate/ledger 결합은 seed하지 않습니다: [patch:555](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:555)-[579](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a3_rung7_fe_v_boundary_mass.patch:579).

`M_V=1.709%` 대 `1.07%`는 +0.639%p, 상대편차 약 59.7%입니다. rung7 patch에서 1.709, 1.07, 목표 M_V, boundary rate scale/fudge를 전수 검색했지만 조정 코드는 없습니다. 따라서 편차를 튜닝으로 숨긴 흔적은 **없습니다**. 다만 이 수치 자체는 사용자 제공 비교이며, B3 금지와 현 소스의 rung7 미적용 때문에 별도 실행 산출물로 재검증하지 않았습니다.

## 8. clamp/floor/cap 및 D6 — [FAIL]

- 신규 물리값 clamp/floor/cap: **PASS**. 일곱 패치에 새 population/rate 보정, M_V 목표 고정, 사후 재정규화는 없습니다. `q/Zq`는 정의에 필요한 정규화이며, `1e-12`, `1e-8`, `1e-4`는 acceptance gate이지 값을 바꾸는 clamp가 아닙니다.
- 단, `-1e-14` 음수 허용대는 값 수정은 아니지만 fail-closed 계약을 약화하는 deadband입니다.
- D6형 항등식 재발: **FAIL**. rung5의 η와 일반 matrix debit fixture는 실제 결함 주입이므로 개선됐지만, rung7의 `boundary_row_residual`은 conservation residual의 동어 반복입니다. Boundary flux도 최종 행렬 대신 같은 assembly plane을 다시 읽고, 필수 boundary-specific route/row seed가 없습니다.

## C2 차단 5건 결론

패치 원문만 보면:

1. projection builder 단일화 — 해소
2. frozen harness 실패 폐기 — 해소
3. top-stage helper 우회 — 해소
4. GPU bypass frozen-only 계측 — 해소
5. η 상수 감사·seed 부재 — 해소

하지만 최종 현 소스에는 1·2·3·4·5가 모두 다시 존재하거나 A3 구현이 없습니다. 따라서 실제 납품 상태는 **전체 FAIL**입니다.