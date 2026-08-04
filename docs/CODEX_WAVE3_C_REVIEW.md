정적 독립 감사 결론은 **FAIL-TOPOLOGY**, 동반 판정은 **FAIL-NUMERICS**입니다. A 보고서는 열람하지 않았으며 수정·실행·git 작업도 하지 않았습니다.

| 항목 | 판정 | 핵심 근거 |
|---|---|---|
| ① §3 행렬 계약 | **FAIL** | identity/checksum과 채널 완전성 미충족 |
| ② ARTIS 구조 등가성 | **FAIL** | 원소-wide 골격은 있으나 target별 bf/NT 구조가 다름 |
| ③ fail-closed commit | **FAIL** | commit 차단은 있으나 fallback이 실제 baseline이 아님 |
| ④ OFF byte 불변 | **PASS(정적)** | 미설정과 명시 0이 같은 무부작용 분기 |
| ⑤ §7 clamp 처분 | **FAIL** | 우회 일부는 맞지만 발화계측이 허위이고 C48/C65가 남음 |
| ⑥ 수치 위험 | **FAIL** | SVD/rcond 계약 및 scaled residual 정의 불충족 |

### ① 행렬 계약 — FAIL

부분별 판정:

- **PASS — canonical II–IV 배치:** stage별 SL 수를 누적해 `[II, III, IV]` 순서로 `N`을 구성합니다. `src/lumina_element_wide.c:852-864`.
- **FAIL — identity/checksum 검증:** `checksum_mismatch`는 `memset`으로 0이 된 뒤 실제 비교·설정되는 곳이 없습니다. 그런데 이를 PASS 조건으로 사용합니다. `src/lumina_element_wide.c:286-303`, `src/lumina_element_wide.c:906-913`. 체크섬도 level·route 일부만 해시하며 line, σ grid, collision table, ionization threshold 등 명세의 atomic identity 전체를 포함하지 않습니다. `src/lumina_element_wide.c:395-418`.
- **FAIL — 7채널 완전성:** 7개 plane 자체는 존재합니다. `src/lumina_element_wide.c:22-25`, `src/lumina.h:918-926`. 하지만 thermal collisional bf는 별도 ARTIS parity gate가 켜져야만 생성되므로 EW만 켜도 `coll_bf`가 조용히 0일 수 있습니다. `src/lumina_plasma.c:15744-15759`. 활성 channel inventory를 PASS 조건에서 검사하지 않습니다. `src/lumina_element_wide.c:958-970`.
- **FAIL — column-sum 검사의 독립성:** 검사 전에 각 diagonal을 off-diagonal 합의 음수로 강제 재작성합니다. 따라서 배치 오류를 탐지해야 할 column-sum 검사가 사실상 구성상 항상 닫히는 자기충족 검사가 됩니다. `src/lumina_element_wide.c:702-710`, `src/lumina_element_wide.c:896-925`.
- **PASS — 단일 원소보존행:** raw rate를 합친 뒤 row 0만 전부 1로 덮고 `b[0]=n_element`; 나머지 RHS는 calloc 0입니다. `src/lumina_element_wide.c:867-875`, `src/lumina_element_wide.c:899-903`.
- **PASS — EW 내부 TOPSTAGE_IV 중복 배제:** reservoir는 `!ew_capture`일 때만 적용되고, 물리 plane 캡처 뒤 조기 반환합니다. `src/lumina_plasma.c:15820-15826`, `src/lumina_plasma.c:16123-16129`.

### ② ARTIS `nltepop.cc:1165-1260` 구조 등가성 — FAIL

원소별 하나의 차원, ion별 bb/bf 조립, row 0 보존이라는 상위 골격은 ARTIS와 같습니다. Lumina는 두 pair producer를 캡처해 하나의 행렬로 합칩니다. `src/lumina_element_wide.c:882-903`; ARTIS는 `../artis-ref/nltepop.cc:1218-1260`.

하지만 실질 topology는 같지 않습니다.

- ARTIS는 각 photoionization target마다 upper identity와 `epsilon_trans`를 정하고 target별 ionization/recombination helper를 호출합니다. `../artis-ref/nltepop.cc:581-615`.
- Lumina는 lower level당 threshold와 `R_bf/R_rec`을 먼저 한 번 계산합니다. `src/lumina_plasma.c:15460-15464`, `src/lumina_plasma.c:15541-15673`. 이후 동일 rate를 CSR probability로 나눠 모든 upper target에 배치합니다. `src/lumina_element_wide.c:236-269`. 즉 upper-target energy/threshold별 rate가 아닙니다.
- ARTIS NT ionization은 lower ion의 모든 level을 순회하며 upper-ion 경로를 배치합니다. `../artis-ref/nltepop.cc:630-651`. Lumina는 lower ground에서 다음 ion ground로 한 항만 넣습니다. `src/lumina_plasma.c:16099-16119`.
- ARTIS autoion plane은 autoionization과 inverse capture를 level-resolved로 모두 포함합니다. `../artis-ref/nltepop.cc:656-709`. Lumina plane은 ground-to-ground DR만 담고 autoion은 manifest에서 inactive 처리합니다. `src/lumina_plasma.c:16023-16042`, `src/lumina_element_wide.c:984-985`.

따라서 “같은 원소-wide 외형”은 맞지만 ARTIS process topology 등가는 아닙니다.

### ③ fail-closed commit — FAIL

좋은 부분은 명확합니다.

- 모든 gate가 `pass`여야 population을 씁니다. `src/lumina_element_wide.c:958-970`, `src/lumina_element_wide.c:989-999`.
- status가 1인 대상만 legacy pair solve/save-restore/damping을 건너뜁니다. `src/lumina_plasma.c:17078-17084`, `src/lumina_plasma.c:17097-17103`, `src/lumina_plasma.c:17124-17130`.

그러나 실패 시 돌아가는 경로가 명세의 baseline이 아닙니다.

- EW가 켜지면 allow-list와 무관하게 전체 NLTE layout을 33-slot EW layout으로 교체합니다. `src/lumina_plasma.c:13969-13976`.
- pair table도 Fe/S의 III–IV pair를 포함한 EW table로 전역 교체됩니다. `src/lumina_plasma.c:8118-8132`.
- 따라서 `COMMIT=0` shadow나 EW 실패 뒤 실행되는 legacy solve는 기존 31-slot/16-pair baseline이 아니라 변경된 33-slot/18-pair solve입니다. `src/lumina_plasma.c:17030-17085`.
- 그럼에도 verdict는 `EW_FAIL_FALLBACK_BASELINE`이라고 기록합니다. `src/lumina_element_wide.c:971`.
- `ew_status`의 calloc 실패도 검사하지 않고 바로 역참조합니다. `src/lumina_plasma.c:17000-17012`.

이는 대상 밖 원소·셸 불변과 진짜 baseline fallback 양쪽을 모두 위반합니다. 통합부의 first failure는 `src/lumina_plasma.c:13969`입니다.

### ④ OFF byte 불변 적대 수색 — PASS(정적)

- master가 미설정이거나 `0`이면 같은 early-return 분기를 탑니다. banner, dump-dir 처리, allocation이 없습니다. `src/lumina_element_wide.c:41-50`.
- OFF capture hook은 첫 조건에서 즉시 반환합니다. `src/lumina_element_wide.c:191-194`.
- OFF에서는 base layout과 base pair table을 선택합니다. `src/lumina_plasma.c:13969-13976`, `src/lumina_plasma.c:8152-8159`.

따라서 정적 호출 그래프상 미설정과 명시 `0` 사이 산술·RNG·출력 부작용은 발견되지 않았습니다. 실행 금지 지시에 따라 실제 `cmp` 증명은 수행하지 않았습니다.

단, 이는 master OFF에만 해당합니다. `COMMIT=0` shadow 및 대상 밖 영역은 위 ③ 사유로 불변이 아닙니다.

### ⑤ §7 A형 clamp 처분 — FAIL

- **PASS:** C13/C14/C15/C17/C19 계열 pair 후처리와 C64 anchor는 EW 조기 반환 뒤에 있어 실행되지 않습니다. `src/lumina_plasma.c:16123-16129`, `src/lumina_plasma.c:16176-16429`, `src/lumina_plasma.c:16548-16753`.
- **FAIL:** `guard_firing_count`는 실제 계측이 아니라 상수 0입니다. 구성된 guard 수는 별도로 세지만 PASS 조건에는 사용하지 않습니다. `src/lumina_element_wide.c:713-725`, `src/lumina_element_wide.c:957-970`.
- **FAIL — C48:** `LUMINA_SUPER_CUTOFF`은 원자자료 로딩 중 level identity를 강제로 lump하지만 EW guard 목록이나 PASS counter에 포함되지 않습니다. `src/lumina_atomic.c:761-780`.
- **FAIL — C65:** EW와 `LUMINA_NLTE_STAGE4`를 함께 설정하면 downstream Gph 소비자가 여전히 stage-IV `b_k` cap을 적용할 수 있습니다. `src/lumina_plasma.c:9967-9975`, `src/lumina_plasma.c:10023-10030`. EW parser는 이 충돌 gate를 거부하지 않습니다. `src/lumina_element_wide.c:48-111`.
- **FAIL — 미계측 fallback/cap:** σ 부재 시 Kramers fallback이 그대로 허용되고 `n_star_ratio`는 `1e30`으로 cap되지만 fallback/guard counter에 반영되지 않습니다. `src/lumina_plasma.c:15488-15503`, `src/lumina_plasma.c:15640-15657`.

즉 lane-local 우회 의도는 맞지만 “발화 0을 측정해 PASS”하는 계약은 충족하지 않습니다.

### ⑥ 수치 위험 — FAIL

구현된 요소는 있습니다.

- ARTIS식 row/column equilibration: `src/lumina_element_wide.c:609-633`.
- partial-pivot LU: `src/lumina_element_wide.c:428-478`.
- 최대 10회 refinement: `src/lumina_element_wide.c:641-686`.

하지만 acceptance 수치 계약은 깨집니다.

- `kappa_2`는 명세가 요구한 SVD가 아니라 고정 32회 power/inverse iteration 추정입니다. 수렴 오차나 경계가 없습니다. `src/lumina_element_wide.c:503-549`.
- `rcond`도 solver의 독립 condition estimate가 아니라 동일 추정치의 단순 역수입니다. `src/lumina_element_wide.c:682-684`.
- scaled SE residual 분모가 `max(inflow,outflow,|x_i|)`입니다. rate flow와 population을 직접 비교해 단위가 맞지 않으며, 명세의 `n_i/t_ref`가 없습니다. 느린 rate에서는 잔차를 과소평가해 false PASS가 가능합니다. `src/lumina_element_wide.c:795-806`; 명세 `docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md:301`.
- equilibration norm의 단순 `v*v` 합은 큰 rate span에서 overflow 위험이 있습니다. `src/lumina_element_wide.c:615-622`.
- boundary active-set은 population과 Sobolev opacity만 검사합니다. IV→V drain rate/heating이 활성인데 V population과 V line opacity가 0인 경우에도 `boundary_process_coverage=1`이 될 수 있습니다. `src/lumina_element_wide.c:935-956`.

최종적으로 OFF 정적 불변과 단일 보존행/TOPSTAGE 배제는 통과하지만, topology·fallback baseline·clamp 계측·condition/residual 계약 때문에 현재 구현은 Stage 2A PASS 또는 PASS-WITH-SCOPE로 인정할 수 없습니다.