# Codex C — Wave-3.2 독립 리뷰

작성일: 2026-08-01  
범위: `src/lumina_plasma.c`, `src/lumina_element_wide.c` 현재 상태의 정적 검토와 R6 설계 재구성  
정본: `docs/WAVE32_REPAIR_BATCH_SPEC_2026-08-01.md`

## 0. 검토 규율과 총평

- `CODEX_WAVE32_B_*` 산출물은 열람하지 않았다.
- `src/`는 수정하지 않았고 테스트도 실행하지 않았다.
- 허용된 데이터 인벤토리 명령 `python3 scripts/wave32_fe5_inventory.py`만 실행했다.
- 구현자 요약은 파일 위치를 찾는 참고로만 사용했고, 아래 판정은 코드 실물로 내렸다.

| 항목 | 판정 | 요지 |
|---|---|---|
| R1 | **FAIL** | τ/source 쓰기 제한 자체는 권위 불변식 기반이지만, EW 무장이 `super_mode`와 전역 파생 레이아웃을 여전히 바꾸므로 COMMIT=0의 일반적인 byte-불변식이 소스상 성립하지 않는다. |
| R3 | **FAIL** | 공유 헬퍼는 있으나 pair 레인에 `C2_MATRIX_BF` 조건 캐시와 GPU 우회 조건이 복제되어 단일 진실원이 아니다. EW의 JEQB 배선 자체는 유효하다. |
| R4 | **UNRESOLVED** | 역 autoionization은 구현되지 않았고 단방향 DR은 유지됐다. 즉 무단 구현은 없다는 하위 판정은 PASS이나, 물리 처분은 명세대로 동결 상태다. |
| R5 | **FAIL** | D6 게이트는 구조적으로 실패 가능하지만, 일부 manifest 카운터는 실측이 아니며 Kramers 폴백도 pair 레인과 내용 동등하지 않다. |
| 신규 clamp/floor/cap | **PASS** | Wave-3.2 신설 경로에서 값을 조정해 통과시키는 신규 clamp/floor/cap은 발견하지 못했다. |
| R6 | **PASS (설계안 제출)** | Fe V 완전 stage는 데이터 부족. Fe V를 단일 경계질량 미지수로 두고 Fe IV↔V 양방향 플럭스와 폐합 원장을 명시하는 대안을 아래에 제안한다. 구현은 하지 않았다. |

따라서 이 정적 리뷰의 배치 종합 판정은 **FAIL**이다. 발견 사항의 처분은 이 문서에 기록하는 것으로 끝내며 코드 수정은 하지 않았다.

## 1. R1 — shadow/off-target 불변식

### 판정: **FAIL**

정공법 부분은 확인된다.

- `pair_owned[]`는 인구값이 아니라 현재 `nlte_get_pairs()`가 반환한 실제 pair 레인 소유 슬롯으로 만들어진다 (`src/lumina_plasma.c:16858-16867`).
- 각 line은 `candidate_only_slot = !pair_owned[ion_idx]`로 분류된다 (`src/lumina_plasma.c:16895-16903`). 후보 전용 슬롯의 셸 쓰기 권한은 `COMMIT`, 허용 (Z,shell), 성공 상태 `g_ew_tau_authority[...] == 1`을 모두 요구한다 (`src/lumina_plasma.c:16947-16954`). 인구를 읽는 것은 이 판정 뒤다 (`src/lumina_plasma.c:16956-16957`). 따라서 영-인구를 보고 τ를 건너뛰는 증상 가드는 아니다.
- EW 레이아웃에서도 pair 소유 표는 Fe/S IV 후보 슬롯을 제외한 원래 16개 물리 pair를 재배치해 반환한다 (`src/lumina_plasma.c:8159-8175`). EW 해가 실제로 인구를 쓰는 것도 `pass && commit_requested`일 때뿐이다 (`src/lumina_element_wide.c:1470-1480`).
- τ와 source는 같은 후보 권한 분기 아래에서 함께 보호된다. 권한이 없으면 둘 다 쓰기 전에 `continue`한다 (`src/lumina_plasma.c:16947-16954`); 권한이 있으면 τ는 `16966-16970`, source는 `16972-16987`에서 기록된다.

그러나 R1의 전체 byte-불변식은 소스상 포섭되지 않았다.

- EW 무장은 target table과 슬롯 수 자체를 33-slot 레이아웃으로 바꾼다 (`src/lumina_plasma.c:14014-14024`). 그 결과 level offset/total과 양방향 level map도 확대된 레이아웃에서 다시 만들어진다 (`src/lumina_plasma.c:14027-14040`, `14060-14078`).
- 더 결정적으로 `super_mode`는 사용자가 `LUMINA_SUPER_LEVELS`를 켜지 않아도 `element_wide`만으로 강제된다 (`src/lumina_plasma.c:14123-14130`). `fl_to_super`, `super_anchor_global`, `within_sl_frac`도 이 확대된 전역 레이아웃 크기로 구성된다 (`src/lumina_plasma.c:14101-14134`).

즉 이번 수리는 line τ/source의 후보 슬롯 오염은 정확히 차단했지만, `armed, COMMIT=0`이 `unarmed`와 일반적으로 같은 solve 표현을 사용한다는 불변식까지 만들지는 않았다. 특히 unarmed에서 `LUMINA_SUPER_LEVELS=0`인 구성은 armed와 `super_mode`가 달라질 수 있다. 특정 리플레이가 양쪽 모두 이미 super mode인 경우 byte-identical일 수는 있으나, 그것은 환경의 우연한 동치이지 이 코드가 보장하는 불변식이 아니다. 명세가 요구한 “super_mode/레이아웃 파생 배열까지 포섭” 기준으로 FAIL이다.

## 2. R3 — bf 장 선택의 단일 진실원과 JEQB

### 판정: **FAIL**

공유 헬퍼 자체는 올바른 모양이다. `nlte_bf_field_source()` 한 곳이 `BF_JEQB` 우선, 그다음 `(artis_parity || C2_MATRIX_BF) && bf_rate_estimator`를 판정하고 J 또는 Planck 값을 선택한다 (`src/lumina_plasma.c:370-389`). pair는 이를 `src/lumina_plasma.c:15632-15634`, `15644-15647`에서, EW는 `src/lumina_element_wide.c:308`, `358-361`에서 호출한다.

그러나 pair 레인에 같은 조건의 일부가 복제되어 있다.

- `c2mx`가 `LUMINA_C2_MATRIX_BF`를 별도로 읽고 캐시한다 (`src/lumina_plasma.c:15552-15567`).
- GPU lookup 사용 여부는 공유 헬퍼의 `bf_field_source`가 아니라 `!bf_jeqb && !c2mx`로 다시 결정된다 (`src/lumina_plasma.c:15632-15638`). bin별 R_bf 누산 여부도 `!use_gpu_R_bf || bf_jeqb || c2mx`라는 별도 조건을 쓴다 (`src/lumina_plasma.c:15655-15668`).
- 따라서 parity만 켜지고 `c2mx=0`, `use_gpu_R_bf=1`이면 공유 헬퍼는 estimator source(1)를 선언하지만 pair R_bf는 GPU lookup을 유지할 수 있다. EW는 같은 상황에서 estimator를 직접 소비한다 (`src/lumina_element_wide.c:362-371`). 단일 진실원이라는 주장과 실제 분기 제어가 일치하지 않는다.

JEQB falsifier의 EW 배선은 별도 하위 판정 **PASS**다. 헬퍼 결과를 `field_source`로 보존하고 (`src/lumina_element_wide.c:308`), source 2이면 각 bin에서 `pref*J`를 쓰면서 `bf_jeqb_bins`를 실제 증가시킨다 (`src/lumina_element_wide.c:358-375`). 이 경로에서는 estimator 분기로 들어가지 않는다. 다만 이 하위 PASS가 조건 복제와 pair/EW 불일치를 상쇄하지는 않는다.

## 3. R4 — 단방향 DR IV→III

### 판정: **UNRESOLVED**; 무단 구현 여부는 **PASS**

- 현재도 DR은 upper ground에서 lower ground로 `R_dr` 하나를 넣는 단방향 항이다 (`src/lumina_plasma.c:16075-16093`). 역 autoionization/detailed-balance 항은 이 블록에 없다.
- `LUMINA_DR_FLOOR_CMS`는 EW capture에서 적용하지 않고 pair 레인에서만 적용한다 (`src/lumina_plasma.c:16078-16086`). 이는 기존 불일치를 해결한 것이 아니라 그대로 드러낸다.
- EW는 같은 단방향 `R_dr`을 `NLTE_EW_AUTOION_DR` plane에 관측할 뿐이다 (`src/lumina_plasma.c:16088-16093`). 그 뒤 EW capture는 time-dependent/pin/topstage/repair 구간 전에 반환한다 (`src/lumina_plasma.c:16178-16184`).

따라서 reverse 항이나 net-rate 관례를 임의로 발명하지 않았다는 점은 PASS다. 반면 명세가 요구한 CMFGEN 원전 감사로 물리 관례를 확정할 근거는 두 검토 대상 소스 안에 없으므로 R4 본체는 계속 UNRESOLVED다.

## 4. R5 — 계기 정직성

### 4.1 D6 독립 조립 게이트

판정: **PASS (구조적으로 FAIL 가능)**

- 각 전이의 matrix inflow와 diagonal debit은 `src/lumina_element_wide.c:258-260`에서 기록된다.
- 별도 `expected_outflow[channel][source]` 원장은 matrix를 읽지 않고 rate를 독립 누산한다 (`src/lumina_element_wide.c:261-265`).
- 검사기는 실제 matrix의 off-diagonal 합과 diagonal을 각각 원장과 비교한다 (`src/lumina_element_wide.c:1030-1045`). 최대 residual은 topology와 numerical gate 모두에 들어간다 (`src/lumina_element_wide.c:1374-1383`, `1440-1445`).

따라서 예를 들어 diagonal `-= rate`의 부호·열·누락을 seed하면 원장 값은 유지된 채 `e_debit` 또는 `e_in`이 비영이 되어 임계값을 넘길 수 있다. 과거의 같은 문장쌍 `+r/-r`을 다시 더해 0을 확인하는 항등식은 아니다. 단, 실제 seeded-defect 실행 시연은 이번 읽기 전용 C 범위가 아니므로 주장하지 않는다.

### 4.2 D7 카운터

판정: **FAIL**

실측 카운터와 명목상 변수만 있는 카운터가 섞여 있다.

- channel event, Kramers, deletion/fallback, nstar/nonfinite, bf estimator/J/JEQB는 실제 발생 지점에서 증가한다 (`src/lumina_element_wide.c:223-233`, `265`, `286`, `320`, `367-375`, `391`, `409`, `451`). hot/cold rebuild도 실제 성공 반환 때 증가하고 두 matrix를 `memcmp`한다 (`src/lumina_element_wide.c:1300-1347`). 이들은 실측이다.
- 반면 `save_restore_calls`, `per_ion_pin_calls`, `topstage_IV_calls`는 구조체에 선언될 뿐 (`src/lumina_element_wide.c:206-214`) 전체 파일에 증가 지점이 없다. 0으로 초기화된 값을 diagnostics/manifest에 출력할 뿐이다 (`src/lumina_element_wide.c:1463`, `1467`). 이는 하드코딩 리터럴을 0-초기화 변수로 바꾼 것일 뿐 실카운터가 아니다.
- `candidate_pair_owner_calls`도 capture를 무조건 시작한 직후 `!nlte_ew_capture_active()`를 읽어 누산한다 (`src/lumina_element_wide.c:1287-1293`). 정상 제어 흐름에서는 직전 `ew_capture_begin()`이 `active=1`을 무조건 썼으므로 (`src/lumina_element_wide.c:181-199`) 구조상 항상 0이다. 실제 pair-owner 진입 지점의 계측이 아니다.

조기 반환이 해당 경로를 막는다는 정적 사실과, 호출 횟수를 실제 계측했다는 주장은 다르다. 명세의 “카운터가 실측인지” 기준으로 FAIL이다.

### 4.3 D3 Kramers 폴백 내용 동등성

판정: **FAIL**

edge σ0만은 공유되어 있다. pair와 EW가 모두 `nlte_bf_kramers_sigma0()`를 호출한다 (`src/lumina_plasma.c:6719-6729`, `15539-15543`; `src/lumina_element_wide.c:302-307`). 그러나 전체 산술 계약은 같지 않다.

1. sigma-row 가용성 조건이 다르다. pair는 CMFGEN frequency grid 수까지 NLTE grid와 같아야 row를 사용한다 (`src/lumina_plasma.c:15541-15543`, `15600-15603`). EW는 loaded/pointer/row flag만 검사하고 grid 길이 일치를 확인하지 않는다 (`src/lumina_element_wide.c:282-285`). grid mismatch에서 pair는 Kramers, EW는 table row가 된다.
2. pair의 inverse Saha factor는 `1e30`에서 cap한다 (`src/lumina_plasma.c:15695-15712`). EW는 log 값이 `DBL_MAX` 표현범위를 넘으면 route 전체를 거부하고, 그 전까지는 cap 없이 지수화한다 (`src/lumina_element_wide.c:384-395`). 내용이 다르다.
3. pair는 recombination을 upper super-level column에서 lower row로 weight 없이 넣는다 (`src/lumina_plasma.c:15753-15759`). EW는 Kramers가 선택한 upper anchor의 `within_sl_frac`을 inverse rate에 곱한다 (`src/lumina_element_wide.c:312-314`, `333-335`, `414-415`). super-level이 비자명하면 같은 rate가 아니다.
4. collisional bf는 pair에서 `parity_on` 조건부다 (`src/lumina_plasma.c:15797-15815`). EW는 같은 형태의 산술을 parity 조건 없이 수행한다 (`src/lumina_element_wide.c:396-405`).
5. R3에서 지적한 field-source GPU 우회 차이도 Kramers의 `rad_ion` 내용에 그대로 들어간다.

폴백/삭제 카운터 자체는 실제 발생 지점에 연결되어 있고 (`src/lumina_element_wide.c:285-286`, `315-321`), `target_fail`은 gate를 닫는다 (`src/lumina_element_wide.c:1430-1442`). 그러나 명세가 요구한 것은 카운터 존재뿐 아니라 기준선 pair 레인과의 내용 동등성이므로 D3는 FAIL이다.

## 5. 신규 clamp/floor/cap 전수 확인

### 판정: **PASS**

Wave-3.2 신설 코드에서 해를 조정해 gate를 통과시키는 신규 clamp, floor, cap은 발견하지 못했다.

- EW의 `nstar_cap`이라는 이름은 부정확하지만 실제 동작은 값을 상한에 고정하는 cap이 아니다. `log_nstar`가 double 표현범위를 벗어나면 해당 route를 거부하고 카운터를 올린다 (`src/lumina_element_wide.c:390-392`). 이 카운터는 `guard_firing_count`에 포함되어 topology gate를 실패시킨다 (`src/lumina_element_wide.c:1418-1432`). 즉 숨은 마스킹층은 아니다.
- Kramers는 명세 R5가 요구한 데이터 부재 폴백이며, 발화가 계수된다 (`src/lumina_element_wide.c:282-286`). 이를 신규 수치 floor로 분류하지 않았다.
- pair 레인의 기존 `n_star_ratio > 1e30` cap (`src/lumina_plasma.c:15711`)은 이번 Wave-3.2 신설물이 아니다. 다만 위 R5에서 EW와의 내용 불일치 좌표로 기록했다.
- boundary, residual, rank, condition-number 임계값은 값을 바꾸는 clamp가 아니라 fail-closed 판정식이다 (`src/lumina_element_wide.c:1427-1449`).

따라서 신규 좌표로 보고할 clamp/floor/cap 위반은 0건이다.

## 6. R6 — Fe V 데이터 실측과 명시적 경계-질량 stage 설계

### 6.1 데이터 갭 실측

`scripts/wave32_fe5_inventory.py`는 Fe ion_number 1..4의 levels, CMFGEN σ row, MA RR source/route/target, 이온화에너지를 집계하고 (`scripts/wave32_fe5_inventory.py:21-83`), Fe V의 모든 σ와 MA source가 함께 있을 때만 full-stage AVAILABLE로 판정한다 (`scripts/wave32_fe5_inventory.py:84-89`). 실행 결과는 다음과 같다.

| stage | levels | σ rows | σ coverage | ma_rr source/routes | valid routes | target ion | IP (eV) |
|---|---:|---:|---:|---:|---:|---|---:|
| Fe II | 2698 | 2576 | 95.478132% | 2576/2576 | 2576 | Fe III | 16.1877488957 |
| Fe III | 1500 | 1500 | 100% | 1500/1500 | 1500 | Fe IV | 30.6513735284 |
| Fe IV | 200 | 200 | 100% | 200/200 | 200 | Fe V | 54.8010156928 |
| Fe V | 200 | 200 | 100% | 0/0 | 0 | 없음 | 73.9723203388 |

스크립트의 최종 판정은 `FeV_full_stage_verdict=INSUFFICIENT (FeV ma_rr sources 0/200)`였다. 또한 Fe II의 σ 결손은 2698-2576 = 122개로, R5 D3의 실측과 일치한다.

결론: Fe IV→V interface를 만드는 데 필요한 Fe IV lower rows와 Fe V target identity는 200/200 존재하지만, Fe V 자체를 완전 NLTE active stage로 승격해 V→VI continuum까지 닫을 MA RR source는 0/200이다. 따라서 II–V “완전 stage” 구현은 거부하고 아래의 제한된 경계질량 stage만 설계 대상으로 삼는다.

### 6.2 제안 상태공간

각 셸에 기존 Fe II/III/IV super-level 미지수 뒤로 scalar `M_V` 하나를 추가한다.

\[
x = (n_{\mathrm{Fe\,II},1..}, n_{\mathrm{Fe\,III},1..},
     n_{\mathrm{Fe\,IV},1..}, M_V).
\]

`M_V`는 Fe V 200개 준위를 독립 방정식으로 푸는 것이 아니라 그 stage의 총 질량만 보유한다. Fe V bb, Fe V→VI bf, Fe V line τ/source 권한은 만들지 않는다. 따라서 없는 `ma_rr`를 Kramers로 대체해 “완전 Fe V”인 것처럼 가장하지 않는다.

Fe V 내부 target fraction은 새 자유 파라미터가 아니라, 읽어온 200개 Fe V level의 `(E_t,g_t)`와 현재 `T_e`로 만든 정규화 Boltzmann projection을 사용한다.

\[
q_t(T_e)=\frac{g_t\exp[-(E_t-E_0)/kT_e]}{\sum_u g_u\exp[-(E_u-E_0)/kT_e]},
\qquad \sum_t q_t=1.
\]

이 projection은 rate를 조정하는 floor/cap이 아니며, `q_t`의 합, min/max, checksum을 artifact에 기록한다. 비유한·음수·합 불일치 시 보정하지 말고 fail closed한다.

### 6.3 Fe IV↔V interface 조립

사용 가능한 Fe IV 200/200 MA route만 interface 생산자로 인정한다. 각 Fe IV lower level `l`과 Fe V target `t`, route probability `p_lt`에 대해 현재 EW와 같은 target별 threshold, σ row, 공유 radiation-field 선택, Milne inverse를 계산한다.

- forward IV→V: `M_V` row / Fe IV source column에 `p_lt * R_ion(l,t) * f_l`을 더하고 source diagonal에서 뺀다.
- reverse V→IV: Fe IV target row / `M_V` column에 `q_t * p_lt * R_rec(t,l)`을 더하고 `M_V` diagonal에서 뺀다.
- collisional ionization/3-body inverse와 nonthermal producer가 실제 존재할 때도 같은 `p_lt`, `q_t` 규칙과 별도 channel plane을 사용한다.
- DR/autoionization은 R4가 해결되기 전에는 현 단방향 관례를 이 새 IV↔V 경계에 복제하지 않는다. producer 부재로 manifest에 명시하고 commit gate를 닫는다.

핵심은 `M_V`를 단순 사후 재정규화 통으로 쓰지 않고, IV↔V 양방향 rate가 들어간 실제 matrix unknown으로 둔다는 점이다. 음수 해, 비유한 해, route 누락을 floor/cap/재정규화하지 않고 baseline으로 fail closed한다.

### 6.4 보존행과 외부 경계

한 행을 다음 보존식으로 교체한다.

\[
\sum n_{\mathrm{II}}+\sum n_{\mathrm{III}}+
\sum n_{\mathrm{IV}}+M_V
=n_{\mathrm{Fe,total}}-M_{\mathrm{outside}}.
\]

여기서 `M_outside`는 upstream population table의 Fe I 및 Fe VI 이상 고정 질량이다. 이를 0으로 가정하거나 `M_V`에 몰래 합치지 않고 입력값과 분율을 기록한다. Fe VI 이상 질량 또는 Fe V↔VI process가 사전등록 허용치보다 크거나 측정 불가능하면 이 boundary-stage는 commit 불가다. 이 설계는 Fe V에서 창을 닫는 근사이므로, 오른쪽 경계의 무시 가능성이 계측으로 확인될 때만 유효하다.

보존행은 solve 안에 들어가며 solve 후 stage 합을 1로 다시 나누지 않는다.

### 6.5 필수 플럭스 폐합 원장

신규 artifact `lumina_ew_boundary_mass.csv`를 셸/iteration마다 한 행 이상 기록하도록 설계한다. 최소 필드는 다음과 같다.

- identity: run, iter, shell, Z, `stage=FeV_boundary_mass`, atomic/checksum, route count/valid count;
- mass: `n_Fe_total`, `M_outside`, `M_V_before`, `M_V_after`, `sum_II`, `sum_III`, `sum_IV`, conservation residual;
- projection: `q_count`, `sum_q`, `q_min`, `q_max`, `q_checksum`;
- channel별 gross flux: `Phi_IV_to_V_rad`, `Phi_V_to_IV_rad`, coll, nonthermal, DR/autoion;
- totals: `Phi_forward`, `Phi_reverse`, `Phi_net = Phi_forward-Phi_reverse`;
- 독립 폐합: 실제 matrix의 `M_V` column/row에서 다시 읽은 flux와 event ledger의 차이, `boundary_row_residual`, scale-normalized residual;
- scope: Fe I fraction, Fe VI+ fraction, right-boundary producer coverage, verdict/fallback reason.

event ledger는 matrix 쓰기와 별도 배열에 gross forward/reverse를 누산하고, 최종 matrix에서 재계산한 값과 비교한다. `+r/-r` 동일 문장쌍의 합을 다시 확인하는 방식은 금지한다. seed로 interface diagonal debit, target row, `q_t`, 또는 route 하나를 훼손했을 때 각각 residual/coverage가 FAIL 가능해야 한다.

사전 gate는 다음처럼 고정한다.

- data: Fe IV levels/σ/valid Fe V targets `200/200`, Fe V levels `200`, IP finite; 하나라도 다르면 FAIL;
- projection: `|sum_q-1| <= 1e-12`, 모든 q finite/nonnegative; 보정 없이 FAIL;
- matrix/ledger: channel assembly 및 boundary flux residual `<=1e-12`;
- solve: 기존 EW의 conservation `<=1e-12`, scaled SE residual `<=1e-10`, nonfinite/negative 0;
- external scope: Fe I/Fe VI+ 질량과 V↔VI producer coverage를 반드시 기록하며, 임계값은 구현 발주 전에 운전석이 수치로 승인해야 한다. 승인 전 COMMIT=0만 허용한다.

### 6.6 사전등록 기대

구현·판정런 전에 아래 방향을 고정한다.

1. s0 Fe IV의 `elem/anchor = 1.0111`은 약 `1.000`으로 이동한다. 1.0111이 전부 누락 경계질량 때문이라는 단순 역산은 `1-1/1.0111 = 0.010978`이므로, `M_V/n_Fe`가 약 1.1% 규모로 기록될 것을 기대한다. 이는 튜닝 목표가 아니라 경계질량 원인설명의 독립 수치 대조다.
2. s0 Fe II/III/IV 각각에서 `d_k(elem) < d_k(pair)`가 모두 성립해야 한다. 어느 한 stage라도 실패하면 R6 회복 주장은 FAIL이다.
3. s8은 유의 변화 없음이 기대다. 경계질량을 넣은 뒤 s8의 stage별 오차나 D가 의미 있게 움직이면 “경계 bookkeeping”이 아니라 radiation-field 내용이라는 기존 귀속을 재검토할 신규 신호로 기록한다. 차이가 기대와 반대여도 재튜닝하지 않는다.
4. `M_V`, gross forward/reverse flux, net flux, 보존 residual을 함께 제시하지 않은 1.000 근접은 폐합 증거로 인정하지 않는다.

이 R6 안은 설계 대안일 뿐이며, 명세대로 운전석 검수 전 구현해서는 안 된다.

## 7. 최종 처분

- 차단 발견: R1 전역 `super_mode`/파생 레이아웃 불변식 미포섭, R3 조건 복제와 pair GPU 우회, R5 비실측 0 카운터, R5 Kramers 내용 불일치.
- 유지 발견: R1 τ/source의 직접 권한 판정은 정공법, EW JEQB는 유효, D6 원장은 구조적으로 실패 가능, R4 무단 역항 구현 없음, 신규 clamp/floor/cap 0.
- R6: Fe V full-stage는 데이터 부족으로 거부하고, 명시적 scalar 경계질량 + 양방향 IV↔V flux + 독립 폐합 원장 설계만 제출.

처분은 기록으로 한정한다. `src/` 수정 및 테스트 실행은 하지 않았다.
