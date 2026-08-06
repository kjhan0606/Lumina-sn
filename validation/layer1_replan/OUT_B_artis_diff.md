# 판독 결과

`[실측]`은 소스에 직접 드러난 배선·수식이고, `[추정]`은 그 제어 흐름에서 도출한 실행 결과다.

## 1. ARTIS 반복 1회의 물리 사슬

ARTIS의 반복 단위는 `update_grid → estimator 초기화 → packet 수송 → estimator 합산`이다. 즉 물질 상태는 수송 전에 갱신되며, 수송에서 만들어진 estimator는 다음 반복의 입력이 된다 (`artis/sn3d.cc:632-649`, `artis/sn3d.cc:670-686`).

| 순서 | 산출량 | 함수·수식 | 입력 | 근거 |
|---:|---|---|---|---|
| 1 | 정규화된 \(J,\nu J\), line/BF estimator | `normalise_J`, `normalise_nuJ`, `normalise_bf_estimators` | 직전 수송의 path-length·line·BF estimator, 셀 부피, \(\Delta t\) | `[실측]` `artis/update_grid.cc:439-472`, `artis/radfield.cc:819-873`, `artis/update_grid.cc:594-596` |
| 2 | \(T_J,T_R,W\), 선택 시 주파수-bin별 \(T_R,W\) | `fit_parameters`; \(T_R=h\langle\nu\rangle/(3.832\,k)\), \(W=\pi J/(\sigma T_R^4)\) | 정규화한 \(J,\nu J\) | `[실측]` `artis/radfield.cc:361-399`, `artis/radfield.cc:735-815`, 호출 `artis/update_grid.cc:482-485` |
| 3a | 비-NLTE 이온의 분배함수 \(U_i\) | `calculate_cellpartfuncts` | 현재 ground/level population; 첫 pass는 임시 \(n_0=1\) | `[실측]` `artis/update_grid.cc:202-207`, `artis/ltepop.cc:169-206`, `artis/ltepop.cc:383-390` |
| 3b | BF heating coefficient | `calculate_bfheatingcoeffs` | 고정된 직전 수송 복사장 | `[실측]` `artis/update_grid.cc:176-189` |
| 4 | \(T_e\) | `call_T_e_finder`, TOMS-748 thermal-balance root | BF·FF·충돌·비열적 heating, FF·FB·충돌·팽창 cooling | `[실측]` `artis/update_grid.cc:211-216`, `artis/thermalbalance.cc:69-113`, `artis/thermalbalance.cc:115-174`, `artis/thermalbalance.cc:236-304` |
| 4a | trial별 \(n_e\), ion population | `T_e_eqn_heating_minus_cooling` 안에서 `calculate_ion_balance_nne` | trial \(T_e\), partition, Saha 또는 rate balance | `[실측]` `artis/thermalbalance.cc:141-150`, `artis/ltepop.cc:435-469` |
| 5 | NLTE ion·level·super-level population | 원소별 `solve_nlte_pops_element`; 한 원소의 사용 이온들을 하나의 SE 행렬로 풂 | \(T_e,n_e\), bound-bound, 광/충돌 ionization·recombination, non-thermal·autoionization rates | `[실측]` `artis/update_grid.cc:236-245`, `artis/nltepop.cc:1165-1168`, `artis/nltepop.cc:1205-1247`, `artis/nltepop.cc:1249-1289`, `artis/nltepop.cc:1333-1368` |
| 6 | NLTE 결과를 반영한 새 \(U_i\) | NLTE element마다 `calculate_cellpartfuncts` 재호출 | 명시적 NLTE level population과 super-level constituent population | `[실측]` `artis/update_grid.cc:240-245`, `artis/ltepop.cc:146-166`, `artis/ltepop.cc:169-196` |
| 7 | 수렴한 \(n_e\), 최종 ion ground population | `calculate_ion_balance_nne`; NLTE solver가 설정한 원소는 덮어쓰지 않음 | NLTE 결과, 비-NLTE 원소의 Saha/rate balance | `[실측]` `artis/update_grid.cc:249-266`, `artis/ltepop.cc:454-469` |
| 8 | 비명시 준위 population | `calculate_levelpop`; 명시 NLTE→저장값, super-level→Boltzmann 분배, 나머지→Boltzmann | ground/NLTE/super-level population, \(T_e\) 또는 구성에 따라 \(T_J\) | `[실측]` `artis/ltepop.cc:136-166`, `artis/ltepop.cc:350-380` |
| 9 | cooling table 및 선택적 expansion opacity | `calculate_cooling_rates`, `calculate_expansion_opacities` | 갱신된 \(T_e,n_e\), ion·level populations | `[실측]` `artis/update_grid.cc:542-561`, `artis/rpkt.cc:921-955` |
| 10 | 새 반복용 빈 estimator | `zero_estimators` | 기존 estimator 제거 | `[실측]` `artis/sn3d.cc:670-673`, `artis/radfield.cc:656-673` |
| 11 | packet 궤적·상호작용·새 estimator | `update_packets`; 내부 packet step | 갱신된 물질 상태와 opacity/tau | `[실측]` `artis/sn3d.cc:676-680`, `artis/rpkt.cc:82-169`, `artis/radfield.cc:675-713` |
| 11a | line Sobolev \(\tau\) | `get_tau_sobolev`; \(\max[(B_{lu}n_l-B_{ul}n_u)hc\,t/(4\pi),0]\) | 그 시점의 lower/upper level population | `[실측]` 수송 중 호출 `artis/rpkt.cc:146-169`, 수식 `artis/rpkt.cc:59-80` |
| 11b | macro-atom action rates와 누적 선택분포 | cache가 비어 있으면 `calculate_macroatom_transitionrates`, 그 뒤 누적합으로 난수 선택 | 현재 \(T_e,n_e\), level populations, 방사·충돌·비열적 전이율 | `[실측]` `artis/macroatom.cc:61-190`, `artis/macroatom.cc:321-329`, `artis/macroatom.cc:332-402` |
| 12 | 다음 반복의 전역 estimator | `mpi_reduce_estimators` | 각 rank의 \(J\), heating, photoionization, deposition estimator | `[실측]` `artis/sn3d.cc:681-686`, `artis/sn3d.cc:456-518` |

LTE/grey 반복에서는 별도 thermal-balance root 대신 \(T_e=T_R=T_J,\ W=1\)로 놓고 partition 및 Saha ionization을 계산한다 (`artis/update_grid.cc:447-467`).

---

## 2. ★ ARTIS 반복 0: 첫 수송 전 LTE start

| 부트스트랩 단계 | 첫 입력·산출량 | ARTIS 경로 | 근거 |
|---:|---|---|---|
| 0 | fresh-run packet과 빈 estimator | `packet_init`, `zero_estimators` | `[실측]` `artis/sn3d.cc:898-902` |
| 1 | 초기 \(T_e\) | `update_grid_cell` 진입 전에 이미 설정됨. 코드가 명시한 공급자는 trapped-energy-release 계산 또는 gridsave | `[실측]` `artis/update_grid.cc:397-400` |
| 2 | LTE mode | `lte_iteration = timestep < num_lte_timesteps`; 첫 timestep은 반드시 LTE ionization balance를 풀도록 assert | `[실측]` `artis/update_grid.cc:584-586` |
| 3 | 모든 이온의 첫 partition \(U_i\) | 복사장 estimator를 정규화하지 않고 곧바로 `calculate_cellpartfuncts` | `[실측]` `artis/update_grid.cc:397-424`; 첫-pass 처리 `artis/ltepop.cc:177-205` |
| 4 | \(n_e\) | \([0,\rho/m_H]\)에서 charge-conservation root를 TOMS-748로 계산 | `[실측]` `artis/ltepop.cc:245-267`, `artis/ltepop.cc:435-458` |
| 5 | ion populations | `force_saha=true`; \(U_i/U_{i+1},T_e,\chi_i,n_e\)로 Saha ladder를 정규화 | `[실측]` `artis/ltepop.cc:29-40`, `artis/ltepop.cc:314-347`, `artis/ltepop.cc:435-465` |
| 6 | ground populations | \(n_{i,0}=n_i g_{i,0}/U_i\) | `[실측]` `artis/ltepop.cc:392-432` |
| 7 | excited-level populations | 사용 시점에 ground로부터 Boltzmann 계산; 아직 유효한 NLTE population이 없으면 이 경로가 선택됨 | `[실측]` `artis/ltepop.cc:136-166`, `artis/ltepop.cc:350-380` |
| 8 | pre-transport opacity/tau | 선택적 expansion opacity는 갱신된 population으로 미리 계산; 개별 line \(\tau\)는 수송 중 같은 population에서 계산 | `[실측]` `artis/update_grid.cc:555-561`, `artis/rpkt.cc:921-955`, `artis/rpkt.cc:59-80` |
| 9 | 첫 macro-atom 전이분포 | 첫 활성화 시 현재 LTE-start 상태로 lazy 계산 | `[실측]` `artis/macroatom.cc:321-329`, `artis/macroatom.cc:383-402` |
| 10 | 첫 packet 수송 | 위 상태가 확정된 뒤 `zero_estimators → update_packets` | `[실측]` `artis/sn3d.cc:647-679` |

재시작에서는 저장된 상태를 유지하여 fresh-run용 `calculate_ion_balance_nne` 호출을 건너뛴다. 저장된 \(W=1\)인데 유효한 \(\Gamma\) estimator가 없는 셀은 grey/LTE로 표시된다 (`artis/update_grid.cc:409-426`).

---

## 3. 이미 확인된 사실의 확인/반박

| 사실 | 판정 | 검증 내용과 ARTIS 대응 |
|---|---|---|
| ARTIS top ion은 준위 1개 | **확인 — 옵션 활성 경로** | `[실측]` top ion의 `nlevelslimit=1`, 그리고 최소 한 준위를 assert한다 (`artis/input.cc:1226-1234`). 준위가 하나면 transition table을 읽지 않는다 (`artis/input.cc:1280-1296`). BF target도 top ground level 0으로 보낸다 (`artis/input.cc:141-153`). |
| Lumina는 ionization energy \(n\)개에서 population \(n+1\)개를 만든다 | **확인** | `[실측]` `total_ion_pops += n_ioniz + 1` (`lumina/lumina_atomic.c:1979-1994`); 각 population의 level 수는 실제 `(Z,stage)` 일치 level만 센다 (`lumina/lumina_atomic.c:2024-2035`). |
| Lumina의 zero-level population은 15/74이며 전부 원소 top | **소스 내 실측기록 확인** | `[실측]` 동일 측정값과 “전부 원소 최상단”이 `lumina/population_contract.c:92-104` 및 `lumina/lumina_plasma.c:6724-6729`에 기록되어 있다. 구조적으로 top population을 추가하는 로더와도 일치한다 (`lumina/lumina_atomic.c:1979-2035`). |
| zero-level top의 partition | **확인** | `[실측]` Lumina는 `hi==lo`이면 \(Z=1\)을 반환한다 (`lumina/population_contract.c:90-111`). ARTIS의 single-level top은 \(U=g_{\rm ground}\)이다 (`artis/ltepop.cc:188-196`). |
| `lumina_plasma.c:2347`의 “LTE-Saha pin으로 fail-closed” | **반박** | `[실측]` 주석은 fallback을 선언하지만 (`lumina/lumina_plasma.c:2340-2347`), 계산한 `phi_neb`는 `(void)`로 폐기되고 `r1_use==0`이면 `POP_BF_STALE`를 반환한다 (`lumina/lumina_plasma.c:2643-2649`). ARTIS 대응은 첫 timestep LTE 강제와 실제 `phi_saha` 사용이다 (`artis/update_grid.cc:584-586`, `artis/ltepop.cc:29-40`, `artis/ltepop.cc:435-465`). |
| `parity_field_built`는 수송 복사장을 요구 | **확인** | `[실측]` `NLTEConfig`, enabled 상태, `RADIATION_FIELD_VIEW_OK`, \(J_\nu\), validity, shell 범위를 모두 요구한다 (`lumina/lumina_plasma.c:2364-2377`). 이 view는 packet 수송·MC commit 뒤에만 refresh된다 (`lumina/lumina_main.c:484-546`). |
| Lumina 첫 tau 준비 시 복사장/NLTE 포인터 상태 | **확인** | `[실측]` seed \(T_e\) 발행과 `lumina_prepare_solver_owned_tau`는 `lumina_main.c:253-265`; `nlte_init`와 `bf_set_nlte_pops`는 그보다 뒤인 `lumina_main.c:296-305`다. `r1_use`는 그 포인터와 transported field를 모두 요구한다 (`lumina/lumina_plasma.c:2529-2530`). |
| 첫 수송 도달 여부 | **현재 기본 CPU 경로에서는 도달 불가로 추정** | `[추정]` 첫 `compute_plasma_state`가 ion population 계산을 호출하고 (`lumina/lumina_plasma.c:6680-6722`), 첫 ion pair에서 `r1_use==0 → POP_BF_STALE`가 되며 (`lumina/lumina_plasma.c:2529-2530`, `lumina/lumina_plasma.c:2643-2649`), `lumina_prepare_solver_owned_tau`가 실패를 반환한다 (`lumina/lumina_plasma.c:6594-6618`). 이 호출은 packet loop보다 앞이다 (`lumina/lumina_main.c:253-265`, `lumina/lumina_main.c:409-490`). |

---

## 4. ARTIS–Lumina 대조표

| 항목 | 분류 | ARTIS | Lumina |
|---|---|---|---|
| **1. 복사장 없는 첫 population 공급자** | **MISSING** | `[실측]` 첫 timestep을 LTE로 강제하고 실제 Saha \(n_e\)·ion population을 만든다 (`artis/update_grid.cc:397-427`, `artis/update_grid.cc:584-586`, `artis/ltepop.cc:435-465`). | `[실측]` seed \(T_e\)와 tau-bootstrap 틀은 있으나 (`lumina/lumina_main.c:253-265`), Saha 결과를 폐기하고 field가 없으면 stale을 반환한다 (`lumina/lumina_plasma.c:2643-2649`). 따라서 “성공하는 LTE population 공급자”가 없다. |
| **초기 \(T_e,n_e\) seed** | **DIVERGENT** | `[실측]` \(T_e\)는 trapped-energy release 또는 gridsave에서 이미 배정되고, fresh \(n_e\)는 Saha charge root로 구한다 (`artis/update_grid.cc:397-426`, `artis/ltepop.cc:245-267`). | `[실측]` 모든 shell의 \(T_e\) seed를 `T_inner`로 놓고 (`lumina/lumina_atomic.c:1057-1061`), 초기 \(n_e\)는 reference opacity에서 복사한다 (`lumina/lumina_main.c:146-157`). 이후 bootstrap에서 iterative \(n_e\) 계산을 시도한다 (`lumina/lumina_plasma.c:6700-6722`). |
| **2. top ion 준위 구조** | **DIVERGENT** | `[실측]` 옵션 활성 시 ground 1개, \(U=g_0\), 내부 bound-bound transition 없음 (`artis/input.cc:1226-1234`, `artis/input.cc:1280-1296`, `artis/ltepop.cc:188-196`). | `[실측]` 기본 loader의 extra top population은 level 0개이고 \(Z=1\) (`lumina/lumina_atomic.c:1979-2035`, `lumina/population_contract.c:90-105`). |
| **top ion continuum/macro transition** | **DIVERGENT** | `[실측]` top으로 가는 BF target을 level 0에 연결하되 top 내부 전이는 없다 (`artis/input.cc:141-153`, `artis/input.cc:1280-1296`). | `[실측]` 기본 zero-level top에는 level target이 없다. 별도의 default-off O IV synthetic anchor만 존재하며, 그 macro block도 비어 있다 (`lumina/lumina_atomic.c:2672-2683`, `lumina/lumina_atomic.c:2718-2727`). |
| **3. Saha/rate-SE 선택 조건** | **DIVERGENT** | `[실측]` LTE iteration·thick cell은 Saha; 비-NLTE 원소의 얇은 non-LTE 셀은 estimator 기반 rate balance; NLTE 원소는 SE matrix 결과를 보존한다 (`artis/ltepop.cc:47-85`, `artis/ltepop.cc:435-465`). | `[실측]` production ion ratio는 transported BF view가 있을 때만 rate-SE이며, 계산된 LTE/nebular Saha 값은 shadow 값이다 (`lumina/lumina_plasma.c:2529-2530`, `lumina/lumina_plasma.c:2643-2654`). |
| **간략 ion rate balance의 항** | **DIVERGENT** | `[실측]` 비-NLTE 원소 경로는 ground photoionization estimator + nonthermal ionization 대 spontaneous recombination이며 collisional recombination은 상수 `false`로 빠져 있다 (`artis/ltepop.cc:62-85`). | `[실측]` rate-SE ratio는 canonical \(J_\nu\) photoionization, radiative recombination, Seaton collisional ionization 및 three-body inverse, 선택적 NT 항을 사용한다 (`lumina/lumina_plasma.c:2386-2439`, `lumina/lumina_plasma.c:2449-2490`). |
| **NLTE ionization·level SE의 행렬 범위** | **DIVERGENT** | `[실측]` 한 원소의 여러 ion과 level을 단일 SE matrix에 넣고 bound-bound/ionization/NT/autoionization을 함께 푼다 (`artis/nltepop.cc:1165-1168`, `artis/nltepop.cc:1205-1289`). | `[실측]` 기본 경로는 지정된 adjacent-ion pair들을 순차 풀이하고 CE 반복으로 결합한다 (`lumina/lumina_plasma.c:15246-15265`, `lumina/lumina_plasma.c:15330-15421`). element-wide 경로는 별도 gate다 (`lumina/lumina_plasma.c:15278-15328`). |
| **SE 실패/저온의 LTE population fallback** | **MISSING** | `[실측]` `T_e==MINTEMP`, singular/invalid solution에서 원소 population을 LTE로 설정한다 (`artis/nltepop.cc:1183-1191`, `artis/nltepop.cc:1319-1326`). | `[실측]` stale radiation/line view 또는 population solve failure는 오류 반환과 transaction abort로 끝난다 (`lumina/lumina_plasma.c:15161-15188`, `lumina/lumina_plasma.c:15218-15231`, `lumina/lumina_plasma.c:15514-15517`). |
| **4. LTE bound-level Boltzmann 수식 자체** | **EQUIVALENT** | `[실측]` \(n_l/n_0=(g_l/g_0)e^{-\Delta E/kT_{\rm exc}}\) (`artis/ltepop.cc:350-368`). | `[실측]` \(f_l=g_l e^{-(E_l-E_0)/kT_e}/Z\) (`lumina/population_contract.c:125-138`). 단, ARTIS의 \(T_{\rm exc}\)는 구성에 따라 \(T_e\) 또는 \(T_J\)다 (`artis/ltepop.cc:361-367`). |
| **전체 partition 함수** | **DIVERGENT** | `[실측]` \(U=g_0[1+\sum_{l>0}n_l/n_0]\); 따라서 유효 NLTE population과 super-level population이 \(U\)에 반영된다 (`artis/ltepop.cc:169-196`). | `[실측]` production \(Z(T_e)\)는 level membership, \(E_l,g_l,T_e\)만으로 전체 full level을 합산한다 (`lumina/population_contract.c:86-118`, `lumina/lumina_plasma.c:2151-2178`). NLTE population은 이 합에 들어가지 않는다. |
| **super-level 합산·투영** | **DIVERGENT** | `[실측]` 한 이온의 lumped levels를 super-level partition으로 합산하고 constituent를 Boltzmann 비로 되푼다 (`artis/nltepop.cc:428-443`, `artis/nltepop.cc:1410-1420`, `artis/ltepop.cc:153-166`). | `[실측]` `super_level` mapping은 임의 다중 group 또는 identity이며 (`lumina/lumina_atomic.c:1430-1468`), NLTE projection에서 full-level→super mapping과 within-SL fraction을 별도로 만든다 (`lumina/lumina_plasma.c:12091-12118`, `lumina/lumina_plasma.c:12126-12173`). 반면 production partition은 super mapping을 전달받지 않는다 (`lumina/lumina_plasma.c:2153-2163`). |
| **packet path-length radiation estimator** | **EQUIVALENT** | `[실측]` packet 이동거리로 \(J,\nu J\), BF 및 line estimator를 누적하고 수송 뒤 rank reduction한다 (`artis/radfield.cc:675-713`, `artis/sn3d.cc:679-686`). | `[실측]` packet별 estimator를 thread-local로 누적·reduce하고 `MC_PATH_LENGTH` provenance로 canonical field를 commit한다 (`lumina/lumina_main.c:458-545`). |
| **5. line \(\tau\)의 소유·갱신 시점** | **DIVERGENT** | `[실측]` 상태 갱신이 수송보다 먼저이며 (`artis/sn3d.cc:647-679`), 개별 Sobolev \(\tau\)는 line encounter 시 현재 \(n_l,n_u\)로 계산된다 (`artis/rpkt.cc:59-80`, `artis/rpkt.cc:146-169`). | `[실측]` disk tau는 stale seed로 표시되고 (`lumina/lumina_atomic.c:1122-1142`), solver가 population에서 bulk tau를 만든다 (`lumina/lumina_plasma.c:6793-6806`). 정상 outer loop에서는 수송 후 `T_e→plasma→NLTE→tau`이고 (`lumina/lumina_main.c:409-546`, `lumina/lumina_main.c:625-683`), 그 tau는 다음 수송에서 사용된다. |
| **NLTE 후 tau 재계산** | **DIVERGENT** | `[실측]` 별도 published tau 배열보다 수송 시 current level population을 조회한다 (`artis/rpkt.cc:59-80`). | `[실측]` NLTE ion-stage writeback 뒤 별도 `nlte_update_tau_sobolev`로 배열을 다시 쓴다 (`lumina/lumina_plasma.c:15535-15544`). |
| **6. \(T_e\)의 반복 내 위치** | **DIVERGENT** | `[실측]` 직전 수송 estimator를 읽어 물질 상태와 \(T_e\)를 먼저 풀고 그 상태로 다음 packet 수송을 한다 (`artis/sn3d.cc:647-686`, `artis/update_grid.cc:176-266`). | `[실측]` 현 outer iteration의 packet 수송·radiation commit 뒤, `iter>0`에서 \(T_e\)를 풀고 이어 plasma/NLTE/tau를 갱신한다 (`lumina/lumina_main.c:409-546`, `lumina/lumina_main.c:611-683`). |
| **thermal-balance trial에서 population 결합** | **DIVERGENT** | `[실측]` 매 trial \(T_e\)마다 ion balance와 \(n_e\), cooling, heating을 다시 계산한다 (`artis/thermalbalance.cc:141-174`). | `[실측]` A210 trial은 이미 발행된 opacity/emissivity와 고정 context의 \(n_e\)를 사용하며, photo/line/FF absorption, recombination/line/FF emission, Compton, gamma, adiabatic 항을 합산한다 (`lumina/lumina_plasma.c:11917-11965`, `lumina/lumina_plasma.c:11979-12009`). plasma population 계산은 \(T_e\) solve 뒤다 (`lumina/lumina_main.c:651-663`). |
| **macro-atom transition probability 갱신** | **DIVERGENT** | `[실측]` packet 활성화 시 cache가 없으면 현재 \(T_e,n_e\), population, radiation rates에서 즉시 계산한다 (`artis/macroatom.cc:61-190`, `artis/macroatom.cc:383-402`). | `[실측]` 기본값은 reference `transition_probabilities.npy`이고 (`lumina/lumina_atomic.c:1152-1165`), dynamic 재계산은 기본 off이며 선택 시 plasma/NLTE 갱신 뒤에만 실행된다 (`lumina/lumina_main.c:183-188`, `lumina/lumina_main.c:705-710`). |

## 요약 판정

| 우선 쟁점 | 판정 |
|---|---|
| 부트스트랩/LTE start | **MISSING** — Lumina에는 seed \(T_e\)와 bootstrap 호출은 있으나, 복사장 없는 population을 실제 공급하는 LTE-Saha 경로가 실행되지 않는다 (`lumina/lumina_main.c:253-265`, `lumina/lumina_plasma.c:2643-2649`; ARTIS `artis/update_grid.cc:397-427`). |
| top ion | **DIVERGENT** — ARTIS ground 1개/\(U=g_0\)/전이 없음 대 Lumina level 0개/\(Z=1\) (`artis/input.cc:1226-1296`; `lumina/population_contract.c:90-105`). |
| ionization balance | **DIVERGENT** — ARTIS는 LTE/thick Saha와 non-LTE rate-SE를 조건부 전환하지만 Lumina production ion ladder는 transported-field rate-SE만 허용한다 (`artis/ltepop.cc:435-465`; `lumina/lumina_plasma.c:2529-2530`, `lumina/lumina_plasma.c:2643-2654`). |
| partition | **DIVERGENT** — ARTIS는 현재 population 비로 \(U\)를 구성하고 Lumina는 원자 membership과 \(T_e\)만 합산한다 (`artis/ltepop.cc:169-196`; `lumina/population_contract.c:86-118`). |
| tau/opacity 시점 | **DIVERGENT** — ARTIS는 pre-transport 상태와 on-demand line tau, Lumina는 post-transport bulk publication을 사용한다 (`artis/sn3d.cc:647-686`, `artis/rpkt.cc:59-80`; `lumina/lumina_main.c:611-683`). |
| \(T_e\) | **DIVERGENT** — ARTIS trial은 population/\(n_e\)와 결합되고 수송 전에 위치하며, Lumina A210은 발행된 opacity/emissivity와 고정 \(n_e\) context를 사용해 수송 뒤 실행된다 (`artis/thermalbalance.cc:141-174`; `lumina/lumina_plasma.c:11979-12009`, `lumina/lumina_main.c:625-663`). |

이 분류는 배선·수식·순서의 차이만 기술하며 어느 구현의 물리적 타당성이 우위인지는 판정하지 않는다.