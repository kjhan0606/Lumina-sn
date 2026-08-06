## 0. 실행 도달성

[실측] 현재 코드는 루프 전에 덱 기반 `T_e`를 1세대로 발행하고 `lumina_prepare_solver_owned_tau()`를 호출하며, NLTE 및 복사장 owner 초기화와 반복 루프는 그 뒤에 있다(`lumina/lumina_main.c:253-265`, `lumina/lumina_main.c:296-305`, `lumina/lumina_main.c:409`).

[실측] pre-loop plasma 계산은 Z 후보를 만든 다음 `n_e` 반복 중 ion population을 요구하지만, 이 시점의 `g_bf_nlte_pops`는 아직 `NULL`이고 transported view도 없으므로 활성 다단 이온 사다리는 `POP_BF_STALE`을 반환한다(`lumina/lumina_plasma.c:6680-6684`, `lumina/lumina_plasma.c:6711-6713`, `lumina/lumina_plasma.c:7196-7197`, `lumina/lumina_plasma.c:2526-2530`, `lumina/lumina_plasma.c:2643-2649`).

[실측] 기본 `n_e` 경로에서는 그 상태가 `compute_electron_density()`의 `-1`을 거쳐 외부에서 `POP_NE_NOT_CONVERGED`로 치환되고, population transaction이 abort된 뒤 `main`이 `EXIT_FAILURE`를 반환한다(`lumina/lumina_plasma.c:2775-2781`, `lumina/lumina_plasma.c:6711-6713`, `lumina/lumina_plasma.c:6667-6676`, `lumina/lumina_plasma.c:6602-6617`, `lumina/lumina_main.c:263-265`).

## 1. 사슬 표

[실측] 아래 순서는 소스의 반복 본문 순서이며, `iter=0`에서는 1–4까지만 예정되고 plasma 갱신 블록은 `iter > 0` 조건 때문에 실행되지 않는다(`lumina/lumina_main.c:603-616`).

| 산출량 | 계산 함수 | 실제 입력 | 호출 위치 | 반복 내 순서 |
|---|---|---|---|---:|
| 수송된 packet 경로·escape·기본 추정자 | `single_packet_loop()` | 초기 packet의 `T_inner`, geometry, `OpacityState`의 `tau_sobolev`·`transition_probabilities`·`electron_density`, 선택적 BF 및 plasma(`lumina/lumina_main.c:17-68`, `lumina/lumina.h:209-235`) | `lumina/lumina_main.c:484-490` | 1 |
| thread-local → 전역 MC 추정자 | 직접 합산, `radiation_field_accumulator_reduce()` | 각 packet의 `local_est` 및 path-length accumulator(`lumina/lumina_main.c:517-531`) | `lumina/lumina_main.c:522-531` | 2 |
| canonical 복사장 추정자 `J_nu`, validity, count, generation 및 line `J̄` | `radiation_field_commit()` | raw path length, count, volume, `time_simulation`, line sum/sumsq/count, generation=`iter+1`(`lumina/lumina_main.c:545`) | `lumina/lumina_main.c:545` | 3 |
| checked `J_nu`/line-`J̄` view | `radiation_field_read_view()`, `radiation_field_line_jbar_view()` | 방금 commit된 epoch·shell 수·generation과 q-set/profile identity(`lumina/radiation_field.c:706-768`, `lumina/radiation_field.c:771-811`) | `lumina/lumina_main.c:546-552` | 4 |
| scalar radiation state | `solve_radiation_field()` | 인자는 받지만 모두 `(void)` 처리되며 산출물을 쓰지 않음(`lumina/lumina_plasma.c:1054-1067`) | `lumina/lumina_main.c:605-609` | 5; 실질 산출 없음 |
| `T_e`, `n_e`의 A2-10 publication | `compute_radiative_equilibrium_te()` → `a210_production_solve()` → `a210_solve_transaction()` | 현재 checked `J_nu`; generation이 맞는 CPU opacity/emissivity; committed population; 기존 `n_e`; gamma; epoch(`lumina/lumina_plasma.c:11979-12009`) | `lumina/lumina_main.c:633-644` | 6; `iter>0`이며 RADEQ/self-consistent gate가 켜진 경우 |
| `T_e_generation` | `main`에서 증가/무효화 | `te_qualified != 0`일 때만 `++`; 실패 시 0(`lumina/lumina_main.c:651-656`) | `lumina/lumina_main.c:651-656` | 7 |
| 분배함수 `Z(T_e)` | `compute_partition_functions()` → `population_partition_build()` | atomic level membership·energy·`g`, `T_e`, nonzero `T_e_generation`, 다음 population generation(`lumina/lumina_plasma.c:2151-2178`, `lumina/population_contract.c:113-118`) | `compute_plasma_state()` 안 `lumina/lumina_plasma.c:6680-6684` | 8 |
| 임시 ion population과 수렴한 `n_e` | `compute_electron_density()` ↔ `compute_ion_populations_shell()` | seed `n_e`, `T_e`, Z, rho·abundance·mass, canonical BF `J_nu`, 원자 BF/재결합 자료(`lumina/lumina_plasma.c:2496-2504`, `lumina/lumina_plasma.c:2550-2560`, `lumina/lumina_plasma.c:2752-2795`) | `lumina/lumina_plasma.c:6704-6715` | 9; ion↔charge 고정점 반복 |
| 최종 ion population | `compute_ion_populations()` → `compute_ion_populations_shell()` | 수렴한 `n_e`, `T_e`, Z, canonical BF view 및 element density(`lumina/lumina_plasma.c:2526-2530`, `lumina/lumina_plasma.c:2584-2594`, `lumina/lumina_plasma.c:2680-2705`) | `lumina/lumina_plasma.c:6718-6722` | 10 |
| committed Z, `n_e`, ion population | `population_transaction_commit()` | 위 세 후보 전체; 하나라도 실패하면 모두 rollback(`lumina/population_contract.c:211-217`, `lumina/lumina_plasma.c:6778-6787`) | `lumina/lumina_plasma.c:6781` | 11 |
| LTE 준위 population `n_lower`, `n_upper` | `population_lte_level_fraction()` 후 `n_ion × fraction` | committed ion population, `T_e`, Z, level energy·`g`(`lumina/population_contract.c:125-138`, `lumina/lumina_plasma.c:2928-2945`) | `compute_tau_sobolev()` 내부 `lumina/lumina_plasma.c:2931-2945` | 12 |
| LTE `tau_sobolev` | `compute_tau_sobolev()` → `a208_signed_sobolev()` | `f_lu`, wavelength, epoch, `n_lower/n_upper`, `g_lower/g_upper`, required tau generation(`lumina/lumina_plasma.c:2843-2855`, `lumina/lumina_plasma.c:2947-2953`) | `lumina/lumina_plasma.c:6793-6795` | 13 |
| BF/FF opacity | `compute_bf_opacity()` | 갱신된 `T_e`, `n_e`, ion/level population(`lumina/lumina_main.c:665-667`) | `lumina/lumina_main.c:665-667` | 14 |
| 명시적 NLTE 준위 population | `nlte_solve_all()` → `nlte_solve_ion_shell()` | current `J_nu`, line `J̄`, Z, `T_e`, `n_e`, ion totals, atomic radiative/collisional rates(`lumina/lumina_plasma.c:15161-15187`, `lumina/lumina_plasma.c:15233-15243`, `lumina/lumina_plasma.c:15395-15413`) | `lumina/lumina_main.c:669-678` | 15; NLTE gate가 켜진 경우 |
| NLTE 선의 `tau_sobolev` overwrite | `nlte_update_tau_sobolev()` | solved NLTE level populations 및 atomic line data(`lumina/lumina_plasma.c:15535-15544`) | `lumina/lumina_plasma.c:15544` | 16 |
| signed opacity/emissivity publication | `a208_publish_cpu_opacity()`, `a209_publish_cpu_emissivity()` | population·partition·`T_e`·`n_e`·tau·radiation generation 및 line source(`lumina/lumina_plasma.c:8036-8059`, `lumina/lumina_plasma.c:8127-8159`) | `lumina/lumina_main.c:685-702` | 17 |
| 전이확률 | `compute_transition_probabilities()` | `T_e`, `n_e`, tau→β, A/B 계수, 선택적 committed NLTE/LTE 준위 population과 `J_nu`; block별 rate 정규화(`lumina/lumina_plasma.c:4325-4332`, `lumina/lumina_plasma.c:4370-4382`, `lumina/lumina_plasma.c:4813-4831`, `lumina/lumina_plasma.c:5037-5048`) | `lumina/lumina_main.c:705-710` | 18; dynamic gate와 hold 조건 충족 시 |

## 2. 세대·신선도 계약 표

| 계약명 | 검사 함수 | 요구 세대 | 위반 시 상태코드 |
|---|---|---|---|
| tau freshness | `tau_sobolev_assert_fresh()` (`lumina/lumina_plasma.c:6508-6528`) | `computed_generation != 0`이고 `computed ≥ required`(`lumina/lumina_plasma.c:6509-6518`) | `-1`(`lumina/lumina_plasma.c:6518`) |
| radiation-field 생산 연속성 | `radiation_field_begin_mc()` / `radiation_field_commit()` (`lumina/radiation_field.c:117-141`, `lumina/radiation_field.c:519-531`) | 요청 generation이 기존 computed generation+1(`lumina/radiation_field.c:127-129`, `lumina/radiation_field.c:524-527`) | `-1`(`lumina/radiation_field.c:123-129`, `lumina/radiation_field.c:531`) |
| radiation-field checked view | `radiation_field_read_view()` (`lumina/radiation_field.c:706-768`) | expected generation이 nonzero이고 required=computed=expected(`lumina/radiation_field.c:729-732`) | `RADIATION_FIELD_VIEW_STALE_GENERATION`(`lumina/radiation_field.c:729-732`) |
| partition build | `population_partition_build()` (`lumina/population_contract.c:113-118`) | population required generation과 `T_e_generation`이 모두 nonzero(`lumina/population_contract.c:113-115`) | `POP_STALE_DERIVED_TEMPERATURE`(`lumina/population_contract.c:114-115`) |
| partition 소비 view | `population_partition_view_check()` (`lumina/population_contract.c:120-123`) | required=computed=requested population generation, 동일 `T_e_generation`, shell/item 수와 T/atomic hash 일치(`lumina/population_contract.c:120-123`) | `POP_STALE_DERIVED_TEMPERATURE`(`lumina/population_contract.c:121-123`) |
| transported BF field | `parity_field_built()` (`lumina/lumina_plasma.c:2372-2377`) | view status OK, `J_nu`·validity 존재, 유효 shell(`lumina/lumina_plasma.c:2372-2376`) | 호출자가 `POP_BF_STALE` 반환(`lumina/lumina_plasma.c:2391-2393`, `lumina/lumina_plasma.c:2647-2649`) |
| BF/BB rate-view 동세대 | `population_rate_views_check()` (`lumina/population_contract.c:140-152`) | BF generation=BB generation=required rate generation이고 required가 nonzero(`lumina/population_contract.c:148-151`) | `POP_STALE_DERIVED_TEMPERATURE` 또는 입력 BF/BB 상태 그대로(`lumina/population_contract.c:144-151`) |
| population transaction | `population_transaction_begin()` / `population_transaction_commit()` (`lumina/population_contract.c:211-217`) | begin의 required generation이 nonzero; caller는 committed+1을 전달(`lumina/population_contract.c:211-213`, `lumina/lumina_plasma.c:6649-6659`) | begin `-1`; commit 검증 실패는 `POP_NONFINITE` 또는 transaction 상태(`lumina/population_contract.c:212-217`) |
| NLTE population view | `nlte_solve_all()` (`lumina/lumina_plasma.c:15126-15187`) | radiation view와 line view가 모두 OK이고 같은 generation(`lumina/lumina_plasma.c:15161-15182`) | `POP_BF_STALE`, `POP_BB_STALE`, `POP_PROFILE_MISMATCH`, `POP_QUERY_HASH_MISMATCH`, 또는 `POP_STALE_DERIVED_TEMPERATURE` 후 `-1`(`lumina/lumina_plasma.c:15161-15187`) |
| A2-10 입력 publication | `a210_production_solve()` (`lumina/lumina_plasma.c:11979-12009`) | radiation view OK; opacity/emissivity nonzero; emissivity의 opacity·radiation·population generation이 현재 값과 일치(`lumina/lumina_plasma.c:11983-11992`) | 함수 반환 `0`, `blocked_stale++`; `RadeqStatus`는 직접 반환하지 않음(`lumina/lumina_plasma.c:11981-11992`) |
| A2-10 T_e transaction | `a210_solve_transaction()` (`lumina/radeq_publication.c:19-21`) | requested `T_e` generation nonzero; 성공 시 그 generation을 committed에 기록(`lumina/radeq_publication.c:19-21`) | 잘못된 인자/zero generation은 `2`; solve·schema/hash 계열 실패는 `4` 또는 `5`(`lumina/radeq_publication.c:19-21`) |

## 3. ★ 반복 0 입력 표

| 첫 수송에 연결되는 입력 | 분류 | 반복 0 시작 시 실제 출처·상태 |
|---|---|---|
| geometry, epoch | `DECK` | `geometry.csv`와 `config.json:time_explosion_s`에서 읽음(`lumina/lumina_atomic.c:947-953`, `lumina/lumina_atomic.c:991-996`) |
| `T_inner` | `DECK` | `config.json:T_inner_K`가 소유하며 기본 경로에서는 그대로 effective 값이 됨(`lumina/lumina_atomic.c:991-999`, `lumina/lumina_atomic.c:810-823`) |
| `T_e[]` 값과 generation-1 publication | `COMPUTED` | `T_inner`를 shell마다 복제하고 plasma에 복사한 뒤, manifest와 generation=1을 pre-loop에서 생성함(`lumina/lumina_atomic.c:1057-1061`, `lumina/lumina_main.c:151-157`, `lumina/lumina_plasma.c:6575-6586`) |
| seed `n_e[]` | `DECK` | `electron_densities.csv:n_e`를 읽어 plasma에 그대로 복사함(`lumina/lumina_atomic.c:1008-1012`, `lumina/lumina_main.c:146-149`) |
| rho | `DECK` | `density.csv:rho`에서 읽음(`lumina/lumina_atomic.c:1021-1023`) |
| abundance·mass·atomic level/line 자료 | `DECK` | abundance·mass와 line A/B/f/level identity를 파일에서 읽음(`lumina/lumina_atomic.c:1315-1337`, `lumina/lumina_atomic.c:1496-1508`) |
| 초기 전이확률 | `DECK` | `transition_probabilities.npy`에서 읽으며 dynamic 재계산은 `iter>0` plasma 블록 끝에서만 가능함(`lumina/lumina_atomic.c:1152-1165`, `lumina/lumina_main.c:705-710`) |
| transported canonical `J_nu`/validity/generation | `ABSENT` | pre-loop prepare가 먼저이고 owner 초기화·`bf_set_nlte_pops()`는 뒤에 있으며, 첫 commit은 packet 수송 뒤에만 있음; 요구 계약은 `parity_field_built`, 위반 상태는 `POP_BF_STALE`(`lumina/lumina_main.c:263-265`, `lumina/lumina_main.c:302-305`, `lumina/lumina_main.c:545-546`, `lumina/lumina_plasma.c:2372-2377`, `lumina/lumina_plasma.c:2647-2649`) |
| committed Z(T_e) | `ABSENT` | 후보 Z는 transaction work buffer에 계산되지만 후속 ion 실패 시 prior stamp를 복원하고 work buffer를 폐기함; 요구 계약은 `population_transaction_commit`/partition stamp(`lumina/lumina_plasma.c:6651-6675`, `lumina/lumina_plasma.c:6680-6684`) |
| solver-owned `n_e`와 ion population | `ABSENT` | `n_e` 고정점 내부의 첫 ion 계산이 transported BF view 부재로 실패하고 전체 transaction이 abort됨; 상태는 내부 `POP_BF_STALE`, 기본 외부 상태는 `POP_NE_NOT_CONVERGED`(`lumina/lumina_plasma.c:2775-2781`, `lumina/lumina_plasma.c:6711-6713`) |
| fresh LTE lower/upper level population | `ABSENT` | 이 값은 committed ion·Z로부터 `compute_tau_sobolev()` 안에서 계산되지만 transaction 실패로 해당 호출에 도달하지 못함; population transaction이 부분 publish를 차단함(`lumina/lumina_plasma.c:6778-6795`, `lumina/lumina_plasma.c:2931-2945`) |
| explicit NLTE level population | `ABSENT` | buffer 초기화 자체가 pre-loop prepare 뒤의 `nlte_init()` 경로에 있고 실제 solve는 transported views를 요구함(`lumina/lumina_main.c:263-265`, `lumina/lumina_main.c:302-305`, `lumina/lumina_plasma.c:12126-12140`, `lumina/lumina_plasma.c:15161-15187`) |
| fresh solver-owned `tau_sobolev` | `ABSENT` | 디스크 tau는 `required=1/computed=0`으로 즉시 stale 처리되고, solver tau 계산은 population commit 뒤에만 실행됨; 요구 계약은 `tau_sobolev_assert_fresh`, 위반 코드는 `-1`(`lumina/lumina_atomic.c:1122-1142`, `lumina/lumina_plasma.c:6781-6806`, `lumina/lumina_plasma.c:6508-6519`) |
| current-generation CPU opacity/emissivity publication | `ABSENT` | 발행 호출은 `iter>0`에서 T_e/plasma 계산보다 뒤에만 존재하지만 A2-10은 T_e 계산 전에 현재 radiation·population 세대와의 일치를 요구함(`lumina/lumina_main.c:641-644`, `lumina/lumina_main.c:685-702`, `lumina/lumina_plasma.c:11983-11992`) |

## 4. 끊긴 간선

| 끊긴 간선 | 요구 계약 | 관측 상태 |
|---|---|---|
| transported `J_nu` → pre-loop ion population | `parity_field_built()`(`lumina/lumina_plasma.c:2372-2377`) | `g_bf_nlte_pops==NULL`이고 view가 없으므로 `r1_use==0`; `POP_BF_STALE`(`lumina/lumina_plasma.c:7196-7197`, `lumina/lumina_plasma.c:2529-2530`, `lumina/lumina_plasma.c:2647-2649`) |
| ion population → converged `n_e` → population commit | `population_transaction_begin/commit()`(`lumina/population_contract.c:211-217`) | 내부 BF stale가 `compute_electron_density()`의 실패가 되고, 기본 경로는 `POP_NE_NOT_CONVERGED`로 transaction을 abort함(`lumina/lumina_plasma.c:2777-2781`, `lumina/lumina_plasma.c:6711-6713`) |
| committed Z/ion/level population → fresh tau | `compute_plasma_state()`의 transaction-before-tau 순서(`lumina/lumina_plasma.c:6778-6795`) | population commit 이전에 종료되어 `compute_tau_sobolev()`와 `tau_sobolev_mark_computed()`가 실행되지 않음(`lumina/lumina_plasma.c:6672-6675`, `lumina/lumina_plasma.c:6793-6806`) |
| fresh tau → 첫 transport consumer | `tau_sobolev_assert_fresh()`(`lumina/lumina_plasma.c:6508-6528`) | disk tau는 required=1/computed=0이고 solver tau도 미발행; assert에 도달하면 `-1`이지만 현재는 앞선 plasma 실패로 assert 호출 자체가 생략됨(`lumina/lumina_atomic.c:1138-1142`, `lumina/lumina_plasma.c:6602-6618`) |
| 현재 radiation generation → 같은 세대 opacity/emissivity → `T_e` | `a210_production_solve()`(`lumina/lumina_plasma.c:11979-12009`) | 새 radiation commit/read는 T_e보다 앞이고 opacity/emissivity 발행은 T_e보다 뒤이므로, T_e 호출 시 현재 radiation generation과 일치하는 publication이 없음(`lumina/lumina_main.c:545-546`, `lumina/lumina_main.c:641-644`, `lumina/lumina_main.c:685-702`) |
| qualified `T_e_generation` → Z/population | `compute_plasma_state()` 및 `population_partition_build()`(`lumina/lumina_plasma.c:6641-6646`, `lumina/population_contract.c:113-118`) | A2-10이 stale로 `0`을 반환하면 main이 generation을 0으로 만들고, 다음 plasma 호출은 `POP_INVALID_TE`로 종료함(`lumina/lumina_main.c:651-658`, `lumina/lumina_plasma.c:6641-6646`) |

## 5. 제시된 네 사실 검증

| 사실 | 판정 | 코드 근거 |
|---|---|---|
| `lumina_prepare_solver_owned_tau`는 반복 앞에서 호출되고 그 안의 plasma 계산은 published `T_e`를 요구한다 | **[실측] 확인** | 호출은 루프 전 `lumina/lumina_main.c:263-265`; 내부 호출은 `lumina/lumina_plasma.c:6594-6618`; `T_e_generation==0` 거부는 `lumina/lumina_plasma.c:6641-6646` |
| 세대를 올리는 곳은 루프 안뿐이며 RADEQ qualification 뒤에만 오른다 | **[실측] 반박** | 루프 안의 RADEQ 증가는 qualification 뒤에만 일어나는 점은 맞지만(`lumina/lumina_main.c:651-656`), pre-loop `lumina_publish_seed_te()`가 별도로 generation 1을 발행하므로 “루프 안뿐”은 틀림(`lumina/lumina_main.c:259-261`, `lumina/lumina_plasma.c:6581-6586`) |
| `parity_field_built`는 OK view와 `J_nu`를 요구한다 | **[실측] 확인** | status OK, `J_nu`, validity 및 shell 범위를 모두 검사함(`lumina/lumina_plasma.c:2372-2377`) |
| 2347 주석은 LTE-Saha fallback을 말하지만 2644–2649는 Saha 값을 버리고 stale을 반환한다 | **[실측] 확인** | 주석은 “Fail-closed to B2 LTE-Saha pin”이라고 명시함(`lumina/lumina_plasma.c:2341-2347`); 실제 코드는 `(void)phi_neb` 후 `!r1_use`에서 `POP_BF_STALE`을 반환함(`lumina/lumina_plasma.c:2643-2649`) |