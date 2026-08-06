[실측] 결론은 “계약 수리”가 아니라 “공급자·호출 위상·세대 소유자 수리”다. 0층 계약 10건은 모두 유지한다. E는 D를 반박하지 않았지만, 두 팔의 일치는 차분 진단일 뿐 최종 인증이 아니라고 확정한다([OUT_E:12–14](/tmp/claude-10396/codex_wiring/OUT_E_fable_certification.md:12), [OUT_E:128–146](/tmp/claude-10396/codex_wiring/OUT_E_fable_certification.md:128)). 따라서 아래 목표에서는 결정론 팔을 상태 갱신용 장 소유자로 고정하고 MC 팔을 동세대 shadow로 둔다. 최종 심판은 외부 CMFGEN이다.

## 1. 생존 함수 목록

`A2-04…10`은 `A2-00` 아래의 구체 간선 계약으로 표기했다.

| 함수 | 파일:행 | 무엇을 계산하는가 | 보증 계약 | 왜 생존인가 |
|---|---|---|---|---|
| `[실측] load_tardis_reference_data` | [lumina_atomic.c:896](/tmp/claude-10396/codex_wiring/lumina/lumina_atomic.c:896) | 기하·밀도·원소 조성·원자자료의 단위/형상/식별자를 적재한다. | `C2-EXEC`, `K-SHAPE`, `D-BUILD`, `CONFIG-PREC`, `A2-00` | 조성 항등식과 배열 형상을 보존한다. `T_e=T_inner` 생성과 덱 전이확률 소비는 이 함수의 생존 범위에서 제외하고 아래 수리 대상으로 분리한다. |
| `[실측] lumina_publish_seed_te` | [lumina_plasma.c:6554–6592](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:6554) | 양의 유한한 seed \(T_e\)의 manifest를 만들고 `T_e_generation:0→1`을 한 번만 발행한다. | `GEN-GUARD`, `TE-DEAD`, `A2-00` | 첫 물질 상태에 필요한 선언적 온도 출처다. 첫 복사장 commit에서 seed 권한이 폐기된다([radiation_field.c:604–609](/tmp/claude-10396/codex_wiring/lumina/radiation_field.c:604)). 프로파일 공급자는 별도 수리한다. |
| `[실측] population_partition_build`, `population_partition_view_check` | [population_contract.c:113–123](/tmp/claude-10396/codex_wiring/lumina/population_contract.c:113) | \(Z_i(T_e)=\sum_l g_l e^{-(E_l-E_0)/kT_e}\)와 `T_e/atomic/population-generation` stamp를 계산·검증한다. | `GEN-GUARD`, `A2-00` | 통계역학식과 manifest 동세대 검사가 있다. 단, 빈 최상단 이온을 `Z=1`로 만드는 하위 함수는 수리 대상이다. |
| `[실측] population_lte_level_fraction` | [population_contract.c:125–138](/tmp/claude-10396/codex_wiring/lumina/population_contract.c:125) | LTE Boltzmann 준위분율 \(g_l e^{-\Delta E/kT}/Z\)을 계산한다. | `GEN-GUARD`, `A2-00` | 양성·정규화된 partition을 요구하고 비유한값을 `POP_NONFINITE`로 거부한다. 부트스트랩과 최종 \(\tau\) 계산에 모두 자리가 있다. |
| `[실측] population_transaction_begin/commit/abort` | [population_contract.c:211–217](/tmp/claude-10396/codex_wiring/lumina/population_contract.c:211) | 이온·준위·\(n_e\)·partition을 작업 사본에서 계산한 뒤 원자적으로 발행한다. | `GEN-GUARD`, `A2-00` | 실패 시 공개 상태를 보존하고, 유한성 확인 후에만 committed generation을 진행시킨다. |
| `[실측] nlte_bf_gamma_canonical`, `bf_rate_gamma_from_view` | [lumina_plasma.c:453–535](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:453), [bf_rate_jnu.c:23–128](/tmp/claude-10396/codex_wiring/lumina/bf_rate_jnu.c:23) | 검사된 \(J_\nu\)와 단면으로 photoionization 적분을 계산한다. | `GEN-GUARD`, `A2-05⊂A2-00` | 구간별 선형 단면 적분이며 `STALE/UNSAMPLED/OOG/EXACT_ZERO`를 구분한다. stale에서 값을 제조하지 않는다. |
| `[실측] parity_rate_se_ratio_checked` | [lumina_plasma.c:2446–2491](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:2446) | canonical BF rate와 재결합률로 인접 이온의 rate-SE 비를 계산한다. | `GEN-GUARD`, `Z-INERT`, `A2-07⊂A2-00` | 분자·분모·\(n_e\)를 검사하고 0/0, 비유한값, rank 부족을 실패 폐쇄한다. 반복 중 Saha fallback을 하지 않는 부분은 그대로 유지한다. |
| `[실측] lumina_zinert_element_inactive`, `lumina_zinert_validate` | [lumina_plasma.c:1933–2101](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:1933) | 비활성 원소의 물질·전이 기여가 정확히 0인지 검증한다. | `Z-INERT`, `D-BUILD` | 존재하지 않는 원소가 전하·불투명도·전이확률을 오염시키지 않는 보존 경계다. |
| `[실측] compute_tau_sobolev`, `tau_sobolev_require_refresh/mark_computed/assert_fresh` | [lumina_plasma.c:2843–2956](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:2843), [lumina_plasma.c:6488–6528](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:6488) | 준위분율과 \(f_{lu}\), 팽창시간으로 Sobolev \(\tau\)를 계산하고 required/computed generation을 검사한다. | `K-FRESH`, `GEN-GUARD`, `A2-08⊂A2-00` | Sobolev 식에 근거하며 덱의 오래된 \(\tau\)가 첫 수송에 들어가는 것을 막는다. 문제는 함수가 아니라 반복 0 공급자의 부재다. |
| `[실측] compute_bf_opacity` | [lumina_plasma.c:7319](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:7319) | 발행된 준위·이온 상태로 BF 흡수/방출 격자를 만든다. | `D-BUILD`, `Z-INERT`, `A2-08⊂A2-00` | population과 원자자료에서 다시 계산되며 signed-opacity 발행의 실제 공급자다. |
| `[실측] single_packet_loop` 인터페이스, `radiation_field_accumulator_add/reduce` | [lumina.h:893](/tmp/claude-10396/codex_wiring/lumina/lumina.h:893), [lumina_main.c:484–531](/tmp/claude-10396/codex_wiring/lumina/lumina_main.c:484), [radiation_field.c:170–207](/tmp/claude-10396/codex_wiring/lumina/radiation_field.c:170) | packet 경로와 comoving path-length 추정자를 축적·reduce한다. | `C2-EXEC`, `H-TRANSFORM`, `A2-04/06⊂A2-00` | MC 추정식과 reduce 경로가 사슬에 있다. 단, 발췌에는 `single_packet_loop` 본체가 없으므로 생존 판정은 인터페이스·호출 배선 범위다. |
| `[실측] radiation_field_begin_mc`, `radiation_field_commit`, `radiation_field_read_view`, `radiation_field_line_jbar_view` | [radiation_field.c:117–141](/tmp/claude-10396/codex_wiring/lumina/radiation_field.c:117), [radiation_field.c:519–662](/tmp/claude-10396/codex_wiring/lumina/radiation_field.c:519), [radiation_field.c:706–811](/tmp/claude-10396/codex_wiring/lumina/radiation_field.c:706) | 복사장의 요청세대=`computed+1`을 강제하고 \(J_\nu,\bar J\)를 원자적으로 발행·조회한다. | `GEN-GUARD`, `TE-DEAD`, `A2-00` | 단위·frame·epoch·shell·q-set·profile identity를 검사한다. 두 팔도 이 동일 commit choke point를 쓸 수 있다. |
| `[실측] formal_solve_bin`, `cmfgen_solve_J` | [lumina_cmfgen.c:2404–2525](/tmp/claude-10396/codex_wiring/lumina/lumina_cmfgen.c:2404), [lumina_cmfgen.c:2528](/tmp/claude-10396/codex_wiring/lumina/lumina_cmfgen.c:2528) | 고정된 \(\chi,\eta\)에서 결정론 formal transfer로 \(J_\nu\)를 계산한다. | `H-TRANSFORM`, `A2-00` | 복사수송 방정식과 광학깊이 극한 검사가 있다. 조립 source의 CMFGEN 일치는 별도 최종 인증 대상이다. |
| `[실측] a208_publish_cpu_opacity`, `a209_publish_cpu_emissivity` | [lumina_plasma.c:8036–8160](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:8036) | signed \(\chi_{\rm es,bb,bf,ff}\), \(\eta_{\rm bb,bf,ff}\)와 그것이 읽은 field/population/\(T_e\)/\(\tau\) 세대를 발행한다. | `K-FRESH`, `GEN-GUARD`, `A2-08/09⊂A2-00` | 계산·검증·원자 commit 본체는 살아 있다. 현재 잘못된 것은 호출 시각과 pure lane의 a209 호출 부재다. |
| `[실측] a210_production_residual`, `a210_solve_transaction`, `compute_radiative_equilibrium_te` | [lumina_plasma.c:11917–12020](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:11917), [radeq_publication.c:20–21](/tmp/claude-10396/codex_wiring/lumina/radeq_publication.c:20) | 동일 세대 \(J,\chi,\eta\)의 가열–냉각 잔차를 풀고 적격 \(T_e\)만 원자 발행한다. | `GEN-GUARD`, `TE-DEAD`, `A2-10⊂A2-00` | 에너지 보존식·선항 소유권·잔차 대장이 있다. root가 없으면 값을 clamp하지 않고 실패한다. |
| `[실측] nlte_solve_all`, `nlte_update_tau_sobolev` | [lumina_plasma.c:15126–15627](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:15126), [lumina_plasma.c:14873](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:14873) | 동세대 BF/BB rates로 SE 준위계수를 풀어 population transaction을 commit하고 새 \(\tau\)를 만든다. | `GEN-GUARD`, `Z-INERT`, `K-FRESH`, `A2-07⊂A2-00` | rank·비유한값·field/line generation·q/profile을 검사한다. 결정론 \(\bar J\) 공급만 연결되면 사슬에 그대로 들어간다. |
| `[실측] cmf_q_at_z`, `cmfgen_write_spectrum_obs` | [lumina_cmfgen.c:2965–2973](/tmp/claude-10396/codex_wiring/lumina/lumina_cmfgen.c:2965), [lumina_cmfgen.c:3212](/tmp/claude-10396/codex_wiring/lumina/lumina_cmfgen.c:3212) | comoving 해를 observer-frame spectrum으로 변환한다. | `H-TRANSFORM` | frame 변환을 한 소유자에게 모으며 최종 CMFGEN 스펙트럼 비교 지점에 자리한다. |

## 2. 구현할 함수 목록

| 순서 | 함수(신설/수리) | 무엇을 공급하는가 | 없어서 끊긴 간선 | 의존하는 것 |
|---:|---|---|---|---|
| 0 | `[실측→수리] inject_topstage_continuum_levels` | 모든 원소의 최상단 이온에 정본 바닥준위 \(E_0,g_0\); offset/hash/NLTE mapping을 만든 뒤가 아니라 만들기 전에 삽입 | atomic model → partition/SE. 현재 O IV 한 개, 기본 off이며 offset이 stale([lumina_atomic.c:2676–2728](/tmp/claude-10396/codex_wiring/lumina/lumina_atomic.c:2676)); ARTIS는 전 원소에 single-level top ion([artis/input.cc:1226–1234](/tmp/claude-10396/codex_wiring/artis/input.cc:1226)) | 원소별 이온목록과 정본 ground degeneracy. 환경 스위치 없이 atomic-model 불변식으로 실행 |
| 1 | `[실측→신설] build_bootstrap_te_profile` | 한 번만 쓰는 양의 셸별 seed \(T_e\)와 provenance | geometry/boundary → seed \(T_e\). 현재 전 셸 `T_inner` 복제는 비물리([lumina_atomic.c:1058–1061](/tmp/claude-10396/codex_wiring/lumina/lumina_atomic.c:1058)) | 기하와 내부경계 조건. CMFGEN 답이나 `plasma_state.csv`를 주입하지 않으며 노브·floor가 아님. 구체식 선택은 현 자료만으로 확정 불가 `[추정]` |
| 2 | `[실측→신설] bootstrap_lte_saha_state` | seed \(T_e\)에서 partition, 전하중성 \(n_e\), Saha 이온, LTE 준위와 population generation 1 | seed \(T_e\) → 첫 \(\tau/\chi/\eta\). 계산한 `phi_neb`를 버리고 `POP_BF_STALE` 반환([lumina_plasma.c:2643–2650](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:2643)) | 수리된 최상단 준위, 조성, `population_transaction_*`. 선언적 exactly-once 초기조건이며 반복 중 fallback이 아님 |
| 3 | `[실측→수리] population_partition_ion` | 실제 바닥준위에서 얻은 top-ion partition | top ion → \(Z\). 현재 `hi==lo`에서 임의 `Z=1`([population_contract.c:92–105](/tmp/claude-10396/codex_wiring/lumina/population_contract.c:92)) | 순서 0. 수리 뒤 빈 이온은 `POP_ATOMIC_MISSING`; 값을 합성하지 않음 |
| 4 | `[실측→수리] compute_electron_density`, `compute_plasma_state` | 전하보존 잔차가 장부화된 \(n_e\), 마지막 \(n_e\)에서 다시 계산한 이온분율; `Z→n_e→ions`만 commit | \(Z\) → charge-consistent ion state. 현재 5% 기준 후 감쇠값과 직전 이온분율을 함께 발행([lumina_plasma.c:2752–2811](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:2752)) | rate-SE 또는 bootstrap-Saha 공급자. 수렴 실패는 `POP_NE_NOT_CONVERGED`; clamp/floor 금지 |
| 5 | `[실측→신설] twoarm_field_epoch_begin`, `twoarm_commit_barrier` | 동일 immutable material manifest와 동일 논리세대 \(r\)를 읽는 `MC`/`DET` owner 두 슬롯 | 한 상태 → 독립 두 장 → 동세대 비교. CPU에는 한 owner/한 팔만 있음([lumina_main.c:429–557](/tmp/claude-10396/codex_wiring/lumina/lumina_main.c:429)) | `radiation_field_begin/commit`; 각 owner는 자기 `computed+1`, barrier는 material/q/profile identity 일치를 요구 |
| 6 | `[실측→신설] deterministic_line_jbar` 및 `[수리] cmfgen_commit_jnu` | formal intensity에서 같은 q-set hash·profile identity의 결정론 \(\bar J_l\), \(J_\nu\)와 한 commit | DET formal solve → line view → SE. 현재 raw `jbar_line_det`는 canonical owner에 발행되지 않아 `POP_BB_STALE`([lumina_cmfgen.c:5193–5208](/tmp/claude-10396/codex_wiring/lumina/lumina_cmfgen.c:5193), [lumina_plasma.c:15168–15177](/tmp/claude-10396/codex_wiring/lumina/lumina_plasma.c:15168)) | 동일 q-set, profile ID/hash, formal intensity. 보간 floor나 MC 값 대입 금지 |
| 7 | `[실측→수리] main`, `cmfgen_run`의 material-update orchestration | commit barrier 직후, 어떤 물질 mutation보다 먼저 a208+a209를 발행하고 그 다음 A2-10 호출 | field → a208/a209 → \(T_e\). 현재 MC는 `commit→T_e→publish`([lumina_main.c:545–702](/tmp/claude-10396/codex_wiring/lumina/lumina_main.c:545)); pure lane은 a209 없음([lumina_cmfgen.c:5157–5208](/tmp/claude-10396/codex_wiring/lumina/lumina_cmfgen.c:5157)) | 순서 5·6, 기존 a208/a209/A2-10 함수. pure lane도 a209를 동일하게 실행 |
| 8 | `[실측→수리] T_e generation commit 호출부` | 적격 A2-10 transaction만 `t→t+1`; 비적격이면 이전 committed \(T_e,t\)를 그대로 유지하고 다음 물질 갱신은 차단 | A2-10 → partition. 현재 비적격 시 `T_e_generation=0`, 적격 시 transaction 밖에서 다시 증가([lumina_main.c:645–655](/tmp/claude-10396/codex_wiring/lumina/lumina_main.c:645)) | `a210_solve_transaction` 단일 세대 소유자. 실패를 이전 온도로 “계속 진행”하는 fallback은 허용하지 않음 |
| 9 | `[실측→수리] compute_transition_probabilities` + `[신설] transition_probability_publish` | 새 population/\(\tau\) manifest에 결박된 전이확률과 generation | 최종 준위·\(\tau\) → 다음 수송. 현재 덱 NPY가 기본이고 재계산은 opt-in([lumina_atomic.c:1152–1165](/tmp/claude-10396/codex_wiring/lumina/lumina_atomic.c:1152), [lumina_main.c:705–710](/tmp/claude-10396/codex_wiring/lumina/lumina_main.c:705)) | 현 함수의 물리 계산 핵심, `Z-INERT`, 새 freshness stamp. `void` 실패 은폐를 상태 반환으로 바꿈 |
| 10 | `[실측→신설] compare_twoarm_with_cmfgen` | 셸×대역 및 line별 \(J^{MC},J^{DET},J^{CMFGEN}\) 3열 오차지도 | 두 팔 일치 → 물리 인증. E가 공통 source/grid/deck bias는 두 팔 차분에서 상쇄된다고 확인([OUT_E:46–65](/tmp/claude-10396/codex_wiring/OUT_E_fable_certification.md:46)) | 동일 observer transform·shell/frequency mapping과 외부 CMFGEN 결과. 두 팔 일치를 pass/fail 물리판정으로 승격하지 않음 |

[실측] `solve_radiation_field`와 `coupled_*` no-op 호출은 생존 함수도 새 물리 공급자도 아니다. 순서 7에서 호출 간선을 제거하고 실제 formal/A2-10 경로만 남긴다. 새 환경 노브는 하나도 추가하지 않는다.

## 3. 통합 배선도

`S#`는 위 생존 함수, `R#`는 구현 목록 순서다. `M_m={T_e(t),Z,n_e,ions,levels,pop(m),tau(k),chi,eta,transition(p)}`는 수송 중 불변인 물질 manifest다.

```text
반복 0 — 선언적 부트스트랩(exactly once)
────────────────────────────────────────────────────────────────────────────
 deck/atomic/geometry
       │ E01: C2-EXEC·K-SHAPE·D-BUILD·CONFIG-PREC
       v
 S1 load_tardis_reference_data
       │
       +--> R0 all-element top-ion ground levels
       │      (offset/hash/NLTE map 전에 구성)
       │
       +--> R1 physical seed-T_e profile
              │ E02: GEN-GUARD·TE-DEAD
              v
          S2 lumina_publish_seed_te              t: 0 → 1
              │ E03: A2-00, exactly-once
              v
          R2 bootstrap_lte_saha_state
              │  Z → charge-neutral n_e → ions → LTE levels
              │  S4 population transaction       m: 0 → 1
              v
          S8 tau + S9 BF/FF chi,eta
              │ tau mark                          k: 0 → 1
              v
          R9 transition_probability_publish       p: 0 → 1
              │ E04: K-FRESH/assert material manifest
              v
          S10 DET formal solve
              │ J^DET_nu
              +--> R6 deterministic line-Jbar (same q/profile)
              │
              v
          S11 radiation_field_commit[DET]          r: 0 → 1
              │ E05: field commit 직후, 물질 변경 전
              v
          S12 a208 + a209 (pure lane도 둘 다)       o=e=r=1
              │
              v
          S13 A2-10 T_e transaction                t: 1 → 2
              │
              v
          S3/S5/R4 rate-SE material update
              └───────────────> 첫 완전 M_m ───────────────┐
                                                           │
반복 r — 두 팔의 공진화                                     │
───────────────────────────────────────────────────────────┘
                     immutable M_m
        {same pop,Te,n_e,tau,chi,eta,transition manifest}
                          │
              R5 twoarm_field_epoch_begin
                          │
                 ┌────────┴────────┐
                 │                 │
       MC ARM    │                 │    DET ARM (상태장 소유자, 고정)
                 v                 v
       S6 packet transport    S10 formal transport
                 │                 │
       S6/S7 path & line      R6 deterministic
       estimators             J_nu + line-Jbar
                 │                 │
                 v                 v
       S11 commit[MC,r]       S11 commit[DET,r]
       canonical schema       canonical schema
                 │                 │
                 └────────┬────────┘
                          v
             R5 commit barrier / arm comparison
        same r + same M hash + same q-set/profile required
                          │
                          │ DET view is fixed state owner;
                          │ MC view writes only shadow/diagnostics
                          v
       S12 a208 + a209, immediately after commit barrier
                          │ o/e bind {r,m,t,k}
                          v
       S13 A2-10 heating=cooling transaction
                          │ success only: t → t+1
                          v
       S3 partition build/view                     Z
                          │
                          v
       R4 charge-conserving n_e iteration           n_e
                          │
                          v
       S5 rate-SE ion ladder + S4 atomic commit     ions, m → m+1
                          │
                          v
       S14 NLTE SE level transaction                levels, m+1 → m+2
                          │
                          v
       S8 final Sobolev tau + S9 BF/FF chi,eta      k → k+1
                          │
                          v
       R9 mandatory transition probabilities       p → p+1
                          │
                          └────────────> next immutable M_(m+2)

       Field[MC,r] ───────┐
       Field[DET,r] ──────┼── R10 three-column map ── 외부 CMFGEN (최종 심판)
       CMFGEN reference ──┘
```

[추정] “같은 곳에 쓴다”는 두 팔이 같은 메모리를 덮어쓴다는 뜻이 아니다. 둘 다 동일한 `radiation_field_commit()` 스키마와 세대 규칙에 쓰되 `Field[MC]`, `Field[DET]` 슬롯은 분리한다. 그래야 MC가 결정론 장을 덮어쓰지 않으면서 동세대 차분이 가능하다.

### 간선표

| ID | 간선 | 지키는 계약 | 위반 시 상태코드 |
|---|---|---|---|
| E01 | 입력 → 정형 atomic/material model | `C2-EXEC`, `K-SHAPE`, `D-BUILD`, `CONFIG-PREC` | `[실측] -1/EXIT_FAILURE`; obsolete scalar·unknown env는 loader가 차단 |
| E02 | top levels/geometry → seed \(T_e\) 발행 | `GEN-GUARD`, `TE-DEAD`, `A2-00` | `[실측] POP_ATOMIC_MISSING`, `POP_INVALID_TE`; 재발행은 `POP_FORBIDDEN_FALLBACK`로 표면화 |
| E03 | seed \(T_e\) → bootstrap Saha state | `GEN-GUARD`, `A2-00` | `[실측] POP_INVALID_PARTITION`, `POP_NE_NOT_CONVERGED`, `POP_NONFINITE`, `POP_SOLVE_FAILED`; `[신설] BOOTSTRAP_REENTRY` |
| E04 | LTE levels → \(\tau,\chi,\eta,\) transition → 첫 수송 | `K-FRESH`, `Z-INERT`, `A2-00` | `[실측] A208_INVALID_TE`, `A208_INVALID_POPULATION`, `A208_NONFINITE`, freshness `-1`; `[신설] TRANSITION_STALE` |
| E05 | 첫 DET solve → \(J_\nu,\bar J\) generation 1 | `H-TRANSFORM`, `GEN-GUARD`, `A2-05/06` | `[실측] commit -1`, `RADIATION_FIELD_VIEW_STALE_GENERATION`, `LINE_JBAR_VIEW_QHASH`, `LINE_JBAR_VIEW_PROFILE` |
| E06 | immutable `M_m` → MC transport | `C2-EXEC`, `H-TRANSFORM`, `K-FRESH`, `D-BUILD` | `[실측] accumulator -1`, freshness -1; `[신설] TRANSITION_STALE`, `MATERIAL_MANIFEST_STALE` |
| E07 | MC estimator → `Field[MC,r]` | `GEN-GUARD`, `A2-04/06` | `[실측] commit -1`, `RADIATION_FIELD_VIEW_UNITS_FRAME`, `…EPOCH_SHELLS`, `…STALE_GENERATION` |
| E08 | immutable `M_m` → DET formal solve | `H-TRANSFORM`, `K-FRESH`, `A2-08/09` | `[실측]` 음의/비유한 opacity는 CMF lane return 3; freshness -1 |
| E09 | DET \(J_\nu,\bar J\) → `Field[DET,r]` | `GEN-GUARD`, `A2-05/06` | `[실측] LINE_JBAR_VIEW_STALE_GENERATION`, `…QHASH`, `…PROFILE`; commit -1 |
| E10 | 두 field commit → barrier | `GEN-GUARD`, `A2-00` | `[신설] TWOARM_GENERATION_MISMATCH`, `TWOARM_MATERIAL_MISMATCH`, `TWOARM_QPROFILE_MISMATCH` |
| E11 | DET field + 같은 `M_m` → a208/a209 | `K-FRESH`, `GEN-GUARD`, `A2-08/09` | `[실측]` a208/a209 return 2(할당), 3(stale), 5(invalid/partial commit) |
| E12 | `{J,χ,η,r,m,t,k}` → A2-10 \(T_e\) | `TE-DEAD`, `GEN-GUARD`, `A2-10` | `[실측] RADEQ_STALE_RF/BF/LINE/POP/OPACITY/EMISSIVITY`, `RADEQ_TERM_SCHEMA`, `RADEQ_NO_BRACKET`, `RADEQ_HEAT_RESIDUAL`, `RADEQ_NONFINITE` |
| E13 | 적격 \(T_e(t+1)\) → partition \(Z\) | `GEN-GUARD`, `A2-00` | `[실측] POP_INVALID_TE`, `POP_INVALID_PARTITION`, `POP_STALE_DERIVED_TEMPERATURE`, `POP_ATOMIC_MISSING` |
| E14 | \(Z,J,n_e\) → charge-consistent ions | `Z-INERT`, `GEN-GUARD`, `A2-07` | `[실측] POP_BF_STALE/UNSAMPLED/OOG/MISS`, `POP_NE_NOT_CONVERGED`, `POP_RANK_INCOMPLETE`, `POP_NONFINITE` |
| E15 | ions → atomic population commit | `GEN-GUARD`, `A2-00` | `[실측] POP_SOLVE_FAILED`, `POP_NONFINITE`; 실패 시 generation 불변 |
| E16 | DET \(J_\nu,\bar J\) + ions → NLTE levels | `GEN-GUARD`, `A2-06/07` | `[실측] POP_BF_STALE`, `POP_BB_STALE`, `POP_PROFILE_MISMATCH`, `POP_QUERY_HASH_MISMATCH`, `POP_RANK_INCOMPLETE` |
| E17 | levels → final \(\tau,\chi,\eta\) | `K-FRESH`, `Z-INERT`, `A2-08/09` | `[실측] A208_INVALID_POPULATION`, `A208_SOURCE_CANCELLATION_SINGULAR`, `A208_NONFINITE`, freshness -1 |
| E18 | final state → transition probabilities | `Z-INERT`, `GEN-GUARD`, `A2-00` | `[신설] TRANSITION_STALE`, `TRANSITION_NONFINITE`, `TRANSITION_NORMALIZATION_FAILED` |
| E19 | 두 팔 + 외부 CMFGEN → 인증지도 | `H-TRANSFORM`, `A2-00` | `[실측]` 현 runtime enum 없음; `[신설] CERT_ARM_MISMATCH`, `CERT_CMFGEN_MISMATCH`. 후자만 물리판정 실패 |

### 세대 장부

| 장부 | 증가 지점 | 그것을 요구하는 소비자 |
|---|---|---|
| `[실측] T_e generation t` | seed publisher가 `0→1`; 이후 A2-10 transaction 성공만 `t→t+1` | partition stamp, plasma/population transaction, a208/a209 |
| `[실측+수리] radiation generation r` | 각 arm owner의 `requested=computed+1`; barrier가 두 arm의 같은 논리 `r` 확인 | checked field view, a209, A2-10, BF/BB rate-SE, NLTE |
| `[실측] population generation m` | bootstrap commit; steady 상태에서 ion commit과 NLTE-level commit이 각각 한 번 진행 | partition stamp, \(\tau\), BF opacity, a208/a209, 다음 수송 manifest |
| `[실측] tau generation k` | material mutation이 required를 증가; final level population 뒤 계산 성공 시 computed가 따라감 | MC/DET 수송, a208, line source |
| `[실측] opacity/emissivity generation o/e` | field barrier 직후 a208/a209 atomic commit | A2-10의 동세대 삼중항 |
| `[신설] transition generation p` | 최종 population·\(\tau\) 뒤 필수 재계산 성공 시 증가 | 다음 MC/DET 수송; 덱 NPY는 generation 0 seed로도 소비 불가 |

### 이 배선도가 서면 무엇이 처음으로 가능해지는가

- 반복 0에서 물리적인 LTE(Saha) 물질 상태가 solver-owned \(\tau/\chi/\eta\)를 공급해 첫 결정론 수송에 도달한다.
- 같은 물질·세대·q/profile을 읽은 \(J^{MC}\)와 \(J^{DET}\)를 셸·대역·선별로 직접 분리해 비교할 수 있다.
- \(T_e\rightarrow Z\rightarrow n_e\rightarrow\) 이온\(\rightarrow\)준위\(\rightarrow\tau\rightarrow\)전이확률의 폐합 결과를 외부 CMFGEN으로 처음 물리 인증할 수 있다.