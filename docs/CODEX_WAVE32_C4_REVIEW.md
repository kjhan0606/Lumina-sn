# Codex C4 — A4 패치 리뷰

## 최종 판정: **FAIL**

패치 정적 판정은 **PASS 2건 / FAIL 4건 / UNRESOLVED 0건**입니다. C3의 OOM 전파와 D6형 감사 구조는 개선됐지만, 잔존 경쟁, iteration=10 우회, I/O 오류 누락, 보존행 감사의 NaN fail-open 때문에 FAIL 좌표가 전부 해소되지는 않았습니다.

| 패치 | 판정 | 핵심 결과 |
|---|---|---|
| rung1 RC 의미론 | **FAIL** | verdict/RC 분리는 됐지만 artifact write/close 오류 누락 |
| rung2 atomic | **FAIL** | 지정 8개는 atomic이나 `target_fail++` 경쟁 잔존 |
| rung3 OOM | **PASS** | CPU/GPU 상위 전파 및 legacy 반환값 폐기 제거 |
| rung4 writer/consumer | **FAIL** | checksum/flag는 정확하나 iteration=10 계약 우회 가능 |
| rung5 M_V 감사 | **FAIL** | 주요 구조는 개선됐지만 `b[0]=NaN` 감사 fail-open |
| rung6 전수 검사 | **PASS** | 신규 물리 clamp/floor/cap 및 D6형 동어반복 없음 |

### 1. rung1 RC 의미론 — **FAIL**

판정 상태와 운영 오류의 분리는 대부분 올바릅니다.

- topology/numerical/boundary gate를 모두 `pass`에 포함하고 별도 `verdict_pass_out`으로 전달합니다:  
  > `int pass=topology_gate_pass&&numerical_gate_pass&&boundary_gate_pass;`  
  > `*verdict_pass_out = pass;`  
  [rung1:85](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung1_rc_semantics.patch:85)

- 완결된 `EW_PASS`, `SCOPE_FAIL`, gate FAIL은 모두 RC 0으로 귀결됩니다:  
  > `return 0;`  
  [rung1:119](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung1_rc_semantics.patch:119)

- config 오류는 RC -1이고, production caller도 이를 중단으로 전파합니다: [rung1:15](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung1_rc_semantics.patch:15), [rung1:170](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung1_rc_semantics.patch:170).

하지만 I/O 오류 폐합이 불완전합니다. `ew_open_dump()`는 `fopen()` 실패만 `ew_dump_failed`에 반영합니다: [rung1:62](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung1_rc_semantics.patch:62). 실제 solution/diagnostics/provenance artifact들은 `fprintf()` 결과와 `fclose()` 결과를 무시합니다: [rung5:218](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:218), [rung5:221](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:221), [rung5:223](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:223). 따라서 open 성공 뒤 write/flush/close가 실패하면 `ew_dump_failed`가 0인 채 RC 0과 commit 경로가 가능하며, “I/O 오류=RC≠0”은 완전하지 않습니다.

추가로 OOM fixture는 실제 production allocator를 실패시키지 않고 wrapper에서 즉시 `-1`을 반환하므로, production OOM 분류 자체의 음성대조는 아닙니다: [rung1:293](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung1_rc_semantics.patch:293).

### 2. rung2 atomic — **FAIL**

지정된 8개 카운터 자체는 모두 atomic update로 바뀝니다.

- runtime 3개 및 atomic snapshot: [rung2:6](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung2_atomic_counters.patch:6)-[47](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung2_atomic_counters.patch:47)
- `kramers_fallback`, `continuum_deleted`: [rung2:69](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung2_atomic_counters.patch:69)-[93](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung2_atomic_counters.patch:93)
- bf estimator/pref-J/JEQB 3개: [rung2:97](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung2_atomic_counters.patch:97)-[159](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung2_atomic_counters.patch:159)

그러나 같은 공유 `ew_cap` 경로에 plain increment가 남습니다:

> `ew_cap.target_fail++;`

[rung2:83](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung2_atomic_counters.patch:83)

바로 인접한 `continuum_deleted`에는 atomic이 필요하다고 수정하면서 `target_fail`은 보호하지 않았으므로 잔존 경쟁입니다. Selftest도 runtime 3개만 호출해 이 결함과 나머지 다섯 카운터를 검증하지 않습니다: [rung2:199](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung2_atomic_counters.patch:199)-[219](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung2_atomic_counters.patch:219).

### 3. rung3 OOM 전파 — **PASS**

C3의 두 차단점이 패치상 해소됩니다.

- legacy wrapper가 `void` 및 `(void)` 폐기에서 `int` 반환으로 바뀝니다: [rung3:7](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:7)-[17](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:17).
- CPU `nlte_solve_all()`이 `int`가 되고 within-SL 실패를 `-1`로 반환합니다: [rung3:19](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:19)-[30](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:30).
- CPU caller는 main, CMFGEN, fixture/test 경로에서 모두 반환값을 검사합니다: [rung3:156](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:156)-[168](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:168), [rung3:338](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:338)-[365](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:365).
- GPU solve 역시 `int`로 바뀌고 within-SL OOM을 반환하며, CPU fallback과 다섯 GPU 호출 지점에서 최상위 `EXIT_FAILURE`까지 전파합니다: [rung3:178](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:178)-[207](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:207), [rung3:246](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:246)-[334](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch:334).

### 4. writer/consumer 계약 — **FAIL**

Writer는 기대값을 강제하지 않고 전달받은 실제 metadata를 기록하도록 개선됐습니다. Binary header와 sidecar 모두 `field_generation` 및 `post_damping` 인자를 그대로 씁니다: [rung4:6](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:6)-[29](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:29), [rung4:31](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:31)-[44](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:44).

Consumer도 다음은 정확합니다.

- checksum을 payload 전체 SHA-256과 비교: [rung4:375](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:375)
- unknown/coherent/frequency/post-damp flag 검사: [rung4:312](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:312)-[328](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:328)
- η total과 두 독립 저장 성분을 bitwise 비교하므로 항등형이 아님: [rung4:342](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:342)-[355](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:355)

그러나 iteration=10은 고정 계약이 아니라 기본 인자일 뿐입니다:

> `parser.add_argument("--expected-iteration", type=int, default=10)`  
> `parser.add_argument("--allow-pre-damp", action="store_true")`

[rung4:381](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:381)-[393](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:393)

검사는 상수 10이 아니라 호출자가 준 `expected_iteration`과 비교합니다: [rung4:296](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:296)-[325](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch:325). 따라서 `--expected-iteration 7`과 generation 7 조합을 승인할 수 있어 C3의 “wanted=7, expected=7” 우회가 소비자 측에서 재현됩니다. Post-damp 필수 flag도 `--allow-pre-damp`로 우회할 수 있습니다.

### 5. M_V 독립 감사 — **FAIL**

대부분의 C3 결함은 구조적으로 해소됐습니다.

- 음수 기준은 기존 `-1e-14` deadband를 제거하고 `rcond`, `N·DBL_EPSILON`, `||x||∞` 기반 bound를 사용합니다. 해 자체는 수정하지 않습니다: [rung5:46](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:46)-[68](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:68), [rung5:162](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:162)-[172](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:172).
- 보존행 감사가 최종 `Anorm` 0번 행과 `b[0]`을 실제 재독합니다: [rung5:75](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:75)-[95](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:95).
- flux 감사가 최종 `Araw`와 독립 per-target ledger를 비교합니다: [rung5:104](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:104)-[143](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:143).
- 세 fixture는 실제로 conservation coefficient, boundary route, q 분배를 각각 손상합니다: [rung5:272](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:272)-[292](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:292).

하지만 보존행 감사는 `b[0]`의 유한성을 검사하지 않습니다. `b[0]=NaN`이면 `rhs_residual`과 `equation_residual`이 NaN이고, 두 `>` 비교가 모두 false가 되어 `coeff_worst==0`을 반환할 수 있습니다: [rung5:80](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:80)-[95](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:95). 즉 최종 RHS를 재독하기는 하지만 NaN 결함에는 fail-open입니다. Fixture도 coefficient만 손상하고 `b[0]` 비유한 결함은 주입하지 않습니다.

### 6. 신규 clamp/floor/cap 및 D6 — **PASS**

- 추가행 전수에서 population/rate를 수정하는 신규 clamp, floor, cap, 사후 재정규화는 없습니다.
- `negative_error_bound`는 분류용이며 raw solution은 그대로 둔다고 명시됩니다: [rung5:46](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:46)-[64](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:64).
- `1e-12` 등 신규 상수는 fixture 판정/결함 크기일 뿐 production 물리값 수정이 아닙니다: [rung5:294](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:294)-[320](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch:320).
- C3의 동어 반복 row residual과 중간 plane flux 비교는 각각 최종 `Anorm/b` 및 최종 `Araw` 대 독립 ledger로 대체됐으므로 D6형 항등식 재발은 없습니다. 위 rung5 FAIL은 독립성 문제가 아니라 NaN fail-closed 누락입니다.
- rung6은 실행 deck 환경 고정만 추가하며 물리 계산이나 감사식을 추가하지 않습니다: [rung6:22](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung6_runtime_deck.patch:22)-[55](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung6_runtime_deck.patch:55).

B4 산출물과 워킹트리 소스는 열람하지 않았으며, 패치 적용·빌드·실행도 수행하지 않았습니다.