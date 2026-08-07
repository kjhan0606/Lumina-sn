## 질문 1 — `iter=0`의 `T_e` 자격 실패

### 1. 직접 원인

- [실측] `compute_radiative_equilibrium_te()`는 `a210_production_solve()`로 바로 위임되며, 이 생산 풀이의 입력은 검사된 복사장뿐 아니라 `CpuOpacityPublication`과 `CpuEmissivityPublication`이다([lumina_plasma.c:12063](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12063), [lumina_plasma.c:12096](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12096)).

- [실측] 이 풀이가 실제 에너지 잔차에 쓰는 방출 입력은 `em->eta_bf`, `em->eta_bb`, `em->eta_ff`이며, 각각 재결합·선·자유–자유 냉각을 공급한다([lumina_plasma.c:12011](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12011), [lumina_plasma.c:12017](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12017)).

- [실측] 입구 검사는 `opacity generation`, `emissivity generation`, `emissivity↔opacity`, `emissivity↔radiation`, `emissivity↔population` 동세대를 모두 요구하며 하나라도 없으면 `blocked_stale`로 `0`을 반환한다([lumina_plasma.c:12067](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12067), [lumina_plasma.c:12071](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12071)).

- [실측] 결정론 루프는 `a208_publish_cpu_opacity()`를 현재 복사장 계산 전에 호출하고, 이후 `cmfgen_commit_jnu()`로 복사장을 커밋한 다음 `a209_publish_cpu_emissivity()` 호출 없이 곧바로 `compute_radiative_equilibrium_te()`로 들어간다([lumina_cmfgen.c:5160](/tmp/claude-10396/codex_hyp/lumina/lumina_cmfgen.c:5160), [lumina_cmfgen.c:5202](/tmp/claude-10396/codex_hyp/lumina/lumina_cmfgen.c:5202), [lumina_cmfgen.c:5323](/tmp/claude-10396/codex_hyp/lumina/lumina_cmfgen.c:5323)).

- [가설] 따라서 현재 죽음의 직접 원인은 현재 결정론 \(J_\nu\)에 결박된 `CpuEmissivityPublication`—물리적으로는 \(\eta_{\rm bf},\eta_{\rm bb},\eta_{\rm ff}\)—이 없어서 A2-10 동세대 입구를 통과하지 못한 것이다([lumina_plasma.c:12070](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12070), [OUT_F_functions_and_wiring.md:38](/tmp/claude-10396/codex_hyp/OUT_F_functions_and_wiring.md:38)).

- [실측] 더구나 현재 `cmfgen_commit_jnu()` 요청에는 line-\(\bar J\)가 없으므로 commit은 line cache의 `computed_generation`을 0으로 만들고, 지금 `a209`를 단순 호출하더라도 `line_view_generation==0`에서 거부된다([lumina_cmfgen.c:3429](/tmp/claude-10396/codex_hyp/lumina/lumina_cmfgen.c:3429), [radiation_field.c:655](/tmp/claude-10396/codex_hyp/lumina/radiation_field.c:655), [lumina_plasma.c:8225](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:8225)).

- [가설] 이 line-view 부재는 R7을 연결한 다음 드러날 R6 후속 차단점이지만, 현재 실행에서는 `a209` 시도 자체가 없으므로 최초 관측 사망의 직접 원인은 R7 위상 단절이다([OUT_F_functions_and_wiring.md:37](/tmp/claude-10396/codex_hyp/OUT_F_functions_and_wiring.md:37), [OUT_F_functions_and_wiring.md:38](/tmp/claude-10396/codex_hyp/OUT_F_functions_and_wiring.md:38)).

### 2. `T_e_generation: 1→0`

- [실측] 값을 0으로 쓰는 정확한 지점은 `compute_radiative_equilibrium_te()`가 0을 반환한 직후의 `if (!te_qualified) plasma->T_e_generation = 0;`이다([lumina_cmfgen.c:5324](/tmp/claude-10396/codex_hyp/lumina/lumina_cmfgen.c:5324), [lumina_cmfgen.c:5326](/tmp/claude-10396/codex_hyp/lumina/lumina_cmfgen.c:5326)).

- [실측] 이어지는 fatal 메시지는 이미 0으로 덮인 값을 출력하므로 로그의 `te_generation=0`은 풀이 입구의 값이 아니라 실패 후 값이다([lumina_cmfgen.c:5327](/tmp/claude-10396/codex_hyp/lumina/lumina_cmfgen.c:5327), [run_g1c_full.log:144](/tmp/claude-10396/codex_hyp/run_g1c_full.log:144)).

- [가설] 따라서 `1→0`은 자격 실패의 원인이 아니라 결과이며, 동시에 이전 committed 세대를 보존하지 않은 별도의 R8 위반이다([OUT_F_functions_and_wiring.md:39](/tmp/claude-10396/codex_hyp/OUT_F_functions_and_wiring.md:39)).

### 3. R7인가 R8인가

- [가설] `te_qualified==0`의 원인은 R7, 즉 현재 field commit 뒤의 `a208+a209` 발행 위상 부재이다([OUT_F_functions_and_wiring.md:38](/tmp/claude-10396/codex_hyp/OUT_F_functions_and_wiring.md:38), [OUT_F_functions_and_wiring.md:84](/tmp/claude-10396/codex_hyp/OUT_F_functions_and_wiring.md:84)).

- [가설] 실패 후 `T_e_generation`을 0으로 지우는 행위만 R8 소관이며, 현재 fatal 주석이 실패 전체를 R8이라고 부르는 것은 원인과 후처리를 혼동한 것이다([lumina_cmfgen.c:5326](/tmp/claude-10396/codex_hyp/lumina/lumina_cmfgen.c:5326), [lumina_cmfgen.c:5328](/tmp/claude-10396/codex_hyp/lumina/lumina_cmfgen.c:5328)).

### 4. 확정·기각 계측

- [가설] `cmfgen_run()`의 A2-10 호출 직전에 `T_e_generation`, `radfield_view_status/generation`, `line_view_status/generation`, opacity의 `required/committed/radiation/population/te/tau generation`, emissivity의 `required/committed/opacity/radfield/line/population/te generation`, 그리고 `atom->population_committed_generation`을 한 행으로 출력해야 한다([lumina_cmfgen.c:5323](/tmp/claude-10396/codex_hyp/lumina/lumina_cmfgen.c:5323), [lumina_plasma.c:12067](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12067)).

- [가설] 동시에 A2-10의 합성 `blocked_stale` 대신 12071–12076의 어느 술어가 처음 실패했는지를 출력하고, 그 입구를 통과했다면 `a210_solve_transaction()` 반환값과 `no_bracket/blocked_schema/nonconverged`를 출력해야 한다([lumina_plasma.c:12071](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12071), [radeq_publication.c:20](/tmp/claude-10396/codex_hyp/lumina/radeq_publication.c:20)).

- [가설] 예상 계측은 `radfield=OK/gen1`, 유효한 opacity commit, `emissivity committed=0`, `line generation=0`, `A2-10 blocked_stale`이며, emissivity를 포함한 모든 입구 세대가 실제로 일치하고 transaction의 수치 풀이에서만 실패한다면 이 가설은 기각된다([lumina_plasma.c:12068](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12068), [lumina_plasma.c:12086](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:12086)).

- [실측] 계측 시 주의할 점은 `OpacityState opacity`가 자동 변수인데 `PlasmaState`와 달리 명시적으로 영초기화되지 않아, 미발행 emissivity 필드가 반드시 숫자 0이라고 소스만으로 단정할 수 없다는 것이다([lumina_main.c:86](/tmp/claude-10396/codex_hyp/lumina/lumina_main.c:86), [lumina_main.c:95](/tmp/claude-10396/codex_hyp/lumina/lumina_main.c:95)).

## 질문 2 — 최상단 이온 바닥 \(g\)

### 1. Sc IV·Ti V·V II의 출처

- [실측] 현재 Lumina 저장소에는 그 세 이온의 별도 바닥준위 표가 없고, 현재 런타임 로더는 덱의 `levels.csv`에서만 `energy_eV`와 `g`를 읽는다([lumina_atomic.c:1406](/tmp/claude-10396/codex_hyp/lumina/lumina_atomic.c:1406), [population_contract.c:100](/tmp/claude-10396/codex_hyp/lumina/population_contract.c:100)).

- [실측] 다만 디스크에는 독립된 외부 앵커인 Cloudy Stout 원자자료가 있으며, 그 형식은 `.nrg`의 두 번째 필드를 에너지(cm\(^{-1}\)), 세 번째 필드를 통계중량 \(g\)로 정의한다([StoutFormat:13](</gpfs/kjhan/cloudy-master/data/stout/StoutFormat:13>), [StoutFormat:17](</gpfs/kjhan/cloudy-master/data/stout/StoutFormat:17>)).

- [실측] Sc IV의 최저 준위는 `E=0.000 cm⁻¹, g=1, 3s²3p⁶ ¹S₀`이고 파일은 NIST 2014-09-16을 출처로 명시한다([sc_4.nrg:2](</gpfs/kjhan/cloudy-master/data/stout/sc/sc_4/sc_4.nrg:2>), [sc_4.nrg:131](</gpfs/kjhan/cloudy-master/data/stout/sc/sc_4/sc_4.nrg:131>)).

- [실측] Ti V의 최저 준위도 `E=0.000 cm⁻¹, g=1, 3s²3p⁶ ¹S₀`이며 NIST 2014-09-16 출처다([ti_5.nrg:2](</gpfs/kjhan/cloudy-master/data/stout/ti/ti_5/ti_5.nrg:2>), [ti_5.nrg:67](</gpfs/kjhan/cloudy-master/data/stout/ti/ti_5/ti_5.nrg:67>)).

- [실측] V II의 최저 준위는 `E=0.000 cm⁻¹, g=1, 3d⁴ a ⁵D₀`이며 역시 NIST 2014-09-16 출처다([v_2.nrg:2](</gpfs/kjhan/cloudy-master/data/stout/v/v_2/v_2.nrg:2>), [v_2.nrg:326](</gpfs/kjhan/cloudy-master/data/stout/v/v_2/v_2.nrg:326>)).

- [가설] 따라서 세 이온은 이 Cloudy/Stout의 NIST-앵커 레코드를 외부 provenance와 함께 가져올 수 있으며, 세 값은 모두 \(E_0=0,\ g_0=1\)이다([sc_4.nrg:2](</gpfs/kjhan/cloudy-master/data/stout/sc/sc_4/sc_4.nrg:2>), [ti_5.nrg:2](</gpfs/kjhan/cloudy-master/data/stout/ti/ti_5/ti_5.nrg:2>), [v_2.nrg:2](</gpfs/kjhan/cloudy-master/data/stout/v/v_2/v_2.nrg:2>)).

### 2. 자료가 끝내 없을 때의 처분

- [가설] 외부 앵커도 확보되지 않으면 해당 원소의 최상단 이온을 `POP_ATOMIC_MISSING`으로 거부하고 population transaction 전체를 발행하지 않는 것이 규약에 맞다([OUT_F_functions_and_wiring.md:34](/tmp/claude-10396/codex_hyp/OUT_F_functions_and_wiring.md:34), [population_contract.c:86](/tmp/claude-10396/codex_hyp/lumina/population_contract.c:86)).

- [실측] 현재 `hi==lo`에서 `Z=1`을 반환하는 것은 명시적으로 “정확한 임시 대입”이며, 뒤의 ×30 게이트는 그 근사의 런별 영향 상한만 검사한다([population_contract.c:92](/tmp/claude-10396/codex_hyp/lumina/population_contract.c:92), [lumina_plasma.c:6808](/tmp/claude-10396/codex_hyp/lumina/lumina_plasma.c:6808)).

- [가설] 이번 분율 상한이 작다는 사실은 잘못된 \(g\)가 물질 상태에 미치는 영향이 작았다는 것만 보이며, \(g=1\) 자체를 원자자료로 검증하지 않으므로 R0 착지 뒤에도 유지할 정본 규칙은 아니다([run_g1c_full.log:69](/tmp/claude-10396/codex_hyp/run_g1c_full.log:69), [OUT_F_functions_and_wiring.md:31](/tmp/claude-10396/codex_hyp/OUT_F_functions_and_wiring.md:31)).

### 3. CMFGEN osc의 정확한 읽기 규약

- [실측] CMFGEN 독자는 헤더의 선언된 준위 수만큼 레코드를 순서대로 읽고, 준위명 뒤 첫 숫자를 `STAT_WT`, 다음 숫자를 excitation energy(cm\(^{-1}\))로 해석한다([genosc_v5.f:189](</gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v5.f:189>), [genosc_v5.f:197](</gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v5.f:197>), [genosc_v5.f:211](</gpfs/kjhan/cmfgen_src/cur_cmf/subs/genosc_v5.f:211>)).

- [실측] CMFGEN의 별도 소비자는 첫 준위 레코드가 `E=0.0000`인지 검사한 뒤 그 레코드의 첫 숫자를 “statistical weight of ground state”로 사용하므로, 첫 레코드가 바닥이라는 것은 단순 경험칙이 아니라 파일 소비 계약이다([read_seq_time_file_v1.f:228](</gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/read_seq_time_file_v1.f:228>), [read_seq_time_file_v1.f:238](</gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/read_seq_time_file_v1.f:238>), [read_seq_time_file_v1.f:243](</gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/read_seq_time_file_v1.f:243>)).

- [가설] 따라서 규약은 “첫 비어 있지 않은 행을 맹신”하는 것이 아니라, 선언된 level block을 형식대로 파싱한 뒤 첫 레코드의 `ID=1`, 유일한 최소 excitation energy \(E_{\rm cm}=0\) 또는 부호 있는 `−0`, 유한·양수·정수형 \(g\)를 모두 확인하고 그 `g`를 \(g_0\)로 채택하는 것이다([cmfgen_parser.py:114](</gpfs/kjhan/lumina_runner2/scripts/cmfgen_parser.py:114>), [cmfgen_parser.py:147](</gpfs/kjhan/lumina_runner2/scripts/cmfgen_parser.py:147>), [cmfgen_parser.py:164](</gpfs/kjhan/lumina_runner2/scripts/cmfgen_parser.py:164>)).

- [가설] 이 규약에서 \(g_0\)는 첫 CMFGEN 세부준위의 통계중량이지 같은 ground term의 fine-structure \(g\) 합이 아니므로, 예를 들어 O IV는 첫 레코드의 \(g_0=2\)이며 현재 별도 실험 앵커의 `g=6`을 그대로 재사용하면 다른 물리 객체를 섞게 된다([O IV osc_data:20](</gpfs/kjhan/cmfgen_21jun23/atomic/OXY/IV/19apr23/osc_data:20>), [lumina_atomic.c:2672](/tmp/claude-10396/codex_hyp/lumina/lumina_atomic.c:2672)).

- [실측] 같은 규약으로 얻는 12개 \(g_0\)는 C IV 2, O IV 2, Mg IV 4, Al V 4, Si VI 4, S VI 2, Ca VI 4, Cr V 5, Mn IV 1, Fe VII 5, Co VII 4, Ni VII 1이다([C IV:18](</gpfs/kjhan/cmfgen_21jun23/atomic/CARB/IV/19apr23/osc_data:18>), [O IV:20](</gpfs/kjhan/cmfgen_21jun23/atomic/OXY/IV/19apr23/osc_data:20>), [Mg IV:21](</gpfs/kjhan/cmfgen_21jun23/atomic/MG/IV/19apr23/osc_data:21>), [Al V:23](</gpfs/kjhan/cmfgen_21jun23/atomic/AL/V/19apr23/osc_data:23>), [Si VI:19](</gpfs/kjhan/cmfgen_21jun23/atomic/SIL/VI/19apr23/osc_data:19>), [S VI:21](</gpfs/kjhan/cmfgen_21jun23/atomic/SUL/VI/19apr23/osc_data:21>), [Ca VI:18](</gpfs/kjhan/cmfgen_21jun23/atomic/CA/VI/19apr23/osc_data:18>), [Cr V:15](</gpfs/kjhan/cmfgen_21jun23/atomic/CHRO/V/19apr23/osc_data:15>), [Mn IV:15](</gpfs/kjhan/cmfgen_21jun23/atomic/MAN/IV/19apr23/osc_data:15>), [Fe VII:18](</gpfs/kjhan/cmfgen_21jun23/atomic/FE/VII/19apr23/osc_data:18>), [Co VII:15](</gpfs/kjhan/cmfgen_21jun23/atomic/COB/VII/19apr23/osc_data:15>), [Ni VII:18](</gpfs/kjhan/cmfgen_21jun23/atomic/NICK/VII/19apr23/osc_data:18>)).