# 사전등록 — DET-COLDSTART: 공개 CMFGEN 이온분율 초기화 차가운 출발 시험 (쌍안정 판별)

**표적:** Lumina 이온 인구를 공개 StaNdaRT CMFGEN toy06 19.480 d 이온분율(차가운 가지)로 초기화한 뒤 생산 반복을 돌렸을 때 sh10–30 의 S III/S II 가 차가운 값 근처에 머무는가(COLD-STAY), 뜨거운 가지로 돌아가는가(HOT-RETURN). 상태 = **기안**(실행 승인·구현·판정 아님). 기안자 = Fable fresh(`claude-fable-5-1`, 개정25-k). 최종목적 = MC 수송으로 CMFGEN 수준 SN Ia 스펙트럼 재현. 잣대 = **공개 파일만**(자체 CMFGEN 런 인용 금지, 재점검 목록 §0).

## §0. 선행 사실 (저자 실행 = 로그인 노드 비연산 조회; 값은 referent, 예측 아님)
1. **공개 파일**: `data/standart_data1/toy06/ionfrac_{si,s,ca,fe,co,ni}_toy06_cmfgen.txt`, 블록 `#TIME:    19.480`(S 파일 행 2841–2943), `#NVEL: 100`, vel_mid 1,025–40,325 km/s; 열 = `s0..s4`(Si/S/Ca = I..V) 또는 `fe0..fe5`(Fe/Co/Ni = I..VI). 중성 열은 1e-99(S·Si·Ca·Co·Ni) 또는 ≤9.3534e-10(Fe, 19.480 블록 최대). sh24 부근 S: v21,630 (0.5726, 0.4145) x=0.724 · v22,335 (0.5259, 0.4569) x=0.869. 공개 phys 19.480: T·ρ·n_e·n_atom(v21,630: 9,231 K·1.36e7·7.73e6).
2. **Lumina**: `geometry.csv` 50 셸(v_in/v_out cm/s; sh9 10,452–11,180 · sh24 21,372–22,100 · sh30 25,740–26,468 km/s). `active_ions.csv` 27 이온 = Si/S/Ca II–V, Fe/Co/Ni II–VI — 공개 최상단 단계와 **일치**, Lumina 에 없는 공개 단계는 **I 뿐**, 공개에 없는 Lumina 단계 없음. 쌍둥이 T_e·침적은 공개 파생(재점검 §2).
3. **초기화 경로(코드, 동결 트리)**: `cmfgen_run`(lumina_cmfgen.c:8767) pass 0 = init pass: bootstrap Saha 물질(pop gen 1; lumina_main.c:259–266 창) → assemble → solve_J(rad gen 1) → `lumina_init_seed_material_predictor`(lumina_plasma.c:17422, 호출 :9069) → `nlte_population_candidate_build_bundle`(:22615; NE_CLOSURE 루프 ≤32회·tol 1e-8·감쇠 0.5) → `prepare_through_core`(:22582: begin→opacity view→ionization_at_ne→precore tau→solve_core) → `nlte_ne_closure_candidate("SEED")`(:21236) → `commit_seed_material`(:23131) = **pop gen 2**. 논리 반복 k: assemble(gen k+2 물질) → solve_J → `lumina_r7_publish_and_solve_te` → 커밋 gen k+3 → `physics_comparison_dump_if_requested` iter k(ion.csv population_generation = k+3, 398939 iter0000 gen 3 실측). 이온 단계 해 = `compute_ion_populations_shell`(:3566) 이 원소별 인접단계 비로 n_element 를 분배해 `atom->ion_number_density[(ip_start+k)*n_shells+s]` 에 씀(:3800–3822); n_e 는 NE_CLOSURE=1 경로에서 매 회 `prepare_through_core(…, trial_ne)`→`ionization_at_ne`(solve_stage_ne=0: `compute_electron_density`(:3878) 의 내부 반복은 쓰지 않음)→준위 해 순서로 폐합되며, 폐합 대상 전하 = `nlte_ne_closure_charge`(:21133) 의 **혼합 전하합** Q_closed = Σ_{대상 II·III} q·Σ_l n_l + Σ_{나머지 단계} q·n_stage — 단계 배열 전하합이 아니다(398939 sh24 iter0000: n_e 14,911,827.36 = 혼합합, 단계 배열 합 14,880,539.82). ion.csv `Z16_stage_q1/q2` = 전하 1/2 = S II/S III 단계 배열, `Z16_nlte_q1/q2` = 같은 이온의 준위 합(physics_comparison.c:445).
   ⚠`LUMINA_NLTE_START_ITER` 는 lumina_main.c:819/844 의 MC 루프만 소비하고 PURE_CMFGEN 경로는 읽지 않는다(cmfgen.c·plasma.c grep 0건) ⟹ DET 레인에 「NLTE 시작 반복 2」는 없다: 준위 NLTE 해는 SEED 부터 매 반복.
   **준위 해의 보존 행(코드)**: 398939 exports 는 `LUMINA_NLTE_ION_LOCK=0`·`LUMINA_NLTE_PER_ION_RESCALE=0`·`LUMINA_NLTE_LOCK_START_ITER=0`. 이 설정에서 S II–III 쌍의 준위 해(`nlte_solve_all_impl`, CPU; `LUMINA_NLTE_ASSEMBLE_GPU=0` 이라 `lumina_cuda.cu` 의 준위 해 경로는 DET 레인에서 돌지 않음)는 보존 행 하나 = **두 단계 준위 합만** `nlte_pair_total_density`(:1958, 단계 배열 II+III 합)에 묶고(:20258), 해 뒤 재규격화도 합만(:20793 부근) ⟹ 준위 해가 J 로 II/III 를 다시 나눈다(398939 sh24 iter0000: 단계비 69.596 대 준위합비 665.05). `nlte_ion_lock_active(current_iter)`(:11433, env 정적 캐시, `enabled && current_iter >= start_iter`)=1 이면 두 보존 행이 단계 배열의 n_lo·n_hi 에 각각 묶이고(:20238–20263) 재규격화도 이온별(:20674 판정)이며 generator 캡처가 꺼진다(:20457). `current_iter` 는 cmfgen.c:8871 이 init pass=0·논리 반복 k=k 로 설정 ⟹ SEED 와 iter 0 은 구별되지 않는다. **기존 노브 재사용 불가**: LOCK_START_ITER 는 「그 반복부터 계속 잠금」이라 pin-then-release(앞 N 반복만) 의 반대 방향이고, ION_LOCK=1 전역은 base 솔버 자체를 바꾼다(x_hot 계보 이탈). bf 흡수는 S II 준위 NLTE 인구(:9113), Milne 방출은 S III 단계 배열(:9082·:9173)을 쓰므로 b_l = n_l/n_l*(T_e,n_e,n_III) 는 두 단계 총량이 함께 정해져야 차가워진다(단계비 주입만으로는 보장 없음 — astra 1).
4. **런 내 증인**: `lumina_pop_djdn.csv`(열 gen,shell,Z,stage,x_lag,x_map,…,epoch): epoch e 의 x_lag = **커밋 gen e 의 비**, x_star = gen e+1. 398939 sh24 Z16: epoch1 x_lag 1044.7(bootstrap Saha)→x_star 81.36(SEED gen 2); epoch2 x_lag 81.36→69.596 = iter0000 ion.csv 와 9자리 일치. ⟹ **epoch-2 x_lag 행 = SEED 커밋 상태**(epoch e 의 x_lag = 커밋 gen e, epoch 안 전 행 동일; 「x_star = 다음 커밋」은 epoch 마지막 채택 행에만 해당하고 `gen` 열은 sweep 계수이지 물질 세대가 아니다 — astra 5). 이 증인은 단계 배열만 보며 준위 합·b·장은 증명하지 않는다(그 몫 = §4 G-COLD).
5. **대조군·base = 398939**(`/gpfs/kjhan/lumina/cmfgen_map/pathb_xshell_20260922/run2`, a100, ITER 6, 47,833 s, SUCCESS, binary 35c6dbc2; h100 run1 44,224 s). DR-ON 385315 대비 exports 델타 = CONT_NLTE_SOURCE=1·POP_DJDN=1·POP_DJDN_CHI=1·CMAX=1.0·ITER 6·XSHELL_RESPONSE=1; deck/sigma/topion sha·fixed_te_profile(f6db3543) 는 DR-ON 과 동일. **선택 근거**: (i) 가설 기전 S_bf=B/b 는 연속체 소스가 인구에 반응해야 존재한다 — DR-ON 은 S_fixed=χ·B(T_e)(09-17 판독 A) 라 **이 소스 되먹임 경로(b→S_bf→J)가 닫혀 있다**(인구 의존 불투명도·선 수송 등 다른 되먹임은 DR-ON 에도 있으므로 「쌍안정이 원리적으로 닫혀 있다」는 말이 아니다 — astra 7); 발주서 referent(S_bf/B 22·b≈0.05·x_lag 134)도 이 계보다; (ii) 6반복 뜨거운 궤적을 가진 유일한 완주 런; (iii) DJDN 은 Picard 몫 ln ρ 를 감쇠하지 않고 Newton 보정 |c|≤C_max 만 캡(`pop_djdn_log_newton`, :2591–2600) ⟹ 유지 안 되는 차가운 상태는 1반복 안에 되돌아갈 수 있다(감쇠가 복귀를 숨기지 않음). exports 델타 = §2 의 env 2키(`+LUMINA_INIT_IONFRAC_DIR`·`+LUMINA_INIT_IONFRAC_PIN_ITERS`) — 하나의 조작(차가운 출발)을 켜는 두 키.
6. **동결 트리 ≠ 398939 스테이징 소스**: `run2/input/current_source_identity.json` 의 4파일(lumina.h·lumina_plasma.c·nlte_population_candidate.h·env_universe.h) sha 가 트리 현행과 불일치(트리 mtime 09-22 12:44–13:01 < 스테이징 09-23 10:23). 운전자가 발주 전 **빌드 base = 398939 바이너리 소스**를 확정해야 G-ID 가 선다(불일치면 base 계보 캐비앗 기재). 금지구간 lumina_cmfgen.c:2848–3212 sha 8d3e3d76 확인.
7. **뜨거운 가지는 수렴 상태가 아니다**: 398939 sh24 x = 69.6/73.7/66.5/133.6/109.4/150.1(반복 0–5); sh16 8.97→2.53, sh18 16→2.02, sh30 723→42,858→10,498(관측). 그래서 판정 셸을 §5 규칙으로 고른다.

## §1. 물리 계약 1개
**질문:** 「398939 와 동일한 계보(같은 봉인 입력·exports·솔버)에서 SEED 와 논리 반복 0–1 동안 S 이온 단계 총량(단계 배열과 준위 합 둘 다)을 공개 CMFGEN 19.480 d 분율에 고정해 장이 차가운 물질 위에서 이완하게 한 뒤 반복 2 부터 풀어 놓으면, 판정 셸 J 의 S III/S II 가 반복 3–5 동안 차가운 기준(공개) 곁에 머무는가(이 초기화·솔버·창에서 차가운 상태가 **유지됨**), 뜨거운 기준(398939 반복 3–5) 쪽으로 돌아가는가(**유지 안 됨**)?」
**말하지 않는 것:** 공개 CMFGEN 재현·어느 가지가 물리적으로 옳은가·쌍안정의 존재/부재 일반론·「Lumina 방정식이 차가운 가지를 지지한다」는 수렴 주장(6반복·미수렴 x_hot 창의 유지/복귀 관측까지만)·차가운 상태의 물리적 안정성(해제 후 이동은 솔버 반복이지 시간진화가 아니다)·NT/시간의존 효과·Si/Ca/철족 판정·자유 T_e.

## §2. 초기화 규칙과 주입 설계 — pin-then-release (env 2개, 기본 OFF)
- **env** `LUMINA_INIT_IONFRAC_DIR` = 6 공개 파일 사본 디렉토리(스테이징에 복사, sha256 = repo 원본). 미설정 = OFF: getenv 1회 외 어떤 문장도 실행되지 않는다(byte-동일의 근거 = 진입 조건 하나). `LUMINA_INIT_IONFRAC_PIN_ITERS` = N_pin(정수 ≥1) = 고정을 유지하는 논리 반복 수; DIR 설정 시 필수(미설정·0·비정수 = fail-closed 종료, 조용한 기본값 없음). **이 런 = 2**(SEED + 반복 0·1 고정, 반복 2 해제). 설계 선택 근거: (a) 준위까지 LTE 분배로 주입하는 안은 뜨거운 rad gen 1 위에서 SEED 준위 해가 곧바로 다시 나누므로(§0-3) 한 반복도 차갑지 않다; (b) 주입을 bootstrap 으로 앞당겨도 rad gen 1 하나만 차가운 물질을 보고 SEED 해는 여전히 자유라 ALI 8 회 초기장 의존(XSHELL 판정문:41)을 못 벗는다; (c) pin-then-release 만이 「장이 차가운 물질과 일관되게 이완한 상태」를 해제 직전에 만들고 그 상태를 표준 덤프(ion.csv iter0000·0001)로 증언한다.
- **속도 보간**: 셸 v_mid=(v_in+v_out)/2. 19.480 블록에서 v_mid 를 감싸는 두 행(i,i+1)으로 **각 단계 k 의 ln f_k 를 ln v 에 선형 보간**, 그 뒤 Σ_k f_k=1 재규격화. 범위 밖(v_mid<1,025 또는 >40,325; 현 격자엔 없음)·블록 부재·열 수 불일치 = fail-closed 종료(floor·대체 없음).
- **단계 사상**: Lumina II ← 공개 I+II(합, 총량 보존); III..최상단 ← 공개 같은 단계. 원소별 「공개 단계 수 = Lumina 단계 수 + 1」 검사, 아니면 종료.
- **armed 판정(1곳)**: `nlte_population_candidate_build_bundle`(:22615) 진입에서 armed ← (DIR 설정) ∧ (`request->public_nlte->current_iter` < N_pin), 반환 전(실패 경로 포함) 해제. `current_iter` 는 init pass(SEED)=0·논리 반복 k=k(cmfgen.c:8871) ⟹ N_pin=2 는 SEED·반복 0·반복 1 의 후보 solve 전부를 고정하고 반복 2 부터 armed=0 = 생산 해 그대로(그것이 시험). `lumina_cmfgen.c` 는 건드리지 않는다.
- **주입 지점 ① 단계 배열(1곳)**: `compute_ion_populations_shell` 이 원소 e·셸 s 의 단계 밀도를 쓴 직후(:3800–3822 블록 끝, 같은 호출) — armed 일 때만 n_ion(k) ← f_k·n_element(n_element = 그 호출의 율 해가 분배한 총량 ⟹ 원소 총량 보존). 후보의 n_e 폐합 루프·ionization_at_ne 서브반복 전부가 같은 주입 해를 되받는다.
- **주입 지점 ② 준위 합 고정(기존 잠금 경로 재사용, 판정식 3곳에 OR)**: 준위 해의 잠금 판정식 세 곳 — 보존 행 선택(:20027 `ion_lock_mode`)·generator 캡처 조건(:20457)·해 뒤 재규격화(:20674 `lock`) — 에 armed 를 OR 한다(static helper 1개; 잠금 경로 본문은 무변경). armed 이면 S II·S III 준위 합이 각각 주입된 단계 총량에 묶이고(:20238–20263 의 두 보존 행) 재규격화도 이온별 ⟹ 준위합비 = 단계비 = x_pub, b_ground(S II) = (1/x_pub)·f_ground/(n_e Φ_1(T_e)) 로 차가운 값(f_ground 는 해가 정함), Milne S_bf 의 n_III 도 주입값. **주의**: 고정 창의 준위 해는 두 보존 행 선형 해(legacy ION_LOCK 경로, generator 캡처 OFF)이고 해제 후 = 생산 경로 — 창 안 준위 분포는 측정 대상이 아니다. 새 방정식·새 폐합 규칙 없음. `LUMINA_NLTE_ION_LOCK`·`LOCK_START_ITER`·`PER_ION_RESCALE` 은 398939 값(0) 그대로(§0-3: 재사용 불가). 구현자 확인 항목: XSHELL 캡처 게이트(cmfgen.c:8899/8913/9004)가 generator 캡처 경로에 의존하지 않음(J 솔브 쪽) — 의존하면 보고 후 정지.
- **n_e 폐합**: 기존 NE_CLOSURE 가 혼합 전하합(§0-3)으로 폐합한다. 고정 창에서는 준위 합 = 주입 단계 총량이므로 Q_closed = Σq·n_inj(재규격화 부동소수 오차 내) — ② 의 귀결이지 새 규칙이 아니며, ② 없이는 성립하지 않는다(astra 2).
- **고정 창의 DJDN**: x_lag 스냅샷은 커밋 배열(주입값), x_map·x_star 는 율 해가 내지만 곧 ① 이 덮어쓴다 ⟹ epoch 2·3·4 의 x_lag = x_pub(런 내 증인, G-INJ(b)). 해제 반복(2)의 x_lag = x_pub, x_star = 첫 자유 결과(ion.csv iter0002).
- **준위**: LTE(Boltzmann) 강제 코드를 넣지 않는다 — 준위 분포는 고정된 이온 총량 아래 기존 해가 정한다(②). ⚠원안의 「기존 준위 단계가 주입된 단계 총량으로 규격화한다」는 ION_LOCK=0 코드와 달랐다(합만 보존, astra 1) — ② 가 그 정정이다.
- **증거 writer**: SEED(init pass, current_iter 0) armed 마지막 회차의 값을 `lumina_init_ionfrac_applied.csv`(shell,Z,stage,v_mid,f_applied,n_ion,n_element) 로 cwd 에 1회 기록(반복 0·1 의 고정 상태는 표준 ion.csv 가 증언하므로 따로 쓰지 않는다).
- clamp·floor·부호 교정 없음. `LUMINA_ENV_STRICT=1` 이므로 env 이름을 `env_universe.h` 에 등록.

## §3. 변경집합과 산출물
### 변경집합 — DET-COLDSTART (표)
| 경로 | 분류 | 허용 변경 |
|---|---|---|
| `src/lumina_plasma.c` | 주입 | 공개 파일 로더·보간·단계 사상(static), armed 판정(build_bundle 진입/반환 1곳)·단계 주입 1곳·준위 잠금 판정식 3곳(:20027·:20457·:20674)에 armed 를 OR(static helper 1개), applied.csv writer; 율 해·n_e 폐합·커밋·DJDN 산술·잠금 경로 본문 무변경 |
| `src/lumina.h` | 테스트 API | 로더·보간 함수 선언 1개(selftest 진입용) |
| `src/env_universe.h` | strict env | `LUMINA_INIT_IONFRAC_DIR`·`LUMINA_INIT_IONFRAC_PIN_ITERS` 등록 |
| `tests/a2_10_radeq_selftest.c` | 직접시험 | 케이스 1개: 50셸×6원소 f_applied 표를 stdout CSV 로(판독기 G-INJ(a) 와 같은 비교에 물림) |
| `scripts/coldstart_judge.py` | 판독·판정 | 신규(repo). referent 표·G-INJ·G-COLD·G-RUN·r_s(k)·셸별 분류 지도·분기·NC 플래그·`--classify-only`(NC-HOT 전용) |
### 변경집합 끝
`expected_extra` = 위 5파일. 변경 금지: `src/lumina_cmfgen.c`(금지구간 포함)·`src/lumina_cuda.cu`·`Makefile`·봉인 입력·398939 산출(읽기만). 산출: `coldstart_referents.csv`(런 전 생성·sha 봉인) · `coldstart_result.json`(게이트·r 표·셸별 분류 지도·분기·입력 해시) · applied.csv · 표준 dump. 런 루트 `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/run1/`.

## §4. 게이트와 음성대조 (허용오차: 산술 항등 상대 1e-6; 물리 판정은 §5 의 r 띠)
- **G-ID**: 봉인 입력 sha(deck/sigma/topion/fixed_te) = 398939 · exports 델타 = `+LUMINA_INIT_IONFRAC_DIR`·`+LUMINA_INIT_IONFRAC_PIN_ITERS=2` 두 키뿐(ION_LOCK/LOCK_START_ITER/PER_ION_RESCALE = 0 그대로) · ITER 6·ALI 8·XSHELL_RESPONSE 1 유지(XSHELL=0 은 base 교체이므로 이 계약 밖, §6) · 소스 = 집행 2a 확정 base(현 동결 트리) + 변경집합 파일만(sha 대조) · 6 공개 사본 sha256 = repo 원본 · 바이너리 sha 기록 · h200 아님.
- **G-INJ**(주입 재현): (a) applied.csv 의 f_applied = 판독기가 repo 원본으로 §2 규칙을 독립 재계산한 값(전 셸×원소×단계, 1e-6) ∧ n_element > 0 인 셀은 n_ion/n_element = f_applied(1e-9), n_element = 0 인 셀(예: sh24 Fe/Co/Ni)은 전 단계 n_ion == 0 정확 영(나눗셈 없음, astra 6); (b) `lumina_pop_djdn.csv` epoch 2·3·4(= SEED·반복 0·반복 1 커밋; §0-4) 의 Z 16·stage 1·s∈J 마지막 채택 행 x_lag = x_pub,s(1e-6). (c) 는 삭제 — (b) 가 SEED 경로 실행을 이미 증명한다(grok L4).
- **G-COLD**(해제 직전 상태가 차가운가 — r 의 0 점이 실제로 차갑다는 전제): ion.csv iter0000·iter0001(고정 창의 커밋 gen 3·4), s∈J: `Z16_stage_q2/q1` = x_pub,s(1e-6) ∧ `Z16_nlte_q2/nlte_q1` = x_pub,s(1e-6; 이온별 재규격화는 정확) ∧ `Z16_nlte_q1` = `Z16_stage_q1`(1e-9). 장 이완 증인(게이트 아님·판정문 기재 필수): spectral.csv 에서 S II 바닥 엣지(23.34 eV = 5.643e15 Hz)를 담는 bin 의 J_nu, s∈J, iter0000→0001 상대 변화와 398939 같은 반복 대비 비 — 「장이 차가운 물질 위에서 이완했는가」 서술의 유일한 근거(수치 예측 미등록).
- **G-RUN**: terminal_state=SUCCESS·model_rc=0·outer_iterations=6·ion.csv iter0000–0005 실재·gen 3–8·J 셸 Z16 q1,q2 유한 양수. 고정 창(iter<2) 안에서 준위 해 실패(NLTE_ION_SOLVE_NEGATIVE_POPULATION/NONFINITE 계열 진단, 그 진단의 ion_lock_active=1)로 MODEL_FAILED 면 G-RUN FAIL 이 아니라 **분기 PIN-FAIL**(§5); 그 밖의 실패 = X.
| NC | 주입 | 반드시 FAIL / 결과 | 시점 |
|---|---|---|---|
| NC-INJ | 사본 디렉토리의 S 파일 19.480 블록 s1 열 ×2(재규격화 없음; 실제 어긋남 ln 2) | selftest CSV 대 판독기 → G-INJ(a) FAIL; 런 후 applied.csv 대 변조 원본 → FAIL | 런 전(selftest)·런 후 |
| NC-HOT | **분류식 시험**(`--classify-only`, 게이트 우회 — 게이트를 걸면 G-INJ(b)·G-COLD 가 FAIL 이라 X 가 되므로 분리): 398939 ion.csv iter3–5 투입 | HOT-RETURN(J 선정 조건상 전 r ≥ 2/3 — **구성상 참**, 판독기 배선 시험이지 판별력 증거 아님) ; x=x_pub 합성 → COLD-STAY(r=0·d=0) | 런 전 |
| NC-COLD | 판독기 G-COLD 에 398939 iter0000·0001 ion.csv 투입 | G-COLD FAIL(sh24 단계비 69.6·준위합비 665 ≠ x_pub 0.744) | 런 전 |
OFF byte-동일 = 진입 조건 1개(소스 검사, Fable 코드 감사 첫 항목) — OFF GPU 재런(13 h)은 하지 않는다(오프라인 주장, DR-ON 전례와 같이 기재).

## §5. 기대치·분기
**판정 통계**: s∈J, k∈{3,4,5}: r_s(k) = ln(x_s(k)/x_pub,s) / ln(x_hot,s/x_pub,s). x_s(k) = iter k ion.csv `Z16_stage_q2/q1`; x_pub,s = §2 보간의 S III/S II(= 고정 상태의 비 — ln 보간이라 재규격화와 무관); x_hot,s = 398939 반복 3–5 기하평균. 0 = 차가운 고정 상태, 1 = 뜨거운 기준. **절대 조건** d_s(k) = |log10(x_s(k)/x_pub,s)| ≤ 0.5 — r 띠만으로는 sh24 x ≤ 4.16·sh30 x ≤ 30.2 도 「차가운」이 되므로(astra 3) COLD 는 공개값 반 자릿수(3.16×) 안까지만 인정한다(0.5 dex 는 도출값이 아니라 선택: 잰 어긋남 1–4 dex 의 최소 gap 의 절반). **셸별 분류**(k=3,4,5 전부로): COLD_s ⟺ 전 k 에 r ≤ 1/3 ∧ d ≤ 0.5; HOT_s ⟺ 전 k 에 r ≥ 2/3; 그 밖 MID_s. 방향 = sign(r_s(5) − r_s(3)), |Δr| > 0.1 이면 UP/DOWN, 아니면 FLAT. 해제 반복 r_s(2) 도 기록(첫 자유 결과; HOT-RETURN 의 다음 걸음이 소비).
**판정 셸 J**(규칙: gap=log10(x_hot/x_pub) ≥ 1.0 ∧ 뜨거운 산포 log10(max/min, 반복 3–5) ≤ gap/3 ∧ 차가운 띠 |log10 x_pub(v_out)/x_pub(v_in)| ≤ gap/3) = **{10–15, 19–30}, 18 셸**. 제외: 9(차가운 띠 0.68 > 0.34, IGE 혼재), 16(산포 0.88 > 0.50), 17(gap 0.84), 18(산포 0.97 > 0.48).
Referent(sh: x_pub / x_hot / gap dex; 저자 계산, 판독기가 `coldstart_referents.csv` 로 재생성해야 함): 10: .241/15.4/1.80 · 11: .181/20.1/2.05 · 12: .186/5.27/1.45 · 13: .197/3.74/1.28 · 14: .210/3.00/1.15 · 15: .233/2.50/1.03 · 19: .349/18.9/1.73 · 20: .392/24.8/1.80 · 21: .490/45.6/1.97 · 22: .572/33.6/1.77 · 23: .696/42.8/1.79 · **24: .744/130/2.24** · 25: .880/369/2.62 · 26: .959/1.51e3/3.20 · 27: 1.10/2.52e3/3.36 · 28: 1.19/3.74e3/3.50 · 29: 1.30/8.75e3/3.83 · 30: 1.43/1.33e4/3.97.
**경계 근거**: [1/3, 2/3] 띠는 gap 의 1/3 을 해상도 단위로 잡은 **선택**이다 — J 선정이 뜨거운 자기 산포를 gap/3 이하로 묶으므로 뜨거운 쪽 1/3 은 그 산포 안이고, 차가운 쪽 1/3 은 절대 조건 0.5 dex 가 다시 좁힌다. 두 우물·분리선 1/2 같은 쌍안정 모형에서 도출한 것이 아니다. 한계(문면): x_hot 는 미수렴(sh30 창 산포 0.91 dex, sh24 66–150)이고 창은 3반복이라 「안정 가지」·「이완 완료」를 증명하지 않는다 — 결과는 이 창의 유지/복귀 관측이다. 반복 창 3–5 = 뜨거운 referent 와 같은 반복 지수(지수만 같고 ALI·DJDN 이력은 다르다 — 고정 창 2반복이 끼므로 「동지수 비교」는 근사). 자유 반복 = 2,3,4,5(4회). ITER 6 근거: (i) Picard 몫 무감쇠라 복귀는 1반복이면 보이나 (ii) 준위 NLTE 되먹임(S II 바닥 b)이 커밋을 거쳐 장에 들어가는 데 ≥2반복, (iii) 대조군과 같은 창. N_pin=2 근거: SEED 물질 위 J 1회 + 고정 커밋 위 J 2회 = 해제 전 차가운 물질 위 J 3회(ALI 8×3) — ALI 8 초기장 의존을 완전히 벗었다는 주장은 하지 않고 G-COLD 의 장 이완 증인으로 기재한다. N_pin 을 더 늘리면 자유 창이 3–5 에서 줄어든다.
**구성상 참**: r_s(k), k≥2 는 런 없이 못 얻는다(방향 예측 미등록, 갈림길 시험). 예외 명시 — G-COLD 의 고정 창 값(r=0)과 NC-HOT 의 두 결과는 **구성상 참**(고정·선정 조건의 귀결)이며 판독기 배선 시험이다(astra 6). 기록(분기 밖, 같은 ion.csv·spectral.csv 파싱): r_s(2)(해제 반복) · G-COLD 의 장 이완 증인. 뺀 것(grok L4): SEED n_e 대 공개 n_e 비 · Si/Ca/Fe r · S II 바닥 분율(반복별 준위 덤프 없음).
| 분기 | 판정식 | 뜻 | 다음 |
|---|---|---|---|
| **COLD-STAY** | 게이트 전부 PASS ∧ 전 s∈J 가 COLD_s(r ≤ 1/3 ∧ d ≤ 0.5, k=3–5) | 이 초기화·솔버·4 자유 반복 창에서 차가운 상태가 유지됐다(공개값 반 자릿수 안) — 이력 의존의 관측이지 안정 가지·수렴 증명 아님 | 뜨거운 대 차가운 상태의 스펙트럼·EUV 창 J/B 대조(런 0회 판독) 후 user 결정(ITER 연장 포함) |
| **HOT-RETURN** | 게이트 전부 PASS ∧ 전 s∈J 가 HOT_s(r ≥ 2/3, k=3–5) | 해제 후 솔버 반복이 뜨거운 기준으로 돌아갔다 — 「차가운 물리 상태가 불안정」과 동일시 금지(DJDN 은 뜨거운 x_map 을 1반복에 받을 수 있고, 이동은 시간진화가 아니다 — astra 2) · 쌍안정 「없음」이 아님 | r_s(2)(해제 반복)와 iter0002 의 S II 엣지 J/B·Γ 판독 → 원천(EUV 채움) 표적 |
| **DRIFT** | 게이트 전부 PASS ∧ 그 밖 | 「미결」로 묶지 않고 **지도**로 판독: 셸별 분류(COLD/HOT/MID)×방향(UP/DOWN/FLAT)×셸군(내측 10–15 / 외측 19–30) 표 필수. 해석 규칙(사전등록): 내측 HOT·외측 COLD(또는 그 역) = 셸 의존 유지 → EUV 창 위치 판독; 전 셸 MID+UP = 진행 중 복귀(ITER 연장 후보); MID+FLAT = 창 부족(연장 후보); 그 밖 = 기재 | 지도 판독 후 user 결정(ITER 연장 여부 포함) |
| **PIN-FAIL** | 고정 창(iter<2) 준위 해 실패로 MODEL_FAILED(§4 G-RUN 의 구분) | 공개 단계 총량을 고정한 준위 방정식이 그 반복의 J 아래 비음수 해를 갖지 않았다 — 솔버 사실이지 「차가운 상태 불가」 물리 주장 아님 | 실패 진단(셸·이온·lo/hi 목표 대 solved 합) 판독 후 user 결정 |
| **X** | 게이트 FAIL 하나라도(PIN-FAIL 아닌 실패 포함) | 미시연 | 계측만 수리, 같은 계약·같은 델타로 1회 재실행 |

## §6. 자원·분장·주장 상한·집행 기록
- **GPU 1런**: partition h100,a100(h200 금지), 프로필 h100-a100x2, `--time 16:00:00`, 예상 12.3–13.5 h(398939·run1 실측; 고정 창은 반복 수를 늘리지 않는다). 재발주 규칙: XSHELL 캡처 게이트(CAPTURE_PRE/POST/VIEW)로 MODEL_FAILED 면 X — 같은 델타로 1회 재실행; `LUMINA_XSHELL_RESPONSE=0` 전환은 base(x_hot 계보)를 바꾸므로 이 계약 안의 예외가 아니라 **계약 개정(user 결정)**이다(G-ID 와 정합, astra 7). 고정 창 준위 해 실패 = PIN-FAIL(재발주 아님). 그 밖 실패는 계측 수리 후 1회. 런 전 CPU: `selftest_a2_10_radeq`(grammar-debug)·판독기 NC 3종.
- **프리플라이트 합성 root 필요**: docs·CLAUDE.md·data·scripts·validation → repo, src·tests → 동결 트리(§0-6 확정본).
- **분장(개정25-n)**: 기안 Fable fresh **완료**(이 본문) · 기계 게이트·물리대수 사전감사·재감사 astra · Q-GOAL/Q-LEAN grok · 구현 Opus 5.5(중첩 CLI, ID 확인) · 코드 감사 Fable fresh(⚠자기 계약 편향 기재) · 발주·집행·판독·판정 운전자 · 종료감사 Fable+astra — 실제 열은 집행 시 기입.
- **Q-GOAL 자기답**: ① 지도 칸 = sh10–30 S 이온화(공개 대비 2–4 dex, 09-24 판독) — 이 단은 그 칸을 「줄이는」 게 아니라 **차가운 공개 상태에서 출발했을 때 현 방정식·솔버가 그 상태를 유지하는지(이력 의존의 한 관측)**를 잰다 — 「초기화/이력 대 방정식을 가른다」는 6반복·미수렴 x_hot 창으로는 과대(astra 7) ② 산출 = 물리 측정(r·셸별 분류 지도) ③ 대안 = EUV 채움 원천 수사(09-17 후보 B)·NT ON(재점검 1순위, grok ③) — 전자는 HOT-RETURN 뒤의 다음 걸음이고, 후자는 이 단의 NT OFF 조건부를 푸는 별개 레버(이 단이 대체하지 않음; NT 가 차가운 상태의 필요조건이면 HOT-RETURN 을 EUV 수사로 넘기는 오판 경로가 남는다 — 판정문 기재) ④ 건너뛰면 NO-COLD-ROOT(장 고정) 의 맹점을 안고 원천을 판다.
- **Q-LEAN 자기답**: 물리량 = r_s(k)(18셸×3)+셸별 분류 지도. 소비: applied.csv·djdn x_lag(G-INJ 두 증인, 하나는 코드 자기보고·하나는 런 내 독립) · ion.csv iter0000·0001(G-COLD, r 의 0 점 전제) · ion.csv iter2–5(r) · referents.csv(x_pub,x_hot) · spectral.csv 엣지 bin(장 이완 서술 1곳). 게이트 4(ID/INJ/COLD/RUN)·NC 3 — 개정 1 에서 G-INJ(c)·분기 밖 기록 3종을 **뺐고**(grok L4), G-COLD·NC-COLD·env 1키·잠금 OR 3곳을 **더했다**(astra 1: 이것 없이는 r 의 0 점이 차갑지 않아 측정 자체가 서지 않는다). 준위 LTE 강제·OFF GPU 재런·XSHELL 제거·새 폐합 규칙은 여전히 없다. 걸음 수 = 1런.
- **주장 상한**: 「398939 계보에서 SEED·반복 0–1 의 S 이온 단계 총량(단계·준위 합)을 공개 분율에 고정하고 반복 2 에 풀면, 반복 3–5 의 J 셸 S III/S II 는 차가운/뜨거운 기준 사이 r=…, 공개 대비 d=… dex 에 있었다」까지. COLD-STAY 여도 「공개 CMFGEN 재현」·「쌍안정 확정」·「차가운 가지 안정」이 아니고, HOT-RETURN 이어도 「쌍안정 없음」·「차가운 상태 불안정」이 아니다(해제 후 이동 = 솔버 반복). 고정 창의 준위 해는 legacy 두 보존 행 경로라 그 창의 준위 분포는 주장하지 않는다. NT OFF·twin-T_e 핀·DJDN 솔버·ALI 8·미수렴 x_hot 조건부.
- **집행 기록 뼈대**: 1 기안 · 2 base 소스 확정(§0-6)·프리플라이트 rc(합성 root) · 3 astra 사전감사·grok Q-GOAL/Q-LEAN · 4 구현·selftest·NC 3종 · 5 코드 감사(첫 항목 봉인 접촉 가부) · 6 빌드·스테이징·referents 봉인·제출(job id·노드·partition) · 7 판독(G-ID/INJ/RUN·r 표·분기) · 8 판정·종료감사(Fable+astra).
- **집행 1** 기안: Fable `claude-fable-5-1`, 2026-09-24.
- **집행 2a** base 소스 확정(운전자, 2026-09-24): 398939 의 빌드 신원 `run2/input/run_build_identity.json`(cuda_build9, 186 소스 파일 sha)을 현 동결 트리와 전수 대조 — **불일치 0**. §0-6 이 지목한 `current_source_identity.json` 은 09-12 스테이징 내용이 입력 디렉토리 복사로 따라온 레거시 파일이며 빌드 신원이 아니다 ⟹ **base = 현 동결 트리**(금지구간 sha 8d3e3d76 유지).
- **집행 2b** 프리플라이트(운전자, grammar-debug): 합성 root `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/preflight_root`(docs·CLAUDE.md·data·scripts·validation→repo, src·tests→동결 트리) — prereg·referents·canonical **전건 rc=0**(changeset files=5·rules=4 residual=X·refs=15).
- **집행 3** 사전감사 1회차(2026-09-24): astra 물리·대수 «폐합 차단» 2(차가운 초기 상태 미보장·전하 폐합 대상 오기)·«수리 요» 3(경계·구성상 참/게이트 모순·base 주장) — `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/astra_out.txt`; grok Q-GOAL 부합·Q-LEAN «폐합 차단» 0·L4 삭제 후보 G-INJ(c)·분기 밖 기록 — 같은 디렉토리 `grok_out.json`.
- **개정 1 — 사전감사 1회차 전건 반영**(기안 Fable fresh `claude-fable-5-1`, 2026-09-24; OLD/NEW 쌍 적용·행 단위 diff 로 의도 외 소실 0 확인 = 운전자): 설계를 pin-then-release 로 교체(§2: env 2키·armed = current_iter < N_pin·준위 잠금 판정식 3곳 OR·N_pin=2), §0-3 코드 사실(잠금 노브 재사용 불가·혼합 전하합·bf 인구 출처) 추가, G-COLD·NC-COLD 신설, G-INJ(c) 삭제, 절대 조건 d ≤ 0.5 dex·셸별 분류·DRIFT 해석 규칙·PIN-FAIL 분기, NC-HOT 분류식 시험 명시, 0/0 셀 정확 영 검사, DR-ON·「가른다」·XSHELL 예외 문면 정정, Fe 상한 9.3534e-10. 재감사(astra 2회차·grok 재발주) 결과는 집행 4 에. 이 개정으로 prereg-preflight(rules 5·refs 15·changeset files 5) 재실행 필요.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-COLDSTART (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)|scripts/[A-Za-z0-9_/.]+\\.py)",
    "symbol": "LUMINA_INIT_IONFRAC_DIR",
    "roots": ["src", "tests"],
    "expected_extra": ["src/lumina_plasma.c", "src/lumina.h", "src/env_universe.h", "tests/a2_10_radeq_selftest.c", "scripts/coldstart_judge.py"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "cold": "sum(1 for x in v if 0.5 <= x < 1.5)", "hot": "sum(1 for x in v if 1.5 <= x < 2.5)", "pin_fail": "sum(1 for x in v if 3.5 <= x < 4.5)"},
    "rules": [
      {"name": "PIN-FAIL", "predicate": "count > 0 and pin_fail > 0"},
      {"name": "X", "predicate": "count == 0 or (pin_fail == 0 and gate_fail > 0)"},
      {"name": "COLD-STAY", "predicate": "count > 0 and pin_fail == 0 and gate_fail == 0 and cold == count"},
      {"name": "HOT-RETURN", "predicate": "count > 0 and pin_fail == 0 and gate_fail == 0 and hot == count"},
      {"name": "DRIFT", "predicate": "count > 0 and pin_fail == 0 and gate_fail == 0 and cold < count and hot < count"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []}, {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 1]]},
      {"name": "all-cold", "mix": [[1.0, 1]]}, {"name": "all-hot", "mix": [[2.0, 1]]},
      {"name": "split", "mix": [[1.0, 1], [2.0, 1]]}, {"name": "band", "mix": [[3.0, 1]]},
      {"name": "pin-fail", "mix": [[4.0, 1]]}
    ],
    "residual": "X"
  },
  "references": [
    {"path": "docs/PREREG_DET_PATHB_XSHELL_2026-09-22.md"},
    {"path": "docs/VERDICT_DET_PATHB_XSHELL_2026-09-24.md"},
    {"path": "docs/VERDICT_DET_DR_ON_2026-09-19.md"},
    {"path": "docs/RECHECK_SELFRUN_CMFGEN_DEPENDENT_VERDICTS_2026-09-24.md"},
    {"path": "docs/READOUT_S_IONIZATION_EUV_LTE_2026-09-17.md"},
    {"path": "CLAUDE.md"},
    {"path": "data/standart_data1/toy06/ionfrac_s_toy06_cmfgen.txt"},
    {"path": "data/standart_data1/toy06/phys_toy06_cmfgen.txt"},
    {"path": "scripts/run_a2_10_selftest.py"},
    {"path": "src/lumina_plasma.c", "flags_existing": ["lumina_init_seed_material_predictor", "compute_ion_populations_shell", "nlte_population_candidate_build_bundle", "nlte_ne_closure_candidate", "pop_djdn_log_newton", "NLTE_NE_CLOSURE_RELATIVE_TOLERANCE", "nlte_ion_lock_active", "nlte_per_ion_rescale_active", "nlte_ne_closure_charge", "nlte_pair_total_density", "nlte_solve_all_impl"]},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_run", "physics_comparison_dump_if_requested", "lumina_init_seed_material_predictor"]},
    {"path": "src/lumina_main.c", "flags_existing": ["nlte_start_iter", "LUMINA_PURE_CMFGEN_ITER"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_XSHELL_RESPONSE", "LUMINA_NLTE_START_ITER", "LUMINA_NLTE_ION_LOCK", "LUMINA_NLTE_LOCK_START_ITER", "LUMINA_NLTE_PER_ION_RESCALE"]},
    {"path": "src/physics_comparison.c", "flags_existing": ["stage_q1_cm3"]},
    {"path": "tests/a2_10_radeq_selftest.c", "flags_existing": ["cmfgen_xshell_production_selftest"]}
  ]
}
```
