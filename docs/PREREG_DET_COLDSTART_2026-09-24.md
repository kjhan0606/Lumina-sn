# 사전등록 — DET-COLDSTART: 공개 CMFGEN 이온분율 초기화 차가운 출발 시험 (쌍안정 판별)

**표적:** Lumina 이온 인구를 공개 StaNdaRT CMFGEN toy06 19.480 d 이온분율(차가운 가지)로 초기화한 뒤 생산 반복을 돌렸을 때 sh10–30 의 S III/S II 가 차가운 값 근처에 머무는가(COLD-STAY), 뜨거운 가지로 돌아가는가(HOT-RETURN). 상태 = **기안**(실행 승인·구현·판정 아님). 기안자 = Fable fresh(`claude-fable-5-1`, 개정25-k). 최종목적 = MC 수송으로 CMFGEN 수준 SN Ia 스펙트럼 재현. 잣대 = **공개 파일만**(자체 CMFGEN 런 인용 금지, 재점검 목록 §0).

## §0. 선행 사실 (저자 실행 = 로그인 노드 비연산 조회; 값은 referent, 예측 아님)
1. **공개 파일**: `data/standart_data1/toy06/ionfrac_{si,s,ca,fe,co,ni}_toy06_cmfgen.txt`, 블록 `#TIME:    19.480`(S 파일 행 2841–2943), `#NVEL: 100`, vel_mid 1,025–40,325 km/s; 열 = `s0..s4`(Si/S/Ca = I..V) 또는 `fe0..fe5`(Fe/Co/Ni = I..VI). 중성 열은 1e-99(S·Si·Ca·Co·Ni) 또는 ≤7e-10(Fe). sh24 부근 S: v21,630 (0.5726, 0.4145) x=0.724 · v22,335 (0.5259, 0.4569) x=0.869. 공개 phys 19.480: T·ρ·n_e·n_atom(v21,630: 9,231 K·1.36e7·7.73e6).
2. **Lumina**: `geometry.csv` 50 셸(v_in/v_out cm/s; sh9 10,452–11,180 · sh24 21,372–22,100 · sh30 25,740–26,468 km/s). `active_ions.csv` 27 이온 = Si/S/Ca II–V, Fe/Co/Ni II–VI — 공개 최상단 단계와 **일치**, Lumina 에 없는 공개 단계는 **I 뿐**, 공개에 없는 Lumina 단계 없음. 쌍둥이 T_e·침적은 공개 파생(재점검 §2).
3. **초기화 경로(코드, 동결 트리)**: `cmfgen_run`(lumina_cmfgen.c:8767) pass 0 = init pass: bootstrap Saha 물질(pop gen 1; lumina_main.c:259–266 창) → assemble → solve_J(rad gen 1) → `lumina_init_seed_material_predictor`(lumina_plasma.c:17422, 호출 :9069) → `nlte_population_candidate_build_bundle`(:22615; NE_CLOSURE 루프 ≤32회·tol 1e-8·감쇠 0.5) → `prepare_through_core`(:22582: begin→opacity view→ionization_at_ne→precore tau→solve_core) → `nlte_ne_closure_candidate("SEED")`(:21236) → `commit_seed_material`(:23131) = **pop gen 2**. 논리 반복 k: assemble(gen k+2 물질) → solve_J → `lumina_r7_publish_and_solve_te` → 커밋 gen k+3 → `physics_comparison_dump_if_requested` iter k(ion.csv population_generation = k+3, 398939 iter0000 gen 3 실측). 이온 단계 해 = `compute_ion_populations_shell`(:3566) 이 원소별 인접단계 비로 n_element 를 분배해 `atom->ion_number_density[(ip_start+k)*n_shells+s]` 에 씀(:3800–3822); n_e 는 `compute_electron_density`(:3878) 가 같은 셸 해를 되풀이하며 Σq·n_ion 으로 폐합. ion.csv `Z16_stage_q1/q2` = 전하 1/2 = S II/S III(physics_comparison.c:445).
   ⚠`LUMINA_NLTE_START_ITER` 는 lumina_main.c:819/844 의 MC 루프만 소비하고 PURE_CMFGEN 경로는 읽지 않는다(cmfgen.c·plasma.c grep 0건) ⟹ DET 레인에 「NLTE 시작 반복 2」는 없다: 준위 NLTE 해는 SEED 부터 매 반복.
4. **런 내 증인**: `lumina_pop_djdn.csv`(열 gen,shell,Z,stage,x_lag,x_map,…,epoch): epoch e 의 x_lag = **커밋 gen e 의 비**, x_star = gen e+1. 398939 sh24 Z16: epoch1 x_lag 1044.7(bootstrap Saha)→x_star 81.36(SEED gen 2); epoch2 x_lag 81.36→69.596 = iter0000 ion.csv 와 9자리 일치. ⟹ **epoch-2 x_lag 행 = SEED 커밋 상태**.
5. **대조군·base = 398939**(`/gpfs/kjhan/lumina/cmfgen_map/pathb_xshell_20260922/run2`, a100, ITER 6, 47,833 s, SUCCESS, binary 35c6dbc2; h100 run1 44,224 s). DR-ON 385315 대비 exports 델타 = CONT_NLTE_SOURCE=1·POP_DJDN=1·POP_DJDN_CHI=1·CMAX=1.0·ITER 6·XSHELL_RESPONSE=1; deck/sigma/topion sha·fixed_te_profile(f6db3543) 는 DR-ON 과 동일. **선택 근거**: (i) 가설 기전 S_bf=B/b 는 연속체 소스가 인구에 반응해야 존재한다 — DR-ON 은 S_fixed=χ·B(T_e)(09-17 판독 A) 라 b 가 장에 못 들어가 쌍안정이 **원리적으로 닫혀 있고**, 발주서 referent(S_bf/B 22·b≈0.05·x_lag 134)도 이 계보다; (ii) 6반복 뜨거운 궤적을 가진 유일한 완주 런; (iii) DJDN 은 Picard 몫 ln ρ 를 감쇠하지 않고 Newton 보정 |c|≤C_max 만 캡(`pop_djdn_log_newton`, :2591–2600) ⟹ 유지 안 되는 차가운 상태는 1반복 안에 되돌아갈 수 있다(감쇠가 복귀를 숨기지 않음). 단일 변수 = exports 델타 `+LUMINA_INIT_IONFRAC_DIR` 1키.
6. **동결 트리 ≠ 398939 스테이징 소스**: `run2/input/current_source_identity.json` 의 4파일(lumina.h·lumina_plasma.c·nlte_population_candidate.h·env_universe.h) sha 가 트리 현행과 불일치(트리 mtime 09-22 12:44–13:01 < 스테이징 09-23 10:23). 운전자가 발주 전 **빌드 base = 398939 바이너리 소스**를 확정해야 G-ID 가 선다(불일치면 base 계보 캐비앗 기재). 금지구간 lumina_cmfgen.c:2848–3212 sha 8d3e3d76 확인.
7. **뜨거운 가지는 수렴 상태가 아니다**: 398939 sh24 x = 69.6/73.7/66.5/133.6/109.4/150.1(반복 0–5); sh16 8.97→2.53, sh18 16→2.02, sh30 723→42,858→10,498(관측). 그래서 판정 셸을 §5 규칙으로 고른다.

## §1. 물리 계약 1개
**질문:** 「398939 와 동일한 계보(같은 봉인 입력·exports·솔버)에서 SEED 커밋의 이온 단계 밀도만 공개 CMFGEN 19.480 d 분율로 바꿔 6 반복을 돌리면, 판정 셸 J 의 S III/S II 가 반복 3–5 동안 차가운 기준(공개) 쪽 1/3 안에 머무는가(Lumina 방정식이 차가운 가지를 지지), 뜨거운 기준(398939 반복 3–5) 쪽 2/3 밖으로 돌아가는가?」
**말하지 않는 것:** 공개 CMFGEN 재현·어느 가지가 물리적으로 옳은가·쌍안정의 존재/부재 일반론·NT/시간의존 효과·Si/Ca/철족 판정(기록만)·자유 T_e.

## §2. 초기화 규칙과 주입 설계 (env 1개, 기본 OFF)
- **env** `LUMINA_INIT_IONFRAC_DIR` = 6 공개 파일 사본 디렉토리(스테이징에 복사, sha256 = repo 원본). 미설정 = OFF: getenv 1회 외 어떤 문장도 실행되지 않는다(byte-동일의 근거 = 진입 조건 하나).
- **속도 보간**: 셸 v_mid=(v_in+v_out)/2. 19.480 블록에서 v_mid 를 감싸는 두 행(i,i+1)으로 **각 단계 k 의 ln f_k 를 ln v 에 선형 보간**, 그 뒤 Σ_k f_k=1 재규격화. 범위 밖(v_mid<1,025 또는 >40,325; 현 격자엔 없음)·블록 부재·열 수 불일치 = fail-closed 종료(floor·대체 없음).
- **단계 사상**: Lumina II ← 공개 I+II(합, 총량 보존); III..최상단 ← 공개 같은 단계. 원소별 「공개 단계 수 = Lumina 단계 수 + 1」 검사, 아니면 종료.
- **주입 지점(1곳)**: `compute_ion_populations_shell` 이 원소 e·셸 s 의 단계 밀도를 쓴 직후(:3800–3822 블록 끝, 같은 호출) — **armed 일 때만** n_ion(k) ← f_k·n_element(n_element = 그 호출의 율 해가 분배한 총량 ⟹ 원소 총량 보존). armed 는 `lumina_init_seed_material_predictor` 가 build_bundle 호출 전 세우고 반환 전(실패 경로 포함) 내린다 ⟹ SEED 후보의 n_e 폐합 루프·ionization_at_ne 서브반복 전부가 같은 주입 해를 되받아 **n_e → Σq·n_ion(주입) 으로 기존 NE_CLOSURE 가 폐합**(새 폐합 규칙 없음). 논리 반복 0 부터 armed=0 = 생산 율 해 그대로(그것이 시험).
- **준위**: SEED 후보의 기존 준위 단계(solve_core)가 주입된 단계 총량으로 규격화한 분포를 쓴다 — LTE 강제 코드를 넣지 않는다(⚠발주서의 「LTE(Boltzmann)」 규칙과 다름: Γ_ion 은 이미 LTE 준위 가중(S1 §0-5)이고, 강제 코드는 표면만 늘린다). 판독기가 SEED 준위의 S II 바닥 분율을 기록(관측).
- **증거 writer**: armed 마지막 회차의 값을 `lumina_init_ionfrac_applied.csv`(shell,Z,stage,v_mid,f_applied,n_ion,n_element) 로 cwd 에 1회 기록.
- clamp·floor·부호 교정 없음. `LUMINA_ENV_STRICT=1` 이므로 env 이름을 `env_universe.h` 에 등록.

## §3. 변경집합과 산출물
### 변경집합 — DET-COLDSTART (표)
| 경로 | 분류 | 허용 변경 |
|---|---|---|
| `src/lumina_plasma.c` | 주입 | 공개 파일 로더·보간·단계 사상(static), armed 플래그·주입 지점 1곳, applied.csv writer; 율 해·n_e 폐합·커밋·DJDN 산술 무변경 |
| `src/lumina.h` | 테스트 API | 로더·보간 함수 선언 1개(selftest 진입용) |
| `src/env_universe.h` | strict env | `LUMINA_INIT_IONFRAC_DIR` 등록 |
| `tests/a2_10_radeq_selftest.c` | 직접시험 | 케이스 1개: 50셸×6원소 f_applied 표를 stdout CSV 로(판독기 G-INJ(a) 와 같은 비교에 물림) |
| `scripts/coldstart_judge.py` | 판독·판정 | 신규(repo). referent 표·G-INJ·G-RUN·r_s(k)·분기·NC 플래그 |
### 변경집합 끝
`expected_extra` = 위 5파일. 변경 금지: `src/lumina_cmfgen.c`(금지구간 포함)·`src/lumina_cuda.cu`·`Makefile`·봉인 입력·398939 산출(읽기만). 산출: `coldstart_referents.csv`(런 전 생성·sha 봉인) · `coldstart_result.json`(게이트·r 표·분기·입력 해시) · applied.csv · 표준 dump. 런 루트 `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/run1/`.

## §4. 게이트와 음성대조 (허용오차: 산술 항등 상대 1e-6; 물리 판정은 §5 의 r 띠)
- **G-ID**: 봉인 입력 sha(deck/sigma/topion/fixed_te) = 398939 · exports 델타 = `+LUMINA_INIT_IONFRAC_DIR` 하나 · ITER 6·ALI 8·XSHELL_RESPONSE 1 유지 · 소스 = §0-6 확정 base + 변경집합 파일만(sha 대조) · 6 공개 사본 sha256 = repo 원본 · 바이너리 sha 기록 · h200 아님.
- **G-INJ**(주입 재현): (a) applied.csv 의 f_applied = 판독기가 repo 원본으로 §2 규칙을 독립 재계산한 값(전 셸×원소×단계, 1e-6) ∧ n_ion/n_element = f_applied(1e-9); (b) `lumina_pop_djdn.csv` epoch 2·Z 16·stage 1·s∈J 의 마지막 행 x_lag = x_pub,s(1e-6; §0-4 런 내 증인); (c) stderr 에 `[A2-INIT][SEED-MATERIAL]` 1회 ∧ `[A2-07][NE-CLOSURE] commit=SEED` 통과.
- **G-RUN**: terminal_state=SUCCESS·model_rc=0·outer_iterations=6·ion.csv iter0000–0005 실재·gen 3–8·J 셸 Z16 q1,q2 유한 양수.
| NC | 주입 | 반드시 FAIL / 결과 | 시점 |
|---|---|---|---|
| NC-INJ | 사본 디렉토리의 S 파일 19.480 블록 s1 열 ×2(재규격화 없음; 실제 어긋남 ln 2) | selftest CSV 대 판독기 → G-INJ(a) FAIL; 런 후 applied.csv 대 변조 원본 → FAIL | 런 전(selftest)·런 후 |
| NC-HOT | 판독기에 398939 ion.csv 를 「런」으로 투입 | 분기 = HOT-RETURN(전 J 셸 r=1 ±산포) ; x=x_pub 합성 ion.csv 투입 → COLD-STAY | 런 전 |
OFF byte-동일 = 진입 조건 1개(소스 검사, Fable 코드 감사 첫 항목) — OFF GPU 재런(13 h)은 하지 않는다(오프라인 주장, DR-ON 전례와 같이 기재).

## §5. 기대치·분기
**판정 통계**: s∈J, k∈{3,4,5}: r_s(k) = ln(x_s(k)/x_pub,s) / ln(x_hot,s/x_pub,s). x_s(k) = iter k ion.csv `Z16_stage_q2/q1`; x_pub,s = §2 보간의 S III/S II(= 주입 상태의 비 — ln 보간이라 재규격화와 무관); x_hot,s = 398939 반복 3–5 기하평균. 0 = 차가운 초기 상태, 1 = 뜨거운 가지.
**판정 셸 J**(규칙: gap=log10(x_hot/x_pub) ≥ 1.0 ∧ 뜨거운 산포 log10(max/min, 반복 3–5) ≤ gap/3 ∧ 차가운 띠 |log10 x_pub(v_out)/x_pub(v_in)| ≤ gap/3) = **{10–15, 19–30}, 18 셸**. 제외: 9(차가운 띠 0.68 > 0.34, IGE 혼재), 16(산포 0.88 > 0.50), 17(gap 0.84), 18(산포 0.97 > 0.48).
Referent(sh: x_pub / x_hot / gap dex; 저자 계산, 판독기가 `coldstart_referents.csv` 로 재생성해야 함): 10: .241/15.4/1.80 · 11: .181/20.1/2.05 · 12: .186/5.27/1.45 · 13: .197/3.74/1.28 · 14: .210/3.00/1.15 · 15: .233/2.50/1.03 · 19: .349/18.9/1.73 · 20: .392/24.8/1.80 · 21: .490/45.6/1.97 · 22: .572/33.6/1.77 · 23: .696/42.8/1.79 · **24: .744/130/2.24** · 25: .880/369/2.62 · 26: .959/1.51e3/3.20 · 27: 1.10/2.52e3/3.36 · 28: 1.19/3.74e3/3.50 · 29: 1.30/8.75e3/3.83 · 30: 1.43/1.33e4/3.97.
**경계 근거**: J 선정이 뜨거운 가지 자기 산포를 gap/3 이하로 묶으므로 [1/3, 2/3] 띠 = 해상도 1단위를 log 중점(대칭 두 우물 그림의 분리선 1/2) 둘레에 둔 것. 반복 창 3–5 = 뜨거운 referent 와 같은 반복 지수(ALI·DJDN 상태 동지수 비교). ITER 6 근거: (i) Picard 몫 무감쇠라 복귀는 1반복이면 보이나 (ii) 준위 NLTE 되먹임(S II 바닥 b)이 커밋을 거쳐 장에 들어가는 데 ≥2반복, (iii) 대조군과 같은 창. ITER 4 는 창이 1반복이 되고 동지수 비교가 깨진다(GPU 차 ≈4 h).
**구성상 참 등록 없음**: r_s(k) 는 런 없이 못 얻는다. 방향 예측을 등록하지 않는다(갈림길 시험). 기록(분기 밖): SEED n_e 대 공개 n_e(v_mid) 비 · Si/Ca/Fe 의 r · 반복 0–2 궤적 · S II 바닥 분율.
| 분기 | 판정식 | 뜻 | 다음 |
|---|---|---|---|
| **COLD-STAY** | 게이트 전부 PASS ∧ max_{s∈J,k} r_s(k) ≤ 1/3 | Lumina 방정식이 이 초기화·6반복·솔버에서 차가운 가지를 지지 — 가지 = 이력/초기화 | 뜨거운 가지 대 차가운 가지의 스펙트럼·EUV 창 J/B 대조(런 0회 판독) 후 user 결정 |
| **HOT-RETURN** | 게이트 전부 PASS ∧ min r_s(k) ≥ 2/3 | 이 초기화·반복·솔버로는 차가운 상태 유지 안 됨 — 쌍안정 「없음」이 아님 | 복귀 1반복째의 S II 엣지 J/B·Γ 판독 → 원천(EUV 채움) 표적 |
| **DRIFT** | 게이트 전부 PASS ∧ 그 밖(띠 안, 또는 셸 분할·단조 이동) | 미결; **셸별 r 지도(3 반복)** 기재 필수 | 지도 판독 후 user 결정(ITER 연장 여부 포함) |
| **X** | 게이트 FAIL 하나라도 | 미시연 | 계측만 수리, 같은 계약 1회 재실행 |

## §6. 자원·분장·주장 상한·집행 기록
- **GPU 1런**: partition h100,a100(h200 금지), 프로필 h100-a100x2, `--time 16:00:00`, 예상 12.3–13.5 h(398939·run1 실측). 재발주 규칙: XSHELL 캡처 게이트(CAPTURE_PRE/POST/VIEW)로 MODEL_FAILED 면 X → `LUMINA_XSHELL_RESPONSE=0` 으로 1회 재발주(델타 2키, 기재); 그 밖 실패는 계측 수리 후 1회. 런 전 CPU: `selftest_a2_10_radeq`(grammar-debug)·판독기 NC 2종.
- **프리플라이트 합성 root 필요**: docs·CLAUDE.md·data·scripts·validation → repo, src·tests → 동결 트리(§0-6 확정본).
- **분장(개정25-n)**: 기안 Fable fresh **완료**(이 본문) · 기계 게이트·물리대수 사전감사·재감사 astra · Q-GOAL/Q-LEAN grok · 구현 Opus 5.5(중첩 CLI, ID 확인) · 코드 감사 Fable fresh(⚠자기 계약 편향 기재) · 발주·집행·판독·판정 운전자 · 종료감사 Fable+astra — 실제 열은 집행 시 기입.
- **Q-GOAL 자기답**: ① 지도 칸 = sh10–30 S 이온화(공개 대비 2–4 dex, 09-24 판독) — 이 단은 그 칸을 「줄이는」 게 아니라 **원인 축(초기화/이력 대 방정식)을 가른다** ② 산출 = 물리 측정(r 지도) ③ 대안 = EUV 채움 원천 수사(09-17 후보 B) — 그것은 HOT-RETURN 뒤의 다음 걸음이고, 쌍안정이면 원천 수사가 헛돈다 ④ 건너뛰면 NO-COLD-ROOT(장 고정) 의 맹점을 안고 원천을 판다.
- **Q-LEAN 자기답**: 물리량 = r_s(k)(18셸×3). 소비: applied.csv·djdn x_lag(G-INJ 두 증인, 하나는 코드 자기보고·하나는 런 내 독립) · ion.csv(r) · referents.csv(x_pub,x_hot). 게이트 3(ID/INJ/RUN)·NC 2 — 미소비 계측 없음; 준위 LTE 강제·OFF GPU 재런·XSHELL 제거를 **뺐다**. 걸음 수 = 1런.
- **주장 상한**: 「398939 계보에서 SEED 이온 단계를 공개 분율로 두면 반복 3–5 의 J 셸 S III/S II 는 차가운/뜨거운 기준 사이 r=… 에 있었다」까지. COLD-STAY 여도 「공개 CMFGEN 재현」·「쌍안정 확정」이 아니고, HOT-RETURN 이어도 「쌍안정 없음」이 아니다. NT OFF·twin-T_e 핀·DJDN 솔버 조건부.
- **집행 기록 뼈대**: 1 기안 · 2 base 소스 확정(§0-6)·프리플라이트 rc(합성 root) · 3 astra 사전감사·grok Q-GOAL/Q-LEAN · 4 구현·selftest·NC 2종 · 5 코드 감사(첫 항목 봉인 접촉 가부) · 6 빌드·스테이징·referents 봉인·제출(job id·노드·partition) · 7 판독(G-ID/INJ/RUN·r 표·분기) · 8 판정·종료감사(Fable+astra).
- **집행 1** 기안: Fable `claude-fable-5-1`, 2026-09-24.
- **집행 2a** base 소스 확정(운전자, 2026-09-24): 398939 의 빌드 신원 `run2/input/run_build_identity.json`(cuda_build9, 186 소스 파일 sha)을 현 동결 트리와 전수 대조 — **불일치 0**. §0-6 이 지목한 `current_source_identity.json` 은 09-12 스테이징 내용이 입력 디렉토리 복사로 따라온 레거시 파일이며 빌드 신원이 아니다 ⟹ **base = 현 동결 트리**(금지구간 sha 8d3e3d76 유지).
- **집행 2b** 프리플라이트(운전자, grammar-debug): 합성 root `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/preflight_root`(docs·CLAUDE.md·data·scripts·validation→repo, src·tests→동결 트리) — prereg·referents·canonical **전건 rc=0**(changeset files=5·rules=4 residual=X·refs=15).

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
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "cold": "sum(1 for x in v if 0.5 <= x < 1.5)", "hot": "sum(1 for x in v if 1.5 <= x < 2.5)"},
    "rules": [
      {"name": "X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "COLD-STAY", "predicate": "count > 0 and gate_fail == 0 and cold == count"},
      {"name": "HOT-RETURN", "predicate": "count > 0 and gate_fail == 0 and hot == count"},
      {"name": "DRIFT", "predicate": "count > 0 and gate_fail == 0 and cold < count and hot < count"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []}, {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 1]]},
      {"name": "all-cold", "mix": [[1.0, 1]]}, {"name": "all-hot", "mix": [[2.0, 1]]},
      {"name": "split", "mix": [[1.0, 1], [2.0, 1]]}, {"name": "band", "mix": [[3.0, 1]]}
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
    {"path": "src/lumina_plasma.c", "flags_existing": ["lumina_init_seed_material_predictor", "compute_ion_populations_shell", "nlte_population_candidate_build_bundle", "nlte_ne_closure_candidate", "pop_djdn_log_newton", "NLTE_NE_CLOSURE_RELATIVE_TOLERANCE"]},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["cmfgen_run", "physics_comparison_dump_if_requested", "lumina_init_seed_material_predictor"]},
    {"path": "src/lumina_main.c", "flags_existing": ["nlte_start_iter", "LUMINA_PURE_CMFGEN_ITER"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_XSHELL_RESPONSE", "LUMINA_NLTE_START_ITER"]},
    {"path": "src/physics_comparison.c", "flags_existing": ["stage_q1_cm3"]},
    {"path": "tests/a2_10_radeq_selftest.c", "flags_existing": ["cmfgen_xshell_production_selftest"]}
  ]
}
```
