# 하드코딩·cap/floor 전수조사 — 파트 C: 숨은 곳 전-코드 스윕 (2026-07-29)

- 임무: partA(52항목)·partB(36항목)·ADVERSARIAL(신규 7건)의 **그물을 빠져나간 것**만 찾기. 거기 있는 항목은 재보고하지 않음(내용 기준 대조).
- 기준선: **parity42** (`logs/coevolve_consume_parity42/`, RESOLVED CONFIG 112 vars 전문 재확인). 읽기 전용 — src/ 무수정.
- 층 판별식: memory/feedback_clamps_are_not_physics_fix_the_solver.md (①제약 ②위생 ③답 대체).
- **memory grep 수행 확인**: SUPER_CUTOFF·FB_MULTI/kpacket_fb·J_CAP_FACTOR/J_NU_UV_CAP/W_CAP/BINNED_J/FIXED_TRAD·TRIPWIRE/RESID_CHECK/EQUILIBRATE·VSPEC·interpolate_zeta/ZETA_OVERRIDE 전부 `~/.claude/projects/.../memory/*.md`에 grep — 히트 항목은 **[기지-메모리]**(메모리에는 있으나 clamp census에는 누락), 미히트만 **[신규]**.
- 발화 실측 소스: parity42 stdout/stderr 배너, repo 루트 DIAG-T2 census CSV(`lumina_census_kpkt_exit.csv`·`lumina_census_emission.csv`, mtime 07-29 11:09 = parity42 종료 직전 기록; parity43은 15:42 종료로 이 파일들을 아직 안 덮음 — **귀속 주의**: 재확인은 파일 mtime으로).

## A. 신규 항목 표 (census 밖에서 발견된 것)

### C-1. 활성 (parity42에서 실제로 답에 관여)

| # | 사이트 [내용앵커] | 값 | 층 | 활성? | 발화 실측 (parity42) | 편향 방향 | 처분 |
|---|---|---|---|---|---|---|---|
| H1 | [기지-메모리★★] **k-packet fb 채널 = Kramers α + 단일 대표 에지**: (확률) `alpha = 2.6e-13*stage²*(Te/1e4)^-0.75; C_fb += alpha*n_ion*ne*kTe` (plasma.c, compute_transition_probabilities kpacket 블록) + (방출) `kpacket_fb_nu[s] = dom_edge_nu`(최대-n_ion 이온의 지상 에지 하나) → 커널 -3 exit `nu_fb = nu_edge − (kTe/h)·ln ξ` (cuda.cu 단일-에지 분기). 실물 Milne+다중에지(FB-MULTI)는 **게이트 뒤에 존재하나 OFF** | 2.6e-13/z²/T^-0.75/에지 1개 | **③성** (수소형 근사가 실물 Milne을 대체; per-ion σ_bf 데이터가 로드돼 있는데 안 씀) | **ON** (KPACKET=1, `[FB-MULTI] p_fb` 배너 0건=real-Milne 경로 미실행 확증) | 최종 iter k-exit census: **fb 3,684 / ff 1,724 / collexc 1,136,516** (fb 점유 0.32%) — fb 광자는 전부 셸당 1개 에지+지수꼬리 스펙트럼. Kramers-vs-Milne p_fb 편차는 이 런 미계측 [정적-미측정] (FB-MULTI 런 배너가 "code-Kramers vs real"을 인쇄하는 계기 존재) | fb-EUV 방출 스펙트럼이 δ(에지)-형상으로 붕괴 + p_fb 자체 오척(방향 미상) — **EUV/FUV 기근 사슬의 방출측 인접 항목** | FB-MULTI(+FB-MILNE C2) A/B 1순위 후보; 최소한 p_fb Kramers/Milne 정적 대조표 |
| H2 | [기지-메모리★] **SUPER_CUTOFF=100 super-level 럼핑**: `level_num>=K → level_super=K` (atomic.c:732-744) + within-SL 재분배 = **Boltzmann@T_e** (within_sl_frac) | K=100 | (ARTIS ION_NLEVELS_EXCITED_NLTE 미러 — 의도적 물리해상도 캡; 층 분류로는 SL 내부 LTE 강제=③성) | **ON** (SUPER_LEVELS=1) | 배너 실측: **"K=100: 21,581 levels lumped"** = 전 준위의 **81.2%** (26,592 중)가 이온당 1개 SL로 붕괴, SL 내부 b_k≡LTE(T_e) 상대 | 고여기 준위의 독립 NLTE 이탈 불능(형광 캐스케이드 상부 해상도 절단) — ARTIS 정합 목적이므로 '결함'이 아니라 '선택'이나, CMFGEN(풀 준위)과의 差 원인 후보 | census 등재 + K 민감도는 CMFGEN 差 조사 시 A/B (지금은 ARTIS-parity 캠페인 목적상 유지) |
| H3 | [신규] **launch_parity42 런처가 `LUMINA_LINE_THERM=1 LUMINA_LINE_THERM_SMAX=49` export** — D4 parity-disable(`!artis_parity_enabled()`)가 유일한 방어선. 게다가 비활성 배너가 "LUMINA_LINE_THERM **unset**"이라고 오표기(cuda.cu:6421) | S→B(T_e) 전셸(≤49) | ③ (켜지면 전 셸 라인 재방출 열화) | **OFF-실효** (ARTIS_PARITY=1이 강제 차단; 배너 확인) | 발화 0 (D4 배너 + "[LTHERM] ... unset" 인쇄) | **잠복 트랩**: ARTIS_PARITY=0 대조런을 이 런처 계열로 만들면 LTHERM이 조용히 켜져 단일변수 A/B가 오염됨. 배너도 set/disabled를 구분 못함 | 런처에서 export 제거 또는 배너를 "set but parity-disabled"로 정정 (env 체인 완전검증 원칙) |
| H4 | [기지-계열] bf 격자 마지막-빈 평탄값: `bin >= n_freq_bins-1 → chi_bf[last]` (host bf_get_chi/bf_get_eta, device d_bf_get_chi 동일) + 격자 밖(λ<100Å, λ>20000Å) → 0 | last-bin | ② | ON | [정적-미측정] — 마지막 빈=100Å 인접이라 도달 자체가 희귀 | 최상단 빈 내부 무보간(미미) | 무해 기록 |
| H5 | [기지-계열] MC J_nu 추정기 창: `comov_nu > nu_min`만 binning (d_update_base_estimators) — λ>20000Å 크로싱은 J_nu에 누락 | 100-20000Å | ② | ON | L9와 동일 뿌리(binned 격자) — partB L9 "창밖 comov 드랍" 문구로 기지 | IR 장 결손(기지 계열) | L9에 흡수 |

### C-2. ARMED-비발화 / 경로-비활성 (게이트는 코드에 실존, parity42에선 안 밟음)

| # | 사이트 [내용앵커] | 값 | 층 | 활성? | 발화 실측 | 편향 | 처분 |
|---|---|---|---|---|---|---|---|
| H6 | [기지-메모리] **JBAR-POPS 트립와이어**: `slb_max > 100 ‖ !isfinite → pops 롤백 + jbar 소비 영구 차단` (cuda.cu:9115-9132, "REVERT + DISABLE jbar") + 잣대 자체에 `x>200‖x<1e-6 continue` 컷 | S/B>100 | **③** (답 롤백 + 이후 소비 차단) | **경로 비활성** — THEN_MC loop-B 전용(자체 로그 41758 "THEN-MC loop only, NOT this pure loop"); coevolve 미통과 | 발화 불능 (배너 0) | (THEN_MC 런에서) 강형광 해를 통째로 되돌림 — 165510 봉쇄 장치의 잔존 | THEN_MC 재사용 시 재심사; census 등재 |
| H7 | [신규] **NLTE_RESID_CHECK**: `‖Ax−b‖/‖b‖ > tol(1e-3) → h_info=88888 → Boltzmann 폴백 라우팅` + **NLTE_EQUILIBRATE** 반복 행/열 균형화 (cuda.cu:598-640, 760-800) | 1e-3 | 라우팅=③성(단 '미해결 표시'형), 균형화=①(정확 변환) | **OFF** (둘 다 default 0, env unset; [NLTE-RESID] 0건) | 0 | (켜면) near-singular 셸을 LTE로 — INV_CEIL과 동일 계열의 별도 그물 | census 등재; 켤 때는 INV_CEIL 장부와 합산 감시 |
| H8 | [기지-메모리] **solve_radiation_field 노브 가족**: `LUMINA_W_CAP`(W 상한) · `LUMINA_BINNED_J_ESTIMATOR`(모멘트→형상적합 대체, T 탐색 [1500,50000]K 브래킷) · `LUMINA_FIXED_TRAD_PROFILE`(T_rad/W 파일 강제) · `LUMINA_FIXED_NE_PROFILE`(n_e 파일 강제, plasma.c:5691) | — | ③ 가족 | **OFF** (전부 unset) + **함수 자체 미호출**(PURE_CMFGEN 경로는 solve_radiation_field를 안 탐) | 0 | — | 폐기 후보 목록에 병합 |
| H9 | [기지-메모리] **binned-J up-rate cap/floor**: `LUMINA_J_CAP_FACTOR`/`LUMINA_J_FLOOR_FACTOR` (J를 factor×W·B(T_rad)로 상하한; plasma.c:2888-2908) + **`LUMINA_J_NU_UV_CAP`** (2000-3500Å J_nu ≤ W_cap·B; nlte_apply_uv_jnu_cap, plasma.c:13086) | env | ③ | **OFF** (전부 unset; project_uv_jnu_cap_closed로 기폐기) | 0 | — | 금지 노브 목록에 병합 |
| H10 | [신규] legacy bf 대역 재방출: 256회 Planck 기각샘플 실패 → **대역 내 균일 ν 발명** (cuda.cu d_bf_absorption_event_band:3404-3410) | 256회/uniform | ② (값 발명이나 대역 유계) | 경로 자체 사장 — bf 흡수는 전량 MA 활성화(D3) 경유 | **emission census `bf_reemit` 열 총합 = 0.0** (etype-8 발화 0 실측) | — | 사장 경로 기록 |
| H11 | [신규] S42 off-diagonal 안정화: `1/fmax(1−Λ*r, 1e-3)`(증폭 ≤1e3 캡) + `M>1→1` (plasma.c:11439-11446) — coupled-Newton/tri-response 전용 | 1e-3 | ② | **OFF** (COUPLED_NEWTON=0, RADEQ_LINE_RE=0; [S42-INERT] 포함 관련 배너 0) | 0 | — | 비활성 기록 |
| H12 | [기지-계열] maser 클램프 미러: `if (coeff < 0) coeff = 0` (IUP-JBLUE up-rate, plasma.c:3821) — partA #13(stim_corr)과 동일 물리의 별도 사이트 | 0 | ① | OFF (IUP_JBLUE unset → jblue_line NULL) | 0 | 역전 증폭 차단 | #13에 병합 |
| H13 | [기지-계열] CPU 수송 잔존물: `ma_iter<5000`(M2 미러)·`loop_count<100000`(M5 기지)·line-id init `lo==n_lines→n_lines−1`·P8 고아준위→공명산란 폴백 (transport.c) | — | ②③ | **OFF** (GPU 런) | 0 | — | M2/M5에 병합 |

### C-3. 정적 정수 캡 (물리 커버리지 잠재 절단 — 현재 전부 비구속)

| # | 사이트 | 값 | 구속 여부 (parity42) | 비고 |
|---|---|---|---|---|
| H14 | `KPKT_FB_NEDGE=16` (lumina.h:35) — FB-MULTI 셸당 fb 에지 수 | 16 | FB_MULTI OFF → 미사용; 켜도 상위 16개 가중치 유지 | H1 수리 시 함께 심사 |
| H15 | `NLTE_BASE_IONS=31 / NLTE_MAX_IONS=38 / NLTE_PAIR_COUNT=23` (lumina.h) — NLTE 네트워크 크기 | 38 | 설계 크기 = 사용 크기 | ADVERSARIAL D-3(네트워크 밖 이온)의 구조적 뿌리 — 기지 |
| H16 | `LUMINA_MAX_COL_IONS=64` (lumina.h:475) — Ω 테이블 이온 수 | 64 | 40 로드 < 64 → 비구속 | — |
| H17 | INJECT2 SED 샘플 `NB>1024→1024` (cuda.cu:7470) | 1024 | N_FREQ_BINS=1000 < 1024 → 비구속 | — |
| H18 | `CE_MAX_REACTIONS=20/CE_N_REACTIONS=17`·`DR_MAX_TERMS=10` (lumina.h) | — | 테이블=설계 크기 | 데이터 커버리지(클램프 아님) |
| H19 | census/이벤트 축: `CENSUS_MAX_SHELLS=256`·EventRec.shell uint8·`nu_comov/energy` float 캐스트·obs `sh[256]`(P7 기지) | 256/f32 | NS=50 → 비구속; float은 계기 정밀도 | 계기 층 |

### C-4. 위생 확정 (신규 발견이나 무해 판정)

| # | 사이트 | 판정 근거 |
|---|---|---|
| H20 | VSPEC v-packet 컷 `tau_total > 50 → return` (cuda.cu:5057) + VSPEC 창밖 무기록 | P_esc<2e-22; virtual 스펙트럼=계기(판정 잣대는 formal). M7 인접 |
| H21 | Bjorkman-Wood Planck 샘플러 `l<=1000` 절단 + `ξ<1e-300` 플로어 (main.c·cuda.cu:5086·plasma.c:6414, 3곳 동일) | Σl⁻⁴ 절단오차 ~1e-12; RNG 가드는 #48 계열 |
| H22 | cmf_solve GPU: `dtau>1e-4` 테일러 분기·수렴잣대 분모 `+1e-30`·비분할 분기 `chih>0?:1.0` (cmf_solve.cu) | 테일러=정확도 스위치; 분모ε는 P5의 기지 잣대결함 내부; 비분할 분기는 ADV_SPLIT=1로 사장 |
| H23 | interpolate_zeta 격자 끝 클램프(T 밖→끝값) + 이온 부재→ζ=1(LTE) (plasma.c:1373-1408) | parity B2/R1 폐로가 ζ 경로를 대체(비parity 전용); ZETA_OVERRIDE 노브는 [기지-메모리, closed] |
| H24 | 분배함수 `boltz<500` 컷·`Z_total<1e-300→1e-300`·bf 빌더 host `n_ion<1e-30 continue`/`U_next<1→1` (plasma.c) | partA #18/#37의 host 미러 — 동일 내용 |
| H25 | 로더 위생: multiplicity `[0,127]` 클램프·col_data `n_temp>256` fail-closed (atomic.c:910, 1142) | fail-closed 모범 / 표현 한계 |
| H26 | equilibration 수렴 `maxdev<1e-3 break`(12회 캡) (cuda.cu:594) | EQUILIBRATE OFF; 정확 변환의 반복 예산 |

## B. 커버리지 통계 ("놓친 게 없다"의 검증 가능 형태)

패턴 스윕 = grep 히트 전수 → {기지(census 내용 일치)/신규(위 표)/무관·위생} 3분류. 히트 수는 grep 라인 수(주석 포함 원시값).

| 파일 | P1 fmin/fmax | P2 센티널(1e-30/-100/-300/1e10/1e28/1e30/1e99) | P3 exp가드(>50..700.0) | P4 문턱 continue/break/return | P5 자기-클램프 `if(x<C)x=C` | P6 #define한계·__constant__ | P7 float/TF32 |
|---|---|---|---|---|---|---|---|
| lumina.h | 0 | 0 | 0 | 0 | 1 | 14 | 2 |
| lumina_cuda.cu | 0 | 37 | 2 | 16 | 69 | 12 | 4 |
| lumina_plasma.c | 5 | 92 | 17 | 70 | 109 | 11 | 1 |
| lumina_cmfgen.c | 3 | 10 | 5 | 4 | 34 | 0 | 0 |
| lumina_atomic.c | 0 | 0 | 0 | 4 | 3 | 0 | 0 |
| lumina_main.c | 0 | 4 | 0 | 0 | 7 | 0 | 0 |
| lumina_transport.c | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lumina_nlte_assemble.cu | 0 | 2 | 1 | 6 | 2 | 0 | 0 |
| lumina_nlte_gemm.cu | 0 | 1 | 0 | 0 | 6 | 0 | 30 |
| lumina_bf_gemm.cu | 0 | 1 | 1 | 2 | 1 | 0 | 29 |
| lumina_cmf_solve.cu | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 헤더 2종(cmfgen.h·radeq_col_pairs.h) | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| **계** | **8** | **148** | **26** | **102** | **232** | **39** | **66** |

- 총 원시 히트 **~621** (패턴 중복 포함). 삼분류 결과: **기지(census 대응) 다수** — cuda.cu P2 37건 전수 열람 → 전부 partA #1-14/#48/#50 및 그 미러(라인 목록은 본 조사 로그에 보존); P7 66건 → 전부 #15(TF32/FP32 스테이징)+계기 캐스트; P3 26건 → #37/#18/F4/C12 계열 exp 가드.
- **신규 = 위 표 H1-H26** (실질 신규 앵커 26개; 이 중 활성 답-관여는 H1·H2·H3 셋).
- **무관 목록(한 줄)**: 로그/CSV 포맷 상수·진행률 인쇄 간격·문자열 버퍼 크기·CUDA 블록/타일 크기(TPB=256, GEMM tile)·bench/selftest 파일(lumina_cmf_selftest.c, cmf_pcygni_b1.c, lumina_radeq_col_pairs_bench.c, bench_*.c — 비생산 경로)·JBLUE-ANCHOR ±3dex 버킷(계기)·EventRec float 캐스트(계기)·D4/E3로 parity-차단된 노브 군(EPS_UV/IR·2STEP·BSRC·KPEMISS 가족 — partA #51-52 계열, 배너로 차단 확인).
- 스크립트 레그: `run_coevolve_s01.sh` 기조사 / `parity_baseline.env` = 게이트 export만(수치 물리 컷 없음, LTE_FLOOR=0 근거문서 포함) / `sbatch_parity_runner.sh` = 스풀 러너(LUMINA 변수 0개) / `launch_parity42_gphoff.sh` = **H3 트랩 1건** 외 전부 RESOLVED CONFIG와 일치.
- 검증 한계 고지: plasma.c(16,943줄)는 라인-by-라인이 아니라 함수-단위 스윕(미커버였던 solve_radiation_field·compute_plasma_state·compute_bf_opacity·compute_transition_probabilities·gamma/rescale/DR·kpacket 빌더 전부 열람) + 패턴 grep의 이중 그물. 패턴에 안 걸리는 형태(예: 곱셈 상수로 위장한 스케일)는 이 방법의 사각 — 0건 주장 불가.

## C. 상위 위험 5 상세

### 1. H1 — k-packet fb: Kramers 확률 + 단일-에지 방출 (활성, 기지-메모리였으나 census 누락)
활성 경로(FB_MULTI OFF)의 fb 채널은 (i) 분기확률 C_fb를 수소형 `2.6e-13·z²·(T/1e4)^-0.75`로 만들고(로드된 CMFGEN σ_bf를 안 씀), (ii) 방출 주파수를 "그 셸에서 n_ion 최대인 재결합 이온의 지상 에지" **하나**에 kT_e 지수꼬리로 얹는다. 최종 iter 실측 fb-exit 3,684건(전 k-exit의 0.32%)이지만, 이 채널이 정확히 **EUV 재결합-에지 광자 제조소**다 — 현 캠페인의 "EUV/FUV 기근 → bf가열·선펌핑 고사" 사슬의 방출측 항이 근사 2중(확률 오척+스펙트럼 붕괴)으로 눌려 있다. 코드 자체가 수리본(FB-MULTI: per-continuum Milne 가중 + 상위 16 에지 CDF + real p_fb 교체)을 갖고 있으므로 A/B 비용이 낮다. 주의: FB-MULTI를 켤 때 [FB-MULTI] p_fb 배너가 Kramers-vs-real을 셸 3곳에서 인쇄 — 이것부터 정적 대조.

### 2. H2 — SUPER_CUTOFF=100: 준위의 81.2%가 SL-내부 LTE (활성, 의도적)
21,581/26,592 준위가 이온당 1개 super-level로 붕괴하고 SL 내부 분배는 Boltzmann@T_e 고정. ARTIS 레시피의 충실한 미러라서 parity 캠페인 안에서는 '옳음'이지만, 층 분류로는 **활성 ③(고여기 준위 b_k≡1 강제)**이고 census에 없었다. CMFGEN(풀 준위)과의 이온화/형광 差를 읽을 때 이 캡이 差의 일부를 나르는지 분리 불가 — parity 종결 후 CMFGEN-差 조사로 넘어갈 때 K 민감도 A/B가 필요한 항목으로 등재해 둔다.

### 3. H3 — 런처 LTHERM 잠복 트랩 (실효 OFF이나 A/B 오염 위험)
launch_parity42가 `LINE_THERM=1, SMAX=49`(=전 셸)를 export하고, 이를 끄는 것은 코드의 D4 parity-disable 하나뿐이다. ARTIS_PARITY=0 대조런을 같은 런처 계보로 만들면 **전 셸 라인 재방출이 조용히 B(T_e)로 열화**되어 "parity 게이트 하나만 바꾼 A/B"가 다변수가 된다. 비활성 배너가 "unset"으로 오표기되는 것(set-but-disabled를 구분 안 함)도 감시자 결함. env 체인 완전검증 원칙(메모리)의 실사례.

### 4. H6 — JBAR-POPS 트립와이어 (경로 비활성이나 ③의 표본)
S/B>100이면 그 반복의 pops를 통째로 되돌리고 **런의 나머지 동안 jbar 소비를 차단**한다(165510 폭주 봉쇄 장치의 잔존). 지금은 THEN_MC loop-B 전용이라 coevolve에 무해하지만, "강형광 해 = 오류"라는 가정이 하드코딩된 전형적 답-대체 장치이고, 문턱(100)은 현 캠페인이 실물로 확인한 초열장(J/B 22-59)과 한 자릿수 거리다. THEN_MC 계열을 재가동하는 순간 재심사 필수.

### 5. H7 — RESID_CHECK/EQUILIBRATE (OFF; 켜기 전 장부 설계 필요)
info=0 near-singular 가비지를 잡는 별도의 그물(잔차>1e-3 → Boltzmann 폴백 라우팅). 지금은 OFF고 parity42의 [NLTE-FALLBACK]=0이라 무주제지만, 켜는 순간 INV_CEIL·grey·singular와 **다른 문턱의 4번째 폴백 원인**이 같은 배너로 합류한다. 켤 때는 원인별 분리 카운터부터(계량→수리 순서).

## D. 미확정 ([정적-미측정] — 계량 부채)

1. **H1 p_fb Kramers/Milne 편차** — FB-MULTI 배너(셸 0/16/49 인쇄)로 정적 계측 가능; 이 런에는 없음.
2. H1 fb-exit의 **에너지 가중** 점유(카운트 0.32%는 건수 기준; erg 기준 미계측 — emission census fb 열은 파장빈별로 존재하나 이번 조사에서 미적분).
3. H2 럼핑이 라인 커버리지에 닿는 정도 — "상부 준위가 SL-100에 속한 라인 수" 미계측 (line_list 조인으로 오프라인 가능).
4. H4 최상단-빈 도달 빈도 — 카운터 없음.
5. census CSV 귀속 — repo 루트 파일이 후속 런에 덮이는 구조(mtime 검증으로만 귀속 가능). 판정용 census는 로그 디렉토리로 복사하는 관행 필요.

## E. 재현 명령 (읽기 전용)

```bash
# H1: 활성 경로 확인 (real-Milne 배너 부재 = Kramers 경로)
grep -c "\[FB-MULTI\] p_fb" logs/coevolve_consume_parity42/stdout.log   # 0
awk -F, 'NR>1{ff+=$2;fb+=$3;ce+=$4}END{print ff,fb,ce}' lumina_census_kpkt_exit.csv  # 1724 3684 1136516
# H2: 럼핑 실측
grep "super-cutoff" logs/coevolve_consume_parity42/stdout.log            # K=100: 21581 levels lumped
# H3: 트랩 확인
grep "LINE_THERM" scripts/launch_parity42_gphoff.sh logs/coevolve_consume_parity42/stdout.log
# H6: 경로 비활성 확인
grep -n "THEN-MC loop only" logs/coevolve_consume_parity42/stdout.log    # 41758
# H10: 사장 확인
awk -F, 'NR>1{s+=$10}END{print s}' lumina_census_emission.csv            # 0
```
