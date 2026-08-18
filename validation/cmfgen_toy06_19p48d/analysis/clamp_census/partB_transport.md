# 하드코딩·cap/floor 전수조사 — 파트 B: 수송·formal·로더 계층

- 기준선: **parity42** (`logs/coevolve_consume_parity42/`, 활성 게이트=stdout.log RESOLVED CONFIG 112 vars). 읽기 전용 조사 — src/ 무수정.
- 판별식: `memory/feedback_clamps_are_not_physics_fix_the_solver.md` (1층=위반불가 제약 / 2층=부동소수점 위생[증폭기 위험] / 3층=답의 대체).
- **memory grep 확인 완료**: fine 창 하드코딩(사례 26, campaign:521)·FINE_SL_CLAMP·S_l=0→B 폴백 사이트·jbar β=성운 τ(사례 23, campaign:388)·linedump 지연 스냅샷·JBAR_MIN 소비사슬(campaign:162)·모드-2 차분 기각 — 전부 [기지]로 표기, 발화 실측만 갱신.
- 범위: lumina_cmfgen.c 전체 / lumina_transport.c / lumina_cuda.cu(MC 수송·업로드·스펙트럼) / lumina_main.c / lumina_atomic.c / lumina_cmf_solve.cu / formal 적분(lumina_plasma.c 내 `compute_formal_integral_spectrum` — 스펙트럼 산출이므로 B 소관; plasma.c 인용은 내용 앵커).

## 표 (사이트 | 값 | 층 | parity42 활성? | 발화 실측 | 편향 | 처분 제안)

### B-1. formal 적분 (판정 잣대)
| # | 사이트 | 값 | 층 | 활성? | 발화 실측 (parity42) | 편향 | 처분 |
|---|---|---|---|---|---|---|---|
| F1 | formal S_l 폴백: `S<=0 → W·B(T_rad)` (`compute_formal_integral_spectrum`, "Fallback dilute-LTE" 주석) | W·B(T_rad) | **3층** | ON(무게이트) | [정적-미측정] — 폴백률 카운터 없음. 단 T_rad=10470 전셸 핀(L5) 위에서 작동 | S_l 미기록 라인을 광구색 연속으로 대체 → 그 파장의 방출 발명 | 폴백률 카운터 추가 후 판정 |
| F2 | formal **S_l·τ 소비점 미스매치**: S=`line_source_S`(NLTE 최종해) × τ=`tau_sobolev`(성운 재기록) 쌍 비보장 + `fi_use_cont=0` 기본(연속 감쇠·방출 없음) | — | **3층(구조)** | ON | **[FORMAL-CONS] L=7.732e43 = 24.99×L_inj** (stdout:41846). 대역분해(본 조사): 2000-3000Å 8.44×L_inj + 3000-4000Å 8.92×L_inj = 초과의 70%; 단일 빈 3342Å 스파이크 = 빈폭 적분 ~9e41 erg/s ≈ 0.3×L_inj. **사전등록 비회귀선(FORMAL-CONS ≤4×, parity34 P3) 위반 상태** | 에너지 위로 발명 — 잣대 자체 비보존 | 상세 §상위-1 |
| F3 | `fi_tau_cutoff=1e-5` (env `LUMINA_FI_TAU_CUTOFF`) | 1e-5 | 2층 | ON(기본) | s8 fine창 τ 중앙값 1.05e-8 ⟹ 대다수 라인 스킵(속도 목적) | (1−e^-τ)≤1e-5 라인 생략 — 무시 가능 | 유지 |
| F4 | `tau_sob>500→1`, `dtau_c>500→1` exp 가드 | 500 | 2층 | ON | [정적-미측정] | 없음(항등 보존) | 유지 |
| F5 | `n_impact=100` 하드코딩 (cuda.cu 호출부 2곳, `compute_formal_integral_spectrum(...,100)`) | 100 | 2층(해상도) | ON | dp=r_outer/100, 50셸 대비 셸당 평균 2 광선 | p-격자 조악 → 코어림(p≈r_phot) 계단 오차 | env화+수렴 A/B |
| F6 | formal 창=출력 격자 `spec_min/max=500/20000Å, 2000빈` (main.c, env `LUMINA_SPEC_RANGE`) | 504.9–19995.1Å | 2층 | ON(기본) | CSV 첫/끝 빈 504.875/19995.125Å 실측 일치. 창 밖 fl은 FORMAL-CONS 적분에서 제외(코드 주석 명시) | 창 절단 vs 과방출 구분은 로그에 인쇄됨 | 유지 |
| F7 | [기지] τ 관측자 창 결함(사례 26): blocking 관측자=linedump=fine 창 1000-4000Å만 | — | 계기 | ON | 덤프 파장범위 1000.01-4000.00Å 재확인. **10000-14000Å 12× 초과대역(parity42 C6)은 구조적 불가시** | 창 밖 결함이 "0"으로 위장 | [기지] — 관측자 창 확장 |

### B-2. cmfgen.c 조립·솔브(binned)
| # | 사이트 | 값 | 층 | 활성? | 발화 실측 | 편향 | 처분 |
|---|---|---|---|---|---|---|---|
| C1 | [기지] binned assemble S_l 폴백: `src_nlte(LUMINA_CMFGEN_SRC_NLTE) 기본 0 ⟹ Sl:=0 ⟹ Sl=B(T_e)` (cmfgen.c:225-227) | B(T_e) | **3층** | ON(게이트 기본 OFF) | **100% 발화** — 게이트 미설정이므로 binned η_line 전량이 B(T_e). 코드 주석 자백: "the B(T_e) fallback (the de-facto thermostat the champion metrics rest on)" | 형광 삭제·열화 적색 편향(기지 확정사슬의 뿌리) | §상위-3 |
| C2 | [기지] fine deposit S_l 폴백 `Sl<=0→B(T_e)` (cmfgen.c:2359/2398)·obs sob `Sl<=0→B` (1284) | B(T_e) | **3층** | ON | **fine창 제로율 실측: s8 20.90% / s45 99.04% / s49 99.04%** (cmf_fine_linedump_s*.csv, S_l==0 비율) | 외곽 셸 방출원 사실상 전면 B(T_e) 대체 | §상위-3 |
| C3 | [기지] `LUMINA_CMF_FINE_SL_CLAMP` (2133/2363-2366, 2402-2405) + obs `g_sob_sl_clamp` (1171/1285) | 0=off | 3층(금지 노브) | **OFF**(미설정) | 발화 0 (clamp off; n_clamped 진단은 FINE_DIAG off라 미인쇄) | — | [기지] 금지 유지 |
| C4 | eps_phys 클램프: `eps_floor=1e-5, eps_cap=1.0`, 테이블 부재 `el<0→1.0(열화)` (cmfgen.c:176-194, 231-234) | 1e-5/1.0 | floor=3층, cap=1층(ε≤1 물리) | ON(`LINE_EPS_PHYS=1`) | [정적-미측정] — 발화 카운터 없음 | floor: 초포화 산란선의 ε을 1e-5로 올려 흡수 발명(소량) / el<0→1: 테이블 밖 라인 전부 열화 | 발화 카운터 심기 |
| C5 | A4 닫힌형 분모 가드 `el+be−el·be+1e-300` (cmfgen.c:246) | 1e-300 | 2층 | ON | 미측정(무해: 가산 가드, el≥1e-5 보장 하) | 없음 | 유지 |
| C6 | 라인 스킵: `tau<=1e-12 continue`(216-218)·`nu_l 창밖(100-20000Å) continue`(220) | 1e-12 | 2층 | ON | fine창 기준 τ≤1e-12 비율: s8 20.6% / s45 99.5%(대부분 τ=1e-100 센티널) | (1−e^-τ)≤1e-12 무시 — 무해 | 유지 |
| C7 | EPAY 군: `EPAY_TAU=2.0`(기본)·`EPAY_TAUBIN=10`(env)·`EPAY_SMIN=5`(env)·`EPAY_HOTF=0`(env⟹hot_regime 항상 참) (cmfgen.c:283-297, 464-515) | τ_es<2, τ_bin>10 면제 | **3층**(보존 강제=1층 목적, 스케일 대체=3층 수단) | ON(`EPAY=2`) | **[CMF-EPAY] scale 12회/런 실측**: s0=0(SMIN 면제) 전 반복; s25 1.39-2.05; s38 0.74-2.32; **s49 0.024-0.214** (외곽 방출을 Kirchhoff 대비 2-21%로 재척도) | 외곽 열방출 강한 하향 — 보존 강제의 실측 크기 | 스케일 시계열을 표준 진단으로 |
| C8 | ALI 분모 가드: 대각 `denom>1e-10 ? ALI가속 : J_fs`(786-788)·tri `dA<1e-10→1e-10`(739/748) | 1e-10 | 2층(증폭 소지) | ON(대각 경로; LAMBDA_TRI off) | [정적-미측정] | Λ*r→1(산란 지배 빈)에서 가속 포기=수렴 지연으로 나타남(값 오염은 아님) | 유지+포기율 카운터 |
| C9 | `J<0 또는 비유한 → 0` (758, 797) | 0 | 1층(J≥0)+침묵 | ON | 카운터 없음 [정적-미측정] | NaN 원인 은폐(0 고정 영구화 주석 자백) | 카운터 추가 |
| C10 | binned 솔브 예산: `ALI_ITER` 기본 8(설정 8)·`ali_tol=1e-3` 기본 | 8/1e-3 | 2층 | ON | 조기정지 로그 없음 | 미수렴시 J 저평가(라그) | 수렴잔차 인쇄 |
| C11 | window_color 대역 5619-6083/9000-11000Å·이분 800-30000K·실패시 t_color=-1 (813-863) | 하드코딩 | 2층 | ON(함수 호출됨) | [정적-미측정] — 소비자 radeq_set_tail_color(파트 A) | 대역 선택 자체가 165584 JDUMP 기반(주석) — toy06 재검증 안 됨 | 대역 재검증 |
| C12 | `cm_planck` x>700→0·`denom<=0→0` (36-43) | 700 | 2층 | ON | — | 없음 | 유지 |
| C13 | inner BB: `INNER_BB_SCALE` (기본 1.0; 설정 1.0) | 1.0 | — | no-op | — | — | — |

### B-3. cmfgen.c fine 생산자·obs 경로
| # | 사이트 | 값 | 층 | 활성? | 발화 실측 | 편향 | 처분 |
|---|---|---|---|---|---|---|---|
| P1 | [기지] **fine 창 `lam_lo=1000/lam_hi=4000` 하드코딩** (cmfgen.c:2120-2122, 게이트 `LUMINA_CMF_FINE_LAMLO/LAMHI` 미설정) | 1000-4000Å | **3층(범위 대체)** | ON | 덤프 실측 1000.01-4000.00Å. **소비자 전수(본 조사)**: ①`jbar_line_det`(창밖=-1 센티널)→plasma.c up-rate 소비자("g_ctp_lineres_jbar && jbar_line_det[..]>=0" 앵커)+pops 소비자("lineres_jbar_pops && opacity->jbar_line_det" 앵커, `LINERES_CONSUME`) — 창밖 라인은 binned/MC-jbar 폴백 ②`FINE_PHOTOION`(OFF — 미설정) ③`FINE_EMERGENT/_OBS`(OFF) ④linedump 계기(사례 26). ⟹ **활성 소비자는 jbar_line_det 하나 + 계기**; NLTE 율의 EUV(400-906Å)·광학(>4000Å) 펌프장은 전부 binned/MC로 구동 | 창 밖 형광·EUV 엔진 구조적 배제 | [기지] — 창 확장 또는 EUV 별도 생산자 |
| P2 | fine 해상도 상수: `vdop=1e6`(10 km/s)·`ppd=12`·`FINE_TAUMIN=1e-12`(전량 deposit)·가우시안 ±4σ+1빈 (2123-2126, 2267, 2350) | 기본값 | 2층 | ON(전부 기본) | NF≈ln4/(vdop/c/12)≈5.0e5 셀; skip_weak=0(TAUMIN 기본) | vdop 10 km/s는 열속도 가정 — 원소별 미분화 | vdop 근거 문서화 |
| P3 | fine 워밍스타트 `fs.J=B(T_e)` (2435) | B | 2층 | ON | — | 미수렴시 B(T_e) 잔재=열화 편향(P5와 결합) | P5와 함께 |
| P4 | [기지] linedump 지연/창 — τ는 소비점 실측으로 수리됨(2026-07-28 컬럼) | — | 계기 | ON | 3셸 리스트 8,45,49 정상 기록 | — | [기지] |
| P5 | **fine GPU 라그 솔버**: `cmf_solve_J_gpu` tol=1e-4 하드코딩(cmfgen.c:1657)·조기정지 `maxrel<tol && it>0`(cmf_solve.cu:323)·라그 어드벡션 O(NB) 수렴 필요 | 1e-4 | 2층(증폭 소지) | ON(`SOLVE_GPU=1`+`ADV_SPLIT=1`+`FINE_ALI=20000`) | **경고 실물 발화**(stderr:5525): "field at n_ali=20000 is likely NOT converged"(필요 O(NB≈5e5)) — 최종 반복의 det jbar 생산 호출 | 라인 코어 J 미수렴 → jbar_line_det 오염 가능; per-iter 변화량 기반 조기정지는 라그 수렴을 오판 가능 | §상위-4 |
| P6 | obs 경로 상수: `NRO=256`·`NObs=3000`·`DVRES=30 km/s`·nsub 캡 256(binned march)/4096(sob march)·`OBS_TAUMIN=1e-6` 기본·백라이트 `Wd=0.5`(r<r_phot) (1119-1121, 1192-1197, 1306-1311, 1407, 1959-1961) | 기본값 | 2층 | ON(전부 기본; 산출=lumina_spectrum.csv obs-frame SOBOLEV) | [정적-미측정] — nsub 캡 도달 카운터 없음 | nsub 캡 도달 시 공명 미해상(적분 오차) | 캡 도달 카운터 |
| P7 | obs `sh[256]` 고정 스택 + `nshell<256` 절단 (1420-1427 등 4곳) | 256 | 1층(NS=50≪256) | ON | 발화 불능 | 없음 | 유지 |
| P8 | cmf_solve_J: `NCORE=16` 코어광선·`cloc[300]/mu[300]` 고정버퍼(NS+1≤300)·CPU 수렴 `maxrel<1e-4 && it>0` (1580-1600, 1690, 1750, 1761) | 상수 | 2층 | ON(GPU 경유) | — | — | 유지 |
| P9 | jbar 추출 `den>0 ? num/den : -1` 센티널 (2527) | -1 | 1층 | ON | 창밖 -1 → 소비자 폴백(P1) | — | 유지 |

### B-4. MC 수송 (cuda.cu 커널·transport.c)
| # | 사이트 | 값 | 층 | 활성? | 발화 실측 | 편향 | 처분 |
|---|---|---|---|---|---|---|---|
| M1 | **상호작용 캡 + 에너지 드랍**: `d_max_interactions`(기본 100000, env 50000)·`d_cap_real_only=0`(경계 횡단도 카운트)·`d_cap_force_escape=0`(캡히트=**에너지 삭제**, `d_E_truncated`) (cuda.cu:2632-2668, 5240, 5761-5785) | 50000 | **3층**(수송해 절단) | ON([CAP] 배너 stdout:208-212) | 최종 반복 escaped=99657/100000, DIFFUSE_INNER_BC=1(재흡수 재방출) ⟹ **≤343 패킷(0.34%)이 캡 절단·에너지 드랍**. per-iter [CAP] 카운터는 coevolve 경로에서 미인쇄 — **에너지 분율 미확정** | 깊은-트랩(UV 형광 캐스케이드 중) 패킷 선택 삭제 → UV 결손 방향 | §상위-2 |
| M2 | MA 내부 캐스케이드 캡 `d_ma_internal_cap=5000` (CPU도 `ma_iter<5000`) | 5000 | 3층(절단) | ON | **[MA-CAP-EXIT] 실측: it0-8=0, it9=1, it10=2, it11=17** (증가 추세; 17/1e5=0.017%) | 캡 탈출=진입 UV 코히런트 재방출(주석 자백) — 미미하나 증가 중 | 추세 감시 |
| M3 | `d_max_total_steps=2000000` 절대 한도 | 2e6 | 1층(안전) | ON | 도달 카운터 없음 [정적-미측정] | — | 유지 |
| M4 | `CLOSE_LINE_THRESHOLD=1e-14`·`MISS_DISTANCE=1e99` (lumina.h:25-26, TARDIS 동일) | TARDIS 패리티 | 2층 | ON | — | TARDIS와 동일 관례 | 유지 |
| M5 | CPU 경로 `loop_count<100000` (transport.c:523) | 100000 | 3층(절단) | OFF(GPU 런) | — | — | 유지 |
| M6 | τ NaN 위생: `!isfinite→1e-100`+카운터 (cuda.cu:9264, loop-B) / τ 하한·센티널 `1e-100` (plasma.c `compute_tau_sobolev` 앵커: skip-Z·ion 미발견·레벨 미발견·산술 하한 4중) | 1e-100 | 2층 | ON(플라즈마측) | linedump τ=1e-100 정확값 다수 실측(s45 τ 중앙값=1e-100) — **주로 "미매핑 이온/레벨" 센티널** = 조용한 라인 무효화 | 미매핑 라인이 τ-플로어로 위장(카운터 없음) | 미매핑 카운터 분리 |
| M7 | 스펙트럼 창 하드코딩: VSPEC 500-20000Å/2000빈·`N_VPACKETS=10`(cuda.cu:148-151)·coevolve MC 방출 CEMC 500-20000Å/1000빈(8016-8019)·창밖 escape 무기록(빈 범위체크) | 상수 | 2층 | ON(CEMC; VSPEC=최종 반복만) | CEMC 산출 정상 기록(99657 escaped) | `LUMINA_SPEC_RANGE` 변경 시 VSPEC/CEMC은 **불추종**(하드코딩) — 불일치 위험 | 상수 공유화 |
| M8 | coevolve INJECT2 `tau_floor=2.0`(`LUMINA_COEVOLVE_TAU_FLOOR` 기본) (cuda.cu:7407-7413) | 2.0 | 2층 | ON(INJECT=2) | **[COEVOLVE-INJECT2] smin=0 (tau<2.00) L_tot=1.088e43** — 불투명핵 접힘 미발화(전 셸 τ<2) | 발화 0 | 유지 |
| M9 | [기지] `JBAR_MIN=3`(env; 기본 10, mode2=50) 소비 문턱 | 3 | 2층 | ON | **jbar_dump 모드 분포 실측(Si II+III, 50셸)**: iter7 mode3=87.8%/mode0=12.2% → iter9 mode3=84.5%/**mode0(binned 폴백)=15.5%** (폴백 점유 증가 추세) | 교차<3 라인은 binned J(과열 13× 기지)로 펌프 | [기지] |
| M10 | [기지] jbar β=성운 τ 맹목(사례 23) | — | 계기 | ON | 89% 바이트동일 기왕 실측 — 재측정 불요 | — | [기지] |
| M11 | 이벤트로그 CAP: `EVENT_LOG_CAP=400`(MB) | 400MB | 계기 | ON | **it11: 1,301,315,460 events 중 901,315,460 dropped (69.3%)** (stdout:41733) | 계기 층 — 물리 무영향; 단 이벤트 쿼리 배터리의 대표성 주의 | 최종 반복만 CAP 상향 고려 |

### B-5. 로더 (atomic.c·main.c)
| # | 사이트 | 값 | 층 | 활성? | 발화 실측 | 편향 | 처분 |
|---|---|---|---|---|---|---|---|
| L1 | tau_sobolev 차원 불일치 → WARNING+**0 배열로 진행** (atomic.c:435-440) | zeros | **3층**(무효데이터 진행) | 가드 존재 | **미발화** — 로그 "tau_sobolev: [2584132 x 50] (expect [2584132 x 50])" 일치. (초기 0은 compute_tau_sobolev가 1반복 내 덮으므로 실해악은 transition_probabilities 쪽이 큼) | 발화 시 조용한 무불투명도 런 | fail-closed(abort)로 전환 |
| L2 | transition_probabilities 열 불일치 → WARNING+0 배열 (atomic.c:451-455) | zeros | **3층** | 가드 존재 | **미발화**([7752396 x 50] 일치) | 발화 시 MA 분기 전멸 | fail-closed로 전환 |
| L3 | 필수 config 6키 부재→abort·`LUMINA_SPEC_RANGE` sscanf 검증 (atomic.c:308-333, main.c:266-278) | — | 1층(fail-closed 모범) | ON | 미발화(정상 로드) | — | 유지 |
| L4 | `T_e_T_rad_ratio` 기본 0.9(config 부재시) (atomic.c:335-338) | 0.9 | 3층(시드) | env=1.0로 대체 | 발화 무관(override) | — | 유지 |
| L5 | **`TRAD_COLOR_FIX=1`: 로드 직후 `T_rad[s>=1]:=T_rad[0]`** (atomic.c:372-377) | 전셸 10470K | **3층**(참조 프로파일 대체) | **ON** | 발화 실측: 로그 "T_rad[s>=1] := T_rad[0]=10470 K" + **최종 lumina_plasma_state.csv 전 50셸 T_rad=10470.093240 유지**(pure-CMFGEN 경로는 T_rad 재해석 없음) | compute_tau_sobolev 성운 인구(W·e^{−E/kT_rad})·formal 폴백 W·B(T_rad)·분배함수의 온도축이 전부 단일값 | §상위-5 |
| L6 | sigma_bf 부재→Kramers 폴백·부분 커버리지 (atomic.c:960-965) | Kramers | 3층(폴백) | 로드됨 | 26087/26592 레벨(98.1%) 커버 — **505 레벨은 σ_bf 무데이터**(처리는 bf 빌더측=파트 A 경계) | 미커버 레벨 bf 누락/Kramers | 커버리지 카운터 유지 |
| L7 | col_data 폴백(FeIII vR·ion Axelrod floor)·MA-RADRECOMB 맵 부재→ground-only (atomic.c:1058, 1122, 1222) | 폴백 | 3층(폴백) | 부분 | 7슬롯 로드 성공(로그); 맵 13이온 로드 | 미로드 이온=근사 충돌률(파트 A 물리) | 파트 A와 공유 |
| L8 | 라인리스트 극단 적색 꼬리 로드(λ까지 9.997e11 Å) (로그:129) | — | 데이터 | ON | binned 창(100-20000Å) 밖 라인은 J 조립 스킵(C6)·MC 순회에는 잔존 | 미미(τ 극소) | 유지 |
| L9 | **binned J 격자 하드코딩**: `NLTE_N_FREQ_BINS=1000`, 100-20000Å (lumina.h:487-489) — 결정론 J·MC 추정기·bf 격자 공통 | 1000빈 | 2층(해상도) | ON | Δlnν=5.3e-3(≈1590 km/s/빈) — 라인 대비 조악(기지 binned-J 회색화의 뿌리) | 창밖(>20000Å) comov 추정 드랍 | [기지 계열] |

## 상위 5 상세

### 1. formal 적분 비보존 25× — 잣대가 등록 비회귀선(≤4×)을 6배 초과한 채 표류 (F2)
`compute_formal_integral_spectrum`은 (a) 라인 소스로 `line_source_S`(최종 NLTE 해)를, τ로는 성운 재기록 `tau_sobolev`를 짝 비보장 상태로 소비하고, (b) 기본값 `fi_use_cont=0`이라 e-산란·연속 감쇠가 0이며, (c) S_l≤0 폴백이 W·B(T_rad)(binned/fine의 B(T_e)와 **다른** 대체답)이다. parity42 실측: **L=24.99×L_inj**, 본 조사 대역분해로 초과의 70%가 2000-4000Å(각각 8.4/8.9×L_inj), 단일 빈 3342Å에 0.3×L_inj 스파이크. 대조군: fine 창 내부의 짝맞춤 소비(linedump)는 유계 — S/B 극단(최대 1.07e6)은 전부 τ~1e-33 라인이라 방출량 S_l·(1−e^-τ)≤3e-33으로 무해. ⟹ 25×는 fine-창 짝맞춤이 아니라 **formal 소비점의 짝 비보장 + 무감쇠 + T_rad 핀 폴백** 조합에 있다(기전 확정은 offline-first 절차로 별도 — parity42 verdict C6 "미규명"과 정합). 처분 제안: formal에 linedump와 같은 "소비점 τ·S_l 짝 관측자"와 폴백률 카운터를 먼저 심는다(계량→수리 순서).

### 2. 캡히트=에너지 삭제 기본값 (M1)
`d_cap_real_only=0`(경계 횡단·diffuse-BC 반환도 상호작용으로 카운트) + `d_cap_force_escape=0`(캡히트 패킷 에너지 삭제)이 parity42 유효 설정. MAX_INTERACTIONS=50000. 실측: 최종 반복 escaped 99657/100000, 재흡수는 diffuse-BC로 재방출되므로 **≤343 패킷(0.34%)이 절단-삭제**로 추정되나 coevolve 경로는 per-iter [CAP] 카운터를 인쇄하지 않아 **에너지 분율 미확정**. 편향 방향은 구조적으로 명확: 깊이 트랩된(=UV 형광 깔때기 한복판의) 패킷만 선택적으로 삭제 → L_emitted 결손·UV 재처리 결손. 처분: coevolve 경로에도 [CAP]/[CAP-SHELL]/E_truncated 인쇄 배선(계기 부채 상환) 후, FORCE_ESCAPE=1(보존형)을 A/B.

### 3. 경로별 3원화된 S_l 대체답 (C1·C2·F1)
같은 물리량(라인 소스)의 폴백이 소비자마다 다르다: binned 조립은 게이트 기본 OFF로 **100% B(T_e)**(사실상의 서모스탯 — 주석 자백), fine 생산자는 NLTE S_l 소비하되 제로율 **s8 20.9% / s45·s49 99.04%**가 B(T_e)로, formal은 **W·B(T_rad=10470핀)**로 폴백. 판별식의 "파생 양 전부가 일관된 대체 경로" 원칙 위반의 전형 — τ와 S_l이 서로 다른 이야기를 하는 구조가 경로 간에도 반복된다. 특히 s45/s49의 99% 제로율은 "외곽 방출원=전면 열화 가정"이 실측으로 확인된 것(형광사 인접). 처분: 폴백을 값이 아닌 "미해결 표시"로 통일하고, (η,χ) 쌍 물화 재정식화(메모리 정본 ② 구조 수정)의 소비자 목록에 formal을 추가.

### 4. fine GPU 라그 솔버 미수렴 경고 실물 발화 (P5)
parity42의 det jbar(전선B가 소비하는 물건)를 만든 최종-반복 fine 솔브에서 stderr 경고가 실제 발화: a_lam≠0 라그 스킴은 O(NB≈5e5) 반복이 필요한데 n_ali=20000. 조기정지 조건(maxrel<1e-4)은 라그 반복의 per-iter 변화량이라 미수렴을 수렴으로 오판할 수 있는 형태(뺄셈판 사례와 동류의 계기 결함). ADV_SPLIT=1이 연속에는 무해하지만 라인 코어의 주파수 재분배는 여전히 라그를 탄다. **jbar_line_det의 라인코어 J가 과소/미수렴인지 여부 미확정** — GPU=2 자가검증을 fine 격자 축소판(NB 1e4급)에서 ALAM on/off A/B로 오프라인 확정 가능(런 발주 불요, 기존 selftest 배선 존재). 처분: 판정 전 이 오프라인 검증.

### 5. TRAD_COLOR_FIX 전셸 T_rad 핀 (L5)
로더가 참조 데이터의 T_rad 프로파일을 셸0 값(10470K)으로 덮고, pure-CMFGEN 경로는 T_rad를 재해석하지 않으므로 **런 전체·최종 상태까지 전 50셸 동일**(plasma_state.csv 실측). 소비자: compute_tau_sobolev의 성운 준위 인구(W·exp(−E/kT_rad)) — 즉 **수송이 보는 τ 전체의 온도축**, formal 폴백 W·B(T_rad), 분배함수. 챔피언 구성으로 문서화된 의도적 게이트지만, 층 분류상 참조 프로파일의 답-대체(3층)이며 "T_rad 전셸 10470핀=잣대 결함" 사고(기지)와 같은 값이 물리 경로에도 박혀 있는 형태. 처분: 게이트 유지하되, τ·formal 소비자 관점에서 W(r)만으로 반경 구조를 지는 현 구성의 민감도를 캠페인 종결 전 1회 A/B.

## 발화 총괄 (parity42 실측)
| 실측 항목 | 값 |
|---|---|
| FORMAL-CONS | **24.99×L_inj** (등록 비회귀 ≤4× 위반); 초과 70%=2000-4000Å; 3342Å 단일빈 0.3×L_inj |
| fine S_l=0→B(T_e) 폴백률 | s8 20.90% / s45 99.04% / s49 99.04% (창내 793,505선) |
| binned η_line B(T_e) 대체 | 100% (SRC_NLTE 게이트 기본 OFF) |
| S/B 극단 | 최대 1.07e6(s45), >1e4 48-62선/셸 — 전부 τ≲1e-33이라 S_l·(1−e^-τ)≤3e-33 무해(짝맞춤 유계 확인) |
| τ≤1e-12(deposit 제외) | s8 20.6% / s45 99.5% / s49 99.6%; τ=1e-100 센티널이 외곽 중앙값 |
| MC 캡 절단 | ≤343/100000 패킷(0.34%), 에너지 분율 미확정; MA 내부캡 탈출 0→17 증가 |
| EPAY 재척도 | s49 0.024-0.214 (외곽 방출 2-21%로), s0=0(면제), 12회/런 |
| jbar 폴백(mode 0) 점유 | 12.2%→15.5% (iter7→9, Si II+III; 증가 추세) |
| c1 초빈 표현 | 'pin' 101→136빈/1200, W 극단(>5 또는 <1e-4) 2366/14400행, 최대 W=6554 (TEPIN 빈에서 W가 진폭 흡수 — W,T_R을 물리로 읽으면 오독; 소비자=광이온 적분, 파트 A 경계) |
| 이벤트로그 드랍 | 901,315,460/1,301,315,460 = 69.3% (CAP 400MB, 계기 층) |
| fine GPU 미수렴 경고 | 1회 발화(최종 반복 생산 호출, n_ali=20000 vs O(5e5)) |
| INJECT2 τ-floor | 미발화(smin=0) |
| 로더 차원 가드(L1/L2) | 미발화(차원 일치) |

## 미확정 (측정 불능이었던 것)
- M1 캡 절단의 **에너지 분율**(coevolve 경로 [CAP]/E_truncated 미인쇄) — 계기 배선 필요.
- C4 eps_floor=1e-5 및 el<0→1.0 폴백의 발화율 — 카운터 없음.
- P5 fine J 라인코어의 실제 수렴도 — GPU=2 축소판 A/B로 오프라인 확정 가능.
- F1 formal S_l 폴백률(라인·셸별) — 카운터 없음.
- C11 window_color 대역의 toy06 유효성 — 소비자(파트 A radeq)와 합동 필요.
- P6 obs nsub 캡(256/4096) 도달률 — 카운터 없음.
