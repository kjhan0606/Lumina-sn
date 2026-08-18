# 클램프 전수조사 적대 재검증 (2026-07-29)

- 검증자: 원 조사에 관여하지 않은 독립 에이전트. **읽기 전용**(src/ 무수정).
- 대상: `CANONICAL.md` / `INNOCENT.md` / `partA_solver.md` / `partB_transport.md`.
- 원칙: 조사 보고서의 수치를 믿지 않고 **원시 산출물에서 독립 재계산**. 소스 주장은 **파일을 열어 내용 앵커로 확인**(행번호는 편집 중이라 신뢰하지 않음).
- 기준선: `logs/coevolve_consume_parity42/` (RESOLVED CONFIG 112 vars, argv `... 100000 12 spectrum nlte`).
- 추가 조인 데이터: `data/tardis_reference_toy06_19p48d_sivcaiv/{line_list.csv, levels.csv, geometry.csv}` (line_id ↔ (Z,ion,lower,upper,f_lu) ↔ (g,E) 조인으로 덤프만으로는 불가능했던 귀속을 결정).

## 스탬프

| 구분 | 수 |
|---|---|
| CONFIRMED (수치·소스 모두 재현) | 13 |
| REFUTED (수치는 재현되나 **귀속/근거가 틀림**) | 3 (P1b-근거, P2-귀속, P7-산출물) |
| UNVERIFIABLE (현 산출물로 판정 불가) | 2 (P1-인과귀속, P8-소비 실재) |
| 무죄 8건 중 강등 | **0** (8건 모두 해당 경로에 계기가 실제 배선됨) |
| 무죄 8건 중 근거 교체 | 1 (#7 fine 짝맞춤 — 결론만 생존) |
| 무죄 8건 중 재분류 | 1 (#6 β-소멸 — "발화 0" 항목이 아니라 희소성 주장) |
| 조사 보고서와 1% 이상 다른 재계산 값 | 5 (§C) |
| 신규 발견 | 7 (§D) |

---

## A. 문제 측 재검증 (P1–P9)

| # | 판정 | 재계산/소스 확인 | 조사 보고서와의 차이 |
|---|---|---|---|
| **P1a** 대역분해 | **CONFIRMED** | 총 **24.986×L_inj**(배너 24.99와 일치). 2000-3000Å **8.488**, 3000-4000Å **8.998**, 합 **17.485 = 총량의 69.98% / 초과분(L−L_inj)의 72.90%** | 8.44/8.92 → +0.57%/+0.87% (1% 미만) |
| **P1b** fine창 짝소비 유계 | **REFUTED(근거) / CONFIRMED(결론)** | "S/B 극단은 **전부** τ~1e-33이라 기여 ≤3e-33"은 거짓. s8의 S/B>1e3 라인 95개 중 **최대 τ=2.95e-6, 최대 S_l·(1−e^−τ)=1.32e-6** (주장 상한의 10²⁷배). 단 결론은 독립 지표로 생존: fine창 **총 초열 초과분 = 열적 기준 대비 +2.24%(s8) / +0.0007%(s45) / ~0%(s49)** | 상한 3e-33 → 실측 1.32e-6 (§C-1) |
| **P1c** formal 폴백·무감쇠 소스 | **CONFIRMED** | `compute_formal_integral_spectrum` 내 앵커 확인: `double S = line_source_S[l*n_shells+shell]; if (S <= 0.0) S = plasma->W[shell] * planck_bnu(plasma->T_rad[shell], nu_l);` (무게이트·무카운터), `int fi_use_cont = _env_cont ? atoi(_env_cont) : 0;` (env 미설정 ⟹ **0**), τ는 별도 배열 `opacity->tau_sobolev[l*n_shells+shell]` | 없음 |
| **P1d** 3342Å 스파이크 | **CONFIRMED(수치) / 성격 정정** | 3342.125Å 빈 = **0.2949×L_inj** (≈9.13e41 erg/s). 그러나 **단일빈 스파이크가 아님**: 3273-3381Å 전 빈이 ≥0.20×L_inj인 고원이며 **3244-3400Å 합계 3.733×L_inj** | 성격 오기 (§D-7) |
| **P1** 25×의 **소비점 귀속** | **UNVERIFIABLE** | 지목된 3요소 중 **폴백 leg는 실측으로 배제**: formal 자신의 게이트(τ≥`fi_tau_cutoff`=1e-5) 통과 라인 중 S_l≤0(⟹W·B(T_rad)) 비율은 s8 **525/110,357 = 0.48%**, 초과 지배대역만 보면 2000-3000Å **0.07%(17선)**, 3000-4000Å **0.06%(3선)**. **무감쇠 leg의 크기도 유계**: 광구→외곽 반경 τ_es = **1.486**(n_e·σ_T·Δr 적분) ⟹ 연속 감쇠를 켜도 최대 ~e^−1.5(≈4×) 억제. 두 leg 모두 25×를 설명 못함 ⟹ 남은 후보는 짝 비보장 + 무싱크 라인합 누적이나 **현 산출물로는 확정 불가** | partB §상위-1의 3요소 귀속은 근거 미달 |
| **P2** S_l denom 컷 진성 27,776 | **REFUTED (귀속)** | 카운트는 정확 재현(**27,776 = 3.5004%**). 그러나 line_list 조인 결과 **27,776선 전부가 NLTE 네트워크 밖 이온**(Ni I 6337, Ca IV 2883, Ni IV 2673, Fe IV 2314, Co IV 2273, Fe V 2260, Fe I 1886, S V 1629, S IV 1589, Co I 1335, Ca V 848, S I 582, Si I 541, Si IV 442, Ca I 184). 이 이온들은 **창내 라인의 100%가 S_l==0** — per-line 컷이 아니라 `calloc` 0 센티널(atomic.c: "0 (calloc default) signals \"use fallback\"")이다. **NLTE 이온 라인 중 (S_l==0 & τ>floor)는 세 셸 모두 0건.** 구조적으로도 denom≤0 ⟺ stim_corr=(ratio−1)/ratio≤0 ⟹ τ가 1e-100로 바닥 ⟹ 컷 발화는 τ 소멸과 동시에만 일어나 흡수·방출이 함께 사라짐 | 27,776 → **denom 컷 발화 0** (§C-2) |
| **P3** S_l 3원화 | **CONFIRMED (해석 1건 정정)** | binned: `int src_nlte = 0; { const char *sn = getenv("LUMINA_CMFGEN_SRC_NLTE"); if (sn && atoi(sn)) src_nlte = 1; }` → `Sl = src_nlte ? ... : 0.0; if (Sl <= 0.0) Sl = cm_planck(nu_l, Te);` **env 미설정 ⟹ 100% B(T_e)** ✓. 자백 주석 원문 확인: *"the B(T_e) fallback (the de-facto thermostat the champion metrics rest on)"* ✓. fine은 **다른 게이트** `int src_nlte = (opac->line_source_S != NULL);` (항상 ON) ✓. 폴백률 재계산 **20.9027% / 99.0372% / 99.0427%** ✓ | **정정**: 외곽 99%는 "외곽 방출원 전면 열화"의 증거가 아님 — 그중 752,579선(s45)이 **τ=1e-100**이라 소스와 무관하게 방출 0. 실질 열화 대체는 **s8 3.50% / s45·s49 1.10%** (§C-3) |
| **P4** OMEGA_FLOOR | **CONFIRMED + 신규계량** | 배너 원문 일치. 순서 확인: real close-coupling Υ 대입 블록(`cf >= 0.0 → radeq_lines[k].coeff = cf; realsub_count++`) **다음에** `if (p_om_floor > 0.0) { double c_min = ARTIS_COL_CONST * p_om_floor; if (coeff < c_min) coeff = c_min; }` ⟹ **실측 Υ<1도 1로 상향** ✓. floored 카운터 부재 ✓("미계측" 정직). **신규**: 테이블을 오프라인 재구성해 발화율 확정 = **2,278,264 / 2,584,132 = 88.16%** (§D-1) | 미계측 → 88.16% |
| **P5** C1 250kK 레일 | **CONFIRMED (성격 2건 정정)** | it11(=배너 it12): fit 700 / pin 136 / empty 364. **T_R=250kK 정확일치 250셀, 전부 fit 모드**, **J 점유 7.0730%**, empty **364/1200=30.33%**, rail-zeroed(W=0 & J>0) **0**, DEGEN 배너 **0** ✓. `Jrow[f] = 0.0` 소비 확인 ✓ | **정정①** "광학 코스빈 0-13"은 동어반복 — 빈 14-23은 TEPIN 핀 또는 empty라 **구조상 fit-레일이 불가능**. 실제 발화는 **IR(빈0, 15999-19986Å, 29셸)부터 FUV(빈13, 906-1131Å)까지** 산포. **정정②** empty 364셀은 **전부 λ<583.4Å 대역**(빈16-23)에 있고, 그중 **빈20-23(λ<240.8Å)은 전 50셸 J≡0**(빈16/17/18/19는 각 31/37/47/49셸). **추가** 레일 셀 수는 반복마다 단조증가 33→250 (J점유 6.4-12.6%) — 7.07%는 정상상태가 아님 |
| **P6** JBAR_MIN 폴백 | **CONFIRMED (정확일치)** | iter11 293,550행: mode0 **50,632 (17.248%)**, mode3 **242,918 (82.752%)**, `jbar_count<3` 50,632 = mode0과 **1:1 정확일치**. mode3 β: >0.99 **98.84%**, <1e-3 **0.09%(208선)** | 없음 |
| **P7** legacy 1e-30 흔적 | **CONFIRMED(수치) / REFUTED(산출물 귀속)** | `lumina_levelpop_resolve_raw.csv`에서 **2,276 / b_k>1e4 85 / max 8.18e4(S II lev47 s13) / b_k>1e8 0** — 전부 정확 재현. **그러나 이 파일은 순수 관찰자 산출물**: `LUMINA_NLTE_FINAL_RESOLVE` 블록이 pops/tau/S_l/jbar를 저장→재솔브→**복원**하며 로그도 *"converged state restored ...; downstream spectra unaffected"*를 인쇄한다. 실전(=스펙트럼을 만든) 상태인 `lumina_levelpop.csv`는 **2,727 (+19.8%) / b_k>1e4 = 0 / max b_k 7,652** | 2,276→2,727, 85→0 (§C-4). "하한치"라는 단서 자체는 타당(클램프 지점 카운터 부재 확인) |
| **P8** fine 라그솔버 경고 | **CONFIRMED (+중요 정정)** | 경고 문구 원문 일치, `tol` 하드코딩 확인(`... nsamp, n_ali_iter, 1e-4, &gpu_iters)` 2곳 + CPU `if (maxrel < 1e-4 && it > 0)`) ✓. 경고는 **static 래치가 없어 호출마다 인쇄**되는데 stderr에 **정확히 1회** — 직후가 3개 LINEDUMP 줄, 그 다음이 `[DIP-TRACE] it11` ⟹ **fine 생산자는 런 전체에서 최종 반복 1회만 실행**. 소스 근거: `LUMINA_CMF_LINERES_JBAR=2` ⟹ `produce_now = (lineres_jbar >= 2 && it == pc_iter - 1)` | "발화=최종 반복" CONFIRMED. **정정**: 매 반복 오염이 아니라 **1회 생산** |
| **P8'** det jbar 소비 실재 | **UNVERIFIABLE** | 소비자 2곳: (i) `g_ctp_lineres_jbar && opacity->jbar_line_det[...] >= 0.0` (게이트 = 같은 env, **활성**), (ii) `LUMINA_CMF_LINERES_CONSUME` (**미설정 → 비활성**, `[cmf_consume]` 배너 0건). 생산이 최종 반복 끝에 1회뿐이라 다운스트림 실소비 여부는 산출물로 판정 불가 | partB의 "판정 전 오프라인 검증" 권고는 유효하나 **우선순위는 하향** |
| **P9** 캡 절단 | **CONFIRMED (+강화 가능)** | `99657 escaped` ✓. 소스 기본값 `__constant__ int d_cap_real_only = 0; __constant__ int d_cap_force_escape = 0;` + 배너 `[CAP] LUMINA_CAP_REAL_ONLY=0 / LUMINA_CAP_FORCE_ESCAPE=0` ✓ (캡히트 시 `atomicAdd(&d_E_truncated_dev, pkt_energy); d_escaped_flag[p]=0;` = 에너지 삭제). **강화**: `pkt_status = 2`(reabsorbed) 대입은 **소스 전체에서 1곳**이며 `d_diffuse_inner_bc`의 else 분기 ⟹ DIFFUSE_INNER_BC=1인 이 런에서 도달 불가 ⟹ 미탈출 **343 = 캡/스텝천장 삭제 정확값**(≤가 아님, 0.343%). "에너지 분율 미확정"은 **정직**: `LUMINA_ENERGY_BUDGET=1`이 설정돼 있으나 `[E-BUDGET]` 인쇄는 co-evolve가 우회하는 loop-B 경로에 있어 stdout 0건 | ≤343 → **=343** |

---

## B. 무죄 측 재검증 (INNOCENT.md 8건)

규칙: ①발화 0 재확인 ②**카운터/배너가 그 경로에 실제 배선됐는지 소스 확인** ③기준선 한정 vs 구조적.

| 항목 | 판정 | 계기 배선 확인 (②) | 분류 (③) |
|---|---|---|---|
| INV_CEIL=1e4 | **무죄 유지** | 배선 확인. `need_fallback`의 **모든** 원인(ret≠0/info≠0/force_lte/grey/ncrit/비유한/INV_CEIL)이 단일 `if (need_fallback)` 블록으로 합류하고 그 안에서 `static int gpu_fb_warn = 0; if (gpu_fb_warn < 16) fprintf(stderr, "[NLTE-FALLBACK] ...")` ⟹ **배너 0건 = 진짜 0건**(첫 16건은 반드시 인쇄) | **기준선 한정** (더 강한 역전이면 발화) |
| NLTE pair-solve Boltzmann 폴백 | **무죄 유지** | 위와 동일 배선. grey 경로(`LUMINA_NLTE_GREY_TAU=2` **설정됨**, ARMED)도 같은 배너를 통과하므로 partA #6의 grey 무발화도 동시 확증 | 기준선 한정 |
| radeq no-root HOLD | **무죄 유지** | **독립 계기 2개가 일치**: (a) `g_tehold_status`가 pin_lo=2/pin_hi=3/root-found=1로 갈리며 셸×반복마다 인쇄 → **600/600 전부 root-found**; (b) 별도 리덕션 카운터 `n_pin_hi/n_pin_lo`가 `[SIMUL it=N] done: pins hi=%ld lo=%ld` 로 인쇄 → 12반복 전부 0/0 | 기준선 한정 |
| 로더 차원 가드 | **무죄 유지** | 배선 확인: 실제 차원을 **무조건 인쇄**(`tau_sobolev: [%d x %d] (expect [%d x %d])`)하고 불일치 시에만 `WARNING ... reinitializing` — 로그의 실제/기대 차원 일치 ⟹ 미발화 확정 | **데이터 의존**(이 데이터셋 한정) |
| INJECT2 τ-floor(2.0) | **무죄 유지** | 배선 확인: `smin`은 τ<floor인 첫 셸이고 접힘 루프는 `for (s = 0; s < smin; ...)`. 배너 `smin=0` ⟹ 루프 0회 = 미발화 | 기준선 한정 |
| β-소멸(mode-3) | **무죄 유지 / 재분류** | 이 항목은 애초에 "발화 0"이 아니라 **분포 주장**. 재계산 β>0.99 98.84%, **β<1e-3 208선(0.09%) 존재** ⟹ "0"이 아니라 "희소" | 기준선 한정 |
| fine 창 내 τ/S 짝맞춤 소비 | **결론 유지 / 근거 교체** | 제시된 근거(τ~1e-33 ⟹ ≤3e-33)는 **REFUTED**(§A P1b). 대체 근거로 결론만 생존: 창내 총 방출 대비 초열 초과 **+2.24%(s8)**. 추가 경고: 이 항목을 **formal 25×의 대조군으로 쓰는 것은 범주 오류** — 덤프량은 라인별 국소 방출 프록시(S·(1−e^−τ))이지 광선적분 광도가 아니고, fine 솔브에는 formal에 없는 연속 싱크(chi_es/chi_abs)가 있다 | 기준선 한정 |
| bisection 브래킷/스텝 | **무죄 유지** | 배선 확인(위 radeq 항목의 n_pin_hi/lo). 단 보고서의 단서대로 **첫-부호변화 행진의 분지 선택**과 **TE_STEP_CLAMP 발화**는 카운터 부재 — 이 두 하위항목은 여전히 미계측 | 기준선 한정 |

**강등 0건**: 8건 모두 해당 경로에 인쇄/카운터가 실제로 배선돼 있어 "카운터 부재로 인한 0"이 아니었다.

---

## C. 조사 보고서와 1% 이상 다른 재계산 값

| # | 항목 | 보고서 | 재계산 | 차이 |
|---|---|---|---|---|
| C-1 | fine창 초열 라인의 최대 방출 기여 (s8, S/B>1e3군) | ≤3e-33 | **1.324e-6** (τ 최대 2.95e-6) | 10²⁷배 |
| C-2 | S_l denom 컷 "진성 발화" (s8) | 27,776선 | **0선** (27,776은 전부 NLTE 네트워크 밖 이온의 calloc 센티널) | 귀속 전복 |
| C-3 | 외곽 셸 실질 B(T_e) 대체율 (s45/s49) | 99.04% | **1.10%** (τ>floor 조건; 나머지는 τ=1e-100이라 방출 0) | 90배 |
| C-4 | 실전 상태의 1e-30 흔적 / b_k>1e4 | 2,276 / 85 | **2,727 / 0** (`lumina_levelpop.csv`) | +19.8% / 전량 |
| C-5 | s8 SoverB 최댓값 | 1.07e6 ("세 셸 공통") | 셸별 **4.953e5(s8) / 1.0706e6(s45) / 4.745e5(s49)** | s8에서 −54% |

(참고: 1% 미만 차이 — 2000-3000Å 8.44→**8.488**(+0.57%), 3000-4000Å 8.92→**8.998**(+0.87%), mode0 17.2→**17.248%**, J점유 7.07→**7.0730%**, 폴백률 20.9→**20.9027%**, β>0.99 98.8→**98.84%**. 전부 사실상 일치.)

---

## D. 신규 발견

### D-1. OMEGA_FLOOR 발화율은 **미계측이 아니라 오프라인 계산 가능**했고, 값은 88.16%
radeq 테이블 구축 로직(`find_ion_pop_idx` → level_num 매칭 → `dE>0` 필터 → parity 분기 coeff)을 `line_list.csv`+`levels.csv`로 재구성했더니 **엔트리 수가 2,584,132로 배너와 정확히 일치**(필터 재현 검증). 그 위에 실제 floor 조건 `coeff < ARTIS_COL_CONST*1.0`을 적용:

| | 수 | 비율 |
|---|---|---|
| **floored 총계** | **2,278,264** | **88.16%** |
| 허용선(f_lu>1e-10) | 2,259,424 / 2,555,321 | 88.42% |
| 금지선(f_lu≤1e-10) | 18,840 / 28,811 | 65.39% |

증폭 배율 `c_min/coeff` 분위수: **p50 = 253×, p90 = 9.0e4, p99 = 2.5e7, max 4.2e9**.
이온별: Co III 90.2%, Co II 90.7%, Fe II 91.3%, Fe III 93.3%, Cr III 89.1%, Mn III 92.4%, Ni III 91.6%.
닫힌형 판별식: 금지선은 `g_lo·g_up < 100`이면 무조건 floored(Axelrod 0.01 스케일), 허용선은 `f_lu·g_lo·(Ry/dE)²·(dE/k)·1.586e-10 < 8.629e-6`이면 floored.
단서: real-Υ 대체 9,958선(0.39%)은 대체 후 값에 floor가 걸리므로 위 수치의 불확도는 ±0.39%p. **냉각 기여 가중 비율은 여전히 미계측**(라인 컬링·인구 가중 필요).
⟹ CANONICAL.md Ⅱ의 최우선 "계량 선행" 항목은 **런 없이 해소됨**. 88%는 "드문 안전망"이 아니라 **테이블의 기본 동작**이다.

### D-2. 클램프 발화 인구조사가 **관찰자 산출물** 위에서 수행됨
`lumina_levelpop_resolve_raw.csv`(및 `_ema`)는 `LUMINA_NLTE_FINAL_RESOLVE` 관찰자 블록의 반사실 재솔브 결과이고, 블록 말미에서 pops/tau/S_l/jbar와 device tau가 **전부 복원**된다(로그: "downstream spectra unaffected"). 실전 상태는 `lumina_levelpop.csv`(로그 순서상 재솔브 **이전**에 기록). 세 파일의 클램프 흔적이 실제로 다르다(2,727 / 2,362 / 2,276; b_k>1e4 0 / 84 / 85). **발화 인구조사는 `lumina_levelpop.csv`로 재수행해야 한다.**

### D-3. denom 컷은 구조적으로 "단독 발화" 불가
`stim_corr = 1 − (g_lo n_u)/(g_up n_l) = denom/ratio`. 따라서 `denom ≤ 0` ⟺ `stim_corr ≤ 0` → 0으로 클램프 → `tau_nlte`가 1e-100 바닥. 즉 컷이 걸리는 순간 그 라인은 흡수도 방출도 하지 않는다(항등식 보존). 실측도 이와 정확히 일치: s8에서 **NLTE 이온의 S_l==0 수(132,578) = τ==1e-100 수(132,578)** 완전일치. 진짜 결함은 다른 곳 — **NLTE 네트워크 밖 이온(창내 33,286선)이 τ는 성운값으로 살아 있는 채 소스만 0**이어서 소비자가 B(T_e)를 발명한다(s8에서 τ>floor인 27,776선). **수리 방향이 바뀐다**: (η,χ) 쌍 물화로는 이 라인들이 고쳐지지 않는다. 필요한 것은 네트워크 커버리지 확장 또는 "미해결" 명시 표기.

### D-4. formal 폴백(W·B(T_rad))은 25×의 운반자가 아니다 — 실측 ≤0.5%
formal 자신의 컷(τ≥1e-5)을 통과해 실제로 적분에 참여하는 라인 중 S_l≤0인 비율: s8 **0.48%**(525/110,357), 초과 지배대역만 보면 **2000-3000Å 0.07%(17선) / 3000-4000Å 0.06%(3선)**. 외곽 셸은 비율이 높지만(9.6-11.4%) 참여 라인 자체가 146/123선뿐. 동시에 누락된 연속 싱크의 크기도 유계다: **광구→외곽 반경 τ_es = 1.486**(n_e·σ_T·Δr). ⟹ partB §상위-1이 지목한 세 leg 중 둘은 크기로 배제되고, 25×의 실제 운반자는 **미확정**.

### D-5. formal 잣대의 영점이 1.000이 아니다 (+6.8% 구적 편향)
`n_impact=100`, `dp = r_outer/100 = 6.783e13 cm`, `r_phot = 6.564e14 cm` ⟹ **코어 원반을 표본하는 광선은 10개뿐**. 백라이트 항의 이산합 `Σ_{p<r_phot} p·dp = 50 dp²` vs 해석해 `r_phot²/2 = 46.83 dp²` ⟹ FORMAL-CONS 분자가 **+6.8%** 과대. 라인 방출이 0이어도 비율은 1.068이 된다. (25× 판정은 불변이나, 등록 비회귀선 "≤4×"류의 정밀 비교에는 영점 보정이 필요.)

### D-6. fine 생산자는 런당 1회만 실행된다
`LUMINA_CMF_LINERES_JBAR=2`의 의미가 "최종 순수반복에서만 생산"(소스 주석: *"per-iter production is the fix8-era dominant bottleneck (~56 min/iter measured; 12x avoided)"*). 래치 없는 경고가 stderr에 1회뿐인 것이 이를 독립 확증한다. ⟹ 미수렴 fine 장의 잠재 오염은 **최종 반복 1회분**으로 국한되고, 유일 활성 소비자는 macro-atom internal-up 분기 하나다(pops 소비자 `LINERES_CONSUME`은 미설정).

### D-7. 3342Å "스파이크"는 폭 ~130Å 고원이고, 같은 대역이 fine 덤프에서도 초열이다
3244-3400Å 합계 **3.733×L_inj**. 그리고 s8 fine 덤프에서 **열적 초과가 가장 큰 라인들이 정확히 이 대역**(3832/3591/3388/3094/3999/3242Å, S/B 1.5-3.2, τ 0.2-2.8, 다수가 S II). 2000-4000Å에서 참여 라인의 **53-58%가 S_l>B(T_e)**. ⟹ 이 대역의 초열성은 formal 고유 아티팩트가 아니라 **NLTE 소스 자체가 갖고 있는 성질**이며, formal은 그것을 싱크 없이 누적한다(가설 — D-4의 미확정과 함께 다음 조사 대상).

---

## E. 재현 명령 (전부 읽기 전용)

```bash
cd logs/coevolve_consume_parity42

# P1a/P1d/D-5/D-7: formal 대역분해·고원·구적
python3 - <<'PY'
import csv
rows=[(float(a),float(b)) for a,b in list(csv.reader(open('lumina_spectrum_formal.csv')))[1:]]
dl=(rows[1][0]-rows[0][0])*1e-8; L=3.094761e42
tot=sum(f*dl for _,f in rows); print('total',tot/L)
for a,b in [(2000,3000),(3000,4000),(3244,3400)]:
    print(a,b,sum(f*dl for l,f in rows if a<=l<b)/L)
PY

# P2/P1b/P3: fine 덤프 (S_l==0, τ, 초열)
awk -F, 'NR>1{n++;if($7==0){z++;if($11>1e-100)g++}}END{print n,z,g}' cmf_fine_linedump_s8.csv

# P5: C1 레일/empty
python3 -c "import csv,collections;r=[x for x in csv.DictReader(open('lumina_c1_bins.csv')) if x['iter']=='11'];print(collections.Counter(x['mode'] for x in r));print(sum(1 for x in r if abs(float(x['T_R'])-250000)<1e-6))"

# P6: jbar 모드/β
awk -F, 'NR>1&&$1==11{m[$10]++;if($8<3)c++}END{for(k in m)print k,m[k];print "cnt<3",c}' lumina_jbar_dump.csv

# P7: 실전 vs 관찰자 산출물
for f in lumina_levelpop.csv lumina_levelpop_resolve_raw.csv lumina_levelpop_resolve_ema.csv; do
  awk -F, -v F=$f 'NR>1{if($7=="1.000000e-30")c++;if($9+0>1e4)b++}END{print F,c,b}' $f; done

# P8/P9: 계기 배너
grep -c "NOT converged" stderr.log; grep -c "cmf_fine" stderr.log
grep -E "\[CAP\]|escaped|E-BUDGET" stdout.log

# D-1: OMEGA_FLOOR 발화율 (line_list+levels 조인, ~13초)
#   scratchpad/omega.py 참조 — coeff 공식은 plasma.c parity 분기와 동일 상수 사용
```

---

## F. 판정 요약 (한 줄)

원 조사의 **측정값은 거의 전부 재현**된다(대역분해·폴백률·모드분포·레일·escaped·배너). 무너진 것은 **세 건의 귀속**이다: ① S_l denom 컷의 "진성 발화 27,776"은 실은 **NLTE 네트워크 밖 이온의 미기록 센티널**이고 컷 자체는 0회 발화(구조적으로도 τ 소멸과 동시에만 가능), ② 클램프 인구조사가 **다운스트림에 영향을 주지 않는 관찰자 재솔브 산출물** 위에서 이뤄졌으며 실전 상태의 수치는 다르다, ③ formal 25×의 3요소 귀속 중 **폴백 leg(≤0.5%)와 무감쇠 leg(τ_es=1.49)는 크기로 배제**되어 인과는 미확정으로 되돌아간다. 반대로 최대 미계측 항목이던 **OMEGA_FLOOR는 런 없이 88.16%로 계량**됐다.
