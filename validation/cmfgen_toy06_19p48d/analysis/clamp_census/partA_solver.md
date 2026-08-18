# 하드코딩·cap/floor 전수조사 — 파트 A: NLTE 솔버·인구·열평형 계층

- 기준선: **parity42** (`logs/coevolve_consume_parity42/`, RESOLVED CONFIG 112 vars 확인).
  핵심 게이트 상태: `LUMINA_NLTE_LTE_FLOOR=0`, `LUMINA_NLTE_FLOOR_MODE` unset(=0), `LUMINA_NLTE_BK_CEIL` unset(=0),
  `LUMINA_NLTE_INV_CEIL=1e4`(**ON**), `LUMINA_RADEQ_OMEGA_FLOOR=1`(**ON**), `LUMINA_NLTE_JBAR_POPS=3`+`LUMINA_JBAR_MIN=3`(**ON**),
  `LUMINA_ARTIS_PARITY=1`, `LUMINA_RADEQ_SIMUL=1`, `LUMINA_NLTE_ASSEMBLE_GPU=0`(GPU bb-assembly OFF), `LUMINA_NLTE_SKIP_Z=`(빈값).
  ⟹ **floor-off 세계에서 legacy `x<0→1e-30` writeback 클램프가 실전 경로.**
- 층 판별식: memory/feedback_clamps_are_not_physics_fix_the_solver.md (1층=정확해가 위반 불가 / 2층=부동소수점 위생·증폭기 위험 / 3층=답의 대체).
- 발화 실측 소스: `lumina_levelpop_resolve_raw.csv`(1,051,900행), `cmf_fine_linedump_s{8,45,49}.csv`(각 793,505행),
  `lumina_jbar_dump.csv`(iter11 293,550행), `lumina_c1_bins.csv`(iter11 1,200셀), stdout/stderr 배너.
- 기지 여부: `grep -rn` memory/*.md 로 확인함 (INV_CEIL·OMEGA_FLOOR·1e-30·1e-100·BK_CEIL·n_star_ratio·LTE_FLOOR·FLOORM·denom>1e-30·W>1e4·ff_pref·35eV·SAHACONST 전부 히트 → [기지] 표기; 미히트 항목만 [신규]).
- **주의(행번호)**: src/lumina_plasma.c 는 편집 중 — 모든 사이트는 함수명+코드 조각 앵커로 인용.

## A. 총괄 표

층: ①=물리 제약(안전) ②=부동소수점 위생(증폭기 확인 요) ③=답의 대체.
발화열의 `[정적-미측정]` = 기존 산출물로 잴 수 없음(계측 부채).

### A-1. NLTE solve·writeback (src/lumina_cuda.cu, `nlte_solve_all_gpu`)

| # | 사이트 [내용앵커] | 값 | 층 | 활성? | 발화 실측 (parity42) | 편향 방향 | 처분 |
|---|---|---|---|---|---|---|---|
| 1 | [기지★] legacy negative-clamp: writeback else-분기 `if (x[i] < 0.0) x[i] = 1e-30;` | 1e-30 | ②(증폭기: 뒤에 per-ion rescale·`1/boltz=exp(+ΔE/kT)` 곱함) | **ON** | n_k==1e-30 정확일치 **2,276행** (Si III 1,965 + S III 311; 셸 43-49 및 20-28 집중). **주의: 하한치** — rescale≠1이면 1e-30 흔적이 지워짐. b_k>1e4 85행(최대 8.2e4, S II lev47 s13), b_k>1e8 **0행** (floor-off 세계에서 1e8 comb 소멸 확인) | 미량이온 여기준위에 인구 발명→과방출 (현 런에선 규모 작음) | 솔버수리(log-space/b_k-space 정식화) + 발화 카운터 상설화 |
| 2 | [기지] INV_CEIL 게이트: `nlte_inv_ceiling()`… `if (x[i]/xg > ceil_ratio) need_fallback=1` → 셸 전체 Boltzmann 대체 | 1e4 (env 명시; 코드 기본값도 1e4 — **env 미설정이어도 켜짐**) | ③(해 기각→LTE 대체) | **ON** | **발화 0** — `[NLTE-FALLBACK]` stderr 0건 (first-16 프린트 로직상 0건=진짜 0) | 강한 진성 역전/형광 초과를 LTE로 대체(잠재) | 이 런에선 휴면; 계량 유지, 장기적으로 미해결-표시 방식으로 |
| 3 | [기지] LTE floor: `lte_floor` 분기 `x[i] = (boltz > 0.0) ? boltz : 1e-30` (sub-resolution `xmax*1e-12` 기준) | LTE@Te, subres=1e-12 상대 | ③ | OFF (`=0`) | 0 (게이트 꺼짐; parity37에서 1000-1300Å 방출 99.9% 제조 전과) | (켜지면) 과방출·초열 S_l/B | 폐기 방향 유지 |
| 4 | [기지] FLOORM mode1: `floorm_mode==1` LTE-relative floor + `cap = floorm_bkmax*boltz_abs` | BKMAX=1e3 | ③ | OFF | 0 (`[FLOORM]` stdout 0건) | (켜지면) b_k≤1e3 강제 | 보류 |
| 5 | [기지] BK_CEIL: `bk_ceil > 0.0` 분기 `cap = bk_ceil*boltz` | 기본 0=off | ③ | OFF | 0 | (켜지면) b_k 상한 발명 | 179184 판정대로 폐기 |
| 6 | [신규] grey/LTE 과도 대체: `nlte->current_iter < grey_iters && nlte->shell_tau[s] >= grey_tau` → fallback | GREY_TAU=2, GREY_ITERS=2 | ③(과도기 한정) | ARMED | **발화 0** (`[NLTE-FALLBACK]` 0건 — grey 발화도 같은 배너를 지나므로 0 확정) | (발화 시) 초기 iter 두꺼운 셸 LTE화 | 무해 확인; 제약유지 |
| 7 | [신규] dead-pair skip: `nlte_skip_dead_pairs()` … `if (n_tot < 1e-10) continue`(제로 행렬→singular→fallback) | 1e-10 cm⁻³ 절대치 | ② | **OFF** (`LUMINA_NLTE_SKIP_DEAD` unset) | 0 (경로 자체 비활성; singular 경고 0건) | (켜지면) 희박 셸 이온 ~0 대체 | 제약유지(성능 게이트) |
| 8 | [신규] CE 수렴 잣대 절대 문턱: `if (old_total > 1.0)`만 max_rel_change 에 포함 (+`has_prior` 판정 동일 문턱) | 1.0 cm⁻³ | ② | **ON** | CE 1-2회 수렴(마지막 iter들 3e-15) — 문턱이 판정을 바꾼 증거 없음. 단 <1 cm⁻³ 이온은 수렴 감시 밖 [정적-미측정] | 미량이온 총량 드리프트 무감시 | 계량필요(상대문턱 or 전이온 포함) |
| 9 | [기지-확장] CE 캡: parity 분기 `ce_max_iter=20; ce_damping=1.0` (비parity 5/0.5), `ce_threshold=1e-2` | 20회/1%/무감쇠 | ①-②(수렴 기계) | **ON** | 캡 도달 0 (1-2회 수렴) | 없음(수렴 시) | 제약유지 |
| 10 | [신규] BK_PARTIAL 참조 floor: `pfloor = pmax * 1e-30; x[i] *= (p > pfloor) ? p : pfloor` | 상대 1e-30 | ② | OFF (`LUMINA_NLTE_BK_PARTIAL` unset) | 0 | — | 보류 |
| 11 | [신규] Boltzmann fallback 온도 선택: `plasma->T_e[s] > 0.0 ? T_e : T_rad` + per-ion rescale `sum>0 && n_total>0` else scale=1 | — | ① | ON(경로), 발화 0 | fallback 자체 0회 | — | 제약유지 |

### A-2. tau/S_l writer (src/lumina_cuda.cu, `nlte_solve_all_gpu` 내 "Updating tau_sobolev" 블록; CPU 미러 `nlte_update_tau_sobolev` 동일 상수)

| # | 사이트 [내용앵커] | 값 | 층 | 활성? | 발화 실측 | 편향 방향 | 처분 |
|---|---|---|---|---|---|---|---|
| 12 | [기지] tau NaN-catch: `if (!(tau_nlte > 1e-100)) tau_nlte = 1e-100;` | 1e-100 | ② | **ON** | **s8: 138,088/793,505 (17.4%) · s45: 777,167 (97.9%) · s49: 777,211 (97.9%)** — 대부분 부재-이온 라인(n_lower=0; levelpop의 n_k≤0 838,446행과 정합) + 역전(stim_corr=0) 혼합, 두 원인 분리 불가 [부분 미측정] | 소비측이 τ~0으로 읽으므로 흡수·방출 쌍이 함께 소멸 → 항등식 보존, 실질 무해 | 제약유지 (단 흡수/방출 항등식 유지 조건부) |
| 13 | [기지] 유도방출 클램프: `if (stim_corr < 0.0) stim_corr = 0.0;` | 0 | ①(τ→0이면 흡수·방출 동시 소멸 — 항등식 보존, 정본 메모리 명시) | **ON** | 개별 카운터 없음; #12 발화에 흡수됨 [정적-미측정] | 역전 라인의 증폭 차단(마이크로 반전 무시) | 제약유지 |
| 14 | [기지★★] S_l denom 컷: `double denom = ratio - 1.0; if (denom > 1e-30) S_l = src_prefac / denom;` else **S_l=0**→소비자 B(T_e) 폴백 | 1e-30 | ②이 ③으로 작동 (문턱 양쪽 30자릿수 불연속) | **ON** | S_l==0 전체: s8 165,864(20.9%)/s45 785,865(99.0%)/s49 785,909(99.0%). **그중 τ>floor(하준위 인구 실존)인 진성 발화: s8 27,776 (전체 forest의 3.5%) · s45/s49 각 8,698** — 이 라인들은 실제로 B(T_e) LTE 소스로 대체됨. 문턱 통과측 초열 꼬리: SoverB>1e3 약 95-100줄/셸, **최대 1.07e6 (line 215015, 1002.6Å, 세 셸 공통)** — floor-off 세계에서 parity37의 1e46 대비 극적으로 완화되었으나 불연속 구조 자체는 잔존 | 문턱 아래=강제 열화(B 폴백, 저형광 편향) / 문턱 바로 위=1e5-1e6× 초열 S_l (1002Å 초열장과 동일 대역!) | **솔버수리 1순위**: (η,χ) 쌍 물화 or 융합 닫힌형 `S_l(1−e^{−τ}) = prefac·C·f_lu·λ·t·(g_l/g_u)·n_u·β(τ)` (정본 메모리 ② 형식 — d 소거) |

### A-3. GEMM 커널 (src/lumina_nlte_gemm.cu · lumina_bf_gemm.cu · lumina_nlte_assemble.cu)

| # | 사이트 [내용앵커] | 값 | 층 | 활성? | 발화 실측 | 편향 방향 | 처분 |
|---|---|---|---|---|---|---|---|
| 15 | [신규] TF32 정밀도: `CUBLAS_COMPUTE_32F_FAST_TF32` (R_bf 광이온율 + chi_bf 불투명도 GEMM) + FP32 스테이징 | ~1e-3 상대오차 하드코딩 | (클램프 아님 — 정밀도 부채) | **ON** (RATES_GEMM 기본 on, BF_OPACITY=1) | N/A — 오차장 미계측 [정적-미측정] | 광이온율·bf 불투명도에 ~0.1% 무작위 편향 | 계량필요(FP64 대조 1회) |
| 16 | [기지] Kramers 폴백 σ₀: `sigma_0 = 7.91e-18 / (Z_eff²)`, `if (Z_eff_int < 1) Z_eff_int = 1` (get_bf_sigma0 "(est)" 연계) | 7.91e-18 | (하드코딩 물리) | **ON** | **156/10,120 활성 광이온 준위가 Kramers (1.5%; 배너 "[NLTE-GEMM] init: … 9964 CMFGEN + 156 Kramers")** | 해당 준위 bf율 근사 | 계량필요(어느 이온인지 목록화) |
| 17 | [신규] χ 부재 센티널: `if (chi_eV < 0.0) chi_eV = 1e10; /* impossibly high */` → K열 0 | 1e10 eV | ① (데이터 부재→율 0) | ON | [정적-미측정] (부재 이온화에너지 수) | 해당 쌍 광이온 0 | 제약유지 |
| 18 | bf n_level 커널: `if (n_ion_s < 1e-30 \|\| Z_part_s < 1e-300 …) n_level=0`; `if (boltz > 50.0) n_level=0` | 1e-30/1e-300/50 | ② | **ON** | [정적-미측정] | 극미량 흡수자 탈락(과소 불투명도, exp(-50)≈2e-22라 무해) | 제약유지 |
| 19 | [신규] bf 불투명도 인구 = **dilute-LTE(W,T_rad)** 고정 (`bf_compute_n_level_kernel`은 NLTE 인구를 안 씀) | — | (하드코딩 물리 선택) | **ON** | N/A | NLTE로 부양/고갈된 준위의 bf 흡수 불일치 (양방향) | 계량필요 — NLTE-인구 chi_bf와 1회 대조 |
| 20 | GPU bb-assembly (`nlte_assemble_bb_kernel`): `a_J_at_nu` 그리드 밖→**1e-30 반환**, `a_planck_bnu` x>500→0, `gbar=0.2`, `ups=max(ups_vr, A_AXELROD_OMEGA=1.0)` | — | ②/③ | **OFF** (ASSEMBLE_GPU=0 + ARTIS_PARITY 블로커 → CPU 조립) | 0 | — | 비활성 확인만 |

### A-4. CPU 조립기 jbar 소비자 (src/lumina_plasma.c, `nlte_assemble_rate_matrix` bb 루프)

| # | 사이트 [내용앵커] | 값 | 층 | 활성? | 발화 실측 | 편향 방향 | 처분 |
|---|---|---|---|---|---|---|---|
| 21 | [기지-갱신] 크로싱 게이트: `jbar_count[…] >= jbar_min` 아니면 binned-J 폴백(mode 0) | JBAR_MIN=3 (기본 10/50) | ②(MC 노이즈 게이트) | **ON** | iter11: **50,632/293,550 (17.2%) 라인이 binned-J 폴백(mode 0)**, 82.8%가 mode-3 소비 (mode 분포 {0:50632, 3:242918} — count<3 행과 정확 일치) | 폴백 라인은 열적 binned J→b_k→1 (형광 결손 방향) | 계량 유지; JBAR_MIN 은 노이즈-바이어스 트레이드 노브 |
| 22 | [신규-측정] mode-3 β·J_inc: `beta = radeq_beta_esc(tau_l); R_absorb = B_lu*beta*J_jbar` | — | ①(정확 Sobolev/MALI 형) | **ON** | β 분포(iter11 mode-3): **중앙값 1.000, β>0.99 98.8%, β<1e-3 0.1%** — β-소멸 펌프는 희귀 | 두꺼운 라인만 열화(물리적) | 제약유지 |
| 23 | [기지] mode-2 차분 클램프: `if (!(bJext > 0.0)) bJext = 0.0` | 0 | ② (뺄셈판 병 — 정본 메모리) | OFF (mode 3; LINERES_CONSUME unset) | 0 | — | mode-2 자체 기각 유지 |
| 24 | [기지] binned J 그리드 밖: `nlte_get_J_at_nu` … `if (nu <= nu_min \|\| nu >= nu_max) … return 1e-30`(radeq 미러 주석 "same out-of-range 1e-30" 동일) | 1e-30 | ②→③(값 발명; 라디에이티브 율 ~0 강제) | **ON** (mode-0 폴백/그리드 밖 라인에) | [정적-미측정] (그리드 밖 라인 수 미계측) | 그리드 밖 라인 복사율 소멸→충돌 지배→열화 | 계량필요 |
| 25 | [기지] MC J_nu 빈-빈 floor: `nlte_normalize_j_nu` … `nlte->J_nu[idx] = 1e-30; /* floor */` | 1e-30 | ② | parity C1 경로가 대체 (아래 #43-46) | C1 경로에선 empty→**0.0** (1e-30 아님) — 발화는 C1 표 참조 | — | 제약유지 |
| 26 | [기지] parity 충돌율 디스패치: `int forb = (f_lu <= 1e-10)` (E1 vs M1/E2 프록시) + `ry_over_dE = 13.6057/(dE_eV>0?dE_eV:1e30)` | 1e-10 / 1e30 | (하드코딩 프록시) | **ON** (ARTIS_PARITY) | [정적-미측정] (forbidden 분류 라인 수) | 금지선 오분류 시 충돌율 체계 오류 | 계량필요 |
| 27 | [기지] 재결합 n_star_ratio 캡: `if (n_star_ratio > 1e30) n_star_ratio = 1e30;` (bf/rec 조립; 로그-공간 판 "no 1e30 cap" 주석과 공존) | 1e30 | ② | **ON** (경로) | [정적-미측정] | 냉 셸 재결합율 상한→저재결합(과이온 방향) | 계량필요 |
| 28 | bb 연결 판정: `if (bb_connected && (total_up + total_down) > 1e-30)` | 1e-30 | ② | ON | [정적-미측정] | 극약결합 준위 고립 처리 | 제약유지 |

### A-5. radeq 열 솔버 (src/lumina_plasma.c, `radeq_build_line_cooling_table`·`radeq_simul_all`·`simul_r1`·`simul_ladder`)

| # | 사이트 [내용앵커] | 값 | 층 | 활성? | 발화 실측 | 편향 방향 | 처분 |
|---|---|---|---|---|---|---|---|
| 29 | [기지★] Ω floor: parity 분기 `if (p_om_floor > 0.0) { c_min = ARTIS_COL_CONST * p_om_floor; if (coeff < c_min) coeff = c_min; }` — **real Upsilon 대입 후에 적용**(진짜 close-coupling Ω<1도 끌어올림) | Υ≥1 (RADEQ_OMEGA_FLOOR=1) | **③** (충돌강도 발명; dig_C5: CMFGEN 관례 0.1의 10×) | **ON** (배너 "[ARTIS-PARITY OMEGA-FLOOR] radeq ETLA coeff floored at Upsilon>=1 (+real-Upsilon for 9958 lines)") | 테이블 2,584,132 bb 중 floor 발화 비율 **미덤프 [정적-미측정]**. real-Upsilon 대입 9,958라인(8 이온 테이블)은 실측 — 이 중 Ω<1이던 것도 1로 상향됨(수 미계측) | 금지·반금지선 충돌냉각 과대→**T_e 저온 편향**(nebular valley 냉각재 인위 부양) | **계량필요 1순위**: floored-line 수·냉각 기여 셸별 카운터; 그 후 0.1/실측 Ω A/B |
| 30 | [기지] T_e 이분법: `double Tlo = 3500.0, Thi = 140000.0;` + 24-스텝 첫-부호변화 행진 + 45회 이분 + **no-root→HOLD** (`f_lo<=0 … T_e = plasma->T_e[s]>100 ? prev : Tlo`) | [3500,140000]K / 24/45 | ①-②(브래킷·행진 해상도) + HOLD는 ③성(이전값 대체이나 '미해결 표시'에 가까움) | **ON** | **600/600 셸-솔브 root-found ([SIMUL it=1..12] pins hi=0 lo=0; TEHOLD 전 셸 radeq_root=root-found 600건)** — no-root HOLD·브래킷 핀 발화 0 | (발화 시) T_e 동결 | 이 런 무해; 첫-부호변화=최저근 선택은 다근 창에서 분지 선택 하드코딩 — 계량 유지 |
| 31 | [기지] T_e 스텝 클램프: `radeq_te_step_clamp` … `if (T_new > 2.0*T_old) return 2.0*T_old; if (T_new < 0.5*T_old) return 0.5*T_old;` | [0.5,2]×T_old (ARTIS 미러) | ①-②(고정점 불변 수렴 감쇠) | **ON** (TE_STEP_CLAMP=1) | 개별 카운터 없음; TEHOLD 로그상 스텝 3-6% 수준(예: 10470→10839K)이라 **발화 0 추정** [정적-부분측정] | 없음(고정점 보존) | 제약유지 |
| 32 | [기지] 감쇠 상수단: RADEQ_DAMP=0.5·SIMUL_ION_DAMP 기본 0.5·J_DAMP=0.5·COEVOLVE_JBAR_DAMP=0.5·CN_DAMP=0.5 | 0.5 | ① (고정점 보존) | **ON** | N/A | 수렴 경로만 변경 | 제약유지 |
| 33 | [기지] H_photo lagged: Gph/Hex를 이분법 밖에서 **lagged binned J**로 1회 구축 ("photoion integrals from the lagged binned J", all-level 분기 주석 "Gph is built once, outside the T bisection -- intended approximation") | — | (operator-split 하드코딩) | **ON** | N/A (구조적) | 장-온도 비동시성: J 갱신 지연분만큼 H_photo 위상차 | 계량필요(수렴 후 잔차로) |
| 34 | [기지] ff 냉각: `sh.ff_pref = 1.426e-27 * 1.2;` 후 `c_ff_d = sh.ff_pref * n_e * n_e * sqrt(T)` — **n_e² 사용(Σz²n_ion 아님)** + Gaunt 1.2 하드코딩 (D5 가열측은 `3.69255e8/√T·Σq²n_ion`으로 별도) | 1.426e-27×1.2, n_e² | (하드코딩 물리 오류 — 기지) | **ON** (매 simul_r1 평가) | N/A (구조적; 매 평가 발화) | 고전하 혼합에서 ff 냉각 왜곡(순수 H 기준 관계식) — 냉각·가열측 불일치 | **솔버수리** (Σz²n_ion으로 교체) |
| 35 | [기지] HRESP 신뢰영역: `radeq_Hresp` … `LUMINA_HRESP_CLAMP` … `|dB| <= fac*max(blag, |B|/10)` | 1.0 | ③(선형화 트러스트) | env=1.0이나 **호출 경로 비활성** — radeq_Hresp 소비처는 coupled-Newton 계열(`COUPLED_NEWTON=0`)뿐, `simul_r1`은 미호출 | 0 (경로 추정) [정적-미측정] | — | 비활성 기록 |
| 36 | [신규] 라인 컬링: `double cull = 0.01; … thr = cull*(H_dep>0?H_dep:1e-30)/n_lines` — 최대기여 상계가 thr 미만이면 탈락 | 0.01 (기본) | ②(집계 편향 <1%·H_dep로 **상계 보장**) | **ON** (RADEQ_LINE_CULL unset→0.01) | 컬링된 라인 수 미덤프 [정적-미측정] | 총 냉각 ≤1% 과소 (구조적 상계) | 제약유지 |
| 37 | [기지] 분배함수 컷 3종+LUT: `simul_build_ulut` 24점 log-T [3500,140000] 양끝 클램프·`x < 300.0` 컷; Gph U_ion `x < 50.0` 컷 + `if (!(U_ion >= 1.0)) U_ion = 1.0`; `bf_rate_pop` `bz >= 500.0 → 0` + `U<=0→1` | 24점/300/50/500/U≥1 | ② | **ON** | [정적-미측정] | x<50 컷: 고여기 준위 절단(기여 e⁻⁵⁰, 무해); U 하한 1은 값 발명이나 U≥g₀≥1이라 사실상 ① | 제약유지 |
| 38 | [기지-경계] simul_ladder 이온화 사다리: `if (r > 1e28) r = 1e28;` · `if (y[j+1] > 1e280) {…/=1e280}` · SIM_SAHACONST(주석 2× 오류 기지) · 레벨없는 top-rung 절단(`SIMUL_CAP_TOPION=1` **ON**) · `g_col = 0.1/0.2/0.3` 하드코딩 | 1e28/1e280 | ②(스케일 재정규는 ①) | **ON** | **이온화 계층 = 파트 B 관할** — 여기선 존재만 기록 | (r-캡) 극단 과이온 상한 | 파트 B로 이관 |
| 39 | [신규] `radeq_beta_esc`: `if (tau <= 1e-6) return 1.0; if (tau > 700.0) return 1.0/tau;` | 1e-6/700 | ①(잘 스케일된 닫힌형 — 정본 메모리가 모범례로 지목) | **ON** | N/A | 없음(점근 정확) | 제약유지·`-expm1` 규칙화의 모델 |
| 40 | [기지] fb 냉각 무게: `g_fb_cool_kt ? (kT) : (chi + kT)` (FB_COOL_KT=1) | kT | (물리 선택 게이트) | **ON** | N/A | 레거시 대비 fb 냉각 축소(ARTIS 정합 방향) | 제약유지 |
| 41 | [신규] `simul_line_term`: `if (den <= 0.0) return 0.0;` (den=C_ul+R_ul) | 0 | ② | ON | [정적-미측정] | 죽은 라인 냉각 0 | 제약유지 |

### A-6. superbin/C1 장 구축 (src/lumina_plasma.c, `nlte_build_perbin_dilute_field`)

| # | 사이트 [내용앵커] | 값 | 층 | 활성? | 발화 실측 (iter11) | 편향 방향 | 처분 |
|---|---|---|---|---|---|---|---|
| 43 | [기지] W 레일: `if (W > 1e4 \|\| !isfinite(W)) { … TR_MAX 재적합 … else { TR = 0.0; W = 0.0; } }` (fit·pin 두 분기 동일) | 1e4 / TR_MAX=250kK | ③(색온도 250kK 대체 또는 장 소거) | **ON** | rail-zeroed(W=0 비-empty) **0**; **T_R==250kK 발행 250/700 fit 빈(35.7%)** — 광학 코스빈 0-13, 전 50셸, **J 점유 7.07%**, W 3.1e-6..0.135. (주의: find_bin_T_R 자체 250kK 상한 도달과 rail-refit 채택이 산출물에서 구분 불가 — 두 경로 모두 T_R=250kK) | 해당 빈 내 스펙트럼 기울기를 250kK RJ 형상으로 대체 (J 적분은 W가 보존) — J의 7%에 형상 왜곡 | **계량필요**: 레일 경로 태그 분리 덤프; 이후 빈 세분/nu_bar 적합 개선 |
| 44 | [기지] TEPIN: `if (tepin_on && nu_lo >= nu_superbin) { … T_R := T_e; W := J/∫B(T_e) }` (λ≤1085Å; pin_c=1은 degen 면제) | T_R:=T_e | (ARTIS 정본 미러 — 물리 선택) | **ON** (C1_SUPERBIN_TEPIN=1) | **134-138 빈/iter 핀, 50셸 전부** ("[C1-SUPERBIN-TEPIN] it 12: 136 coarse bins") | EUV 색온도를 국소 T_e로 고정(ARTIS 동작) | 제약유지 |
| 45 | [기지] DEGEN 폴백: `TR_c[c] >= 0.95*TR_MAX && raw_frac < 1e-3 → raw per-Hz` | 0.95/1e-3 | ②(레일 아티팩트의 정직-원시 대체) | ARMED (C1_DEGEN_FALLBACK=1) | **발화 0** ("[C1-DEGEN-FALLBACK] it" 배너 0건) — #43의 250kK 발행 빈들은 raw_frac≥1e-3라 미포착 | (발화 시) 레일 빈을 원시장으로 | 기준 재검토(250kK 광학 빈이 그물 밖) |
| 46 | [신규-측정] empty 빈: `if (f_first[c] < 0 \|\| !(J_c > 0.0)) → W=0, TR=0` → 해당 대역 `Jrow[f]=0.0` | 0 | ①(패킷 0 = 통계 사실; ARTIS fit_parameters:796 동일) | **ON** | **364/1,200 (셸,코스빈) 셀 empty (30.3%)** — 해당 대역 J≡0으로 소비(광이온·펌프 정확히 0) | 희박 대역 율 0 (기근을 정확히 0으로 고착) | 제약유지 + 점유율 카운터 |

### A-7. k-packet·전송측 상수 (src/lumina_cuda.cu 소비 + src/lumina_plasma.c 구축)

| # | 사이트 [내용앵커] | 값 | 층 | 활성? | 발화 실측 | 편향 방향 | 처분 |
|---|---|---|---|---|---|---|---|
| 47 | [기지-측정] MA-REAL-UPSILON 커버리지: 실 close-coupling Ω 적중 **9,958라인 vs covered-ion vR 폴백 1,424,522라인** (배너 실측) | 0.7% real | (데이터 커버리지 하드코딩) | **ON** | 배너 census 그대로 | 99.3% 라인이 vR/Bethe 프록시 충돌율 | 계량 완료; 데이터 확장이 유일 해법 |
| 48 | RNG 가드: `if (xi < 1e-30) xi = 1e-30;` (log(0) 방지, 다수) · `if (r1 < 1e-300) r1 = 1e-300;` | 1e-30/1e-300 | ① | ON | N/A | 없음 | 제약유지 |
| 49 | 전송 안전캡: `__constant__ int d_max_total_steps = 2000000;` + `LUMINA_MAX_INTERACTIONS=50000` | 2e6/5e4 | ②(안전망이나 발화 시 에너지 유실) | **ON** | [정적-미측정] — EVENT_LOG 쿼리로 잴 수 있으나 본 조사 범위 밖 | (발화 시) 패킷 조기 폐기=플럭스 결손 | 계량필요(이벤트로그 카운트) |
| 50 | [신규] KPEMISS_TE_POP 분배함수 floor: `if (Ze < 1e-300) Ze = 1e-300;` + `bz < 500` 컷 | 1e-300 | ② | OFF (KPEMISS_TE_POP unset) | 0 | — | 보류 |
| 51 | uvopt 방출 부스트 4창 (`uvopt_emit_boost*`, 기본 1.0) | 1.0=off | ③(켜면 율 조작) | OFF | 0 | — | 폐기 후보(오버피팅 노브) |
| 52 | [기지] DETFLUOR S_l 캡: emergent 분기 `setenv("LUMINA_CMF_FINE_SL_CLAMP", ceil?:"10")` (cmfgen.c 소비) | 10×B | ③ | **OFF** (parity42 EMERGENT_MODE 배너 없음, env unset) | 0 | — | 파트 B 관할 소비처 |

## B. 심각도 상위 5 상세

### 1. S_l denom 컷 (#14) — 활성 ②→③, 진성 발화 s8 27,776라인
`if (denom > 1e-30) S_l = src_prefac / denom;` else S_l=0. 실측으로 두 얼굴이 다 있다:
(a) **강제 열화면**: 하준위 인구가 실존(τ>floor)하는데 S_l=0→소비자 B(T_e) 폴백이 s8에서 27,776라인(포토스피어 forest의 3.5%). NLTE 이온에 LTE 소스를 강제 — SKIP_Z 사건과 같은 기전의 국소판.
(b) **초열면**: 문턱 바로 위 denom에서 S_l/B 최대 1.07e6이 실체화(1002.6Å — 현 캠페인의 Si III ¹P° 부양 초열장 1113Å대와 같은 FUV 창). floor-off 세계라 parity37의 1e46은 사라졌지만 **불연속 자체는 남아 있고, 그 스케일(1e6)이 지금 판정에 걸리는 크기**다.
처분: 솔버수리 — 정본 메모리의 융합 닫힌형(②)으로 d 소거: `S_l·(1−e^{−τ}) ∝ n_u·β(τ)`. 물화 배열을 (η,χ) 쌍으로 바꾸면 이 사이트 자체가 소멸.

### 2. RADEQ_OMEGA_FLOOR=1 (#29) — 활성 ③, 발화율 미계측
parity 분기에서 real close-coupling Upsilon을 **대입한 뒤** floor를 적용하므로, 실측 Ω<1인 전이(금지선 다수)도 Ω=1로 발명된다. 실데이터를 덮는 유일한 활성 3층 항목. 2.58M 라인 테이블 중 floored 비율·냉각 기여가 덤프되지 않음 — **계량이 선행 조건** (dig_C5의 "CMFGEN 관례 0.1의 10×" 기록과 결합하면 T_e 저온 편향 용의). 이 floor는 radeq ETLA 냉각 전용이라 NLTE 행렬 충돌율(artis_col_rates)과 별개 장부 — 두 계층의 Ω 불일치도 잠재 결함.

### 3. legacy 1e-30 writeback (#1) — 활성 ②(증폭기), 발화 하한 2,276
floor-off 세계의 실전 경로. 실측 발화는 Si III(1,965)·S III(311)에 국한, b_k>1e8 꼬리 0 — parity36-38 판정(플로어 제거 무손실)과 정합. 단 **exact-1e-30 카운트는 하한**: per-ion rescale이 1이 아니면 1e-30 흔적이 지워진다(증폭기 원리 그대로 — 발명값에 스케일이 곱해진 채 유통). 진짜 발화율은 클램프 지점 카운터로만 잴 수 있음(floorm_clamp 카운터가 mode1 전용인 것을 legacy 분기로 확장하는 1줄 계측이 선결).

### 4. C1 W>1e4 레일 → T_R=250kK 발행 (#43) — 활성 ③, 250빈/iter·J의 7.1%
예상과 달리 발화가 **EUV가 아니라 광학 코스빈(0-13)**에 몰린다: 전 50셸에서 250/700 fit 빈이 T_R=250kK로 발행되고, 이 빈들이 총 J의 7.07%를 나른다. J 적분은 W가 보존하므로 피해는 빈 내 스펙트럼 형상(250kK RJ 기울기 대체)에 국한되나, 이 장은 bb 레이트(nlte_get_J_at_nu)와 광이온 GEMM이 그대로 소비한다. 250kK가 find_bin_T_R 탐색 상한 도달인지 rail-refit 채택인지 산출물로 구분 불가 — 경로 태그 1컬럼 계측이 선결.

### 5. jbar_count<3 binned 폴백 (#21) — 활성 ②, 17.2%
소비 라인의 17.2%가 MC 크로싱 부족으로 열적 binned J에 폴백(mode 0). 이 폴백은 b_k→1 방향(형광 결손)으로 계통 편향 — JBAR_MIN=3으로 이미 기본(10)보다 공격적으로 낮춰져 있으나, 남은 17.2%는 sparse-line 이온의 구조적 한계. β 분포 실측(98.8%가 β>0.99)은 mode-3 자체의 β-소멸은 무죄임을 보여줌 — 남은 결손은 크로싱 통계가 지배.

## C. 발화 실측 총괄 (parity42)

| 계측 | 결과 |
|---|---|
| n_k==1e-30 (writeback 클램프 흔적) | 2,276행 (Si III 1,965 / S III 311; 하한치) |
| n_k≤0 (부재이온/영-인구 행) | 838,446/1,051,900 (그중 −0.0 표기 205,157) |
| b_k>1e4 / >1e8 | 85행(최대 8.2e4) / **0행** |
| tau_sob==1e-100 | s8 17.4% · s45 97.9% · s49 97.9% |
| S_l==0 전체 / 그중 τ>floor 진성 | s8 20.9% / **27,776** · s45 99.0% / 8,698 · s49 99.0% / 8,698 |
| SoverB 최대 | 1.07e6 (line 215015, 1002.6Å) — >1e3 라인 ~100줄/셸 |
| jbar mode 분포 (iter11) | mode-3 242,918 (82.8%) / mode-0 폴백 50,632 (17.2%, =count<3과 일치) |
| mode-3 β 분포 | 중앙값 1.000; β>0.99 98.8%; β<1e-3 0.1% |
| NLTE Boltzmann fallback (INV_CEIL·grey·singular·NaN 합산) | **0회** (stderr [NLTE-FALLBACK]·singular 0건) |
| radeq 이분법 | 600/600 root-found; pins hi=lo=0; HOLD 발화 0 |
| C1: TEPIN / 250kK 발행 / empty / DEGEN / rail-zero | 136빈 / 250빈(J의 7.07%) / 364빈(30.3%) / 0 / 0 |
| OMEGA_FLOOR | 배너로 활성 확정 (real-Upsilon 9,958라인 위에 floor); floored 비율 미계측 |
| Kramers σ₀ 폴백 준위 | 156/10,120 (1.5%) |
| MA 충돌 real Ω 커버리지 | 9,958 / 1,434,480 (0.7%) |
| CE 수렴 | 1-2회, 캡(20)·fallback 미도달 |

## D. 미확정 ([정적-미측정] 목록 — 계측 부채)

1. **#29 OMEGA_FLOOR floored-line 비율·냉각 기여** — 미덤프. 셸×이온별 카운터 1개면 잰다 (최우선).
2. **#1 legacy 클램프 진짜 발화율** — exact-1e-30은 rescale≠1이면 소실. floorm_clamp 카운터의 legacy-분기 확장 필요.
3. #12/#13 tau-floor 발화의 부재이온 vs 역전(stim_corr) 분해 — 현 덤프 스키마로 불가.
4. #43 250kK 발행의 경로(find_bin_T_R 상한 vs rail-refit) 구분 — mode 컬럼에 태그 추가 필요.
5. #27 n_star_ratio 1e30 캡 · #24 그리드-밖 1e-30 J · #37 분배함수 컷 · #36 컬링 라인 수 · #49 MAX_INTERACTIONS/step-ceiling 발화(이벤트로그로 가능) — 전부 무카운터.
6. #15 TF32 오차장 — FP64 1회 대조 없이는 크기 불명.
7. #31 TE_STEP_CLAMP — 정황상 발화 0(스텝 3-6%)이나 반복 전체를 관통하는 직접 카운터 없음.
8. levelpop의 −0.0 205,157행 — 부호 있는 0의 산출 경로(xfl=x·frac에서 frac=0 곱 추정) 미추적; 소비엔 무해하나 writer 경로 규명 미완.
