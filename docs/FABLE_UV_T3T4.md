# FABLE_UV_T3T4 — T3(T_rad 핀 감사) · T4(형광 행렬 채널 커버리지 감사)

- 감사자: fable (독립 레인). `docs/CODEX_UV_T1T2*.md`는 **열람하지 않았다**(독립성 유지). 열람한 것: `docs/UV_CENSUS_CONSOLIDATION.md`, `docs/FABLE_UV_CENSUS.md`(본인 직전 산출), `docs/CODEX_EMISS_E8/E10/E11/E12/E13(+_SUMMARY).md`.
- 모드: **읽기 전용**. 생산 `src/` 수정 0, 신규 모델/GPU 런 0, 커밋 0. 신규 스크립트는 스크래치패드에만 작성(`t3_trad.py`, `t4_events.py`, `t4_bands.py`, `t4_fb.py` — 본 문서 §부록 C에 재현 명령).
- 1차 실측 대상: `RUN=/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828` (parity59, `docs/PARITY59_FLUORMAT.env`).
- 표기: **[F]** 본 감사 실측 · **[D]** 기존 문서 인용 · **[A]** ARTIS 원본 대조.

---

## 0. 두 줄 결론

> **T3 — T_rad ≡ 10470.09 K 핀은 실물이고 살아 있으나 UV 과잉과 무관하다(설명력 ≈ 0%).**
> parity59에서 매크로원자 상향률의 **펌프 항은 T_rad를 소비하지 않는다**(`IUP_TRAD` OFF, `j_cap=j_floor=0` ⇒ `W·B(T_rad)` 비교자 불활성). 남은 유일한 T_rad 항은 자극방출 보정이며 UV선에서 **≤0.4%**다. 핀이 실제로 먹이는 곳은 `chi_bf` 인구·비-NLTE 선 τ·radeq 희석 비교자 — 전부 B1–B4에서 `chi_abs/chi_tot ≤ 5.5e-3`, `eta_fixed/eta_tot ≤ 1.1e-3`인 채널 안이다. **이온화·열 장부 항목으로 강등**하고 UV 수리 후보에서 뺀다(§5.3 사전등록 임계 5% 대비 실측 0.0015%).

> **T4 — 형광 행렬의 채널 커버리지 갭은 실물이나 0.235%이고, E10/E12/E13 실패의 원인이 아니다. 실패의 실제 기전은 따로 특정했다.**
> 구조적 부재 입력(bf-활성 + ff-heat-활성 매크로원자)은 매크로원자 **진입 에너지의 0.2353%**, 출구 선방출 에너지의 **0.2343%**(B0에서도 0.64%)다. 반면 **측정된 R은 대각을 빼면 사실상 rank-1(입력 무관)** 이다 — 밴드별 off-diagonal 출력 SED의 총변동거리(TVD)가 0.018–0.094. 그 보편 출력 SED `q`의 B0 몫은 **10.53%**, 결정론 방출률(s8)의 B0 몫은 **1.43%**. 따라서 R을 적용하면 B0 방출률이 **×7.35** 되는 것이 **산술적 필연**이며(예측 B1 ×1.073 vs E12 실측 ×1.151), "B2→B0 지배"는 병리가 아니라 **B2가 입력 가중의 35.6%를 차지한다는 사실의 동어반복**이다.

**부수 최대 발견(잣대):** E12/E13이 읽은 형광 행렬 파일은 **런 중에 덮어써졌다**. 현재 디스크의 `fluor_matrix_iter10`은 헤더 `iteration=11`, 468,330 edge, sha `08ff3312…`인데 E12가 기록한 것은 `iteration 10`, 473,045 edge, sha `2b65dba6…`다. sha 사이드카가 페이로드와 **함께** 갱신되므로 `sha256sum -c`는 여전히 PASS하고, 소비자 어디에도 **행렬 iteration 계약 검사가 없다**. ⇒ E12/E13은 현재 런디렉토리에서 **재현 불가**이며, 오늘 같은 명령을 돌리면 다른 행렬을 조용히 소비한다(§8).

---

# T3 — T_rad 핀 감사

## 1. 핀의 성립 조건 재확인 [F]

| # | 사실 | 좌표 |
|---|---|---|
| 1 | 적재 시 `T_rad[s≥1] := T_rad[0]` 평탄화 | `src/lumina_atomic.c:377-382`; `LUMINA_TRAD_COLOR_FIX=1`; `stdout.log:139` |
| 2 | `T_rad`/`W`를 갱신하는 **유일** 함수 `solve_radiation_field` | `src/lumina_plasma.c:847-983` |
| 3 | 그 호출부는 **2곳뿐** — CPU 드라이버 `src/lumina_main.c:536`, 고전 MC 루프 `src/lumina_cuda.cu:10447` | grep 전수 |
| 4 | 본 런은 PURE-CMFGEN 블록(`cuda.cu:7737-9885`)을 타고 **`:9881`에서 `return 0`** 한다(`LUMINA_CMFGEN_THEN_MC` 미설정 — `stdout.log`에 `[THEN-MC]` 배너 0건) ⇒ `:10447`에 **도달하지 않는다** | `cuda.cu:9879-9882`; `stdout.log:332` |
| 5 | 결과: 50셸 `T_rad = 10470.093240` uniq=1, `W` = 모델파일 값 그대로(0.2979→0.0108), 12 반복 내내 불변 | `$RUN/lumina_plasma_state.csv` |

**정정 1건 (본인 직전 보고 대비):** 직전 전수조사는 `solve_radiation_field`의 유일 호출부를 `cuda.cu:10447`이라고 썼다. 정확히는 **호출부는 2곳**(`main.c:536` + `cuda.cu:10447`)이고, `main.c:536`은 별도 CPU 드라이버 바이너리 소속이라 본 런의 실행 경로에 없다. 결론은 동일하나 좌표를 바로잡는다.

## 2. T3-1 — 핀의 소비처 전수 (parity59 기준 활성/불활성) [F]

### 2.1 활성 소비처

| # | 소비처 | 좌표 | T_rad가 하는 일 | 크기 |
|---|---|---|---|---|
| **A1** | `compute_tau_sobolev` | `src/lumina_plasma.c:2636-2682` (`beta_rad=1/(kT_rad)`, `weight = meta ? 1 : W`) | 희석-Boltzmann 준위인구 → 전 선의 τ | **비-NLTE 선만 잔존**. NLTE 매핑 2,410,046/2,584,132(93.26%)은 `nlte_update_tau_sobolev`(`plasma.c:16945`, T_e 기반)가 덮어씀. 호출 사슬 `cuda.cu:8203 → plasma.c:6391` |
| **A2** | `compute_bf_opacity` 준위인구 | `src/lumina_plasma.c:7145-7150` | `chi_bf`의 `n_level = n_ion·W·g·e^{−E/kT_rad}/Z` | 전 bf 연속체(`LUMINA_BF_NLTE_POPS` 미설정 ⇒ NLTE 대체 없음) |
| **A3** | `bf_rate_pop` | `src/lumina_plasma.c:8836-8846`, 소비 `:11627`·`:13354` | radeq bf-가열용 인구 | `LUMINA_BF_RATE_POPS=1` |
| **A4** | radeq 희석 비교자 | `src/lumina_plasma.c:11564-11660` (`J_eff = β·W·B_ν(T_rad) + (1−β)·S̄`, `H_photo_dilute`) | T_e 근 탐색의 가열 항 | 활성 |
| **A5** | 매크로원자 내부상향 **자극방출 보정** | `src/lumina_plasma.c:4483-4502` | `coeff = B_lu − B_ul·(w_u g_u)/(w_l g_l)·e^{−ΔE/kT_rad}` | **상향률에 남은 유일한 T_rad 항**. §4에서 정량 |
| **A6** | GPU bf 레인 준위인구 | `src/lumina_bf_gemm.cu:82-95` (`beta_rad=1/(kT_rad_s)`) | A2의 GPU 사본 | 활성 |
| **A7** | 진단·덤프 | `cuda.cu:1969-1997`, `:9842-9845`, `:10628-10633` | 기록만 | 무해 |

### 2.2 불활성 소비처 (증거 포함) — parity59에서 T_rad를 **먹지 않는다**

| 소비처 | 좌표 | 왜 불활성인가 | 증거 |
|---|---|---|---|
| 분배함수 `Z` | `plasma.c:1901-1907` | ARTIS-PARITY B3가 `T_part := T_e, W := 1`로 전환 | `stdout.log:414` `[ARTIS-PARITY B3] partition functions at T_e, undiluted (W=1)` |
| 성운 Saha 이온 분할 | `plasma.c:2219-2387` (`compute_ion_populations_shell`) | `compute_ion_populations`가 **즉시 return** (`plasma.c:2392: if (g_simul_on == 1) return;`) | `LUMINA_RADEQ_SIMUL=1` (`stdout.log:85`); `[SIMUL it=N] done` 12회가 항상 `[Plasma] Computing ion populations...`보다 **먼저**(362<417, 3802<3856, …) ⇒ `g_simul_on=1` 확정. 결정적 증거: return 직후에 있는 배너 `[ARTIS-PARITY B2] ionization closure = LTE Saha at T_e`(`plasma.c:2395-2398`)가 **로그에 0건** |
| 전자밀도 반복해 | `plasma.c:2424-2500` | `plasma.c:6372: if (g_simul_on != 1) compute_electron_density(...)` | 동상 |
| **내부상향 펌프** `B_lu·W·B_ν(T_rad)` | `plasma.c:4362-4363`, `:4508` | `LUMINA_MACROATOM_IUP_TRAD` 미설정(`g_ctp_iup_trad=0`), `use_j_nu=1` | `stdout.log:459` `J_src=MC_histogram`; `stdout.log:440` `[IUP-JBLUE] … (B_lu − B_ul n_u/n_l)*beta*J_blue` |
| J cap/floor 비교자 `J_lte = W·B(T_rad)` | `plasma.c:4402-4412` | `j_cap_effective=0`, `j_floor_effective=0` | `stdout.log:459` `j_cap=0, j_floor=0` |
| MC 레인 `Planck(T_rad)` 열화 (EPS_UV/EPS_IR) | `cuda.cu:5325,5332,5350,5612,5618`, `:6431` | 게이트 미설정 | `LUMINA_EPS_UV`/`EPS_IR` 부재 |
| 레거시 bf 재방출 `Planck(T_rad)` | `cuda.cu:6429-6432`, `plasma.c:7632-7645` | k-packet 풀이 켜져 있어 `act_level`이 항상 채워짐(`cuda.cu:6229-6241`) | 방출 센서스 `bf_reemit` 열 전 파장 **0.0** |
| NLTE 폴백 `Boltzmann@T_rad` | `cuda.cu:1465-1498` | `LUMINA_NLTE_FALLBACK_TE=1` ⇒ T_e | env |
| `nlte_assemble` `dilute_TR = T_rad[0]` | `lumina_nlte_assemble.cu:428` | `LUMINA_NLTE_ASSEMBLE_GPU=0` | env |
| `T_e = ratio·T_rad` fail-closed 폴백 | `plasma.c:2806,2849,11315,11529,11558,11703` | **한 번도 발화 안 함** | `radeq_root=root-found` **600/600**(50셸×12iter), no-root 0건 |
| UV `J_nu` cap `W·B_ν(T_rad)` | `plasma.c:14407-14440` (호출 `cuda.cu:8213`, 매 반복) | `LUMINA_J_NU_UV_CAP` 미설정 ⇒ `if (!enabled) return;` | env |
| EPAY `hot_regime = (Te > hotf·T_rad)` | `lumina_cmfgen.c:1134` | `LUMINA_CMF_EPAY_HOTF=0` ⇒ 술어가 T_rad-무관(항상 참) | env |

> **소비처 감사의 1차 결론:** ARTIS-PARITY(B3)와 RADEQ_SIMUL이 이미 T_rad의 **최대 소비처 두 개**(분배함수·성운 이온화)를 T_e 쪽으로 옮겨 놓았다. parity59에서 핀이 남아 있는 곳은 **연속체 불투명도 인구(A2/A6)·비-NLTE 선 τ(A1)·radeq 가열 비교자(A3/A4)·자극방출 보정(A5)** 넷뿐이다.

## 3. T3-2 — 실제 셸별 T_rad 재구성 [F]

동결 payload `emiss_ab_iter10.A`(LCMFCE01, 헤더 `iteration=10`, `nr=50 × nnu=1000`, λ 100.2–19933.3 Å)의 `J`(마지막 배열, 작성부 `lumina_cmfgen.c:349-350`)에서 재구성.

### 3.1 방법 A — 코드 자신의 추정자(Lucy/TARDIS 모멘트)
`solve_radiation_field`가 호출됐다면 계산했을 값: `T_rad = T_RADIATIVE_CONSTANT·(∫νJdν)/(∫Jdν)`, `T_RADIATIVE_CONSTANT = 1.2523374827e-11` (`lumina.h:45`), `W = πJ_tot/(σT^4)`.

| 셸 | T_pin [K] | **T_rad^mom [K]** | 비 | W_pin | W_mom | T_e [K] |
|---|---|---|---|---|---|---|
| 0 | 10470.1 | **19397.4** | **1.853** | 2.979e-1 | 2.153 | 21227.6 |
| 3 | 10470.1 | **15349.0** | **1.466** | 1.018e-1 | 4.469 | 15667.6 |
| 8 | 10470.1 | **14152.2** | **1.352** | 3.888e-2 | 1.429 | 12003.6 |
| 16 | 10470.1 | **12468.9** | **1.191** | 1.525e-2 | 2.262e-1 | 9334.3 |
| 25 | 10470.1 | **11930.3** | 1.139 | 7.593e-3 | 1.055e-1 | 8475.7 |
| 40 | 10470.1 | **11736.5** | 1.121 | 3.424e-3 | 5.033e-2 | 10456.1 |
| 49 | 10470.1 | **11704.0** | 1.118 | 2.390e-3 | 3.552e-2 | 13052.1 |

**핀 대비 실제 색온도는 1.12×(외곽)–1.85×(내부).** 단조 감소. 동시에 `W_mom > 1`이 s0–s11 전체에서 나온다(최대 4.96 @ s2) — `plasma.c:853-858`이 경고한 railing 그대로다. ⇒ **장은 희석-Planck가 아니며, "진짜 T_rad" 단일값은 존재하지 않는다.** 이것 자체가 잣대 사실이다.

### 3.2 방법 B — 코드 자신의 per-bin 적합(C1 superbin)
`$RUN/lumina_census_perbin_field.csv`(셸×24 coarse bin의 `W_bin,T_R_bin_K,J_bin,J/B(T_e)`)와 `$RUN/lumina_c1_bins.csv`(`mode ∈ {fit,pin,empty,degen}`, 14,400행: fit 8,368 / empty 4,454 / pin 1,546 / degen 32).

s0 UV 대역 실측(핀 10470 K 대비):

| bin | λ [Å] | W_bin | **T_R_bin [K]** | J/B(T_e) |
|---|---|---|---|---|
| 8 | 2740–3406 | 3.695e-1 | **34104.0** | 1.023 |
| 9 | 2194–2741 | 1.900 | **16573.0** | 0.840 |
| 10 | 1756–2194 | 1.570e-2 | **178861.1** | 0.969 |
| 11 | 1413–1756 | 3.972 | **18033.9** | 1.842 |
| 12 | 1131–1413 | 1.648 | **19106.1** | 0.909 |
| 13 | 906–1131 | 7.400e-1 | **21980.4** | 0.929 |

핀(10470 K)의 **1.6–17×**. 다만 `W_bin`이 1을 크게 넘거나(3.97) 1e-2 이하로 railing하는 빈이 섞여 있어 per-bin (W,T_R) 쌍도 물리적 희석-Planck가 아니다. **s16 이상에서는 `T_R_bin`이 250000 K로 rail되고 `W_bin ~ 1e-5`가 되는 빈이 다수** — 이는 본인 직전 보고의 **U1**(J_blue 과대 vs 구간장 기근)에서 예고한 railing이며, 여기서 **railing 실물 확인**으로 격상한다(방향 확정은 여전히 T2b 소관).

### 3.3 방법 C — 소비자 관점: `R_band = ⟨J⟩_band / (W_pin·B_ν(T_pin))`
소비처가 `J` 대신 `W·B(T_rad)`를 쓸 때의 오차 배율(>1 = 핀이 장을 과소평가).

| 셸 | B0 600–1000 | B1 1000–1500 | B2 1500–2000 | B3 2000–2500 | B4 2500–3000 | BALL |
|---|---|---|---|---|---|---|
| 0 | 1.61e4 | 1.08e3 | 2.20e2 | 1.55e2 | 1.07e2 | 1.07e2 |
| 3 | 4.31e3 | 1.16e3 | 2.52e2 | 4.49e2 | 4.88e2 | 2.33e2 |
| 8 | 1.56e3 | 7.59e2 | 1.46e2 | 1.35e2 | 2.83e2 | 1.15e2 |
| 16 | 8.49e1 | 1.37e2 | 2.92e1 | 2.15e1 | 5.43e1 | 2.11e1 |
| 49 | 3.33e1 | 6.04e1 | 1.52e1 | 1.70e1 | 4.78e1 | 1.55e1 |

**핀은 UV 장을 1.5–3 dex 과소평가한다.** 따라서 "핀은 무해"라는 결론은 *핀이 정확해서*가 아니라 **UV를 지배하는 항들이 핀을 소비하지 않기 때문**이다 — §4·§5가 그것을 증명한다.

## 4. T3-3 — 상향률·p_iup에 대한 영향 정량 [F, 산술]

### 4.1 parity59에서 상향률의 정확한 형태

```c
/* src/lumina_plasma.c:4483-4502 (활성 분기) */
double coeff = atom->line_B_lu[line_id];
… dboltz = (E_up − E_lo)/(k T_rad);
   nu_nl  = (w_up g_up)/(w_lo g_lo) · exp(−dboltz);
   coeff -= atom->line_B_ul[line_id] * nu_nl;
if (coeff < 0.0) coeff = 0.0;      /* maser clamp */
rate = coeff * beta * J_blue;      /* :4502 */
```
`B_ul = B_lu·g_l/g_u`를 대입하면 **T_rad 의존성이 전부 하나의 인자로 폐합**된다:

> **`rate_iup = B_lu · [ 1 − (w_up/w_lo)·e^{−hν/kT_rad} ] · β · J_blue`**

즉 `J_blue`(MC blue-wing 추정자, it10 커버 85.7% — `stdout.log:35890`)가 펌프이고, **T_rad는 대괄호 안 자극방출 보정에만 들어간다.**

### 4.2 사전등록 → 실측

**사전등록(측정 전):** UV선(λ ≤ 3000 Å ⇒ hν ≥ 4.133 eV)에서 보정항 `(w_up/w_lo)e^{−hν/kT}`는 T=15349 K(s3 재구성값)에서도 ≪1이므로, 핀 수리에 따른 `p_iup` 변화는 **1% 미만**일 것이다. 만약 **5% 이상** 나오면 핀을 재순환 고리의 공범으로 승격한다.

**실측(s3: `W=0.1018`, `T_pin=10470.09`, `T_mom=15349.0`, per-bin `T_R≈12265–15266`):**

| λ [Å] | hν [eV] | 사례 | `[·]` @ T_pin | `[·]` @ T_mom=15349 | Δ(상향계수) |
|---|---|---|---|---|---|
| 1500 | 8.266 | 하부 metastable / 상부 비-meta (`w_up/w_lo=W`) | 0.9999893 | 0.9998033 | **−0.0186%** |
| 1500 | 8.266 | 양쪽 동종 (`w_up/w_lo=1`) | 0.9998950 | 0.9980679 | **−0.183%** |
| 3000 | 4.133 | 하부 metastable / 상부 비-meta | 0.9989567 | 0.9955254 | **−0.344%** |
| 3000 | 4.133 | 양쪽 동종 | 0.9897512 | 0.9560448 | **−3.41%** |
| 3000 | 4.133 | 양쪽 동종, per-bin `T_R=15266.2`(s3 bin 11) | 0.9897512 | 0.9567834 | −3.33% |

`w_up/w_lo ≤ 1/W`이 되는 유일한 조합(상부 metastable + 하부 비-metastable)은 **선이 존재하면 상부가 metastable일 수 없으므로 물리적으로 배제**된다(⇒ maser clamp는 UV에서 발화 불가; 발화 조건 `hν < kT_rad·ln(1/W) = 2.0615 eV`, 즉 **λ > 6014.6 Å**).

**판정: 핀을 실제 색온도로 고쳐도 강한 UV Fe III의 상향률 계수는 0.02–3.4% **감소**한다.** `p_iup = 0.8859`(s3, Fe III, `stdout.log:35829-35845`)를 만들려면 `p_iup/p_emit ≈ 50`이 필요한데, 3.4% 변화로는 자릿수가 맞지 않는다. **사전등록 임계(5%) 미달 ⇒ 공범 아님.**

### 4.3 교차 확인 — 결정론 레인에서도 핀이 잡을 손잡이가 없다 [F]

payload에서 직접(dν 가중):

| 셸 | `chi_abs/chi_tot` B1 | B2 | B3 | B4 | `eta_coherent/eta_total` B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.29e-4 | 4.88e-4 | 1.72e-4 | 1.40e-4 | 0.999835 | 0.999559 | 0.999857 | 0.999904 |
| 3 | 2.54e-4 | 5.35e-4 | 2.14e-4 | 4.04e-4 | 0.999906 | 0.999624 | 0.999907 | 0.999889 |
| 8 | 1.20e-3 | 6.28e-4 | 6.13e-5 | 1.97e-4 | 0.999638 | 0.999282 | 0.999939 | 0.999929 |
| 16 | 1.84e-3 | 5.50e-3 | 2.15e-5 | 4.11e-5 | 0.999508 | 0.989397 | 0.999972 | 0.999980 |

세 가지가 동시에 성립한다.
1. **B1–B4의 흡수 불투명도는 `eps_eff ≈ 1.9e-4`**(E8 실측 **[D]**)와 자릿수·값 모두 일치한다(s0 B1 2.29e-4, B3 1.72e-4, B4 1.40e-4). ⇒ 그 대역의 `chi_abs`는 **ε·χ_line(선 열파괴, T_e 기반)** 이 지배하고 `chi_bf`(A2/A6, 유일한 큰 T_rad 소비처)는 부차적이다.
2. `S_fixed/B(T_e) ≈ chi_abs/chi_tot`(s3 B2: 8.54e-4 vs 5.35e-4; s0 B1: 3.28e-4 vs 2.29e-4) ⇒ **고정 방출률은 T_e 열원**이지 T_rad 열원이 아니다.
3. `chi_bf`가 지배하는 유일한 대역은 **B0(600–1000 Å)**(s16에서 `chi_abs/chi_tot=0.609`). 그러나 흡수 지배 대역에서는 국소 극한 `J → eta_fixed/chi_abs → B(T_e)`이므로 **진폭이 χ_bf에 대해 1차로 무감**하다. 실측이 이를 뒷받침한다: s0 UV bin 8–13에서 `J/B(T_e) = 0.84–1.84`, s8에서 `0.36–2.22` — **UV 진폭의 잣대는 T_e이지 T_rad가 아니다.**

## 5. T3-4 — 판독: 핀 수리는 UV 과잉의 몇 %를 설명하는가

> **≈ 0%. 핀 수리 단독으로 UV 과잉은 움직이지 않는다.**

근거 사슬:
1. 상향률 펌프가 T_rad를 안 먹는다(§2.2, §4.1) ⇒ `p_iup` 무관(§4.2, ≤3.4%, 방향은 오히려 감소).
2. 결정론 UV 방출률의 99.9%가 `chi_es·J`(결맞음 재순환)이고(§4.3), 핀이 만지는 `eta_fixed`는 B1–B4에서 ≤1.1e-3 몫이며 그마저 T_e 열원이다.
3. E8이 폐합한 진폭 기전(`chi_coherent = chi_e + (1−ε)chi_line`, 이득 5247.49× **[D]**)에 T_rad는 등장하지 않는다. `chi_line`은 τ에서 오는데, τ의 T_rad 경로는 비-NLTE 선에만 남고(§2.1 A1) 그 몫은 아래 §5.1에서 0.0015%다.

### 5.1 직전 보고 §5.3 사전등록의 처분 [F]

> 사전등록 원문(FABLE_UV_CENSUS §5.3): *"NLTE 집합 밖 이온이 소유한 UV τ가 τ 합의 5% 미만이면 → C3는 UV에 대해 무해로 강등하고 잣대 결함으로만 등재. 20% 초과면 승격."*

실측(`stdout.log:33441-33459`, s0, 성운 패스 직후 = T_rad 경로 τ 전체):
- `[TAU-DIAG] UV(1700–3000) τ>1 = 1093/343613, Στ = 227,895.6`
- `[TAU-BY-ION]` 상위: Z=27 ion2 1.993e5(87.4%), Z=28 ion2 2.416e4(10.6%), Z=26 ion2 2.356e3(1.0%), Z=27 ion1 1.305e3, Z=28 ion1 7.541e2, Z=26 ion1 2.871e1 — **전부 NLTE 집합**(`$RUN/lumina_levelpop.csv` s3에 (26,1),(26,2),(27,1),(27,2),(28,1),(28,2) 존재, b_k 해 있음).
- **NLTE 집합 밖**(ion 0, 3, 4): 2.181 + 0.8375 + 0.5028 + 2.12e-3 + 9.479e-5 + 4.802e-5 + 4.118e-6 = **3.523**
- 몫 = 3.523 / 227,895.6 = **1.546e-5 = 0.0015%**

NLTE 갱신 후 τ가 바뀌어도 결론은 유지된다: s0의 지배 이온 b_k는 O(0.04–1.8)(`Z=26 ion2 p50=0.549`, `Z=27 ion2 0.458`, `Z=28 ion2 0.041`) ⇒ NLTE τ가 최대 ~25× 줄어도 비-NLTE 몫은 0.04% 수준이다.

**⇒ 사전등록 임계 5% 대비 0.0015%. C3(T_rad 핀)를 "UV에 대해 무해"로 강등하고, 아래 장부에만 등재한다.**

### 5.2 핀을 어디에 등재하는가 (수리는 별건 — "틀린 값은 조용히 대장 기재")

| 등재 항목 | 영향 받는 물리 | 좌표 | 방향 |
|---|---|---|---|
| **P1** | `chi_bf` 준위인구가 실제보다 차갑다(s3에서 E=5 eV 준위 기준 `e^{−E/kT}` 비 = **5.8×** 과소) | `plasma.c:7145-7150`, `bf_gemm.cu:82-95` | bf 불투명도·EUV 차폐 과소 |
| **P2** | **정규화 불일치**: 분자는 `W·e^{−E/kT_rad}`, 분모 `Z`는 ARTIS-PARITY B3에 의해 `Z(T_e, W=1)` | `plasma.c:2640` / `:7148` vs `:1903-1907` | 인구 합이 `n_ion`을 보존하지 않음(신규 발견) |
| **P3** | radeq 가열 비교자 `W·B_ν(T_rad)`가 실제 장을 1.5–3 dex 과소평가(§3.3) | `plasma.c:11564-11660` | T_e 근을 낮은 쪽으로 편향 |
| **P4** | 비-NLTE 선 τ (0.0015%) | `plasma.c:2636-2682` | 무시 가능 |
| **P5** | `W`도 동결(모델파일 값) — `solve_radiation_field` 미도달의 두 번째 결과 | `plasma.c:951-952` 미실행 | 위 전부의 공통 승수 |

**P2가 본 감사의 신규 발견이다.** `LUMINA_ARTIS_PARITY=1`이 분배함수만 T_e로 옮기고 소비처(τ·χ_bf)의 Boltzmann 인자는 T_rad에 남겨 두어, `Σ_k n_k ≠ n_ion`이 되었다. 이것은 핀과 **독립**인 결함이며 핀을 고쳐도 남는다(핀을 고치면 오히려 T_rad→T_e로 수렴해 완화). 수리는 별도 판단.

---

# T4 — 형광 행렬 채널 커버리지 감사

## 6. T4-1 — 매크로원자 활성화 경로 전수 [F]

`transport_kernel` 내 `d_macro_atom_interaction(` 호출은 **정확히 3곳**이고, `d_line_scatter_event(` 호출은 **1곳**이며, `d_fluor_matrix_record(` 호출은 **1곳**이다(grep 전수, `src/lumina_cuda.cu`).

| # | 활성화 경로 | 진입 좌표 | MA 호출 | 형광행렬 기록 | MA-FATE 센서스 | 이벤트 채널 태그 |
|---|---|---|---|---|---|---|
| **L** | **선 흡수** (`interaction_type==1`) | `cuda.cu:6092-6132` | `:5368` (in `d_line_scatter_event`) | **YES** `:6127-6132` | YES `:5626` (`d_ma_fate_record_zi`) | 진입 `EVCH_MA_ACT_BB`; 출구 `MA_RAD_DEEXC`(:5270,:5303) / `KPKT_COLLEXC_BB`(:5414) / `KPKT_FF`(:5448) / `KPKT_FB`(:5542) / `KPKT_BTE`(:5561) / `MA_RAD_RECOMB`(:5571) / `KPKT_MACAP`(:5596) / `HEAT_LINETHERM`(:5258,5264,5291,5297,5394,5400,5584,5590) |
| **BF** | **bf 연속체 흡수** (`cont_chan==2`, `act_level≥0`) | `cuda.cu:6170-6265` | `:6265` | **NO** | YES `:6417` (`d_ma_fate_record`) | 진입 `EVCH_RPKT_BF_ABS`(:6174); 출구 `KPKT_COLLEXC`(:6279) / `KPKT_FF`(:6295) / `KPKT_FB`(:6378) / `KPKT_BTE`(:6390) / `MA_RAD_RECOMB`(:6400) / `KPKT_MACAP`(:6412) |
| **BF-k** | bf에 target 없음 → **k-packet 풀 재여기** | `cuda.cu:6229-6241` (그 뒤 BF와 합류) | `:6265` | **NO** | YES `:6417` | 동상 |
| **FFH** | **자유-자유 가열** (`cont_chan==1`) → k-packet 재여기 | `cuda.cu:6437-6505` | `:6466` | **NO** | **NO** (기록 호출 자체가 없음) | 진입 `EVCH_HEAT_FF`(:6446); 출구 `KPKT_COLLEXC`(:6476) / `MA_RAD_RECOMB`(:6487) / `KPKT_FF`(:6503) |
| **BFL** | bf, `act_level<0` **그리고** k-packet 풀 OFF → `Planck(T_rad)` 재방출 (MA 아님) | `cuda.cu:6418-6435` | — | **NO** | — | `EVCH_BF_REEMIT_LEGACY`(:6435) — parity59에서 **0건**(센서스 `bf_reemit`=0.0) |
| **ES** | Thomson 산란 (MA 아님) | `cuda.cu:6506-6511` | — | — | — | `EVCH_RPKT_ESCATTER` |
| **IBC** | 내부 경계 재주입 (MA 아님) | `cuda.cu:6060-6085` | — | — | — | — |

**구조 판정:** 형광 행렬은 `interaction_type == 1` **LINE 분기 하나**에만 걸려 있다. 따라서 **BF·BF-k·FFH 세 활성화 클래스의 입력이 연산자에 원천 부재**하다. 부수로 **FFH는 MA-FATE 센서스에도 없다**(`:6466` 뒤에 `d_ma_fate_record` 호출 없음) — 이는 신규 계측 공백이다.

### 6.1 잣대 선행 감사 — 어떤 계량기가 믿을 만한가 [F]

| 계량기 | 상태 | 근거 |
|---|---|---|
| `$RUN/lumina_census_*.csv` (kpkt_exit, heating, ma_fate, emission) | **완전(무-cap)** | `d_census_accumulate`가 `d_event_record` 안에서 **cap 검사·λ 필터·escatter 필터보다 먼저** 발화한다 (`cuda.cu:4610-4612`, 주석 원문 *"census fans out BEFORE the display filters … independent of the lambda filter + CAP truncation"*) |
| `$RUN/lumina_events.bin` | **41.2% 접두부** | `stdout.log:37972` `it11: 970557187 events (570557187 dropped)`; cap 400M (`LUMINA_EVENT_LOG_CAP=400`) |
| `$RUN/fluor_matrix_iter10` | **iteration=11** (파일명과 불일치) | 헤더 실측 `iteration: 11`, `events_total=483,936,781`, `edges=468,330`, `Eabs=2975.9288` — `stdout.log:37965`(it11)와 정확히 일치. `stdout.log:35887`(it10)는 509,203,774 / 473,045 / 3065.5032 |

**이벤트 로그 편향 정량:** 무-cap 센서스 대비 접두부의 누락 채널 비율은 `KPKT_COLLEXC/rad_deexc` = 496,956/199,931,517 = 2.4856e-3 (로그) vs 1,289,180/485,105,515 = **2.6575e-3** (센서스) ⇒ 접두부가 **6.5% 낮게** 잡는다. bf 흡수도 같은 방향(로그 497,840 = 센서스 1,291,431의 38.55%, 전체 접두부 비율 41.21% 대비 −6.5%). ⇒ **접두부의 에너지 가중 비율은 6.5% 상향 보정해 쓴다**(아래에 보정 전/후 병기).

## 7. T4-2 — 누락분의 크기 (실측)

### 7.1 카운트(무-cap 센서스, it11) [F]

| 양 | 값 | 출처 |
|---|---|---|
| 매크로원자 종착 총계 | **485,228,598** | `lumina_census_ma_fate.csv` (rad_deexc 485,105,515 / col_deexc 4,351 / rad_recomb 118,732) |
| **bf + ff-heat 활성화** 진입 | **1,291,834** | `lumina_census_heating.csv` (bf 1,291,431 = `EVCH_RPKT_BF_ABS`; ff 403 = `EVCH_HEAT_FF`) |
| 그중 선광자로 탈출(`EVCH_KPKT_COLLEXC`, **bf/ffh 전용 태그**) | **1,289,180** | `lumina_census_kpkt_exit.csv` `collexc` 열 |
| 누락 클래스 몫 (카운트) | **1,291,834 / 485,228,598 = 0.2662%** | |

> `collexc` 열이 정확히 누락 클래스인 근거: `EVCH_KPKT_COLLEXC`(0x12)를 쓰는 사이트는 `cuda.cu:6279`(bf 활성)와 `:6476`(ff-heat 활성) **두 곳뿐**이고, 선 활성 경로는 `EVCH_KPKT_COLLEXC_BB`(0x16)를 쓰며 이는 k-packet exit 히스토그램에서 **의도적으로 제외**돼 있다(`cuda.cu:4542-4548`).

### 7.2 에너지 가중, 파장 대역 분해(이벤트 로그 접두부 41.2%) [F]

**매크로원자 진입 에너지** (공변, 흡수 시점):

| 밴드 [Å] | E(선 흡수) | E(bf 흡수) | E(ff-heat) | **(bf+ffh)/선** |
|---|---|---|---|---|
| 300–600 | 9.884 | 1.279 | 0 | **12.94%** |
| **B0 600–1000** | 162.07 | 2.397 | 8.2e-6 | **1.479%** |
| **B1 1000–1500** | 401.60 | 2.070e-2 | 0 | **0.0052%** |
| **B2 1500–2000** | 624.26 | 6.108e-4 | 2.3e-5 | **0.0001%** |
| **B3 2000–2500** | 155.96 | 1.269e-4 | 7.1e-6 | 0.0001% |
| **B4 2500–3000** | 113.64 | 4.776e-5 | 8.5e-6 | 0.0000% |
| 3000–4000 | 31.37 | 8.4e-5 | 7.5e-5 | 0.0005% |
| 4000–20000 | 71.72 | 7.4e-4 | 6.1e-4 | 0.0019% |
| **TOTAL** | **1572.68** | **3.6989** | **1.298e-3** | **0.2353%** (보정 후 **≈0.2504%**) |

**매크로원자 출구(선광자) 에너지**:

| 밴드 | E(포함=선 활성) | E(누락=bf/ffh 활성) | 누락 % | 보정 후 |
|---|---|---|---|---|
| 300–600 | 11.136 | 1.910e-2 | 0.171% | 0.182% |
| **B0** | 163.52 | 1.0476 | **0.637%** | **0.678%** |
| **B1** | 400.63 | 1.3529 | **0.337%** | 0.358% |
| **B2** | 623.75 | 0.96828 | 0.155% | 0.165% |
| **B3** | 155.86 | 6.332e-2 | 0.041% | 0.043% |
| **B4** | 113.44 | 8.213e-2 | 0.072% | 0.077% |
| **TOTAL** | **1572.24** | **3.6927** | **0.234%** | **0.250%** |

### 7.3 부수 실측 — U2·U3 해소 및 신규 센서스 결함 1건 [F]

| # | 항목 | 결과 |
|---|---|---|
| **U2 해소** | `linetherm` 열이 전 파장 0인 이유 | **배선 버그 아님.** `stdout.log:269`: *"[LTHERM] LUMINA_LINE_THERM=1 SET but DISABLED by ARTIS-PARITY (D4: no ARTIS analog)"* (`cuda.cu:7266-7268`: `ltherm_on = env && !artis_parity_enabled() && !fix_no_ltherm`). 이벤트 로그 4억 건 중 `EVCH_HEAT_LINETHERM` **0건**으로 독립 확인. ⇒ U2 **CLOSED**. |
| **U3 해소** | fb 방출이 900 Å 미만인지 | **물리다.** `EVCH_KPKT_FB` 1,223건 전량 λ ∈ [252.3, 758.5] Å, 중앙값 601.5 Å, **E(λ<900 Å)=100.00%**. 센서스 창 `[900, 30000] Å` 밖 ⇒ `fb` 열 0.0은 정상. ⇒ U3 **CLOSED**(단, 방출 센서스가 fb 채널에 **구조적으로 맹목**이라는 잣대 사실로 재등재). |
| **신규 N1** | `EVCH_MA_RAD_RECOMB`(0x3A)가 **방출 센서스의 emch 스위치에 없다** (`cuda.cu:4506-4521`) | 실측: 48,735건, E=0.3824(접두부), λ 353–25117 Å, **중앙값 1302.7 Å, 에너지의 73.7%가 센서스 창 안**. ⇒ `lumina_census_emission.csv`의 채널 열 합은 총 방출에너지와 **일치하지 않는다**(누락 ≈ 선방출의 0.024%). 작지만 계량기 결함. |
| **신규 N2** | FFH 경로에 `d_ma_fate_record` 없음 (`cuda.cu:6466` 뒤) | MA-FATE 센서스가 ff-heat 활성화를 누락. 규모 403건 ⇒ 현재는 무해, 계측 완전성 문제. |

## 8. T4-3 — 누락이 E10/E12의 B2→B0 지배와 형상 실패를 설명하는가

### 8.1 사전등록

> **사전등록(측정 전):** 누락 클래스가 B0 유입 에너지의 **10% 이상**을 차지하면 커버리지 불완전이 형상 실패의 유력 원인으로 승격한다. **1% 미만**이면 기각하고 다른 기전을 찾는다.

**실측: B0 출구 에너지의 0.637%(보정 0.678%), 전체 0.234%(보정 0.250%). ⇒ 1% 미만, 기각.**

E12는 B0 `J_det/CMFGEN` **8.29 → 26.43**(×3.19), B2→B0 점유율 **54.92%**(문턱 2.042%의 26.9배)를 보고했다 **[D]**. **0.68% 규모의 입력 클래스 누락으로 ×3.19의 대역 붕괴나 54.92%의 off-diagonal을 만들 수 없다.** 커버리지 갭은 실물이나 **원인이 아니다.**

### 8.2 그러면 무엇이 원인인가 — 본 감사가 특정한 기전 [F, 신규]

행렬을 직접 읽어 12개 밴드로 접었다(`R[j,i] = edge_output_energy/terminal_energy[i]`, LFMAT001은 ν 오름차순).

**(1) 행-정규화 밴드 전이표 (%)** — 각 행이 자기 대각을 빼면 **거의 같은 출력 SED**를 낸다:

| in \ out | 300-600 | B0 | B1 | B2 | B3 | B4 | 3-4k | 4-6k | 6-10k |
|---|---|---|---|---|---|---|---|---|---|
| 300-600 | 2.77 | 13.18 | 25.82 | 41.07 | 4.13 | 6.29 | 1.54 | 3.44 | 1.61 |
| **B0** | 1.01 | **21.61** | 22.24 | 35.89 | 6.31 | 5.77 | 1.90 | 3.76 | 1.36 |
| **B1** | 0.74 | 9.78 | **31.54** | 37.82 | 7.34 | 6.54 | 1.90 | 3.17 | 1.07 |
| **B2** | 0.63 | 9.27 | 24.23 | **44.77** | 8.95 | 6.36 | 1.79 | 2.83 | 1.08 |
| **B3** | 0.33 | 6.76 | 19.02 | 36.44 | **24.70** | 7.74 | 2.01 | 2.25 | 0.68 |
| **B4** | 0.69 | 8.19 | 22.81 | 36.06 | 10.35 | **13.67** | 2.20 | 4.07 | 1.82 |
| 4-6k | 0.79 | 12.02 | 24.03 | 36.18 | 6.13 | 8.39 | 2.39 | **7.29** | 2.58 |
| 6-10k | 0.97 | 10.47 | 20.90 | 37.03 | 4.58 | 9.31 | 1.77 | 6.15 | **8.57** |

**(2) rank-1 정량:** 보편 출력 SED `q` = 열 합 = (300-600 0.694%, **B0 10.530%**, B1 25.220%, **B2 40.077%**, B3 9.737%, B4 7.141%, 3-4k 1.952%, 4-6k 3.266%, 6-10k 1.275%).

| 입력 밴드 | 입력 가중 | TVD(row, q) | **TVD(off-diag, q_off)** |
|---|---|---|---|
| 300-600 | 0.613% | 0.0688 | 0.0667 |
| B0 | 10.425% | 0.1202 | **0.0345** |
| B1 | 25.283% | 0.0637 | **0.0244** |
| B2 | 40.105% | 0.0469 | **0.0179** |
| B3 | 9.739% | 0.1561 | 0.0686 |
| B4 | 7.152% | 0.0878 | 0.0399 |
| 4-6k | 3.298% | 0.0870 | 0.0601 |
| 6-10k | 1.292% | 0.1276 | 0.0768 |

**대각을 제거하면 모든 입력 밴드의 재분배 SED가 서로 TVD 0.018–0.077 안에 있다 — 즉 off-diagonal 재분배의 92–98%가 입력 무관이다.** 1000빈 원행렬(행-확률화, 입력>0인 752행)의 SVD도 같은 말을 한다: `σ = [2.140, 1.000, 0.607, 0.597, 0.530, …]`, rank-1이 Frobenius 에너지의 **59.8%**, rank-2가 **72.9%**(σ₂=1.000은 대각/항등 성분).

> **⇒ 측정된 `R`은 형광 재분배 커널이 아니라 `R ≈ (1−d_i)·q + d_i·δ_ij` — "고정된 캐스케이드 SED로 열화시키는 연산자"다.** MC 매크로원자가 어떤 진동수로 들어오든 거의 같은 SED로 뱉는다는 실측이며, 이는 §T4-1의 커버리지와 무관한 **연산자 자체의 성질**이다.

**(3) 산술 폐합 — E10/E12의 형상 실패를 rank-1만으로 재현한다** [F]

s8 동결 방출률의 밴드 몫(payload, `eta_coherent·dν`): B0 **1.4306%**, B1 23.5125%, B2 35.5736%, B3 20.5607%, B4 11.5438%, 300-600 0.0017%.

E10식 적용의 1차 예측 `η_new = η_tot − a + q·Σa`:

| 밴드 | `η_tot` 몫 | `q` 몫 | **예측 배율** | E12 실측 배율 **[D]** |
|---|---|---|---|---|
| 300–600 | 0.0059% | 0.694% | **×118.5** | (미보고) |
| **B0** | 1.4313% | **10.530%** | **×7.354** | **×3.19** (8.29→26.43) |
| **B1** | 23.510% | 25.220% | **×1.073** | **×1.151** (4.916→5.659) |
| B2 | 35.582% | 40.077% | ×1.127 | (미보고) |
| B3 | 20.552% | 9.737% | ×0.474 | (미보고) |
| B4 | 11.539% | 7.141% | ×0.619 | (미보고) |

**B1은 예측 ×1.073 vs 실측 ×1.151 — 6.8% 이내로 맞는다.** B0은 예측 ×7.35 vs 실측 ×3.19: 부호·자릿수 일치, 차이는 방출률 배율이 `J` 배율로 바뀔 때 다른 셸·다른 대역에서 오는 수송 기여와 결맞음 증폭의 포화 때문이다.

**이 대조의 한계(정직 기재) 3가지:** (i) 본 예측은 **방출률 배율**, E12 수치는 **`J_det/CMFGEN` 배율**이므로 동일 양이 아니다 — 수송 후 비율은 일반적으로 더 완만해진다. (ii) 본 예측의 `q`는 **iteration-11 행렬**에서, E12 수치는 iteration-10 행렬에서 나왔다(§10) — 두 행렬의 `Eabs`는 3065.50 vs 2975.93로 3% 차이라 `q`의 밴드 몫 변화는 작을 것으로 보이나 **직접 확인하지 못했다**(it10 행렬이 디스크에 없음). (iii) 예측은 국소 1차(`η_new = η_tot − a + q·Σa`)이며 셸 간 결합을 무시한다. ⇒ 이 대조는 **기전의 방향·자릿수 확증**이지 정량 재현이 아니다. 정량 판정은 §9.4의 T5가 담당한다.

**(4) "B2→B0 지배"의 정체** [F]: rank-1 연산자에서는 **모든 출력 열의 최대 기여자가 자동으로 최대 입력 밴드**가 된다. 행렬 자체의 B0 출력 열 조성(에너지 가중, 가중치 없음)은 `from B2 = 35.31%`이고, s8 `a_i` 가중(본 감사 proxy `a = eta_coherent·dν`)으로는 **`from B2 = 35.97%`** — `a`의 B2 몫 **35.57%**와 **0.4%p 이내로 일치**한다. 즉 B2→B0 점유율은 **B2의 입력 가중을 그대로 되읽은 값**이다.
(주의: E12의 54.92%는 정식 applicator + `a_i = (1−ε_MC)χ_line·J·Δν` 가중 + iteration-10 행렬로 얻은 값이고, 본 감사 수치는 proxy 가중 + 현재 디스크의 iteration-11 행렬이다. 두 차이를 분리하지 못했으므로 **본 수치는 E12의 대체가 아니라 독립 방증**으로만 쓴다.)

### 8.3 판정

> **누락 채널은 형상 실패의 원인이 아니다(기각, 사전등록 임계 1% 대비 0.68%).**
> **실제 기전은 "MC 캐스케이드의 보편 출력 SED `q`와 결정론 방출률 SED의 밴드 불일치"이며, 이는 rank-1 연산자를 어떤 입력에 적용하든 입력의 스펙트럼 형상을 `q`로 갈아치우기 때문에 발생한다.** 이는 기존 대장의 [메트릭-맹 전역 재분배](feedback_metric_blind_global_redistrib) 실패 양식과 동형이다. 행렬을 무편향으로 만들거나(E12) 미러링하거나(E13) 색인을 검증해도(E13) 이 성질은 바뀌지 않는다 — **연산자가 rank-1인 한, "R을 고쳐서" 형상을 살릴 길은 없다.**

## 9. T4-4 — 커버리지 확장에 필요한 최소 계측 변경 (설계, 구현 금지)

### 9.1 설계 원칙 (중요)

**bf/ff-heat 입력을 기존 행렬에 합치면 안 된다.** E10 소비자는 `η_j ← η_j − a_j/Δν_j + Σ_i R[j,i]·a_i/Δν_j`에서 `a_i`를 **선 반환량**(`(1−ε)χ_line·J·Δν`)으로 잡는다. bf 진입분을 같은 `R`에 섞으면 `R`의 의미가 "임의 채널 흡수 조건부 재분배"로 바뀌어, **소비자가 곱하는 `a_i`(선 전용)와 공변량이 어긋난다.** 얻는 커버리지(0.25%)보다 도입되는 공변량 오차가 크다.

⇒ **활성화 클래스별로 분리된 행렬(또는 3번째 인덱스 `class ∈ {LINE, BF, FFH}`)로 기록한다.**

### 9.2 최소 변경 목록 (호출부 · 무엇을)

| # | 파일:라인 | 추가할 것 | 비고 |
|---|---|---|---|
| **C1** | `src/lumina_cuda.cu:6170` 직후 | `double fluor_in_nu_bf = comov_nu_bf2;` `double fluor_in_e_bf = pkt_energy * d_get_doppler_factor(pkt_r,pkt_mu,t_exp);` (게이트 안에서만) | 진입 캡처. `:6250`의 `old_doppler`와 **동일 mu** 사용 필수 |
| **C2** | `src/lumina_cuda.cu:6417` 직전 (`d_ma_fate_record` 옆) | `d_fluor_matrix_record_cls(FLUOR_CLS_BF, fluor_in_nu_bf, fluor_in_e_bf, pkt_nu*d_out, pkt_energy*d_out, pkt_shell_id, via_kpkt, route_ok)` | `via_kpkt = (bf_kinetic_to_kpkt || act_level<0 경유)` — `:6229-6241` 분기에서 세팅 |
| **C3** | `src/lumina_cuda.cu:6447` 직후 | 동일 패턴으로 `comov_nu_ff`/`comov_e_ff` 캡처 | |
| **C4** | `src/lumina_cuda.cu:6505` 직전 | `d_fluor_matrix_record_cls(FLUOR_CLS_FFH, …)` + **`d_ma_fate_record(...)`(신규 N2 수리)** | ff-heat는 정의상 항상 `via_kpkt=1` |
| **C5** | `src/lumina_cuda.cu:6435` 직전 | 레거시 bf 재방출(`FLUOR_CLS_BFLEGACY`) 기록 — parity59에서 0건이지만 fail-open 방지 | |
| **C6** | `src/lumina_cuda.cu:4506-4521` | `case EVCH_MA_RAD_RECOMB: emch = 8; break;` + `CENSUS_NEMCH` 확장 + CSV 헤더 열 추가 | **신규 N1 수리**. 방출 센서스 채널 합의 폐합 회복 |
| **C7** | `src/lumina_cuda.cu:4246-4300` (`cuda_fluor_matrix_init`) | dense accumulator를 `[class][ib][ob]`로 확장 (3 class × 8 MB = 24 MB 추가; 셸군 행렬은 LINE 클래스만 유지해 상한 보존) | E11의 유한 상한 논증 유지 |
| **C8** | LFMAT001 v2 헤더 | `n_classes`, class별 `events/absorbed/reemitted/terminal` 장부. **v1 리더가 v2를 조용히 읽지 않도록 version bump + fail-closed** | |
| **C9** | `scripts/emiss_e11_fluor_matrix.py` | `read_fluor_matrix`에 **`expected_iteration` 계약 인자 추가(기본 필수)** — §10의 사고 재발 방지 | 소비자 3종(`emiss_e10_apply_redistribution.py`, `emiss_e12_diagnose.py`, `emiss_e13_index_audit.py`) 모두 통과 |

### 9.3 확장으로 무엇이 바뀌는가 — 사전등록

- **바뀌는 것:** 현재 텅 빈 **EUV 입력 열**(λ<1000 Å)이 채워진다. bf 진입 에너지의 **12.94%가 300–600 Å, 1.48%가 B0**에 있고(§7.2) 선 흡수는 그 대역에 거의 없다. 즉 확장은 **EUV→UV 하향변환 지도**를 처음으로 제공한다 — memory의 "s12+ FUV 기근"·"Co IV 형광 깔때기" 전선에 직접 쓰이는 자산이다.
- **바뀌지 않는 것(사전등록):** `q`의 rank-1 성질과 B2→B0 점유율은 **변하지 않는다**(가중 0.25%). 확장 후 재측정에서 B2→B0가 5%p 이상 움직이면 본 감사의 rank-1 진단이 반증된 것이다.

### 9.4 확장보다 먼저 해야 할 것 — 신규 오프라인 시험 T5 (사전등록)

> **T5 (오프라인, 신규 런 0):** E10 applicator에 **순수 rank-1 대리 연산자** `R*[j,i] := q_j` (대각·off-diagonal 구조 전부 제거)를 넣고 같은 s8 동결 payload에 적용한다.
> - **PASS 판정(=본 감사의 진단 확증):** `R*`가 E12의 B0(26.43)·B1(5.659)·B2→B0 점유율을 **각각 15% 이내**로 재현한다 ⇒ 실측 R의 off-diagonal 구조는 결과에 거의 기여하지 않으며, **형광 행렬 수리 노선(E10–E13) 전체를 종결**하고 Stage 3.2(VEF/ALI)로 자원을 옮긴다.
> - **FAIL 판정:** 재현 못 하면 R의 구조가 실제로 일하고 있다는 뜻이므로 §9.2 커버리지 확장이 정당화된다.
>
> T5는 `scripts/emiss_e10_apply_redistribution.py --matrix-format formal`에 대리 행렬 fixture(`write_fixture_matrix`)를 먹이는 것만으로 실행 가능하다. **비용: CPU 수 분.**

---

## 10. 잣대 사고 등재 — 형광 행렬 아티팩트가 런 중 교체되었다 [F, 신규·중대]

| 사실 | 값 |
|---|---|
| E12가 기록한 행렬 | `iteration 10`, sparse edges **473,045**, sha256 **`2b65dba6…d01c99b`** (`docs/CODEX_EMISS_E12.md:42,67,94`) |
| E12가 쓴 런디렉토리 | `E12_RUN=/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828` (`docs/CODEX_EMISS_E12.md:205`) — **본 감사와 동일** |
| **현재 디스크의 같은 경로** | 헤더 `iteration: 11`, edges **468,330**, sha256 **`08ff3312…5af6`** |
| 시각 | payload `chieta_iter10`/`emiss_ab_iter10.A` 08:36:00 → E12 산출물 08:55–09:03 → E13 09:05–09:32 → **행렬 파일 mtime 09:45:28** → stdout 종료 09:51 |
| 왜 탐지되지 않았나 | (i) 생산 writer가 **매 MC pass마다 같은 경로를 덮어쓴다**(E11 §2.2 설계 그대로) (ii) `.sha256` 사이드카가 페이로드와 **함께** 갱신되므로 `sha256sum -c`는 언제나 PASS (iii) `read_fluor_matrix`는 `iteration`을 헤더 dict에 담기만 하고 **검증하지 않는다**; 소비자 3종 어디에도 행렬 iteration 계약이 없다(`emiss_e10_apply_redistribution.py:287`의 `"iteration"`은 **E9 payload** 헤더) |

**귀결 3가지**
1. **E12·E13은 현재 런디렉토리에서 재현 불가능하다.** 문서의 재현 명령을 오늘 실행하면 다른 행렬(it11)을 조용히 소비한다.
2. **E10/E12/E13은 iteration-10 행렬을 iteration-10 장에 적용한 것이 맞다**(타이밍상). 즉 결론 자체가 무효화되지는 않는다. 그러나 그 정합은 **설계가 아니라 우연**이었다.
3. **"fail-closed 사이드카"가 아티팩트 교체를 막지 못한다**는 일반 교훈. 페이로드와 함께 재계산되는 해시는 무결성은 지키지만 **동일성(identity)은 지키지 못한다.** 계약은 **내용 안의 불변식**(여기서는 헤더 `iteration`)으로 걸어야 한다 ⇒ §9.2 C9.

**권고(계측):** `LUMINA_FLUOR_MATRIX_DUMP` 경로에 iteration suffix를 강제하거나(생산 변경), 최소한 소비자에 `--expected-matrix-iteration`을 필수 인자로 넣는다(오프라인 변경만으로 가능).

---

## 11. UNRESOLVED 갱신

| # | 상태 | 내용 |
|---|---|---|
| U1 | **유지·보강** | `J_blue` 과대 vs C1 구간장 기근. 본 감사에서 **railing 실물 확인**(s16 이상 다수 빈이 `T_R_bin=250000 K`, `W_bin~1e-5`, `mode` 분포 fit 8368/empty 4454/pin 1546/degen 32). 방향 확정은 여전히 T2b 소관 |
| U2 | **CLOSED** | `linetherm`=0은 `ARTIS-PARITY D4`가 `LUMINA_LINE_THERM`을 무력화한 결과(`stdout.log:269`, `cuda.cu:7266-7268`). 배선 버그 아님 |
| U3 | **CLOSED(재분류)** | fb 방출 100%가 λ<900 Å(252–759 Å, 중앙값 601.5 Å) = 물리. 단 **방출 센서스가 fb 채널에 구조적 맹목**이라는 잣대 사실로 재등재 |
| U4 | **CLOSED** | 형광 행렬 입력 클래스 편향 크기 = **0.235%**(진입 에너지) / **0.234%**(출구), B0에서도 0.64%. 형상 실패 설명력 없음 |
| U5 | 유지 | `diag_macro_branch`의 준위 균등가중 — `p_iup=88%`의 트래픽 가중값 미측정 |
| U6 | **하향** | 자극방출 포화의 조건부 발산: §4.1에서 T_rad 의존이 `1 − (w_up/w_lo)e^{−hν/kT_rad}` 하나로 폐합됨을 증명. maser clamp 발화 조건은 `hν < kT_rad ln(1/W)` (s3에서 λ>6014.6 Å) ⇒ **UV에서는 원리적으로 불가**. IR 잔여만 미조사 |
| U7 | 유지 | 7D "97.7%"의 순환성·비교대상 오정합 |
| U8 | 유지 | EPAY 독립 장부 부재 |
| U9 | 유지 | ARTIS 20.2% 재현 |
| U10 | 유지 | TF32 R_bf 레인 오차 미계량 |
| **N1** | **신규** | `EVCH_MA_RAD_RECOMB`가 방출 센서스 emch 스위치에 없음(`cuda.cu:4506-4521`) ⇒ 채널 열 합이 총 방출과 불일치(선방출의 0.024%) |
| **N2** | **신규** | ff-heat 활성화 경로(`cuda.cu:6466`)에 `d_ma_fate_record` 없음 ⇒ MA-FATE 센서스 누락(403건) |
| **N3** | **신규** | 준위인구 정규화 불일치: 분자 `W·e^{−E/kT_rad}` vs 분모 `Z(T_e, W=1)` (`plasma.c:2640/7148` vs `:1903-1907`) ⇒ `Σ_k n_k ≠ n_ion`. T_rad 핀과 **독립**인 결함 |
| **N4** | **신규·중대** | 형광 행렬 아티팩트가 런 중 교체됨 + 소비자에 iteration 계약 부재(§10) ⇒ E12/E13 재현 불가 |
| **N5** | **미해결** | `q`(보편 출력 SED)를 무엇이 결정하는가. rank-1이라는 **현상**은 확정했으나 그 **씨앗**(방출 가중 `A_ul·β·hν` × 준위인구? 아니면 `p_iup≈0.886`의 상향 지배로 인한 캐스케이드 정상상태?)은 본 감사로 분리 불가 |

---

## 12. 상신 (우선순위)

1. **T5(§9.4)를 최우선.** 오프라인·CPU 수 분. rank-1 대리 연산자가 E12를 재현하면 **E10–E13 노선을 종결**할 수 있다 — 형광 행렬 수리에 더 투입할 근거가 사라진다.
2. **T_rad 핀은 UV 수리 후보에서 제외**하고 P1–P5 장부(§5.2)로 이관. 별도로 **N3(정규화 불일치)** 는 핀과 독립 결함이므로 따로 판단.
3. **N4(아티팩트 교체) 즉시 처방**: 소비자 3종에 행렬 `iteration` 계약 강제(오프라인 변경만). 이것 없이는 E-시리즈 어떤 후속도 증거력이 없다.
4. **커버리지 확장(§9.2)은 T5 이후.** 그 가치는 "E10 수리"가 아니라 **EUV 입력 열 확보**(FUV 기근·Co IV 깔때기 전선의 자산)에 있다 — 목적을 그렇게 재정의해 발주할 것.
5. **N1·N2는 계측 부채 상환분**으로 묶어서 처리(둘 다 CSV 열/호출 1줄).

---

## 부록 A — 본 감사가 직접 실측한 값 (전부 `RUN=/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828`)

| 값 | 출처 |
|---|---|
| `T_rad` 50셸 uniq=1 = 10470.093240; `W` 0.2979→0.0108 | `$RUN/lumina_plasma_state.csv` |
| 재구성 `T_rad^mom` s0 19397.4 / s3 15349.0 / s8 14152.2 / s49 11704.0; `W_mom>1` (s0–s11) | `$RUN/emiss_ab_iter10.A` (LCMFCE01, `iteration=10`) + `scripts/…` 없음 → 스크래치 `t3_trad.py` |
| `R_band = ⟨J⟩/(W_pin B(T_pin))`: s3 B0 4.31e3 / B1 1.16e3 / B2 2.52e2 | 동상 |
| `chi_abs/chi_tot` s0 B1 2.29e-4 · B2 4.88e-4; `eta_coh/eta_tot` ≥0.9995 (B1–B4 전 셸) | 동상 |
| s0 UV per-bin `T_R_bin` 16573–178861 K, `W_bin` 1.57e-2–3.97 | `$RUN/lumina_census_perbin_field.csv` |
| c1 bins mode: fit 8368 / empty 4454 / pin 1546 / degen 32 (14,400행) | `$RUN/lumina_c1_bins.csv` |
| `radeq_root=root-found` 600/600, no-root 0 | `$RUN/stdout.log` (`[TEHOLD]`) |
| 비-NLTE 이온의 UV(1700–3000) Στ 몫 = 3.523/227,895.6 = **1.546e-5** | `$RUN/stdout.log:33441-33459` |
| b_k s0: Fe III p50 0.549 / Co III 0.458 / Ni III 0.041 / Fe II 0.991 / Co II 1.29 | `$RUN/lumina_levelpop.csv` |
| MA fate 총 485,228,598 (rad_deexc 485,105,515 / col_deexc 4,351 / rad_recomb 118,732) | `$RUN/lumina_census_ma_fate.csv` |
| kpkt exit 전 셸: ff 1,335 / fb 3,016 / **collexc 1,289,180** / bte 0 / macap 0 | `$RUN/lumina_census_kpkt_exit.csv` |
| heating: bf **1,291,431** / ff **403** | `$RUN/lumina_census_heating.csv` |
| emission: line **2725.989** / escatter 23.0863 / ff 6.3e-3 / fb·bte·macap·bf_reemit·linetherm **0.0** | `$RUN/lumina_census_emission.csv` |
| 이벤트 로그 it11: 970,557,187 events, **570,557,187 dropped** (cap 400M) | `$RUN/stdout.log:37972` |
| 이벤트 로그 접두부 채널: MA_ACT_BB 199,492,039 (E=1572.68) / RPKT_BF_ABS 497,840 (E=3.6989) / KPKT_COLLEXC 496,956 (E=3.6927) / HEAT_FF 165 (E=1.298e-3) / MA_RAD_RECOMB 48,735 (E=0.3824) / KPKT_FB 1,223 (E=9.705e-3) / KPKT_FF 545 / HEAT_LINETHERM **0** | `$RUN/lumina_events.bin` (스크래치 `t4_events.py`, `t4_bands.py`) |
| KPKT_FB λ ∈ [252.3, 758.5] Å, E(λ<900 Å)=100.00% | 스크래치 `t4_fb.py` |
| 형광 행렬 파일 헤더: **iteration 11**, events 483,936,781, edges 468,330, Eabs 2975.9288, rel 2.80e-8 | `$RUN/fluor_matrix_iter10` |
| 행렬 `q` = (B0 10.530%, B1 25.220%, B2 40.077%, B3 9.737%, B4 7.141%) | 동상 |
| off-diagonal TVD(row,q): B0 0.0345 / B1 0.0244 / B2 0.0179 / B3 0.0686 / B4 0.0399 | 동상 |
| 1000빈 행-확률 연산자 SVD: σ=[2.140, 1.000, 0.607, 0.597, 0.530, …], rank-1 = 59.8% | 동상 |
| s8 `eta_coherent·dν` 밴드 몫: B0 1.4306% / B1 23.5125% / B2 35.5736% / B3 20.5607% / B4 11.5438% | `$RUN/emiss_ab_iter10.A` |

## 부록 B — 본 감사가 직접 확인한 소스 좌표 (T3T4 신규분)

| 좌표 | 내용 |
|---|---|
| `src/lumina_plasma.c:4483-4502` | 내부상향 = `B_lu[1 − (w_up/w_lo)e^{−hν/kT_rad}]·β·J_blue` — **T_rad 의존의 폐합 형태** |
| `src/lumina_plasma.c:2392` | `if (g_simul_on == 1) return;` — 성운 Saha 이온 분할 무력화 |
| `src/lumina_plasma.c:1903-1907` | ARTIS-PARITY B3: `T_part := T_e, W := 1` (분배함수만) |
| `src/lumina_plasma.c:2640` / `:7148` | 소비처는 여전히 `e^{−E/kT_rad}` + `W` — **N3 정규화 불일치의 두 좌표** |
| `src/lumina_plasma.c:7145-7150` | `chi_bf` 준위인구 = 희석-Boltzmann@T_rad |
| `src/lumina_plasma.c:8836-8846` | `bf_rate_pop` |
| `src/lumina_plasma.c:11564-11660` | radeq `J_eff = β·W·B_ν(T_rad) + (1−β)·S̄` |
| `src/lumina_bf_gemm.cu:82-95` | GPU bf 레인의 같은 희석-Boltzmann@T_rad |
| `src/lumina_cuda.cu:9879-9882` | PURE-CMFGEN 블록의 `return 0` — `:10447`(solve_radiation_field) 미도달의 직접 증거 |
| `src/lumina_cuda.cu:6092-6132` | LINE 분기 + **형광행렬 유일 호출부** |
| `src/lumina_cuda.cu:6265` | **bf 활성 MA** — 형광행렬 기록 없음 |
| `src/lumina_cuda.cu:6466` | **ff-heat 활성 MA** — 형광행렬·MA-FATE 둘 다 기록 없음 |
| `src/lumina_cuda.cu:4506-4521` | 방출 센서스 emch 스위치 — `EVCH_MA_RAD_RECOMB` 부재(N1) |
| `src/lumina_cuda.cu:4542-4548` | `EVCH_KPKT_COLLEXC_BB`를 kx에서 의도적 제외 ⇒ `collexc` 열 = **bf/ffh 전용** |
| `src/lumina_cuda.cu:4610-4612` | 센서스가 cap·필터보다 **먼저** 발화 ⇒ 센서스는 무-cap |
| `src/lumina_cuda.cu:7266-7268` | `ltherm_on = env && !artis_parity_enabled() && !fix_no_ltherm` (U2 해소) |
| `src/lumina_cmfgen.c:280-352` | LCMFCE01 배열 순서: `r_edge, nu, dnu, chi_tot, chi_es, chi_tot·S_fixed, chi_es·J, eta_total_audit, J` |
| `scripts/emiss_e11_fluor_matrix.py:85,177` | 헤더 `iteration` 파싱만 하고 **검증 없음**(N4) |

## 부록 C — 재현 명령 (스크래치 스크립트, 생산 트리 미변경)

```bash
SP=/tmp/claude-10396/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/50011b1c-ea14-4956-add2-6a1c0478ce63/scratchpad
RUN=/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828

# T3: 셸별 T_rad 재구성 (모멘트 / 밴드 색온도 / R_band)
python3 $SP/t3_trad.py $RUN/emiss_ab_iter10.A

# T4: 이벤트 로그 채널 장부 (8 GB 스트리밍, ~3분)
python3 $SP/t4_events.py $RUN/lumina_events.bin
python3 $SP/t4_bands.py  $RUN/lumina_events.bin
python3 $SP/t4_fb.py     $RUN/lumina_events.bin

# T4: 행렬 헤더 / rank 구조 (기존 리더 사용, 읽기 전용)
python3 scripts/emiss_e11_fluor_matrix.py $RUN/fluor_matrix_iter10
```
