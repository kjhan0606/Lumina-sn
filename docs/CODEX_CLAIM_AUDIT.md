# 판정 요약

현재 산출물이 직접 지지하는 범위는 다음까지다.

- 현 결정론 source에는 same-bin coherent return이 크다.
- s8 BALL에서 post-EPAY source 분해상 `S_fixed/S_total`이 매우 작다.
- uniform reinjection과 계측된 LINE 행렬 reinjection은 모두 UV 형상을 개선하지 못했다.
- population-native T2는 필요한 권위 데이터가 없어 `UNRESOLVED`다.
- 따라서 Stage 3.2 ALI/MALI는 강하게 정당화된 다음 후보지만, 산출물이 “유일한 해법”이나 “원인 확정”까지 증명하지는 않았다.

아래에서 INFLATED를 결정 영향도 순으로 먼저 배치했다.

---

# INFLATED

## 1. T1이 재분배 노선 전체를 닫았고 ALI가 유일한 본진이다

> “재분배-연산자 노선(E10–E13)은 **산술적으로 실패가 예정**돼 있었다(rank-1 R × SED 불일치). T1이 그 노선 전체를 닫았다: **선 전달은 소스 사전주입이 아니라 수송 해 안에서 자기일관되게 풀려야 한다 = ALI**. 따라서 **Stage 3.2(VEF/ALI)가 UV 수리의 유일한 본진**이며, 나머지는 계기 수리(N4·T2 덤프·N2)와 독립 결함 등재(N3)다.”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:64)

근거 산출물:

- `validation/uv_t1t2/t1_construction.json`
  - `R_probability_per_output_bin = 0.001`
  - `n_output_bins = 1000`
  - `removed_line_return_energy = injected_uniform_energy = 0.0067651957989972485`
  - `relative_energy_error = 0.0`
- `validation/uv_t1t2/t1_summary.json`
  - `preregistered_readout = "SAME_BIN_COHERENCE_ASSUMPTION_IS_THE_PROBLEM"`
- [CODEX_UV_T1T2.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_UV_T1T2.md:51): 제거한 source energy를 1000개 출력 빈에 같은 확률로 사전주입하는 shell-8 frozen construction.
- `validation/emiss_t5/verdict.json`
  - `route_disposition = "UNRESOLVED"`
  - 이유: artifact/request contract 충돌과 same-generation B4 잔차.

판정: **INFLATED**

넘은 범위:

T1은 “shell 8의 동결된 source에서 제거 energy를 균일하게 사전주입하는 대리모형”을 반증했다. owner-resolved redistribution, line/shell covariance 보존, transport 안에서의 redistribution, 다른 Λ 분할까지 모두 반증한 것은 아니다. 특히 T5의 최종 산출물 자체가 노선을 `UNRESOLVED`로 둔다.

결정 영향:

E10–E13을 종결하고 인력·계기를 Stage 3.2에 집중시키며 ALI를 유일 노선으로 승격했다. 이번 감사에서 가장 영향이 큰 과장이다.

정확한 재서술:

> T1은 shell-8 frozen payload에서 제거한 coherent-return energy를 균일 source로 사전주입하는 대리모형을 반증했다. 이는 ALI/MALI 같은 transfer-consistent 접근을 우선 시험할 근거지만, ALI의 유일성이나 모든 redistribution 노선의 실패를 증명하지는 않는다.

---

## 2. E8이 UV 11.977배의 직접 원인을 완전히 설명했다

> “필요한 이득 5247.41×와 0.00152% 이내로 일치하여 UV 11.977× 과잉의 amplitude를 완전히 설명합니다.”

출처: [CODEX_EMISS_E8_SUMMARY.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8_SUMMARY.md:8)

> “진폭의 직접 원인 … E8 폐합(이득 5247.49 vs 필요 5247.41)”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:9)

근거 산출물: `validation/emiss_e8/summary.json`

정의 문자열:

- `cell_eps_eff = "eta_fixed/(eta_fixed+eta_coherent)"`
- `band_eps_eff_source = "integral[(eta_fixed/chi_total)dnu]/integral[(eta_total/chi_total)dnu]"`
- `literal_eta_integral_check = "integral eta_fixed dnu/integral eta_total dnu"`
- `recycle_gain_source = "1/eps_eff_source=S_total_band/S_fixed_band"`

s8 BALL:

- `eps_eff_source = 0.00019056728565`
- `recycle_gain_source = 5247.4903894`
- `J_over_CMFGEN = 11.97709747`
- 필요한 이득 `5247.410557`
- `S_total/J = 1.00001521364`

전체 보고서는 다음 한계를 명시한다.

> “동일 시점 대수 폐합이지 독립 인과 실험 아니다.”

출처: [CODEX_EMISS_E8.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8.md:16)

판정: **INFLATED**

넘은 범위:

두 5247배 수치는 모두 같은 분모 `S_fixed`를 공유한다. “필요한 이득” 계산에서도 CMFGEN 값이 소거되어, 독립적인 CMFGEN 폐합이 아니다. `S_total/J≈1`도 source가 바로 그 `J`를 포함하는 fixed-point 자기일관성 검사다. CMFGEN의 대응 `ETAL/CHIL` 또는 `eps`는 산출물 자체에서 `UNRESOLVED`다.

결정 영향:

same-bin recycling을 직접 원인으로 확정하고 Stage 3.2의 물리 표적을 정하는 데 사용됐다.

정확한 재서술:

> 현재 payload의 post-EPAY source 분해에서는 s8 BALL의 fixed-source 몫이 `1.90567×10⁻⁴`이고 coherent-return source가 지배한다. `S_total/J≈1.000015`는 그 source 분해의 자기일관성을 보이지만, CMFGEN 대비 11.977배 과잉의 독립 인과 폐합은 아니다.

---

## 3. Stage 3.1이 MC 장을 독립적으로 재현해 MC 수송을 기각했다

> “Stage31의 인증 formal solver가 같은 `chi,eta`로 `J_MC` UV의 97.7181%를 재현했다. 따라서 전체 UV 진폭에 대한 ‘MC 수송 연산자 단독 결함’은 기각됐다.”

출처: [CODEX_UV_CENSUS.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_UV_CENSUS.md:28)

근거 산출물: `docs/s31_results/stage31_bench_round7d.json`

- `schema = "stage31-cmf-field-bench-v1"`
- BALL `J_det_over_J_MC = 0.9771809393334623`
- `eta_input = "captured eta_total"`
- `chi_coherent_input = 0.0`

생산자 확인:

- [stage31_cmf_field_bench.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/stage31_cmf_field_bench.py:613): `j_producer = arrays[8]`
- [CODEX_STAGE31_BENCH7D.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_BENCH7D.md:112): `J_MC`는 sidecar payload의 `J_producer`.
- 해당 `eta_total`에는 `chi_es * J_producer`가 이미 들어간다.

판정: **INFLATED**

넘은 범위:

이는 독립적인 live MC estimator와 formal solver의 비교가 아니다. 같은 deterministic producer가 만든 `J_producer`가 source 안과 비교 대상 양쪽에 사용된 replay-consistency 시험이다. “같은 생산자의 산출물을 두 번 센” 사례에 해당한다.

결정 영향:

MC transport를 주원인 후보에서 제외하고 χ/η assembly와 ALI 쪽으로 노선을 집중하는 데 사용됐다.

정확한 재서술:

> Stage 3.1은 captured `J_producer`를 포함해 구성된 `eta_total`을 풀었을 때 그 deterministic producer field의 97.7181%를 재현했다. live MC estimator에 대한 독립 수송 검증은 아직 없다.

---

## 4. 실제 행렬이 사실상 rank-1이고 B0 ×7.35가 필연이다

> “R이 대각 제외 사실상 rank-1(SVD 59.8%), 보편 출력 SED q의 B0 몫 10.53% vs 결정론 방출률 B0 몫 1.43% ⟹ 적용 시 B0 ×7.35는 산술적 필연.”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:55)

근거 산출물: `validation/emiss_t5/rank1_residual_summary.json`

- 대각 포함:
  - `optimal_rank1_fraction_of_frobenius_energy = 0.59803247535`
  - `optimal_rank1_relative_frobenius_residual = 0.63400908877`
- 대각 제외:
  - rank-1 fraction `0.62457`
  - relative residual `0.61272`
- `q_B0 = 0.105297`
- `q_B2 = 0.400766`
- 입력 세대: iteration 11, SHA `08ff…`, 468,330 edges.

`validation/emiss_t5/verdict.json`:

- preregistered B0/B1/B2→B0 proxy는 15% 기준 통과.
- `route_disposition = "UNRESOLVED"`
- same-generation B4 잔차는 약 20.6–21.8%.

판정: **INFLATED**

넘은 범위:

59.8%는 “rank-1 항등식”이 아니며 Frobenius residual이 0.63이다. ×7.35는 q의 B0 source share를 기존 source share로 나눈 pure rank-1 emission proxy다. transported `J`의 실제 반사실이 아니고, iteration-11 q와 iteration-10 E12 결과도 섞여 있다.

결정 영향:

E10–E13 실패를 구조적으로 예정된 결과로 선언하고 해당 노선을 닫는 근거가 됐다.

정확한 재서술:

> Rank-1 proxy는 사전등록된 B0/B1/B2→B0 요약량을 15% 안에서 근사하지만 전체 행렬에는 약 0.61–0.63의 상대 Frobenius residual이 남는다. ×7.35는 pure-q source-share 예측이며 실제 transported B0 반사실은 아니다. 최종 노선 판정은 `UNRESOLVED`다.

---

## 5. N9가 “조립된 η_line의 폐기 비율”을 실측했고 η-only 시험을 무효화했다

> “thin bin·s≥5의 `S_fixed`가 `w_n·(bf_Milne_η + χ_line,th·B(T_e))`로 재구성되어 조립된 `η_line`이 폐기됨. 실측 하한 s8 UV 빈의 ≥30.6%, s16+에서 98.7–100%. ⟹ η만 바꾸는 모든 시험(E4 B-lane·E5 B2-lane·T2의 population-native η)은 그 영역에서 무효(inert)였다.”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:75)

근거 산출물:

`/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_t2n9_offline/n9_summary.json`

- `schema = "lumina-uv-n9-v1"`
- `energy_definition = "eta_fixed_post_EPAY * exact_frequency_overlap * shell_volume"`
- disposition counts: `5000/10696/34304/0`
- `epay_scale_not_reproducible = true`
- 항등 정의: `eta_rate_line = chi_line_th * B_nu(Te)`

`n9_energy_shell_band.csv`:

- s8 BALL post-energy fraction `0.9956303809148374`
- s≥5 B1–B4는 정확히 `1.0`.

생산자 계약:

- `linepop_iter10.manifest.json`: `eta_line_epoch = "pre-EPAY, pre-split (assemble line loop)"`
- 실제 산출 비율은 `eta_fixed_post_EPAY`에 대한 것이다.
- writer의 disposition 재구성은 실제 production branch의 `acc_w > 0` 조건을 보존하지 않는다. 실제 branch는 [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1704), writer 측 분류는 [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:904).

판정: **INFLATED**

넘은 범위:

production branch가 조건 충족 시 pre-line shape를 바꾸는 것은 코드상 사실이다. 그러나 N9의 비율은 “폐기된 pre-EPAY `eta_line` energy”가 아니라 “writer가 rate-shape로 분류한 셀에 놓인 post-EPAY fixed emissivity”다. 여기에는 continuum/deposition도 포함되며 실제 branch count도 완전히 복원되지 않는다.

B1–B4의 정확한 `1.0`은 측정된 100% 폐기율이 아니다. mask가 해당 밴드의 모든 denominator 셀을 덮어 numerator와 denominator가 같은 배열이 된 항등식이다.

결정 영향:

E4/E5/T2의 η-only 결과를 물리적으로 inert였다고 폐기하고, T2 및 계기 수리 방향을 바꿨다.

정확한 재서술:

> N9는 post-EPAY `eta_fixed` 중 writer가 rate-shape disposition으로 재구성한 셀에 놓인 energy fraction을 측정한다. 이는 pre-EPAY assembled `eta_line`의 실제 폐기 fraction이나 모든 η-only 반사실의 inert fraction이 아니다. 정확한 branch coverage는 `acc_w`와 EPAY scale 입력이 직렬화되지 않아 `UNRESOLVED`다.

---

## 6. E9가 산란 재순환 기전을 진폭 수준에서 확정했다

> “BALL 사전등록 `0.93665×` 대비 산술 `0.93596×`, 재조립 source `0.92791×`, stage31 `J_det=0.93229×`로 모두 적중했습니다. 산란 재순환 기전은 진폭 수준에서 확정입니다.”

출처: [CODEX_EMISS_E9_SUMMARY.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E9_SUMMARY.md:5)

근거 산출물:

`validation/emiss_e9/emiss_e9_effective_iter10.manifest.json`

- `construction = "cellwise J_old*(eps_old/eps_MC), then eta_fixed+[chi_es_proxy+(1-eps_MC)chi_line_proxy]*J_effective"`
- `chi_es_proxy = "per-shell minimum payload chi_coherent (capture-epoch line-free proxy)"`
- `e9_diagnostic_only = true`
- `repair_implemented = false`

`validation/emiss_e9/summary.json`:

- `eps_MC = 0.002436822`
- `gain = 410.3705`

판정: **INFLATED**

넘은 범위:

E9는 사전 구성한 frozen scalar proxy에 formal solver가 예상대로 반응하는지 확인했다. native χ/η, population feedback, line-owner coupling 또는 물리적 destruction operator를 직접 시험하지 않았다.

결정 영향:

redistribution E10–E13을 개시하고 E8의 인과 해석을 강화했다.

정확한 재서술:

> E9는 capture-epoch proxy와 scalar `eps_MC`로 구성한 frozen source가 사전등록된 BALL 진폭 예측을 재현함을 보였다. 이는 해당 대리모형의 solver response 검증이며 native coupled 기전의 확정은 아니다.

---

## 7. T4의 0.2353%가 전체 사건의 실측 비율이므로 채널 누락은 기각된다

> “누락(bf·ff-heat) 진입 0.2353%·출구 0.2343%·B0 0.64%, 임계 1% 미달.”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:55)

근거 산출물/로그:

- 같은 런의 이벤트 로그:
  - attempted events `970,557,187`
  - stored prefix `400,000,000`
  - dropped `570,557,187`
- `validation/emiss_e9/redistribution_summary.json`
  - `status = "TRUNCATED_PREFIX-not-an-unbiased-random-sample"`
  - `stored_fraction = 0.41213439595`

판정: **INFLATED**

넘은 범위:

0.2353%, 0.2343%, 0.64%는 비무작위 4억 atomic-reservation prefix 안의 조건부 비율이다. 전체 iteration을 대표한다는 산출물 정의가 없으므로 1% 의사결정 threshold와 직접 비교할 수 없다.

결정 영향:

bf·k-packet 등 채널 확장을 기각하고 ALI-only 노선을 강화했다.

정확한 재서술:

> 저장된 4억-event prefix에서는 누락 채널 몫이 진입 0.2353%, 출구 0.2343%, B0 0.64%였다. prefix가 전체 사건의 비편향 표본이 아니므로 full-iteration 비율과 1% threshold 판정은 `UNRESOLVED`다.

---

## 8. T3가 T_rad 핀의 UV 설명력을 약 0으로 측정했다

> “T3 T_rad 핀 | **기각(공범 아님)** | 핀 수리 시 UV 상향계수 0.02–3.4%(임계 5% 미달).”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:54)

근거 산출물의 실제 범위:

- `p_iup` 보정 proxy 사례: 약 0.02–3.4%.
- s0, 1700–3000 Å에서 non-NLTE τ share: `1.546×10⁻⁵`.
- repaired-pin 상태로 수행한 frozen `J` solve, flux solve 또는 production counterfactual은 없다.

판정: **INFLATED**

넘은 범위:

측정된 것은 일부 pumping coefficient와 구조적 τ proxy의 민감도다. UV field나 emergent flux의 변화율을 측정한 것이 아니다.

결정 영향:

T_rad를 UV 후보에서 제거하고 ALI 노선으로 자원을 집중했다.

정확한 재서술:

> parity59에서 감사한 pumping proxy의 변화는 최대 3.4%였고 s0의 선택 창에서 non-NLTE τ share는 매우 작았다. repaired T_rad가 UV field와 flux에 미치는 직접 영향은 측정되지 않았다.

---

## 9. “UV가 42.9%에서 움직인 적이 없다”와 “UV 표적 변경이 없었다”

> “UV는 2026-07-06(ff58168) 이후 42.9%에서 움직인 적이 없고, 그 사이 모든 변경은 (a) UV를 측정조차 안 했거나 (b) byte-identical이거나 (c) 역방향이었다. ‘수리를 시도했으나 실패’가 아니라 ‘UV를 표적으로 삼은 변경이 사실상 없었다’ — 이것이 1년 정체의 실체.”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:30)

근거:

- 보존된 accepted measurement에는 42.9% 이후 지속 개선 증거가 없다.
- 반대로 여러 변경은 UV를 측정하지 않았다.
- 마지막 관련 commit은 `47bfa20`이고 이후 소스·문서에는 미커밋 상태가 존재한다.

판정: **INFLATED**

넘은 범위:

“개선 측정이 없다”는 “실제로 움직이지 않았다”와 같지 않다. 측정하지 않은 구간의 UV 값과 변경 의도는 산출물로 확정할 수 없다.

결정 영향:

1년 정체 원인 서사와 향후 mandatory UV regression 정책의 근거가 됐다.

정확한 재서술:

> 보존된 accepted measurement에서는 42.9% 이후 지속 개선을 확인할 수 없다. UV를 측정하지 않은 변경이 많고 미커밋 계보도 있어 실제 중간 trajectory와 모든 변경의 표적은 `UNVERIFIABLE`하다.

---

## 10. N9/T2의 상위 이온을 Fe III·Ni III·Co III로 보고했다

> “상위 이온 **Fe III 5,924행**, Ni III 5,662행…”

출처: [VERIFICATION_REGISTERS.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERIFICATION_REGISTERS.md:23)

근거 산출물:

`t2_nonpositive_population_forensics.json`

- 필드 `ion = 3`
- 원값은 0-기반 `ion_number`.
- 저장 convention에서 `ion=1`이 II, `ion=2`가 III이므로 `ion=3`은 IV다.

판정: **INFLATED**

넘은 범위:

행 수와 ion index는 산출물에 있으나 분광학적 표기가 한 단계 낮게 번역됐다.

결정 영향:

population coverage 보강 대상 이온의 우선순위를 잘못 표시할 수 있다.

정확한 재서술:

> 상위 결손은 Fe IV 5,924행, Ni IV 5,662행이다. 같은 방식으로 `ion=3`인 Co도 Co IV로 표기해야 한다.

---

## 11. “명시적 self-coupling의 5247×q⁻² 이득”이 측정된 결합 이득이다

> “Stage 3.2가 제거해야 할 것은 물리적 fluorescence가 아니라 명시적 self-coupling의 \(5247\times q^{-2}\) 이득이다.”

출처: [CODEX_STAGE32_ALI_DESIGN.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE32_ALI_DESIGN.md:545)

근거:

- 5247은 E8의 `S_total/S_fixed` source-decomposition ratio.
- 설계 문서에서 q는 population/source sensitivity의 근역전 척도.
- 실제 coupled operator는 문서에서 schematic하게
  `M_coupled ~ resolvent Λ · dS/dn · dn/dJ`
  로 정의된다.
- q 분포, coupled spectral radius, 실제 곱 `5247×q⁻²`는 측정되지 않았다.
- 같은 문서는 native coupled UV 결과를 `UNRESOLVED`로 둔다.

판정: **INFLATED**

넘은 범위:

서로 다른 진단 척도를 곱한 위험도 heuristic을 실제 측정된 iteration gain처럼 서술했다.

결정 영향:

MALI 구조와 preconditioner 설계를 선택하는 논거다.

정확한 재서술:

> E8 source 분해는 s8 BALL에서 약 5247의 resolvent-scale 위험을 보이며, near-inversion population source 민감도는 q에 강하게 의존할 수 있다. 실제 coupled product와 spectral radius는 아직 측정되지 않았다.

---

# SUPPORTED

## 12. 현재 source에 same-bin coherent line term이 존재한다

> “`chi_es = chi_e + (chi_ln − chi_ln_th)`”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:9)

근거: [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1641)

해당 source는 fixed term과 `chi_es`를 분리하고 transfer source에 same-bin `J`를 소비한다.

판정: **SUPPORTED**

한계:

현재 checkout의 코드 구조에 대한 판정이다. capture binary가 바로 이 dirty source에서 빌드됐다는 commit/binary provenance는 별도 확인되지 않았다.

---

## 13. E8의 수치적 source 분해와 CMFGEN 대응량 `UNRESOLVED`

주장:

> “s8 UV `chi_coherent/chi_total = 97.7713%` … `eps_eff=1.90567×10⁻⁴`, 재순환 이득은 5247.49×.”

> “CMFGEN 등가 `eps`와 배율은 `ETAL/CHIL` depth-frequency dump 부재로 `UNRESOLVED` 처리했습니다.”

출처: [CODEX_EMISS_E8_SUMMARY.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8_SUMMARY.md:5)

근거: `validation/emiss_e8/summary.json`의 앞서 인용한 정의와 수치.

판정: **SUPPORTED**

단, “완전히 설명한다”는 별도 INFLATED 판정이다.

---

## 14. T1의 해당 construction 안에서 B0가 악화했다

주장:

> “균일 R에서도 B0 8.29→25.91”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:51)

근거:

- `t1_construction.json`: 1000개 빈 균일 확률, 제거 energy 전량 재주입.
- `t1_summary.json`: B0 `8.29055 → 25.90856`.
- 반복 출력 hash 일치.

판정: **SUPPORTED**

이 판정은 해당 균일 source-preinjection contract에만 적용된다.

---

## 15. T2 population-native 시험은 권위 데이터 부족으로 `UNRESOLVED`

주장:

> “T2 native χ+η | UNRESOLVED(계기 부족)”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:52)

근거:

`t2_C_population_coverage.json`

- selected rows `1,169,145`
- missing rows `28,949`

`t2_nonpositive_population_forensics.json` 정의:

> “affected row iff tau_from_pops, n_lower, n_upper, or S_l_pop is nonpositive; no recorded A value is substituted…”

- `actual_solver_negative_population_rows = 0`
- `undefined_minus_one_sentinel_rows = 28949`
- window: 600–3000 Å
- `tau_min = 1e-12`
- selected shells: 5

판정: **SUPPORTED**

단, 결손 이온의 III 표기는 IV로 고쳐 읽어야 한다.

---

## 16. E10/E12 적용 결과가 해당 계약 안에서 형상을 악화했다

주장:

> “E10 prefix는 B0 8.29→20.91, E12 full은 B0 26.43으로 악화했다.”

근거:

- `validation/emiss_e10/diagnosis_summary.json`
  - `shape_gate = false`
- `validation/emiss_e12/diagnosis_summary.json`
  - readout: shape moves away.
- E12 strict guard는 `UNRESOLVED`; formal-tolerance auxiliary도 실패.

판정: **SUPPORTED**

한계:

이는 저장된 LINE-matrix와 해당 source projection contract의 결과다. 모든 redistribution physics의 결과로 일반화할 수 없다.

---

## 17. E13 native index 방향이 맞고 mirror는 더 악화한다

근거 산출물: `validation/emiss_e13/index_branch_audit.json`

- native frequency mean ratio `1.003594`
- mirror `0.99697`
- native/mirror local output 비교 포함.

판정: **SUPPORTED**

동일 행렬에서 파생한 방향성 검사이며 독립 물리 검증은 아니다.

---

## 18. T5의 최종 노선 판정은 `UNRESOLVED`

근거: `validation/emiss_t5/verdict.json`

- `route_disposition = "UNRESOLVED"`
- preregistered rank-1 proxy 일부는 통과.
- artifact/request optical metrics 및 B4 residual은 통과하지 못함.

판정: **SUPPORTED**

따라서 이를 상위 요약에서 “재분배 노선 폐쇄”로 바꾼 부분이 과장이다.

---

## 19. N4: 행렬 artifact가 iteration 11로 덮어써져 E12 세대 재현성이 손실됐다

근거:

- 현재 `fluor_matrix_iter10`: iteration 11, 468,330 edges, SHA `08ff…`
- E12 기록: iteration 10, 473,045 edges, SHA `2b65…`

판정: **SUPPORTED**

현재 파일명과 sidecar 일치만으로 과거 E12 입력 세대를 복원할 수 없다.

---

## 20. E11의 direct LINE accumulator는 기존 4억 cap의 영향을 받지 않는다

근거 정의:

- `R[j,i] = edge_output_energy / terminal_energy[i]`
- 기록 범위는 unique LINE interaction call.
- direct accumulator는 기존 event-log cap과 별도다.

판정: **SUPPORTED**

단, bf·k-packet-only 입력까지 포함한 전체 physical channel matrix라는 뜻은 아니다.

---

# UNVERIFIABLE

## 21. “올바른 물리가 이미 구현돼 있고 꺼져 있다”

> “올바른 물리가 **이미 구현돼 있고 꺼져 있음** … ⟹ **1년 노브 전멸의 이유는 물리 부재가 아니라 연산자 분할의 수치 실패.**”

출처: [UV_CENSUS_CONSOLIDATION.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:16)

근거:

현재 source에는 `LUMINA_CMFGEN_SRC_NLTE` default-OFF gate와 “correct physics”, “EXPLOSIVE”라는 주석이 존재한다.

판정: **UNVERIFIABLE**

이유:

gate와 과거 관찰을 기록한 주석은 확인되지만, 해당 source term이 CMFGEN과 동등한 “올바른 물리”인지 검증하는 ETAL/CHIL 또는 coupled oracle이 없다. 이것이 1년 실패의 유일 원인이라는 산출물도 없다.

허용되는 재서술:

> 현재 source에는 default-OFF인 population-ratio source 후보와 과거 explosive 동작을 기록한 주석이 있다. 후보의 정확성 및 장기 실패에 대한 인과성은 검증되지 않았다.

---

## 22. E12/E13의 원래 iteration-10 raw matrix를 현재 다시 재생할 수 있다

근거:

N4에서 파일이 iteration 11로 덮어써졌다. iteration-10의 derived 보고 수치는 남아 있으나 원래 raw artifact와 SHA가 현재 경로에 없다.

판정: **UNVERIFIABLE**

기존 derived 결과 자체를 거짓으로 판정할 근거는 없지만, 현재 동일 입력으로 재생하거나 source-to-report를 완전 검증할 수 없다.

---

## 23. “파이프라인 결정론”의 독립 경로 ≥2 검증

> “파이프라인 결정론 | 호스트·OMP 교차 0행 + 반복 0행 | 원장”

출처: [VERIFICATION_REGISTERS.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERIFICATION_REGISTERS.md:16)

근거 표기는 구체 artifact가 아니라 “원장”뿐이다. 같은 문서의 후속 기록에는 pipeline이 bit-reproducible하지 않다는 jitter 설명도 있다.

판정: **UNVERIFIABLE**

정확한 재서술:

> 일부 동일-binary 반복 시험에서는 0행 차이 또는 동일 SHA가 관찰됐다. 전체 호스트·OMP 파이프라인의 결정론을 입증하는 특정 독립 산출물은 현재 특정할 수 없다.

---

# 특별 감사

## 정확히 1.000 또는 0.000인 값

| 값 | 성격 | 판정 |
|---|---|---|
| N9 s≥5 B1–B4 `1.000000…` | mask 뒤 numerator와 denominator가 동일한 post-EPAY 배열 | 항등식, 폐기율 측정 아님 |
| N9 `S_rate_line/B` 오차 `2.22×10⁻¹⁶` | `eta=chi·B`, `S=eta/chi` | 대수 항등식 |
| T1 energy error `0.0` | 제거 energy를 명시적으로 같은 양 재주입 | construction/accounting 항등식 |
| E8 Pearson/Spearman `−1` | `gain=1/eps` 정의 | 정의상 항등식 |
| E11 identity fixture closure `0` | identity wiring fixture | 배선 항등식 |
| E12 destination-sum closure `0` 수준 | 정규화·장부 closure | emergent flux 측정 아님 |
| T2 `outside_BALL_rows=0` | writer가 애초 600–3000 Å만 기록 | window 항등/건전성 검사 |
| clamp/nonfinite/negative counters `0` | 실제 계수기 관측 | 항등식 아님 |
| 반복 SHA 동일 | 같은 입력·binary 반복 관측 | 국소 재현성 관측, 전역 결정론 아님 |

오늘 확인된 B1–B4 `1.0000000`은 요청에서 지적한 대로 “전 셀 mask 뒤의 항등식”이다.

## “N배” 값의 분모

| 주장값 | 분모 | 판정 |
|---|---|---|
| 5247.49× recycle gain | post-EPAY `S_fixed` | 정의된 source ratio. 독립 CMFGEN 측정 아님 |
| 필요한 5247.41× | 같은 `S_fixed`; 구성상 CMFGEN 항이 소거됨 | 독립 폐합 아님 |
| MC destruction 12.787× | deterministic `eps_eff_source` | 서로 다른 정의의 양; 물리적 destruction 확률비로 읽으면 과장 |
| B0 ×7.35 | frozen deterministic source의 B0 share 1.43% | rank-1 emission proxy, transported-J 측정 아님 |
| edge 5.1258× 증가 | 비무작위 4억 prefix의 92,287 edges | 전체-event 배율 아님 |
| N9 99.563% | post-EPAY fixed emissivity 전체 disposition | 폐기 전 `eta_line`이 분모가 아님 |
| T4 0.2353% | cap prefix의 event/energy 합 | full iteration 추정치 아님 |
| 438× fixed-source 부족 | CMFGEN 대비 정의된 post source ratio | source/field 비교는 측정됐지만 인과 배율은 아님 |

## 독립 경로처럼 보고됐으나 같은 생산자를 쓴 항목

- Stage31 `J_det` 대 `J_MC`: `J_MC`가 live MC oracle이 아니라 capture sidecar의 deterministic `J_producer`.
- E8의 “측정 이득”과 “필요 이득”: 둘 다 같은 `S_fixed`와 payload decomposition을 소비.
- N9 manifest/offline 일치: offline 분류가 같은 writer disposition을 재구성한 serialization-consistency 검사.
- T5 full-R와 rank-1 q: q가 바로 같은 R에서 파생된 compression 검사.
- E13 native/mirror: 같은 행렬의 두 판독.
- E10/E12는 instrumentation은 다르지만 같은 MC producer/deck에서 나온 행렬이다. 독립 물리 모델 두 개의 합치는 아니다.
- Codex/Fable의 상호 비열람은 해석자 독립성이지 산출물 생산자 독립성이 아니다.

## 4억 cap prefix 영향

직접 영향받은 주장:

- E8의 1,856,667 event pair와 same/different-line/bin 95.1164%.
- E9 prefix matrix의 305 input bins, 92,287 edges, energy closure.
- E10의 prefix-derived redistribution 결과.
- T4의 0.2353%, 0.2343%, B0 0.64%.

직접 영향받지 않은 항목:

- E8 payload의 `eta_fixed`, `eta_coherent`, `chi_total`, `J/CMFGEN` source 분해.
- 별도 thermal-destruction counter.
- E11/E12 direct LINE accumulator.

다만 E11/E12는 cap과 별개로 LINE interaction contract만 덮으며 전체 채널 완전성은 지지하지 않는다.

# 최종 노선 판정

현 증거는 Stage 3.2 ALI/MALI를 다음 구현·계측 우선순위로 삼기에 충분하다. 하지만 다음 네 문장은 현재 산출물 범위를 넘는다.

- “E8이 직접 원인을 완전히 폐합했다.”
- “T1이 모든 재분배 노선을 닫았다.”
- “ALI가 유일한 본진이다.”
- “N9가 η-only 시험의 inert fraction을 실측했다.”

산출물에 맞는 운전 문장은 다음이다.

> 현재 payload는 post-EPAY same-bin coherent source가 지배적임을 보이고, 시험된 uniform 및 LINE-matrix source-preinjection 대리모형은 UV 형상을 개선하지 못했다. 따라서 population-native χ/η와 transfer-consistent ALI/MALI는 우선 시험할 강한 후보다. 다만 직접 인과 폐합, 해법의 유일성, 전체 redistribution 노선의 폐쇄 및 η-only 시험의 전역 무효화는 아직 `UNRESOLVED`다.

파일 수정, 신규 모델/GPU 실행, 커밋은 수행하지 않았다.