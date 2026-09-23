# 사전등록 — DET-CMFGEN-FIXEDPOINT: CMFGEN 상태에 대한 Lumina 연산자 1회 적용 (고정점 시험)

**표적:** 같은 반복의 CMFGEN 상태(인구·T·n_e)를 Lumina 에 주입하고 Lumina 의 생산 bin 조립기·형식해를 한 번 적용해, CMFGEN 이 그 상태에서 낸 J_ν·Γ(S II)·x(S III/S II) 를 Lumina 연산자가 재현하는지 잰다.

상태 = **기안**. 실행 승인·구현 완료·판정이 아니다.
기안자 = **Fable fresh** (`claude-fable-5-1`, 개정25-k). 최종목적 = **MC 수송으로 CMFGEN 수준 SN Ia 스펙트럼 재현**.

## §0. 정정·선행 사실 (저자 실행 = 로그인 노드 비연산 조회)

1. **이 단은 연산자 비교이지 수렴 시험이 아니다.** 1주일간의 ∂J/∂N 사다리(DJDN-MVP→DRIVE-D→LOCAL-D→XSHELL)는 「Lumina 방정식의 참해 = CMFGEN 해, decoupled solve 가 못 간다」를 전제했고 시험한 적 없다. 반대 가설 「Lumina 방정식 자체가 다른 해를 갖는다」와 가르려면 **같은 상태에서 두 코드의 연산자 출력**을 비교하면 된다. 상태가 수렴해일 필요는 없다 — J_C = T_C(상태), Γ_C = R_C(상태, J_C) 가 **같은 반복에서** 나왔다는 쌍 정합만 필요하다(§4 G-DECK).
2. **CMFGEN 권위 판단(저자):** FIX_T 해제 conv 런(`toy06_19.48d_conv`)은 **it49 에서 발산, 수렴 EDDFACTOR 없음**(`R8_RESUME_NOTE.txt` 928–931행) ⟹ 권위 아님. FIX_T 자체런 `toy06_19.48d/` 는 텍스트 산출(RVTJ·S2OUT, 07-18 03:11 = it40)과 바이너리(SCRTEMP·EDDFACTOR, 14:47 = it65)가 **다른 반복**이다 ⟹ 쌍 정합 불가. **채택 = `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/`**(POINT1 IREC=66 NITSF=66; SCRTEMP·EDDFACTOR·RVTJ·S2PRRR·S2OUT·SIIIOUT 전부 07-18 21:37–38, it63–66 LAMBDA 4회, `run_jnu4.info`). 이 상태는 **미수렴**이다(`toy06_19.48d/OUTGEN` it60–63 「Maximum %」 1e5–1e7 @depth 26·48) — 형성셸 v<10,000 km/s 만 수렴 주장(R8 149행). 인구·T 는 참조해로 씨앗됐다(R8 379행 순환 caveat). FIX_T=T 는 Lumina `LUMINA_FIXED_TE_PROFILE` 레인의 쌍둥이다.
3. **율 쪽 반쪽 시험은 안 되어 있다.** 09-17 장부 §2 는 Lumina 자기 J 로 Lumina 이온비를 폐합한 것(0.5–3 %), §1 은 `data/cmfgen_jtable_toy06_19p48d.bin` 을 σ∝ν⁻³ 대역평균한 함의 장(S II 9–24 셸 자체런/CMFGEN 함의 3.9e3×)이다. **CMFGEN J 를 Lumina σ 표로 준위별 적분한 Γ 를 CMFGEN 자신의 Γ(`S2PRRR`)와 비교한 적은 없다.** 이 단이 그 반쪽(§2.4 R_Γ)을 오프라인으로 포함한다.
4. **Lumina 주입 경로:** T_e·n_e·T_rad 프로파일만 있고 이온분율·준위 인구 주입은 없다(브리프 확정 사실 1). bin 조립기(`cmfgen_assemble_impl`, 동결 트리 1942행~)는 `opac->tau_sobolev`·`line_source_S`(←`nlte_update_tau_sobolev`)·`bf->chi_bf`(←`compute_bf_opacity`)·`plasma->T_e` 만 소비하고 GPU fine-jbar(`jbar_line_det`)는 소비하지 않는다 ⟹ **CPU 단독 가능**. 생산 exports(`pathb_xshell_20260922/run2/input/resolved_lumina.exports`)는 `LUMINA_CMFGEN_SRC_NLTE`·`LUMINA_BF_NLTE_POPS` 미설정 ⟹ 조립기 선 소스 = B(T_e) 폴백(ε 두준위 `LINE_EPS_PHYS=1`), χ_bf 준위 가중 = 희석-Boltzmann. **주입 준위 인구는 χ_line(tau_sobolev)에만 들어간다** — 이것이 현행 Lumina 연산자이며 그대로 잰다.
5. **CPU 하네스 정정:** `cmfgen_xshell_production_selftest`(동결 트리 `lumina_cmfgen.c:8716`)는 합성 4셸·1준위 상태다. 실물 덱 CPU 경로 = `Makefile` `TARGET=lumina`(gcc) + `LUMINA_PURE_CMFGEN=1` → `lumina_main.c:402 cmfgen_run`. 이 단은 후자를 쓴다.
6. **목표 셸 referent (인용, 예측 아님):** sh24 = v 21,372–22,100 km/s(`geometry.csv`, 중앙 21,736). 공개 CMFGEN 19.48 d, 21,630 km/s 행: f_I/II/III/IV = 1e-99/0.5726/0.4145/0.01287 ⟹ **x_C,pub = 0.724**, T 9231.3 K(= 쌍둥이 sh24), n_e 1.360e7, n_atom 7.730e6. 자체런 RVTJ 깊이 d33 22,127 / d34 21,534 km/s 가 sh24 를 감싼다. Lumina base(391837 계보, run2 gen 133) sh24 S II: Γ 2.8245e-2, n_e 1.4914e7, n_eα_rr 3.819e-5, n_eα_dr 5.680e-5, C_ion 5.9e-15, C_rec 4.5e-16, x_map 297.4 / x_lag 133.6(XSHELL §0). ⟹ **설명할 간극 = x 185–411×**(브리프의 f_II 3.2–20× 는 다른 셸 구간의 값). 충돌 항은 Γ 의 1e-13 — 무시 가능(§2.4).
7. **부분 주입은 무의미하다.** 09-17 장부 §4 가 sh24 창 J 를 **외곽 셸(41–49) S V·Ca V 선 방출**에 귀속했다 ⟹ 목표 셸 S·Si 만 CMFGEN 으로 두고 나머지를 Lumina 로 두면 창 J 는 Lumina 상태가 만든다. **전 셸·전 종 주입**만 고정점 시험이다.

## §1. 물리 계약 1개

**계약 질문:** 「jnu4 IREC=66 의 CMFGEN 상태(전 셸 이온밀도·NLTE 준위 인구·n_e; T_e 는 쌍둥이 프로파일)를 Lumina 생산 상태에 주입하고, 생산 조립기(`nlte_update_tau_sobolev`→`compute_bf_opacity`→`cmfgen_assemble`)와 대각 ALI 형식해(`cmfgen_solve_J`)를 J_init = J_C 에서 수렴까지 적용했을 때, sh24 의 S II 광이온화 창 J·Γ(S II)·1스텝 x(S III/S II) 가 CMFGEN 의 J_C·Γ_C·x_C 를 τ=1.5 배 안에서 재현하는가?」

### 말하지 않는 것
- 「공개(수렴) CMFGEN 해가 Lumina 고정점이다」 — 시험 상태는 it66 미수렴 상태다.
- 「Lumina 반복이 그 고정점으로 수렴한다」(끌개·유일성).
- 「어느 코드가 옳다」. 「CMFGEN f_II 에 도달한다」. 「MC 스펙트럼」.
- 생산 8회 절단 ALI 의 충분성(별도 부산물 기재만).

### Q-GOAL — 기안자 자기답
결론 = **PASS(조건부)**. ① 어긋남 지도의 **중간 셸 S II/III 칸(sh9–24, x 10²×)** 을 직접 줄이지는 않는다 — 그 칸이 **솔버 칸인지 방정식 칸인지** 를 가른다. ② 산출은 물리 측정(두 연산자의 출력비 R_J·R_Γ·R_x, bin 별 J 비). ③ 더 가까운 대안 = XSHELL 후속 셸간 선형화 — 그러나 그 전제가 이 단의 피시험 명제다. ④ 건너뛰면 잘못된 전제 위에 N×N 결합을 지을 수 있다(사다리 4단 연속 「불충분」의 원인 후보).

### Q-LEAN — 기안자 자기답
결론 = **APPROPRIATE**. 물리 산출 5개: R_J(창 J 재현비), R_Γ(율 반쪽), R_x(1스텝 이온비), 창 bin 별 J_L/J_C 프로파일(FP-C 귀속용), ρ₁(J_C 에서 ALI 1스텝 잔차). 계측 3개: 덱 빌더(오프라인)·주입 훅(C)·판정 스크립트 — 각각 §2 산출에 소비. 게이트 4·음성대조 3(§4) — 각각 막는 실사고를 명시. 삭제한 것: 새 스펙트럼·전 셸 J CSV·GPU 런·NLTE solve·SRC_NLTE 대안 레인·준위 인구 전량 CSV 증거(해시만).

## §2. 기전·측정 정의

### 2.1 CMFGEN 상태 S_C (덱, 오프라인 `scripts/fixedpoint_build_deck.py`)
- 원천 = `toy06_19.48d_jnu4/`: SCRTEMP 레코드 **IREC−1 과 IREC**(둘 다 읽음), EDDFACTOR(J_ν, NCF=166,152 × ND=90), RVTJ(v·r·T·n_e·n_atom), S2PRRR(S II 55 초준위 광이온화율 × 90 깊이), XzVOUT(322 전준위 이탈계수, 판독기 검증용).
- **쌍 정합 [저자 판단·미확정]:** CMFGEN 반복은 인구→J(COMP_J_BLANK, `new_main/cmfgen_sub.f:1815/2298/3512`)→인구 갱신→SCR_RITE_V2 순이므로 EDDFACTOR 는 **갱신 전 인구(IREC−1)** 에서 계산됐다고 본다. 구현자가 반복 루프 안의 SCR_RITE_V2 호출 위치를 인용해 확정한다. 덱은 확정된 쪽을 쓰고, 두 레코드의 목표 깊이 x_C 차 δ_pair 를 기록한다(§4 G-REF).
- **준위 인구 = 선택 (i):** 초준위 인구 → 전준위: CMFGEN 자신의 정의(`F_TO_S` 매핑 + 초준위 내 Boltzmann, `SUP_TO_FULL_V4`, cmfgen_sub.f:1168)로 편다. Lumina 준위 목록은 같은 osc 파일(`active_ions.csv` osc_sha256)이므로 준위 대응은 에너지 일치(≤1e-4 상대)로 검증, 미대응 0 강제. (ii) 공개 이온분율+LTE 는 **Lumina 의 구성(S_l=B·LTE 준위)을 CMFGEN 이라 부르는 것**이라 기각 — S II Γ 의 준안정 준위 엣지(577/611 Å)를 오염시킨다. 단 §0-4 에 따라 현행 연산자에서 준위 인구는 χ_line 에만 들어가므로 (i) 의 비용은 판독기뿐이다.
- 격자: CMFGEN 90 깊이 → Lumina 50 셸, **속도 log-선형 보간**(인구·n_e·J bin 값). 이온 밀도는 보간 후 Σ_준위 = 이온 밀도로 재정규화하지 않는다(둘 다 보간, 정합은 G-DECK 이 잰다). T_e = 쌍둥이 프로파일(sh24 9231.3 K = 공개 행) 고정; 자체런 T 보간값과의 차를 기록(<1 % 기대치 아님, 기재).
- J_C(shell, bin): EDDFACTOR J_ν 를 Lumina 1234 log-bin(`lumina_frequency_grid.h`: 5.841e13–4.036e16 Hz)에 bin 평균 후 속도 보간. 판독기 = `validation/cmfgen_toy06_19p48d/analysis/extract_jnu.py` 의 `read_eddfactor` 재사용.
- Lumina 에 있고 CMFGEN 모델에 없는 (Z,stage) 는 주입하지 않고 목록·창 불투명도 기여를 기록한다.

### 2.2 연산자 적용 (C 훅, `cmfgen_run` pass 0 진입 직후, `LUMINA_FIXEDPOINT_DECK=<dir>`)
1. 덱을 읽어 `plasma->T_e`·`plasma->n_electron`·`atom->ion_number_density`·`nlte->nlte_level_populations` 를 덮어쓴다(차원 불일치·결측 셸·비유한 값 → fail-closed rc≠0). 덮어쓴 배열의 SHA-256 을 증거에 남긴다.
2. 생산 생산자 순서 그대로: `nlte_update_tau_sobolev` → `compute_bf_opacity` → `cmfgen_assemble`. 생산 env(base exports)를 그대로 쓴다.
3. `cs.J ← J_C`, `cmfgen_solve_J(…, n_ali=1)` 반복 k=1,2,… 까지 max|ΔJ/J|<1e-4(전 셸·전 bin, J>0) 또는 k=200(FAIL). k=1·8·k_conv 에서 창 bin(§2.3 W)×50 셸을 저장. **수렴해를 채택하는 이유:** 고정점 문제는 Lumina 방정식의 해를 묻는 것이고 8회 절단은 그 근사다; J_init=J_C 는 「J_C 가 고정점이면 머문다」를 ρ₁ 로 곧바로 보여 준다.
4. `xshell_canonicalize` 로 canonical 뷰 → `parity_gamma_phot_explicit_field`(동결 트리 `lumina.h:1594`)로 Γ_L(J_C)·Γ_L(J_L^{(1)})·Γ_L(J_L^{(8)})·Γ_L(J_L^{conv}) — sh24 와 전 50 셸, Z=16 stage 1(주)·Z=14 stage 1(부).
5. `cmfgen_dump_frozen_chieta` 로 조립 χ_tot·χ_es·S_fixed 를 남기고 **fine-jbar·commit·R7 전에 return 0**(`LUMINA_VALIDATION_EXIT_AFTER_R6` 패턴). 생산 배열·commit 경로·금지구간 `lumina_cmfgen.c:2848-3212` 무접촉.

### 2.3 창 W
S II 엣지: 4S° 531 Å · 2D° 577 Å · 2P° 611 Å(`S2_F_OSCDAT` E 로 재계산). W = [ν_edge(2P°), 3·ν_edge(4S°)] = 177–611 Å(≈234 bin; 531–611 Å 는 ≈26 bin). 표에는 σ_ground 가중 창 평균과 bin 별 프로파일을 둔다.

### 2.4 측정량 (sh24 주, 전 셸 부산물)
- **R_J = Γ_L(J_L^{conv}) / Γ_L(J_C)** — 같은 σ·같은 준위 가중으로 J 만 바뀐 비 = σ-가중 창 J 재현비.
- **R_Γ = Γ_L(J_C) / Γ_C,eff**, Γ_C,eff = Σ_l n_l Γ_l / N(S II) (S2PRRR 준위별 율 × SCRTEMP 초준위 인구, 목표 깊이 d33/d34 보간). 보조로 바닥 준위 비 Γ_L,ground(J_C)/Γ_C,ground(PRRR 1행). Γ_L 은 09-17 `rebalance_check.py` 계열(σ 표 `cmfgen_sigma_bf.bin`·Lumina 준위 가중)로 오프라인 재적분하며, 그 도구는 Lumina 생산 Γ 를 0.5–3 %(Si II 5–9 %) 로 재현한다(장부 §2).
- **R_x = x_1 / x_C**, x_1 = (Γ_L(J_L^{conv}) + C_ion) / (n_e,C·α_L + C_rec), α_L = (n_eα_rr+n_eα_dr)_base/n_e,base (T 동일 ⟹ α 동일), C 항은 base 값을 n_e 비로 스케일(§0-6: Γ 의 1e-13, 결과에 무영향). x_C = n(S III)/n(S II) 덱(IREC 채택 레코드) sh24 보간값.
- **ρ₁ = max_{b∈W} |J_L^{(1)}−J_C|/J_C @sh24**. 부산물: R_J^{(8)} (절단 생산 ALI 의 값).
- 유한 교란 비이므로 clamp·부호 교정·0–1 제한 없음.

## §3. 변경집합과 산출물

### 변경집합 — DET-CMFGEN-FIXEDPOINT (표)

| 경로 | 분류 | 허용 변경 |
|---|---|---|
| `src/env_universe.h` | 계측 게이트 | `LUMINA_FIXEDPOINT_DECK` 를 strict env universe 에 등록 |
| `src/lumina_cmfgen.c` | 주입·측정 훅 | `cmfgen_run` pass 0 진입부의 static 훅 `fixedpoint_load_deck`·`fixedpoint_inject_state`·`fixedpoint_measure`·`fixedpoint_write_evidence`; 기본 OFF, OFF 시 byte-동일; 생산 solve·commit·금지구간 무접촉 |
| `scripts/fixedpoint_build_deck.py` | 오프라인 덱 빌더 | 신규: SCRTEMP/EDDFACTOR/RVTJ/S2PRRR/XzVOUT 판독·검증·보간·덱 기록·referent JSON |
| `scripts/fixedpoint_judge.py` | 판정 스크립트 | 신규: R_J·R_Γ·R_x·ρ₁·ε_C·게이트·분기·NC 판정 |

### 변경집합 끝

`expected_extra` 는 위 4파일과 정확히 같다. 변경 금지: `Makefile`(CPU `lumina` 타깃 그대로 사용)·`src/lumina.h`·`src/lumina_cmfgen.h`·`tests/*`·`src/lumina_cuda.cu`·`formal_solve_bin`·대각 ALI 갱신식·`cmfgen_commit_jnu`·인구/n_e/준위 solve·χ/η 조립식·금지구간 `src/lumina_cmfgen.c:2848-3212`(sha `8d3e3d76…`). 소비만 하는 기존 함수: `nlte_update_tau_sobolev`·`compute_bf_opacity`·`cmfgen_assemble`·`cmfgen_solve_J`·`xshell_canonicalize`·`parity_gamma_phot_explicit_field`·`cmfgen_dump_frozen_chieta`.

### 산출물
- 덱: `fp_state.csv`(shell,Z,stage,n_ion) · `fp_levels.csv`(shell,Z,stage,level_idx,n_level) · `fp_thermo.csv`(shell,T_e,n_e) · `fp_jnu.bin`(50×1234 binary64) · `fp_referent.json`(IREC 채택·δ_pair·d33/d34·x_C·Γ_C·판독기 검증치·해시).
- 훅: `fp_fields.bin`(J_L k=1·8·conv 창 bin×50 셸, Γ 배열) · `fp_result.json`(해시·k_conv·Γ·유한성·게이트) · 기존 frozen chieta dump.
- 판정: `fp_verdict.json`. 전 bin J CSV·스펙트럼·준위 인구 전량 증거는 만들지 않는다.

## §4. 게이트와 음성대조

허용오차: 배열·해시 = bit-동일; 산술 항등 = 상대 `1e-6`; 물리 재현 = **τ = 1.5** (§5 근거).

### G-DECK — 덱 무결(판독기·쌍 정합)
- (a) 원소 총량: Σ_q n(Z,q) = A_Z·n_atom(RVTJ) 상대 ≤1e-3, Z∈{14,16,20}, 전 깊이. (b) 전하 폐합: Σ q·n = n_e(SCRTEMP) 상대 ≤1e-2(모델 종 전부). (c) SCRTEMP(IREC) 전준위 인구 대 XzVOUT 이탈계수 재구성 ≤1e-6 — 판독기·DC 규약(`writedc_v3.f:96` log10 여부 [미확정]) 동시 확정. (d) EDDFACTOR: 외곽 깊이 ∫J dν/(σT⁴/π) 가 jnu4 기록 0.945 를 1e-3 로 재현. (e) 준위 대응 미대응 0. (f) 쌍 정합 인용(cmfgen_sub.f 행) 기재.
- 막는 실사고: 08-28 표 「이름을 측정 대신 씀」 — 잘못된 레코드·잘못된 DC 규약을 CMFGEN 이라 부르는 것.

### G-INJ — 주입 재현
- 훅이 덮어쓴 4배열의 SHA-256 = 덱 파일에서 오프라인 계산한 SHA-256(bit). 결측·NaN·차원 불일치 시 rc≠0.
- OFF(`LUMINA_FIXEDPOINT_DECK` 미설정) 실행이 base 와 stdout/stderr/산출 byte-동일.

### G-REF — referent 불확도
- ε_C = max(|ln x_C^{IREC}/x_C^{IREC−1}|@sh24, |ln x_C(d33)/x_C(d34)|) 를 원시값으로 기록. **ε_C ≤ ln(τ)/2 = 0.203** 이어야 FP 분기를 읽는다(초과 = FP-X: referent 가 τ 를 못 받친다). Γ_C,ground(d33·d34) 유한(S2PRRR 하위 행 NaN 실재 — 바닥 행은 유한 확인).

### G-FIN — 산출 유한·수렴
- J_L·Γ·R 전부 유한; k_conv < 200; 창 bin 중 J_C≤0 인 bin 0.

### 음성대조

| NC | 주입 | 반드시 FAIL |
|---|---|---|
| NC-OFF | 게이트 미설정 | 신규 산출 0·base byte-동일(위반 시 G-INJ) |
| NC-DECK | 덱 빌더에 A_S ×1.01 변조 | G-DECK (a) |
| NC-PAIR | XzVOUT 한 준위 DC 변조 | G-DECK (c) |
| NC-SHELL | `fp_state.csv` 에서 셸 한 행 삭제 | 훅 rc≠0 (fail-closed) |

## §5. 기대치와 분기

### 허용오차 τ = 1.5 의 근거
간극 = sh24 x 185–411×(§0-6), 브리프 최소 3.2×. referent 잡음 = LAMBDA 반복 쌍 차 δ_pair·속도 보간·Γ 도구 폐합 0.5–9 % — 합쳐 ≲20 % 로 예상되나 **G-REF 가 실측**한다. τ=1.5 는 최소 간극(3.2)의 절반 이하이면서 잡음 상한(G-REF 0.203 = 1.23×)의 배 이상이다. 「구성상 참」 등록 없음: R_J·R_Γ·R_x·ρ₁·k_conv·bin 지배 위치 전부 미지.

### 런 없이 얻을 수 없는 수
R_J, R_Γ, R_x, ρ₁, k_conv, R_J^{(8)}, ε_C, δ_pair, 창 어느 bin·어느 셸이 R_J 편차를 지배하는지. 알려진 값(§0-6)은 referent 이지 예상 결과가 아니다.

### 분기 표 (배타; 게이트 FAIL 하나라도 있으면 FP-X)

| 분기 | 판정식 | 뜻 | 다음 결정 |
|---|---|---|---|
| **FP-A** | 게이트 PASS ∧ R_J∈[1/τ,τ] ∧ R_x∈[1/τ,τ] | CMFGEN it66 상태가 Lumina 연산자의 고정점(창·율·이온비) | 사다리 전제 지지. 다음 = 끌개 시험(주입 상태에서 생산 반복 1회 후 이탈량) 별도 계약 |
| **FP-B** | 게이트 PASS ∧ R_J∈[1/τ,τ] ∧ R_x∉ | 전달은 같고 율 쪽(σ 가중·α·준위 가중)이 다름 | R_Γ·바닥준위 비로 σ/가중/α 중 하나를 지목 → 원자자료·율 계약. 사다리 정지 |
| **FP-C** | 게이트 PASS ∧ R_J∉[1/τ,τ] | 전달·불투명도·방출률 쪽 | chieta dump + 09-17 `window_leak/ion_share` 도구로 지배 bin·셸·종 기록(S_l=B 폴백·χ_bf 준위 가중·내경계 3,900 km/s 가 1차 용의). 사다리 정지, 지목 요소 단일 수리 계약 |
| **FP-X** | 게이트 FAIL 또는 ε_C>0.203 또는 필수값 부재 | 미시연 | 계측·referent 만 수리, 같은 계약 재현 |

FP-B·C 안에서 「ρ₁ 작고 R_J 큼(고정점이나 ALI 가 떠남)」 등 세부는 판정문에 수치로 기재하되 새 분기를 만들지 않는다.

## §6. 결정·자원·보존
- **GPU 0회.** 덱 빌더·판정 = grammar-debug CPU(분 단위); 훅 = CPU `lumina` 바이너리(`LUMINA_PURE_CMFGEN=1 LUMINA_PURE_CMFGEN_ITER=1`, base exports + `LUMINA_FIXEDPOINT_DECK`), 예상 수십 분 이내(ALI ≤200 × 50셸×1234bin×58 ray). 초과 시 lageunha(OMP=60)로 폴백, 근거 기재.
- 대상 트리 = 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source`(XSHELL 변경집합 포함본). 새 run root `/gpfs/kjhan/lumina/cmfgen_map/fixedpoint_20260924/run1/`; 실패 시 재사용 금지. 봉인 입력·jnu4 는 읽기만.
- FP-A 도 구현 승인이 아니다. 후속 표적은 user 결정.

## §7. 분장 장부 (개정25-n)

| 단계 | 규약상 담당 | 실제·상태 |
|---|---|---|
| 기획·표적 선정 | 운전자(Opus 5.5) | 완료 — user 2026-09-24 승인 |
| 계약 기안 | Fable fresh | **완료** — `claude-fable-5-1`, 이 본문 |
| 기계 프리플라이트 | astra | 미집행 |
| 물리·대수 사전감사 | astra | 미집행 |
| Q-GOAL·Q-LEAN 경량 사전감사 | grok | 미집행 |
| 구현·직접 selftest | Opus 5.5(중첩 CLI) | 미집행 |
| 코드 감사 | Fable fresh | 미집행 (⚠자기 계약 편향 기재) |
| 재감사 | astra | 미집행 |
| 발주·집행·판독·판정 | 운전자 | 미집행 |
| 종료감사 | Fable + astra | 미집행 |

## §8. 주장 상한
최대치: 「jnu4 IREC=66 CMFGEN 상태를 주입한 Lumina 생산 조립기·수렴 대각 ALI 는 sh24 S II 창의 σ-가중 J 를 R_J 배, Γ 를 R_Γ 배(율 반쪽), 1스텝 이온비를 R_x 배로 냈다; J_C 에서의 1스텝 잔차는 ρ₁ 이었다.」 FP-A 면 「이 상태에서 두 연산자가 τ 안에서 일치한다」까지. 확장하지 않는 것: 공개 수렴해에 대한 같은 결론(상태가 다름·미수렴), Lumina 반복의 끌개·유일성, 절단 8회 ALI 의 충분성, 어느 코드의 정당성, 준위 인구 주입의 효과(현행 연산자에서 χ_line 에만 들어감), CMFGEN f_II 도달, MC 스펙트럼.

## §9. 집행 기록 뼈대
1. **기안** — `claude-fable-5-1`, 2026-09-24, read-only 본문 반환, 파일화 미실행.
2. **저자 확인** — jnu4 스냅샷 시각·IREC, conv 런 발산(R8 928–931), 조립기 fine-jbar 무소비, CPU 타깃, 공개 행 referent, base sh24 채널 행: 완료. **[미확정 → 구현자]** 쌍 정합 순서(cmfgen_sub.f), XzVOUT DC 규약(writedc_v3 log10 여부).
3. **기계 프리플라이트** — `check_prereg_preflight`·`check_gate_referents`·`check_doc_canonical`: **집행 2026-09-23 23:40, 전건 rc=0**(운전자, grammar-debug). 참조가 repo(docs·validation·CLAUDE.md)와 동결 트리(src·scripts)에 나뉘어 단일 `--root` 로는 각각 한쪽이 FAIL(repo root: `xshell_canonicalize` 부재 / 동결 root: docs 부재) ⟹ 실제 배치를 그대로 잇는 합성 root `/gpfs/kjhan/lumina/cmfgen_map/fixedpoint_20260924/preflight_root`(docs·CLAUDE.md·validation→repo, src·scripts→동결 트리 symlink)로 실행: changeset files=4·branch-partition residual=FP-X·references refs=9·GATE-REF·DOC-CANON PASS. ⚠`--root` 는 동결 트리(`parity_gamma_phot_explicit_field`·`xshell_canonicalize` 는 repo 가 아니라 거기 있다).
4. **사전감사(astra·grok)** — 모델 ID / 폐합 차단 / 처분:
5. **구현·직접시험** — 모델 ID / 변경집합 실제 / 외 접촉 / NC 4종 시연 / OFF byte 동일:
6. **코드 감사(Fable)·재감사(astra)** — 첫 판정 봉인 접촉 가부 / 금지구간 불변 / 생산 경로 무접촉 / 차단·수리:
7. **덱 빌드** — jnu4 해시 / IREC 채택 / G-DECK (a)–(f) 원시값 / ε_C·δ_pair:
8. **훅 실행** — 노드 / 바이너리 SHA / exports 델타 / k_conv / 경과:
9. **판독** — G-DECK / G-INJ / G-REF / G-FIN / R_J·R_Γ·R_x·ρ₁·R_J^{(8)} / 지배 bin·셸·종 / 분기:
10. **판정·종료감사** — 운전자 판정 / 주장 상한 준수 / 종료감사 모델 ID·재현 / 최종 처분:

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-CMFGEN-FIXEDPOINT (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:src/[A-Za-z0-9_/.]+\\.(?:c|h)|scripts/[A-Za-z0-9_/.]+\\.py)",
    "symbol": "LUMINA_FIXEDPOINT_DECK",
    "roots": ["src", "scripts"],
    "expected_extra": [
      "src/env_universe.h",
      "src/lumina_cmfgen.c",
      "scripts/fixedpoint_build_deck.py",
      "scripts/fixedpoint_judge.py"
    ]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    "metrics": {
      "count": "len(v)",
      "gate_fail": "sum(1 for x in v if x < 0.5)",
      "fixedpoint": "sum(1 for x in v if 0.5 <= x < 1.5)",
      "rate_side": "sum(1 for x in v if 1.5 <= x < 2.5)",
      "transport_side": "sum(1 for x in v if x >= 2.5)"
    },
    "rules": [
      {"name": "FP-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "FP-A", "predicate": "count > 0 and gate_fail == 0 and fixedpoint == count"},
      {"name": "FP-B", "predicate": "count > 0 and gate_fail == 0 and rate_side == count"},
      {"name": "FP-C", "predicate": "count > 0 and gate_fail == 0 and transport_side == count"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 1]]},
      {"name": "fixedpoint", "mix": [[1.0, 1]]},
      {"name": "rate-side", "mix": [[2.0, 1]]},
      {"name": "transport-side", "mix": [[3.0, 1]]}
    ],
    "residual": "FP-X"
  },
  "references": [
    {"path": "docs/PREREG_DET_PATHB_XSHELL_2026-09-22.md"},
    {"path": "docs/READOUT_IMPLIED_J_LEDGER_2026-09-17.md"},
    {"path": "docs/READOUT_S_IONIZATION_EUV_LTE_2026-09-17.md"},
    {"path": "docs/VERDICT_DET_PATHB_LOCAL_2026-09-22.md"},
    {"path": "CLAUDE.md"},
    {"path": "validation/cmfgen_toy06_19p48d/analysis/extract_jnu.py"},
    {
      "path": "src/lumina_cmfgen.c",
      "flags_existing": ["cmfgen_run", "cmfgen_assemble", "cmfgen_solve_J", "cmfgen_dump_frozen_chieta", "xshell_canonicalize"],
      "flags_planned": ["LUMINA_FIXEDPOINT_DECK", "fixedpoint_load_deck", "fixedpoint_inject_state", "fixedpoint_measure", "fixedpoint_write_evidence"]
    },
    {
      "path": "src/lumina.h",
      "flags_existing": ["nlte_update_tau_sobolev", "compute_bf_opacity", "parity_gamma_phot_explicit_field"]
    },
    {
      "path": "src/env_universe.h",
      "flags_existing": ["LUMINA_FIXED_TE_PROFILE", "LUMINA_CMFGEN_ALI_ITER", "LUMINA_CMFGEN_SRC_NLTE"],
      "flags_planned": ["LUMINA_FIXEDPOINT_DECK"]
    }
  ]
}
```
