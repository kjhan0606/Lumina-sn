# 사전등록 — DET-BB-ALI: 생산 bb 율에 CMFGEN Sobolev 순 복사 괄호(β·J_cont, β·A)를 걸어 FUV Λ-정체가 S II 과이온화의 원천인지 1런으로 가른다 (2026-09-25)

**표적:** 398939 뜨거운 base 와 같은 봉인 입력·솔버·반복 수에서, 생산 준위 행렬의 bound-bound 율만 「자기항 포함 J̄」(R = B·J̄, A) 에서 「CMFGEN 비겹침 Sobolev 순 복사 괄호」(R = B·β·J_cont, A·β) 로 바꾸고(env 1키 신설 + 기존 원장 캡처 1키 ON), 판정 코어 셸의 S II 광이온화율 Γ 와 S III/S II 가 제어(398939 산출) 대비 등록 문턱 이상으로 어느 방향으로 움직이는가로 「두꺼운 FUV 선 소스의 Λ-정체가 FUV 장(→ 과이온화)의 원천」을 확증(ALI-SOURCE)·부분(ALI-PARTIAL)·기각(ALI-NULL)·역행(ALI-ADVERSE) 한다. 상태 = **기안**(실행 승인·구현·판정 아님). 기안자 = Fable fresh(`claude-fable-5-1`, 개정25-b; grok 대행 초안 `bb_ali_20260925/draft_raw.md` 를 개정 — 집행 기록 1–3). user 결정(2026-09-25) = 「실험으로 확증」(2단계 판독 §7); 순서 1 원인 분리(확증) → **2 EUV/UV 원천(이 단)** → 3 NT 생산 반영. 최종목적 = MC 수송으로 CMFGEN 수준 SN Ia 스펙트럼 재현. 잣대 = **공개 파일 + 398939 기존 산출**(자체 CMFGEN 런 인용 금지).

## §0. 선행 사실 (저자 실행 = 로그인 노드 비연산 조회; 값은 referent, 예측 아님)

1. **1단계 확증** (`docs/READOUT_CAUSE1_SII_IONIZATION_CHANNEL_2026-09-25.md`, astra 3회차 확증 가): sh10–24 차가운 창에서 S II→III 광이온화 직접 율 기여의 69–82 % 가 명시 들뜬 준위(E ≥ 4 eV, 준위 < 100) 경로, 그 광자의 79–91 % 가 λ < 2000 Å(sh24 는 < 912 Å 최대). 단계 균형 Γ = Σ_l f_l^LTE(T_e)·Γ_l(J)(`lumina_plasma.c`:3141–3250) 가 `lumina_ion_channels.csv` gamma_phot 와 0.2–0.8 % 일치. x_phot = 0.40·w·x_Saha(DR = 분모 60 %). sh30 은 바닥 채널만으로도 과이온화(코어 제외).
2. **2단계 판독** (`docs/READOUT_CAUSE2_FUV_LINE_LAMBDA_STALL_2026-09-25.md`): 생산 선 J̄ = β·J_cont + (1−β)·S_old 의 Λ-반복 구조는 확인(§6 astra), 원천 연결은 오프라인 확증 불가; §7 sh15 Γ-가중 국소 방출 귀속 = S II 선 44 %·S III 14 %·Si II 9 %·Si III 5 %·bf+ff 15 %·잔여 13 %, τ > 100 두꺼운 선 29 %. 398939 최종 인구의 mapped FUV 선(900–2000 Å): sh24 불투명도 가중 S_l/B = 1.82(S III 2.91·S II 2.04), β < 10⁻² 몫 39 %; sh15 1.55, 54 %(`coldstart_20260924/cause2/fuv_lines_beta.txt`). user 결정 = bb 율 표준 가속 1런.
3. **생산 bb 율(동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src`, 저자 실행 sed)**: `lumina_plasma.c`:19089–19110 「Production split」 — JEQB 아니면 `nlte_bb_rate_graph_contains` → `nlte_bb_jbar_canonical`(정의 :585 부근; `line_jbar_lookup` 으로 checked view 의 `jbar` 만 반환, VALID/EXACT_ZERO 일 때 rc 1, 그 밖 0 이고 `Jbar_view` 0) → `R_absorb = B_lu·Jbar_view`, `R_stim = B_ul·Jbar_view`, `R_spont = A_ul`, `jd_beta = 1.0`. 그 위 :19003–19087 은 `A2_06_DIAGNOSTIC_SHADOW`(dilute·mode 3 β·J_inc·mode 2·mode 1, `mali_on`) 로 **전부 덮어써진다**. 이 함수는 `nlte_assemble_rate_matrix_impl`(:18656, 셸·이온쌍 단위; n_e 폐합 sweep 마다 호출 — 398939 는 세대당 21–27 회) 이고, `opacity`(OpacityState)·`nlte->current_iter`·`opacity->jbar_line_det_operator` 에 접근할 수 있다.
4. **plasma 가 받는 뷰에는 β·J_cont 가 없다**: `LineJbarValue`(`radiation_field.h`:427–433) = {jbar, validity, count, se, statistic_kind}. 뷰의 jbar 는 `cmfgen_commit_jnu`(`lumina_cmfgen.c`:3992–)가 `opac->jbar_line_det[lid*NS+s]` 를 **그대로 복사**한 값(:4212 `raw` → `line_jbar[dst] = raw`; −1 → UNSAMPLED/0). ⟹ grok 초안 §3 「`nlte_bb_jbar_canonical` 이 기존 생산자 값을 노출」은 plasma.c·env_universe.h 만으로 **불성립**(운전자 발주 전 확인, 집행 기록 2).
5. **β·J_cont 의 유일한 생산 지점 = fine 조립기** `lumina_cmfgen.c`:7078–7118(`cmfgen_fine_jbar`, Sobolev 연산자일 때): `line_net_sobolev_material(n_upper, A_ul, ν, τ, t_exp, SRCE_CHK, 1)` → `line_net_sobolev_radiation(material, J_cont=line_value[s], …)` → `opac->jbar_line_det[cell] = radiation.jbar`(cell = l·NS + s; `l` 은 `opac->line_list_nu[l]`·`g_fine_atom->line_A_ul[l]` 을 함께 쓰는 **원자 선 인덱스** — plasma 의 `line` 과 같은 번호 공간; plasma :18962 도 `jbar_line_det[line*n_shells+shell]` 로 읽는다). `line_net_rate.c`:296–340: `continuum_term = β·J_cont`, `local_emission_term = η·(c t/ν)·(1−β)/τ_eff`(= (1−β)S), `jbar = continuum_term + local_emission_term`(같은 순서의 double 덧셈 1회). β = `line_net_cmfgen_exponx(τ_eff)`(:264–, CMFGEN `subs/exponx.f` 전사: |τ| < 10⁻³ 다항식, τ < 40 (1−e^{−τ})/τ, 그 밖 1/τ; value ≤ 0·비유한이면 rc −1). τ_eff = τ_raw, 단 τ_raw < −0.5 이면 SRCE_CHK 양수 치환(:250–259) ⟹ **τ_eff ≥ −0.5 · β ∈ (0, 1.297]** — β > 1 은 정상 경로(398939 stderr `[cmf_fine][SOBOLEV-LINE-OPERATOR]` 6회 전부 `raw_negative`=11,034–26,764 셀·`srce_chk_applied=0`·`beta_max`=1.0007–1.0050).
6. **기존 read-only 원장(캡처)**: `LUMINA_A210_SPRODUCER_CAPTURE`(`env_universe.h`:25; 파싱 :5527–5536 값 0/1 아니면 BLOCKED; :5595 Sobolev 연산자일 때만) 가 1 이면 같은 셀에 `line_producer_continuum_term`(= β·J_cont as produced)·`line_producer_local_emission_term`·`line_producer_eta`·`line_producer_tau_eff`·`line_producer_provenance` 를 쓴다(:7111–7118; 할당 :6987–7030 NL·NS 5배열, 진입 시 free/NULL :5498–5505, 종료 시 `line_producer_terms_captured = 1`·`_n_shells = NS` :7343–7345 + stderr `[cmf_fine][SPRODUCER-CAPTURE] status=CAPTURED … identity=jbar==continuum_term+local_emission_term`; 센티널 double −1·provenance 0xff, `lumina.h`:290–302 「Never consumed by the NLTE/rate publisher」). plasma.c:16815–16850 의 진단 소비자(A210 saturation diag)는 stride/포인터를 검사하고 미일치면 센티널을 유지한다(행렬 아님). 398939 는 이 키 = 0(`input/resolved_lumina.exports`). **크기**: 398939 `jbar_cells`=109,014,300(= 2,180,286 선 × 50 셸) ⟹ 4 double + 1 byte 배열 = **3.60 GB** 호스트 메모리; 398939 MaxRSS = 32,365,304 KB(sacct .batch), `--mem=64G`.
7. **base = 398939**(`/gpfs/kjhan/lumina/cmfgen_map/pathb_xshell_20260922/run2`, DET-PATHB-XSHELL 판정런; `docs/VERDICT_DET_PATHB_XSHELL_2026-09-24.md`): exports 델타 대 391837 = `+LUMINA_XSHELL_RESPONSE=1` 뿐; `LUMINA_PURE_CMFGEN_ITER=6`·`LUMINA_CMFGEN_ALI_ITER=8`·`LUMINA_CMF_LINERES_JBAR=2`·`LUMINA_ION_CHANNEL_DUMP=1`·`LUMINA_POP_DJDN=1`·`LUMINA_POP_DJDN_CHI=1`·`LUMINA_A210_LINE_SATURATION_DIAG=2`·`LUMINA_NLTE_LEVEL_DUMP=1`·`LUMINA_FIXED_TE_PROFILE`(T_e 고정)·`LUMINA_ENV_STRICT=1`; `LUMINA_MALI`·`LUMINA_NLTE_JEQB`·`LUMINA_ION_NT`·`LUMINA_INIT_IONFRAC_DIR` 부재. partition a100(syn101) 2 GPU, 47,833 s, `binary_sha256` 35c6dbc2…, terminal SUCCESS. **반복 구조**(`lumina_cmfgen.c`:8862–8871·9032–9049; stderr `[A2-07][NE-CLOSURE]`): pass 0 = init(`INIT_SHARED_GAUSSIAN` 연산자, `current_iter`=0, SEED 커밋 gen 2, sweep 26회) → 논리 반복 k = 0..5(Sobolev 연산자, 복사 gen k+2, R7 인구 커밋 gen k+3, sweep 27·25·23·22·23·21) ⟹ 채널 덤프 `gen` 1–167 = 누적 sweep(Σ = 167), 반복 k 의 마지막 sweep = 26·53·78·101·123·146·167(cause1 `rates_by_iter.py` 사상). `physics_DET_iter000k.{ion,shell,spectral}.csv` 는 R7 뒤 덤프(k = 0..5, spectral 74–51,323 Å). 준위 덤프 `nlte_levels_final.csv` 는 최종(gen 8)만. DET-COLDSTART-NT run2(`docs/VERDICT_DET_COLDSTART_NT_2026-09-25.md`, DRIFT)는 차가운 출발 + NT 상한이 겹친 런이라 **이 단의 제어가 아니다**.
8. **제어 referent(398939, 저자 실행 awk/python, 코어 12셸)** — x = `Z16_stage_q2_cm3/Z16_stage_q1_cm3`(ion.csv), Γ = S II `gamma_phot`(채널 덤프, 반복 마지막 sweep), S_band = Ση_true Δν / Σ(χ_bb+χ_bf+χ_ff)B_ν(T_e)Δν(100–2000 Å, spectral+shell):

| 셸 | x k=5 | x_pub | gap dex | Δlog x (k3→5) | Γ k=5 | Δlog Γ (k3→5) | S_band k=5 | Δlog S (k3→5) |
|---|---|---|---|---|---|---|---|---|
| 10 | 15.0 | 0.241 | 1.79 | +0.01 | 0.0465 | −0.00 | 0.176 | −0.07 |
| 11 | 20.3 | 0.181 | 2.05 | −0.06 | 0.0443 | −0.00 | 0.145 | −0.04 |
| 12 | 5.22 | 0.186 | 1.45 | −0.01 | 0.0261 | −0.01 | 0.823 | −0.00 |
| 13 | 3.67 | 0.197 | 1.27 | −0.02 | 0.0144 | −0.02 | 0.911 | −0.01 |
| 14 | 2.91 | 0.210 | 1.14 | −0.03 | 0.00897 | −0.03 | 1.003 | −0.01 |
| 15 | 2.25 | 0.233 | 0.98 | −0.11 | 0.00547 | −0.07 | 1.126 | −0.01 |
| 19 | 16.3 | 0.349 | 1.67 | −0.13 | 0.00204 | −0.13 | 1.622 | −0.02 |
| 20 | 21.0 | 0.392 | 1.73 | −0.14 | 0.00205 | −0.14 | 1.658 | −0.02 |
| 21 | 40.8 | 0.490 | 1.92 | −0.08 | 0.00307 | −0.11 | 1.415 | −0.02 |
| 22 | 34.0 | 0.572 | 1.77 | +0.01 | 0.00489 | −0.06 | 1.300 | −0.03 |
| 23 | 45.7 | 0.696 | 1.82 | +0.05 | 0.0151 | +0.05 | 1.097 | −0.03 |
| 24 | 150 | 0.744 | 2.30 | +0.05 | 0.0387 | +0.05 | 1.102 | −0.03 |

⟹ 제어 자체의 2반복(k 3→5) 표류: |Δlog x| 최대 0.14·중앙값 0.05, |Δlog Γ| 최대 0.14·중앙값 0.06, |Δlog S| 최대 0.07·중앙값 0.02. 단일 셸의 1반복 점프는 그보다 크다(sh24 Γ k 2→3 +0.49 dex, x +0.30 dex) — 판정 통계를 **코어 12셸 중앙값 · k=4,5 기하평균**으로 두는 근거. x_ctrl 는 채널 덤프의 r(Γ/D; sh24 k=5 408)과 다르다(커밋 단계비 150) — δΓ 와 δx 는 서로 대체되지 않는다. x_pub = `coldstart_20260924/coldstart_referents.csv`(sha 5910685e…, 공개 toy06 19.48 d 분율 v_mid 보간). T_e 고정·n_e 는 k 3→5 에 3자리 불변.
9. **현 동결 트리** 는 DET-COLDSTART(-NT) 변경집합(pin·재규격·NT 경로, env 부재 시 armed 0)을 포함하므로 398939 바이너리와 byte-동일이 아니다. 이 계약의 OFF 경로 동일성은 **코드 diff**(A2-06 세 대입 무변경)로 검수하고 OFF GPU 재런은 하지 않는다(기재).

## §1. 물리 계약 1개

**질문:** 「398939 계보(같은 봉인 입력·솔버·ITER 6·ALI 8·T_e 고정; exports 델타 = `LUMINA_BB_NRB=1` 신설 + `LUMINA_A210_SPRODUCER_CAPTURE` 0→1)에서 논리 반복 0–5 의 생산 준위 행렬 bb 율을 비겹침 Sobolev 순 복사 괄호 R_absorb = B_lu·β·J_cont, R_stim = B_ul·β·J_cont, R_spont = A_ul·β(β·J_cont·β 는 같은 반복의 fine 조립기가 J̄ 를 만들 때 쓴 바로 그 값)로 바꾸면, 코어 셸 J_core 의 S II 광이온화율 Γ 와 S III/S II 가 398939 같은 반복 지수 대비 등록 문턱(Γ 0.3 dex·x 0.5 dex) 이상 **내려가는가**(SOURCE), 한쪽만인가(PARTIAL), 어느 쪽도 문턱 안인가(NULL), 오히려 오르는가(ADVERSE)?」

**말하지 않는 것:** 공개 CMFGEN 재현 · 전면 ALI(fine 격자 근사 Λ*) · 겹침 선·연속체가 만드는 J_cont 자체의 원천(fine 격자 J 는 전 선의 불투명도·방출을 포함한 blanketed 장이며 NRB 는 **국소 Sobolev 자기항**만 상쇄한다) · NT 물리 강도(NT 단계 OFF·준위 pref 0.05 는 398939 그대로) · 차가운 출발 · 공개 방출식에 β 를 한 번 더 곱하는 것 · 수렴 일반론·끌개 · Si/Ca/철족 판정 · 자유 T_e · SEED 반복(init 연산자)의 율 형태.

## §2. 수리·원천·가드·base 설계

- **(a) 수리 형태 = Sobolev 순 복사 괄호, J_inc = J_cont 직접.** 근거: 생산자(`line_net_rate.c`:296) 가 이미 CMFGEN 비겹침 Sobolev J̄ = β J_cont + (1−β) S_old 를 만들고, 두 준위 대수에서 n_l B_lu J̄ − n_u(A + B_ul J̄) 의 (1−β)S 항은 S 가 **현재** 인구의 것일 때 정확히 (1−β) n_u A 와 상쇄돼 순 율 = β[n_l B_lu J_cont − n_u(A + B_ul J_cont)] 가 된다(MALI 항등식). 현 코드는 S_old(직전 커밋 인구)를 넣어 그 상쇄를 반복에 맡긴다 — 그것이 2단계가 판독한 Λ-정체(교정 ∝ β). NRB 는 **율 자리에서 그 상쇄를 대수로 수행**하는 CMFGEN Sobolev 발자국이며 행렬은 인구에 선형으로 남는다. 세부 균형: J_cont = B_ν(T_e) 이면 β 가 상쇄돼 n_u/n_l → 볼츠만(selftest §3).
- **(b) 채택하지 않는 것**: 전면 ALI(J 솔버에 Λ* 를 새로 조립 — 다른 계약; NULL/PARTIAL 뒤 후보) · `LUMINA_MALI`(β·J̄ 는 β(1−β)S 자기항이 남는다) · J̄ 로부터 β·J_cont = J̄ − (1−β)S_old 역산(두꺼운 선에서 근접 대수 상쇄·부호 반전 자리를 만든다; mode 2 가 그 자리를 clamp 로 덮은 전례 :19056).
- **(c) β·J_cont 의 원천 = (A) 기존 SPRODUCER 원장을 행렬이 소비**(exports 에 `LUMINA_A210_SPRODUCER_CAPTURE=1`). 근거(초안 §3 경로 불성립 → 세 후보 비교): (A) writer 코드 0행, 셀·인덱스·세대가 jbar_line_det 와 **같은 호출·같은 대입 블록**(§0-5·6), stride 검사 관용구·항등식 로그가 이미 있음; 대가 = 호스트 메모리 +3.60 GB(§0-6; 32.4 + 3.6 = 36 GB < 64 GiB) · 「never consumed」 계약의 **개정**(아래) · A210 saturation 진단 CSV 의 producer 열이 값을 갖게 됨(진단 산출 변화, 물리값 무변경). (B) cmfgen.c 에 전용 2배열(β·J_cont, β) 발행 — 메모리 절반이나 cmfgen.c·lumina.h·lumina_atomic.c 에 할당·기록·해제 ~50행 신설(생산자 함수 안 편집). (C) plasma 가 τ_sobolev 로 β 를 재계산하고 J_cont 는 뷰에서 역산 — (b) 로 기각. ⟹ **판별력은 셋이 같고 변경 최소·검증 가능성 최대는 (A)**. **계약 개정 문면(이 사전등록이 정본, `lumina.h` 주석은 무변경)**: 「`line_producer_*` 는 `LUMINA_BB_NRB=1` 일 때 준위 행렬 bb 율의 **입력**이다. 그 밖에는 종전대로 read-only 원장이다.」
- **(d) 행렬 소비 규칙(plasma.c :19089–19110 Production split 안, JEQB 분기 뒤·A2-06 대입 자리)**: `LUMINA_BB_NRB=1` 이고 `nlte_bb_jbar_canonical` 이 rc 1(VALID/EXACT_ZERO) 을 준 셀에서만 —
  cell = line·n_shells + shell; ct = `line_producer_continuum_term[cell]`, le = `line_producer_local_emission_term[cell]`, τ_eff = `line_producer_tau_eff[cell]`;
  전제 검사(fail-closed, 하나라도 위반 = `[BB-NRB][FATAL] reason=<…> iter= line= shell= …` 1행 + 기존 BLOCKED 반환 경로로 실패 → 런 종료 rc≠0): ① `line_producer_terms_captured == 1` ∧ `_n_shells == n_shells` ∧ 5 포인터 비NULL (`LEDGER_ABSENT`) ② ct 유한 ∧ ct ≥ 0 (센티널 −1·비유한 = `CONT_TERM_INVALID`; ct = 0 허용) ③ **항등 |Jbar_view − (ct + le)| ≤ 10⁻¹²·Jbar_view**(EXACT_ZERO 는 양변 0) (`LEDGER_MISMATCH`) ④ `line_net_cmfgen_exponx(τ_eff, &β, &c)` rc 0 (`BETA_UNDEFINED`).
  율: **R_absorb = B_lu·ct, R_stim = B_ul·ct, R_spont = A_ul·β**; `J_line = Jbar_view` 유지(예산 진단), `jd_mode = 8`·`jd_beta = β`(기존 jbar 덤프 writer 표기, 값 소비 없음).
  rc 0 셀(OOG·UNSAMPLED·MISS·STALE)은 종전대로 R_absorb = R_stim = 0, R_spont = A_ul(원장 셀도 없고 β 도 정의되지 않는다 — 창 밖·미표본 선의 기존 처리, 무변경).
  **연산자 조건**: `opacity->jbar_line_det_operator == 2`(CMFGEN_NONOVERLAP_SOBOLEV) 일 때만 위 규칙; == 1(init Gaussian, SEED 뿐)이면 OFF 형태 그대로(Sobolev 분해가 없는 J̄ 에 β 는 정의되지 않는다) — **조용한 폴백이 아니라 태그로 증언**(아래). 그 밖 값 = FATAL `OPERATOR_UNKNOWN`.
- **(e) β 범위·음의 τ (초안 「β ∉ (0,1] → FATAL」 폐기)**: β 는 exponx 가 τ_eff 에서 주는 값을 **그대로** 쓴다. τ_eff ∈ [−0.5, 0) 에서 β ∈ (1, 1.297] 은 정확해가 허용하는 정상 물리 경로(§0-5 실측 beta_max > 1)이고, τ_raw < −0.5 는 생산자가 이미 SRCE_CHK 로 치환했다. 가드는 exponx rc(비유한·≤ 0)와 원장 센티널뿐 — 범위 clamp·부호 교정·|J| 치환·J̄ 대체 금지. ⚠A2-06 셸의 `radeq_beta_esc`(:14617; 분기점 다름)는 쓰지 않는다 — 생산자와 **같은 함수·같은 입력**이라야 bit-동일 β 다.
- **(f) 세대·호출 정합(초안 (iii))**: 새 게이트 없이 (d)③ 의 셀별 항등식이 담당한다 — 원장 셀과 뷰 값은 같은 `cmfgen_fine_jbar` 호출이 같은 `radiation` 구조체에서 썼고(`jbar = continuum_term + local_emission_term` :330), 뷰는 그 jbar 의 그대로 복사(§0-4)이므로 같은 세대이면 항등식은 반올림 없이 성립하고, 원장이 옛 호출의 것이면(진입 시 free/NULL 이라 원리상 불가하나) 항등식이 깨져 FATAL 이다. 반복 안 n_e 폐합 sweep 은 J̄·원장 모두 고정이므로 sweep 사이 세대 문제는 없다.
- **(g) 증인 태그(1종)**: `nlte_assemble_rate_matrix_impl` 안에서 (iter, operator) 쌍이 바뀔 때 1행(omp critical, 정적 마지막 쌍 비교) `[BB-NRB] iter=<current_iter> operator=<1|2> armed=<0|1> captured=<0|1> n_shells=<NS>`. 기대: SEED `iter=0 operator=1 armed=0` 1행 + k=0..5 `operator=2 armed=1 captured=1` 각 1행. 값 변경 없음.
- **(h) env**: `LUMINA_BB_NRB` 신설(`env_universe.h` 등록 + `LUMINA_ENV_UNIVERSE_COUNT` 531→532; 정렬 위치 유지 — `scripts/derive_env_universe.py` 재도출 또는 동일 결과 수동 1행; ENV_STRICT 가 받아야 한다). 값 "0"/미설정 = OFF(A2-06 세 대입 그대로, 태그 0행, 원장 미소비), "1" = ON, 그 밖 = 파싱 시점 FATAL(`a210_sproducer_capture_request_parse` 관용구). JEQB 가 켜지면 종전처럼 JEQB 분기가 이긴다(G-ID 가 부재 요구).
- **(i) base = 398939 + 델타 2키.** 과이온화·FUV 정체·초열적 두꺼운 선이 그 산출에 있고(원인 1·2 소비), run2(차가운 출발·NT 상한·XSHELL 0)를 섞으면 해제 불연속·pref 1.0 이 한 관측점에 붙어 원천이 갈리지 않는다. `LUMINA_XSHELL_RESPONSE=1` 유지(캡처만, 398939 동형). 제어 = 398939 **산출**(OFF 재런 없음, §0-9 기재).
- clamp·floor·부호 교정 없음. 공개 방출 writer(`emissivity_publication.c`:52 η = n_u A β hν/(4πΔν))·fine 조립기·J 솔버·커밋·coldstart·NT **무변경**(β 이중 적용 금지).

## §3. 변경집합과 산출물

### 변경집합 — DET-BB-ALI (표)
| 경로 | 분류 | 허용 변경 |
|---|---|---|
| `src/line_net_rate.h` | 순수 대수 API | 추가만: `LineNetBracketRates {beta, R_absorb, R_stim, R_spont}` 와 `int line_net_bracket_rates(double continuum_term, double local_emission_term, double tau_eff, double jbar_view, double B_lu, double B_ul, double A_ul, LineNetBracketRates *out)`(rc 0 / 음수 = §2(d) ②③④ 사유 코드). 기존 선언 무변경 |
| `src/line_net_rate.c` | 순수 대수 | 파일 끝에 위 함수 정의 **추가만**(§2(d) 검사 + 세 율; exponx 호출). `line_net_sobolev_material`·`line_net_cmfgen_exponx`·`line_net_sobolev_radiation` 등 기존 함수 **무변경**(diff 가 추가 블록 하나여야 한다) |
| `src/lumina_plasma.c` | 생산 bb 율 | :19089–19110 Production split 안에 `LUMINA_BB_NRB` 가지(§2(d): 원장 전제 검사 ① → `line_net_bracket_rates` → 세 율 대입·FATAL 1행+실패 반환·`jd_mode`/`jd_beta`), env 1회 파싱(정적), §2(g) 태그 1곳. JEQB 우선순위·OFF 세 대입·shadow 블록·jbar 덤프·나머지 전부 **무변경** |
| `src/env_universe.h` | 등록 | `LUMINA_BB_NRB` 1행 + COUNT 갱신. 다른 키 무변경 |
| `tests/bb_nrb_selftest.c` | 대수 selftest(GPU·plasma 링크 없음; `gcc -O2 -Wall -Wextra -Werror -std=c11 -Isrc -o selftest_bb_nrb tests/bb_nrb_selftest.c src/line_net_rate.c -lm`, Makefile 무변경) | §4 「selftest」 항목 전부 |
| `scripts/bb_ali_judge.py` | 판독·판정 | 직전 계약 판독기 `coldstart_judge` 계보(G-ID 신원 사슬·referent 생성 관용구 승계; 그 파일은 변경집합 아님 — 선언 블록 references 에 지목): G-ID/G-FIRE/G-RUN, δ 표(k = 2..5 궤적 + k=4,5 기하평균), §5 분기 5종, NC 3종(`--selftest`). `line_list.csv`·`nlte_levels_final.csv` 미오픈 |
### 변경집합 끝

`expected_extra` = 위 6파일. 변경 금지: `src/lumina_cmfgen.c` · `src/emissivity_publication.c` · `src/lumina.h`(주석 포함) · `src/lumina_atomic.c` · `src/lumina_cuda.cu` · `src/radiation_field.c` · `Makefile` · 봉인 입력 · 398939 산출. **산출**: `bb_ali_referents.csv`(§0-8 표를 398939 산출에서 재생성: 셸별 x_ctrl(k=2..5)·Γ_ctrl(k=2..5)·S_band_ctrl(k=3..5, 100–2000 및 <912/912–1400/1400–2000 Å)·x_pub — 오프라인 1회, sha 를 G-ID 가 대조) · `bb_ali_result.json`(게이트·δ 표·분기) · 표준 dump(398939 동형) · **지지 계측(판독기 밖, 오프라인 1회)**: 기존 `coldstart_20260924/cause2/fuv_lines_beta.py` 를 런 루트 인자만 바꿔 실행(`nlte_levels_final.csv` + `line_list.csv` 스트리밍 1회; 신설 스크립트 없음) → 두꺼운 선 S_l/B(sh15·sh24). 런 루트 `/gpfs/kjhan/lumina/cmfgen_map/bb_ali_20260925/run1/`. 빌드 = `nt/build_nt.sh` 계보(코드 감사 통과 트리, 186 경로 `source_sha_at_build.txt`·`binary.sha256`·`build.rc`), 스테이징 = `nt/stage_nt.sh` 계보(빌드 기록 복사·재해시 금지·승인 sha 검증 — 직전 계약 §3 도구 요구 (i)–(iv) 승계) + exports 델타 = **정확히 2키**. 예상 `--mem` 64G 유지(§0-6).

## §4. 게이트와 음성대조 (허용오차: 산술 항등 상대 1e-6; 물리 문턱은 §5)

게이트 **3종**. 새 dump 0 · 새 writer 0 · 새 stderr 태그 1종.

- **G-ID**: 봉인 입력 sha = 398939(deck·topion·sigma·fixed_te·deposition; 승계) · exports 델타 대 398939 `resolved_lumina.exports` = **정확히** {`+LUMINA_BB_NRB=1`, `LUMINA_A210_SPRODUCER_CAPTURE` "0"→"1"}, 그 밖 추가·삭제·변경 0(⟹ `LUMINA_MALI`·`LUMINA_NLTE_JEQB`·`LUMINA_ION_NT`·`LUMINA_INIT_IONFRAC_DIR` 부재·`LUMINA_XSHELL_RESPONSE=1`·ITER 6·ALI 8 자동) · 소스–바이너리 신원 사슬 = 직전 계약 §4 G-ID (a)–(e) 승계(승인 sha = 빌드 기록 = manifest = 트리 현행; 빌드 `binary.sha256` = manifest = footer = `input/lumina_cuda` 실측), 대상 = 변경집합의 소스 4경로 `src/line_net_rate.h`·`src/line_net_rate.c`·`src/lumina_plasma.c`·`src/env_universe.h`; 빌드 기록 대 직전 계약 빌드 기록(`coldstart_20260924/nt/cuda_build_nt1/source_sha_at_build.txt`) 차이 = **정확히 그 4경로**(부분집합·∅ FAIL) · `bb_ali_referents.csv` sha 기록 · h200 아님.
- **G-FIRE**(수리 발화·원장 도달): stderr 에 (1) `[BB-NRB] iter=k operator=2 armed=1 captured=1` 가 k = 0..5 각 **정확히 1행**, (2) `operator=2 armed=0` 0행, (3) `[BB-NRB] iter=0 operator=1 armed=0` ≤ 1행, (4) `[cmf_fine][SPRODUCER-CAPTURE] status=CAPTURED` **정확히 6행**(기존 writer — env 키가 생산자에 닿았다는 증거), (5) `[BB-NRB][FATAL]` 0행. 398939 로그에는 (1)(4) 가 없으므로 env 설정만으로는 발화를 증언하지 못한다 — 태그 1종은 유지(Q-LEAN L4 자기답 §6).
- **G-RUN**: SUCCESS · model_rc 0 · outer 6 · `physics_DET_iter000k.ion.csv` k = 0..5 존재·J 셸 x 유한 양수 · k = 3..5 `.spectral.csv`·`.shell.csv` 존재 · `lumina_ion_channels.csv` 의 최대 gen = stderr `[A2-07][NE-CLOSURE]` iterations 합 Σ(반복 마지막 sweep 사상의 전제 — Γ 가 소비) · G-FIRE 와 별개의 `[cmf_fine][BLOCKED]`·`[R6]`·`[R7][FATAL]` 0행. 실패 식별은 X.

| NC | 주입 | 결과 | 시점 |
|---|---|---|---|
| **selftest**(`tests/bb_nrb_selftest.c`) | (1) 픽스처 (n_upper, A_ul, ν, τ_raw ∈ {0, ±10⁻⁴, 0.3, −0.3, −0.49, −0.8, 5, 50, 10³}, t_exp, J_cont) 를 `line_net_sobolev_material`+`_radiation` 에 넣어 (ct, le, jbar, τ_eff, β_prod) 를 얻고 `line_net_bracket_rates(ct, le, τ_eff, jbar, …)` 호출 → β **bit-동일** β_prod, R_absorb = B_lu·ct, R_stim = B_ul·ct, R_spont = A·β bit-동일 (2) τ_raw = −0.3 에서 rc 0 ∧ β = (e^{0.3}−1)/0.3 (상대 1e-12) — β > 1 통과 (3) J_cont = B_ν(T) 일 때 (R_absorb)/(R_spont + R_stim) 로 준 n_u/n_l = (g_u/g_l)e^{−hν/kT} (상대 1e-12) — β 상쇄 (4) 음성: ct = −1 → rc ≠ 0 · τ_eff = NaN → rc ≠ 0 · jbar_view = (ct+le)(1+10⁻⁶) → rc ≠ 0 · jbar_view = ct+le 정확 → rc 0 · EXACT_ZERO(0,0,0,0) → rc 0·R = (0,0,A) | 전건 통과·음성 4건 rc≠0 | 구현 직후, 코드 감사 전 |
| **NC-ID** | (1) 398939 exports 원문(델타 0) → FAIL (2) `LUMINA_BB_NRB=1` 만 추가(캡처 0 유지) → FAIL (3) referents csv 1바이트 변조 → FAIL | 3건 FAIL | 런 전 |
| **NC-FIRE** | (1) 398939 stderr 원문 → FAIL(태그 0·CAPTURED 0) (2) 같은 로그에 armed 6행 + CAPTURED 6행 삽입 → 조건 PASS (3) (2) 에서 k=3 행 삭제 → FAIL (4) (2) 에 `iter=2 operator=2 armed=0` 1행 추가 → FAIL | (1)(3)(4) FAIL·(2) PASS | 런 전 |
| **NC-CLASS** | referent 표 위 δ 치환 픽스처: (S) med δΓ = −0.5 ∧ med δx = −0.8 → SOURCE · (S0) −0.3 ∧ −0.5(경계 포함) → SOURCE · (P1) −0.5 ∧ −0.2 → PARTIAL · (P2) 0 ∧ −0.8 → PARTIAL · (N) 0 ∧ 0 → NULL · (A) +0.5 ∧ 0 → ADVERSE · (A2) +0.1 ∧ +0.6 → ADVERSE | 7 이름 각 1회 적중·교차 0 | 런 전 |

OFF byte-동일: **코드 diff**(Production split 의 OFF 세 대입·shadow 블록 무변경, line_net_rate.c 추가 블록만) — 코드 감사 첫 항목과 함께 검수. OFF GPU 재런 없음.

## §5. 기대치·분기

**판정 셸**: 보고 J = {10–15, 19–30}(직전 계약 계승). **분기 코어 J_core = {10–15, 19–24}**(12셸; 원인 1 확증 셸·원인 2 sh15/24 표가 여기). sh25–30 은 표로만(sh30 = 바닥 채널 반례). **비교 = 같은 논리 반복 지수**(제어 398939 산출 대 이 런).

**정의**: x_s(k) = `Z16_stage_q2_cm3/Z16_stage_q1_cm3`(`physics_DET_iter000k.ion.csv`) · Γ_s(k) = S II `gamma_phot`(채널 덤프, 반복 k 마지막 sweep = NE-CLOSURE 누적) · S_s(k) = 100–2000 Å 대역 Ση_trueΔν/Σχ_abs B_ν(T_e)Δν(spectral+shell; 하위 대역 3개 병기). Q̄_s = √(Q_s(4)·Q_s(5)), **δQ_s = log10(Q̄_s^NRB/Q̄_s^ctrl)**, 판정 통계 = **med_{J_core} δQ**. k = 2..5 셸별 δ 는 궤적 표로 기재. 두꺼운 선 S_l/B(sh15·sh24, 최종 인구, `fuv_lines_beta.py`) 는 귀속 지지 열.

**문턱과 근거**: field_moved ⇔ med δΓ ≤ −0.3(제어 2반복 표류 최대 0.14 dex 의 2배·중앙값 0.06 의 5배; 코어 중앙값이 단일 셸 점프 0.49 를 흡수) · x_dropped ⇔ med δx ≤ −0.5(제어 표류 최대 0.14 의 3.5배·중앙값 10배; 공개값까지 gap 1.0–2.3 dex 의 1/4–1/2 — 「원천」이 물렸다면 최소 이 자릿수) · up_Γ ⇔ med δΓ ≥ +0.3 · up_x ⇔ med δx ≥ +0.5 · 지지 source_moved ⇔ med δS ≤ −0.15(제어 최대 0.07 의 2배; 분기 항 아님). 공개값 도달은 요구하지 않는다(NT pref 0.05·unmapped 13 %·얇은 선 25 %·J_cont 원천이 남는다).

**구성상 참(예측 아님)**: OFF 세 율 = A2-06(diff) · 원장 항등식(§2(f))은 같은 호출이면 반올림 없이 성립 · β = 생산자 β(같은 함수·입력) · J_cont = B 이면 볼츠만 · G-FIRE 행 수 = pass 구조(SEED 1 + Sobolev 6) · x_ctrl·Γ_ctrl·S_ctrl 는 398939 파일의 재적분 · NC-CLASS 픽스처 이름 · 메모리 +3.6 GB.

**예측(등록, 가설 H = 「Λ-정체 두꺼운 FUV 선 소스가 FUV 장과 과이온화의 원천」)**: H 가 참이면 SOURCE — β → 0 인 두꺼운 선의 복사 펌핑(직전 초열적 S_old 가 J̄ 를 통해 n_u 를 떠받치던 항)이 사라져 두꺼운 S II/III FUV 선 S_l/B(sh24 S III 2.91·S II 2.04, 반복당 1 % 고정)가 그 고정에서 풀리고(지지: 최종 S_l/B 변화 ≥ 10 %·source_moved), 명시 들뜬 준위 이온화 대역의 J 가 내려 Γ·x 가 문턱 이상 내려간다. **진폭은 오프라인으로 도출되지 않는다**(2단계 §6 확증 불가) — 문턱은 잡음 바닥·gap 비율의 하한이지 예측 진폭이 아니다. H 가 거짓이면 NULL(두꺼운 선 소스는 다른 채널 — 재결합 폭포·겹침 J_cont — 이 떠받친다) 또는 PARTIAL. ADVERSE(NRB 로 더 이온화)는 J_cont 자체가 초열적이라 자기항 제거가 β J_cont 펌핑을 상대적으로 키우는 경우 가능 — 기재.

| 분기 | 판정식 | 뜻 | 다음 |
|---|---|---|---|
| **ALI-SOURCE** | 게이트 3 PASS ∧ field_moved ∧ x_dropped | 율 자기항(Λ-정체)이 이 솔버·이 반복 창에서 S II 이온화 장과 과이온화를 문턱 이상 떠받친 원천으로 확증. 공개 재현 아님 | 3단계(NT 생산 반영) 설계를 NRB 기준선 위에서 user 에 보고; 잔여 gap 귀속(source_moved·S_l/B 표) 기재 |
| **ALI-PARTIAL** | 게이트 PASS ∧ (field_moved XOR x_dropped) | 한쪽만: Γ 만(0.3 ≤ |δΓ|, x < 0.5 dex — 실재하나 불충분) 또는 x 만(Γ 밖 경로 — DJDN/n_e/쌍 해; 귀속 실패). 반대 부호 동반 시 기재 | 움직인 쪽 표 + J_cont(겹침·연속체) 후보 보고; 전면 ALI 또는 겹침 계약은 user 결정 |
| **ALI-NULL** | 게이트 PASS ∧ ¬field_moved ∧ ¬x_dropped ∧ ¬up_Γ ∧ ¬up_x | 순 복사 괄호를 걸어도 문턱 안 — 국소 자기항은 원천이 아님(기각) | 전면 ALI·J_cont 원천·원인 1 채널의 다른 장 조립. 같은 NRB 재런 아님 |
| **ALI-ADVERSE** | 게이트 PASS ∧ ¬field_moved ∧ ¬x_dropped ∧ (up_Γ ∨ up_x) | 문턱 이상 **역행** — 자기항 제거가 이온화를 키움; J_cont 초열성의 직접 증거 후보 | J_cont 분해(연속체 대 겹침 선) 오프라인 판독 후 user 결정 |
| **X** | G-ID/G-FIRE/G-RUN 중 FAIL(FATAL 포함) | 미시연 | 계측·계약 위반만 수리, 같은 델타 1회 재실행 |

## §6. 자원·분장·주장 상한·집행 기록

- **GPU 1런**: partition h100,a100(h200 금지), 프로필 h100-a100x2, `--mem=64G`(예상 RSS ≈ 36 GB), `--time 16:00:00`, 예상 ≤ 13.5 h(398939 동형; 원장 읽기 추가 비용 무시). 재발주: X = 같은 델타 1회. 그 밖 = 재발주 아님.
- **런 전 CPU**(grammar-debug): selftest 컴파일·실행 · 판독기 NC 3종 · referents csv 생성 · (런 후) `fuv_lines_beta.py` 1회.
- **프리플라이트 합성 root**: docs·CLAUDE.md·data·scripts → repo, src·tests → 동결 트리.
- **분장(개정25-o/p, 이 단)**: 기안 = **Fable fresh**(grok 대행 초안 개정; 집행 1–3) · 기계 게이트 = 운전자 · 사전감사 = **astra**(물리·대수) + **경량 Q-GOAL/Q-LEAN = Opus 4.8 fresh**(개정25-d, 기안자와 계열 분리) · 구현 + 직접 selftest = **운전자**(개정25-o) · 코드 간단 감사 = **Fable fresh**(첫 항목 = 봉인 접촉 가부; ⚠자기 계약 편향 기재) · 발주·집행·판독·**판정 = 운전자** · 종료감사 = **Fable + astra**(개정25-m 두 축). ⚠구조적 귀결(기재, 폐합 조건 아님): 운전자가 구현·판정을 겸한다 — 남는 독립 검사 = ① 코드 감사(Fable) ② 종료감사 2겹 ③ 기계 게이트·NC ④ user. 실제 열은 집행 시 기입(`단계 / 규약상 / 실제 / 위반`).
- **Q-GOAL 자기답**: ① 지도 칸 = sh10–30 S 과이온화(공개 대비 1.0–2.3 dex, 코어) 의 **장 원천** — 원인 1(채널) 확증, 원인 2(Λ-정체 → 장) 미확증; 이 단은 그 연결을 1런으로 가른다 ② 산출 = 물리 측정(δΓ·δx 분기, δS·S_l/B 귀속) ③ 대안 = 전면 ALI(J 솔버 계약, NULL/PARTIAL 뒤) · run2 위 NRB(판별력 하락) · 오프라인 귀속(이미 확증 불가) ④ 건너뛰면 미확증 원천을 안은 채 3단계 NT 로 간다.
- **Q-LEAN 자기답**: 물리량 = J_core 12셸 δΓ·δx(분기) + δS_band·두꺼운 선 S_l/B(귀속). 소비: ion.csv k=2..5(x) · 채널 덤프 + NE-CLOSURE Σ(Γ; Σ 는 G-RUN 이자 Γ 사상의 전제) · spectral/shell k=3..5(S) · stderr 태그(G-FIRE) · exports·빌드 기록·manifest(G-ID) · referents csv · 최종 준위 덤프(+line_list 스트리밍, 기존 스크립트, 판독기 밖). 게이트 3 · NC 3 + selftest 1 · env +1 신설 +1 기존 ON · writer 0 · 새 dump 0 · 태그 1종(7행) · 코드 ≈ plasma ≤ 60행·line_net_rate ≤ 40행·헤더 ≤ 15행. 초안 대비 **뺀 것**: d_exc(단계 균형은 LTE 분율을 써서 x 가 소비하지 않음)·δJ 대역(Γ 가 대체)·β 범위 FATAL·G-ID 의 개별 키 부재 목록(델타 정확성 규칙 하나가 대체)·jbar 역산. **넣지 않은 것**: OFF 재런·전면 Λ*·MALI 재생·차가운 출발 중첩·반복별 준위 덤프·DJDN 삼중 대조. L4: 태그를 기재로 내리면 G-FIRE 가 env 설정만 믿게 되고, 현 MALI 처럼 「켰으나 덮어써진」 거짓 발화를 통과시킨다 — 유지; 항등식 ③ 은 게이트가 아니라 코드 계약(세대 정합의 유일한 증인)이라 유지. 걸음 수 = 1런.
- **주장 상한**: 「398939 계보 + `LUMINA_BB_NRB=1`(+캡처 1) 에서 반복 4·5 의 J_core 는 med δΓ = …, med δx = …, med δS = … 이었고 분기는 … 였다」까지. 어느 분기도 공개 재현·전면 ALI 불필요 증명·NT 결론·J_cont 원천 판정이 아니다. 제어는 398939 산출이며 현 트리의 휴면 coldstart 코드가 OFF 경로에 잔류한다(§0-9). NRB 는 국소 Sobolev 자기항만 상쇄하며 fine 격자 J_cont 의 겹침·연속체 자기 기여는 남는다. SEED 반복은 init 연산자라 Λ-형태 그대로다. G-ID 신원 사슬은 기록 사본의 결합이지 재컴파일 재현이 아니다.
- **집행 기록 뼈대**: 1 grok 대행 초안 · 2 운전자 발주 전 확인 결함 · 3 Fable 개정 기안(이 본문) · 4 파일화·기계 게이트 rc(prereg·referents·canonical·dispatch) · 5 astra 물리·대수 사전감사 + Opus 4.8 Q-GOAL/Q-LEAN(회차·차단) · 6 구현(운전자)·selftest·NC 3종 · 7 코드 감사 Fable(첫 항목 = 봉인 접촉 가부; line_net_rate.c 추가 블록만·OFF 대입 무변경·env 등록) · 8 빌드·스테이징·referents 봉인·제출(job id·노드·partition·`OWNER.txt`) · 9 판독(G-ID/FIRE/RUN·δ 표·궤적·S_l/B·분기) · 10 판정(운전자)·종료감사(Fable + astra)·폐합 계급.
- **집행 1** grok 대행 초안(Grok 4.6, 2026-09-25 13:12, `bb_ali_20260925/draft_raw.md`, read-only): 수리 형태(a)·base·분기 3종·게이트 3종 골격. 동결 트리 sed 는 **[미실행]** 로 남김.
- **집행 2** 운전자 발주 전 확인(2026-09-25, `[미실행]` 실행): **결함 1 — 구현 경로 불성립**(뷰에 β·J_cont 없음; 생산 지점은 cmfgen.c fine 조립기; 초안이 cmfgen.c 를 변경 금지로 묶음; β∈(0,1] FATAL 이 정상 경로를 죽임; 선 인덱스 공간 미확인). grok 재발주(`grok_rev1.md`) 산출 0바이트 = 미시연.
- **집행 3** Fable 개정 기안(`claude-fable-5-1`, 2026-09-25, read-only 본문 반환): §0-3~9 저자 실행(sed·grep·awk·python 재적분, 로그인 노드 비연산·소형 CSV) — 원천 (A) 채택·가드 재설계·항등식 계약·분기 4+X·문턱 근거 표·인덱스 공간 확인(§0-5).

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-BB-ALI (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)|scripts/[A-Za-z0-9_/.]+\\.py)",
    "symbol": "bb_nrb",
    "roots": ["src", "tests"],
    "expected_extra": ["src/line_net_rate.h", "src/line_net_rate.c", "src/lumina_plasma.c", "src/env_universe.h", "tests/bb_nrb_selftest.c", "scripts/bb_ali_judge.py"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    "metrics": {
      "count": "len(v)",
      "gate_fail": "sum(1 for x in v if x < 0.5)",
      "source": "sum(1 for x in v if 0.5 <= x < 1.5)",
      "partial": "sum(1 for x in v if 1.5 <= x < 2.5)",
      "null": "sum(1 for x in v if 2.5 <= x < 3.5)",
      "adverse": "sum(1 for x in v if 3.5 <= x < 4.5)",
      "spread": "(max(v) - min(v)) if len(v) > 0 else 0.0"
    },
    "rules": [
      {"name": "X", "predicate": "count == 0 or gate_fail > 0 or spread > 0.5"},
      {"name": "ALI-SOURCE", "predicate": "count > 0 and gate_fail == 0 and spread <= 0.5 and source == count"},
      {"name": "ALI-PARTIAL", "predicate": "count > 0 and gate_fail == 0 and spread <= 0.5 and partial == count"},
      {"name": "ALI-NULL", "predicate": "count > 0 and gate_fail == 0 and spread <= 0.5 and null == count"},
      {"name": "ALI-ADVERSE", "predicate": "count > 0 and gate_fail == 0 and spread <= 0.5 and adverse == count"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1]]},
      {"name": "all-source", "mix": [[1.0, 1]]},
      {"name": "all-partial", "mix": [[2.0, 1]]},
      {"name": "all-null", "mix": [[3.0, 1]]},
      {"name": "all-adverse", "mix": [[4.0, 1]]},
      {"name": "mix-src-null", "mix": [[1.0, 1], [3.0, 1]]},
      {"name": "mix-null-adverse", "mix": [[3.0, 1], [4.0, 1]]},
      {"name": "gate-and-source", "mix": [[0.0, 1], [1.0, 1]]}
    ],
    "residual": "X"
  },
  "references": [
    {"path": "docs/PREREG_DET_COLDSTART_NT_2026-09-24.md"},
    {"path": "docs/VERDICT_DET_COLDSTART_NT_2026-09-25.md"},
    {"path": "docs/VERDICT_DET_PATHB_XSHELL_2026-09-24.md"},
    {"path": "docs/READOUT_CAUSE1_SII_IONIZATION_CHANNEL_2026-09-25.md"},
    {"path": "docs/READOUT_CAUSE2_FUV_LINE_LAMBDA_STALL_2026-09-25.md"},
    {"path": "CLAUDE.md"},
    {"path": "data/standart_data1/toy06/ionfrac_s_toy06_cmfgen.txt"},
    {"path": "src/lumina_plasma.c", "flags_existing": ["nlte_bb_jbar_canonical", "nlte_bb_rate_graph_contains", "nlte_assemble_rate_matrix_impl", "line_producer_terms_captured", "line_producer_terms_n_shells", "A2_06_DIAGNOSTIC_SHADOW_END", "LUMINA_NLTE_JEQB", "radeq_beta_esc"], "flags_planned": ["LUMINA_BB_NRB", "line_net_bracket_rates"]},
    {"path": "src/lumina_cmfgen.c", "flags_existing": ["sproducer_capture", "line_producer_continuum_term", "line_producer_tau_eff", "CMF_FINE_LINE_OPERATOR_CMFGEN_NONOVERLAP_SOBOLEV", "jbar_line_det_operator", "line_net_sobolev_radiation", "SPRODUCER-CAPTURE"]},
    {"path": "src/line_net_rate.c", "flags_existing": ["line_net_cmfgen_exponx", "line_net_sobolev_material", "line_net_sobolev_radiation", "LINE_NET_NEGATIVE_OPACITY_CMFGEN_SRCE_CHK"], "flags_planned": ["line_net_bracket_rates"]},
    {"path": "src/line_net_rate.h", "flags_existing": ["LineNetSobolevRadiation", "line_net_cmfgen_exponx"], "flags_planned": ["LineNetBracketRates"]},
    {"path": "src/lumina.h", "flags_existing": ["line_producer_continuum_term", "line_producer_local_emission_term", "line_producer_tau_eff", "line_producer_provenance", "jbar_line_det_operator"]},
    {"path": "src/radiation_field.h", "flags_existing": ["LineJbarValue", "line_jbar_lookup"]},
    {"path": "src/emissivity_publication.c", "flags_existing": ["a209_sobolev_line_eta"]},
    {"path": "src/env_universe.h", "flags_existing": ["LUMINA_A210_SPRODUCER_CAPTURE", "LUMINA_MALI", "LUMINA_NLTE_JEQB", "LUMINA_ENV_UNIVERSE_COUNT"], "flags_planned": ["LUMINA_BB_NRB"]},
    {"path": "scripts/coldstart_judge.py", "flags_existing": ["g_id", "g_run", "make_referents", "EXPORT_DELTA"]}
  ]
}
```
