# relT3 "미통과" 심층 분석 — 독립 분석가(fable) 보고서

작성: 2026-08-01. 신규 런 없음 — 전부 기존 파일 실측. `docs/CODEX_RELT3_*` 미열람(독립성 준수). 선행 확정 사실은 [CODEX_RELT2_POSTMORTEM](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_RELT2_POSTMORTEM.md)만 참조.

## 판정 요약 (한 문단)

**(c) 혼합 — 단, 성분이 명확히 갈린다.** (1) "% correction" 잣대는 relT3 국면에서 **구조적으로 붕괴**해 있다: 반환 MAXCH는 terminal-ion 행(인구 1e-36~1e-59 cm⁻³)이 매 iteration 100~110% 감소를 제안해 **항상 1e7%로 고정**되고, 극값 % 는 인구 5.8e-41 cm⁻³짜리 Ca V 초고준위가 만든다. (2) 물리 가중 지표로 재면 상태는 **발산하지 않는다**: n_e·이온분율·질량가중 평균 전하가 it40→55 전 구간 0.1~2% 이내에서 준-정지, 심부(d≥53)는 완전 동결. 그러나 **실수렴도 아니다**: d16~32의 여기/부이온화 자유도가 ±10% 캡에 걸린 채 감쇠 없는 리밋사이클·크롤을 계속한다(Si III 분율@d27가 0.50~0.60을 15 iteration 내내 왕복; d21 Ca IV 바닥이 매 스텝 +20~40% 요구로 16 iteration에 2.0× 단조 성장). (3) it51 full step은 **방향은 물리적(진행 중인 Ca 이온화 크롤과 동부호), 크기는 선형화 유효범위 밖**이었고, 적용 결과 방출 스펙트럼은 3.7%만 움직였으나 **외곽 radiative-equilibrium 잔차가 32× 악화**(d1: −2.36e5 → −7.6e6), moment-luminosity 2.15× 점프 — probe 분기 폐기는 정당했다. 이후 "LAMBDA 제안 100× 악화"는 대부분 잣대 악화다(pop-가중 스텝 크기는 오히려 2~3× 감소).

---

## 0. 데이터 계보와 방법 검증 (잣대부터 감사)

### 0.1 상태 파일 계보 — 바이트 수준 확증

- `relT3/SCRTEMP` 선두 40 iteration = `modern/SCRTEMP` 전체와 **byte-identical** (`cmp -n 52435952` 통과). relT3는 modern it40 상태의 순수 continuation이다.
- `relT3_probe1/SCRTEMP` 선두 50 iteration = `relT3/SCRTEMP` 전체와 **byte-identical** (`cmp -n 65536752` 통과). probe1은 relT3 it50에서 분기했다.
- 따라서 it1~55는 단일 연속 궤적: it41–50 = relT3 P0 (LAMBDA, `MAX_LAM=1.10`), it51 = full BA (`MAX_LIN=1.05`), it52–55 = 자동 LAMBDA 강제.

### 0.2 SCRTEMP 디코더 (독립 재구성)

포맷은 소스로 확정: 레코드 16,376 B(=4094×4, [dir_acc_pars_gen.f:24](/gpfs/kjhan/cmfgen_src/cur_cmf/unix/dir_acc_pars_gen.f)), 레코드당 2047 double, R/V/SIGMA 2레코드 + iteration당 80레코드(POPS(1800,90)) — [scr_read_v2.f](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/scr_read_v2.f). 파일 크기 항등식: 2+80×55 레코드 × 16376 B = 72,087,152 B = probe1 실측 ✓.

디코더 검증(외부 앵커): it50 상태 vs `relT3/RVTJ`(13:37 기록, [RVTJ:52,65](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/RVTJ)) — **n_e 최대 상대오차 4.2e-8, T 4.2e-8, R 4.3e-11**. 변수 배치는 MODEL_SPEC ISF 합산으로 구축하고 CORRECTION_LINK의 I(STEQ) 앵커 12개(NkV49→1730, CaV38→681, CoV12→1448, NkSEV→1798 등)로 전수 검증.

### 0.3 % 잣대의 소스 수준 의미론 (판단 전 확정)

- SOL(J,L)>0=감소, <0=증가; 화면의 "% increase"=100×|SOL| — [solveba_v13.f:155-156,198-199](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f).
- **반환 MAXCH: 감소 제안 ≥99.999%면 무조건 1e7% 치환** — [solveba_v13.f:207-213]. relT3 it46–50 모두 DEC_VEC 상위 10개가 1.00~1.03 → 반환 MAXCH ≡ 1e7% ([OUTGEN:139-140,171-172,203-204,235-236,274-275](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/OUTGEN)). `EPS_TERM=0.1%` ([MODEL:314](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/MODEL))는 이 국면에서 **구조적으로 도달 불가능**.
- 적용 스텝(MAJOR 스케일링): 깊이별 SCALE은 "major"(pop > 1e-10×n_e) 변수만으로 결정, 이후 **전 변수 per-var 클립** [−(1−1/C), +(C−1)] (LAMBDA C=1.10: +10%/−9.09%; full C=1.05: +5%/−4.76%) — [fiddle_pop_corrections_v2.f:156,161,184-192](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f). 상태 diff 실측과 일치(모든 LAMBDA 스텝 max|ΔN/N|=1.000e-1, full 스텝 5.000e-2).
- LAMBDA LIMIT 옵션은 **SOL>1.1만** 0.999로 치환 — [solveba_v13.f:129-141]. terminal 행의 SOL≈1.087~1.098은 이 그물을 **정확히 빠져나간다**(아래 §4).

### 0.4 유실 기록 (정직 신고)

it41–45의 OUTGEN/batch.log는 스틴트 재발주 스크립트의 `rm -f OUTGEN batch.log ...` ([slurm_cmfgen_relt3.sh:40](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/slurm_cmfgen_relt3.sh))로 소실됐고 slurm .out은 래퍼 2줄뿐(각 130 B, `seq_logs/relt3_1948_slurm-398721.out` 등). 과제 지시문의 "it41–50 극값 6.3e4%→1.5e3%"는 **파일로 재검증 불가 — UNRESOLVED**(생존한 it46–50 패턴과는 정합적). 상태 자체(SCRTEMP it41–45)는 온전하며 본 보고서의 물리 지표는 전부 상태 실측이다.

---

## 1. Q1 — 물리 가중 수렴 지표의 독립 재구성: 상태는 수렴/정체/발산?

### 1.1 n_e(depth): 준-정지

it40→55 (16 iteration, full step 포함) 실측:

| depth | n_e(it40) | n_e(it55) | 총변화 | 비고 |
|---:|---:|---:|---:|---|
| d8 | 2.1136e5 | 2.1027e5 | −0.52% | far-outer |
| d21 | 1.5825e6 | 1.5844e6 | **+0.12%** | 극값 % 발생 지점 |
| d27 | 4.6619e6 | 4.7574e6 | +2.0% | 최대 변동 깊이 |
| d31 | 9.3404e6 | 9.2567e6 | −0.90% | |
| d50 | 3.7717e8 | 3.7731e8 | +0.04% | |
| d80 | 1.4240e10 | 1.4216e10 | −0.17% | 심부 |

iteration당 max_d |Δln n_e| = 3~7e-3 (LAMBDA), full step은 2.2e-3. **n_e는 어느 깊이에서도 발산 기미가 없다.**

### 1.2 이온분율/질량가중 이온화: 정지

지배 이온분율(종-정규화, d21): SkIV 0.985→0.990, SIV 0.990→0.991, CaIII 1.000→1.000, FeIV 1.000, CoIV 0.9994, NkIV 0.995→0.996 (it40→55). 수-가중(shell r²Δr) 전역 평균 전하: Fe 2.8912→2.8895, Co 2.8260→2.8242, Nk 2.7611→2.7594, Ca 2.00010→2.00011 — **15 iteration 총변화 ≤0.07%**. 거시 이온화 상태는 사실상 고정점에 있다.

### 1.3 그러나 "실수렴"은 아니다 — 캡-구속 정체(stalled limit cycle)

(a) **d27 Si III/IV 실물 진동**: Si III 분율@d27 = 0.601, 0.578, 0.555, 0.531, 0.521, 0.518, 0.496, 0.526, 0.502, 0.534, 0.564 | 0.562, 0.557, 0.550, 0.543, 0.535 (it40→55). 감쇠 없는 ±5% 왕복 + 크롤. major-변수 방향 코사인@d27: it45–48 구간 −0.40~−0.67(진동), it48–50 +0.99(크롤), it52–55 +0.96~+1.00(크롤) — 스텝 크기 |Δln|₂ = 0.13~0.27로 **일정**(감쇠 없음).

(b) **d21 Ca IV 바닥 크롤**: pop(CaIV g.s., d21) = 0.079 → 0.159 cm⁻³ (it40→55, 단조 2.0×). LAMBDA 제안이 매번 −0.17~−0.40(= +17~40% 증가 요구)로 소진되지 않음. Ca IV 분율은 1.9e-6→3.9e-6 — 거시적으론 무의미하나 **고정점에 도달하지 못했다는 신호**.

(c) **d21 지배 준위(13개, 종 인구 ≥1% 보유)의 최대 |Δln|**: it52→55에서 1.000e-1 — **여전히 ±10% 캡에 걸려 있다**(CoIV 저준위 SL3/SL5 = var1383/1385). 심부(d53+)는 0.1% 이상 변하는 변수 0개([CORRECTION_SUM:58 이하](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/CORRECTION_SUM)).

(d) **pop-가중 스텝 노름** Σ|ΔN|/ΣN (깊이별 max): it40–50 = 2.1~3.8e-2 (항상 d26–27), it51–55 = 1.2~1.4e-2. 감쇠 추세가 아니라 **plateau**.

### 1.4 % 잣대와의 대조 — 극값의 물리적 실체

| 극값 주인공 | 변수 | 깊이 | 인구(cm⁻³, it50) | 종 대비 | n_e 대비 |
|---|---|---|---:|---:|---:|
| it46/48 스파이크 +1.4~1.6e5% | CaV SL70 (`3p3(2Do)8g` 초고준위) | d21 | **5.3e-41** | 1.3e-45 | 3.4e-47 |
| it54/55 스파이크 +2.8~7.5e5% | CaV SL62 | d21 | 1.2e-38 | 2.9e-43 | 7.6e-45 |
| terminal 감소 100~110% | NkSEV(=Ni VII) | d27 | **2.9e-59** | 2.8e-55 | 6.1e-66 |
| 〃 | CoSEV, FeSEV | d27 | 2.7e-55, 1.1e-54 | 2.6e-51, 1.1e-50 | ~1e-61 |

d21 셸 부피 ≈ 3.1e46 cm³ → 1원자 문턱 = 3.2e-47 cm⁻³. **d21에서 1798개 변수 중 757개가 pop<1e-20, 148개는 셸 전체에 원자 1개 미만.** it51에서 |SOL|>100%였던 728개 변수 중 450개가 pop<1e-20, 150개는 sub-1-atom. CORRECTION_SUM의 "d21에서 410개 변수 100% 이상"([CORRECTION_SUM:26](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/CORRECTION_SUM))은 이 유령 인구의 통계다.

**Q1 판정: 물리 상태는 (i) 심부·거시 이온화 — 수렴/동결, (ii) d16–32 여기·부이온화 — 감쇠 없는 캡-구속 정체(수렴도 발산도 아님), (iii) % 잣대 — 물리와 분리된 별도 신호(trace 지배).**

---

## 2. Q2 — it51 +3.84e7%@d21의 변수 동정과 성격

### 2.1 동정 (직접 실측 — 추정 아님)

probe1 `STEQ_VALS`는 iteration당 1블록(it51–55)의 raw STEQ + **STEQ SOLUTION ARRAY**를 보존한다(블록 시작 [STEQ_VALS:2,49091,98180,147269,196358](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/STEQ_VALS); it51 solution은 32880행부터). 파서 검증: 10개 블록 전부의 max inc/dec가 OUTGEN 화면값과 일치(it51: 3.839e7%@d21 ✓, it55: 7.483e5%@d21 ✓ 등 10/10).

**it51 d21 최대 증가 = var 713 = Ca V SL70 (`3s2_3p3(2Do)8g_3Ho/1Do…`), SOL = −3.8393e5.** 차순위: var 714 = **CaSIX(Ca VI 이온) −2.2537e5**, CaV SL33 −1.15e5, SL53, SL43, SL60 … — d21 상위 20개 중 19개가 Ca V 준위 + Ca VI. 전역(전 깊이) top-20도 전부 d19–21의 CaV/CaSIX. 최대 감소는 CaV SL5 +48.2@d27 (4823% — OUTGEN의 "4.82E+03" ✓).

### 2.2 full/LAMBDA 증폭 실측

같은 변수의 it50(LAMBDA) 제안과 비교 (d21):

| 변수 | SOL(it50, Λ) | SOL(it51, full) | 배율 |
|---|---:|---:|---:|
| CaV70 | +0.852 (85% 감소) | −3.84e5 (×384k 증가) | **−4.5e5 (부호 반전)** |
| CaSIX | +0.994 | −2.25e5 | −2.3e5 |
| CaV33 | +0.974 | −1.15e5 | −1.2e5 |
| CaIV g.s.(603) | −0.218 (+22%) | **−213.5 (×214)** | +981 (동부호) |

### 2.3 성격 논증: 물리인가 수치인가 — **혼합, 층위 분해**

**(i) 방향은 물리적이다.** 근거: (a) Ca IV 바닥@d21은 LAMBDA 하에서 매 iteration +20~40%를 일관 요구하며 16 iteration 동안 2.0× 단조 성장 중(§1.3b) — full BA의 ×214 제안은 이 크롤의 고정점 점프 시도로 동부호. (b) 사다리 배율 구조: SOL(CaIV g.s.)=−213.5 → SOL(CaV70)=−3.84e5 ≈ 1800×, SOL(CaSIX)=−2.25e5 ≈ 1000× — 이온화 사다리(CaIV→CaV→CaVI)를 따라 평형비가 연쇄 재조정되는 선형 응답의 전형. 극값은 CaIV 신호의 사다리-증폭 영상이지 독립 모드가 아니다. (c) LAMBDA의 격회 스파이크(it46/48/54/55)도 동일 가족(CaV70/62) — full 선형화가 "만든" 방향이 아니라 원래 있던 느린 방향을 크게 밟은 것.

**(ii) 크기는 수치적으로 무효하다.** (a) ×1e5 상대 보정은 선형화 유효범위 밖(1차 Newton의 외삽). (b) 그 방향의 절대 규모: CaV70 인구 5.6e-41 cm⁻³ — ×3.8e5를 다 줘도 2.1e-35 cm⁻³로 물리 무의미. (c) **국소 야코비안의 조건화 실측**(probe1 `BA_ASCI_N_D5/D41/DND` = it55의 depth-block C_MAT; 열이 POPS로 스케일된 분수-보정 변수계, [generate_full_matrix_v3.f:385-411](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/generate_full_matrix_v3.f)): raw cond(d5) = 6.2e51; DGEEQU식 평형화 후에도 **cond(d5)=9.7e9 vs cond(d41)=5.3e7, cond(d90)=1.4e7** — far-outer 블록이 100~1000× 나쁘다. 평형화 행렬의 최소 특이벡터 성분: **1위(σ=9.3e-10) Si V 고준위 클러스터, 2위(σ=1.5e-6) Ca V 고준위 클러스터**. 역대 폭주 가족(385770=Si V, relT2=Si III, relT3=Ca V)이 정확히 국소 야코비안의 근사-널 공간이다. 근사-널 방향은 RHS의 작은 변화(예: full 선형화가 추가하는 dJ/dN 항)에 해가 극민감 — 배율의 신뢰도는 0이다. (d) 그 dJ/dN 계수 자체가 취약 지대에서 계산된다: d21은 **flux-mean 불투명도가 음수**(χ_Flux=−2.203e-7 vs χ_Ross=+1.059e-8, [MEANOPAC:22](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/MEANOPAC))이고 NEG_OPAC에 음불투명 전이 15,174건이 기록된 영역이며, it51의 moment solve는 5.09e15 Hz(EUV)에서 수렴 실패를 냈다([probe1 OUTGEN:129-132](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN)).

**(iii) 검증 폐합**: 동일 파이프라인으로 C_MAT(d5)·x=RHS를 평형화 solve하면 SOL(it55,d5)를 major 변수 중앙오차 4.5e-6, 전 변수 100%가 1% 이내로 재현 — 행렬 해석은 실제 솔버 경로와 일치한다.

**소결**: BA 전역 선형화가 키운 것은 "실제 불안정의 새 방향"이 아니라 **실재하는 느린 이온화-완화 방향의 크기를, 근사-널(trace) 사다리를 통해 무효한 배율로 외삽**한 것. "다중근" 증거는 없음(단, full BA d21 블록 자체는 미보존이라 고유값 부호 확정은 UNRESOLVED — §6).

---

## 3. Q3 — full step(±5%)이 상태를 실제로 어디로 옮겼나

### 3.1 상태 diff (SCRTEMP 실측)

- 클립 통계: **+5% 클립 10,828 (var,depth), −4.76% 클립 12,104**. 분포가 웅변적: d43–53에서 깊이당 ~300–770개 +5%, d54–87에서 깊이당 ~200–720개 −4.76% — LAMBDA가 **한 번도 건드리지 않던 심부**(iteration당 popw(d56–90)=1.9e-6)를 full step은 popw 7.7e-3으로 밀었다(×4000). 내용: 예컨대 d60에서 Co II 바닥들 −0.6%, Co III 바닥 +1.6% — Co II→III 방향 심부 재이온화 제안.
- 종 수 보존은 유지(전 깊이 |Δ(종 총수)|/총수 ≤ 2e-6, 대부분 1e-10 이하) — solve는 보존 제약을 존중했다.
- 전역 pop-가중 거리: it50 기준 ‖it51‖=7.5e-3. 참고로 직전 LAMBDA 밴드폭(it46–49 vs it50)은 1.4e-5~4.9e-5 — **full step은 LAMBDA 왕복 밴드의 ~150×를 한 번에 이동**했고, it52–55는 it50으로 되돌아오지 않고 그 자리에서 감(거리 7.5→8.1e-3).

### 3.2 상태 악화인가 — 세 개의 물리 잣대

| 잣대 | it50 상태 | it51 스텝 후 | 판정 |
|---|---:|---:|---|
| 방출 스펙트럼(OBSFLUX 적분) | — | 총 +3.7% (912–2000Å +3.8%, 광학대 ±0.5%) | 관측계 영향 경미 |
| radiative-equilibrium 잔차 d1 | −2.36e5 (it46–51 정체) | **−7.6e6 (it52–55)** | **×32 악화** |
| RE 잔차 d21/d27 | +3.4e2 / +2.0e4 | +4.6e3 / +1.35e5 | ×13 / ×6 악화 |
| RE 잔차 d83 | −6.23e6 | −5.30e6 | 15% 개선 |
| moment-L(d=1) | 6.19~6.64e10 | 1.33e11 (it52–55 안정) | ×2.15 점프 |

교란 변인 통제: it51 블록의 RE(d1) = −2.363e5 ≈ relT3 it50 블록 −2.364e5 — **JEW/EDDFACTOR 재구축(probe 분기 시)은 잔차를 바꾸지 않았다**. 따라서 it52의 ×32는 온전히 it51 스텝(상태 변화)의 몫이다.

### 3.3 "LAMBDA 제안 100× 악화"의 해부 — 상태 악화 vs 잣대 악화

- **잣대 성분(지배적)**: it52–55 극값 2.1~7.5e5%의 주인공은 it54/55 기준 CaV SL62@d21 (인구 1.9e-38) — pre-probe 스파이크와 동일 가족. terminal 행은 1.09x로 여전히 MAXCH=1e7 고정. 한편 pop-가중 스텝은 2.8–3.8e-2(it46–50) → **1.2–1.4e-2(it52–55)로 오히려 2~3× 감소**. 즉 "100×"는 유령 변수의 %가 커진 것.
- **상태 성분(실재)**: full step이 외곽 에너지 일관성(RE)을 ×32 망가뜨린 지점에서 LAMBDA가 재시작됐다. CaV 가족의 % 성장(스파이크의 격회성 소멸, it52→55 단조 악화 −13.9→−2207)은 스텝 후 상승한 EUV/FUV 장(§3.2의 d1 과방출)이 Ca 사다리를 더 세게 미는 것과 정합.

**Q3 판정: 관측계·인구계로는 소폭 이동(3.7% / 0.75%), 에너지 일관성으로는 명백한 후퇴(×32). "probe 분기 폐기, it50 보존" 결정은 물리 잣대로 정당하다.** (거리로도 it52–55는 it50으로 수렴하지 않고 새 준-정체점 주위를 돈다.)

---

## 4. Q4 — terminal-ion 행(100~110% 감소 고착)의 실체와 처방

### 4.1 실체 — 행렬 수준 해부 (BA_ASCI d5 블록 실측, 해석 3중 검증)

terminal 변수(NkSEV=Ni VII, CoSEV, FeSEV, CaSIX, …)의 시스템 내 위치:

- **자기 행 = 종 수-보존 방정식**: row 1798(NkSEV)의 성분이 정확히 각 Nk 변수의 인구와 일치(NkIV1 계수 2.767e-6 = pop(NkIV1,d5)=2.7679e-6 ✓; row 714의 CaIV1 계수 2.2e3 = pop 2246.7 ✓, CaIII1 1.9e3=1936.8 ✓). 보존 행에서 terminal 변수 자신의 계수 = 자기 인구 = **1e-36** — 보존식은 이 변수를 전혀 구속하지 못한다.
- **자기 열 ≈ 널**: col 1798의 최대 성분은 전하보존(ED) 행의 6.8e-36(=전하6×pop ✓ CaSIX로 교차검증: 5×6.638e-15=3.319e-14 = ED행 계수 3.3e-14 ✓)과 자기 대각 1.1e-36, 그 다음이 NkSIX 준위 행들의 재결합 항 4.9e-42. 행렬 전형 스케일(1e-6~1e3) 대비 **30~40자릿수 아래**.
- 결과: terminal 변수의 보정은 1e-36급 계수들 사이의 몫으로 결정되는 **수치 잔향**이다. 구조가 매 iteration 같으므로 값도 안정적으로 SOL≈+1.087~1.098에 앉는다([probe1 CORRECTION_LINK:23-26](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/CORRECTION_LINK): NkSEV 1.0978, CaSIX 1.0967, CoSEV 1.0937, FeSEV 1.0871@d27). 이는 modern 부검의 "잔차 −7e-56 → 보정 +1.0064"(사례 21)의 행렬-해부학적 완성이다.
- 이 값이 유해한 경로는 물리아 아니라 **제어 로직**: (a) SOL>0.99999 → 반환 MAXCH≡1e7% → EPS_TERM 영구 미달 + LAM_VAL=400 초과로 auto-LAMBDA 고착; (b) LIMIT 그물(>1.1만 치환, [solveba_v13.f:133])을 1.09가 정확히 통과; (c) 적용 시 매 iteration −9.09%(캡)로 서서히 0을 향해 감쇠할 뿐(NkSEV@d27: 7.4e-59→2.6e-59, it40→55) 절대 도달하지 않아 제안이 재생산된다. DO_LEVEL_CHK(10th-max 반환, [solveba_v13.f:262-269])는 문턱(10% 미만 조건)이 커서 이 국면에서 사문화.

### 4.2 처방 후보 (검증 용이 순)

1. **변수 소거/동결(최우선)**: pop < ε×(종 총수) (예 ε=1e-20, 또는 sub-1-atom/shell)인 terminal-ion·초고준위 변수는 선형화에서 제외(SOL:=0) 하고 하위 이온과의 비를 동결. 정확해가 위반할 수 없는 조건(유령 인구의 임의 유한 배율은 관측·수지 불변)이므로 "кламп는 물리가 아니다" 원칙과 충돌하지 않음 — 이것은 물리 수정이 아니라 **잣대·자유도 위생**.
2. **수렴 지표 재정의**: 반환 MAXCH를 pop-가중(예: 종 인구 x% 이상 보유 변수의 max|SOL|, 또는 Σ|ΔN|/ΣN)으로 병행 산출·판정. 현 DO_LEVEL_CHK의 자리(이미 훅 존재)에 이식 가능.
3. **LIMIT 구멍 봉합**: SOL∈(0.999,1.1] 구간도 치환하도록 문턱 수정(1줄) — 반환 MAXCH 1e7 고정의 즉효 완화(단 근본책은 1).
4. terminal 행의 별도 스케일링(관측된 대안): ADJUST_CORRECTIONS 파일로 깊이별 RELAX/POP_LIM 조정이 소스에 이미 내장([fiddle_pop_corrections_v2.f:95-143]) — 단 변수-선택성이 없어 1의 대체재는 아님.

---

## 5. Q5 — 종합 판정과 차기 프로브 1건 (사전등록)

### 5.1 판정: **(c) 혼합** — 성분표

| 성분 | 판정 | 결정 근거(실측) |
|---|---|---|
| % 잣대 | **아티팩트 확정** | MAXCH≡1e7(terminal 1.09 고착), 극값=인구 1e-38~1e-41 변수, d21의 100%+ 보정 728개 중 450개 pop<1e-20 |
| 거시 물리 상태 | **준-정지(발산 아님)** | n_e ≤2%/15it, 평균 전하 ≤0.07%, 심부 동결, 스펙트럼 3.7% |
| d16–32 미시 자유도 | **캡-구속 정체(수렴 아님)** | Si III@d27 0.50–0.60 무감쇠 왕복, CaIV@d21 2.0× 단조 크롤, 지배 준위 |Δln|=캡 10% 지속 |
| it51 full step | **방향 물리/크기 무효, 순효과 후퇴** | RE(d1) ×32 악화(전달재구축 통제됨), moment-L ×2.15; 스펙트럼 +3.7% |
| probe 폐기 결정 | **정당** | 위 행 |

relT2 부검의 "modern은 불안정 궤적 위" 서사는 이렇게 정정된다: **궤적은 '폭주 중'이 아니라 '캡이 만든 유계 정체'다.** 다만 EPS_TERM 기준 수렴은 현 잣대·현 자유도 구성으로는 영원히 불가능하다.

### 5.2 차기 프로브 P-TF ("trace-freeze LAMBDA") — 사전등록

**개입(단일)**: it50 상태에서 재개하는 LAMBDA 5 iteration. 유일 변경 = solveba_v13/fiddle에 위생 게이트 1개: `POPS(J,I) < 1e-20 × SPECIES_TOT(spec(J),I)` 인 변수는 SOL(J,I):=0 (적용·통계 모두에서 제외). T 고정, MAX_LAM=1.10, 그 외 relT3 P0와 동일. (구현 대안: 통계만 게이트해 '측정용 MAXCH'를 병행 출력하는 무개입판 — 더 보수적.)

**사전등록 기대**:
- E1 (잣대 개방): 반환 MAXCH가 1e7%에서 **major-변수 실값(예상 1e2~1e4%)**으로 즉시 내려온다. terminal 1.09 제안 소멸.
- E2 (물리 불변): n_e drift/it ≤7e-3, popw(d26–27) 1e-3~4e-2, Si III@d27 진동 진폭 ±5% — 전부 relT3 P0 실측 밴드 내 유지(2× 이내). 이는 "trace가 물리 상태를 끌고 있지 않았다"의 직접 검증.
- E3 (실물 미수렴 노출): 게이트 후 MAXCH의 주인은 d21–32의 CoIV 저준위/CaIV/SkIII 실물 크롤로 이동하고, 5 iteration 내 감쇠하지 않는다(코사인 +0.8 이상 지속). → 다음 단계(실물 자유도용 감쇠/앵커 설계)의 표적이 확정된다.
- **반증 조건**: 게이트 후에도 MAXCH가 1e5%+로 남고 그 주인이 pop-가중 major라면, (a) 실체 발산으로 재격상하고 본 보고서의 (c) 판정을 폐기한다.
- 비용: CPU slurm 1스틴트(~55분, relT3 실측 기준). 판정은 이벤트/파일 배터리(OUTGEN MAXCH, CORRECTION_LINK 주인공, SCRTEMP 물리 지표 재계산)로 1회.

---

## 6. UNRESOLVED (정직 목록)

1. it41–45의 OUTGEN 수치(§0.4) — 파일 소실. 지시문 수치는 driver-보고로만 존재.
2. it51 full-BA의 **d21 블록 행렬 자체** — 미보존(BA_ASCI는 d5/d41/d90만, 그마저 it55 LAMBDA 것). 따라서 full 선형화의 d21 고유값 부호(진짜 불안정 모드/다중근 여부)는 미확정. §2의 판정은 해(SOL)·사다리 구조·조건화 방증에 기반.
3. moment-L(d=1)=1.33e11의 절대 기준 대비 평가 — 기준 L 부재(base 런 자체가 5e11→2.7e10 진동 후 NaN, [toy06_19.48d/OUTGEN it63-65]). 상대 비교(×2.15)만 유효.
4. terminal SOL이 하필 +1.087~1.098에 앉는 대수적 이유(재결합/보존 항 비율의 고정점으로 추정) — 미도출.
5. d27 Si III/IV 진동의 근원(J-피드백 지연 vs 원자데이터 민감성) — 본 과제 범위 밖, P-TF의 E3가 표적화함.

## 7. 재현 자료

분석 스크립트(전부 재실행 가능): `/tmp/claude-10396/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/50011b1c-ea14-4956-add2-6a1c0478ce63/scratchpad/` — `scrtemp_lib.py`(디코더+변수맵), `run_metrics.py`(Q1 지표), `parse_steq.py`(SOL 파서+it51 동정), `limit_cycle.py`, `re_resid.py`(RE 잔차), `ba_analysis.py`/`ba_solve2.py`(C_MAT 해부), `final_checks.py`, `obsflux_cmp.py`. 중간 산출물: `sol_relT3.npy`, `sol_probe1.npy`.

주요 원자료: [relT3/OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/OUTGEN), [probe1/OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN), 양 디렉토리의 STEQ_VALS/CORRECTION_LINK/CORRECTION_SUM/SCRTEMP/POINT1/RVTJ/MEANOPAC/NEG_OPAC/BA_ASCI_N_D{5,41,ND}, [solveba_v13.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f), [fiddle_pop_corrections_v2.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f), [generate_full_matrix_v3.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/generate_full_matrix_v3.f).
