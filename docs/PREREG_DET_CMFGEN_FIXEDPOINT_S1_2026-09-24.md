# 사전등록 — DET-CMFGEN-FIXEDPOINT-S1: 율 반쪽 오프라인 (R_Γ · α 비 · β_C · K)

**표적:** CMFGEN 자신의 장 J_C 와 상태(it66 입력 = SCRTEMP 65)를 그대로 주었을 때, Lumina 생산 율 연산자(σ 표·LTE 준위 가중·canonical 적분기·Milne α)가 CMFGEN 의 S II 광이온화율·재결합 계수를 재현하는가. sh24(CMFGEN d33/d34) 주, 50 셸 부산물. 상태 = **기안**(실행 승인·구현·판정 아님). 기안자 = Fable fresh(`claude-fable-5-1`, 개정25-k). 최종목적 = MC 수송으로 CMFGEN 수준 SN Ia 스펙트럼 재현. 1단계(율 반쪽)만이며 전달 반쪽(주입 훅·형식해)은 이 결과 뒤 user 가 따로 정한다(2026-09-24 결정).

## §0. 선행 사실 (저자 실행 = 로그인 노드 비연산 조회; 값은 referent, 예측 아님)
1. **S2PRRR 규약**: `PR(I,J)+=WSE·HN·JPHOT`, `RR(I,J)+=WSE·HNST·JB_RAT`(prrr_sl_v6.f:126–127) — HN=초준위 인구(cmfgen_sub.f:2400 `ATM(ID)%XzV`) ⟹ **PR·RR 은 체적률[cm⁻³s⁻¹]**. 출력은 그대로(wrrecomchk_v5.f:87·101). 같은 파일의 「Radiative Recombination Coefficient for explicitly treated levels」 = `ΣRR/(ED·DHYD)`(wrrecomchk_v5.f:186·193) = **CMFGEN 자신의 α_C**. 「Net Recombination Rate (% of total)」 = 파서 검증용 항등식(:171–182). DIERECOM 행은 0 이면 안 찍히며 S2PRRR 에 없다 — `VADAT:368 DIE_S2 F,F` ⟹ **CMFGEN 의 S II 재결합은 RR 뿐, DR 없음**. 초준위 1 = 바닥 4S°[3/2] 단독(S2_F_TO_S 13행; LEVEL_SL_STEQ_LINKS:538, STEQ 272). S2=55 SL(STEQ 272–326), SIII=79 SL(327–405), NT=1645(MODEL:9), ND=90.
2. **쌍 정합**: it66 의 EDDFACTOR·PRRR 은 SCRTEMP **65** 인구·ED 에서 계산, SCR_RITE_V2 는 solve 뒤 IREC 증가(solve_for_pops.f:321, scr_read_v2.f:402). RVTJ·S2OUT 은 66. SCRTEMP 는 16,376 B 레코드 4,966 개 = 헤더 2 + 68 상태 × 73(⌈1645·90/2047⌉) — 상태 번호는 이름이 아니라 §4 G-READ(b) ED 항등으로 확정한다.
3. **깊이 referent**(S2PRRR d31–40 블록 저자 조회): d33 v 22,127 km/s·T 9,416 K·ED 1.4938e7·DHYD 1.8491e6·α_C 2.7307e-12; d34 v 21,534·T 9,244·ED 1.6334e7·DHYD 2.2358e6·α_C 2.7361e-12. sh24 = 21,372–22,100(중앙 21,736; geometry.csv). ⚠S2OUT(66) 5열 N(S II)/ΣDHYD = 1.108e-4(d33)·1.874e-4(d34) — **jnu4 상태의 황은 S III 로 거의 전부 이온화돼 있다**(공개 21,630 km/s 행 f_II/f_III=0.5726/0.4145 와 10⁴ 차). 이 상태는 발산 중인 LAMBDA 반복(it67 NaN 직전, IN_ITS) 이므로 x_C 는 평형값이 아니다 — 분기에 쓰지 않는다(astra 3).
4. **Lumina 측 referent**(run2, 391837 계보): sh24 epoch 6 = `lumina_ion_channels.csv` gen 146 행: T_e 9231.3, n_e 1.491381653e7, Γ 2.824526205e-2, n_eα_rr 3.818812176e-5, n_eα_dr 5.679597350e-5, c_ion 5.886e-15, c_rec 4.517e-16. `lumina_xshell_response.json`: J_prod6 = bin 오프셋 3,455,240 부터 50×1234 LE binary64, partition6 5.115445227083729, `gamma_prod6_explicit` = census(상대 1e-6). σ 표 `run2/input/model/cmfgen_sigma_bf.bin`(magic 0x434d4644 v1, 24,542 준위 × 1,234 빈, ν 5.8412785919616062e13–4.0362581455823112e16, has_sigma 24,395; S II 전역 536–857 중 315 행 보유·7 행 Kramers). levels.csv 행순 = 전역 준위 index(536 = S II 바닥 E=0 g=4). σ 원천 = jnu4 와 동일 `SUL/II/19apr23/phot_data_A`(두 atomic_links 19행) ⟹ 차이는 표화·적분·가중이지 자료 vintage 가 아니다.
5. **생산 Γ 경로(코드 확인)**: `parity_gamma_phot_explicit_field`(lumina_plasma.c:3254) → `parity_gamma_phot_checked_impl`(:3139): 준위 l 마다 ν_th=(χ−E_l)/h, f_l=g_l e^{−(E_l−E_0)/kT}/Z (population_contract.c `population_lte_level_fraction`, x≥745 → 0), σ 행 있으면 그 행·없으면 σ_0=`get_bf_sigma0(16,1)`=2.00e-18 Kramers(ν_th/ν_c)³; `nlte_bf_gamma_canonical` → `bf_rate_gamma_legacy_grid`(bf_rate_jnu.c:131): 1234 log-빈마다 계단 σ(문턱 걸친 빈: 표 행은 `sg·(hi−lo)/(hi−ν_th)`, Kramers 는 `σ_0(1−x³)/(3 ln(hi/ν_th))`) → `bf_rate_gamma_from_view`: canonical 3,866 빈(radiation_field.h: j∈[−1398,2468), Δlog = log(ν_max/ν_min)/2468) 마다 K_b=∫σ/(hν)dν(ν≥ν_th 부터, 선형 구간 해석적) 하고 Γ=4πΣ_b J_b K_b, VALID/EXACT_ZERO 만 합, w_miss≤1e-3(bf_rate_jnu.h:51) 아니면 STALE/UNSAMPLED/OOG. canonical J 는 1234 빈의 정확한 2-분할 재빈(radiation_field.c:457–540; 밖은 OUT_OF_GRID). Γ_ion = Σ_l f_l Γ_l. ⟹ 도구는 이 규약을 **그대로** 옮긴다(새 근사 적분 금지). ⚠09-17 `rebalance_check.py` 는 Kramers σ_0=7.91e-18/(q+1)² 와 빈 중심 1/ν 를 써 생산과 다르다(장부 §2 재현비 S II 0.953, Si II 1.22 는 그 도구의 값).

## §1. 물리 계약 1개
**질문:** 「jnu4 it66 의 J_C(EDDFACTOR)·SCRTEMP 65 상태에서, Lumina 생산 율 연산자가 낸 S II 광이온화율 Γ_L(J_C) 와 Milne α_rr,L 이 CMFGEN 자신의 Γ_C,eff(=ΣPR_l/N_C(S II))·α_C(S2PRRR 계수 행)를 d33·d34 에서 τ=1.5 안에 재현하는가?」
**말하지 않는 것:** J 재현(전달 반쪽, 2단계) · x_C 평형(§0-3) · 어느 코드가 옳은가 · DR 의 정당성(CMFGEN 에 없는 채널, §2) · 공개 수렴해에 대한 결론.

## §2. 측정 정의 (스크립트 1개 `scripts/fixedpoint_rate_half.py`, repo; grammar-debug CPU; C 변경 0·GPU 0·CMFGEN 재실행 0)
- **입력**: EDDFACTOR(+_INFO; 판독 = extract_jnu.py `read_eddfactor`) · SCRTEMP(판독 = ptf_gated_metrics.py `ScrtempReader`, NT=1645) · S2PRRR · RVTJ · S2OUT · S2_F_TO_S · run2 σ 표·levels.csv·xshell_response.bin/json·ion_channels.csv. 모두 읽기만.
- **J_C→빈**: 깊이 d 의 J_ν(166,152 점, 주파수 내림차순)를 1234 log-빈에 **선형-주파수 보존 평균**(빈 안 사다리꼴, 빈 경계 절단) → canonical 2-분할(§0-5). EDDFACTOR 미포함 빈은 OUT_OF_GRID.
- **Γ_L(J_C; w_LTE)**: §0-5 규약, T=CMFGEN T(d), Z=Σ_l g_l e^{−E_l/kT}(S II 322 준위). **Γ_L,g** = 바닥 준위(536) 단독 Γ_l. **Γ_L(J_C; w_C)**: 같은 Γ_l 을 CMFGEN 가중으로 합 — SL 인구(SCRTEMP 65)를 F_TO_S(이름·에너지 일치, 미대응 0 강제 fail-closed) 로 전준위에 SL 내 Boltzmann 분배(SUP_TO_FULL 정의).
- **CMFGEN 측**: Γ_C,eff = Σ_l PR_l / N_C(S II)(N_C = Σ SL 인구, 65), Γ_C,g = PR_1/n_SL1, α_C = 계수 행(= ΣRR/(ED·DHYD)); α_C,tot = ΣRR/(ED·N_65(S III)) 도 기록(DHYD 귀속은 G-READ(c)).
- **비**: R_Γ = Γ_L(w_LTE)/Γ_C,eff · **R_σ** = Γ_L(w_C)/Γ_C,eff(σ·적분만) · **R_w** = Γ_L(w_LTE)/Γ_L(w_C)(가중만) · R_Γ,g = Γ_L,g/Γ_C,g · **R_α,rr** = α_C/α_rr,L, α_rr,L = (n_eα_rr/n_e)_{gen146} · R_α,tot = α_C/(α_rr,L+α_dr,L)(기록만; CMFGEN 에 S II DR 이 없으므로 분기에 안 쓴다 — 구성상 예정된 값을 분기에 넣지 않는다).
- **원시값**: β_C = ΣPR/ΣRR 및 (ΣPR+CPR)/(ΣRR+CRR) · x_C(65)=N(S III)/N(S II), x_C(66) · K_rr = Γ_L(w_LTE)/(ED·α_rr,L·x_C(65)), K_tot(α_rr+α_dr) · 준위별 표(322 행: E, g, has_σ, f_LTE, f_C, Γ_l, PR_l/n_l).
- **깊이**: d33·d34 각각 계산(보간 없음); sh24 값 = 두 비의 log-선형 v-보간(가중 0.659 → d34), 쌍 퍼짐 ε_d=|ln R(d33)/R(d34)|. 50 셸 표 = 90 깊이 R 의 v-보간(부산물).
- **잣대(발주서 추가 요구)**: 같은 코드 경로를 J_prod6(sh24)·T_e 9231.3·partition6 에 적용 → δ_yard = |ln Γ_py/0.028245262051479193|.
- **재빈 오차 실측**: Γ_fine = 같은 계단 σ·같은 가중으로 CMFGEN 미세 격자의 J(구간 선형)·정확한 1/ν·문턱 절단을 해석 적분 → ε_bin = |ln Γ_bin/Γ_fine|(eff·g, d33·d34). 측정의 잣대이지 측정값이 아니다.
- clamp·floor·부호 교정 없음. 유한 검사만.

## §3. 변경집합과 산출물
### 변경집합 — DET-CMFGEN-FIXEDPOINT-S1 (표)
| 경로 | 분류 | 허용 변경 |
|---|---|---|
| `scripts/fixedpoint_rate_half.py` | 오프라인 판독·적분·판정 | 신규. 심볼 `FIXEDPOINT_S1`. NC 플래그 3종 내장 |
### 변경집합 끝
`expected_extra` = 위 1파일. 변경 금지: `src/*`·동결 트리·봉인 입력·jnu4(읽기만). 산출: `fp_s1_result.json`(전 스칼라·게이트·분기·입력 해시) · `fp_s1_levels.csv`(322×2 깊이) · `fp_s1_shells.csv`(50 셸). 런 루트 `/gpfs/kjhan/lumina/cmfgen_map/fixedpoint_20260924/s1/`.

## §4. 게이트와 음성대조 (허용오차: 산술 항등 상대 1e-3, 물리 재현 τ=1.5)
- **G-YARD**: δ_yard ≤ 1e-3 ∧ 전 준위 상태 VALID/EXACT_ZERO(w_miss ≤1e-3). 막는 실사고: 생산과 다른 적분기를 생산이라 부르는 것(§0-5 ⚠).
- **G-READ**(판독 무결): (a) 파싱한 PR·RR·CPR·CRR 로 「Net Recombination Rate」행과 α_C 행을 전 90 깊이 1e-3 재현; (b) SCRTEMP 상태 s 의 ED(변수 NT−1) = S2PRRR ED 행(→ s≡65), 다음 상태 ED = RVTJ 「Electron density」(→ 66), 둘 다 전 깊이 1e-3, 두 상태가 파일의 마지막 둘; (c) DHYD(d) 가 N_65(S III) 총량 또는 SL1 인구 중 하나와 1e-3 일치(어느 쪽인지 기록); (d) EDDFACTOR FINISH_REC≠0·전 깊이 유한, S II 문턱 이상 빈 coverage 100 %.
- **G-NOISE**(τ 지지): ε_bin ≤ 0.10 ∧ ε_d ≤ 0.203(=ln τ/2) — R_Γ·R_α,rr 각각. 초과 = referent·재빈이 τ 를 못 받침.
- **G-FIN**: 전 산출 유한, N_C(S II)>0, 미대응 준위 0.
| NC | 주입(스크립트 플래그) | 반드시 FAIL |
|---|---|---|
| NC-YARD | Kramers σ_0 를 7.91e-18/4 로 교체 또는 partition ×1.01 | G-YARD |
| NC-STATE | SCRTEMP 상태를 한 칸 앞(64)으로 읽음 | G-READ(b) |
| NC-PRRR | 파싱한 RR 행 ×1.01 | G-READ(a) |

## §5. 기대치·분기
**τ=1.5 근거**: 설명할 간극은 x 10²–10⁴(§0-3·Lumina x_map 297); τ 는 그 최소의 1/60 이하. 잡음 = δ_yard(≤1e-3)+ε_bin(≤0.10)+ε_d(≤0.203)+T 불일치(d33 9,416/d34 9,244 대 α_rr,L 의 9,231.3 — CMFGEN 자신의 d33/d34 α_C 차 0.2 % 가 그 감도의 실측). 셋을 G-NOISE 가 실측하므로 합쳐도 ln τ=0.405 아래일 때만 분기를 읽는다.
**런 없이 얻을 수 없는 수**: 없음 — 이 단의 산출(R_Γ·R_σ·R_w·R_Γ,g·R_α,rr·β_C·K·ε_bin·ε_d·δ_yard) 이 곧 그 오프라인 계산이다. **구성상 참 등록 없음**: 위 어느 비도 저자가 계산하지 않았다(§0 의 행 값은 인용). 단 R_α,tot 는 CMFGEN 에 DR 이 없어 (α_rr+α_dr)/α_rr 만큼 예정된 이탈이 있으므로 분기 밖(기록).
| 분기 | 판정식(sh24 보간값; d33·d34 개별값 병기) | 뜻 | 다음 결정 |
|---|---|---|---|
| **R-A** | 게이트 전부 PASS ∧ R_Γ∈[1/τ,τ] ∧ R_α,rr∈[1/τ,τ] | 율 결백 — 간극은 J(전달) 쪽 | 2단계(전달 반쪽) 기안(Fable). 주입 상태 선택은 §0-3 을 근거로 user 결정 |
| **R-B** | 게이트 전부 PASS ∧ (R_Γ∉ ∨ R_α,rr∉) | 율 쪽 — R_σ(σ·적분)·R_w(가중)·R_α 중 τ 밖인 것과 크기를 지목, 준위별 표로 위치 | 지목 요소 단일 수리 계약(user). 2단계 보류 |
| **R-X** | 게이트 FAIL 하나라도 | 미시연 | 계측만 수리, 같은 계약 재실행 |

## §6. 자원·분장·주장 상한·집행 기록
- grammar-debug CPU 분 단위(EDDFACTOR 142 MB 1회 통독·SCRTEMP 2 상태). 봉인·jnu4 무접촉(읽기). 프리플라이트 **합성 root 필요**: docs·CLAUDE.md·validation·**scripts → repo**, src → 동결 트리(기존 `preflight_root` 는 scripts 가 동결 트리를 가리켜 `ptf_gated_metrics.py` 가 없다 — 재구성 요).
- 분장(개정25-n): 기안 Fable fresh **완료**(이 본문) · 기계 프리플라이트·물리대수 사전감사·종료감사 astra · Q-GOAL/Q-LEAN grok · 구현 Opus 5.5 · 코드 감사 Fable fresh(⚠자기 계약 편향 기재) · 발주·집행·판독·판정 운전자 — 실제 열은 집행 시 기입.
- 주장 상한: 「jnu4 it66 입력 상태(65)·J_C 에서 Lumina 생산 율 연산자는 CMFGEN 의 S II Γ 를 R_Γ 배(σ·적분 R_σ, 가중 R_w, 바닥 R_Γ,g), α_rr 을 R_α,rr 배로 냈다; 그 상태의 β_C·x_C·K 는 원시값이다.」 확장 않음: J 재현, 평형, 공개 해, DR 정당성, 다른 이온·깊이의 일반화.
- 집행 기록 뼈대: 1 기안 · 2 프리플라이트 rc(합성 root 경로) · 3 사전감사 · 4 구현·NC 3종 시연 · 5 코드 감사 · 6 실행(노드·입력 해시·δ_yard·ε_bin·ε_d) · 7 판독(G-YARD/READ/NOISE/FIN·비 전부·분기) · 8 판정·종료감사.
- **집행 1** 기안: Fable `claude-fable-5-1`, 2026-09-24; 개정 1(Fable OLD/NEW, references 의 미존재 신규 파일 1행 삭제).
- **집행 2** 프리플라이트(운전자, grammar-debug, 2026-09-24): 합성 root `/gpfs/kjhan/lumina/cmfgen_map/fixedpoint_20260924/s1_preflight_root`(docs·CLAUDE.md·validation·scripts→repo, src→동결 트리). 1차 prereg FAIL(reference-missing-path `scripts/fixedpoint_rate_half.py`) → 개정 1 → prereg·referents·canonical **전건 rc=0**(changeset files=1·residual=R-X·refs=11).

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-CMFGEN-FIXEDPOINT-S1 (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/[A-Za-z0-9_/.]+\\.py)",
    "symbol": "FIXEDPOINT_S1",
    "roots": ["scripts"],
    "expected_extra": ["scripts/fixedpoint_rate_half.py"]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    "metrics": {"count": "len(v)", "gate_fail": "sum(1 for x in v if x < 0.5)", "rate_clean": "sum(1 for x in v if 0.5 <= x < 1.5)", "rate_side": "sum(1 for x in v if x >= 1.5)"},
    "rules": [
      {"name": "R-X", "predicate": "count == 0 or gate_fail > 0"},
      {"name": "R-A", "predicate": "count > 0 and gate_fail == 0 and rate_clean == count"},
      {"name": "R-B", "predicate": "count > 0 and gate_fail == 0 and rate_side == count"}
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []}, {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 1]]},
      {"name": "rate-clean", "mix": [[1.0, 1]]}, {"name": "rate-side", "mix": [[2.0, 1]]}
    ],
    "residual": "R-X"
  },
  "references": [
    {"path": "docs/PREREG_DET_CMFGEN_FIXEDPOINT_2026-09-24.md"},
    {"path": "docs/READOUT_IMPLIED_J_LEDGER_2026-09-17.md"},
    {"path": "docs/VERDICT_DET_PATHB_XSHELL_2026-09-24.md"},
    {"path": "docs/PREREG_DET_PATHB_XSHELL_2026-09-22.md"},
    {"path": "CLAUDE.md"},
    {"path": "validation/cmfgen_toy06_19p48d/analysis/extract_jnu.py", "flags_existing": ["read_eddfactor"]},
    {"path": "scripts/ptf_gated_metrics.py", "flags_existing": ["ScrtempReader", "DOUBLES_PER_REC"]},
    {"path": "src/lumina_plasma.c", "flags_existing": ["parity_gamma_phot_explicit_field", "parity_gamma_phot_checked_impl", "nlte_bf_gamma_canonical", "get_bf_sigma0"]},
    {"path": "src/bf_rate_jnu.c", "flags_existing": ["bf_rate_gamma_legacy_grid", "bf_rate_gamma_from_view", "segment_sigma_over_nu"]},
    {"path": "src/radiation_field.h", "flags_existing": ["LUMINA_RADFIELD_J_LO", "LUMINA_RADFIELD_REFINEMENT_K"]},
    {"path": "src/population_contract.c", "flags_existing": ["population_lte_level_fraction"]}
  ]
}
```
