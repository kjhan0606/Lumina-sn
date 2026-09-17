# 판독 — 황 0.395 의 정체: 「S 문제」가 아니라 「EUV 장 = B_ν(T_e)」 (2026-09-17, 런 0회)

> ⚠ **정정 (같은 날 후속, `READOUT_IMPLIED_J_LEDGER_2026-09-17.md` §1-1)**: §2 의 「LTE Saha(T_e) 1 % 일치」는 f_III 포화 자리의 착시다 — 비(比)로는 sh12 2.6×·sh20 48×·sh32 3.8e3× LTE 위. §3·부록 A-1 의 「EUV J = B(T_e) 채움」은 **불투명 빈에서만** 참이며, σ-가중 Γ 를 지배하는 창 빈(557.7/696.7/922.5 Å …)은 J/B = 10³–10⁶ 이다. 원인은 외곽 고온층의 S V·Ca V LTE 선 방출(S_l = B(T_e) 폴백)이 EUV 창으로 안쪽을 비추는 것으로 특정됐다. 이 문서의 §1(지표 착시)·§4·A-2·A-3 은 유지.


저자 = 운전자. **판독이지 판정이 아니다.** 전건 `[저자 실행]`, grammar-debug, 생산 코드 미사용(csv+Decimal/float 직접).
도구 = `/gpfs/kjhan/lumina/cmfgen_map/repro_claude_20260917/readout_{S,thresh,lte,J,Jcmf,tau}.py` (`READOUT_TOOLS.sha256`), 출력 `*.stdout`.
입력 = 09-14 비교의 baseline 지도(`consistent_ionization_state_20260911/retry_20260912/physical_evidence/result/divergence_map.csv`) ·
그 런의 `physics_DET_iter000{0,1}.{shell,spectral}.csv`(DET/FIXED_T, COMMITTED) · 덱 `ionization_energies.csv` ·
CMFGEN J 표 `data/cmfgen_jtable_toy06_19p48d.bin`(7월 jnu4 자체런 — ⚠대장이 이온화 오라클 부적격으로 기록; J 잣대로만 사용).

## 1. 지표의 착시 — S 만 튄 것이 아니다

`E_e` 는 **q=1,2 만** 세고 **질량가중**이다. 그래서 II→III 전이가 질량 무거운 중간 셸(9–10 kK)에 놓인 **황만** 걸렸다.
실제로는 **전 원소가 바깥일수록 한 단계씩 과이온화**(Saha 권위·NLTE 권위 동일 — 이중 권위 간극 아님):

| 셸(T_pub) | CMFGEN → Lumina |
|---|---|
| S II/III, sh10–24 (9–10 kK) | S II **0.80–0.85 → 0.00–0.02**, S III 0.15 → 0.98 |
| Si III/IV, sh28 (9.7 kK) | Si III 0.80 → 0.04, **Si IV 0.08 → 0.96** |
| Ca III/IV/V, sh45 (19 kK) | Ca V 0.12 → **0.67** |
| Fe/Co/Ni, sh0 (19 kK, 내곽) | IV 0.98 → 0.98 — **맞음** |

E_e 기여 상위 8셸(10–16, 9)이 S E_e 의 72 %. S II Δf<0 인 셸 44/46.

## 2. 판별 1 — 중간 셸의 Lumina 이온화 = **LTE Saha(T_e)**

바닥항 통계무게로 LTE Saha(T_pub, n_e)를 셸별 계산(조야한 분배함수 — 계수 100 판별엔 충분):

| sh | T | n_e | f_III: CMFGEN / **Lumina** / LTE(T_e) |
|---|---|---|---|
| 9 | 10114 | 6.2e8 | 0.493 / **0.968** / 0.978 |
| 12 | 10138 | 3.1e8 | 0.154 / **0.996** / 0.990 |
| 20 | 8849 | 4.1e7 | 0.284 / **0.998** / 0.923 |

⟹ sh9–24 전건 Lumina ≈ LTE(T_e), CMFGEN 은 훨씬 아래. 외곽 sh28–36 의 III→IV 는 Lumina 가 **LTE 보다 100–1000배 더** 이온화(둘째 국면).

## 3. 판별 2 — 그 런의 EUV J 는 국소 B_ν(T_e) 다

`physics_DET_iter0001.spectral.csv` 에서 문턱 주파수의 J 를 뽑아 Planck 와 비교:

| 문턱 | sh9 | sh12 | sh20 | sh28 | sh33 | sh36 | sh40 | sh45 |
|---|---|---|---|---|---|---|---|---|
| **532 Å** J/B(T_e) | 0.98 | 0.98 | 0.98 | 159 | 145 | 69 | 1.6 | 1e-3 |
| **370 Å** J/B(T_e) | 1.07 | 1.07 | 1.09 | 1.08 | 1.07 | 1.07 | 804 | 2.5 |
| **356 Å** J/B(T_e) | 0.92 | 0.92 | 0.91 | 0.92 | 0.92 | 0.92 | 383 | 0.06 |

★**상수**(0.98 / 1.07 / 0.92)가 셸·T·τ 와 무관 — 빈 중심 대 문턱의 평가점 차이일 뿐인 **B(T_e) 그 자체**.

## 4. 판별 3 — 열화(확산 극한)가 아니다

370 Å 의 셸당 τ = χ_tot·Δr: sh5 **1.2e5** → sh12 1.1e3 → sh20 147 → sh28 6.4 → sh30 3.1 → **sh33 1.06 → sh36 0.52**.
J/B 는 그 다섯 자릿수 동안 **1.07 로 불변**. τ≈0.5–1 에서 J=B 는 확산 물리가 아니다.
iter0 J 는 셸 무관 상수 seed(532 Å 4.94e-12, 370 Å 1.19e-16) → iter1 에서 정확히 B(T_e)×상수로 이동
⟹ 형식해는 돌았으나 **EUV 소스함수가 LTE(S=B)** 여서 J≈B 가 나온 것으로 읽힌다(기전 후보; 소스 특정은 §7).
⚠이 계보의 `nlte_build_perbin_dilute_field` 는 희석-Planck 압축이 A2-17 에서 폐기(`#if 0`)돼 있으므로 **그 재적합은 원인이 아니다.**

## 5. 세부균형 — 판별 2 가 판별 1 을 함의한다

J=B_ν(T_e) 를 먹은 광이온화 Γ 는 Milne 재결합과 **정확히 세부균형** ⟹ Saha 층의 답 = LTE(T_e). 관측(§2) 그대로.
⟹ **중간 셸의 이온화는 수송된 장이 아니라 T_e 가 정한다.** 쌍둥이 레인은 T_e 를 CMFGEN 값으로 핀하므로
**「올바른 T 에서의 LTE」**가 되며, 지도의 개선(M_Z 0.097→0.031)은 그것이 상당 부분 만든 것이다.
Fable 09-09 B-③ 「전리 계산이 더 옳다고 말할 수 없다」의 실현.

## 6. CMFGEN J 표와의 정량 (⚠자체런 J, 잣대 캐비앳)

| | sh12 | sh20 | sh30 | sh36 | sh40 | sh45 |
|---|---|---|---|---|---|---|
| 532 Å J_L/J_C | 0.30 | **0.005** | 15 | 5.4 | 8.8 | 1.3 |
| 370 Å J_L/J_C | 0.03 | **3e-4** | 1e-3 | 1e-3 | 34 | **5e3** |
| 356 Å J_L/J_C | 0.13 | **2e-4** | 5e-3 | 0.03 | 1.8e3 | **2e4** |

★**모순처럼 보이는 것이 기전이다**: 중간 셸에서 Lumina 의 EUV J 는 CMFGEN 보다 **10–1000배 낮은데** Lumina 가 **더** 이온화돼 있다.
J 가 B(T_e) 면 이온화는 J 의 크기가 아니라 **T_e 가** 정하기 때문이다. CMFGEN 의 EUV 는 B(T_e) 의 10²–10³배인데도 덜
이온화 — CMFGEN 의 EUV 는 열화돼 있지 않고(S II 지배 → 다른 bf 문턱 → 다른 불투명도: **이온화 쌍안정 후보**), 희석된
비국소 장이 지배한다. 7월 `bistability_test` 와 연결.
외곽 sh40–49: Lumina J 가 CMFGEN 의 **10³–10⁴배** — 별개의 「외곽 EUV 과잉」.

## 7. 기전 특정으로 가는 다음 걸음 (런 0회)

1. DET/FIXED_T 레인의 **EUV 소스함수·J 구축 위치**를 소스에서 특정 — S=B(T_e) 가 어디서 오는가(LTE seed 유지? `NLTE_START_ITER=2` 와 2반복의 관계? bf 방출률 η_bf 의 LTE 구성?). `physics_DET` 의 `eta_bf`/`chi_bf` 열이 그 자리다.
2. 같은 런의 `lumina_c1_bins.csv`/`LUMINA_C2_BFR_DUMP` 류 raw J 덤프 유무 확인 — 재적합 전 raw MC J 와 위 J₁ 의 관계.
3. 반증 가능 시험(계약 후보): **EUV 빈의 J 를 CMFGEN 표로 치환(`LUMINA_GPH_JTABLE`, 7월 배선 실존)** 한 판정런 1회 — S II/III 가 CMFGEN 쪽으로 오면 장이 원인, 안 오면 율/자료. 단일 변수·기존 게이트.

## 8. 이 판독이 말하지 않는 것

CMFGEN 이 옳다는 것(잣대일 뿐) · 외곽 과잉의 원인 · 소스 행 단위 귀속 · 스펙트럼 영향의 크기.
**S 0.395 는 증상이고 병소는 EUV 장의 LTE 성**이라는 것이 이 판독의 상한이다.

---

## 부록 A (같은 날 후속, 런 0회) — 채움의 확정과 두 강등

### A-1. EUV J 는 **Planck 채움**이다 — iter0 과 iter1 이 같은 빈-중심 상수를 갖는다
`physics_DET_iter0000` 의 seed J 는 전 셸 균일 **B_ν(T_inner=10,020 K)**(532/370/300/150 Å 의 bin-center 비 = 0.988/1.085/1.004/0.925).
iter1 의 J/B(T_e) 는 셸 12–36 에서 **정확히 같은 상수**(0.98/1.07/1.0/0.93). ⟹ iter1 의 EUV J 는 **같은 Planck 평가 루틴이 T 만
T_inner → 국소 T_e 로 바꿔 채운 값**이다. 수송·확산 어느 쪽도 아니다. 셸 12–36 의 **150–532 Å 전 EUV** 가 그렇다.
CMFGEN J 표 대비 Lumina EUV = **10⁻⁴–10⁻²**(셸 20–36). 실제 장은 **sh≥40 에서만** 나타난다(J/B 1.4e3–1.6e5, CMFGEN 대비 10⁴–10¹⁰ — 별개 「외곽 EUV 과잉」).

### A-2. 강등 1 — §3 「S_bf/B ≫ 1」은 근거로 쓰지 않는다
`physics_DET` 의 manifest 에 population generation 키가 없어 `opacity`(χ) 와 `emissivity`(η) 가 **같은 세대**인지 알 수 없다.
세대가 다르면 η/χ 는 무의미하다(계약이 「동일 공개 세대 연결 미시연」으로 둔 구멍). ⟹ **관측으로만 남기고 결론에서 뺀다.**

### A-3. 강등 2 — 외곽 28–36 의 III→IV 과이온화는 **공개 장으로 설명되지 않는다**
그 셸의 J 는 전 EUV 에서 ≤ B(T_e) 이므로 세부균형상 LTE 를 넘을 수 없는데, 이온 분율은 LTE 비의 **~3,600배**(S IV sh30: 0.805 대 LTE 0.001).
⟹ 이온화 solve 가 **공개 J 가 아닌 다른 장/경로**를 먹었다는 뜻이다. 후보(미판별): 세대 지연(material phase 가 이전 세대 장 소비) ·
`LUMINA_CMF_BF_MILNE=2` 의 재결합 온도 · 성운형 Saha 의 (W, T_rad) 적합 · `g_photoion_mc_J` 는 DET 레인에서 **미장착**(로그에 `armed` 0건).
**중간 셸(황 0.395)의 기전과 외곽의 기전은 다르다.**

### A-4. 이 런의 구조(로그 실측)
DET 레인 · 2 외부반복 · iter0 물질 = `BOOTSTRAP_LTE_SAHA`(seed-T_e LTE/Saha, 루프 진입 전) · iter1 `R7_MATERIAL_PHASE_COMMITTED
phase=A2-10 te_generation=2→3`, `population_generation=4` · 장 commit = `lumina_cmfgen.c:3781`(provenance `CMFGEN_REPLAY`,
producer `PURE_CMFGEN_COMOVING_CONSERVATIVE_REBIN`) · Γ_phot = `nlte_bf_gamma_canonical` → `bf_rate_gamma_legacy_grid(&nlte->radfield_view)`.
CMF 조립에 **`S=B_ν(T_e)` 열화 폴백**(`CMF_LP_F_SL_FALLBACK`, `lumina_cmfgen.c:610/915/1867`)과 **`η_dep=κ·B_ν(T_e)` 주입**(:1799)이 존재 — 채움의 소스 후보이나 **행 단위 귀속은 미완**.

### A-5. 반증 계약의 전제 (다음 메시지에서 확인)
7월 `LUMINA_GPH_JTABLE` 이 A-2 이관 후에도 `radfield_view` 경로에 꽂혀 있는가. 아니면 치환 시험은 **view 에 주입하는 배선 1개**가 선행한다.

