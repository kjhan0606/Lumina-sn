# 운전자 기획 브리프 — 연속체 방출률을 CMFGEN n_i* 형으로 재정의 (2026-09-19, Fable 계약 기안용)

운전자(Claude) 작성. Fable 이 이 증거로 사전등록(계약)을 기안한다(개정25-b). 계약 아님. 표적 명: **DET-CONT-NISTAR** (CONT 를 CMFGEN 방식으로 다시).

## 배경 (실측)
- **CONT 실패**(취소, `docs/READOUT_CONT_NLTE_SUPERTHERMAL_2026-09-18.md`): 연속체 소스를 NLTE 로 바꾸자 초열 재점화(S_bf/B 1e79 인구역전)로 황을 반대 방향 과이온화.
- **근인 규명**(user 질문 → CMFGEN 소스 정찰, `reference_cmfgen_bf_emissivity_lte_referenced.md`): CMFGEN 은 클램프가 아니라 **방출률 정의**로 유계를 지킨다.
  - **CMFGEN**(`/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/genopaeta_v10.f:266-268`): `η = (2hν³/c²)·α·n_i*`, `χ = α·(n_l − n_i*·e^{-hν/kT})`. **n_i\* = LTE Saha 재결합 인구**(실제 다음-이온 밀도 + 가스 T_e). η ∝ n_i\*(고정) ⟹ 역전 불가·열적 유계. `η/χ→B_ν` LTE 에서 구성상 성립(`comp_opac.f:219-221`).
  - **Lumina**(`.../lteunmap_eps_20260917/source/src/lumina_plasma.c` 8047-8617): `S_bf = 2hν³/c²·exp(-x)/Cinv_l`, `Cinv_l = saha_m·n_level/g`(8412-8417). **분모가 고갈되는 실제 준위 인구 n_level 에 비례** ⟹ 과이온화로 n_level→0 이면 S_bf→∞(레이징). 머리 주석 8047-8050 이 `b_l e^x−1` 형의 레이징을 스스로 경고.

## 표적 (기획, user 2026-09-19 「해당 내용을 코드에 반영해」)
**Lumina 연속체 bf 방출률/불투명도 쌍을 CMFGEN 의 LTE-기준 n_i\* 형으로 재정의**한다. NLTE 성질은 방출률 안(고갈되는 n_level)이 아니라 **실제 이온 밀도 DI(NLTE)를 통해서만** 들어오게 한다. 클램프/floor 아님(규약 판별식: 정확해가 위반 가능한 가드는 잘못 — 여기선 **정의 교체**).

## 수리안 방향 (구현자 Opus 4.8 이 상세, 여기선 골격)
- 현행 `S_bf ∝ 1/Cinv_l ∝ 1/n_level` 를 **`η_bf = (2hν³/c²)·σ_bf·n_i*·e^{-h(ν−ν_edge)/kTe}`** 로 바꾼다. `n_i* = SAHACONST·(g_l/g_up)·T_e^{-1.5}·n_next·n_e·e^{(χ−E_l)/kTe}` 형(실제 다음-이온 밀도 n_next, 가스 T_e). 인프라 존재: `SAHACONST=2.0706659e-16`(2429), `saha_m`(8416), `bf_get_chi`.
- 유도방출 항(χ)도 **같은 n_i\* 기준**으로: `χ_bf = σ_bf·(n_l − n_i*·e^{-hν/kTe})`. ⟹ `η/χ→B` 구성상 성립·`b·e^x−1`·`1/n_level` 분모 소멸.
- 게이트 `LUMINA_CMF_BF_MILNE` 계열과의 관계·byte-동일(OFF)·기존 EPAY/OTS 경로 보존은 구현자가 정리.

## 오프라인 검증 (수리안 오프라인 검증 — ★CONT 교훈, 필수)
CONT 는 오프라인(퇴화 극한)이 비선형 초열 불안정을 못 봤다. 이번엔 **재정의가 CONT 폭주 조건에서 유계임을 런 전에 수치로 시연**한다:
- CONT run1 덤프의 폭주 빈(S_bf/B 1e79)의 실제 n_level·n_next·n_e·T_e·σ_bf 를 뽑아, **n_i\* 형으로 S_bf 를 재계산 ⟹ O(1) 유계**(밴드 0.1–10)임을 보인다. selftest 로 박제(인구 고갈 픽스처에서 옛 형=발산·새 형=유계).

## 기대치 (사전등록 후보)
- **G-BND(핵심)**: 재정의 런의 조립기 census `S_bf_over_B_mean`·`_max` 가 전 셸·전 패스 **밴드 0.1–10 유지**(CONT 는 1e79 였음 — 음성대조는 CONT 덤프).
- **E-1 황 이동**: 중간 셸 S f_II 가 **CMFGEN(.85/.80/.71) 방향으로** 이동(CONT 는 반대 .10→.004 로 갔음). 정확 착지값은 오프라인 예측 후 문턱(Fable·구현자).
- **E-2 외곽 유지·E-3 내핵 상한**: 램프 재점화·부작용 없음.
- 런 구성: **DR-on**(DET-DR-ON 이 세운 깨끗한 재결합 base) 위에서. ITER 수는 결정 항목.

## 감사·검증 순서
경량 사전감사(Opus 4.8, Q-GOAL·Q-LEAN) → 게이트 4종 → Opus 4.8 구현(+selftest) → Fable 코드 감사 → 빌드 → 오프라인 유계 시연 → 판정런 → 판정 → 종료감사.
⚠**src 편집 태스크 1개 규칙**: DET-DR-ON 런(385315) 착지·폐합 후 착수(또는 별 트리에서 구현하되 발주는 DR-on 폐합 후).
⚠**보조(이번 계약 밖 후보 기재)**: CMFGEN 의 완전 선형화(∂J/∂N 을 SE 야코비안에 접음)·음-불투명도 처리는 더 깊은 수리 — 이번은 **방출률 정의 교체 하나**에 한정(Q-LEAN).

## 주장 상한
방출률 정의 교체가 초열 폭주를 제거하고 황을 CMFGEN 방향으로 옮기는가까지. 완전 선형화·자유-T_e·스펙트럼·정량 착지값은 말하지 않는다. n_i\* 형의 in-vivo 유계는 런이 최종 확정(오프라인은 폭주 조건 시연까지).
