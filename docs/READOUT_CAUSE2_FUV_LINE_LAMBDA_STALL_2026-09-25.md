# READOUT — 2단계: λ<2000 Å 장의 원천 — 두꺼운 선 소스 함수의 Λ-정체 (2026-09-25, 런 0회)

- 선행: `docs/READOUT_CAUSE1_SII_IONIZATION_CHANNEL_2026-09-25.md`(확증: S II 과이온화의 직접 율 기여 = 명시 들뜬 준위 × λ<2000 Å 장). user 순서 「1→2→3, 원인이 밝혀지면 확증 후 다음」.
- 판독자: 운전자. 입력(읽기 전용): run2 `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/run2`, 398939 `/gpfs/kjhan/lumina/cmfgen_map/pathb_xshell_20260922/run2`, 동결 트리 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src`. 스크립트·산출 `/gpfs/kjhan/lumina/cmfgen_map/coldstart_20260924/cause2/`(source_bands · source_bands_iters · fuv_lines_beta · lambda_stall — .py/.txt).

## 1. λ<2000 Å 장은 선 방출이 채운다 (`source_bands.txt`)

physics_comparison iter1 spectral 덤프, 대역 적분: 912–1400·1400–2000 Å 에서 η_true 의 99–100 % 가 bb(선), χ 의 53–100 % 가 bb; <912 Å 는 χ 100 % bf·η 93–100 % bb. 대역 S_true/B(= Ση_true/Σχ_abs·B): run2 it1 sh15/19/24 의 912–2000 Å 0.82–1.54, 398939 it1 1.0–2.3. 398939 반복 0–5 동안 FUV 대역 S_true/B 는 거의 제자리(sh24 912–1400 Å 1.62 → 1.38, 1400–2000 Å 1.13 → 0.81; `source_bands_iters.txt`).

## 2. 코드 경로 (정적)

- **fine 조립기**(lumina_cmfgen.c:2094–2160): 선 방출 η_l = w·el·S_l, S_l = `opac->line_source_S`(NLTE 인구 소스; ≤0 이면 B 폴백). **mapped 선**(NLTE II–III 이온) el = 1(「unchanged fully-thermal treatment; its NLTE-source eps channel is a separate rung」 — 주석, :160–170): 불투명도 전부를 흡수로·방출은 NLTE S_l. unmapped(IV+) = ε 2준위.
- **준위 율 행렬의 bb 율**(lumina_plasma.c:19088–19110, 생산 분기 「A2-06 view」): R_absorb = B_lu·J̄, R_stim = B_ul·J̄, R_spont = A_ul — **β(탈출 확률)·근사 Λ 연산자(ALI) 없음**. J̄ = `nlte_bb_jbar_canonical`(:585) 의 선-분해 CMF view(`LUMINA_CMF_LINERES_JBAR=2`) — 그 선 자신의 방출(직전 인구의 S_l)을 포함하는 장. `LUMINA_MALI` 미설정(β 곱은 구 분기에만 있고 생산 분기에서 덮어씀).
- ⟹ 두꺼운 선(τ ≫ 1)에 대해 인구 갱신은 **순수 Λ-반복**: J̄ ≈ (1−β)S_l^(k) + β J_ext, 반복당 교정 ∝ β.

## 3. 바깥 셸의 FUV 선은 Si/S II–III 자신의 선이며 초열적 (`fuv_lines_beta.txt`)

398939 최종 NLTE 인구(shells ≥ 16 은 Si/S/Ca 만), mapped 선 900–2000 Å, τ_Sobolev > 10⁻³, 가중 1−e^{−τ}: sh24 — 불투명도 가중 S_l/B = **1.82**(S III 2.91, S II 2.04, Si III 0.97, Si II 0.92), β < 10⁻² 인 몫 39 %(β < 10⁻³ 20 %); sh15 — 1.55, β < 10⁻² 54 %(β < 10⁻³ 35 %).

## 4. Λ-정체의 직접 증거 (`lambda_stall.txt`)

run2 준위 증인(S II·III 전 준위, 커밋 gen 2–8), S II/III FUV 선 2,092 개(900–2000 Å), β 계급별 가중 S_l/B:

| 셸 | 계급 | gen2 | gen3 | gen4 | gen5(해제) | gen6 | gen7 | gen8 |
|---|---|---|---|---|---|---|---|---|
| sh24 | 두꺼운(β<10⁻²) | 0.982 | 3.20 | 3.21 | 3.20 | 3.18 | 3.16 | 3.18 |
| sh24 | 중간 | 0.984 | 3.26 | 4.61 | 7.36 | 5.28 | 4.48 | 3.86 |
| sh24 | 얇은(β>0.5) | 0.936 | 2.59 | 3.01 | 4.97 | 2.78 | 2.13 | 1.76 |
| sh15 | 두꺼운 | 1.01 | 2.19 | 2.19 | 2.18 | 2.18 | 2.17 | 2.17 |
| sh15 | 중간 | 0.947 | 2.00 | 1.92 | 2.06 | 2.15 | 2.12 | 2.04 |
| sh15 | 얇은 | 0.811 | 1.30 | 1.18 | 1.72 | 1.44 | 1.29 | 1.23 |

⟹ 두꺼운 선의 S_l/B 는 gen 3–8 동안 **1 % 안에서 고정** — 해제(gen 5, S II 단계 총량 1.40e6 → 5.5e4·n_e 1.32e7 → 2.11e7, sh24)에도 무반응. 얇은 선은 같은 사건에 즉시 반응(음성대조). 이것은 §2 의 Λ-반복 구조(β 없는 bb 율 + 자기 방출 포함 J̄)가 예측하는 서명이다.

## 5. 결론 (2단계 원천, 주장 상한)

- **말하는 것**: sh15·24 의 λ<2000 Å 장을 채우는 FUV 선 중 두꺼운 선(불투명도의 39–54 %)은 소스 함수가 반복 사이에 고정된다(Λ-정체) — 코드의 bb 율이 β/ALI 없이 자기 방출 포함 J̄ 를 쓰기 때문과 정합. 고정값이 B_ν(T_e) 의 2–3 배(초열적)라 FUV 장을 열적 이상으로 떠받치고, 1단계에서 확증한 들뜬 준위 이온화 채널의 장이 된다. 이 고정은 S 의 이온 상태 변화에 반응하지 않는다 ⟹ 차가운 출발이 한 반복에 버려지는 기전의 장 쪽 원천 후보.
- **말하지 않는 것**: gen 2 → 3 의 1 → 2–3 점프 기전(첫 J 해가 무엇을 가져왔는가) · 고정값이 초열적인 이유(재결합 폭포·선 겹침·초기 장) · IGE 셸(sh ≤ 15 의 Fe/Co/Ni)·IV+ unmapped 선의 몫 · CMFGEN 의 실제 선 소스 · ALI 를 넣으면 어디로 수렴하는가(시험 필요) · Si II·III 선의 반복 궤적(준위 증인은 S 만).
- 다음(확증 후): 3단계 전에 이 원천을 제거·검증하는 수리안 — bb 율의 ALI/Sobolev 순 복사 괄호(net radiative bracket)·선형화 — 을 사전등록으로 설계(CMFGEN 발자국: 선형화·ALI).
