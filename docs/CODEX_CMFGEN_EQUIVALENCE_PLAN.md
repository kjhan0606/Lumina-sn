# Lumina → CMFGEN-동등 수준 단계 계획

## 0. 결론

권고 노선은 **(c) 혼합형**이다.

- **결정론 CMF 수송 + 전 원소/전 이온단계 SE·전하·복사평형 연립계**를 물리 상태의 권위 있는 생산자로 승격한다.
- 기존 **MC/GPU는 macro-atom 재분배, packet fate, emergent 통계, 가속기 경로**로 유지한다.
- MC 피드백은 결정론 해와 operator/rate 수준 동등성이 입증된 뒤에만 권위 경로로 허용한다.
- 목표는 CMFGEN 포트란 클론이 아니라, **같은 물리 잔차·보존법칙을 풀어 CMFGEN 앵커와 같은 해를 내는 것**이다.

예상 총량은 약 **29–51 person-month**, 3–4인 팀 기준 **12–18개월**이다. 가장 큰 위험은 수송 단독 이식이 아니라 **CMF 복사장 응답과 전 준위 SE·전하·온도를 묶는 전역 비선형 해법**이다.

중요한 앵커 주의점이 하나 있다. `toy06_19p48d_modern`은 프로세스상 정상 완주했지만 현재 `RVTJ`는 `Was T fixed? T`이다([RVTJ:1–11](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/RVTJ:1)). 저장소의 검증문도 최종 열평형 acceptance에는 `FIX_T=F` 수렴본이 필요하다고 명시한다([README.md:135–142](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/README.md:135)). 따라서 현 완주본은 **동결 셀 율·Jν·인구 검증 앵커**로 쓸 수 있지만, `T_e`·열평형·최종 스펙트럼의 종국 acceptance에는 **released-T 앵커**를 하나 더 확보해야 한다.

또한 경로 명세상 `/gpfs/kjhan/cmfgen_21jun23/`는 원자자료이고, 실제 솔버 소스는 `/gpfs/kjhan/cmfgen_src/cur_cmf/`이다([CMFGEN_BUILD_RUN_GUIDE.md:9–16](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CMFGEN_BUILD_RUN_GUIDE.md:9)).

---

## 1. “CMFGEN-동등”의 정의

두 종류를 구분해야 한다.

- **E-동등(equation equivalence)**: 같은 원자모델을 투영한 뒤 SE, 전하보존, 복사평형, bf target, detailed balance의 잔차가 같은 방정식을 나타낸다. 수치 선형대수 구현은 달라도 된다.
- **S-동등(solution equivalence)**: CMF, ALI/VEF, MC, GPU 등 풀이 방법은 달라도 수렴한 `Jν`, 인구, `T_e`, `n_e`, 이온분율, 스펙트럼이 사전등록 문턱 안에서 같다.

SE·원자 위상·복사평형은 우선 **E-동등**이 필요하다. 수송·formal·MC는 같은 연속방정식과 경계조건을 보존하면 **S-동등**으로 인정할 수 있다.

### 공통 정량 계약안

Stage 0에서 고정하고 이후 결과에 맞춰 완화하지 않는다.

| 축 | 최종 문턱 |
|---|---|
| 보존/KA | 입자·전하 잔차 `≤10⁻¹⁰`; LTE detailed-balance 잔차 `≤10⁻⁸`; KA 에너지 오차 `≤10⁻³` |
| 동결 셀 율 | 활성 항의 `|log10(Lumina/CMFGEN)|`: median `≤0.03 dex`, p95 `≤0.10 dex`, max `≤0.20 dex` |
| 복사장 | `Jν` median `≤5%`, p95 `≤15%`; σ-weighted Γ 차이 `≤10%` |
| 상태 | `T_e` median/p95 `≤3/5%`; `n_e` `≤5/10%`; 지배 이온분율 절대차 `≤0.05`, 비율 `≤0.10 dex` |
| 준위인구 | 활성 준위 `b_k` 오차 median `≤0.05 dex`, p95 `≤0.15 dex` |
| formal | `L_out/(L_in+L_dep)` 오차 `≤1%`; CMFGEN 대비 `L_bol≤2%`, 주요 대역 flux `≤5%`, EW `≤10%`, 특징속도 `≤500 km/s` |
| 이산화 | 공간·주파수 해상도 2배 시 변화가 각 acceptance 폭의 절반 이하 |

비활성 수치의 log 오차가 판정을 오염하지 않도록 이온분율 `>10⁻⁸` 또는 해당 열/불투명도 기여 `>10⁻⁴`인 항만 주 지표로 삼고, 나머지는 coverage 표에 별도 기록한다.

---

## 2. 6대 구조 대조

| 축 | CMFGEN 실물 | Lumina 현행 | 거리와 동등 조건 |
|---|---|---|---|
| 전 준위 동시 SE 선형화 | 모든 종·이온의 STEQ/BA를 만들고 전하식을 더한 뒤([cmfgen_sub.f:1588–1680](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:1588)), block-banded 계를 푼다([solve_for_pops.f:1–8](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:1), [solveba_v13.f:90–104](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:90)). 온도를 포함한 전 변수를 한 번에 보정한다([solveba_v13.f:311–322](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:311)). | 인접 이온 pair를 셸별로 독립 조립·Gauss 풀이한다([lumina_plasma.c:15723–15763](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15723)). 기본은 5회·0.5 damping이며 element-wide matrix가 잔여라고 코드가 선언한다([lumina_plasma.c:16214–16240](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16214)); 공유 lo-ion은 뒤 pair 해를 저장·복원으로 폐기한다([lumina_plasma.c:16268–16313](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16268)). | **최대 거리. E-동등 필수.** 원소별 모든 포함 단계, 준위/SL, 전하 및 원소보존을 한 잔차계로 만들어야 한다. pair sweep 반복은 대체 증명이 되지 않는다. |
| ALI/VEF 기반 CMF 수송 | 생산 경로는 단순 “ALI 하나”가 아니다. CMF formal ray와 모멘트 방정식을 함께 풀고([comp_j_blank.f:603–768](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_j_blank.f:603)), Eddington factor를 반복 수렴시킨다([comp_j_blank.f:779–808](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_j_blank.f:779)). line formal에는 approximate diagonal Λ operator도 있다([formsol.f:1–16](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/formsol.f:1), [formsol.f:522–533](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/formsol.f:522)). | 기본 `cmfgen_solve_J`는 bin별 short-characteristics+대각/삼대각 ALI이다([lumina_cmfgen.c:539–680](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:539)). 별도 gate가 주파수 결합 CMF sweep을 제공하지만([lumina_cmfgen.c:1535–1546](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1535)), GPU는 `a_lam≠0`에서 `O(Nbin)` 반복이 필요해 현재 예산으로 미수렴한다고 스스로 경고한다([lumina_cmfgen.c:1636–1649](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1636)). | **큼.** 같은 CMF 방정식·경계조건·주파수 advection을 만족해야 하지만 VEF, ALI, Krylov 중 알고리즘은 달라도 된다. 최종 기준은 S-동등과 transport residual이다. |
| 복사평형 열해 | 각 주파수의 `χ_noscat J−η_noscat`를 온도식에 누적하고([cmfgen_sub.f:2305–2321](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2305)), 선율·J의 인구/온도 변분까지 BA에 넣는다([cmfgen_sub.f:2437–2502](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2437)). 이후 SE와 RE를 같은 선형계에서 갱신한다([cmfgen_sub.f:4001–4005](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4001)). | 현 열식은 `H−C` 채널 합으로 풀며([lumina_plasma.c:10463–10485](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:10463)), 여러 선택적 closure가 공존한다. coupled Newton도 기본은 셸 독립이고 depth coupling은 A4로 이월돼 있다([lumina_plasma.c:12336–12339](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:12336)); global 모드는 별도 조건부 gate다([lumina_plasma.c:12401–12410](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:12401)). | **최대 거리. E-동등 필수.** 열 장부만 비슷한 것이 아니라 같은 `J` 응답과 같은 population derivative를 포함한 RE 잔차가 필요하다. |
| super-level | CMFGEN은 full atom의 모든 과정은 유지하면서 작은 SL 인구로 기술한다([steq_multi_v10.f:1–18](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/steq_multi_v10.f:1)). full→SL 매핑으로 충돌률을 투영하고([steq_multi_v10.f:113–181](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/steq_multi_v10.f:113)), 해 뒤 LTE 비율/보간으로 full 인구를 복원하되 각 SL 총량을 정확히 재정규화한다([sup_to_full_v3.f:140–175](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sup_to_full_v3.f:140), [sup_to_full_v3.f:236–250](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sup_to_full_v3.f:236)). | Lumina도 pair matrix를 SL 공간에서 풀고 full 준위는 within-SL Boltzmann fraction으로 재분배한다([lumina_plasma.c:15730–15739](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15730)). 그러나 pair별 보존·핀 구조 때문에 SL 자체보다 **원소 전체 위상**이 다르다([lumina_plasma.c:15809–15820](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15809)). | **중–큼.** SL 사용은 유지 가능하나 full-rate→SL projection과 SL 총량 보존을 E-동등하게 해야 한다. CMFGEN도 모든 fine level을 독립 미지수로 푸는 것은 아니다. |
| 원자데이터·연속과정 위상 | ion마다 모든 `N_XzV_PHOT` 경로를 순회하고 각 경로의 upper target ID를 전달한다([comp_opac.f:86–101](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_opac.f:86), [mod_cmfgen.f:160–184](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/mod_cmfgen.f:160)). bf 불투명도에는 실제 target population을 사용한 stimulated-recombination 차감항이 있다([genopaeta_v10.f:175–205](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/genopaeta_v10.f:175), [genopaeta_v10.f:252–269](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/genopaeta_v10.f:252)). | ion별 첫 mapped target 하나로 `rr_act`를 정한다([lumina_plasma.c:6505–6528](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6505)); neutral은 전부 건너뛴다([lumina_plasma.c:6532–6537](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6532)); `χ=n_lσ`이며 stimulated recombination을 의도적으로 버린다([lumina_plasma.c:6666–6703](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6666)). | **최대 거리. E-동등 필수.** continuum identity, lower level, upper target, threshold, σ, spontaneous/stimulated Milne 쌍을 하나의 불가분 그래프로 보존해야 한다. |
| formal 스펙트럼 | 수렴 CMF `η,χ`에 scattering emissivity를 합하고([cmf_flux_sub_v5.f:2053–2057](/gpfs/kjhan/cmfgen_src/cur_cmf/obs/cmf_flux_sub_v5.f:2053)), 이를 observer-frame solver에 그대로 넘긴다([cmf_flux_sub_v5.f:2133–2166](/gpfs/kjhan/cmfgen_src/cur_cmf/obs/cmf_flux_sub_v5.f:2133)). observer solver는 Doppler/상대론 변환 후([obs_frame_sub_v9.f:683–750](/gpfs/kjhan/cmfgen_src/cur_cmf/obs/obs_frame_sub_v9.f:683)) ray intensity를 적분한다([obs_frame_sub_v9.f:942–1018](/gpfs/kjhan/cmfgen_src/cur_cmf/obs/obs_frame_sub_v9.f:942)). | Lumina의 `compute_cmf_formal_spectrum`은 Gaussian line-overlap ray solver다([lumina_plasma.c:17207–17222](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17207)); e-scatter source는 여러 field fallback을 쓰고 bf는 기본적으로 `Bν(T_e)`다([lumina_plasma.c:17297–17329](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17297)). 별도 formal-integral 경로에는 총입력 대비 ×18.07 비보존이 확정돼 있다([RE³:40–52](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/RE_RE_RE_ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-31.md:40)). | **최대 거리.** 알고리즘 복제는 불필요하지만 CMF 상태의 동일 `η,χ`, observer 변환, 경계조건, 절대 에너지 보존으로 S-동등해야 한다. |

---

## 3. 노선 결정

| 노선 | 예상 비용 | 핵심 위험 | 검증 가능성 | 판정 |
|---|---:|---|---|---|
| (a) MC/GPU 골격 유지, 통계적으로 CMFGEN 해에 접근 | 22–40 PM | noisy `J`로 전역 Jacobian/RE를 닫기 어렵고, 현재 pair-SE·target 위상은 MC packet 수만 늘려도 바뀌지 않음. field producer가 deterministic/MC/binned/fine으로 분열돼 있음([lumina.h:242–265](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:242), [lumina_cuda.cu:7153–7208](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7153)). | 중–낮음. 불일치가 bias인지 MC variance인지 계속 분리해야 함. | 비추천 |
| (b) 결정론 ALI/CMF 코어 전면 이식 | 35–60 PM | CMFGEN의 실체는 ALI만이 아니라 VEF·formal·moment·full linearization 혼합이다. 사실상 두 번째 CMFGEN을 C/CUDA로 재작성할 위험. | 높음. 함수별 대조가 쉬움. | fallback |
| **(c) 혼합** | **29–51 PM** | 두 solver의 권위 경계와 데이터 계약을 엄격히 관리해야 함. | **가장 높음.** Gate B→SE pilot→Jν→전역 상태→formal로 원인을 국소화할 수 있음. | **추천** |

추천 구조는 다음과 같다.

```text
원자 그래프
   ↓
deterministic CMF/VEF·ALI field
   ↓
element-wide SE + charge + RE global solve   ← 권위 있는 상태
   ├─ observer formal spectrum               ← 최종 acceptance
   └─ MC/GPU packet redistribution           ← 별도 방법/통계 lane
```

Lumina가 이미 결정론 CMF 루프와 fine `J̄` 생산자를 보유하므로([lumina_cmfgen.c:3167–3223](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:3167)), 이를 검증 가능한 권위 코어로 승격하는 편이 전면 포팅보다 유리하다.

---

## 4. 단계 계획

### Stage 0 — 기준선·앵커·판정 계약 봉인

- **변경 범위:** 물리 변경 없음. 소스/원자자료/run/effective gate 체크섬, 속도 기반 shell map, field consumer matrix, writer provenance, acceptance 스크립트 계약을 봉인한다.
- **선행 조건:** 없음.
- **Acceptance:**
  - 현 fixed-T 완주본은 Gate B와 J/rate 앵커로 인증.
  - 열평형 종국 앵커는 `FIX_T=F`, CMFGEN 내부 최대 보정 `<1%`, 공개 CMFGEN 대비 `T_e` p95 `<5%`, `n_e` p95 `<15%`, `L_bol<2%`를 요구.
  - D4-OFF parity57을 새 기준선으로 삼되, 생산 기본값 채택을 먼저 명시적으로 결정한다. D4의 재기저 결과는 이미 대장에 있다([VERIFICATION_REGISTERS.md:42](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERIFICATION_REGISTERS.md:42)).
- **검증:** Gate B shell `s0/s8/s45`와 속도 대응([GATE_B:42–60](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/GATE_B_DUAL_ORACLE_SPEC.md:42)); V0–V5를 모든 후속 단계의 공통 래퍼로 사용([VERDICT_PROTOCOL.md:7–35](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_PROTOCOL.md:7)).
- **규모:** 1–2 PM, 2–4주.
- **Go/No-Go:** released-T CMFGEN 앵커가 없으면 Stage 1–3은 진행 가능하나 Stage 4·6 최종 PASS 선언은 금지.

### Stage 1 — 원자 그래프·연속과정의 함수 동등성

- **변경 범위:** future implementation에서 `continuum_id → lower level → upper target → ν_edge → σν → Γ/α/χ/η → packet energy split`을 단일 자료구조로 만든다. neutral bf, stimulated recombination, spontaneous/stimulated Milne, B18 spin predicate를 같은 경로에 통합한다.
- **선행 조건:** Stage 0 manifest와 Gate B 생산함수 observer.
- **Acceptance:**
  - Gate B 세 셀에서 Γ, α, χbf, ηbf, χff, ηff가 공통 rate 문턱 통과.
  - LTE `J=B(T_e)`에서 각 continuum의 net bf current와 `χB−η` 상대 잔차 `≤10⁻⁸`.
  - packet bf event에서 `E_packet=E_ion+E_kinetic` 상대 오차 `≤10⁻¹²`.
  - 유효 continuum의 target coverage 100%; “산출 불가” 0건.
- **검증:** Gate B의 생산 함수 직접 호출 원칙과 결정론 2회 항등을 그대로 재사용한다([GATE_B:23–31](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/GATE_B_DUAL_ORACLE_SPEC.md:23), [GATE_B:82–88](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/GATE_B_DUAL_ORACLE_SPEC.md:82)). CMFGEN PRRR/GENCOOL/EDDFACTOR를 Lane C로, ARTIS를 산식 분해 Lane A로만 쓴다.
- **규모:** 3–5 PM, 6–10주.
- **흡수:** D-1의 identity/schema 부분, D-2, D-3, neutral bf, B18 전부.

### Stage 2 — element-wide SE + CMFGEN식 super-level 파일럿

- **변경 범위:** 우선 S II–IV와 Fe II–IV를 각 원소 하나의 행렬에 넣는다. 전 포함 이온단계, SL 인구, 원소보존, 전하/`n_e` 행을 함께 조립하고 pair owner/save-restore를 파일럿 원소에서 제거한다.
- **선행 조건:** Stage 1의 모든 rate가 oracle 인증됨.
- **Acceptance:**
  - 동일 동결 셀·동일 원자모델에서 행별 정규화 잔차 `≤10⁻¹⁰`, 원소/전하보존 `≤10⁻¹²`.
  - CMFGEN 해와 이온분율 절대차 `≤0.05`, 활성 `b_k` median/p95 `≤0.05/0.15 dex`.
  - matrix permutation, cold/hot initial population 두 경우가 같은 해로 수렴.
  - 파일럿 통과 후 모든 toy06 원소로 확장하며 coverage 100%.
- **검증:** `s0/s8/s45` frozen matrix dump, KA LTE matrix, Gate B rate 재사용, S/Fe 1셀 판정런. 기존 합의도 Gate 2를 S II–IV·Fe II–IV 파일럿으로 정했다([RE³:54–64](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/RE_RE_RE_ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-31.md:54)).
- **규모:** 4–7 PM, 8–14주.
- **흡수:** D-5, G-1, pair-wise ion owner 문제. D-5의 upper-stage-blind 기전은 감사에서 확정돼 있다([V3 audit:438–446](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md:438)).

### Stage 3 — 결정론 주파수결합 CMF 장을 단일 권위 생산자로 승격

- **변경 범위:** 현재 binned SC, fine-window `J̄`, MC coarse field의 역할을 분리한다. 전체 원자율과 RE가 읽는 권위장은 하나의 frequency-coupled CMF 해로 통일한다. VEF 또는 검증된 Λ*/Krylov preconditioner를 넣고 CPU 경로를 먼저 정본으로 삼는다.
- **선행 조건:** Stage 2 파일럿 SE.
- **Acceptance:**
  - frozen CMFGEN `χ,η,pop,T_e,n_e`에서 EDDFACTOR `Jν` 대비 median `≤5%`, p95 `≤15%`.
  - σ-weighted Γ `≤10%`, `Hν/Fν` moment `≤5%`.
  - 주파수·ray·depth 해상도 2배 시 변화가 위 문턱의 절반 이하.
  - pure absorption, pure scattering, homologous redshift KA에서 transport residual `≤10⁻⁴`.
  - GPU 경로는 같은 iteration budget에서 CPU 기준을 만족할 때만 활성. 현재처럼 `a_lam≠0`에서 `O(Nbin)` 반복이면 production 부적격.
- **검증:** KA1–KA3, EDDFACTOR 추출 인프라([CMFGEN_BUILD_RUN_GUIDE.md:105–110](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CMFGEN_BUILD_RUN_GUIDE.md:105)), CMFGEN frozen-state 판정런, CPU↔GPU 이중 실행.
- **규모:** 5–9 PM, 10–18주.
- **흡수:** G-2/G-3을 “ARTIS 복제”가 아니라 consumer provenance와 CMF 주파수 해상도 문제로 폐합.

### Stage 4 — SE·전하·복사평형·CMF 응답의 전역 coupled solve

- **변경 범위:** Stage 2의 element-wide residual, Stage 3의 `δJ/δχ, δη`, 전하/`n_e`, RE 온도행을 하나의 Newton/Krylov 계로 묶는다. depth coupling은 block-banded 또는 matrix-free preconditioner로 처리한다.
- **선행 조건:** Stage 1–3 전부 PASS. released-T CMFGEN 앵커 필요.
- **Acceptance:**
  - 정규화 SE/전하 잔차 `≤10⁻⁸`, RE 잔차 `≤10⁻³`.
  - 서로 다른 초기 `T_e`·이온화 상태에서 최종 `T_e,n_e,Jν` 차이 `<1%`.
  - CMFGEN 대비 상태 공통 문턱: `T_e 3/5%`, `n_e 5/10%`, 이온분율/`b_k` 문턱 통과.
  - 수렴 종료 시 마지막 3회에서 단조 잔차 감소 또는 계약된 globalization 증거가 있어야 하며 floor/clamp로 PASS 금지.
- **검증:** frozen→부분 coupled→full coupled KA 사다리, CMFGEN `RVTJ/PRRR/GENCOOL/Jν`, 동일 binary 쌍둥이 판정런, V0–V5.
- **규모:** 7–12 PM, 16–28주.
- **최고 위험 단계:** 이 단계가 실패하면 full-spectrum 조정으로 우회하지 않는다.

### Stage 5 — MC/GPU 재결합과 event-level 물리 폐합

- **변경 범위:** 결정론 수렴 상태를 MC/GPU에 전달하고 macro-atom 재분배와 packet observer만 수행한다. 이후 선택적으로 MC estimator feedback을 시험하되 deterministic operator와의 동등성을 별도 증명한다.
- **선행 조건:** Stage 4 수렴 상태.
- **Acceptance:**
  - packet energy ledger 오차 `≤0.1%`.
  - 모든 macro-atom action 확률 합 `1±10⁻¹²`; collisional destruction은 한 번만 추첨.
  - 여러 seed의 MC `Jν/J̄` 95% 신뢰구간이 deterministic 값을 포함하고, 합산 bias가 median `≤5%`, p95 `≤15%`.
  - D1의 `ν_edge/ν` 에너지 분할과 continuum target fate coverage 100%.
- **검증:** 기존 `EventRec`/channel census 인프라([lumina.h:66–104](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina.h:66)), packet fate/energy census, D4 known-answer, 동일 seed CPU/GPU 판정런.
- **규모:** 3–5 PM, 6–10주.
- **흡수:** D-1 event 부분, D-4 최종 폐합. 현재 terminal 재추첨 경로는 실제 코드에 존재한다([lumina_cuda.cu:4365–4388](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4365)).

### Stage 6 — 하나의 권위 있는 observer formal

- **변경 범위:** Stage 3–4의 수렴 CMF `χ,η`를 그대로 받아 observer frame으로 변환하는 단일 formal 경로를 만든다. 기존 Gaussian CMF formal과 `compute_formal_integral_spectrum`은 KA/진단 lane으로 내린다.
- **선행 조건:** Stage 4 상태, Stage 5 energy ledger.
- **Acceptance:**
  - KA1 영점, KA2 pure e-scattering, KA3 LTE `S=B` 모두 `≤10⁻³`.
  - production `L_out/(L_in+L_dep)=1±0.01`; ×18.07 완전 소멸.
  - CMFGEN 대비 `L_bol≤2%`, 대역 flux `≤5%`, EW `≤10%`, 특징속도 `≤500 km/s`.
  - 동일 상태를 formal과 MC observer에 넣었을 때 차이가 MC 2σ 또는 위 spectrum 문턱 안.
- **검증:** 기존 KA1 `1.000000028` 위에 KA2/3을 추가하는 합의안을 그대로 사용한다([GATE_B:13–19](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/GATE_B_DUAL_ORACLE_SPEC.md:13)). first-offending source/opacity bin 누적 추적 후 toy06 판정런.
- **규모:** 4–7 PM, 8–14주.
- **흡수:** ×18.07, formal source/opacity mismatch, MC emergent↔formal 레인 혼합.

### Stage 7 — 종국 acceptance와 일반화

- **변경 범위:** toy06 19.48d 전축 acceptance, 인접 epoch, 최소 한 개 독립 모델에서 회귀.
- **선행 조건:** Stage 1–6 모두 PASS.
- **Acceptance:** 공통 정량 계약 전 항목, 서로 다른 seed/GPU/해상도에서 재현. 모든 어긋남 지도 항목은 `PASS`, `explicit non-goal`, 또는 CMFGEN 자체 한계로 provenance가 있어야 하며 `UNVERIFIABLE` 고아는 0개.
- **검증:** runner spool, Gate B, KA, V0–V5, CMFGEN 자체런 epoch 사다리를 재사용한다([CMFGEN_BUILD_RUN_GUIDE.md:112–118](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CMFGEN_BUILD_RUN_GUIDE.md:112)).
- **규모:** 2–4 PM + 계산자원 1–2개월.

---

## 5. 기등재 결함 흡수표

| 항목 | 흡수 단계 | 처분 |
|---|---|---|
| D-1 continuum/level identity 및 `ν_edge/ν` 부재 | Stage 1 + Stage 5 | Stage 1에서 원자 그래프 복원, Stage 5에서 packet energy/fate 실현 |
| D-2 D6 허위 capability | Stage 0 | effective capability manifest로 교체; 미구현 기능은 PASS 배너 금지 |
| D-3 stimulated recombination 부재 | Stage 1 | CMFGEN target-population 보정식과 LTE KA로 인증 |
| D-4 `MA_LINE_DESTRUCT` 이중추첨 | Stage 0 + Stage 5 | 즉시 D4-OFF 기준선; Stage 5에서 단일 fair draw로 구조 폐합 |
| D-5 shared lo-ion save/restore | Stage 2 | element-wide matrix가 ion owner가 되면서 제거 |
| B18 REC_SPINGATE 3건 | Stage 0 + Stage 1 | 과대 배너는 Stage 0, `η_bf`/FB-Milne predicate 단일화는 Stage 1. 세 건의 정의는 대장에 명시돼 있다([VERIFICATION_REGISTERS.md:40](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERIFICATION_REGISTERS.md:40)). |
| neutral bf 누락 | Stage 1 | neutral을 다른 ion stage와 동일 continuum graph로 포함 |
| ×18.07 formal 비보존 | Stage 0 blocker + Stage 6 | 그 전 spectrum은 특성화만 허용; Stage 6에서 절대 에너지 acceptance |
| G-1 per-ion pin | Stage 2 | 원소보존행으로 대체 |
| G-2 field consumer 불일치 | Stage 3 | 권위 field/consumer ID를 하나로 통일 |
| G-3 bin geometry | Stage 3 | CMF convergence 기준으로 재정의; ARTIS exact-bin은 방법시험 전용 |

D-1~D-5와 G-1~G-3의 확정 분류는 V3 감사의 정본과 일치한다([V3 audit:481–500](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md:481)).

---

## 6. 비목표와 반드시 0으로 수렴할 지도

### 비목표

- CMFGEN 파일·서브루틴·반복 순서의 line-by-line 복제.
- CMFGEN과 bitwise-identical `Jν` 또는 스펙트럼.
- ARTIS를 최종 진리로 승격하는 것. 기존 합의대로 ARTIS는 방법 reference이고 CMFGEN이 acceptance target이다([RE³:66–68](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/RE_RE_RE_ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-31.md:66)).
- `JBAR_MIN`, damping, 온도 pin, 경험적 thermalization knob로 full spectrum을 직접 맞추는 것.
- MC/GPU의 폐기. packet redistribution과 대규모 event census는 유지한다.
- CMFGEN의 알려진 근사까지 무비판적으로 모사하는 것. 예를 들어 CMFGEN `STEQ_MULTI_V10`도 collisional ionization은 현재 ground target만 처리한다고 명시한다([steq_multi_v10.f:21–30](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/steq_multi_v10.f:21)); 이런 항목은 “CMFGEN parity”와 “더 완전한 물리” lane을 분리한다.

### 반드시 0으로 수렴할 구조 어긋남

1. pair-wise stage owner/save-restore/pin.
2. continuum identity와 upper-target collapse.
3. neutral bf와 stimulated Milne 누락.
4. full-rate↔super-level 투영 및 총량 불일치.
5. `Jν/J̄/Jblue` producer–consumer 세대 혼식.
6. CMF 주파수 advection 미수렴.
7. RE에서 `δJ/δpopulation, δJ/δT` 누락.
8. formal의 수송 `χ,η`와 별도 writer `τ,S` 혼용.
9. packet energy partition 이중계상.
10. 절대 luminosity 비보존 및 MC/formal 레인 혼합.

---

## 7. 위험과 조기 판별 신호

| 위험 | 조기 신호 | 중단/전환 기준 |
|---|---|---|
| **Stage 4 전역 coupled solve 실패 — 최고 확률** | hot/cold 초기값이 다른 branch, 3회 이상 잔차 정체·증가, floor/clamp 활성 증가, `T_e` 포켓 이동, SE는 줄지만 RE가 증가 | full spectrum 단계 진입 금지. Jacobian-vector finite-difference 검증과 1셀/1원소 문제로 후퇴 |
| CMF 주파수 coupling의 비실용적 수렴 | iteration 수가 `Nbin`에 비례, outer UV `J/B`가 해상도에 따라 수십 배 이동 | GPU lagged-advection 경로 보류, CPU sequential-frequency 정본 또는 implicit frequency block으로 전환 |
| 앵커 자체가 열평형 진리가 아님 | `RVTJ`의 `Was T fixed? T`, 공개 spectrum/ion fraction overlay 미통과 | fixed-T 자료는 frozen oracle로만 사용; Stage 4/6 acceptance 유예 |
| 원자모델 불일치가 solver 오차로 위장 | rate는 맞지만 특정 이온/준위만 지속적으로 큰 `b_k` 차이, continuum target coverage 결손 | 공통 model-atom projection을 먼저 확정; spectrum 튜닝 금지 |
| super-level 축약이 해를 바꿈 | identity mapping과 SL mapping의 이온분율/열률 차이가 acceptance 폭 초과 | 해당 이온의 SL 분할 재설계 또는 full-level 파일럿 |
| KA는 통과하지만 production formal만 에너지 생성 | 특정 주파수 bin부터 누적 `L_out−L_in`가 단조 성장, `S/χ` writer provenance가 둘 이상 | first-offending bin/과정이 특정될 때까지 Stage 6 판정런 반복 금지 |
| MC variance가 bias를 숨김 | seed 간 분산이 변경 효과보다 큼, packet 수 증가 시 평균이 결정론 값으로 수렴하지 않음 | estimator feedback 비권위 유지; packet 수 사다리와 confidence interval 사전등록 |
| 전역 BA 메모리/시간 폭발 | 메모리가 `Nlevel²×Ndepth`로 증가, iteration당 CMFGEN 대비 10배 이상 | dense 이식 중단, block-sparse/matrix-free JFNK와 physics preconditioner로 전환 |

---

최우선 실행 순서는 **Stage 0 앵커/계약 봉인 → Gate B와 KA2/3 → Stage 1 continuum graph → Stage 2 S/Fe element-wide 파일럿**이다. 이 네 관문을 통과하기 전에는 full-spectrum 변화에 “CMFGEN parity 개선”이라는 판정을 붙이지 않는 것이 핵심이다.

이번 작업에서는 파일을 수정하지 않았다.