개정판 v2 원고를 완성했습니다. 읽기 전용 역할에 따라 [원본 계획서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_CMFGEN_EQUIVALENCE_PLAN.md)는 수정하지 않았습니다.

---

# Lumina → CMFGEN-동등 수준 단계 계획 — 개정판 v2

## 0. 결론

확정 노선은 **(c) 혼합형**이며 변경하지 않는다.

- 결정론 frequency-coupled CMF 수송과 전 model-atom super-level SE·전하·복사평형 연립계를 물리 상태의 권위 생산자로 둔다.
- MC/GPU는 macro-atom 재분배, packet fate, emergent 통계와 가속기 경로로 유지한다.
- MC estimator feedback은 결정론 operator/rate와의 동등성이 별도로 입증된 뒤에만 권위 경로에 진입한다.
- 목표는 CMFGEN 구현의 복제가 아니라 같은 원자모델·경계조건·물리 잔차를 풀어 같은 해를 얻는 것이다.

### 앵커 판정 정정

현재 `toy06_19p48d_modern`은 **어떤 축에서도 인증 앵커가 아니다**. `FIX_T=T`일 뿐 아니라 마지막 상태가 수렴하지 않았고, 외곽 luminosity, active-variable correction, moment solve가 모두 실패 기준을 넘었다([C 리뷰:127–140](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EQUIV_PLAN_C_REVIEW.md:127)).

고정온도 산출물에 대한 “Jν/rate/population oracle” 주장은 전부 철회한다. 향후 fixed-T 계산도 오직 released-T 진입을 위한 **staging checkpoint**로만 쓰며, 앵커·oracle·최종 acceptance 자료로 인증하지 않는다.

또한 job `385770`은 안정화되지 않은 상태에서 `FIX_T=T→F`만 바꾼 released-T 재시도가 실제로 발산할 수 있음을 보였다. 보정량 `7.8×10⁵–5.6×10⁶%`, NaN luminosity/optical depth, grey-solution 실패 후 종료됐으며 수렴 EDDFACTOR가 없다([R8_RESUME_NOTE.txt:928–932](/gpfs/kjhan/cmfgen_runs/R8_RESUME_NOTE.txt:928)). 따라서 새 앵커는 별도 검수된 안정화 레시피를 거친 **수렴 released-T run**으로만 생성한다.

### 목표 범위

핵심 프로젝트의 최종 목표는 다음으로 봉인한다.

- **주 acceptance:** `DO_DDT=F`인 matched-physics steady CMFGEN self-run과의 동등성.
- **과학 교차검사:** 공개 StaNdaRT CMFGEN time-dependent 19.48 d 결과와의 비교. 물리 범위가 다르므로 주 acceptance에 혼합하지 않는다.
- **별도 확장:** 공개 57-epoch sequence 동등성을 주장하려면 comoving SE derivative와 time-dependent energy/adiabatic 항을 포함하는 Stage 7T를 추가해야 한다. CMFGEN의 해당 항은 [cmfgen_sub.f:2920–2958](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2920)에 존재한다.

### 재산정 비용

단계별 WBS 합계는 **85–147 person-month**이다. 교차단계 통합·재작업 예비비 15%를 더한 승인 예산은 **98–169 PM**이다.

- 3인 팀, 직접 작업률 80%: 약 **41–70개월**
- 4인 팀, 직접 작업률 80%: 약 **31–53개월**
- time-dependent Stage 7T: 별도 **21–37 PM**

계산 queue 시간과 장비비는 PM에 포함하지 않는다. 기존 29–51 PM 및 12–18개월 추정은 폐기한다.

실제 solver source는 `/gpfs/kjhan/cmfgen_src/cur_cmf/`, 원자자료는 `/gpfs/kjhan/cmfgen_21jun23/atomic/`이다.

---

## 1. “CMFGEN-동등”의 정의

세 종류를 구분한다.

- **M-동등(model-atom equivalence):** 양쪽 solver에 투영된 ion, full level, super-level, line, continuum, target, collision/photoionization 자료와 활성 과정의 ID·수·checksum이 일치한다.
- **E-동등(equation equivalence):** 동일 상태에서 SE, 전하보존, 복사평형, 수송, detailed-balance 잔차와 Jacobian-vector product가 같은 방정식을 나타낸다.
- **S-동등(solution equivalence):** 풀이 알고리즘과 선형대수 구현이 달라도 수렴한 `Jν`, population, `T_e`, `n_e`, 이온분율과 스펙트럼이 사전등록 문턱 안에서 일치한다.

CMFGEN의 미지수는 모든 fine level을 독립적으로 푸는 것이 아니라 **모든 활성 model-atom super-level과 온도·전하 변수**이다. full-level 과정은 SL 계에 투영되고 해 뒤 복원된다.

### 공통 정량 계약

| 축 | 최종 문턱 |
|---|---|
| M-동등 | 활성 ion/SL/full-level/line/continuum/target 수와 ID 100% 일치; energy·g·threshold·mapping checksum 불일치 0건 |
| 보존/KA | 입자·전하 잔차 `≤10⁻¹⁰`; LTE detailed balance `≤10⁻⁸`; event energy ledger `≤10⁻³` |
| 동결 상태 rate | 활성 항 `|log10(Lumina/CMFGEN)|` median `≤0.03 dex`, p95 `≤0.10 dex`, max `≤0.20 dex` |
| 복사장 | `Jν` median `≤5%`, p95 `≤15%`; σ-weighted Γ `≤10%`; `Hν/Fν` moment `≤5%` |
| 상태 | `T_e` median/p95 `≤3/5%`; `n_e≤5/10%`; 지배 이온분율 절대차 `≤0.05`, 비율 `≤0.10 dex` |
| 준위인구 | 활성 `b_k` median `≤0.05 dex`, p95 `≤0.15 dex` |
| formal | 에너지 오차 `≤1%`; `L_bol≤2%`, 주요 대역 flux `≤5%`, EW `≤10%`, 특징속도 `≤500 km/s` |
| E-동등 | 공통 상태의 scaled residual 상대차 `≤10⁻⁶`; centered finite-difference Jacobian-vector 상대차 `≤10⁻⁴` |
| 이산화 | `h, h/2, h/4` 3단 grid에서 Richardson 외삽 오차가 해당 acceptance 폭의 절반 이하 |

### 판정 계산 규약

- active set은 양쪽 코드의 **합집합**이다. 어느 쪽이든 이온분율 `>10⁻⁸`이거나 해당 열률·율·불투명도 기여가 총량의 `>10⁻⁴`이면 활성이다.
- 활성 양수값에는 log-ratio를 쓴다. 한쪽만 0 또는 음수이면 floor로 숨기지 않고 `support/sign mismatch`로 별도 실패 처리한다.
- rate와 population quantile은 활성 `(shell, transition/level)` 표본에 대해 동일 가중으로 계산하고, 질량가중 결과를 보조 지표로 병기한다.
- `Jν`는 공통 log-frequency grid에 보존적으로 재빈닝하고 `Δm·Δlnν` 가중 quantile을 사용한다. shell별 최악값도 별도 보고한다.
- 상태 변수의 depth quantile은 shell mass로 가중한다.
- SE 행은 `max(total inflow,total outflow,n_i/t_ref)`로, 전하행은 `n_e`로, 원소보존행은 원소 총밀도로, RE행은 `max(total heating,total cooling,|deposition|)`로 scaling한다.
- shell map은 속도 기준 monotonic map과 보존 재빈닝 checksum을 Stage 0에서 봉인한다.
- EW·특징속도는 line ID, wavelength window, continuum 규약과 blend 처리법을 manifest에 고정한다.
- MC 다중-bin 판정은 seed ensemble의 95% **simultaneous** confidence band를 사용한다. pointwise interval만으로 PASS하지 않는다.
- grid 검증은 단일 2배 해상도 비교가 아니라 `h, h/2, h/4` 사다리와 관측 수렴차수를 요구한다.

---

## 2. 6대 구조 대조

| 축 | CMFGEN 실물 | Lumina 현행 | 동등 조건 |
|---|---|---|---|
| 전 model-atom SL SE·전하·온도 선형화 | 모든 ion의 STEQ/BA와 전하식을 조립하고([cmfgen_sub.f:1588–1680](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:1588)), block-banded 계를 푼다([solve_for_pops.f:1–8](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:1), [solveba_v13.f:90–104](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:90)). 인구와 온도는 같은 correction vector로 갱신된다([solveba_v13.f:311–322](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:311)). | 한 ion pair·한 shell씩 조립해 Gauss solve한다([lumina_plasma.c:15746–15785](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15746)). 기본 5회·0.5 damping이며 element-wide matrix가 잔여라고 명시한다([lumina_plasma.c:16252–16263](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16252)). 공유 lo-ion은 저장 후 뒤 pair 결과를 복원으로 폐기한다([lumina_plasma.c:16291–16335](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16291)). | 원소별 frozen-`n_e` 파일럿과 전 원소 global charge solve를 분리한다. 최종에는 모든 원소가 공유하는 하나의 `n_e`와 RE가 전역계에 들어가야 한다. |
| CMF formal·moment/VEF 수송 | formal ray와 moment 계열을 함께 호출하고([comp_j_blank.f:603–768](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_j_blank.f:603)), Eddington factor를 반복 수렴시킨다([comp_j_blank.f:779–808](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_j_blank.f:779)). line formal은 approximate diagonal `LAMLINE`을 반환한다([formsol.f:1–16](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/formsol.f:1)). | binned SC+ALI 경로([lumina_cmfgen.c:539–680](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:539))와 별도 frequency-coupled gate([lumina_cmfgen.c:1535–1546](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1535))가 있다. 현 advection은 Courant 문제를 capped operator split으로 우회한다([lumina_cmfgen.c:1554–1564](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1554)). | 현 경로의 단순 승격이 아니라 implicit frequency coupling, 경계조건, moment closure와 수렴 검증을 갖춘 새 권위 solver로 취급한다. |
| 복사평형 열해 | `χ_noscat J−η_noscat`를 적분하고([cmfgen_sub.f:2305–2321](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2305)), line/continuum의 인구·온도·J 변분을 BA에 반영한다([cmfgen_sub.f:2516–2525](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2516), [update_ba_for_line.f:178–208](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/update_ba_for_line.f:178), [update_ba_for_line.f:504–590](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/update_ba_for_line.f:504)). | 현 열식은 `H−C` 채널 합이다([lumina_plasma.c:10479–10501](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:10479)). coupled Newton은 기본적으로 shell-local이다([lumina_plasma.c:12352–12355](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:12352)); A4 global은 조건부 gate다([lumina_plasma.c:12417–12429](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:12417)). 현재 global 핵심도 shell당 `T_e,n_e` 2×2 block이다([lumina_plasma.c:12035–12069](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:12035)). | Stage 4는 증분 수정이 아니라 새 global nonlinear solver다. `δJ/δpopulation`, `δJ/δT`와 depth coupling을 포함해야 한다. |
| super-level | full-atom 과정을 SL population으로 기술한다([steq_multi_v10.f:1–18](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/steq_multi_v10.f:1)). 실제 full collision rate의 SL 합산은 [subcol_multi_v6.f:177–230](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/subcol_multi_v6.f:177)에 있으며, 해 뒤 full population을 복원·재정규화한다([sup_to_full_v3.f:140–175](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sup_to_full_v3.f:140), [sup_to_full_v3.f:236–250](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sup_to_full_v3.f:236)). | pair matrix는 SL에서 풀고 full population은 within-SL Boltzmann fraction으로 복원한다([lumina_plasma.c:15755–15758](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15755), [lumina_plasma.c:16205–16234](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16205)). pair 보존행과 per-ion rescale/pin이 별도로 존재한다([lumina_plasma.c:15686–15701](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15686), [lumina_plasma.c:15832–15843](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15832)). | full-rate→SL projection과 각 SL 총량 보존의 E-동등을 요구한다. |
| 원자자료·bb/bf 과정 위상 | ion별 모든 photoionization target을 순회하고([comp_opac.f:86–101](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_opac.f:86)), target population을 stimulated-recombination 차감에 사용한다([genopaeta_v10.f:175–205](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/genopaeta_v10.f:175), [genopaeta_v10.f:252–269](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/genopaeta_v10.f:252)). | 첫 mapped target만 선택한다([lumina_plasma.c:6541–6544](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6541)); neutral을 건너뛴다([lumina_plasma.c:6551–6553](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6551)); `χ=n_lσ`이며 stimulated recombination을 버린다([lumina_plasma.c:6694–6714](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6694)). | continuum과 line 모두 identity→full level→SL→rate→opacity/emissivity→event fate의 전 사슬을 하나의 graph로 보존한다. |
| observer formal | 수렴 CMF `η,χ`에 scattering emissivity를 합쳐 같은 배열을 observer solver에 전달한다([cmf_flux_sub_v5.f:2053–2057](/gpfs/kjhan/cmfgen_src/cur_cmf/obs/cmf_flux_sub_v5.f:2053), [cmf_flux_sub_v5.f:2133–2166](/gpfs/kjhan/cmfgen_src/cur_cmf/obs/cmf_flux_sub_v5.f:2133)). | Gaussian overlap formal의 실제 구현은 [lumina_plasma.c:17230–17245](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17230)이며, e-scattering fallback과 bf `Bν(T_e)` 처리는 [lumina_plasma.c:17330–17351](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17330)에 있다. | 권위 수송의 동일 `χ,η`, 동일 scattering kernel, observer 변환과 절대 에너지 보존이 필요하다. |

### 추가 필수 축

- **시간의존성:** steady와 time-dependent residual·데이터·판정을 혼합하지 않는다.
- **bound-bound topology:** `line_id→lower/upper full level→SL→stimulated emission→net rate→macro-atom fate`와 per-line LTE KA를 추가한다.
- **전자산란:** coherent, incoherent/redistribution, Compton 열교환 중 활성 범위를 manifest로 봉인하고 moment·RE·formal·MC가 같은 kernel을 사용하게 한다.
- **과정 inventory:** charge exchange, dielectronic/autoionization, non-thermal, Auger/X-ray, two-photon, dissolution, free-free, Rayleigh, dust, clumping, radioactive deposition 및 time-energy 항을 coverage ledger에 등록한다.
- **경계조건:** inner luminosity/deposition, outer incident field, homologous frequency advection과 observer boundary를 독립 계약으로 둔다.

---

## 3. 노선 결정

노선 (c)는 확정한다. 단, “가장 싸다”는 근거로 선택하지 않는다.

| 노선 | 동일 범위 위험포함 비용 | 판정 |
|---|---:|---|
| (a) MC 중심 global solve | 약 90–165 PM, 불확실성 큼 | pair-SE·원자 위상은 별도 재작성해야 하고 stochastic Jacobian/variance-control R&D가 추가된다. 비선택 |
| (b) 결정론 코어 + formal, MC 재결합 제외 | 약 89–153 PM | 결정론 부분은 (c)와 대부분 겹친다. 비용상 정직한 fallback |
| **(c) 혼합형** | **98–169 PM** | 결정론 상태의 검증 가능성과 기존 MC/GPU event 기능을 함께 보존하는 확정 제품 구조 |

```text
sealed model-atom/process graph
               ↓
implicit deterministic CMF + moment/VEF
               ↓
global SL-SE + one charge equation + RE      ← 권위 상태
        ├── single observer formal           ← 주 acceptance
        └── MC/GPU redistribution/observer   ← 통계·방법 lane
```

현 binned solver와 fine `J̄` gate는 재사용 후보 자산이지만, 권위 코어가 이미 존재한다는 증거는 아니다. 기본 경로와 fine producer가 분리돼 있음은 [lumina_cmfgen.c:3167–3223](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:3167)에서 확인된다.

---

## 4. 단계 계획

### Stage 0 — 범위·모델·앵커·판정 계약 재생성

- 현 fixed-T run과 job 385770을 모두 `REJECTED_AS_ORACLE`로 기록한다.
- steady self-run을 주 acceptance로 봉인하고 공개 time-dependent 결과는 science cross-check로 분리한다.
- model-atom projection gate를 만든다. ion/SL/full-level/line/continuum/target 수, energy, `g`, mapping과 데이터 checksum이 일치해야 한다.
- process·boundary·defect coverage ledger와 acceptance manifest를 생성한다.
- `docs/CODEX_RELT_RECIPE.md`에 안정화 레시피가 검수되기 전에는 released-T run을 제출하지 않는다.

Released-T 앵커 생성 순서는 다음과 같다.

1. 원본 run을 in-place 수정하지 않고 source/config/restart checksum을 보존한 복제본을 만든다.
2. `FIX_T=T`에서 active population, `Jν`, moment solve를 실제로 안정화한다. 이는 staging checkpoint일 뿐 앵커가 아니다.
3. job 385770과 다른 안정화 전략을 사전등록한다. continuation state, damping/step cap, LAMBDA↔full-linearization 전환, trace-population scaling, release ladder, rollback/kill gate를 명시한다.
4. `FIX_T=F`로 release해 SE·charge·RE·moment를 다시 수렴시킨다.
5. 최종 판정 구간에서는 임시 population freeze, 온도 hold, 비물리 floor와 convergence mask가 모두 해제돼야 한다.
6. 다음을 동시에 만족해야 released-T 앵커로 인증한다.
   - `RVTJ: Was T fixed? F`
   - moment solver error 0건
   - active correction p95 `<0.1%`, max `<1%`
   - scaled SE/charge residual `≤10⁻⁸`, RE residual `≤10⁻³`
   - 마지막 3회 `T_e,n_e,Jν,L` 변화 `<0.5%`
   - `L_out/(L_inner+L_dep)=1±0.01`
   - grey failure, NaN/Inf, hidden clamp 0건

- **Go/No-Go:** released-T 앵커가 없으면 구조·KA·projection 개발은 가능하지만 CMFGEN rate/J/state/spectrum PASS 선언은 전 단계에서 금지한다.
- **WBS:** 범위·실격 ledger 1–2 PM; 판정 도구 2–3 PM; model projection 2–4 PM; 안정화 레시피·run forensic 2–4 PM.
- **규모:** **7–13 PM**.

### Stage 1 — 원자·과정 graph의 함수 동등성

- continuum graph와 bound-bound graph를 함께 구축한다.
- neutral bf, 복수 upper target, stimulated recombination, spontaneous/stimulated Milne, line stimulated emission과 macro-atom fate를 통일한다.
- process inventory의 모든 항목에 `implemented`, `anchor-inactive explicit non-goal`, `Stage 7T`, `blocked` 중 하나를 부여한다.
- coherent/incoherent electron-scattering kernel 계약과 energy-exchange sign convention을 고정한다.

Acceptance:

- model projection checksum 100%.
- Gate B 셀에서 bf/ff/bb의 Γ, α, χ, η, net rate가 rate 문턱 통과.
- 각 line/continuum의 LTE detailed balance `≤10⁻⁸`.
- target, lower/upper level, SL mapping coverage 100%.
- packet event 에너지 오차 `≤10⁻¹²`.
- **WBS:** continuum 3–5 PM; line topology 2–4 PM; e-scattering contract 2–3 PM; secondary-process ledger·KA 2–4 PM.
- **규모:** **9–16 PM**.

### Stage 2 — element-wide SE와 global charge solve

Stage 2A는 **frozen-`n_e` 원소 파일럿**이다.

- S II–IV와 Fe II–IV를 각 원소 하나의 SL 행렬로 조립한다.
- 원소보존행은 포함하되 전하행을 원소마다 중복 삽입하지 않는다.
- pair owner/save-restore/pin을 파일럿에서 제거한다.

Stage 2B는 **전 원소 global charge solve**다.

- toy06의 모든 활성 원소를 확장한 뒤 하나의 공유 `n_e` 전하식을 결합한다.
- trace population은 row scaling으로 수치적으로 보호하되 최종 해에서 freeze하지 않는다.

Acceptance:

- scaled SE 잔차 `≤10⁻¹⁰`, 원소보존 `≤10⁻¹²`.
- global charge 잔차 `≤10⁻¹⁰`.
- matrix permutation 및 hot/cold 시작점에서 같은 해.
- residual-vector 상대차 `≤10⁻⁶`, Jv 상대차 `≤10⁻⁴`.
- released-T 앵커 대비 ion fraction과 `b_k` 공통 문턱 통과.
- **WBS:** S/Fe 파일럿 3–5 PM; SL projection 2–4 PM; 전 원소 확장 3–5 PM; global charge 3–5 PM; residual/Jv 도구 2–3 PM.
- **규모:** **13–22 PM**.

### Stage 3 — implicit frequency-coupled CMF 권위장

- binned SC, fine `J̄`, MC field의 consumer를 분리하고 권위 producer를 하나로 만든다.
- capped advection을 production acceptance에서 제외하고 implicit frequency block 또는 동등하게 검증된 sequential-frequency formulation을 사용한다.
- inner/outer boundary, homologous frequency advection, electron redistribution을 포함한다.
- CPU를 정본으로 먼저 완성하고 GPU는 동등성 통과 후 활성화한다.

Acceptance:

- released-T 앵커의 frozen `χ,η,pop,T_e,n_e`에서 `Jν`, Γ, moment 문턱 통과.
- pure absorption, coherent scattering, redistribution scattering, homologous-redshift KA 통과.
- transport residual `≤10⁻⁴`.
- `h,h/2,h/4` Richardson 검증 통과.
- GPU가 같은 residual과 해상도 문턱을 만족하지 못하면 비권위 유지.
- **WBS:** implicit transport 5–8 PM; moment/VEF·preconditioner 3–5 PM; boundary·redistribution 2–4 PM; 3-grid harness 2–3 PM; GPU parity 2–4 PM.
- **규모:** **14–24 PM**.

### Stage 4 — SE·전하·RE·CMF 응답의 global nonlinear solve

- Stage 2 residual, 하나의 global charge equation, RE 온도행과 Stage 3 radiative response를 Newton–Krylov 계로 결합한다.
- `δJ/δpopulation`, `δJ/δT`, depth coupling을 포함한다.
- trace population scaling과 물리 preconditioner를 설계한다.
- 마지막 3회 단조감소를 필요조건으로 삼지 않고 trust-region 또는 line-search sufficient-decrease 계약을 사용한다.

Acceptance:

- SE/charge `≤10⁻⁸`, RE `≤10⁻³`.
- centered finite-difference Jv 상대차 `≤10⁻⁴`.
- hot/cold 시작점의 최종 상태 차이 `<1%`.
- line-search가 8회 backtrack 후 실패하는 비율 `<10%`, 최종 5 Newton step에서는 0건.
- preconditioned Krylov p95 `≤80` iteration, 단일 solve `>150` iteration 0건.
- depth grid 2배에서 Krylov iteration 증가는 2배 미만.
- 기준 CPU run peak RSS `≤180 GiB`; dense `Nlevel²×Ndepth` 할당 금지.
- Newton step 비용 `≤6`회 Stage 3 transport solve 상당.
- floor/clamp/freeze가 활성인 상태로 PASS 금지.
- **WBS:** global residual 4–7 PM; radiative Jv 4–7 PM; JFNK/globalization 4–7 PM; preconditioner/scaling 4–6 PM; diagnostics 3–5 PM; 성능예산 2–4 PM.
- **규모:** **21–36 PM**.

### Stage 5 — MC/GPU 재결합과 event-level 폐합

- 결정론 수렴 상태를 read-only contract로 MC/GPU에 전달한다.
- macro-atom fate와 packet observer를 수행하되 상태 producer 권한은 주지 않는다.
- collisional destruction 이중추첨을 제거한다. 기존 `kp_deact` 포함 근거와 terminal 재추첨을 함께 검증한다([lumina_plasma.c:4425–4445](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4425), [lumina_cuda.cu:4365–4388](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4365)).
- 전자산란 redistribution과 continuum energy split도 동일 kernel/ledger를 사용한다.

Acceptance:

- packet energy ledger `≤0.1%`.
- action probability `1±10⁻¹²`.
- continuum/line/scattering fate coverage 100%.
- 여러 seed의 95% simultaneous confidence band가 결정론 결과를 포함.
- CPU/GPU 동일 seed event census 계약 통과.
- **WBS:** 상태 contract 2–3 PM; macro-atom 수정 2–3 PM; redistribution/event ledger 1–2 PM; 통계판정 2–4 PM; GPU parity 1–2 PM.
- **규모:** **8–14 PM**.

### Stage 6 — 하나의 권위 observer formal

- Stage 3–4의 동일 `χ,η`와 scattering kernel을 observer frame으로 전달한다.
- Gaussian formal과 기존 formal-integral 경로는 KA/진단 lane으로 내린다.
- line/EW manifest와 절대 luminosity ledger를 적용한다.

Acceptance:

- 영점, LTE `S=B`, coherent scattering, redistribution-scattering KA `≤10⁻³`.
- `L_out/(L_inner+L_dep)=1±0.01`.
- 기존 ×18.07 비보존 완전 소멸.
- released-T 앵커 대비 spectrum 공통 문턱 통과.
- 같은 상태의 formal과 MC observer 차이가 MC simultaneous 2σ 또는 spectrum 문턱 안.
- **WBS:** observer solver 3–5 PM; redistribution formal 1–2 PM; KA/energy ledger 2–3 PM; spectrum metric 2–3 PM.
- **규모:** **8–13 PM**.

### Stage 7 — 종국 acceptance와 일반화

Stage 7S는 핵심 steady 목표다.

- toy06 19.48 d 전축 acceptance.
- 인접 epoch의 matched-steady anchor.
- 최소 한 개 독립 모델.
- CPU/GPU/seed/grid 재현.
- coverage ledger의 `UNVERIFIABLE` 고아 0건.

Stage 7T는 별도 time-dependent 확장이다.

- epoch restart/data contract.
- comoving SE derivative.
- time-dependent energy와 adiabatic 항.
- radioactive deposition의 시간 장부.
- 최소 3개 연속 epoch의 residual·solution·spectrum 검증.
- Stage 7T 이전에는 “공개 CMFGEN time-dependent sequence와 동등”하다고 표현하지 않는다.

- **Stage 7S WBS:** toy06·인접 epoch 2–3 PM; 독립 모델 2–4 PM; release/회귀 1–2 PM.
- **Stage 7S 규모:** **5–9 PM**.
- **Stage 7T:** 기본 18–32 PM, 위험예비비 포함 **21–37 PM** 별도.

---

## 5. 기등재 결함 흡수표

| 항목 | 흡수 단계 | 처분 |
|---|---|---|
| 기존 fixed-T oracle 주장 | Stage 0 | 전면 철회, `REJECTED_AS_ORACLE` |
| job 385770 released-T 발산 | Stage 0 | 안정화 레시피 선행조건과 rollback gate로 흡수 |
| steady/time-dependent 혼재 | Stage 0 + 7T | 주 acceptance 분리; 공개 sequence 주장은 7T 전 금지 |
| model-atom projection 부재 | Stage 0 | 독립 M-동등 gate |
| D-1 continuum identity/energy split | Stage 1 + 5 | graph와 event ledger |
| bound-bound topology 결손 | Stage 1 + 2 + 5 | line identity부터 fate까지 폐합 |
| D-2 허위 capability | Stage 0 | effective manifest |
| D-3 stimulated recombination | Stage 1 | target-population 식과 LTE KA |
| D-4 이중추첨 | Stage 0 + 5 | D4-OFF 기준선과 단일 draw |
| D-5 shared lo-ion restore | Stage 2 | element-wide matrix로 제거 |
| neutral bf | Stage 1 | 동일 continuum graph에 포함 |
| B18 predicates | Stage 1 | 단일 process predicate |
| G-1 per-ion pin | Stage 2 | 원소보존 + global charge로 대체 |
| G-2 field consumer 혼식 | Stage 3 | producer/consumer ID 단일화 |
| G-3 bin geometry | Stage 3 | 3-grid CMF convergence로 재정의 |
| incoherent e-scattering | Stage 1·3·4·6 | 동일 kernel을 rate/moment/RE/formal에 사용 |
| 과정 inventory 결손 | Stage 0 + 1 | 활성은 구현, 비활성은 명시적 non-goal과 zero-path test |
| acceptance 계산 규약 부재 | Stage 0 | active set·가중·scaling·Jv·grid·EW·MC 규약 봉인 |
| 경계조건 provenance | Stage 0 + 3 + 6 | inner/outer/deposition/observer 계약 |
| ×18.07 formal 비보존 | Stage 6 | 절대 에너지 acceptance |

---

## 6. 비목표와 반드시 0으로 수렴할 지도

### 비목표

- CMFGEN 파일·서브루틴·반복 순서의 line-by-line 복제.
- bitwise-identical `Jν` 또는 스펙트럼.
- ARTIS를 최종 진리로 승격하는 것.
- fixed-T 산출물을 어떤 물리축의 인증 oracle로 사용하는 것.
- 경험적 damping, 온도 pin, trace freeze 또는 spectral knob가 켜진 상태로 최종 PASS하는 것.
- Stage 7T 이전에 공개 time-dependent sequence와의 동등성을 주장하는 것.
- MC/GPU 폐기.
- 앵커에서 비활성인 dust/Rayleigh 등 모든 CMFGEN 옵션의 무조건적 구현. 단, 비활성 여부와 non-goal provenance는 필수다.

### 반드시 0으로 수렴할 구조 어긋남

1. 미수렴/fixed-T 자료의 oracle 사용.
2. steady/time-dependent 판정 혼재.
3. model-atom ID·수·checksum 결손.
4. pair-wise owner/save-restore/pin.
5. 원소별 중복 `n_e`/charge equation.
6. continuum upper-target collapse.
7. bound-bound identity·stimulated-emission·fate 단절.
8. neutral bf와 stimulated Milne 누락.
9. full-rate↔SL projection 및 총량 불일치.
10. e-scattering kernel의 moment/RE/formal 불일치.
11. `Jν/J̄/Jblue` producer–consumer 혼식.
12. CMF frequency advection의 capped 비수렴.
13. RE의 population/T radiative derivative 누락.
14. formal과 transport의 `χ,η` provenance 분리.
15. packet energy partition과 destruction 이중계상.
16. 경계조건·deposition ledger 불일치.
17. 절대 luminosity 비보존.
18. acceptance metric의 active set·가중·scaling 미정.
19. defect/process ledger의 `UNVERIFIABLE` 고아.

---

## 7. 위험과 조기 판별 신호

| 위험 | 조기 신호 | 중단/전환 기준 |
|---|---|---|
| released-T 앵커 재발산 | job 385770형 `10⁵–10⁶%` correction, NaN, grey failure, moment error | blind rerun 금지. 레시피 재설계 후 fixed-T staging부터 재시작 |
| fixed-T staging도 안정화 실패 | active correction 정체, 외곽 luminosity ring, trace ion이 MAXCH 지배 | 앵커 생성 중단. trace scaling·model atom·linearization 전환 검토 |
| Stage 4 global solve 실패 | multistart branch, Jv 불일치, line-search 실패율 증가 | spectrum 진입 금지. 1셀→1원소→frozen-J 사다리로 후퇴 |
| preconditioner 부적합 | Krylov p95 `>80`, grid 증가에 iteration 폭증 | physics block 재설계; dense fallback 금지 |
| CMF frequency coupling 비실용 | iteration이 `Nbin`에 비례, grid에 따라 UV장이 이동 | capped split 폐기, implicit frequency block으로 전환 |
| model-atom 불일치 | rate는 맞지만 특정 ion/level만 지속 이탈 | solver 튜닝 중단, M-gate 재검사 |
| global charge 잘못 조립 | 원소별 파일럿은 통과하지만 전체 `n_e`가 시작값 의존 | Stage 2B 전면 재검토 |
| e-scattering 범위 불일치 | moment는 맞지만 RE/formal 에너지 장부 실패 | kernel provenance가 일치할 때까지 Stage 4/6 중단 |
| time-dependent 결과 혼입 | steady residual PASS인데 공개 profile만 체계적 이탈 | science discrepancy로 기록; Stage 7T 없이 core failure로 재분류하지 않음 |
| formal 에너지 생성 | 특정 bin부터 `L_out−L_in` 누적 증가 | first-offending process가 특정될 때까지 판정런 금지 |
| MC variance가 bias 은폐 | packet 증가에도 평균이 결정론 값으로 미수렴 | feedback 비권위 유지, simultaneous CI 재설계 |
| 메모리·시간 폭발 | RSS `>180 GiB`, Newton step `>6` transport-equivalent | matrix-free/block-sparse 경로 외 구현 중단 |

최우선 순서는 다음과 같다.

**Stage 0 기존 앵커 실격 등록 → steady/time-dependent 범위 봉인 → model-atom projection gate → released-T 안정화 레시피 검수 → Stage 1 continuum·line graph → Stage 2 frozen-`n_e` S/Fe 파일럿**.

수렴 released-T 앵커가 생성되기 전에는 어떠한 현 CMFGEN 산출물에도 “Jν/rate/population oracle” 또는 “CMFGEN parity PASS”라는 판정을 붙이지 않는다.