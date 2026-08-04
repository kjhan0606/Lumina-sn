# Clamp 도입-사유 고고학 보고서

작성일: 2026-07-31  
대상 정본: `docs/CLAMP_FIX_PRIORITY_REGISTRY.md`의 94개 ID

## 판정 규칙과 증거 범위

- **(A) 상위버그 전파 차단:** 상류의 물리·배선·자료 결함이 만든 증상을 하류에서 바꿔치기하거나 잘라낸 경우다. 레지스트리 `upstream_bug`에 특정 결함을 기록했다.
- **(B) 물리 오해:** 물리적으로 허용되는 상태를 금지하거나, 진단용 가정을 물리 해법으로 대신한 경우다.
- **(C) 정당 수치 위생:** 정의역·확률측도·보존측도·유한 표현범위를 보호하고 오차를 유계화한 경우다.
- **(D) 사유 불명:** 현재 코드 주석, 도입 커밋, `docs/*.md`, clamp census 어느 쪽에도 그 수치나 선택을 정당화하는 기록이 없는 경우다.

판정에는 요청된 다섯 증거원만 사용했다. 코드 위치는 레지스트리의 `file:line`을 정본 앵커로 삼고, `git blame`에서 확인되는 도입 커밋을 대조했다. 현재 작업트리에서 `00000000 (Not Committed Yet)`로 나타나는 사이트는 아래에 **uncommitted**라고 명시했다. 대장 근거는 다음과 같다.

- `validation/cmfgen_toy06_19p48d/analysis/clamp_census/INNOCENT.md`
- `validation/cmfgen_toy06_19p48d/analysis/clamp_census/ADVERSARIAL.md`
- `validation/cmfgen_toy06_19p48d/analysis/clamp_census/CANONICAL.md`
- `docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md`
- `docs/VERIFICATION_REGISTERS.md`
- `docs/DETERMINISTIC_EMERGENT_BUGTABLE.md`
- `docs/CODEX_TUNING_CLAMP_CONSULT.md`

메모리류 문서는 사용하지 않았다. 기록이 없는 수치에는 사후 물리 해석을 만들어 붙이지 않았다.

## 분류 총계

| 위험도 | A | B | C | D | 합계 |
|---|---:|---:|---:|---:|---:|
| 위험 | 22 | 14 | 8 | 1 | 45 |
| 조건부 | 8 | 2 | 24 | 4 | 38 |
| 정당 | 0 | 0 | 11 | 0 | 11 |
| **합계** | **30** | **16** | **43** | **5** | **94** |

## 위험 45건 전수 판정

표의 “도입 근거”는 도입 사유를 실제로 말하는 주석·커밋·대장만 적었다. 단순히 현재 코드에 존재한다는 사실은 도입 사유 증거로 승격하지 않았다.

| ID | 판정 | 규명된 도입 사유와 직접 근거 |
|---|---|---|
| C08 | A | `lumina_cmfgen.c:177-234`는 ε를 `C/(C+Aβ)`로 정의하면서 collision/ε table 미구축 시 thermal로 대체한다고 주석화한다. 도입 커밋 `43e509e`의 “physical line-eps machinery”가 같은 목적을 명시한다. 즉 ε 자료 결손을 열화로 가린다. |
| C09 | A | `lumina_cmfgen.c:227` 주석은 NLTE 선원함수가 operator-split 폭주를 일으켜 `B(T_e)`를 “de-facto thermostat”로 썼음을 기록한다(도입 계열 `ac02dfc`). ADVERSARIAL D-3은 27,776건이 실제 clamp 발화가 아니라 NLTE-network-out 이온의 미기록 source 센티널임을 확인한다. |
| C10 | C | `lumina_cmfgen.c:269-518` 주석과 도입 커밋 `8e97890`은 EPAY를 k-packet mirror의 radiative-equilibrium 고정점 및 energy-paid 재척도로 규정하고 “tuning knob가 아님”을 명시한다. 보존측도 위생이다. |
| C13 | A | `lumina_cuda.cu:1543-52`, `lumina_plasma.c:16045-68`의 음수-pop 치환은 cold/near-singular NLTE 해를 소비자에 넘기지 않기 위한 후처리다. `c0fba08`가 LTE floor로 solver 쓰레기를 대체했다고 명시한다. |
| C14 | A | LTE-relative repair는 `c0fba08`의 “orthodox NLTE closure”로 들어왔고, 코드 주석은 negative/sub-resolution population 및 ill-conditioned solve를 사유로 든다. 행렬 결함의 증상을 LTE 쪽에서 가린다. |
| C22 | A | 코드에서 분모 컷과 `tau=1e-100` 센티널이 구조적으로 짝지어져 있다. ADVERSARIAL D-3은 컷 단독 발화 0회 및 실제 원인이 NLTE-network-out source 미기록임을 확정했다. |
| C23 | C | `lumina_bf_gemm.cu:83-93`의 `nion<1e-30`, `U<1e-300` skip과 `lumina_plasma.c:1853`의 division-by-zero 주석은 Boltzmann underflow와 영분모를 막는다. 절대 기여가 표현 한계 아래인 항만 제외하는 정의역 보호다. |
| C24 | A | 원자 로더·GEMM 경로는 σ_bf 미등록 시 Kramers를 택한다고 직접 주석화한다. 실제 단면 자료/평가기 커버리지 결손을 근사 단면이 가린다. |
| C38 | C | 해당 밀도 guard 주석은 NaN 검출, 양의 정의역, 뒤이은 conservation renormalization을 명시한다. 고정 물리 해를 주입하는 용도가 아니라 정규화 측도의 유한성을 지키는 용도다. |
| C40 | A | `b60ecf3`는 이전 endpoint commit이 “audit defect”였고 outer strip attractor/self-illumination을 만들었다고 기록하며, no-root 때 직전 `T_e` HOLD를 넣었다. 500/1000 K 대체도 thermal-root 실패를 하류에서 봉합한다. |
| C46 | A | `lumina_plasma.c:1246-90,1520`은 비유한 또는 `W>1e4`인 장 적합을 재시도하거나 0으로 만든다. `docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md`의 G-3(24-bin 기하 비충실) 및 C1 rail/fit 퇴화와 직접 같은 상류 장 경로다. |
| C47 | B | 고정 `T_rad`, TEPIN, W cap은 실제 radiation field가 아니라 지정 온도·희석장 범위를 해로 강제한다. 코드의 gate 설명은 진단/고정장 가정만 제시하며 물리적으로 그 범위를 보장하는 근거가 없다. |
| C49 | A | `lumina_atomic.c` 로더 주석은 누락·불일치 자료를 0, Kramers, ground-only, Axelrod로 대체한다. 원자 스키마/자료 결손의 직접 마스킹이다. |
| C59 | B | H1 코드·검증 기록은 실제 per-ion σ가 로드되어도 k-packet fb가 Kramers 단일 대표 edge를 사용했음을 확인하고, 뒤에 multi-edge gate가 추가됐다. “한 대표 edge”를 실제 Milne 다중-edge 물리로 간주한 구현 자체가 오류다. |
| C64 | A | isolated/top-stage 행을 Boltzmann anchor로 교체하는 주석은 닫히지 않는 행을 풀기 위한 대체다. 함수감사 D-5가 shared lo-ion save/restore와 upper-stage-blind matrix를 확정했으므로, 이 anchor는 D-5/고립행 증상을 막는다. |
| C66 | A | `lumina_cuda.cu:9370-82` 주석은 run 165510의 operator-split J̄→population 폭주를 containment하기 위해 `S/B>100` rollback과 J̄ 차단을 넣었다고 명시한다. |
| C68 | B | `ef62067` 계열의 thick-line `S_l→B(T_e)`는 superthermal source 가설을 판별하는 diagnostic clamp다. 열적 선원함수를 일반 thick-line 물리로 치환하므로 생산 해법으로는 물리 오해다. |
| C69 | B | `8caa4f1` 커밋 메시지부터 “IGE-blanketing falsifier”로 도입됐다. 실제 IGE forest opacity를 0으로 만드는 진단 가정이며 물리 해법이 아니다. |
| C70 | B | `92afcdf` 계열 주석은 Fe 창의 source multiplier를 oracle falsifier로 규정한다. 물리율에서 유도되지 않은 배율이므로 진단 외 사용은 물리 오해다. |
| C71 | B | line re-emission을 `B(T_e)`로 바꾸는 `LINE_THERM` 경로는 line source의 non-LTE 성분을 없앤다. 코드·런처에는 진단 gate 근거만 있고 열화가 정확한 물리라는 근거는 없다. |
| SC13 | A | atomic-data 생성 스크립트는 데이터 부재 때 cap/Kramers/skip을 택한다. 일부 현재 줄은 **uncommitted**이며, 주석이 자료/evaluator 결손을 직접 사유로 기록한다. |
| SC16 | B | prototype가 `gbar=0.2`, `f≥1e-6`을 placeholder로 고정한다. 실제 원자량에서 유도된 값이 아니라는 코드 표기가 직접 근거다. population 비음수화만 따로 보면 C지만 한 census 항목의 지배 동작은 placeholder 물리다. |
| SC21 | A | 생성 단계에서 level cap 밖 transition과 nonpositive Ω를 버린다. `expand_atomic_data_cmfgen.py`의 FULL_LEVELS 관련 주석·검증은 cap과 downstream 자료 불일치를 원인으로 특정한다. 관련 현재 변경 일부는 **uncommitted**다. |
| C15 | A | FLOORM LTE floor 주석은 음수/작은 population과 `b_k` cap 뒤 rescale을 위한 “FIX-1”이라고 명시한다. 해당 현재 구현은 `git blame`상 **uncommitted**이며, C13/C14와 같은 ill-conditioned solve 마스킹이다. |
| C16 | A | 도입 커밋 `ef62067`의 제목이 “super-thermal cure: b_k departure ceiling”이며, 코드 주석은 약 1–50이어야 한다는 가정으로 `1e20`대 solver garbage를 자른다고 말한다. 상류 행렬 ill-conditioning의 증상 억제다. |
| C27 | A | 빈/창밖 J를 0 또는 fallback으로 바꾸는 각 주석은 estimator/frequency window 미커버를 조건으로 한다. 함수감사 G-3의 24-bin geometry mismatch와 동일한 장 커버리지 결함군이다. |
| C28 | B | `J_CAP/FLOOR_FACTOR`는 unconstrained pumping을 `WB` 범위에 넣어 보는 진단이라고 코드/런 스크립트가 설명한다. ADVERSARIAL의 2000–4000 Å 분석은 참여 라인의 53–58%에서 `S_l>B(T_e)`가 실제 NLTE 성질임을 보여, `J≤WB` 가정은 물리 법칙이 아니다. 신뢰할 도입 커밋 사유는 발견되지 않았다. |
| C29 | B | `lumina_plasma.c:13706-68` 및 `9a4e91d` 계열은 2000–3500 Å의 Jν를 `W_cap Bν`로 눌러 σ_bf gap을 공격한다고 적는다. super-Planckian J를 금지하는 물리 근거가 없으므로 clamp 자체가 오류다. |
| C32 | B | 코드 순서상 real close-coupling Υ를 읽은 뒤 floor가 덮어쓴다. 원 floor는 `b60ecf3`에서 valley coolant 보완으로 도입됐고, real-Υ 뒤에도 적용하는 현재 parity arm은 **uncommitted**다. ADVERSARIAL D-1은 2,278,264/2,584,132 = **88.16%** 발화와 median 약 **253×** 상향을 확인했다. 누락 자료 보완이 아니라 실측 Υ를 발명된 하한으로 치환한다. |
| C34 | A | gbar/Axelrod/forbidden Ω fallback 주석은 collision-strength 자료가 없을 때만 적용됨을 명시한다. 원자 충돌자료 커버리지 결손을 경험식으로 막는다. |
| C36 | A | `9a4e91d`의 주석은 uniform α_DR floor를 “would any extra recombination close the gap?” 진단으로 규정하고 Mazzotta LS-coupling의 low-T near-threshold resonance 누락을 명시한다. K6/저온 DR 자료 결손 마스킹이다. |
| C37 | C | 이온비·연쇄곱 cap 주석은 overflow 방지 후 conservation renormalization을 전제로 한다. 표현범위 보호이고 결과 측도는 다시 정규화된다. |
| C44 | A | `9a4e91d` 계열 `COOL_NONNEG` 주석은 spurious NLTE inversion이 coolant 부호를 뒤집는 것을 막는다고 한다. 같은 파일의 후속 주석은 lagged/non-SE population을 원인으로 적는다. |
| C45 | B | 코드 주석은 upper population의 LTE ceiling 때문에 genuine line heating까지 막힐 수 있음을 스스로 기록한다. deterministic J 아래의 비LTE pumping을 금지한 clamp 자체가 물리 오해다. |
| C48 | A | `66586d6` 및 `lumina_atomic.c:723-39` 주석은 full O II/Co III span이 약 250 orders여서 행렬이 ill-conditioned해져 `SUPER_CUTOFF` lump를 썼다고 명시한다. 수치행렬 결함의 물리 증상을 준위 절단으로 가린다. |
| C52 | A | `9a4e91d` 주석은 dense opacity의 unbounded interaction으로 인한 iter3 hang을 이유로 cap을 넣었다. 같은 주석이 legacy counter가 boundary crossing까지 세고 cap-hit packet energy를 삭제했음을 명시한다. ADVERSARIAL P9은 삭제 packet을 **343개**로 확정했다. 비보존/계수 오배선 마스킹이다. |
| C53 | D | `9a4e91d`의 현재 주석은 5000이 “old hard constant”였다고만 하고, 작은 cap은 diagnostic/non-physical이라고 명시한다. 그러나 원래 5000의 선택 근거나 오차한계는 코드·선행 history·문서·census에서 찾지 못했다. |
| C65 | A | `STAGE4_BK_CAP` 주변 주석은 stage-IV III-combination에 continuum/metastable drain이 없어 `b_k`가 솟는 것을 cap으로 핀했다고 적고, drain 구현 뒤 cap을 퇴역시켜야 한다고 기록한다. 현재 줄은 **uncommitted**다. |
| SC04 | C | decay 뒤 isotope mass fraction을 비음수화하고 총량을 재정규화하는 코드다. 조성 simplex의 round-off 이탈만 되돌리는 측도 보호다. |
| SC05 | B | 86개 launcher가 외곽 `X_Fe≥5e-4`를 강제한다. 코드·문서 어디에도 실제 조성이 그 하한을 가져야 한다는 근거가 없고, 조성 자체를 바꾸는 바닥값이므로 물리 오해다. |
| SC06 | C | continuum 정규화 분모를 peak의 1% 이상으로 두는 report metric이다. 선속을 바꾸지 않고 near-zero continuum에서 점수 폭주만 유계화한다. |
| SC08 | C | `empirical_pcygni_ml.py` 주석은 `tau>8`의 `exp(-tau)` 꼬리가 3.4×10⁻⁴ 이하로 포화됨을 적고 kernel width 정의역을 보호한다. 명시적 오차 유계다. |
| SC11 | C | χ clip과 최소 MC tolerance는 검증 점수/통계 오차의 정의역만 제한하고 production state를 바꾸지 않는다. INNOCENT의 baseline no-fire 검산과도 모순되지 않는다. |
| SC12 | B | level cap은 GPU memory를 이유로 도입됐지만, FULL_LEVELS 검증 주석·설계문서는 잘린 levels가 이온별 Γ의 20–95%를 운반함을 기록한다. 물리적으로 무시 가능하다는 가정이 반증됐다. 현재 확장 작업 일부는 **uncommitted**다. |
| SC15 | B | offline ETLA prototype의 `n_upper≤n_LTE`는 코드상 “no-pumping guard”다. 비LTE pumping을 금지하므로 production 물리로는 잘못된 가정이다. |

## 조건부 38건과 정당 11건

조건부군은 요청 범위대로 코드 주석과 설계문서 수준에서 판정했다.

| 판정 | ID | 주석·문서가 말하는 도입 사유 |
|---|---|---|
| A (8) | C17, C19, C25, C26, C33, C35, C60, C63 | 각각 NLTE solve 실패, cold matrix conditioning, ionization energy 부재, MC crossing 미달, Ω 미등록, collision data/conditioning, C2 estimator·C1 provenance 결손, rate-field/upper-ion routing 결손을 fallback/floor가 가린다. C60과 관련 current path 일부는 **uncommitted**다. |
| B (2) | C21, C61 | maser/inversion opacity를 0으로 만드는 가정(C21), spin selection과 실제 target term을 동일시한 재결합 gate(C61)다. `docs/VERIFICATION_REGISTERS.md`의 Y4는 C61의 selection-rule/target mismatch와 FUV 악화를 확인한다. C61 current gate는 **uncommitted**다. |
| C (24) | C04, C06, C11, C12, C18, C20, C30, C31, C39, C41, C42, C43, C51, C56, C57, C62, C67, SC02, SC07, SC09, SC10, SC14, SC17, SC19 | 표 endpoint, 비음수 opacity, ALI 분모, workspace, negligible-pair skip, sentinel, 반복 감쇠/trust region, 정적 표현범위, sampler tail, provenance/wiring, ODE simplex, 회귀 정의역, fit positivity, 검증 mirror/tolerance다. C31·C62·C67 current 구현은 **uncommitted**이며, C67은 ADVERSARIAL D-5의 +6.8% formal quadrature 및 continuum/τ/S provenance를 수리하는 측도 위생이다. |
| D (4) | C54, C58, C72, SC18 | total-loop ceiling, bf grid 밖 0/last-bin 유지, impact-ray 기본 50, report nearest-neighbor의 선택값·오차상한을 설명하는 기록이 없다. C72는 100→env/default 50 변경 사실만 기록돼 있다. |
| C 확인 (정당 11) | C01, C02, C03, C05, C07, C50, C55, C73, SC01, SC03, SC20 | RNG/지수/Planck/escape 점근식/확률·sqrt/표현범위/계기 저장량/test·plot 정의역 보호다. `INNOCENT.md`의 8개 baseline no-fire 검산은 이 계열이 정상영역 물리를 바꾸지 않음을 확인한다. |

## A형 상위버그별 군집

한 ID가 둘 이상의 상류 결함과 닿을 때는 실제 일괄 제거 판단에 유용한 주 군집에 배치했다. 아래 묶음은 **상위 결함을 먼저 수리한 뒤 동시 제거·A/B 검증할 후보**이지, 선행 삭제 승인 목록이 아니다.

| 상위 결함 군집 | A형 clamp | 상위 수리 후 일괄 제거 관점 |
|---|---|---|
| NLTE 행렬 singular/ill-conditioning·solver 실패 | C13, C14, C15, C16, C17, C19, C48 | full-rank/conditioning과 residual 실패 처리를 수리하면 LTE floor, b ceiling, Boltzmann fallback, superlevel 절단을 한 묶음으로 퇴역 검증할 수 있다. |
| NLTE network/source coverage·operator-split feedback | C09, C22, C40, C44, C66 | network-out source를 명시하고 선원함수–population feedback을 안정화하면 thermal fallback, zero sentinel, HOLD/rollback, negative-cooling floor가 함께 제거 후보가 된다. |
| 원자 자료·평가기 커버리지 | C08, C24, C25, C33, C34, C35, C49, SC13, SC21 | ε, σ_bf, χ, Ω와 loader schema를 완전하게 만든 뒤 Kramers/gbar/Axelrod/skip/floor를 일괄 제거한다. |
| radiation-field sampling·provenance·geometry | C26, C27, C46, C60, C63 | G-3/C1-C2 빈·창·field routing을 수리하면 binned fallback, field zero/refit, rate prior를 함께 퇴역시킬 수 있다. |
| D-5 상위-stage-blind/continuum drain 부재 | C64, C65 | upper-stage drain과 metastable continuum 연결을 고치면 Boltzmann anchor와 stage-IV `b_k` cap을 동시에 제거한다. |
| 저온 DR resonance/K6 자료 결손 | C36 | 실제 ion별 저온 DR 자료를 채운 뒤 uniform α_DR floor를 제거한다. |
| transport interaction 계수·비보존 packet drop | C52 | boundary/event 계수를 분리하고 cap-hit energy를 보존한 뒤 legacy truncation 경로를 제거한다. |

대조 결과, A형을 함수감사 **D-1·D-2·D-3·D-4**에 직접 귀속할 도입 기록은 발견되지 않았다. D-1과 인접한 C59는 “마스킹”이 아니라 대표-edge 물리 오해(B)이며, D-2는 상태보고 결함, D-3/D-4는 해당 clamp의 도입 사유라는 증거가 없다. 추측을 피하기 위해 이 네 결함에 A형을 억지 배정하지 않았다.

## B형 목록 — 즉시 수정 후보군

| 처리 성격 | ID | 이유 |
|---|---|---|
| production/default에서 제거·정물리로 교체 | C21, C32, C45, C47, C59, C61, SC05, SC12, SC15, SC16 | 물리적으로 가능한 maser·superthermal population/field를 금지하거나, 실제 자료·조성·다중-edge·준위계를 인공 가정으로 치환한다. |
| 진단 falsifier로만 격리하고 production 영향 금지 | C28, C29, C68, C69, C70, C71 | 목적 자체가 pumping/source/blanketing 가설을 판별하는 비물리 A/B arm이다. 삭제보다는 default-off·결과 배너·production 불침범을 검증해야 한다. |

최우선은 발화율과 왜곡이 계량된 **C32**(88.16%, median 약 253×), 실제 multi-edge 경로가 이미 존재하는 **C59**, 준위 기여 20–95% 누락이 확인된 **SC12**다.

## D형 목록 — 기록 보강 전 보류

| ID | 불명인 것 | 필요한 최소 증거 |
|---|---|---|
| C53 | legacy MA cascade cap 5000의 선택 근거와 절단 오차 | 최초 도입 기록 또는 cap-depth 수렴/energy 보존 계량 |
| C54 | total step/CPU loop ceiling의 물리·수치 오차상한 | cap-hit 분포와 미완 packet 측도 |
| C58 | bf grid 밖 0 및 마지막 bin 유지의 경계조건 근거 | grid-tail 적분 오차 또는 원 설계문서 |
| C72 | formal impact-ray 기본 50의 구적 오차상한 | n-impact 수렴표; ADVERSARIAL D-5는 n=100에서도 +6.8% 영점 편향을 보임 |
| SC18 | CMFGEN depth/frequency nearest-neighbor의 허용 보간 오차 | interpolation 비교 및 report 오차예산 |

## 검산

- 레지스트리 데이터 행: **94**
- ID 집합: 누락 **0**, 중복 **0**, 추가 **0**
- 위험도 합: 위험 **45**, 조건부 **38**, 정당 **11**
- 도입사유 합: A **30**, B **16**, C **43**, D **5**
- `intro_reason`: 94행 모두 기입
- A형 `upstream_bug`: 30행 모두 기입
- 비-A형 `upstream_bug`: 전부 `—`
- 기존 물리검증 join 4열: 94행 모두 공란 유지
