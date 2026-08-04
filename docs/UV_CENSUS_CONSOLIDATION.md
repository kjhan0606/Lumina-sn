# UV 편중 이중 전수조사 — 운전석 대조표 (2026-08-02)

입력: docs/CODEX_UV_CENSUS.md + docs/FABLE_UV_CENSUS.md (상호 열람 금지 준수 확인).

---

## ⚠ 0. 정정 고지 (2026-08-03) — 이 문서의 주장 6건이 과장으로 판정됨

`docs/CODEX_CLAIM_AUDIT.md`의 독립 해석 감사에서 이 문서의 주장 **11건이 INFLATED**로
판정됐다. 아래 표가 이 문서 안의 해당 문안과 산출물이 실제로 지지하는 범위다.
**본문은 역사 보존을 위해 원문 그대로 두되, 각 지점에 정정 표지를 달았다.** 인용할
때는 반드시 정정판을 쓸 것.

| 이 문서의 문안 | 산출물이 실제로 지지하는 것 |
|---|---|
| §1 "E8 폐합(이득 5247.49 vs 필요 5247.41)" — 진폭의 직접 원인 확정 | **철회 확정(08-03, rung 1 실측).** 두 5247은 같은 분모 `S_fixed`를 공유하는 **동일 시점 대수 항등식**이고, 판정런 189116이 **반복 연산자의 실제 spectral radius를 최초로 측정**했다: s8에서 `ρ=(χ_es/χ_tot)·λ*=0.98332` ⟹ **실측 증폭 `1/(1−ρ)=59.97`**. E8의 5247이 요구하는 `ρ=0.99981`과 **87배 어긋난다**. 두 양은 서로소 배열(`chi_es`·`lambda_star` vs `eta_fixed`·`eta_coherent`)에서 나온 **독립 2번째 경로**다. ⟹ **5247은 iteration amplification이 아니라 post-EPAY source 조성비이며, "recycle gain"이라는 명명 자체가 오도다.** 상세=대장 `0l` |
| §5 "T1이 재분배 노선 전체를 닫았다 / Stage 3.2가 **유일한 본진**" | T1이 반증한 것은 **shell-8 동결 payload에 제거 에너지를 균일 사전주입한 대리모형 하나**다. T5 자신의 `verdict.json`은 `route_disposition="UNRESOLVED"`. owner-resolved 재분배·수송 내 재분배·다른 Λ 분할은 반증되지 않았다 |
| §5 "조립된 `η_line`이 폐기됨 … s16+ 98.7–100%" | 분모가 pre-EPAY `η_line`이 **아니다**. 실제 측정량은 writer가 rate-shape로 분류한 셀에 놓인 **post-EPAY `eta_fixed`** 몫이며 continuum/deposition을 포함한다. **B1–B4의 정확한 `1.0`은 mask 뒤 분자=분모가 된 항등식**이지 100% 폐기율이 아니다 |
| §5 "T3 T_rad 핀 기각(공범 아님)" | 측정된 것은 **pumping proxy 민감도(0.02–3.4%)와 구조적 τ proxy**다. repaired-pin 상태의 UV field·flux 반사실은 측정되지 않았다 |
| §5 "T4 누락 채널 0.2353% ⟹ 임계 1% 미달" | 분모가 **비무작위 4억 event prefix**(시도 9.71억 중 41.2%만 저장, `TRUNCATED_PREFIX-not-an-unbiased-random-sample`)다. full-iteration 비율과 1% 임계 비교는 `UNRESOLVED` |
| §3 "UV는 42.9%에서 움직인 적이 없다" | "개선 측정이 없다"는 "움직이지 않았다"와 다르다. 미측정 구간의 실제 trajectory는 `UNVERIFIABLE` |
| (별건, §2) Stage 3.1 "MC 수송 단독 결함 기각" (`docs/CODEX_UV_CENSUS.md:28`) | `J_MC`가 live MC estimator가 **아니라** capture sidecar의 결정론적 `J_producer`이고, 비교 대상 `eta_total`에 `chi_es·J_producer`가 이미 들어 있다. **같은 생산자를 양쪽에 놓은 replay-consistency**이므로 MC 수송은 아직 무죄가 아니다 |

**운전석 판정**: Stage 3.2 ALI/MALI는 여전히 **최우선 후보**다. 다만 근거가 "유일
노선이라서"가 아니라 **"반복의 spectral radius가 아직 한 번도 측정된 적 없어서"**로
바뀐다. 그 측정이 rung 1이며, 사전등록 정본은
`patches/stage32_rung1_expected_changes_v2.txt`다. 특히 위 표 첫 행의 5247배는
rung 1의 **사전등록 예측 2가 직접 반증 대상으로 삼고 있다**(독립 배열 경로).

관련: `docs/AUDIT_STAGE32_RUNG1_EPSILON_DISCREPANCY.md`(ε 정의·연산자 짝 판정).

---

## 1. 합치 (양측 독립 도달)

| 쟁점 | 판정 |
|---|---|
| 진폭의 직접 원인 | **결정론 레인의 same-bin coherent line recycling** — `chi_es = chi_e + (chi_ln − chi_ln_th)`, 즉 비열화 선 불투명도를 주파수-결맞음 산란으로 수송. 2준위 형식을 다준위 원자에 적용한 것. E8 폐합(이득 5247.49 vs 필요 5247.41) **⚠철회 확정(08-03): 실측 증폭은 `1/(1−ρ)=59.97`(s8)로 5247과 87배 어긋난다. 5247은 반복 증폭이 아니라 source 조성비다 — §0·대장 0l. `chi_es` 분해 자체는 SUPPORTED** |
| p_iup≈88% | **원인 아니라 결과.** fable 대수: J≈W·B(T_rad)면 비율 ~1e-4인데, J_blue/(W·B)≈5.5e3(재순환 고리의 자기 출력) 때문에 ~50까지 오름. 셸 국소(s3 2.37 vs s16 0.05). Codex: JBLUE 이전 0.9918 → JBLUE 0.9618 → parity26 0.8935 ⟹ **IUP-JBLUE는 누명(오히려 낮춤)** |
| 07-07 이후 신규 도입 아님 | line-eps 분할은 06-11(43e509e) "EXPERIMENTAL, do not enable in production"으로 도입, 07-07에 이미 UV 42.9% 관측 |
| 계보 공백 | 마지막 커밋 47bfa20(07-18). 이후 Wave 1/2/3·Stage 3.1·E1–E13 전부 미커밋(fable: 14,662 insertions) — 커밋 단위 증명 불가 |

## 2. 상보 발견 (한쪽만 — 둘 다 실물)

**Codex 고유:** 올바른 물리가 **이미 구현돼 있고 꺼져 있음** — `LUMINA_CMFGEN_SRC_NLTE`(default OFF): 주석 원문 "pop-ratio S_l is **correct physics (fluorescence)** but is **EXPLOSIVE under the operator split**: saturated FUV lines carry S_l up to ~1e16 × local J … the B(T_e) fallback (**the de-facto thermostat the champion metrics rest on**) stays the default." ⟹ **1년 노브 전멸의 이유는 물리 부재가 아니라 연산자 분할의 수치 실패.**

**fable 고유 (신규 결함 2건 + 구조 갭 1건):**
1. **T_rad ≡ 10470.09 K, 전 50셸·미갱신** — pure-CMFGEN 경로가 `solve_radiation_field`(유일 호출부 cuda.cu:10447)에 도달하지 않음. dilute-Boltzmann up-rate가 잘못된 균일 T_rad로 계산됨
2. **k-packet free-bound 방출이 900Å 미만**에 떨어져 방출 census에 0.0 기여(s0에서 fb exit 1,445건)
3. **형광 행렬의 구조적 불완전** — 호출부가 `cuda.cu:6127` LINE 분기 **하나뿐**이라 bf-활성·k-packet-only 입력이 원천 부재. E11의 "무편향" 행렬도 **채널 커버리지는 미완**
4. MC 레인 자체가 UV→UV: 선 방출의 **92.1%가 2896Å 미만**, macro-atom 열적 파괴 8.97e-6, k-packet의 99.56%가 충돌 재여기로 탈출 ⟹ E10/E12/E13이 실패한 이유는 **R을 MC에서 재어 결정론에 적용했기 때문**(R은 UV→UV 지도를 충실히 인코딩)
5. **"IUP-JBLUE = ARTIS-exact" 라벨 오류** — ARTIS는 `DETAILED_LINE_ESTIMATORS_ON=false`로 출하되어 binned field를 읽음(per-line MC estimator 아님)

## 3. 종합 판정 (운전석)

증상은 **한 원인이 아니라 두 레인의 결함이 겹친 것**이다.
- **결정론 레인**: 정확한 S_l이 있으나 연산자 분할이 감당 못 해 꺼져 있고, 그 자리를 2준위 ε·B + same-bin 재순환이 대신한다 ⟹ **수치 문제(ALI 부재)**
- **MC 레인**: 단일 사이클 UV→UV는 이론과 정합(92% vs 이론 89–98%)이나, 다중 사이클 재처리가 emergent 42.9%를 못 내린다. 게다가 형광 행렬 계측이 LINE 분기만 덮어 bf·k-packet 채널이 빠졌다 ⟹ **계측·채널 커버리지 문제**
- **계보 결론**: UV는 2026-07-06(ff58168) 이후 42.9%에서 움직인 적이 없고, 그 사이 모든 변경은 (a) UV를 측정조차 안 했거나 (b) byte-identical이거나 (c) 역방향이었다. **"수리를 시도했으나 실패"가 아니라 "UV를 표적으로 삼은 변경이 사실상 없었다"** — 이것이 1년 정체의 실체.

## 4. 즉시 실행 가능한 오프라인 시험 (양측 추천 병합)

| 시험 | 내용 | 판별 |
|---|---|---|
| **T1**(fable) | 동결 payload에서 `chi_coherent → chi_e`만 남기고, 제거한 `(1−ε)χ_line·J`를 **균일 R**로 되돌림(E10/E12는 MC-측정 R) | 균일 R에서 형상이 개선되면 **R의 형상이 문제**, 여전히 악화면 **결맞음 가정 자체가 문제** |
| **T2**(Codex) | 같은 인구·연속·격자에서 선 조립만 population-native `chi_l[n_l,n_u]`, `eta_l=A_ul n_u`로 교체(EPAY·경계·solver 불변) | BALL이 O(1)로 붕괴하면 진폭 원인 최종 확정 |
| **T3**(신규, 운전석) | **T_rad 핀 감사** — 10470 K 균일 고정이 up-rate·dilute-Boltzmann에 미치는 영향 정량. 실제 T_rad(셸별)로 재계산 시 p_iup 변화 | p_iup이 크게 내려가면 T_rad 핀이 재순환 고리의 **공범** |
| **T4**(신규, 운전석) | 형광 행렬 **채널 커버리지 확장**(bf·k-packet 입력 포함) 후 R 재측정 | 커버리지 확장으로 B2→B0가 사라지면 계측 불완전이 원인 |

## 5. 상신
1. **Stage 3.2(VEF/ALI)를 UV 수리의 본진으로 승격** — Codex 발견(정확한 물리가 연산자 분할 때문에 꺼져 있음)이 이를 직접 지지. ALI가 서면 SRC_NLTE를 켤 수 있고 ε 해킹이 불필요해진다.
2. **T_rad 핀 수리** — 독립 결함이며 up-rate 전반에 영향.
3. **형광 행렬 채널 확장** — 진단 계기의 완전성 확보.
4. **계보 규율**: UV 밴드 표를 회귀 배터리에 상설 편입(변경마다 UV 측정 의무화) — 1년 정체의 재발 방지.

---
# T1–T4 판정 (08-02 오후)

| 시험 | 판정 | 핵심 수치 |
|---|---|---|
| **T1 결맞음 가정** | **RESOLVED — 가정 자체가 문제** | 균일 R에서도 B0 8.29→25.91(MC-R의 26.43과 동급) ⟹ R 형상 무관. 선 광자 운명은 소스 사전주입으로 흉내 불가 |
| **T2 native χ+η** | UNRESOLVED(계기 부족) | iter-10 하위준위 인구·선별 χ 분해 부재. B2-lane은 η만 교체(χ는 bitwise 동일) |
| **T3 T_rad 핀** | **기각(공범 아님)** ⚠**정정 §0: pumping proxy 민감도만 측정됨. UV field·flux 반사실 미측정 ⟹ 기각 근거 불충분** | 핀 수리 시 UV 상향계수 0.02–3.4%(임계 5% 미달). 주 소비처 2개는 이미 사망(B3가 분배함수를 T_e로·RADEQ_SIMUL이 성운 Saha 차단). 재구성 T_rad 19397K(s0)→11704K(s49), W_mom>1 ⟹ 장이 희석-Planck 아님 |
| **T4 채널 커버리지** | **기각(0.24%)** ⚠**정정 §0: 분모가 비무작위 4억 prefix(저장률 41.2%) ⟹ 1% 임계 비교 UNRESOLVED. rank-1·×7.35도 transported-J 반사실 아님** | 누락(bf·ff-heat) 진입 0.2353%·출구 0.2343%·B0 0.64%, 임계 1% 미달. **실패 기전 특정**: R이 대각 제외 사실상 rank-1(SVD 59.8%), 보편 출력 SED q의 B0 몫 10.53% vs 결정론 방출률 B0 몫 1.43% ⟹ 적용 시 B0 ×7.35는 산술적 필연. "B2→B0 지배"=B2 입력 가중 35.6%의 동어반복 |

## 신규 결함 (등재)
- **N4(잣대·최대)**: `fluor_matrix_iter10`이 **런 중 덮어써짐** — 현 파일 iteration=11/468,330 edge/sha 08ff3312, E12 기록 iteration=10/473,045/sha 2b65dba6. sha 사이드카가 함께 갱신돼 `sha256sum -c`는 PASS하고 **소비자 3종 어디에도 행렬 iteration 계약 없음** ⟹ **E12/E13 재현 불가**(같은 명령이 다른 행렬을 조용히 소비). 결론은 타이밍상 유효하나 정합은 우연.
- **N3**: 준위인구 정규화 불일치(분자 W·e^{−E/kT_rad} vs 분모 Z(T_e,W=1)) ⟹ Σn_k ≠ n_ion. T_rad 핀과 독립.
- **N2**: ff-heat 활성화 경로가 MA-FATE 센서스에 부재.
- CLOSED: U2(linetherm=0은 D4 무력화이지 배선 버그 아님), U3(fb 방출 100%가 λ<900Å = 물리).

## 결론 갱신
재분배-연산자 노선(E10–E13)은 **산술적으로 실패가 예정**돼 있었다(rank-1 R × SED 불일치). T1이 그 노선 전체를 닫았다: **선 전달은 소스 사전주입이 아니라 수송 해 안에서 자기일관되게 풀려야 한다 = ALI**. 따라서 **Stage 3.2(VEF/ALI)가 UV 수리의 유일한 본진**이며, 나머지는 계기 수리(N4·T2 덤프·N2)와 독립 결함 등재(N3)다.

> **⚠정정 (§0)**: "유일한 본진"은 과장이다. T1이 반증한 것은 균일 사전주입 대리모형
> 하나이고, T5의 최종 판정은 `UNRESOLVED`다. ALI는 최우선 후보이되 유일 노선이
> 아니며, 재분배 노선은 닫히지 않았다.

---
# 계기 수리 배치 착지 (08-02) + ★N9

**패치 3종 준비 완료**(patches/instr_{rn4_generation_contract,rt2_linepop_dump,rn2_activation_census}.patch): 3가지 순서 적용에서 byte-identical·역적용 청정·nvcc 링크 성공·신규 경고 0·**신규 clamp 0**·seeded 음성 대조 전건 발화. 생산 트리 무변경(mtime 증명).
- **R-N4**: 생산자가 `<path>.iter%03d`로 쓰고 기존 세대 덮어쓰기 FATAL, 소비자 `read_fluor_matrix`에 **필수 키워드 `expected_iteration`**(계약=10) — 잊으면 TypeError로 죽음. 6개 호출부 재배선. 음성 대조가 사고를 정확히 재현(payload+sidecar 동시 교체 → `sha256sum -c` OK인데 소비자는 거부). **원장 정정**: 계약이 전 소비자에 부재한 게 아니라 `emiss_e12_preregister.py:68`에 있었으나 **사전등록 시점에만, 적용 시점엔 없었음**.
- **R-T2**: `LCMFLP01` 아티팩트(같은 세대·읽기전용 replay·per (line,shell) n_l·n_u·τ·S_l^pop·S_l^used·ε_l·w+flags, `chi_line` bitwise 재현 manifest 단언). 8종 오프라인 대조 PASS(1-ulp 드리프트 거부 포함).
- **R-N2**: ff-heat `d_ma_fate_record`+**가산형 무-cap activation census**(기존 컬럼 무접촉)+co-evolve 레인에서 죽어 있던 밴드 원장 관측화(**N6**). 사전등록 변경집합 +403(무cap)/+165(prefix).

## ★N9 (결정적 — 실험 해석을 바꿈)
parity59 설정(EPAY=2·HOTF=0·SMIN=5)에서 **thin bin·s≥5의 `S_fixed`가 `w_n·(bf_Milne_η + χ_line,th·B(T_e))`로 재구성되어 조립된 `η_line`이 폐기됨.** 실측 하한 **s8 UV 빈의 ≥30.6%, s16+에서 98.7–100%**. ⟹ **η만 바꾸는 모든 시험(E4 B-lane·E5 B2-lane·T2의 population-native η)은 그 영역에서 무효(inert)**였다. R-T2 덤프에 per-cell EPAY disposition 열을 넣어 이를 물리적 null로 오독하지 못하게 함.

> **⚠정정 (§0)**: 분모가 pre-EPAY `η_line`이 아니라 **post-EPAY `eta_fixed`**다
> (continuum·deposition 포함). **s16+의 "98.7–100%"와 B1–B4의 정확한 `1.0`은
> mask 뒤 분자=분모가 되는 항등식**이지 측정된 폐기율이 아니다. 게다가 그 disposition
> 열 자체가 **재구성값**이었다 — writer가 production 분기의 `acc_w > 0`을 누락했다
> (`src/lumina_cmfgen.c:904` vs 실제 분기 `:1704`). branch-site 실측은 rung 1 v4에서야
> 들어갔고, 그 값이 나오기 전까지 이 절의 비율은 **UNRESOLVED**다.
> ⟹ "η만 바꾸는 시험이 무효였다"는 결론도 그만큼 유보된다.

## 기타 신규 등재
- **N8**: `d_ma_fate_band_from_nu`의 밴드 7이 **λ<1700Å과 λ≥10000Å을 한 통에 묶음** — ff-heat exit의 **83%(137/165)가 λ<1700Å(중앙값 1526.5Å)**. 캠페인이 쫓던 IR→FUV 상향 변환이 **대각 원소로 위장**돼 있었음(밴드 재정의는 기존 zihist 비교성 파괴 — 등재만).
- **N7**: `EVCH_MA_ACT_BB`가 어떤 census 버킷에도 도달 안 함(T4가 CAP 절단 로그로만 가중할 수밖에 없던 이유).
- **N3 수리 설계**: 이중 분배함수 A안(`Z_LTE(T_e)` 유지 + 신규 `Z_neb(T_rad,W)` 61kB) ⟹ Σn_k=n_ion by construction. **순위: 이온화·가열 원장 항목이지 UV 진폭 후보 아님**(비-NLTE τ 몫 0.0015%). 사전등록 게이트(R_norm 5%/50%) 후 착수.
