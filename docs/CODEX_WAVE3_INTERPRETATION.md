결론은 이렇습니다. **element-wide 자체는 D-5의 올바른 구조적 표적이지만, 현재 II–IV 파일럿의 acceptance 실패는 solver가 아니라 불완전한 active-stage 보존과 검증되지 않은 채널 물리 내용이 지배합니다.** 다음 수는 **(b) 행렬 채널 물리-내용 감사 선행**이 맞습니다.

## ① s0 Fe의 정합 스토리 순위

| 순위 | 스토리 | 판단 |
|---|---|---|
| 1 | **II–IV 절단 상태에서의 보존행 trade-off** | 가장 강한 설명 |
| 2 | **상위 drain을 복구한 올바른 재분배 + 잘못되거나 미검증인 bf/J/rate 내용** | 높은 개연성, 1번과 공존 |
| 3 | **fixed-T·미수렴 앵커 편향** | 평가값에는 크게 작용하나 내부 재분배의 원인은 아님 |
| 4 | solver/conditioning artifact | 현 증거로는 낮음 |

### 1위 — 보존행 trade-off

명세는 II–IV 행렬에 원소 총량 전체를 넣어 `Σx=n_element`로 강제합니다. 동시에 제외 I/V가 활성 문턱을 넘으면 acceptance를 금지하도록 되어 있습니다. [WAVE3 spec:150](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md:150), [WAVE3 spec:39](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md:39)

여기에는 거의 수치적 지문이 있습니다.

- 조건부 s0 Fe 앵커는 `(II,III,IV)=(9.93e−12, 0.000305, 0.989)`이므로 II–IV 합이 약 `0.989305`, 즉 창 밖 질량이 약 `1.07%`입니다. [ABS_STATE:32](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_ABS_STATE_5154.md:32)
- 실제 s0 EW 진단의 boundary fraction은 `1.3749%`이고 producer coverage는 0입니다. [B2:53](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_B2_TEST.md:53)
- EW가 II/III를 크게 drain하면서도 II–IV 합을 1로 강제하면 남는 질량은 거의 전부 Fe IV로 갑니다.
- 실제 후보 Fe IV 오차 `d=0.00479`는 `|log10(1/0.989)|≈0.00480`과 사실상 같습니다. 즉 Fe IV 악화량이 “II–IV 창을 100%로 채운 결과”와 거의 정확히 일치합니다.

따라서 `D: 2.010→0.842`는 거대한 trace-ion log 오차인 Fe II/III가 줄어 얻은 평균 개선이고, Fe IV 악화는 제외-stage 질량을 IV에 떠넘긴 보존행 부작용이라는 설명이 가장 정합적입니다. 둘은 모순이 아닙니다.

### 2위 — 올바른 구조 재분배, 잘못된 물리 내용

III→IV drain을 동일 행렬에 넣는 구조 자체는 D-5를 직접 제거합니다. C2도 CSR 모든 target 순회와 route probability 1회 적용은 소스 구조 한정 PASS로 보았습니다. [C2:10](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_C2_REVIEW.md:10)

하지만 “어떤 행렬을 풀었는가”는 아직 인증되지 않았습니다.

- 외부 atomic checksum과 비교하지 않음
- 실제 소비되는 `col_ion_*` 자료가 checksum 밖
- sigma 없는 continuum이 expected 분모에서도 빠짐
- NT-BB와 AUTOION/DR 기대 활성 여부가 gate 밖

[C2:3](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_C2_REVIEW.md:3)

또한 동일 projection의 ARTIS matrix dump가 없어 실제 support/rate oracle PASS도 없습니다. [B2:68](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_B2_TEST.md:68)

따라서 가장 그럴듯한 합성 스토리는 다음입니다.

> element-wide가 pair-wise의 잘못된 소유권·복원 보상을 제거해 저이온을 강하게 drain한 방향 자체는 맞지만, active window·bf/collision inventory·복사장 입력 중 하나 이상이 틀려 재분배 크기가 물리적으로 맞지 않는다.

s8에서 S와 Fe의 모든 stage가 악화했고, 지배 ion 절대오차도 크게 증가했다는 점이 이를 뒷받침합니다. [B2:29](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_B2_TEST.md:29) 특히 s8 경계 질량을 보존식보다 관대하게 움직여도 25% PASS로 반전되지 않는다는 상한은, s8 실패가 단순 boundary bookkeeping만은 아님을 보여줍니다. [B2:57](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_B2_TEST.md:57)

### 3위 — 앵커 편향

현재 비교 기준은 fixed-T이며 `MAXCH=3.46×10³%`가 남은 조건부 스냅숏입니다. 작은 Fe II/III 분율을 분모로 한 로그 오차는 특히 불안정합니다. [ABS_STATE:12](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_ABS_STATE_5154.md:12)

따라서 s0의 `D` 개선률과 Fe IV의 근소한 stagewise 실패는 앵커가 바뀌면 쉽게 재배열될 수 있습니다. 다만 앵커 편향은 C2 계약 결손이나 현재 EW 내부의 mass forcing을 설명하지 못하므로 원인 순위는 3위입니다.

### 4위 — 수치해 artifact

잔차 약 `1e−15`, 보존 오차 약 0, 양호한 조건수·pivot, permutation 및 byte 재현성은 “조립된 행렬을 잘못 풀었다”는 설명을 크게 약화합니다. [B2:5](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_B2_TEST.md:5), [B2:66](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_B2_TEST.md:66)

다만 이것은 물리적으로 올바른 행렬이라는 증명이 아니라, 주어진 행렬의 해가 청정하다는 뜻입니다.

## ② element-wide가 여전히 옳은 표적인가

**예. 단, 표적은 “고정 II–IV element-wide”가 아니라 “완전한 active-ion window를 가진 element-wide”여야 합니다.**

Pair-wise의 upper-stage-blind 구조와 save/restore 소유권 문제는 명세상 직접 확인된 구조 결함이고, element-wide가 이를 없애는 방식도 옳습니다. [WAVE3 spec:8](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md:8)

그러나 현재 결과가 보여주는 것은 다음입니다.

- **element-wide 필요성:** 유지
- **현재 II–IV 파일럿의 충분성:** 기각
- **현재 acceptance 결과의 지배 결함:** active-window 보존 + 채널 물리내용 계약
- **J/bf 중 무엇이 지배적인지:** 아직 분리 불가

결함 후보의 현재 우선순위는 다음으로 봅니다.

1. active-stage/boundary closure와 atomic/channel inventory
2. bf estimator·target별 rate·collision/DR 내용
3. frozen 복사장 `Jν/Jbar` 입력
4. global charge/`n_e`·Saha 폐합
5. thermal/RE feedback

`n_e`와 global charge는 Stage 2B, `Jν` 생산자는 Stage 3, thermal/RE는 Stage 4로 명시적으로 동결·분리되어 있습니다. 따라서 Wave 3만으로 이들을 무죄로 만들 수는 없지만, 곧바로 thermal 문제로 귀속할 수도 없습니다. [WAVE3 spec:26](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md:26), [Equivalence plan:177](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:177)

## ③ relT2가 판정을 바꿀 개연성

구분해서 봐야 합니다.

- **§4.5 최종 판정:** relT2 인증 전에는 원천적으로 불가능합니다. [WAVE3 spec:246](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md:246)
- **s0 Fe 축 판정:** 바뀔 개연성이 높습니다. Fe IV의 `d` 차이는 `0.00173 dex`뿐이라 앵커 fraction이 약 수천분의 몇 수준으로 움직여도 pair/EW 우열이 바뀔 수 있습니다. Fe II/III도 극소 분율의 동일가중 로그 오차이므로 `D`가 매우 앵커 민감합니다.
- **s8 S 전체 판정:** 변경 가능성은 중간 정도입니다.
- **s8 Fe 전체 실패 반전:** 상대적으로 낮습니다. 지배 ion 절대오차가 `0.057→0.389`로 벌어졌기 때문입니다. [B2:34](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_B2_TEST.md:34)

절대오차만으로도 anchor 이동의 필요조건을 둘 수 있습니다. 두 후보의 우열을 뒤집으려면 지배 ion 앵커가 적어도 오차 차이의 절반만큼 움직여야 하므로, S는 약 `0.032`, Fe는 약 `0.166` 이상의 fraction 이동이 필요합니다. fixed-T 앵커가 실격 상태이므로 불가능하다고 할 수는 없지만, Fe는 상당히 큰 이동입니다.

relT2 재판정 때는 다음을 다시 계산해야 합니다.

- 먼저 앵커 인증: `FIX_T=F`, moment error 0, correction p95/max, SE·charge·RE 잔차, 마지막 3회 `T_e,n_e,Jν,L` 안정성, 광도보존, clamp 0. [Equivalence plan:147](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md:147)
- s0/s8/s20의 **signed** I–V 이온분율과 active-set 합집합
- 기존 `d_k`, 동일가중 `D`, 지배 ion 절대오차를 모두 유지하되 mass-weighted 오차도 보조로 병기
- frozen `T_e,n_e,Jν`에서 채널별 Γ·재결합·충돌 rate support/ratio
- 전체 active SL의 `b_k` median/p95와 residual-vector. 공식 문턱은 [WAVE3 spec:248](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md:248)에 있습니다.

relT2는 acceptance 표적을 바꿀 수 있지만 C2의 topology/계측/atomic projection 계약 FAIL을 수리하지는 않습니다.

## ④ 다음 수

**선택: (b) 행렬 채널 물리-내용 감사 선행.**

이유는 세 가지입니다.

1. solver 오차는 이미 충분히 배제됐습니다.
2. C2의 네 계약 FAIL 중 fail-closed layout과 카운터 정직성은 중요하지만, 성공한 ON 해의 물리적 방향을 직접 설명하지는 않습니다. [C2:17](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_C2_REVIEW.md:17), [C2:24](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_C2_REVIEW.md:24)
3. 반면 checksum 누락, continuum 분모 제외, 불완전한 process inventory와 C48 projection은 “무엇을 풀었는가” 자체를 바꿀 수 있습니다. [C2:30](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_C2_REVIEW.md:30)

따라서 감사 순서는 `identity/active window → channel support → rate coefficient·units·probability → signed stage flux → solution`이어야 합니다. 감사 결과로 물리적으로 영향을 주는 계약 항목을 특정한 다음 수리·재측정해야 합니다.

`(c) relT2 hold`는 권하지 않습니다. relT2를 기다리지 않고도 matrix-content와 boundary closure는 판정할 수 있습니다. `(d)` 역시 현재 불명확성을 다른 셀/원소로 확산할 가능성이 큽니다.

## ⑤ 새 런 없는 최저비용 판별 측정 1개

**기존 s0 Fe solution에 대한 “boundary-mass-only 재정규화” 한 번**을 권합니다.

기존 EW의 II–IV 분율을 새 solve 없이

\[
p^{\mathrm{corr}}_k=(1-f_{\mathrm{boundary}})\,p^{\mathrm{EW}}_k,
\qquad f_{\mathrm{boundary}}=0.013749
\]

로 재정규화하고, Fe IV의 signed ratio·`d_IV`와 전체 `D`만 다시 계산합니다.

- Fe IV가 pair보다 좋아지고 II/III의 큰 `D` 개선이 유지되면: **수수께끼는 거의 보존창 trade-off**입니다.
- Fe IV가 계속 악화하면: **bf/J/rate 내용 또는 앵커 편향**이 추가로 필요합니다.

이는 기존 CSV와 이미 기록된 boundary fraction만 쓰며, rate 재계산·새 solve·새 모델 런이 필요 없습니다. 단, B2가 지적했듯 boundary rate/heating feedback까지 설명하는 측정은 아니며, 순수 질량 bookkeeping 가설만 판별합니다. [B2:59](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE3_B2_TEST.md:59)

지정하신 해석 문서는 열람하지 않았고, 문서 확인 외 수정·테스트/모델 실행·git 작업은 하지 않았습니다.