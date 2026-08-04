# Codex A′-rev2 — KA3 가드 재설계 판정 보고서

결론부터 말하면, 운전석 가설은 절반만 맞습니다.

- 관측 실패 좌표 `k=98, segment=47`에 정확한 cell-average Gaussian 이력을 대입하면 `η_eff`는 음수가 아니라 양수입니다. 따라서 현재 실패를 “정확해도 피할 수 없는 BDF2 구조적 현상”으로 설명할 수 없습니다.
- 그러나 같은 coarse 격자의 더 먼 Gaussian 꼬리에서는 정확해 자체도 `η_eff<0`을 만듭니다. 따라서 `η_eff≥0`은 연속 문제의 불변량이 아니며, fail-closed 판정에서 진단 카운터로 강등하는 분류 자체는 수학적으로 타당합니다.
- 다만 가드만 강등하면 KA3가 통과하지 않습니다. 독립 signed recurrence에서 바로 다음 segment에 실제 이산해가 `I=-2.2011e-8`로 음수가 됩니다. 원 acceptance 역시 `p_obs≈1.33`, finest profile L2 `≈2.99e-3`로 실패할 것으로 예측됩니다.

즉, 가드 재분류는 맞지만 “스킴과 acceptance를 그대로 둔 채 KA3를 살린다”는 결론은 성립하지 않습니다.

## 1. 관측 좌표 산술 검증

입력 기록은 [ka3.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_results/ka3.json)과 [구현 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_IMPL2.md:90)에 일치합니다.

균일 격자에서

\[
\Delta x=\frac{0.74}{127}
=0.005826771653543307,\qquad
a=\frac{A}{L}=10^{-11}\ {\rm cm}^{-1}.
\]

관측 수치해 이력은

\[
I_{k-1}=1.1251344515546073\times10^{-7},
\]
\[
I_{k-2}=5.0547558929601476\times10^{-7},
\]
\[
R_{\rm obs}=\frac{I_{k-2}}{I_{k-1}}
=4.492579429930308>4.
\]

따라서

\[
2I_{k-1}-\frac12I_{k-2}
=-2.771090433708592\times10^{-8},
\]

\[
\eta_{\rm eff}
=\frac{a}{\Delta x}
 \left(2I_{k-1}-\frac12I_{k-2}\right)
=-4.7557903389323\times10^{-17}.
\]

이는 roundoff 부호 반전이 아니라 기록된 수치 이력의 산술적 결과입니다.

### 같은 좌표의 정확한 Gaussian 이력

`segment=47`의 downstream history node는

\[
r=1.421875\times10^{10}\ {\rm cm},\qquad
\theta=a(r-r_{\rm in})=0.0421875.
\]

cell-average Gaussian을

\[
\bar G_h(x;c)=
\frac{\sigma\sqrt{\pi/2}}{h}
\left[
\operatorname{erf}\frac{x+h/2-c}{\sqrt2\sigma}
-\operatorname{erf}\frac{x-h/2-c}{\sqrt2\sigma}
\right]
\]

라 하면 정확해는

\[
I^{\rm ex}(x,r)=e^{-3\theta}\bar G_{\Delta x}(x;-\theta).
\]

같은 이력 주파수에서 80-digit 산술로 얻는 값은

\[
I^{\rm ex}_{96}=4.7547305910730000\times10^{-6},
\]
\[
I^{\rm ex}_{97}=2.2972677906250876\times10^{-6},
\]
\[
R_{\rm ex}=2.069732841106537<4.
\]

따라서

\[
2I^{\rm ex}_{97}-\frac12 I^{\rm ex}_{96}
=2.2171702857136751\times10^{-6},
\]

\[
\eta_{\rm eff}^{\rm ex}
=+3.805143598454551\times10^{-15}.
\]

판정은 명확합니다.

> 관측 좌표에서 정확해는 음의 `η_eff`를 재현하지 않는다. 관측 실패는 정확 Gaussian 꼬리의 필연적 감쇠가 아니라 수치 이력이 정확해보다 과도하게 감쇠된 결과다.

독립 signed recurrence는 현재 solver의 최초 실패를 `≈5×10^-29` 절대차로 재현했습니다. 따라서 좌표 해석이나 로그 항의 오독은 아닙니다.

### 그러나 `η_eff≥0`은 정확 문제의 불변량이 아니다

전체 coarse exact profile을 같은 이력식으로 조사하면:

- 최초 exact 위반: `k=122`, inner boundary
- \(R=4.0117088489\)
- \(\eta_{\rm eff}^{\rm ex}=-8.2393\times10^{-32}\)
- exact 음수 node-plane 수: 36
- 최대 비: `k=127`, inner boundary, \(R=4.4599857352\)
- 최소 exact `η_eff`: `k=123`, 첫 center, `-1.85994e-31`

반면 `Nnu=256,512`에서는 exact negative가 0개입니다.

따라서 `η_eff` 부호는 BDF2 표현의 격자 의존적 단조성 조건이지, 연속 방정식의 물리 불변량은 아닙니다. 가드 강등은 이 더 일반적인 이유로 정당화됩니다. 다만 `k=98`의 훨씬 큰 실패를 정상 exact-tail 현상으로 해석해서는 안 됩니다.

## 2. BDF2 양수 한계 일반식

\(t=-x\), 현재 step \(h_k=t_k-t_{k-1}\), 이전 step \(h_{k-1}\), step ratio

\[
\rho=\frac{h_k}{h_{k-1}}>0
\]

로 두면 비균일 BDF2는

\[
\frac{dI}{dt}\bigg|_k
=
\frac1{h_k}
\left[
\frac{1+2\rho}{1+\rho}I_k
-(1+\rho)I_{k-1}
+\frac{\rho^2}{1+\rho}I_{k-2}
\right].
\]

따라서

\[
\chi_{\rm eff}
=\chi+3a+
\frac{a}{h_k}\frac{1+2\rho}{1+\rho},
\]

\[
\eta_{\rm eff}
=\eta+
\frac{a}{h_k}
\left[
(1+\rho)I_{k-1}
-\frac{\rho^2}{1+\rho}I_{k-2}
\right].
\]

진공 KA3에서 \(\eta=0\)이면

\[
\eta_{\rm eff}\ge0
\iff
\frac{I_{k-2}}{I_{k-1}}
\le
\left(\frac{1+\rho}{\rho}\right)^2.
\]

균일 격자 \(\rho=1\)에서는 정확히

\[
I_{k-2}/I_{k-1}\le4.
\]

이전 step에서의 유효 감쇠율을

\[
\lambda_k=
\frac{\ln(I_{k-2}/I_{k-1})}{h_{k-1}}
\]

로 정의하면 양수 한계는

\[
\lambda_k h_{k-1}
\le 2\ln(1+1/\rho),
\]

또는

\[
\lambda_k h_k
\le2\rho\ln(1+1/\rho).
\]

물리 방출률이 양수일 때의 완화된 정확 조건은

\[
\frac{I_{k-2}}{I_{k-1}}
\le
\frac{(1+\rho)^2}{\rho^2}
+
\frac{1+\rho}{\rho^2}
\frac{h_k\eta}{aI_{k-1}}.
\]

이는 임의 문턱이 아니라 BDF2 계수에서 직접 유도된 한계입니다.

## 3. 가드 재분류

### `η_eff` 부호

다음은 진단 전용으로 강등합니다.

- `bdf_eta_negative_count`
- `bdf_eta_min`
- 최초 `(evaluation, ray, segment, substep, endpoint, k)`
- `I[k-1]`, `I[k-2]`
- 감쇠비 \(R\)
- 해당 step ratio의 이론 한계 \(R_{\max}\)

유한한 음의 `η_eff`는 clamp/floor 없이 그대로 signed SC update에 전달합니다. `η_eff`가 non-finite이면 즉시 실패합니다.

### 실제 해 \(I\)의 음수

임의 `-1e-N` 문턱을 두어서는 안 됩니다. 각 update에 roundoff enclosure를 전파해야 합니다.

SC update를

\[
I_d=E I_u+w_uH_u+w_dH_d
\]

로 쓰고, \(H=\eta_{\rm eff}\), \(E=e^{-\Delta\tau}\), \(w_{u,d}=\psi_{u,d}/\chi_{\rm eff}\)라 하면 오차 반경은 최소한

\[
B_d =
|E|B_u+|w_u|B_{H_u}+|w_d|B_{H_d}
+\gamma_m\left(
|EI_u|+|w_uH_u|+|w_dH_d|
\right)+B_{\rm kernel},
\]

\[
\gamma_m=\frac{mu}{1-mu},\qquad u=2^{-53}
\]

형태로 실제 연산 그래프의 \(m\)을 사용해 계산합니다. 가장 안전한 구현은 각 항을 directed-rounding interval로 평가하는 것입니다.

판정은 다음과 같습니다.

- \(\hat I+B_I<0\): 음수가 수치오차 한계를 초과했으므로 `LCMF_ENEGATIVE`.
- \(\hat I-B_I>0\): 양수 확정.
- \(0\in[\hat I-B_I,\hat I+B_I]\): 부호 미결정이므로 별도 numerical-uncertainty 상태로 fail closed.
- 어느 항이든 non-finite: 즉시 `LCMF_ENONFINITE`.

현재 예상 최초 음의 해에서는

\[
\hat I=-2.2010974\times10^{-8},
\]

SC 세 항 절대값 합이 `2.25572e-8`입니다. 매우 느슨한 100-operation local bound도

\[
\gamma_{100}\sum|T_i|\approx2.50\times10^{-22}
\]

이므로 현재 음수는 단순 roundoff보다 약 \(8.8\times10^{13}\)배 큽니다. 최종 판정은 반드시 누적 interval을 구현해 확인해야 하지만, 진짜 이산 음수로 판정될 가능성이 압도적입니다.

### 보존 및 acceptance

다음 수치는 하나도 완화하지 않습니다.

- profile L1/L2 `≤1e-4`
- centroid absolute error `≤1e-4`
- invariant-area relative error `≤1e-4`
- `1.8≤p_obs≤2.2`
- transport residual `≤1e-4`
- clamp/floor 0
- non-finite 0
- 안정성 한계를 초과한 solution-negative 0

기존의 모호한 `negative_count==0`만 다음처럼 의미를 분리합니다.

- `bdf_eta_negative_count`: 기록 전용
- `solution_negative_excess_count`: acceptance/fail-closed
- `sign_uncertain_count`: fail-closed numerical uncertainty

## 4. 1차 positivity-preserving 대안 기각

1차 implicit upwind는

\[
\chi_{\rm eff}=\chi+3a+a/h,\qquad
\eta_{\rm eff}=\eta+(a/h)I_{k-1}\ge0
\]

이므로 비음수 입력에 대해 단조적입니다.

그러나 주파수 truncation error가 \(O(h)\), 전역 오차도 \(O(h)\)이므로 Richardson 차수는 점근적으로 \(p_{\rm obs}\to1\)입니다. 요구 창 `1.8–2.2`와 직접 충돌합니다.

`η_eff<0`인 곳만 1차로 전환하는 hybrid도 사전 승인할 수 없습니다. 전환 영역의 측도가 충분히 빠르게 사라진다는 증명 없이 전역 2차를 보장하지 못하며, 부호 기반의 해상도 의존적 스킴 변경이 됩니다. 따라서 1차 fallback은 명시적으로 기각합니다.

## 5. KA3 기대 거동 사전등록

### 새 fail-closed 가드가 적용된 실제 예상

`η_eff`만 진단으로 강등하고 solution-negative 가드를 유지하면 coarse run은 다음처럼 진행될 것으로 예상합니다.

| 항목 | 사전 예상 |
|---|---:|
| 최초 `η_eff<0` | `k=98`, segment 47 |
| solution fail 전 음의 η endpoint 평가 | 3회, unique node 2개 |
| 그 시점까지 `η_eff_min` | 약 `-1.475e-16` |
| 최초 안정성 초과 음의 해 | `k=98`, 다음 segment |
| 최초 음의 \(I\) | 약 `-2.2011e-8` |
| p/centroid/area | fail-closed 이전 종료로 미산출 |

따라서 공식 기대 판정은 **FAIL**, PASS가 아닙니다.

### fail을 무시한 forensic signed recurrence 예측

이는 acceptance 실행이 아니라 예상치 설정용 독립 계산입니다.

| grid | η 음수 주파수 plane | η 최소값 | 해 최소값 | profile L2 | centroid error | area error |
|---|---:|---:|---:|---:|---:|---:|
| 32×128 | 30 (`k=98–127`) | `-6.63e-14` | `-1.98e-5` | `2.417e-2` | `1.206e-4` | `1.820e-4` |
| 64×256 | 0* | 양수 | `+3.93e-17` | `7.541e-3` | `6.15e-5` | `9.25e-5` |
| 128×512 | 0* | 양수 | `+5.94e-15` | `2.992e-3` | `3.10e-5` | `4.66e-5` |

\* 안정적인 Gaussian cell-average 평가를 사용할 때.

사전등록 창은 다음으로 둡니다.

- \(p_{\rm obs}({\rm L2})\): `1.25–1.45`, 중심 예측 `1.3335`
- finest centroid error: `(2.5–3.8)e-5`
- finest invariant-area error: `(4.0–5.5)e-5`
- finest profile L2: `(2.5–3.5)e-3`

centroid와 area는 finest에서 기존 문턱을 통과할 것으로 보이지만, `p_obs`와 profile L2는 실패합니다. 이 예측은 PASS 선언이 아닙니다.

현재 `erf(high)-erf(low)` 구현은 미세 격자 꼬리에서 cancellation으로 정확한 양수를 0으로 만들며, 그대로면 64×256과 128×512에서도 각각 29/61개 주파수 plane에 가짜 음의 이력이 생길 것으로 예상됩니다.

## 6. §2.2 개정 문안

[원설계 §2.2](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md:86)의 마지막 문단을 다음 의미로 교체해야 합니다.

> BDF2의 음의 두-step 계수 때문에 유한한 `eta_eff<0`은 연속 전달 문제의 비음수 불변량이 아니다. 이를 clamp/floor하지 않으며, 발생 횟수·최소값·최초 좌표·두 history 항·step ratio·이론 positivity 한계를 기록한다. `eta_eff`의 부호만으로 solve를 종료하지 않는다.  
>  
> fail-closed 조건은 입력 물리량의 음수, 모든 non-finite 값, 그리고 전파된 IEEE-754 roundoff enclosure의 상한까지 음수인 계산해 `I`다. enclosure가 0을 포함하면 부호 미결정으로 별도 fail-closed한다. KA3의 실제 보존 판정은 기존 centroid 및 invariant-area 문턱으로 수행하며 profile·차수·residual을 포함한 모든 기존 acceptance는 유지한다. clamp, floor, tail 제외, 1차 fallback은 금지한다.

KA3 acceptance의 [generic negative count 조항](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md:481)은 `solution_negative_excess_count==0`, `sign_uncertain_count==0`으로 바꾸고 `bdf_eta_negative_count`는 판정에서 제외해야 합니다.

## 7. 구현 지점

- [lumina_cmf_field.c:696](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:696)

  `η_eff` 계산은 유지하되 708–717의 finite-negative 즉시 종료를 카운터·최소값·최초 좌표 기록으로 교체합니다. non-finite만 즉시 종료합니다.

- [lumina_cmf_field.c:451](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:451)

  물리 source용 기존 nonnegative SC와 BDF effective-source용 signed SC를 분리합니다. signed 경로는 interval/error bound를 반환하고 solution sign을 판정합니다. [residual 경로](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:465)도 signed effective source를 허용해야 합니다.

- [lumina_cmf_field.h:44](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.h:44)

  error record에 endpoint/substep, decay ratio, theoretical limit, solution interval 상·하한을 추가합니다. [LCMFResult](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.h:88)의 `negative_count`를 위 세 카운터로 분리합니다.

- [stage31_cmf_ka_driver.c:17](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/stage31_cmf_ka_driver.c:17), [runner:143](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_stage31_cmf_ka.py:143)

  Gaussian cell average는 같은 부호의 먼 꼬리에서 `erf-erf` 대신 안정적인 `erfc` 차를 사용합니다. 이는 floor가 아니라 동일 해석식의 cancellation-free 평가입니다.

- [run_stage31_cmf_ka.py:174](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_stage31_cmf_ka.py:174)

  JSON에 새 diagnostics를 모두 기록하고, `bdf_eta_negative_count`는 acceptance에서 제거합니다. profile, centroid, area, p, residual 문턱은 그대로 둡니다.

필수 회귀는 exact-history 양/음 두 사례, observed numeric-history 재현, signed SC의 해석해 대조, interval-negative/interval-straddling 분기, stable Gaussian tail의 80-digit oracle 대조를 포함해야 합니다.

저장소 파일은 수정하지 않았습니다.