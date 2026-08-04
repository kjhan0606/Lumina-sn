# Codex A′-rev3 — KA3 차수 손실 진단 및 스킴 개정 보고서

결론은 다음과 같습니다.

- `p_obs=1.3335`의 주원인은 **② BDF2와 ray-segment 선형 SC 결합에서 생기는 혼합 절단항**입니다.
- ① `k=1` bootstrap은 KA3 실측 오차에 사실상 기여하지 않습니다.
- ③ coarse 음수·꼬리 norm은 차수 손실을 설명하지 못합니다.
- 최소 차수 개정은 `k=1` trapezoidal 시작과 `k≥2` BDF 이력의 quadratic-exact SC 결합입니다.
- 이 개정은 \(p\simeq2\)를 회복하지만 finest L2는 약 `1.54e-3`으로 예측되어 기존 `1e-4` acceptance에는 여전히 실패합니다.
- 저장소 파일은 수정하지 않았습니다.

## 1. 후보별 forensic 귀속

현 독립 recurrence는 세 격자의 L1/L2·centroid·area를 11–12자리까지 재현했습니다. 기준 실측은 [구현 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_IMPL3.md:77)와 [JSON](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_results/ka3_rev2.json)입니다.

| 실험 | L2: 32×128 / 64×256 / 128×512 | middle→fine \(p\) | 판정 |
|---|---:|---:|---|
| 현 스킴 | `2.417273e-2 / 7.540836e-3 / 2.992171e-3` | `1.333532` | 기준 |
| `k=1`을 exact plane으로 교체 | 기준과 표시 자릿수 동일 | `1.333532` | ① 기각 |
| 공간 이산화를 사실상 제거 (`Ns=4096`) | `2.365931e-2 / 5.982659e-3 / 1.494132e-3` | `2.001480` | ② 확정 |
| 출력 중심 ±4σ만 L2 계산 | `2.417237e-2 / 7.540762e-3 / 2.992155e-3` | `1.333526` | ③ 기각 |
| quadratic-exact SC 개정 모델 | `2.431362e-2 / 6.147832e-3 / 1.535926e-3` | `2.000969` | 차수 회복 |

### ① bootstrap 오염

`k=1` 현재 결과를 exact KA3 plane으로 바꿨을 때 출력 profile의 최대 변화는

- coarse `1.87e-14`
- middle `1.85e-14`
- fine `1.86e-14`

이고 상대 L2 변화는 모두 약 `2.6e-14`입니다. 따라서 이번 fixture에서 첫 두 frequency cell이 8σ blue tail에 위치하기 때문에 1차 시작오차는 비활성입니다.

다만 현재 `k=1` 식은 실제로 backward Euler입니다([소스](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:1012)). 일반적인 2차 보장 증명에는 그대로 둘 수 없으므로 아래 개정안에서는 2차 시작으로 바꿉니다.

### ② BDF2–ray SC 혼합항

현 선형 SC는 각 plane에서 공간적으로 2차인 이력 표현 오차 \(O(h_s^2)\)를 만듭니다. 이 오차가 \(O(1/\Delta x)\)개의 BDF plane에 반복 주입되므로

\[
E_{\rm mix}=O\!\left(\frac{h_s^2}{\Delta x}\right).
\]

현재 동시 세분은 \(h_s\propto\Delta x\)이므로 \(E_{\rm mix}=O(\Delta x)\)입니다.

직접 계산한 “대각 스킴 − 공간연속 기준” 상대 L2는

\[
1.11540\times10^{-2},\
5.61814\times10^{-3},\
2.82360\times10^{-3},\
1.41359\times10^{-3},
\]

이고 연속 refinement 차수는 `0.9894, 0.9926, 0.9982`입니다. 반대로 공간오차를 끈 BDF2 계열은 `1.9835, 2.0015, 2.0047`로 수렴합니다. BDF2 계수 자체는 [rev2 일반식](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_DESIGN_REV2.md:131) 그대로 2차입니다.

### ③ coarse 음수·norm 오염

공식 `p_obs`는 coarse를 사용하지 않고 middle/fine L2 비만 사용합니다([runner](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_stage31_cmf_ka.py:270)). 두 격자의 출력 중심값은 모두 비음수입니다.

또한 coarse 음수 출력 cell의 제곱오차 기여는 전체 L2 분자의 `1.7135e-5`, 즉 `0.00171%`뿐입니다. ±4σ core만 사용해도 \(p=1.333526\)으로 전체 norm의 `1.333532`와 같습니다. 따라서 꼬리/enclosure 영역은 차수 손실의 원인이 아닙니다.

## 2. 최소 스킴 개정

### 2.1 `k=1`: trapezoidal 시작

\(t=-\ln\nu\), \(L(I)=\partial_s I+(\chi+3a)I-\eta\)라 하면

\[
a\frac{I_1-I_0}{\Delta x}
+\frac12\{L(I_1)+L(I_0)\}=0.
\]

`k=0` static solve가 \(\partial_sI_0+\chi_0I_0=\eta_0\)를 만족하므로 구현식은

\[
\chi_{\rm eff,1}=\chi_1+3a+\frac{2a}{\Delta x},
\]

\[
\eta_{\rm eff,1}
=\eta_1+\left(\frac{2a}{\Delta x}-3a\right)I_0.
\]

KA3 최대 \(\Delta x=0.74/127=0.00582677<2/3\)이므로 이 시작 source는 비음수 \(I_0\)에 대해 비음수입니다. Trapezoidal rule은 A-stable이며 이후 BDF2 계수는 바뀌지 않습니다.

### 2.2 `k≥2`: quadratic history formal integral

각 ray segment에서

\[
H(s)=\eta+\frac{a}{\Delta x}
\left(2I_{k-1}(s)-\frac12I_{k-2}(s)\right)
\]

를 두 endpoint와 branch-local 인접 node로 만든 quadratic \(H(u)=c_0+c_1u+c_2u^2\)로 재구성합니다. core boundary reset을 가로지르는 stencil은 금지합니다.

\(\tau=\bar\chi_{\rm eff}\Delta s\)일 때 SC update는

\[
I_d=e^{-\tau}I_u+\Delta s(c_0J_0+c_1J_1+c_2J_2),
\]

\[
J_n=\int_0^1e^{-\tau(1-u)}u^n\,du
\]

를 사용합니다. \(J_0,J_1,J_2\)는 작은 \(\tau\) series와 일반 폐쇄식을 분리해 cancellation 없이 계산합니다.

Quadratic interpolation의 segment local defect는 \(O(h_s^4)\), plane 전체는 \(O(h_s^3)\)이므로

\[
E=O(\Delta x^2)+O\!\left(\frac{h_s^3}{\Delta x}\right).
\]

따라서 \(h_s\propto\Delta x\)인 KA3 세분에서 \(E=O(\Delta x^2)\)가 보장됩니다.

### 안정성·양수성 한계

- BDF2의 A-stability는 계수를 건드리지 않으므로 유지됩니다.
- trapezoidal 시작도 A-stable이며 KA3에서는 위 source 계수가 양수입니다.
- quadratic reconstruction은 무조건 단조적이지 않습니다. forensic 중심해에서도 coarse 최소 출력은 현 `-1.98e-5`에서 약 `-2.48e-5`로 바뀝니다.
- 원 BDF2도 \(I_{k-2}/I_{k-1}>4\)이면 양수성을 보장하지 않습니다([일반 한계](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_DESIGN_REV2.md:162)). 따라서 “무조건 2차이면서 무조건 양수”는 이 선형 BDF 계열에서 주장할 수 없습니다.

즉, 개정은 **2차를 보장하지만 KA3 PASS나 무조건 양수성을 보장하지 않습니다**. 기존 solution-negative 및 sign-uncertain fail-closed는 그대로 유지해야 합니다.

## 3. `sign_uncertain_count` 급증 판정

KA3 p=0 경로의 실제 update 수 대비 비율은 다음과 같습니다.

| grid | sign-uncertain | 전체 비영 frequency update | 비율 |
|---|---:|---:|---:|
| 32×128 | 591 | 8,382 | 7.05% |
| 64×256 | 3,462 | 33,150 | 10.44% |
| 128×512 | 32,933 | 131,838 | 24.98% |

카운트 배율은 `5.858`, `9.513`으로 전체 update 증가율 약 4보다 빠릅니다.

원인은 [이력 interval](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:792), [BDF interval](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:822), [SC interval](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:344)이 매 단계 이전 반경을 독립 worst case로 더하면서 BDF의 부호 상쇄와 단계 간 상관을 버리기 때문입니다.

- fine 최초 uncertainty는 \(x=0.22297\), 즉 blue tail이며 반경/중심값=`1.0123`입니다.
- 그러나 middle 최초 uncertainty는 \(x=-0.12980\), line core 부근이고 중심 `0.5630`에 반경 `0.6248`, 반경/중심값=`1.1097`입니다.

따라서 enclosure는 꼬리에서 먼저 과보수적으로 작동하지만, 문제는 꼬리에만 국한되지 않습니다.

판정은 두 층으로 나뉩니다.

1. `lower≤0≤upper`를 `LCMF_ESIGNUNCERTAIN`로 분류하고 최종 실패시키는 로직 자체는 rev2 계약과 정확히 일치합니다([계약](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_DESIGN_REV2.md:253), [구현](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:882)). **fail-closed는 올바르게 작동했습니다.**
2. 그러나 현재 enclosure가 tight하거나 certified됐다는 결론은 낼 수 없습니다. `gamma_m`이 `DBL_EPSILON=2u`를 사용해 rev2의 \(u=2^{-53}\)보다 약 2배 보수적이고, 이력 상관을 보존하지 않으며, `exp/expm1` 오차의 엄밀한 라이브러리 bound도 별도 증명되지 않았습니다. 이 부분은 **UNRESOLVED**입니다.

카운터를 acceptance에서 빼거나 tolerance로 숨겨서는 안 됩니다. 개선하려면 독립 radius 합산 대신 공유 noise symbol을 유지하는 affine enclosure 또는 BDF 안정성 연산자를 이용한 a-posteriori 전역 roundoff bound가 필요합니다.

## 4. 개정 후 사전등록

다음은 위 trapezoidal-start + branch-local quadratic-exact SC를 정확히 구현한다는 조건의 forensic 중심값과 창입니다.

| grid | 이전 격자 대비 \(p\) 중심값·창 | profile L2 중심값·창 | centroid error 중심값·창 | area error 중심값·창 |
|---|---:|---:|---:|---:|
| 32×128 | — | `2.43136e-2` `[2.38,2.49]e-2` | `8.112e-7` `[6.5e-7,1.0e-6]` | `8.152e-7` `[6.5e-7,1.0e-6]` |
| 64×256 | `1.98362` `[1.94,2.04]` | `6.14783e-3` `[6.00,6.30]e-3` | `2.086e-7` `[1.7,2.6]e-7` | `2.091e-7` `[1.7,2.6]e-7` |
| 128×512 | `2.00097` `[1.96,2.04]` | `1.53593e-3` `[1.50,1.58]e-3` | `5.286e-8` `[4.2,6.5]e-8` | `5.293e-8` `[4.2,6.5]e-8` |

공식 middle/fine `p_obs` 중심 예측은 `2.00097`입니다.

**예측 ≠ PASS.** Finest L2 중심값 `1.53593e-3`은 기존 `1e-4` 문턱의 `15.36×`이고, coarse 음수와 개정 enclosure의 sign-uncertain 카운트도 미해결입니다. acceptance는 [현 기준](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_DESIGN_REV2.md:274) 그대로 유지합니다.

## 5. 구현 지시

- [lumina_cmf_field.c:344](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:344): `lumina_cmf_sc_quadratic_signed`를 추가하고 \(J_0,J_1,J_2\)의 일반식·small-\(\tau\) series·interval 전파를 구현.
- [lumina_cmf_field.c:792](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:792): 선형 `interpolate_history` 대신 비균일 z 좌표의 branch-local 3-point quadratic reconstruction 추가. core reset을 stencil 경계로 취급.
- [lumina_cmf_field.c:1012](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:1012): `k==1`을 위 trapezoidal `chi_eff/eta_eff`로 교체.
- [lumina_cmf_field.c:1026](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:1026): BDF2 계수는 유지하고 effective source polynomial을 quadratic SC에 전달.
- [lumina_cmf_field.c:637](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:637): residual probe도 동일 quadratic source를 사용하도록 동기화.
- [lumina_cmf_field.c:882](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:882) 및 [최종 상태 반환](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:1238): negative/sign-uncertain 분류와 acceptance는 변경 금지.
- [self-test](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/tests/stage31_cmf_skeleton_selftest.c:100): constant/linear/quadratic source 해석값, \(\tau\to0\), branch-boundary stencil, trapezoidal 시작, interval-negative/straddling 회귀 추가.
- [KA runner](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_stage31_cmf_ka.py:198): 위 사전등록 창을 별도 rev3 예측 필드로 기록하되 기존 acceptance 식은 한 글자도 변경하지 않음.

최종 상태: **차수 귀속 RESOLVED, 2차 개정 설계 RESOLVED(KA3 smooth-grid 범위), 무조건 양수성·tight certified enclosure·최종 KA3 PASS는 UNRESOLVED.**