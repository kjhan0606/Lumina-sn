# Codex A′-rev4 — enclosure 정밀화 및 L1 귀속

결론부터 말하면:

- `sign-uncertain` 폭증과 28회 non-finite의 주원인은 단순 선형 roundoff 누적이 아니라, **BDF2의 음의 이력 계수와 quadratic stencil의 음의 가중치를 절대값 반경으로 전파하면서 생긴 지수적 interval instability**다.
- `γ_m`의 과대 설정은 초기 반경을 약 2배 키우지만 폭증의 근본 원인은 아니다.
- 가장 작은 감사 가능한 개정은 중심 계산을 바꾸지 않고 **고정 고정밀도 MPFR directed-rounding certificate replay**를 추가하는 것이다.
- finest L1 초과는 enclosure 꼬리가 아니라 중심 profile 오차다. Enclosure 개정만으로 L1은 통과하지 않으며 `(1024,4096)` 격자 한 단계가 필요하다.

## 1. Enclosure 산술 진단

현재 저장소의 [signed linear SC 경로](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:343)는 rev2 구현이고, rev3 quadratic 구현은 [rung6 patch](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/s31_rung6.patch:88)에만 있다. 아래 폭증 진단은 실제 rev3 작업본 산식을 기준으로 했다.

Finest `512×2048`에서

\[
\Delta x=3.61504640938\times10^{-4},\quad
a=10^{-11},\quad c=a/\Delta x=2.76621621622\times10^{-8},
\]

\[
\Delta s=1.953125\times10^7,\quad
\tau=(3a+1.5c)\Delta s=0.8110008446,\quad
E=e^{-\tau}=0.4444130552.
\]

균일 내부 segment의 quadratic formal-integral history 가중치에 \(c\)를 곱하면

\[
c\,w=(0.129806869,\ 0.270827566,\ -0.0305107411).
\]

따라서

\[
\sum cw_i=0.370123694,\qquad
\sum |cw_i|=0.431145176,
\]

즉 stencil 절대값 전파만으로도 반경이 `1.1648678×` 과대화된다.

여기에 BDF2의 `2 I_{k-1} - 0.5 I_{k-2}`를 현재 구현처럼 독립 반경으로 바꾸면

\[
2B_{k-1}+0.5B_{k-2}
\]

가 된다. 공간 sweep의 감쇠까지 포함한 근사 plane recurrence는 다음과 대비된다.

\[
\text{signed error:}\quad
e_k\simeq1.33237e_{k-1}-0.333093e_{k-2},
\]

\[
\text{현재 radius:}\quad
B_k\lesssim1.55203B_{k-1}+0.388009B_{k-2}+\delta_k.
\]

Signed recurrence의 특성근은 약 `0.99892, 0.33345`로 안정적이지만, 반경 recurrence의 지배근은

\[
\rho_B=1.77111>1
\]

이다. 따라서 현재 반경은 “절대오차를 단계 수에 비례해 선형 누적”하는 것이 아니라, **절대값화로 상관을 잃은 양의 선형 recurrence가 주파수 plane 수에 대해 지수 성장**한다.

실측도 이에 맞는다.

| grid | uncertain / 전체 update | 비율 | 최초 기록 좌표 |
|---|---:|---:|---|
| 128×512 | 51,036 / 131,838 | 38.71% | `k=50, x=0.247593, segment=257`, `I=1.02267e-14`, interval `[-9.36e-16,2.14e-14]` |
| 256×1024 | 242,152 / 525,822 | 46.05% | `k=49, x=0.284555, segment=513`, `I=1.13964e-14`, interval `[-2.16e-15,2.49e-14]` |
| 512×2048 | 1,019,773 / 2,100,222 | 48.56% | 최초 non-finite `k=2041, x=-0.417831, segment=1026`, `I=1.39581e-14` |

28회의 non-finite는 [작업본의 `radius==inf → [-inf,+inf]` 경로](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/s31_rung6.patch:128)에서 시작한다. 최초가 마지막 segment인 `(k,segment)=(2041,1026)`이고 남은 plane이 7개이므로, 뒤쪽으로 한 segment씩 퍼진 `1+2+…+7=28` 패턴과 정확히 일치한다.

`γ_m`도 과대하다. 구현은 `DBL_EPSILON=2u`를 사용하여

- `γ32 = 7.1054e-15` — 올바른 \(u=2^{-53}\) 기준의 2배
- `γ96 = 2.1316e-14` — 올바른 기준의 2배

를 주입한다. 다만 이것은 지수 성장의 seed를 약 2배 키우는 요인일 뿐이다. 더 중요한 문제는 [rev3에서 이미 지적했듯](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_DESIGN_REV3.md:146) `exp/expm1`의 엄밀한 libm 오차 상계와 실제 연산 수 증명이 없다는 점이다. 따라서 현 enclosure는 넓을 뿐 아니라 완전한 의미의 certified bound도 아니다.

## 2. Tight certified bound 확정

후보 판정은 다음과 같다.

| 후보 | 판정 |
|---|---|
| 상대오차만 전파 | 취소가 있는 BDF2에서 `I≈0`이면 정의·보증이 깨짐. 기각 |
| FMA/Kahan/보상합산 | local roundoff는 줄지만 history 상관 손실과 libm bound를 해결하지 못함. 단독 기각 |
| segment마다 반경 reset | incoming/history 오차를 버리면 상계가 아니며, 보존하면 현재 지수 성장이 재현됨. 기각 |
| full affine arithmetic | 선형 상관을 보존해 certified 가능하나 noise symbol 수와 축약 증명이 큼 |
| BDF 안정성 연산자/Krawczyk | 장기적으로 가장 효율적이나 verified inverse 및 전역 안정성 상수 구현 부담이 큼 |
| 고정밀 directed interval replay | 기존 중심 스킴을 그대로 두면서 가장 작은 감사 가능한 개정. 확정 |

확정안은 다음이다.

- binary64 중심 계산은 그대로 유지한다.
- 동일한 discrete recurrence를 MPFR interval arithmetic으로 별도 replay한다.
- 기존 공식 triple에는 고정 `2048 bit`, 신규 `1024×4096`에는 고정 `4096 bit`를 사용한다.
- `+,-,*,/`, Lagrange weight, BDF combination, `exp/expm1`을 각각 `MPFR_RNDD/RNDU`로 평가한다.
- small-\(\tau\) series에는 다음 생략항에 의한 엄밀 remainder를 추가한다.
- 입력 binary64 값은 `mpfr_set_d`로 정확히 point interval화한다.
- 어느 cell이라도 `L≤0≤U`이면 기존과 동일하게 `LCMF_ESIGNUNCERTAIN`; precision을 사후 증가시켜 통과시키지 않는다.

보증은 interval extension의 포함성으로 귀납된다. 입력이 point interval에 포함되고, 각 primitive가 outward-rounded interval이면 `(k,node)` 위상 순서의 모든 정확한 real discrete state가 계산 interval에 포함된다. 따라서 `L>0`은 진짜 양수 증명이며 tolerance나 acceptance 완화가 아니다.

현 폭증률로 환산하면 필요한 정밀도는 대략 plane당 `log2(1.77111)=0.82465 bit`씩 증가한다. 따라서 2048-plane에는 약 1750 bit, 4096-plane에는 약 3445 bit가 필요해 사전등록한 `2048/4096 bit`가 안전 여유를 갖는다.

기존 scalar enclosure 카운터는 `legacy_sign_uncertain_count` 진단으로 남길 수 있지만 acceptance는 새 `certified_sign_uncertain_count==0`을 사용해야 한다. 이는 카운터 제거가 아니라 **더 정밀하고 실제로 증명된 상계로 같은 계약을 평가**하는 것이다. rev2의 판정 의미는 그대로 유지된다([계약](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_DESIGN_REV2.md:253)).

## 3. L1 초과 귀속

Finest L1은

\[
1.0565893300\times10^{-4},
\]

문턱 초과분은

\[
5.6589330\times10^{-6}
\]

이다.

출력 중심 `x=-0.1`, `σ=0.04`를 기준으로 직접 분해하면:

| 영역 | 전체 정규화 L1 기여 | 총 L1 중 비율 |
|---|---:|---:|
| `|x+0.1|≤4σ` 중심 | `1.0537701600e-4` | 99.7332% |
| `|x+0.1|>4σ` 꼬리 | `2.81916998e-7` | 0.2668% |
| `|x+0.1|≤3σ` 중심 | `1.0069969098e-4` | 95.3064% |
| `|x+0.1|>3σ` | `4.95924202e-6` | 4.6936% |

따라서 ±4σ 밖의 모든 꼬리 오차를 0으로 만들어도 중심만으로 문턱을 `5.3770160e-6` 초과한다. ±3σ 중심만으로도 이미 `6.99691e-7` 초과다.

즉 L1은 enclosure와 독립적인 **중심 discretization error**다. Enclosure 개정은 sign/non-finite를 해결할 수 있지만 L1 값을 바꾸지 않는다.

관측 L1 차수는

\[
p_{256\to512}=2.00116967.
\]

한 단계 추가 시

\[
L1_{1024,4096}
 \simeq \frac{1.05658933\times10^{-4}}{2^{2.00116967}}
 =2.63933\times10^{-5}.
\]

정확히 \(p=2\)를 쓰면 `2.64147e-5`다. 따라서 `(1024,4096)` 한 단계면 충분하며, enclosure 개정과 별개로 필요하다.

## 4. 사전등록 및 구현 지시

사전등록:

| 실행 | certified sign-uncertain | non-finite | profile L1 |
|---|---:|---:|---:|
| 기존 128×512 | `0` 기대, 아니면 FAIL | `0` | `1.69387067e-3` 중심 불변 |
| 기존 256×1024 | `0` 기대, 아니면 FAIL | `0` | `4.22978524e-4` 중심 불변 |
| 기존 512×2048 | `0` 기대, 아니면 FAIL | `0` | `1.05658933e-4`, 여전히 FAIL |
| 신규 1024×4096 | `0` 기대, 아니면 FAIL | `0` | 중심 `2.64e-5`, 창 `[2.50,2.80]e-5`, PASS 기대 |

구현 순서:

1. `s31_rung6.patch`의 rev3 중심 스킴을 기준으로 작업한다.
2. 기존 double `gamma/radius` 경로를 acceptance용으로 미세조정하지 말고 MPFR certificate replay를 별도 추가한다.
3. 비대칭 `[lower,upper]` 자체로 판정하며 임의 tolerance와 대칭 재팽창을 금지한다.
4. `certificate_bits`, `certified_min_lower`, `certified_max_width`, 최초 unresolved 좌표를 JSON에 기록한다.
5. runner에 `(1024,4096)`을 추가하고 공식 triple을 `(256,1024)/(512,2048)/(1024,4096)`으로 이동한다.
6. 기존 L1/L2·차수·residual·negative·non-finite acceptance는 한 글자도 완화하지 않는다.

최종 판정은 **enclosure 원인 RESOLVED, certified 개정 설계 RESOLVED, L1은 중심 귀속으로 RESOLVED**다. 예상 최종 KA3 PASS에는 enclosure replay와 격자 한 단계 추가가 둘 다 필요하다. 파일은 수정하지 않았다.