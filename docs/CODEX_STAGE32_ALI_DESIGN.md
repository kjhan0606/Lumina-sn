# Stage 3.2 VEF/ALI 설계 명세 — UV 수리 정본

상태: **DESIGN ONLY / IMPLEMENTATION FORBIDDEN**  
결정: **Sobolev line-space diagonal MALI를 정본으로 채택한다.** coarse-bin Olson 대각 ALI는 연속체·전자산란과 KA용 기반이며, 공간 삼대각 Λ*는 성능 후속 rung이다. VEF는 독립 moment oracle/선택적 공간 preconditioner로 두며, `SRC_NLTE` 활성화의 필수 안정화 장치는 아니다.

## 1. 문제 정식화

### 1.1 현재 반복과 스펙트럼 반경

순수 2준위 선에서

\[
S=(1-\epsilon)J+\epsilon B,\qquad J=\Lambda[S]
\]

이고 현재와 같은 명시적 Λ-반복은

\[
J^{(m+1)}
=\Lambda\{\epsilon B+(1-\epsilon)J^{(m)}\}.
\]

오차 \(e^{(m)}=J^{(m)}-J^\star\)는

\[
e^{(m+1)}=(1-\epsilon)\Lambda e^{(m)}
\]

이므로

\[
\boxed{\rho_{\rm LI}=(1-\epsilon)\lambda_{\max}(\Lambda)}.
\]

실제 coarse cell에서는 \(1-\epsilon\) 대신

\[
a_{sb}={\chi_{\rm coherent}\over\chi_{\rm total}}
\]

가 들어가며 일반식은 \(\rho(\Lambda A)\)이다. Stage 3.1 정본은 실제로 `eta_fixed + chi_coherent*J_previous`를 반복하고 있어 정확히 이 형태다([lumina_cmf_field.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:1933)). 구 assembler 역시 선 산란 잔여를 `chi_es`에 넣고 `S_fixed+rJ`로 소비한다([lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1527), [E8](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8.md:130)).

광학적으로 두꺼운 균질층의 확산 극한에서

\[
{d^2J\over d\tau^2}=3(J-S)
\]

이고 공간 mode \(e^{ik\tau}\)에 대해

\[
\lambda(k)={1\over1+k^2/3}.
\]

총 광학두께를 \(T\), 가장 느린 mode를 \(k_1\simeq\pi/T_{\rm eff}\)라 하면

\[
\rho_{\rm LI}
\simeq {1-\epsilon\over1+\pi^2/(3T_{\rm eff}^2)}
\simeq1-\left[\epsilon+{\pi^2\over3T_{\rm eff}^2}\right].
\]

따라서

\[
N_{\rm efold}\simeq
{1\over\epsilon+\pi^2/(3T_{\rm eff}^2)}.
\]

경계조건에 따라 \(T_{\rm eff}\)의 \(O(1)\) 상수는 달라지므로 이는 두꺼운 층의 점근식이며, 실제 이산 스펙트럼 반경은 KA에서 조립한 Λ 행렬로 측정해야 한다. 고전 ALI가 이 느린 mode를 approximate operator로 제거한다는 근거는 Olson–Auer–Buchler의 operator-splitting 식에 있다([JQSRT 35, 431](https://doi.org/10.1016/0022-4073(86)90030-0)); SC 기반 대각·삼대각 연산자는 Olson–Kunasz가 유도했다([JQSRT 38, 325](https://doi.org/10.1016/0022-4073(87)90027-6)).

\(T\gg\epsilon^{-1/2}\)이면 \(\rho\simeq1-\epsilon\)이다. 따라서:

- \(\epsilon=10^{-4}\): e-fold 약 \(10^4\)회, 오차 \(10^{-4}\) 감소에 약 \(9.21\times10^4\)회.
- E9의 \(\epsilon=0.00243682\): e-fold 410회, \(10^{-4}\) 감소 약 3,780회. 저장소의 기존 산술과 일치한다([E9](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E9.md:307)).

### 1.2 E8의 5247배와의 대응

단일 eigenmode에서

\[
J={\lambda S_{\rm fixed}\over1-(1-\epsilon)\lambda},
\]

\[
{S_{\rm total}\over S_{\rm fixed}}
={1\over1-(1-\epsilon)\lambda}
\simeq {1\over\epsilon+1-\lambda}.
\]

E8 s8 BALL의

\[
{S_{\rm total}\over S_{\rm fixed}}=5247.4904
\]

는 유효 분모

\[
1-(1-\epsilon)\lambda
=1.90567\times10^{-4}
\]

에 해당한다. 이는 E8이 직접 측정한 `eps_eff(source)`와 같고, 필요한 장 이득 5247.4106과 0.00152% 이내로 닫힌다([E8](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8.md:8), [대역표](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8.md:147)).

단, 이것은 “선형 Λ-반복이 수학적으로 발산했다”는 뜻이 아니다. \(\rho<1\)이면 선형 반복은 매우 느리지만 수렴한다. 실제 폭발은 다음 population feedback이 붙을 때 발생한다.

### 1.3 \(S_l/J\) 폭발의 비선형 기전

\[
S_l={C_\nu\over q},\qquad
C_\nu={2h\nu^3\over c^2},\qquad
q={g_u n_l\over g_l n_u}-1.
\]

\(r=n_u/n_l\), \(G=g_u/g_l\)라 하면

\[
{\partial S_l\over\partial r}
={C_\nu G\over r^2q^2}.
\]

따라서 population ratio가 inversion 경계 \(r\to G^{-}\)에 접근하면 \(S_l\sim q^{-1}\), 민감도는 \(q^{-2}\)로 발산한다. 명시적 외부 반복

\[
J^{m}\rightarrow n^{m+1}(J^m)
\rightarrow S_l^{m+1}(n^{m+1})
\rightarrow J^{m+1}
\]

의 선형화는 개략적으로

\[
M_{\rm coupled}\sim
\underbrace{[I-(1-\epsilon)\Lambda]^{-1}}_{\text{E8: } \sim5247}
\Lambda\,
{\partial S_l\over\partial n}
{\partial n\over\partial J}.
\]

따라서 coarse 재순환 resolvent와 \(q^{-2}\) population 민감도가 곱해져 \(\rho(M_{\rm coupled})>1\)이 될 수 있다. 실제 주석도 첫 NLTE-fed iteration에서 \(J=1.8\times10^{-18}\to4.2\times10^{-2}\) 및 \(S_l/J\sim10^{16}\)을 기록한다([lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:756)). 별도 emergent 경로에서는 \(S_l\sim10^{52}\)까지 보고되어 있다([lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:6684)).

현재 writer는 inversion에서 `stim_corr`를 0으로 자르고, \(q\le10^{-30}\)이면 \(S_l=0\)으로 남겨 Planck fallback을 유발한다([lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:17048)). 이 branch는 Stage 3.2의 정본 안정성으로 인정하지 않는다.

## 2. ALI/MALI 설계

### 2.1 연산자 선택

선택 순서는 다음과 같다.

1. **연속체·전자산란:** Stage 3.1 SC에서 직접 얻는 Olson–Auer–Buchler 대각 Λ*.
2. **Sobolev 선:** line-space 국소 연산자
   \[
   \boxed{\Lambda^\ast_{ls,l's'}
   =\delta_{ll'}\delta_{ss'}(1-\beta_{ls})}.
   \]
3. **다준위 population:** 위 line-space Λ*를 Rybicki–Hummer MALI의 approximate rate operator로 소비.
4. **공간 수렴이 부족할 때만:** 같은 SC에서 얻은 shell 삼대각 블록.
5. **VEF:** \(f_\nu=K_\nu/J_\nu\)를 formal sweep에서 갱신하는 독립 moment solve/oracle. ALI와 해가 일치해야 하며 기본 production preconditioner로는 사용하지 않는다.

Rybicki–Hummer I은 비중첩 다준위 선+연속체의 ALI를, II는 중첩 전이와 full continuum를 다룬다([1991 A&A 245, 171](https://www.nist.gov/publications/accelerated-lambda-iteration-method-multilevel-radiative-transfer-i-non-overlapping), [1992 A&A 262, 209](https://jila-pfc.colorado.edu/bibcite/reference/4848)). VEF의 formal/moment 반복은 Auer–Mihalas가 제시했다([MNRAS 149, 65](https://doi.org/10.1093/mnras/149.1.65)). Stage 4와의 hybrid ALI/전역 선형화 방향은 Hubeny–Lanz의 CL/ALI 구조와 같다([ApJ 439, 875](https://doi.org/10.1086/175226)).

구 `lumina_cmfgen.c`의 대각/삼대각 ALI는 참고 구현일 뿐 정본으로 승격하지 않는다. 분모 floor와 음수 \(J\to0\) clamp가 포함돼 있기 때문이다([lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1546), [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1593)). 새 연산자는 Stage 3.1의 fail-closed 산술 규율을 따라야 한다.

### 2.2 Sobolev Λ*와 escape probability

국소 Sobolev 관계는

\[
\bar J_l=(1-\beta_l)S_l+\beta_lJ_{{\rm ext},l},
\qquad
\beta_l={1-e^{-\tau_l}\over\tau_l}.
\]

따라서 line source의 자기 응답은 정확히

\[
{\partial\bar J_l\over\partial S_l}=1-\beta_l.
\]

미시적 열화확률을

\[
\epsilon_0={C\over A+C}
\]

로 정의해

\[
S_l=(1-\epsilon_0)\bar J_l+\epsilon_0B
\]

에 대입하면

\[
D_l=1-(1-\epsilon_0)(1-\beta_l)
=\epsilon_0+\beta_l-\epsilon_0\beta_l,
\]

\[
\boxed{
S_l=
{(1-\epsilon_0)\beta_lJ_{{\rm ext},l}
+\epsilon_0B\over
\epsilon_0+\beta_l-\epsilon_0\beta_l}
}.
\]

즉 \(\tau\gg1\)에서 \(D\simeq\epsilon_0+1/\tau\)이며 trapped self-coupling을 한 번에 역산한다.

현재 `radeq_line_eps_phys`의

\[
\epsilon_{\rm eff}={C\over C+A\beta}
={\epsilon_0\over\epsilon_0+\beta-\epsilon_0\beta}
\]

는 이미 Sobolev trapping을 제거한 “외부장 대비 유효 열화확률”이다. 이를 다시 `(1-eps_eff) chi_line*J_same-bin`에 넣으면 trapping을 coarse transport에서 또 반복한다. Stage 3.2에서는:

- MALI rate equation에는 \(A,B,C,\beta\)의 원시 rate owner를 사용한다.
- `C/(C+Aβ)`는 진단 또는 국소 제거 후의 결과로만 사용한다.
- 이를 line-space Λ*의 \(\epsilon_0\) 자리에 재사용하지 않는다.

### 2.3 ALI 갱신과 안정성

\[
\Lambda=\Lambda^\ast+(\Lambda-\Lambda^\ast)
\]

로 나누면

\[
S^{m+1}
=(1-\epsilon)\left[
\Lambda^\ast S^{m+1}
+(\Lambda-\Lambda^\ast)S^m
\right]+\epsilon B
\]

이고 오차 증폭 행렬은

\[
\boxed{
M_{\rm ALI}=
[I-(1-\epsilon)\Lambda^\ast]^{-1}
(1-\epsilon)(\Lambda-\Lambda^\ast)
}.
\]

고립된 Sobolev zone에서는 \(\Lambda=\Lambda^\ast\)이므로 \(M_{\rm ALI}=0\): 국소 trapping mode는 한 단계에 제거된다. 중첩선·비국소 continuum 때문에 \(\Lambda-\Lambda^\ast\ne0\)이면 수렴률은 KA에서 직접 측정한다. “ALI이므로 무조건 수렴”은 주장하지 않는다.

다준위 문제에서는 \(S_l\) 각각을 독립적인 2준위 식으로 풀지 않는다. 동일한 Λ*를 statistical-equilibrium rate matrix 안에 선조건화하는 MALI를 사용한다. 그렇지 않으면 형광 branch를 다시 2준위 근사로 축약하게 된다.

## 3. 1000-bin 격자와 line-center 투영

### 3.1 coarse-only ALI가 가능한 범위

1000-bin 장에서 line-to-bin 투영을 \(P\), bin-to-line sampling을 \(Q\)라 하면 line-space 전달 연산자는

\[
\Lambda_{\rm line}=Q\,\Lambda_{\rm grid}\,P.
\]

한 bin에 여러 선이 있으면 \(P\)가 그 선들의 서로 다른 \(S_l\)를 opacity-weighted 평균 하나로 축약한다. 따라서 coarse-only 연산자는 같은 bin의 서로 다른 선에 독립적인 \(1-\beta_l\) 대각을 줄 수 없다. 이 rank 손실은 ALI의 선택 문제가 아니라 입력 표현의 한계다.

결론:

- coarse 50×1000 대각 ALI는 electron scattering 및 “bin 하나당 source 하나”인 KA에는 충분하다.
- 다준위 `SRC_NLTE`를 켜는 production 계약에는 불충분하다.
- 최소한 FUV 지배선에 대해 line-center \(J_l\), \(\tau_l\), \(S_l\), population epoch를 보존하는 보조 line-space가 필요하다.

저장소에도 binned \(J\)와 line-resolved \(J_l\)의 대조 생산자가 이미 있으며, 특히 1000–1300 Å 병리가 외곽 셸에 국소화됐다고 기록돼 있다([lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:3378)). 이는 line-center 보조 격자의 필요성을 직접 지지하지만, 기존 fine producer 자체를 정본으로 승인하는 것은 아니다.

### 3.2 권고 투영

각 활성 line-shell에 대해 다음을 보존한다.

\[
w_{ls}={(1-e^{-\tau_{ls}})\nu_l\over
ct_{\rm exp}\Delta\nu_{b(l)}}.
\]

- `deposit`: \(\eta_b=\sum_{l\in b}w_{ls}S_{ls}\)
- `sample`: \(J_{{\rm ext},ls}=Q_l[J_{\rm continuum+other\,lines}]\)
- `local response`: \(\Lambda^\ast_{ls}=1-\beta_{ls}\)
- `epoch`: population/rate/opacity/source/bin-edge SHA를 함께 고정

Sobolev 근사에서는 thermal Doppler profile 전체를 균일하게 해상할 필요는 없다. line center와 주변 continuum interpolation은 필요하다. 그러나 실제 overlap block을 어떤 \(\Delta\nu\) 기준으로 묶을지는 thermal/microturbulent 폭과 velocity-grid 정보가 현재 명세에 완전히 존재하지 않으므로 **UNRESOLVED**다.

권고 active-set은 600–3000 Å에서 \(w_l\), 흡수 power, rate-Jacobian 기여 중 하나가 비영인 선을 정렬해 누적 기여 99.99%까지 포함한다. 이 선택으로 UV 대역 결과가 전선 계산과 1% 이내인지 convergence study로 인증한다. 필요한 선 수와 실제 런타임은 census 전까지 **UNRESOLVED**다.

규모 하한은 이미 크다. 기존 기록상 active transition은 1,681,176개, active line-shell cell은 19,246,925개다([E5](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E5_VERDICT.md:75)). active cell당 double 하나는 약 154 MB이며, 모든 transition×50 shell의 dense double은 약 672 MB다. 여러 line-space field를 dense로 동시에 보존하지 말고 active CSR/SoA를 사용해야 한다.

## 4. `SRC_NLTE`의 안전한 소비

### 4.1 전처리

정본 owner는 \(S_l\) scalar가 아니라 짝을 이룬 population-native opacity/emissivity다.

\[
\chi_l={h\nu\over4\pi}
(n_lB_{lu}-n_uB_{ul})\phi_l,
\qquad
\eta_l={h\nu\over4\pi}n_uA_{ul}\phi_l,
\qquad S_l={\eta_l\over\chi_l}.
\]

이 형태는 이미 E9의 정본 설계에 명시돼 있다([E9](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E9.md:123)).

절차:

1. population, rate, \(\tau\), line mapping의 epoch와 SHA가 모두 같지 않으면 fail.
2. \(n_l,n_u,g_l,g_u,A,B,C\)의 finite/nonnegative 및 detailed-balance 쌍을 검사.
3. \(q\)는 log-ratio와 `expm1` 형태로 계산해 near-cancellation을 줄인다.
4. transport assembly는 \(S_l\)를 먼저 만들어 \(wS_l\)을 곱하지 않고 \(\chi_l,\eta_l\)를 직접 투영한다.
5. \(S_l\)는 진단 및 MALI 응답 계산에만 사용한다.
6. \(q\le0\)인 inversion/maser는 clamp·Planck fallback 없이 별도 상태로 fail-closed한다. maser transfer는 Stage 3.2 범위 밖이며 **UNRESOLVED**다.
7. \(q\to0^+\)여도 \(\eta_l\)가 finite이면 scalar \(S_l\)의 크기만으로 거부하지 않는다. 반드시 \((\chi_l,\eta_l,\tau_l)\)의 짝과 에너지 잔차로 판정한다.

### 4.2 초기화와 반복

초기화는 세 단계로 한다.

1. 기존 수렴 snapshot의 \(J,n,T\)를 읽되 source는 native \(\chi_l,\eta_l\)로 재조립한다.
2. population을 동결한 formal solve를 한 번 수행한다. 이 단계는 source가 고정돼 있으므로 ALI 반복이 아니라 native-source 배선 검증이다.
3. 그 결과를 seed로 MALI population solve를 시작한다.

MALI correction \(\Delta x\)에는 처음 \(\omega=1\)을 사용하고, 결합 잔차가 증가할 때만 backtracking으로

\[
x^{m+1}=x^m+\omega\Delta x,\qquad
\omega=1,{1\over2},{1\over4},\ldots
\]

를 적용한다. 인정 조건은 \(J\), SE, population conservation, line source, bolometric energy 잔차가 모두 감소하는 것이다. 고정 `S_l/B` cap, epsilon floor, \(J\to0\) clamp는 금지한다. Backtracking은 잔차가 3회 연속 감소하지 않으면 해당 rung를 실패 처리하며 물리 source를 바꾸지 않는다.

안정성의 근거는 명시적 \(5247\times q^{-2}\) self-loop를 \(I-(1-\epsilon)\Lambda^\ast\) 또는 MALI rate block 안에 넣는 데 있다. 다만 중첩선 및 전역 population Jacobian의 잔여 spectral radius가 1 미만인지는 사전 보장할 수 없으므로 full-model 결과는 **UNRESOLVED**다.

## 5. 검증 사다리 — 사전등록

### KA-3.2.1: 해석적 2준위 대기

등온 semi-infinite 2준위 대기의 \(\sqrt{\epsilon}\) 법칙

\[
S(0)=\sqrt{\epsilon}\,B
\]

을 표면 oracle로 둔다. Eddington/two-stream 전체 해는

\[
{S(\tau)\over B}
=1-(1-\sqrt{\epsilon})
e^{-\sqrt{3\epsilon}\tau}.
\]

\(\sqrt{\epsilon}\) 표면 법칙은 고전 2준위 NLTE 검증 기준이다([MNRAS 186, 369](https://academic.oup.com/mnras/article/186/2/369/994143)).

시험점:

\[
\epsilon=10^{-2},10^{-4},10^{-6},10^{-8},
\qquad
T=10,10^2,10^4,10^6.
\]

finite slab formal 해는 기존 KA2 Fredholm 80-digit Nyström oracle를 확장한다. 현재 KA2의 적분방정식·수치 oracle 계약은 이미 존재한다([Stage 3.1 설계](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md:436)).

Acceptance:

- direct/high-precision 해 대비 \(J,S\) 상대 L2 \(\le10^{-8}\)
- 표면 \(\sqrt\epsilon\) 상대 오차 \(\le1\%\) — truncation/depth refinement 후
- source residual \(\le10^{-10}\), transport residual \(\le10^{-4}\)
- clamp/nonfinite/inversion count 0

### KA-3.2.2: Λ-반복 대 ALI 수렴률

작은 격자에서는 unit-source formal solves로 이산 Λ를 직접 조립한다.

\[
M_{\rm LI}=(1-\epsilon)\Lambda,
\]

\[
M_{\rm ALI}
=[I-(1-\epsilon)\Lambda^\ast]^{-1}
(1-\epsilon)(\Lambda-\Lambda^\ast).
\]

Acceptance:

- 직접 eigenvalue의 \(\rho\)와 반복 후반 잔차비의 차이 \(\le5\%\)
- ALI와 direct solve의 최종 해 상대 L2 \(\le10^{-10}\)
- 모든 \(T\ge10^2,\epsilon\le10^{-4}\)에서
  \(\rho_{\rm ALI}<\rho_{\rm LI}\)
- 대각 ALI가 100회 안에 \(10^{-8}\) 잔차에 못 도달하거나 \(\rho_{\rm ALI}\ge0.8\)이면 공간 삼대각 rung로 후퇴

### KA-3.2.3: Sobolev 국소 해

단일 line-shell에 대해

\[
\bar J=(1-\beta)S+\beta J_{\rm ext}
\]

및 위의 폐형식 \(S\)를 oracle로 사용한다.

Acceptance:

- \(\tau=10^{-6}\ldots10^{12}\), \(\epsilon_0=10^{-8}\ldots1\)
- analytic \(S,\bar J\) 상대오차 \(\le10^{-12}\)
- \(\epsilon_0\to0\), 유한 \(\tau\)에서 경계 escape로 finite
- 닫힌 무한 pure-scattering domain의 비유일성은 성공으로 위장하지 않고 **UNRESOLVED-SINGULAR**

### KA-3.2.4: coarse projection falsifier

같은 bin 안에 서로 다른 \((S_l,\beta_l)\)를 가진 두 선을 놓는다.

- coarse-only가 line-specific \(J_l\)를 재현하지 못하면 예상된 FAIL.
- line-center projection은 line oracle 각각 1%, bin-integrated energy \(10^{-10}\) 이내.
- UV active-set을 99%, 99.9%, 99.99%로 늘려 band \(J\) 변화가 1% 미만이 되는 지점을 인증.

### 동결 상태 벤치

중요한 사전등록은 세 경우를 분리한다.

1. **ALI만 켜고 source physics 불변:** 수렴한 해의 예측은 정확히  
   \[
   \Delta J_{\rm det}=0.
   \]
   ALI가 CMFGEN 쪽으로 장을 이동시키면 fixed-point 불변성 실패다.

2. **E9 scalar-\(\epsilon_{\rm MC}\) proxy:** 기존 사전등록/실측 비교값은 다음과 같다.

| band Å | 조건부 예측 \(J_{\rm det}/J_{\rm CMFGEN}\) | 기존 frozen solve |
|---|---:|---:|
| 600–1000 | 8.4558 | 8.2906 |
| 1000–1500 | 4.93745 | 4.91614 |
| 1500–2000 | 1.85533 | 1.83988 |
| 2000–2500 | 0.211522 | 0.208361 |
| 2500–3000 | 0.331338 | 0.336805 |
| 600–3000 | **0.936647** | **0.932288** |

이는 scalar source proxy의 검증치일 뿐 native `SRC_NLTE` acceptance가 아니다([E9](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E9.md:95)).

3. **native \(\chi_l,\eta_l\)+MALI:** 정확한 band 수치는 iter-10 하위 population/line별 χ 계기가 없어 **UNRESOLVED**다([UV consolidation](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:50)). 방향 사전등록만 둔다: BALL과 5개 sub-band 중 적어도 4개가 log-distance 기준 CMFGEN 쪽으로 이동해야 하며, 수치 acceptance는 계기 생성 전에 별도 봉인해야 한다.

### UV 회귀표 상설 편입

모든 radiation/population 변경은 s0, s8, s20에서 B0–B4와 BALL을 exact-edge Hz 가중으로 보고한다.

필수 열:

- `J_det/CMFGEN`, `J_det/previous`, `J_det/MC`
- source/SE/energy residual
- LI/ALI iteration 수와 측정 \(\rho\)
- native line opacity/emissivity power
- inversion/undefined/omitted active-cell 수
- population/rate/opacity/source iteration 및 SHA
- binary/source/config/bin-edge SHA
- clamp/floor/cap counter
- 결과 파일의 immutable 경로와 checksum

iteration을 명시하지 않은 overwrite는 실패다. 이는 기존 형광 행렬이 iteration 10에서 11로 덮어써졌는데 sidecar 검사만 통과한 N4 재발을 막는 계약이다([UV consolidation](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:57)). UV 표가 1년간 상설 측정되지 않았다는 계보 결론도 동일 문서에 기록돼 있다([UV consolidation](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/UV_CENSUS_CONSOLIDATION.md:25)).

## 6. Stage 4 인터페이스

Stage 4에 dense \(\delta J/\delta{\rm pop}\) 행렬을 제공하지 않는다. transition 수 규모상 물질화가 불가능하다. 대신 epoch-bound matrix-free JVP를 제공한다.

기본 residual:

\[
F(J,n,T)=J-\Lambda_{\chi(n,T)}[S(J,n,T)].
\]

source-form approximate response:

\[
\delta J\simeq
[I-\Lambda^\ast S_J]^{-1}\Lambda^\ast
(S_n\delta n+S_T\delta T+S_{n_e}\delta n_e).
\]

Opacity 변화까지 포함한 정확한 tangent formal equation은 ray마다

\[
\left({d\over ds}+\chi\right)\delta I
=\delta\eta-I\,\delta\chi
\]

이며, 기존 formal sweep과 같은 순서로 한 번 풀면 \(\delta J\)를 얻는다.

권고 API 계약:

```text
stage32_apply_lambda_star(delta_source -> delta_J_local)
stage32_solve_local_resolvent(rhs -> delta_J)
stage32_apply_transfer_jvp(delta_eta, delta_chi -> delta_J)
stage32_apply_rate_preconditioner(delta_pop -> delta_rates, delta_J)
```

모든 호출은 동일한 `OperatorSnapshot`을 소비해야 한다.

```text
grid_sha
population_epoch
rate_epoch
opacity_epoch
source_epoch
lambda_epoch
active_line_set_sha
```

Stage 4 Newton/GMRES는 위 JVP로 \(\delta J/\delta n\) 작용만 요청한다. Stage 3.2가 제공하지 못하는 부분은 full \(\partial\Lambda/\partial\chi\) 행렬이며, tangent formal sweep이 이를 대신한다. 구 코드에는 삼대각 resolvent를 Stage 4에 넘기려는 인터페이스가 있으나 one-hop·floor·cap 근사이므로 새 JVP의 oracle이 될 수 없다([lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:12653)).

## 7. 단계·규모·후퇴 기준

아래 규모는 설계 추정이며 active-line census 전까지 **UNRESOLVED-ESTIMATE**다.

| rung | 물리 계약 하나 | 예상 규모 | 통과/후퇴 |
|---|---|---:|---|
| R1 | SC formal operator에서 대각 Λ*를 동일 이산화로 산출 | 300–500 LOC, 0.5–1 PW | unit-source Λ 대각 불일치 \(>10^{-10}\)이면 중단 |
| R2 | 2준위 coarse ALI의 fixed-point 불변성 | 500–800 LOC, 1–1.5 PW | direct 해 불일치 또는 \(\rho\ge1\)이면 승인 불가 |
| R3 | Sobolev \(\Lambda^\ast=1-\beta\)와 raw \(\epsilon_0\) 계약 | 500–900 LOC, 1–1.5 PW | \(C/(C+A\beta)\) 이중소비 발견 시 중단 |
| R4 | line-center 투영 및 energy-preserving active CSR | 1,000–2,000 LOC, 2–4 PW | 99.99% active-set에서 UV 변화 \(>1\%\)면 selection 폐기, 더 세밀화 |
| R5 | native \((\chi_l,\eta_l)\) 및 inversion fail-closed | 800–1,500 LOC, 2–3 PW | Planck fallback·source cap·epoch 혼식이 하나라도 있으면 미활성 |
| R6 | Rybicki–Hummer MALI rate preconditioner | 1,000–1,800 LOC, 2–3 PW | 결합 잔차가 line search 3회 연속 증가하면 frozen-pop 단계로 후퇴 |
| R7 | frozen benchmark·UV 상설표·계보 gate | 500–1,000 LOC/scripts, 1–2 PW | 표/SHA/epoch 누락 시 production gate 금지 |
| R8 | 선택적 삼대각 Λ* 또는 VEF moment preconditioner | 800–1,500 LOC, 1–2 PW | 대각 ALI가 이미 기준을 만족하면 수행하지 않음 |
| R9 | Stage 4 tangent/JVP 연결 | 700–1,200 LOC, 1–2 PW | JVP finite-difference 검증 실패 시 Stage 4 차단 |

총 예상치는 약 6–10 kLOC와 12–20 person-week다. line-center active 수, overlap block 크기, GPU 이관 여부가 최대 불확정 요인이다.

허용되는 후퇴는 “더 강한 정본 연산자”뿐이다.

- 대각 실패 → 공간 삼대각 또는 overlap block.
- dense line-space 비용 실패 → active CSR/matrix-free.
- coupled MALI 실패 → frozen-pop native solve까지만 승인하고 Stage 4 대기.
- coarse projection 실패 → line-center refinement.

same-bin coherent 재주입, Planck fallback, epsilon floor, \(S_l/B\) cap으로의 후퇴는 금지한다.

## 최종 판정

Stage 3.2가 제거해야 할 것은 물리적 fluorescence가 아니라 명시적 self-coupling의 \(5247\times q^{-2}\) 이득이다. 이를 위해서는 coarse-bin ALI만으로는 부족하며 다음 세 조건이 동시에 필요하다.

1. Sobolev line-space \(\Lambda^\ast=1-\beta_l\).
2. population-native \((\chi_l,\eta_l)\)의 짝 소비.
3. 동일 Λ*를 rate matrix 안에서 쓰는 Rybicki–Hummer MALI.

이 세 조건과 line-center projection, inversion fail-closed, epoch 계보가 통과한 뒤에만 `LUMINA_CMFGEN_SRC_NLTE=1`을 허용한다. native coupled UV 결과의 정확한 대역 수치는 현재 계기 부족으로 **UNRESOLVED**이며, ALI 자체는 accelerator이므로 source physics를 바꾸지 않은 상태에서는 \(J_{\rm det}\)를 CMFGEN 방향으로 이동시켜서는 안 된다.