# Codex A′-rev1 — Stage 3.1 설계 수정 보고서

판정: **§2.3 개정이 필요하며, 외곽 half-cell 수정만으로는 KA1 차수 회복이 불가능하다.** 관측된 실패는 세 원인의 중첩이다.

1. \(\chi R=100\) 붕괴와 max 오차: 외곽 half-cell 상수 연장.
2. thin/moderate의 intensity 절단 오차: 접선 근방 \(ds=O(h^{1/2})\) 세그먼트에 선형 SC를 적용한 \(O(h^{3/2})\) 오차.
3. thin/moderate의 보고된 \(p_{\rm obs}\simeq1.50\): 비중첩 shell-center 점값에 fine pair-average restriction을 적용한 Richardson 측정 오차가 주성분.

GL 각도 적분이나 단순히 `Nmu`와 `Nr`를 동시에 배증한 사실 자체는 1.50차의 원인이 아니다.

저장소 파일은 변경하지 않았다.

## 1. 현 이산화의 항별 절단 오차

### 1.1 일반 ray segment

현 SC kernel은 [lumina_cmf_field.c:268](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:268)의 선형-source exact formula이다. 매끄러운 \(Q(s)\), \(\chi(s)\)와 일반 segment \(\Delta s=O(h)\)에 대해

\[
\delta I_{\rm seg}=O(\Delta s^3),\qquad
\sum_{\rm ray}\delta I_{\rm seg}=O(h^2).
\]

`radial_value()`의 중심 간 선형 보간도 내부에서 \(O(h^2)\)이다. 따라서 접선과 경계를 제외하면 설계대로 2차다.

### 1.2 접선 근방: 독립적인 \(h^{3/2}\) 성분

[path_build()](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:314)은 radial center를 \(z=\pm\sqrt{r^2-p^2}\)로 변환하고 \(z=0\)을 turning node로 넣는다. 접선 바로 위 중심이 \(r=p+\theta h\)이면

\[
L=z_1=\sqrt{(p+\theta h)^2-p^2}
     =\sqrt{2p\theta h+O(h^2)}
     =O(h^{1/2}).
\]

매끄러운 구면 source는 접선에서

\[
S(\sqrt{p^2+z^2})
 =S(p)+\frac{S'(p)}{2p}z^2+O(z^4).
\]

이를 양 끝값을 잇는 선형 SC로 근사하면, 상수 \(\chi\)의 경우 leading error는

\[
\delta I_{\rm tan}
 =
 \chi\frac{S'(p)}{2p}
 \int_0^L z(L-z)e^{-\chi(L-z)}\,dz
 =
 \frac{\chi S'(p)}{12p}L^3+O(L^4)
 =
 O(h^{3/2}).
\]

KA1의 \(S=1+\tfrac12r^2\)에서는 계수가 양수이므로 선형 chord가 정확한 quadratic source보다 높다. 실제 h4 최대 오차도 이 부호와 일치한다.

| \(\chi R\) | h4 최악 좌표 | 방향 | \(I_{\rm num}-I_{\rm exact}\) |
|---:|---|---|---:|
| \(10^{-3}\) | \(i=121,m=12,r=0.94921875,\mu=0.33406569885893617\) | plus | \(+3.0107020563\times10^{-7}\) |
| \(1\) | \(i=110,m=7,r=0.86328125,\mu=0.13390894062985514\) | plus | \(+2.4087205732\times10^{-4}\) |
| \(100\) | \(i=127,m=9,r=0.99609375,\mu=0.20614212137961885\) | minus | \(-1.1414637337\times10^{-3}\) |

thin/moderate의 최악점은 outermost cell이 아니며, outer constant extension이 만드는 음의 편향과 반대 부호다. 따라서 그 intensity 오차는 외곽 half-cell 단독으로 설명되지 않는다.

### 1.3 외곽 half-cell: thick max 오차의 직접 원인

[radial_value()](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:363)은 \(r\ge r_{N-1/2}\)에서 마지막 중심값을 그대로 반환한다. 매끄러운 \(f\)에 대해

\[
f(R)-f(R-h/2)=\frac h2f'(R)+O(h^2),
\]

이므로 boundary value 자체는 1차다.

\(\chi R=100\) 최악점은 outer boundary에서 해당 incoming target까지가 첫 SC segment다. 이 좌표에서

\[
p=0.9746997101812538,\quad
\Delta s=0.018181520910730553,\quad
\Delta\tau=100\Delta s=1.8181520910730553.
\]

실패 보고서의 \(0.390625=100/(2\cdot128)\)는 radial half-cell optical depth이지 최악 oblique ray의 실제 segment optical depth는 아니다. 실제 값은 1.818로 더 비점근적이다.

상수 연장값은

\[
S_{127}=1.4961013793945312,
\]

따라서 현 수치값은 정확히

\[
S_{127}(1-e^{-1.8181520910730553})
=1.2532460698223082,
\]

이며 이는 보고된 `I_numeric=1.253246069822307`과 일치한다. 즉 thick max 실패는 첫 boundary segment만으로 완전히 재현된다.

### 1.4 GL \(\mu\) quadrature는 원인이 아님

exact intensity만 GL로 적분하고 256-point GL을 기준으로 비교한 angular RMS 오차는 다음과 같다.

| \(\chi R\) | 32×8 | 64×16 | 128×32 |
|---:|---:|---:|---:|
| \(10^{-3}\) | \(7.35\times10^{-11}\) | \(1.88\times10^{-14}\) | \(\sim10^{-18}\) |
| \(1\) | \(7.18\times10^{-8}\) | \(1.86\times10^{-11}\) | \(\sim10^{-15}\) |
| \(100\) | \(3.39\times10^{-8}\) | \(6.91\times10^{-11}\) | \(\sim10^{-15}\) |

이는 관측된 \(10^{-4}\)–\(10^{-3}\)급 오차보다 훨씬 작다. `Nmu` 배증은 충분하며 GL 자체의 재설계는 필요 없다.

### 1.5 \(p_{\rm obs}\simeq1.50\)의 주원인: 잘못된 radial restriction

현 runner의 [difference_norm()](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_stage31_cmf_ka.py:57)은

\[
J_h(r_i)-\frac12[J_{h/2}(r_i-h/4)+J_{h/2}(r_i+h/4)]
\]

를 사용한다. 그러나 solver의 `J`는 cell average가 아니라 shell-center 점값이다.

진공 외곽에서 chord 길이는

\[
\sqrt{R^2-r^2+r^2\mu^2}.
\]

\(\delta=R-r\)로 두고 \(\mu\) 적분하면

\[
J(R-\delta)
  =J(R)+C\,\delta\log\delta+O(\delta).
\]

따라서 outer 몇 개 cell에서 pair-average interpolation defect는 \(O(h)\)이며, unweighted discrete radial L2에서는

\[
\left[h\,O(h^2)\right]^{1/2}=O(h^{3/2}).
\]

실제로 solver 오차를 완전히 제거하고 exact \(J\)만 같은 restriction에 넣어도 다음 차수가 나온다.

| \(\chi R\) | 관측 solver \(p_{\rm obs}\) | exact-field restriction \(p\) |
|---:|---:|---:|
| \(10^{-3}\) | 1.503194 | 1.487327 |
| \(1\) | 1.503583 | 1.473926 |
| \(100\) | 0.464061 | 0.482872 |

따라서 thin/moderate의 보고된 1.50은 주로 Richardson 관측 연산자가 만든 값이다. 다만 §1.2의 실제 SC \(h^{3/2}\) 오차도 독립적으로 존재하므로 restriction만 바꿔서는 충분하지 않다.

정리하면:

- 경계 half-cell 단독 원인: **아님**.
- ray-tangent 국소 성분: **실제 \(h^{3/2}\) 절단 오차**.
- joint refinement 결합: `Nr`–`Nmu` 결합 자체가 아니라 **비중첩 radial 점값 restriction**이 \(h^{3/2}\)를 생성.
- thick 붕괴: **외곽 half-cell의 비점근 optical segment가 직접 원인**.

## 2. §2.3 경계 후보 비교 및 확정안

| 후보 | 2차 보장 | 구현 국소성 | 판정 |
|---|---|---|---|
| 1. 중심값의 one-sided 선형 외삽 | boundary 값 \(O(h^2)\). 아래 path-length 제한과 결합하면 intensity/J 전역 \(O(h^2)\) | `radial_value()` 중심의 최소 변경 | **채택** |
| 2. boundary segment 정확 적분 | 선언된 reconstruction에 대해서는 정확. 하지만 center samples만으로 실제 물리 적분은 “정확”하게 정의할 수 없음 | variable \(\chi,\eta\)용 새 적분기와 residual 재작성 필요 | 불채택 |
| 3. source grid 자체를 face-centered로 변경 | \(R\) 값이 직접 주어지면 정합 가능 | input ABI, frozen schema, indexing 전부 변경. 중심값에서 face를 재구성하면 후보 1과 동일 | production source grid에는 불채택 |

균일 격자에서 외곽 face 값은

\[
\widehat f(R)=\frac32f_{N-1}-\frac12f_{N-2}
            =f(R)-\frac38h^2f''(R)+O(h^3).
\]

내곽 face도

\[
\widehat f(r_{\rm in})=\frac32f_0-\frac12f_1
\]

로 대칭 처리한다. `chi`와 `eta`를 각각 외삽하고, 결과가 음수 또는 non-finite이면 limiter나 clamp를 쓰지 않고 해당 face/frequency를 기록한 뒤 실패해야 한다.

후보 1만 적용하면 접선 \(h^{3/2}\) 성분은 남는다. 따라서 다음 절의 path-length 수정은 확정안의 필수 부분이다.

## 3. 잔여 \(h^{3/2}\) 성분 개정

### 3.1 SC path-length subcycling

각 기존 ray segment에 대해 local radial reconstruction scale \(h_{\rm loc}\)를 정하고

\[
n_{\rm sub}
 =\max\left(1,\left\lceil\frac{\Delta s}{h_{\rm loc}}\right\rceil\right),
\qquad
\Delta s_{\rm sub}=\Delta s/n_{\rm sub}
\]

로 나눈다. 각 subnode에서 기존 `radial_value()`로 \(\chi,\eta\)를 평가하고 기존 `sc_step()`을 반복한다.

그러면 모든 substep에서 \(\Delta s_{\rm sub}=O(h)\)이므로

\[
\sum\delta I_{\rm sub}
  =O(h^{-1})O(h^3)
  =O(h^2).
\]

이 방식은 새 parabolic-SC weight, source limiter 또는 clamp를 도입하지 않으며 이미 검증된 선형 SC kernel을 재사용한다. quasi-uniform mesh와 매끄러운 구면 field, 중심에서 \(S'(0)=0\)이라는 통상 정칙성 조건 아래 2차를 보장한다.

### 3.2 Richardson용 nested face evaluation

production source 배열은 계속 shell-centered로 둔다. 다만 KA1 관측점은

\[
r_j=jR/N_r,\qquad j=0,\ldots,N_r
\]

인 nested face 좌표로 바꾼다. fine-to-coarse restriction은 pair-average가 아니라 정확한 injection

\[
(\mathcal R J_{h/2})_j=J_{h/2,\,2j}
\]

을 사용한다. 이로써 동일 물리 좌표의 해를 비교하며 \(\delta\log\delta\) radial interpolation defect가 사라진다. 제외점은 여전히 0이다.

이는 후보 3의 “source storage 변경”을 채택하는 것이 아니라, KA 평가 좌표만 face-aligned/nested로 만드는 수정이다.

## 4. KA1 격자 재규정과 기대 수렴 사전등록

기존 32/64/128 thick 계열은 \(\chi h\)가 커서 경계 수정 뒤에도 pre-asymptotic cancellation이 강하다. acceptance 수치는 그대로 두고 공통 grid family만 다음으로 재규정한다.

\[
(N_r,N_\mu)
 =(128,32),\ (256,64),\ (512,128).
\]

- 세 \(\chi R\) 모두 동일 계열을 사용한다.
- GL rule, \(N_\mu/N_r=1/4\), joint factor-two refinement는 유지한다.
- `p_obs=log2(||u_h-u_{h/2}||/||u_{h/2}-u_{h/4}||)`는 유지한다.
- fine restriction만 nested face injection으로 바꾼다.
- `I/J <=1e-4`, max `<=3e-4`, \(1.8\le p\le2.2\), residual `<=1e-4`를 포함한 acceptance는 하나도 완화하지 않는다.

개정 구현을 모사한 독립 계산에 근거한 사전등록 예상치는 다음과 같다.

| \(\chi R\) | 예상 \(p_{\rm obs}(J)\) | 사전 예상 구간 | acceptance |
|---:|---:|---:|---:|
| \(10^{-3}\) | 1.98 | 1.90–2.06 | 1.8–2.2 유지 |
| \(1\) | 1.99 | 1.90–2.08 | 1.8–2.2 유지 |
| \(100\) | 2.02 | 1.85–2.15 | 1.8–2.2 유지 |

이는 구현 전 예측이지 PASS 선언이 아니다. 실제 결과가 창 밖이면 acceptance를 재조정하지 않고 다시 FAIL로 기록해야 한다.

## 5. 구현 지시

### C solver

1. [radial_value():363](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:363)

   - 양 끝 constant return을 one-sided linear extrapolation으로 교체.
   - extrapolated `chi/eta`의 non-finite/negative를 오류로 반환할 수 있도록 현재 `double` 반환을 status+out parameter 형태로 변경.
   - limiter와 zero clamp 금지.

2. [path_build():314](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:314)

   - `radial_index`만 받지 말고 임의 `target_r`를 받을 수 있도록 내부 helper 분리.
   - target \(\pm r\mu\) node를 정렬 삽입하고 boundary/tangent와 중복되면 deduplicate.
   - production center 평가 경로는 기존 결과와 호환되게 유지.

3. [solve_static_ray():455](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:455)

   - 각 segment의 `n_sub=ceil(ds/h_loc)` 계산.
   - subnode마다 `chi/eta` 재구성 후 기존 `sc_step()` 호출.
   - residual도 실제 substep별로 누적하여 coarse 원 segment residual로 가장하지 않음.

4. [ray cache:225](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:225), [solve loop:545](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:545)

   - 임의 evaluation radius용 cache builder 추가.
   - 기본 호출은 shell centers, KA 옵션에서는 `r_edge[0..Nr]`를 평가.
   - `LCMFResult.nr`가 source `input.nr`와 달라질 수 있으므로 `n_r_eval`을 명시적으로 보유하도록 API를 정리하는 편이 안전하다.

### KA driver/runner

1. [driver:23](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/stage31_cmf_ka_driver.c:23)

   - KA 출력 좌표를 centers에서 faces \(j/N_r\)로 변경.
   - \(r=0,R\) 포함, 제외점 0.
   - source sampling은 계속 shell center에서 수행.

2. [grid declaration:17](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_stage31_cmf_ka.py:17)

   - 공통 grid를 `(128,32),(256,64),(512,128)`로 변경.

3. [difference_norm():57](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_stage31_cmf_ka.py:57)

   - `0.5*(fine[2i]+fine[2i+1])`를 `fine[2i]` injection으로 교체.
   - coarse/fine radius bitwise 또는 상대 \(10^{-14}\) 일치 검사 추가.

4. exact oracle loop [run_stage31_cmf_ka.py:79](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_stage31_cmf_ka.py:79)

   - 동일 face 좌표에서 exact \(I,J\) 평가.
   - 기존 80-digit oracle와 모든 acceptance 수치는 유지.

### 필수 회귀시험

- 상수·선형 radial field의 boundary extrapolation exactness.
- quadratic field의 boundary value 및 intensity 오차비 \(\simeq4\).
- 고정 \(p\) tangent-ray에서 subcycling 전 \(p\simeq1.5\), 후 \(p\simeq2\).
- exact-field-only Richardson에서 nested injection \(p\simeq2\).
- GL-only 오차가 spatial 오차의 1/20 이하인지 확인.
- extrapolated `chi/eta<0` 입력은 clamp 없이 좌표와 값이 포함된 실패.
- 기존 outer incoming zero와 deterministic hash 3회 재검증.

## 6. UNRESOLVED

- **UNRESOLVED-EXTRAP-POSITIVITY:** production frozen field에서 one-sided extrapolation이 음의 \(\chi\) 또는 \(\eta\)를 만드는지는 실제 입력 없이는 알 수 없다. limiter로 숨기지 말고 fail-closed 해야 한다.
- **UNRESOLVED-NONUNIFORM-MESH:** 위 2차 증명은 quasi-uniform refinement를 전제로 한다. 강하게 graded된 production mesh에는 \(h_{\rm loc}\) 정의와 mesh-ratio 상한을 별도 명시해야 한다.
- **UNRESOLVED-API-EVAL-GRID:** arbitrary evaluation radii를 `LCMFOptions`에 넣을지 별도 diagnostic API로 둘지는 ABI 소유자의 결정이 필요하다. 수치 설계에는 영향이 없다.
- **UNRESOLVED-PREDICTION:** 위 세 예상 \(p_{\rm obs}\)는 구현 전 사전등록값이다. 실제 KA1 재실행 전에는 KA1을 PASS로 바꿀 수 없다.
- KA2, KA3, scattering rung은 여전히 미실행 상태이며 KA1 재통과 전 진입하면 안 된다.