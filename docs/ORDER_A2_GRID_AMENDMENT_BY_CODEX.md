# 개정 발주서 — A2-02 주파수 합집합과 line `\bar J` 계약

- 발주일: 2026-08-05
- 상태: 구현 전 확정 개정안
- 저작: Codex
- 검수·실행·회귀 원장: 운전석
- 적용 대상: `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md`의 §2·§4와 그 직접 하류 단계
- 변경 범위: 이 문서 한 파일뿐이다. 코드·덱·`src`·검증 산출물은 이 발주에서 변경하지 않는다.
- 발주 근거: `docs/OUTSIDE_LOOP_POOL.md`의 A2-02 `BLOCKED` 원장과 커밋
  `43ffe3186f926e887139228d465a8c63fa5c42a8`

---

## 1. 개정 효력과 범위

이 개정은 원 발주서 §7 말미의 “합격선 변경은 원장 근거와 별도 발주” 경로를
집행한다. A2-02의 8000→16000 사다리에서 `Γ`, 대역 `J`, `χ`, `η`는 모두
PASS했으나 line `\bar J`만 FAIL했고, 그 원인이 다음 두 설계 계약으로 분리되어
실측되었기 때문이다.

1. `bound_bound_line_frequency_profile`이 런타임 수송 창 밖의 덱 장부선까지 모두
   합집합에 넣어 하한을 `2.998e6 Hz`로 내렸다.
2. 실제 UV 선숲의 profile-가중 `\bar J`는 전역 성긴 빈을 재빈하는 방식으로
   실용 빈 수 안에서 수렴하지 않았다. 유효 42레코드 중 11건이 1%를 넘었고,
   최악은 s8 Fe II `l61→u1308`의 7.6%였다.

이 문서는 다음 조항만 개정한다.

- 원 발주서 §4의 “bound-bound 선 주파수와 profile 폭” 소비자 정의
- 원 발주서 §2.2의 generation 결박 파생 캐시 해석과 §2.3의 line `\bar J` 계산 경로
- A2-02 사다리의 line 지표와 재판정 절차
- A2-03·A2-04·A2-05·A2-06·A2-12·A2-13의 직접 파급

그 밖의 원 발주서 조항, 특히 단일 `J_ν` 소유권, 공이동 frame, 단위, validity,
1%·0.2% 해상도 합격선, clamp·floor 금지는 그대로다.

---

## 2. 개정 1 — §4 bound-bound 선 소비자 규칙

### 2.1 규범적 물리 창

원 발주서 §4의 “bound-bound 선 주파수와 profile 폭”은 다음으로 교체한다.

> bound-bound 합집합 소비자는 현재 수송 코드가 역사적으로 처리해 온 line-center
> 창 `100 Å ≤ λ_lu ≤ 20000 Å` 안의 전이와, 그 전이에 대해 런타임이 실제 사용하는
> 유한 profile support이다.

파장 경계가 규범값이며 `ν=c/λ`는 파생값이다. 따라서

\[
\mathcal D_{bb}=\left[{c\over20000\,\mathring{\rm A}},
                       {c\over100\,\mathring{\rm A}}\right]
\]

이고, line center `ν_lu`가 `\mathcal D_bb`에 포함될 때만 그 line ID가
`bound_bound_line_frequency_profile` 소비자로 인정된다. 경계는 포함한다.

이 창은 새 세기 cutoff가 아니다. 기존 정본 수송 격자의 명시 범위
100–20000 Å를, 지금까지 line loader와 수송 경로가 암묵적으로 적용해 온 소비 범위의
계약으로 승격한 것이다. 20000–25000 Å 등록 검증 대역은 전역 `J_ν` coverage를
확장하지만 bound-bound rate 소비자 목록을 소급 확장하지 않는다. 이 line 창을 넓히는
일은 해당 전이의 rate·population 파급과 원장 근거를 가진 별도 개정으로만 한다.

### 2.2 line census와 profile support

덱 line list의 모든 행은 다음 둘 중 정확히 하나로 분할한다.

- `BB_IN_DOMAIN`: 유한·양의 `ν_lu`이고 `ν_lu∈D_bb`인 전이
- `BB_EXCLUDED_OUTSIDE_DOMAIN`: 유한·양의 `ν_lu`이지만 `ν_lu∉D_bb`인 전이

비유한 주파수, 0 또는 음의 주파수, 잘못된 level 연결은 제외 목록으로 정상화하지
않는다. 이는 atomic-input 오류이며 합집합 작성 자체를 FAIL시킨다.

`BB_IN_DOMAIN` 전이는 line center 하나가 아니라 등록된 런타임 profile support 전체를
합집합에 기여한다. 현 profile 계약이 `ν_lu(1±4v_D/c)`이면 바로 그 support를 쓰며,
셸별 `v_D`가 다르면 모든 판정 셸 support의 합집합을 쓴다. profile 종류, 정규화,
truncation 배수와 그 입력의 provenance를 합집합 manifest에 기록한다. profile support가
선언되지 않았는데 임의 default 폭을 넣는 것은 FAIL이다. 이후 profile 계약이 넓어지면
합집합은 넓힐 수 있지만, 원장 갱신 없이 좁힐 수 없다.

`A_ul`, `gf`, 예상 population, 예상 rate, line strength에는 어떤 수치 floor도 두지
않는다. 창 안의 약한 선도 소비자다. 이번에 관측된 `A_ul≈1e-16 s^-1` Ni I 전이가
제외되는 이유는 약해서가 아니라 line center가 문서화된 수송 창 밖이기 때문이다.

### 2.3 제외 목록은 판정 산출물이다

A2-02 재실행은 해시 결박된 전량 제외 목록을 먼저 산출해야 한다. 각 행은 적어도 다음을
가진다.

```text
line_id, element, ion, lower, upper, nu_lu_hz, lambda_lu_A,
reason=BB_EXCLUDED_OUTSIDE_DOMAIN, source_row, source_hash,
domain_contract_hash
```

manifest에는 원 line census 행 수, `BB_IN_DOMAIN` 수, 제외 수, 두 집합의 해시와
`in_domain + excluded = finite_positive_input_census` 항등식을 기록한다. 누락·중복은
FAIL이다. 제외된 전이는 atomic bookkeeping과 spontaneous `A_ul`에는 남을 수 있으나,
`J_ν` 또는 `\bar J` 소비자로 가장할 수 없다. active rate graph가 제외 line의
`\bar J`를 요청하면 0이나 작은 값을 돌려주지 말고 `OUT_OF_BB_DOMAIN`으로 실패한다.

다른 등록 소비자가 같은 주파수를 자기 근거로 합집합에 넣는 것은 허용한다. 그러나
그 주파수 coverage가 생겼다는 사실로 제외 line ID의 bound-bound 소비 자격이 되살아나지는
않는다.

### 2.4 §9 clamp·floor 계약과의 관계

이 규칙은 유효한 물리값을 다른 값으로 바꾸는 guard가 아니다. 소비자 외연을 기존 수송
도메인으로 선언하고, 도메인 밖 요청을 상태로 거부한다. 특히 다음은 금지한다.

- 제외 line의 `\bar J`를 0, `1e-30`, 가장자리 빈 또는 이웃 line 값으로 반환
- `A_ul`, `gf`, population 또는 예상 기여도로 제외 여부 결정
- 제외 목록을 만들지 않고 loader 단계에서 조용히 drop
- continuum·observer coverage를 근거로 제외 line을 rate 소비자로 재등록

### 2.5 사전등록 음성대조

다음 음성대조는 개정 합집합 checker와 독립 실행하고, 주입된 결함마다 비0 종료와
명시적 FAIL marker가 나와야 한다.

1. 실측 최악 저주파 전이 Ni I `l461→u462`를 `BB_IN_DOMAIN`에 강제 삽입하면 FAIL.
2. 같은 제외 line ID로 `\bar J` 조회 또는 stimulated rate를 요청하면 값 반환 없이 FAIL.
3. 제외 목록에서 한 행을 삭제·중복하거나 source/domain hash를 바꾸면 census 항등식 또는
   결박 검사가 FAIL.
4. 100 Å와 20000 Å의 안쪽·경계·바깥쪽 fixture가 포함 규칙과 다르면 FAIL.
5. 창 안에 있으나 작은 `A_ul`을 가진 fixture를 strength floor로 제외하면 FAIL.
6. 제외 line과 같은 주파수를 continuum 소비자가 덮을 때 line ID까지 다시 활성화하면 FAIL.

---

## 3. 개정 2 — §2 line `\bar J` 계산 계약

### 3.1 권고안: 전역 격자 + 선택적 direct estimator의 혼합

채택안은 혼합안이다.

- 전역 정본 `J_ν[s][b]`는 개정 1의 합집합과 개정 3의 격자 사다리로 선택한 빈에
  계속 저장한다.
- line `\bar J`만 전역 `J_ν` 빈의 profile 재적분으로 생산하지 않고, 같은 MC
  path-length measure에서 line별로 직접 추정한다.
- 결과는 정본 `RadiationField.generation`에 결박된 읽기 전용 파생 캐시이며 별도
  복사장 정본이 아니다.

수학적 목표는 바뀌지 않는다.

\[
\bar J_{lu,s}=\int\phi_{lu,s}(\nu)J_\nu(s)\,d\nu,
\qquad \int\phi_{lu,s}(\nu)\,d\nu=1.
\]

생산 계산은 같은 generation을 만드는 공이동 MC segment들로 다음과 동형인 선택적
path-length estimator를 쓴다.

\[
\widehat{\bar J}_{lu,s}=
{1\over4\pi V_s\Delta t}
\sum_{p\cap s}\int_{\mathrm{segment}}
\epsilon_p(\ell)\,\phi_{lu,s}(\nu'_p(\ell))\,d\ell .
\]

실제 packet energy·frequency 변환과 segment 분할은 전역 `J_ν` estimator와 동일한
frame·시간·체적·`4π` 정규화 계약을 사용한다. segment 안에서 공이동 주파수가 변하면
line center의 point sample로 대체하지 않고 등록된 segment 적분 규칙을 쓴다.

### 3.2 선택 집합과 캐시 권한

generation `g`의 선택 집합 `Q_g`는 그 generation의 enabled bound-bound rate graph가
요청하는 `BB_IN_DOMAIN` line ID와 셸의 집합이다. 다음을 지킨다.

- `Q_g`는 estimator 누적 전에 고정하고 hash를 남긴다.
- 선택 기준은 rate graph의 의미적 도달성뿐이다. `A_ul`, `gf`, population, 직전
  generation의 rate 또는 estimator 크기로 솎지 않는다.
- generation 도중 rate graph가 바뀌면 같은 cache에 덧대지 않고 다음 generation의
  새 `Q`로 commit한다.
- `Q_g`에 없는 active rate 요청과 `BB_EXCLUDED_OUTSIDE_DOMAIN` 요청은 둘 다 명시적
  오류다. 0 또는 전역 격자 적분 fallback은 금지한다.

파생 캐시는 적어도 다음을 보존한다.

```text
generation, shell_id, line_id, profile_id, profile_hash,
jbar_value, units, frame, validity, sample_count,
variance_or_standard_error, q_set_hash, provenance
```

`MEASURED`, `EXACT_ZERO`, `UNSAMPLED`, `OUT_OF_BB_DOMAIN`은 서로 다른 validity다.
`UNSAMPLED` active line은 rate 계산을 BLOCK하며 `EXACT_ZERO`로 승격하지 않는다.

원 발주서 §2.2의 “비정본 객체는 소비자 API가 될 수 없다”는 다음처럼 좁혀 해석한다.
rate 소비자는 cache 배열이나 별도 setter를 직접 보지 못한다. 오직
`RadiationField` 소유의 generation-checked line-derived view를 통해
`(generation,shell_id,line_id,profile_id)`로 조회한다. 캐시는 독립 lifecycle,
독립 commit 또는 정본과 다른 generation을 가질 수 없다. `J_ν`와 `Q_g` cache는 같은
원자적 commit의 두 view다.

pure-CMFGEN replay에서는 native/fine oracle로 같은 target을 계산한 읽기 전용 cache를
만들 수 있으나 provenance를 `CMFGEN_REPLAY`로 구분한다. 이를 생산 MC estimator의
통계 수렴 증거로 대신할 수 없다.

### 3.3 정본 `J_ν` 정합 게이트

선택적 estimator가 별도 복사장으로 갈라지는 것을 막기 위해 다음 세 게이트를 모두
요구한다.

1. **동일 measure·동일 commit:** 전역 `J_ν`와 `\widehat{\bar J}`가 같은 raw segment
   ledger, frame, `V_s`, `Δt`, packet normalization과 generation을 사용했음을 hash로
   증명한다.
2. **canonical projection closure:** 실제 audit segment에 대해 canonical bin마다
   상수인 정규화 control profile `φ^(N)`을 사전등록한다. direct estimator 값과
   `Σ_b J_b∫_bφ^(N)dν`가 최대 1%, 중앙값 0.2% 이내여야 한다. 이는 실제 좁은 line을
   성긴 빈으로 근사하는 시험이 아니라 정규화·frame·generation wiring 시험이다.
3. **fine diagnostic closure:** 실제 line profile audit cohort에 대해 같은 segment
   ledger로 만든 fine 진단 histogram과 direct estimator를 비교한다. fine histogram은
   런타임 정본이 아니며 profile support 해상도 수렴을 먼저 증명해야 한다. 차이는 최대
   1%, 중앙값 0.2% 이내여야 한다.

세 게이트 중 하나라도 실패하면 line cache는 유효 generation으로 commit할 수 없다.

### 3.4 대안 비교

| 안 | 장점 | 비용·위험 | 판정 |
|---|---|---|---|
| 선택적 estimator만 별도 추가(B) | 좁은 UV line을 전역 빈 폭에서 해방하고 ARTIS `jbar` 방식과 동형 | active line별 count·variance·generation lifecycle과 MC 수렴 게이트가 필요 | 전역 격자 계약과 결합해 채택 |
| 전역 `J_ν`를 fine 192,922빈으로 저장(C) | line 적분과 전역 격자 해상도를 한 표현으로 통일 | `50×192922×8 B≈77 MB`는 수용 가능하지만 이는 J 한 벌뿐이다. commit용 이중 버퍼, validity·variance, CPU↔GPU generation 업로드, 불규칙 rate 접근과 대역폭을 별도 계약해야 한다 | 기술적으로 가능하나 이번 개정에서는 보류 |
| 전역은 사다리 선택, line만 B(혼합) | `Γ`·대역 `J`·`χ`·`η`에 필요한 전역 비용과 line profile 해상도를 분리하며 기존 단일 소유권을 유지 | 두 view의 동일-generation 정합 검증이 필수 | **권고·채택** |

32 GB Ada에서 77 MB 자체는 용량 blocker가 아니다. C를 배제하는 이유는 용량 부족이
아니라, 실측상 이미 수렴한 전역 소비자 모두에게 fine-grid 저장·전송·대역폭 계약을
강제할 근거가 부족하기 때문이다. 향후 C로 전환하려면 J 단일 배열이 아니라 이중
buffer·validity·variance·GPU 업로드 bytes와 rate kernel bandwidth까지 실측한 별도
발주가 필요하다.

### 3.5 사전등록 음성대조

1. cache generation을 `J_ν`보다 하나 이전으로 바꾸면 조회와 commit 모두 FAIL.
2. line ID 또는 profile hash를 서로 바꾸면 `Q_g`/profile 결박 검사 FAIL.
3. `4π`, `V_s`, `Δt` 또는 공이동 frame 변환 하나를 누락한 fixture는 canonical
   projection closure FAIL.
4. `UNSAMPLED`를 0으로 바꾸거나 cache miss를 전역 성긴 `ΣφJΔν`로 fallback하면 FAIL.
5. `Q_g`에서 작은 `A_ul` 또는 작은 직전 estimator line을 제거하면 selection census FAIL.
6. 파생 cache에 독립 setter·독립 generation·정본보다 긴 lifecycle을 허용하면 owner
   checker FAIL.
7. profile 정규화를 1에서 벗어나게 하거나 segment 주파수를 observer frame으로 주입하면
   정합 게이트 FAIL.

---

## 4. 개정 3 — A2-02 사다리 재실행·재판정

### 4.1 종전 결과의 처분

커밋 `43ffe31`의 `BLOCKED` 결과는 폐기하거나 덮어쓰지 않는다. 이는 개정 필요성을 만든
원장 증거로 보존한다. 개정 후 실행은 새 schema·새 artifact 이름·새 회귀 대장 행을
사용하고 종전 결과를 `supersedes`가 아니라 `amends_after`로 참조한다.

### 4.2 재실행 순서

다음 순서를 바꿀 수 없다.

1. 원 line census와 100–20000 Å 계약을 hash 결박한다.
2. `BB_IN_DOMAIN` 원장과 전량 제외 목록을 만들고 개정 1의 음성대조를 통과한다.
3. 다른 여섯 소비자와 개정된 bound-bound 소비자의 주파수 합집합을 다시 만든다.
4. 바뀐 합집합 edge로 fine 진단 dump를 다시 만들고 보존 재빈·validity 검사를 통과한다.
5. 1000, 2000, 4000, 8000, 16000 전역 격자 사다리를 처음부터 다시 실행한다.
6. 같은 generation의 선택적 line estimator packet-effort 사다리와 §3.3 정합 게이트를
   production 자료형에 앞선 읽기 전용 offline segment replay로 실행한다.
7. 두 사다리와 모든 음성대조가 PASS한 뒤에만 A2-02를 PASS로 다시 서명한다.

합집합이 바뀌면 모든 log-bin edge가 바뀌므로 종전 8000→16000의 네 PASS 수치를 새
사다리 PASS로 복사할 수 없다.

### 4.3 전역 격자 사다리

전역 격자 선택 지표에서 종전 `Jbar`를 제거한다. 남는 네 물리 지표는 다음이다.

- 등록 대역별 `∫J_νdν`
- 매칭된 `Γ`
- `χ_ν` 대역 적분
- `η_ν` 대역 적분

각 지표는 기존 그대로 최대 상대변화 `≤1%`, 중앙값 `≤0.2%`여야 한다. 가장 작은
`N→2N` PASS의 `N`을 고른다. 8000→16000도 하나라도 실패하면 전역 격자 결정은
`BLOCKED`다. invalid-eligible record는 0건이어야 하며 missing·unsampled를 제외 또는
0으로 바꾸지 않는다.

### 4.4 `Jbar` 대체 지표: estimator effort 사다리

종전 전역 재빈 `Jbar(N_grid)→Jbar(2N_grid)` 대신
`\widehat{\bar J}(P)→\widehat{\bar J}(2P)`를 잰다. `P`는 임의 상수가 아니라 실행 전
manifest에 기록된 현재 production packet-effort이며, 고정 RNG prefix를 사용해
`P` 표본이 `2P` 표본의 진부분집합이 되게 한다. 첫 쌍이 실패하면 production effort의
두 배씩 늘리되, 실행 전에 resource ceiling과 마지막 쌍을 등록한다. 마지막 쌍도
실패하면 `BLOCKED`다.

이 사다리는 A2-03의 runtime class나 A2-06의 production rate 경로를 미리 요구하지
않는다. A2-01에서 확정한 rate graph와, 같은 packet run에서 전역 `J_ν`와 line
estimator를 함께 재생할 수 있는 read-only raw segment capture를 입력으로 하는 A2-02
offline oracle이다. capture에는 packet/segment ID, 셸, 공이동 frequency trajectory,
energy, path length, `V_s`, `Δt`와 generation 결박이 있어야 한다. 이 자료가 없으면
합성 자료만으로 production 수렴을 주장하지 않고 A2-02는 `BLOCKED_MISSING_SEGMENT_CAPTURE`로
남는다. A2-03은 이 offline으로 검증된 schema를 shadow하고, A2-06은 같은 수식을 production
경로에 이식한 뒤 동일 게이트를 다시 실행한다.

판정 cohort는 값을 보기 전에 고정한다.

- 종전 A2-02에서 유효했던 42 line-shell record를 삭제 없이 positive-control cohort로
  이관한다. 특히 s8 Fe II `l61→u1308`을 필수 고정 record로 둔다.
- 개정 1 뒤 해당 record가 도메인 밖이 되는 예외가 있으면 제외 원장에 이유를 남기고,
  결과를 본 뒤 대체하지 않는다. UV 선숲 coverage가 줄면 같은 deterministic 층화 규칙으로
  실행 전에 보충한다.
- active `Q_g`의 wavelength·ion·shell 층화 표본을 추가할 수 있으나 두 effort 사이에
  membership을 바꿀 수 없다.

각 유효 record의 상대변화는 기존 사다리와 같이 fine 쪽을 분모로 한다.

\[
\delta_i={|\widehat{\bar J}_i(P)-\widehat{\bar J}_i(2P)|
                 \over|\widehat{\bar J}_i(2P)|}.
\]

fine 값이 정확히 0이면 양쪽이 모두 `EXACT_ZERO`일 때만 `δ=0`이며, 한쪽만 0이거나
`UNSAMPLED`이면 FAIL이다. cohort 전체에서 최대 `δ≤1%`, 중앙값 `≤0.2%`,
invalid-eligible 0건을 요구한다. 각 값의 count·variance를 함께 기록하며, 동일 effort의
독립 RNG stream 재현도 같은 1%·0.2% 선을 통과해야 한다. 합격선은 바꾸지 않았다.

### 4.5 A2-02 최종 판정식

```text
A2-02 PASS =
  amended_union_PASS
  AND exclusion_census_PASS
  AND union_negative_controls_PASS
  AND global_grid_ladder_PASS
  AND selective_jbar_effort_ladder_PASS
  AND canonical_projection_closure_PASS
  AND fine_diagnostic_closure_PASS
  AND estimator_negative_controls_PASS
```

하나라도 `BLOCKED`, `NOT_RUN`, invalid-eligible 또는 음성대조 미실행이면 A2-02는 PASS가
아니다. A2-03 발주 보류도 계속된다.

### 4.6 사전등록 음성대조

1. 종전 delta-top-hat/전역 재빈 `Jbar` 결과를 새 estimator schema로 제출하면 FAIL.
2. 두 effort 사이에 cohort 또는 `Q_g` hash를 바꾸면 FAIL.
3. s8 Fe II `l61→u1308`을 cohort에서 제거하면 필수-record 검사 FAIL.
4. packet effort가 실제로 늘지 않았는데 label만 `P→2P`로 바꾸면 segment-ledger count와
   hash 검사가 FAIL.
5. 한 line의 frequency/profile ID를 섞은 fixture는 estimator convergence 또는 fine
   closure FAIL.
6. 마지막 effort 쌍이 1%·0.2%를 넘는데 중앙값만 보고 PASS시키면 FAIL.
7. 종전 합집합의 8000→16000 PASS 값을 새 edge 결과로 복사하면 input/edge hash 검사
   FAIL.

---

## 5. 개정 4 — 하류 프롬프트·구현 파급

### 5.1 단계별 변경표

| 단계 | 프롬프트에 반드시 추가할 계약 | 구현·게이트 파급 |
|---|---|---|
| A2-03 `RadiationField` shadow | amended A2-02 PASS의 선택된 전역 bin edge/hash를 입력으로 요구한다. `LineJbarCache`는 `RadiationField` 안의 generation-checked derived view이며 별도 owner가 아니라고 명시한다. | 전역 배열은 선택된 `N`으로 shadow한다. 192,922-bin fine dump를 런타임 배열 크기로 사용하지 않는다. cache metadata·validity·`Q_g` hash·원자적 commit schema를 shadow에 포함하고, 기존 결과 불변 게이트를 유지한다. |
| A2-04 producer commit | 전역 `J_ν`와 선택적 line accumulator가 같은 raw segment ledger에서 나오고 한 generation으로 함께 commit된다고 명시한다. | 둘 중 하나만 성공한 partial commit을 금지한다. pure-CMFGEN replay cache와 MC estimator provenance를 구분한다. |
| A2-05 CPU bound-free rate | 개정 합집합에서 선택된 전역 `J_ν`로 `Γ`를 직접 적분한다고 명시한다. line 제외 목록은 BF threshold·cross-section coverage를 줄이는 근거가 아니라고 명시한다. | 알고리즘은 종전과 같다. 새 edge에서 A2-02 `Γ` PASS를 재현하고, line cache 없이 작동해야 한다. |
| A2-06 CPU bound-bound rate | 아래 §5.2의 교체 문구를 사용한다. 전역 성긴 빈의 `ΣφJΔν`를 production `\bar J`로 계산하는 경로를 금지한다. | `Q_g` selective accumulator, count·variance·validity, generation-checked 조회를 구현하고 `R_lu=B_lu\widehat{\bar J}`, `R_ul^stim=B_ul\widehat{\bar J}`, `R_ul^sp=A_ul`을 계산한다. L-1bb와 estimator effort/정합 게이트를 모두 통과해야 한다. |
| A2-12 GPU lifecycle | line cache의 ID mapping, generation, validity와 값의 업로드·reset·동기화를 정본 `J_ν` lifecycle에 결박한다고 명시한다. | stale cache, CPU/GPU generation 차이, partial upload는 명시적 실패다. cache upload bytes를 기록한다. |
| A2-13 GPU rate | GPU bound-bound rate의 CPU oracle은 A2-06의 같은 cache view라고 명시한다. GPU에서 coarse `J_ν`를 다시 profile 적분하거나 fine-grid C를 몰래 도입하지 않는다. | GPU는 generation·line ID가 일치하는 cache를 소비해 CPU와 상·하향률을 비교한다. bound-free는 선택된 전역 grid를 계속 쓴다. memory footprint와 cache upload bandwidth를 보고하되 77 MB fine J를 전제하지 않는다. |

### 5.2 A2-06 문구의 규범적 교체

원 발주서 A2-06의 “`\bar J`, 상·하향률 직접 계산”과 §2.3의
“line `\bar J`는 정본으로부터 명시적으로 `\intφJdν` 계산”은 다음 문구로 교체한다.

> line `\bar J`의 물리적 정의는 `\bar J=∫φJ_νdν`로 유지한다. production MC에서는
> 전역 `J_ν` 빈을 재적분하지 않고, 같은 generation·raw path-length measure에서
> `φ` 가중 선택적 estimator `\widehat{\bar J}`를 직접 누적한다. 결과는 정본
> `RadiationField` generation에 원자적으로 결박된 읽기 전용 파생 cache로만 소비한다.
> active line의 cache miss·`UNSAMPLED`·profile mismatch에는 coarse-grid 적분, 0 또는
> 이전 generation fallback을 허용하지 않는다.

L-1bb의 비교 대상과 기존 과학 합격선은 바뀌지 않는다. `\bar J`, 상향률,
stimulated 하향률은 estimator 값을 사용하고 spontaneous 항은 계속 atomic `A_ul`을
사용한다.

### 5.3 단계 순서와 보류 조건

A2-03은 개정 A2-02 전체 PASS 전에는 무효다. A2-04가 원자적 dual-view commit을
제공하지 않으면 A2-06을 시작할 수 없고, A2-12가 cache lifecycle을 이행하지 않으면
A2-13을 시작할 수 없다. A2-05는 line estimator와 독립이지만 amended grid edge가
확정되어야 한다. 개정 A2-02의 estimator PASS는 runtime 구현을 선행한 것이 아니라
read-only segment replay로 수식·selection·수렴을 검증한 것이므로 이 순서에는 순환
의존성이 없다.

C안의 fine-grid 런타임 저장은 A2-03 또는 A2-13 구현자가 편의상 선택할 수 있는
fallback이 아니다. 별도 개정과 GPU 메모리·대역폭 실측 없이 도입하면 계약 위반이다.

### 5.4 사전등록 음성대조

1. A2-03 shadow가 amended A2-02 hash 대신 종전 union hash 또는 192,922 fine shape를
   정본 shape로 사용하면 FAIL.
2. A2-05가 line 제외 목록을 BF coverage 제외 근거로 재사용하면 `Γ` consumer census FAIL.
3. A2-06에 production `Σ_bφ_bJ_bΔν`, cache-miss coarse fallback 또는 독립 `jbar_line`
   owner가 남으면 static owner/read-trace 검사 FAIL.
4. A2-04에서 `J_ν`만 commit하고 line cache commit을 실패시킨 injection이 partial
   generation을 공개하면 FAIL.
5. A2-12/13에서 cache generation을 하나 늦추거나 line ID mapping을 shuffle하면 GPU
   rate 실행 전 FAIL.
6. GPU bound-bound가 cache 대신 fine/coarse `J_ν`를 독자 적분해 우회하면 read-trace FAIL.
7. A2-13에서 bound-free와 bound-bound 중 한쪽만 CPU oracle과 맞는데 단계 전체를
   PASS시키면 FAIL.

---

## 6. 필수 개정 산출물

후속 구현 발주는 최소한 다음 논리 산출물을 서로 다른 hash로 남겨야 한다. 경로명은
프롬프트에서 확정하되 종전 A2-02 산출물을 덮어쓰지 않는다.

1. amended 7-consumer frequency-union manifest
2. 전량 `BB_IN_DOMAIN` ledger와 전량 `BB_EXCLUDED_OUTSIDE_DOMAIN` ledger
3. 개정 합집합 fine diagnostic manifest
4. 전역 1000→16000 grid ladder 결과
5. selective `Jbar` packet-effort ladder 결과와 count·variance
6. canonical projection closure와 fine diagnostic closure 결과
7. 개정 1∼4 음성대조별 marker·종료코드
8. A2-02 amended 회귀 대장 한 행과 운전석 서명

각 결과는 source commit, 원 line-list hash, domain-contract hash, `Q_g` hash,
raw segment-ledger hash, 정본 generation을 해당되는 범위에서 결박한다.

---

## 7. 변경 금지선

이 개정은 다음을 허가하지 않는다.

- 1%·0.2% 합격선 완화
- `J_ν` 단일 정본 소유권 폐기
- 100–20000 Å 밖 line에 작은 `J` 또는 0을 주는 수치 floor
- `A_ul`·`gf`·population 기반 line pruning
- 20000–25000 Å 전역 검증 coverage 축소
- BF·opacity·emissivity·packet·formal·observer 소비 범위를 line 창에 맞춰 축소
- fine 진단 dump를 근거 없이 runtime canonical grid로 승격
- A2-02 `BLOCKED` 원장 또는 종전 결과 파일 덮어쓰기

---

## 8. 개정 인수조건

이 개정의 집행은 다음이 모두 충족될 때만 완료다.

- line 소비자 외연이 100–20000 Å line-center 계약과 등록 profile support로 재현된다.
- 원 line census가 in-domain과 excluded로 무손실 분할되고 제외 line의 rate 재진입이
  음성대조에서 차단된다.
- 전역 격자는 네 기존 지표로 다시 선택되고 각 지표의 1%·0.2% 선이 유지된다.
- selective estimator는 기존 유효 42레코드와 필수 Fe II record를 포함한 사전등록
  cohort에서 packet-effort 1%·0.2% 수렴을 통과한다.
- estimator와 canonical `J_ν`가 동일 measure·generation임을 projection/fine closure로
  증명한다.
- A2-03 이후 프롬프트가 혼합안을 사용하고 C안 또는 coarse fallback을 몰래 도입하지
  않는다.

이 조건 전에는 A2-02 상태가 계속 `BLOCKED`이며 A2-03은 발주 보류다.

---

## 9. 운전석 검수 항목

운전석은 다음을 독립 검수한다.

1. 이 문서 외 코드·덱·`src`·기존 검증 산출물이 이번 개정 저작에서 변경되지 않았는가.
2. 근거 commit이 정확히 `43ffe3186f926e887139228d465a8c63fa5c42a8`인가.
3. line 창의 규범값이 100–20000 Å이고 20000–25000 Å 전역 검증 대역은 유지되는가.
4. line 포함 기준에 `A_ul`, `gf`, population 또는 예상 기여 floor가 없는가.
5. profile support의 종류·폭·정규화·셸 의존성과 provenance가 manifest에 있는가.
6. 원 line census가 in-domain/excluded로 누락·중복 없이 분할되는가.
7. Ni I `l461→u462` 등 저주파 퇴화선이 이유·원행·hash와 함께 제외 목록에 있는가.
8. 제외 line의 `\bar J` 요청이 0/floor가 아니라 `OUT_OF_BB_DOMAIN`으로 실패하는가.
9. 다른 소비자의 동일 주파수 coverage가 제외 line ID를 재활성화하지 않는가.
10. 전역 `J_ν`는 amended union의 새 edge로 1000→16000 전 쌍을 재실행했는가.
11. grid ladder에서 `Jbar`가 제거되고 대역 `J`·`Γ`·`χ`·`η` 네 지표가 모두 남았는가.
12. 네 지표의 최대 1%·중앙값 0.2% 선이 그대로인가.
13. estimator ladder가 실제 packet effort `P→2P`와 고정 prefix를 사용했는가.
14. 종전 유효 42레코드와 s8 Fe II `l61→u1308`이 사전등록 cohort에 남았는가.
15. estimator 값마다 count·variance·validity가 있고 invalid-eligible이 0건인가.
16. canonical projection closure와 fine diagnostic closure가 각각 독립 PASS인가.
17. `J_ν`와 line cache의 raw-ledger·frame·normalization·generation hash가 같은가.
18. cache가 `RadiationField` 밖의 독립 owner·setter·lifecycle을 갖지 않는가.
19. cache miss·`UNSAMPLED`·profile mismatch에 coarse-grid 적분이나 이전 generation
    fallback이 없는가.
20. A2-03은 selected runtime `N`만 shadow하고 192,922-bin fine dump를 정본으로 쓰지
    않는가.
21. A2-05의 BF coverage가 line 제외 규칙 때문에 줄지 않았는가.
22. A2-06 프롬프트가 §5.2 문구로 교체되고 production `ΣφJΔν` 경로를 금지하는가.
23. A2-12/13이 cache ID·generation upload와 CPU oracle parity를 검사하는가.
24. GPU memory·upload bytes·bandwidth 보고가 실제 혼합안 배열을 대상으로 하는가.
25. 개정 1∼4의 모든 음성대조가 독립 marker와 기대 비0 종료를 남겼는가.
26. 종전 `BLOCKED` 결과는 보존되고 amended 회귀 대장에 새 행과 운전석 서명이 있는가.
