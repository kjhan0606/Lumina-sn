# A2-10 one-sided Jbar 증명 설계 (2026-08-15)

상태: **설계만 완료, production 미구현**. k=8 전 셀 census 결과를 먼저 본다.

## 현재 대칭 envelope가 증명하는 것

현재 exact owner는 component마다

```text
|J_exact - J_numeric| <= e_absolute
```

를 증명하고 Gaussian profile에 `e_absolute`를 양의 가중합하여 line `Jbar`의 대칭
absolute bound를 만든다. 따라서 `Jbar_numeric-e_profile < 0`인 line 15/shell 10에서
이 자료만으로 양의 lower bound를 주장할 수 없다. Fable의 one-sided 제안은 현재
자료구조에 이미 들어 있는 정보가 아니며, 대칭 bound의 해석만 바꿔서는 구현할 수 없다.

## 가능한 별도 증명

exact transport는 양의 affine fixed-point 문제다.

```text
F(J) = b + K J
J_exact = b + K J_exact
K >= 0,  rho(K) < 1
```

기존 directed formal sweep에 `J=0`을 넣어 lower rounding으로 계산한
`b_lower = F_lower(0)`은 물리장을 수정하지 않는 별도 증명량이다. 양의 역연산자 때문에

```text
0 <= b_lower <= b <= J_exact
```

가 성립한다. Gaussian profile weight와 denominator도 양수이므로 outward-downward
가중 투영으로 `Jbar_lower`를 만들 수 있다. `chi>0`인 heating 후보는

```text
eta - chi*Jbar_exact < 0
```

를 직접 계산하지 않고도 다음 충분조건으로 인증할 수 있다.

```text
Jbar_lower > eta/chi
```

line 15/shell 10의 threshold는
`1.2946600103991379e-154`다. source-only lower가 이보다 클지는 실측 전에는 알 수
없으며, 0이면 이 경로는 닫히지 않는다. `chi<=0`에는 이 판정을 적용하지 않는다.

## 구현 경계

1. physical `J`, eta, chi, population, signed net을 변경하지 않는다.
2. `max(J-e,0)` 같은 사후 값 수리는 production J에 쓰지 않는다.
3. source-only lower는 별도 typed proof buffer와 provenance로만 발행한다.
4. lower formal sweep의 rounding 계약과 nonnegative recurrence가 실패하면 fail closed한다.
5. line profile 투영은 lower-directed sum/denominator로 독립 검증한다.
6. cooling sign에는 lower bound를 오용하지 않는다. cooling에는 Jbar upper 증명이 필요하다.

## 채택 전 시험

- small-grid direct reference에서 모든 component의
  `0 <= b_lower <= J_exact`를 확인한다.
- lower-bound bit corruption, 음수 recurrence, 잘못된 profile denominator를 각각
  fail-closed하는 음성대조를 둔다.
- k=8 census의 lower형 전 셀에 `Jbar_lower`와 `eta/chi`를 기록한다.
- lower형 전수에서 strict inequality가 성립할 때만 A2-10 heating sign에 사용한다.
- inequality가 성립하지 않는 셀은 기존 `UNRESOLVED_CANCELLATION`으로 남긴다.

이 방법은 diagonal jitter나 positivity floor가 아니다. exact transport의 양의 source가
이미 제공하는 물리적 하한을 별도 directed 계산으로 증명하는 것이다.
