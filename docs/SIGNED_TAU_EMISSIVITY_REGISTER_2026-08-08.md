# Signed Sobolev τ emissivity 사전등록 — 2026-08-08

## Lumina 계약

A2-09는 signed τ를 clamp하지 않는다.

```text
beta(tau) = -expm1(-tau) / tau
beta(0)   = 1
eta_nu    = n_u A_ul h nu beta / (4 pi Delta_nu)
```

`tau=-a<0`이면 `beta=(exp(a)-1)/a`이므로 `a`가 커질수록 지수적으로 증폭된다. 이는
population inversion을 exact zero나 양의 τ로 바꾸지 않는다는 뜻이지, 무한 maser를
허용한다는 뜻이 아니다. `beta` 또는 `eta_nu`가 비유한 값이 되면
`EMISS_NONFINITE` → `[A2-09][BLOCKED] reason=EMISS_DIRECT_LINE_INVALID`로 private
publication 전체를 폐기한다. floor/clamp/fallback은 없다.

자가검사는 `tau=-0.25`에서 `(exp(0.25)-1)/0.25`를 직접 대조하고, `tau=-800`이
이름 있는 nonfinite 오류면으로 종료되는지 확인한다.

## CMFGEN 비교 사전등록

현재 확보한 CMFGEN 근거는 정상 line emission이 `A_ul*n_upper`에서 직접 생산되고,
reference formal transfer가 그 방출·흡수를 자체 주파수/공간 operator로 운반한다는
것까지다. 동일 inverted cell에서 CMFGEN이 signed Sobolev escape factor를 그대로
사용하는지, 별도 maser 처리/제외를 하는지는 아직 정본 증거가 없다. 따라서 지금
동등하다고 선언하지 않는다.

다음 CMFGEN 대조 전에 아래를 고정한다.

1. 동일 atomic model/population snapshot에서 `tau<0` line-shell 목록과
   `(Z, ion, lower, upper, tau, n_u, A_ul)`을 양쪽에서 덤프한다.
2. `tau` 구간을 `[-1e-6,0)`, `[-1e-2,-1e-6)`, `[-1,-1e-2)`, `<-1`로 나눠
   CMFGEN의 포함·제외·증폭 처리를 집계한다.
3. CMFGEN이 clamp/제외하면 Lumina에 조용히 복제하지 않는다. 물리 근거와 에너지
   장부를 다시 Fable에 제출해 별도 판정을 받는다.
4. 기대 결과는 사전 지정하지 않는다. 현재 Lumina의 필수 결과는 오직 signed 값을
   보존하고 비유한 증폭을 이름 있게 차단하는 것이다.
