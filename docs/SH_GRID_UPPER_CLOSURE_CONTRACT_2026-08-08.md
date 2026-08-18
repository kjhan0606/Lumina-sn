# SH-GRID 상한 폐합 계약 — 2026-08-08

## 재개방 증거

A210 trial-consistent ionization의 첫 실패는 양 endpoint 모두 shell 0의
Si V→Si VI ground edge였다. 이 edge는 `166.7674257419 eV`,
`4.0324184137410968e16 Hz`, `74.34557311275296 Å`이며 기존 BF/NLTE 상한
`3.0e16 Hz` 밖이다. 활성 abundance 원소의 모든 adjacent-stage ground edge를
전수 조사한 결과 이 상한을 넘는 항은 이 한 건뿐이다. level 전수 조사에서도
상한 밖 양의 threshold는 Si V level 0 한 건뿐이다.

이는 zero cross section 판정이 아니다. 기존 1178-bin bake는 threshold가 상한 밖인
row를 bake 전에 제외했기 때문에 runtime `has_cmfgen=0`이었고 Kramers fallback을
선택했다. 그러나 sealed `active_ions.csv`가 가리키는 CMFGEN phot source에는 이 row가
실재한다. 1234-bin fresh bake에서는 이 source를 정상 평가해 `has_cmfgen=1`로 등록해야
한다. threshold 자체가 grid 밖이라 적분 support가 없었던 종전 상태를 exact zero로
대입하지 않는다.

## 최소 구조 수리

기존 A2 frequency-union 정본은 BF limiting edge를
`4.032418413741097e16 Hz`로 이미 등록했다. 정식 canonical radiation field의 outward
rounded 상한은 `4.0362581455823112e16 Hz` (`74.27484744208523 Å`)다. 기존 로그 간격
`0.0052983173665480362`를 유지하며 BF/NLTE grid에 정확히 56개 bin을 위로 더하면:

- bin 수: `1178 → 1234`
- 하한: `5.8412785919616062e13 Hz` 불변
- 상한: `3.0e16 → 4.0362581455823112e16 Hz`
- `log(nu_max/nu_min)/1234`: 구 1000-bin 및 1178-bin dlog와 bit-identical
- BF 상한 canonical index: `K*N = 2*1234 = 2468 = J_HI`
- canonical absolute edges, 3866-bin 수, union/edge identity: 불변

따라서 ARTIS의 10 Å superbin을 새로 모사하거나 radiation field를 외삽하지 않는다.
이미 등록·생산되는 canonical 74.27 Å support에 BF/NLTE 소비 격자를 맞춘다.

## 구현·검증 조건

1. CMFGEN sigma 자산은 24,542 level×1234 bin을 원 source에서 전부 재평가한다.
   1178-bin row padding, tail fill, 보간은 금지한다.
   기존 상한 때문에 skip됐던 row의 `has_cmfgen: 0→1`은 threshold가 정확히 새 band에
   속하고 linked phot source가 평가된 경우에만 허용한다.
2. sealed deck의 기존 1178-bin exact-Hyd 자산은 loader-forbidden quarantine에
   recoverable하게 보존한 뒤 새 자산을 원자 교체한다.
3. 기존 baker가 runtime의 `log(max/min)/N` 대신
   `(log(max)-log(min))/N`을 사용해 old grid에서 3-ULP coordinate mismatch가 있었음이
   승격 prefix gate에서 발견됐다. 새 baker는 runtime 산식으로 통일한다. 따라서 기존
   1178-bin sigma 값의 byte 동일을 요구하지 않고, positive-support mask 완전 동일,
   셀 최대 상대변화 `≤1e-7`, 별도 동일-snapshot 물리 폐합을 요구한다. canonical edge
   전수 및 BF↔canonical roundtrip 자체는 기존 값과 bit-identical이어야 한다.
4. 새 high band의 finite/nonnegative sigma, threshold containment, rate/emissivity
   동일-snapshot closure를 별도 기록한다.
   stored sigma는 full-bin linear-frequency average이므로 threshold partial bin에서는
   `sigma_active=sigma_avg*Delta_nu_full/Delta_nu_active`로 적분 질량을 보존한 뒤
   `[nu_threshold,nu_hi]`에서만 canonical J와 결합한다. CPU/GPU rate owner가 같은
   sharp-edge 규칙을 써야 한다.
   photo-rate **정확도** gate는 상대오차 `<=2e-2`만 쓴다. 마지막 half-bin의
   Wien-tail처럼 native rate가 극미소일 때 `|Delta Gamma|*t_epoch<=1e-12`이면
   현재 snapshot에서 `EFFECT_NEGLIGIBLE`로 별도 분류할 수 있으나, 이는 정확도
   PASS를 대신하지 않는다. Milne 적분도 상대 `<=2e-2`를 그대로 요구한다.
5. CPU/OpenMP/CUDA 및 전체 gate battery 뒤 H200 endpoint flight에서 IONIZATION
   `POP_BF_OOG`가 사라져야 한다. 이후 나타나는 첫 실패를 별개 물리 문제로 취급한다.
