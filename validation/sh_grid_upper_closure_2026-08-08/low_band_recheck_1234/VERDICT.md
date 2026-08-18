# SH-GRID 신규 저주파 band 동일-snapshot 폐합

판정: **PASS**

이 시험은 미수렴 CMFGEN capture의 수렴도를 판정하지 않는다. 같은 고정
`EDDFACTOR/RVTJ/POP*`를 두 적분 경로가 소비할 때 새 178개 bin의 BF
광이온화율과 자발 Milne 방출률이 닫히는지만 판정한다.

- 대상: 707 levels x 90 depths
- sigma 재구성 max rel: `0.000000e+00`
- depth 합계 photo-rate max rel: `2.858299e-04`
- depth 합계 eta max rel: `2.854924e-05`
- significant ion-depth photo/eta max rel: `1.119969e-03` / `4.699186e-05`
- level-weighted L1 photo/eta max rel: `7.084012e-04` / `4.547908e-05`
- significant level-depth photo/eta max rel: `1.036785e+00` / `2.480808e-02`

마지막 level-depth 최대값은 진단치다. 임계면의 매우 얇은 한 빈에서
서로 따로 평균된 sigma와 J만으로 sub-bin 공분산을 복원할 수 없으므로,
수용 판정은 전역/ion 합과 기여도 가중 L1 폐합으로 한다.

`*PRRR`은 사용하지 않았다. 이 run에서는 일부 깊이에서 POP와 상태가
다르므로 동일-snapshot truth가 아니기 때문이다.
