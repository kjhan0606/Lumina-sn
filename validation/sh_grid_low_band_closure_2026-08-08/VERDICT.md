# SH-GRID 신규 저주파 band 동일-snapshot 폐합

판정: **FAIL**

이 시험은 미수렴 CMFGEN capture의 수렴도를 판정하지 않는다. 같은 고정
`EDDFACTOR/RVTJ/POP*`를 두 적분 경로가 소비할 때 새 178개 bin의 BF
광이온화율과 자발 Milne 방출률이 닫히는지만 판정한다.

- 대상: 707 levels x 90 depths
- sigma 재구성 max rel: `6.927909e+282`
- depth 합계 photo-rate max rel: `8.392513e-01`
- depth 합계 eta max rel: `7.832753e-01`
- significant ion-depth photo/eta max rel: `8.394547e-01` / `7.835351e-01`
- significant level-depth photo/eta max rel: `4.776335e+00` / `4.804526e+00`

`*PRRR`은 사용하지 않았다. 이 run에서는 일부 깊이에서 POP와 상태가
다르므로 동일-snapshot truth가 아니기 때문이다.
