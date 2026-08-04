완료했습니다. 최종 판독은 **시험된 정식화 좌표 기각**입니다.

- s8에서 `S_A/J_CMFGEN`은 UV 5대역에서 6.91–33.77배입니다.
- B2는 600–1000 Å에서 35.97배로 악화되고, 나머지 대역은 A와 동일합니다.
- thick90 BALL 14개 셸 모두 CMFGEN 쪽으로 이동하지 않았습니다.
- 1208.743248 Å s0에서 `eta_B2/eta_A = 22.4642`: η 급락이 아니라 증가입니다. “η 급감으로 트립” 가설은 반증됐습니다.
- 정확한 비국소 recurrence 원인, line/continuum 완전 분해, CMFGEN 범위 밖 s44–s49 비교는 `UNRESOLVED`로 유지했습니다.

산출물:

- [전체 E6 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E6.md)
- [산술 전용 재현 소비기](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e6_direct_fields.py)
- [300행 대역·셸 결과](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e6/band_shell.csv)
- [1208 Å 셸별 해부](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e6/trip_1208_shells.csv)
- [기계 판독 요약](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e6/summary.json)

재현 검증은 PASS했으며 CMFGEN 적분보존 최대 오차는 `2.22e-16`입니다. 신규 런·수송 solve·clamp·커밋은 수행하지 않았습니다.