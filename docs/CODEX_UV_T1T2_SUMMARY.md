완료했습니다. 전체 결과는 [CODEX_UV_T1T2.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_UV_T1T2.md:1)에 정리했습니다.

- **T1 RESOLVED:** 균일 R에서도 B0가 `8.29055→25.90856×CMFGEN`으로 악화했습니다. 사전등록 판독은 **결맞음/재주입 가정 자체가 문제**입니다.
- **T2 UNRESOLVED:** B2는 η만 교체했고 χ는 A와 bitwise 동일합니다. exact iter-10 하위준위 population과 선별 line-χ 분해가 없어 단일-인자 native χ+η 시험을 구성할 수 없었습니다.
- T1 에너지 폐합 오차 `0`, clamp/fallback `0`, stage31 3회 byte-identical, 모든 solver guard 통과.
- 생산 코드·모델/GPU·커밋은 건드리지 않았습니다.

재현 판독기는 [uv_t1t2_offline.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_t1t2_offline.py:1), 사전등록과 수치 산출물은 [validation/uv_t1t2](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/uv_t1t2)에 있습니다.