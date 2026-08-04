완료했습니다.

핵심 재측정 결과:

| 항목 | 구 `_sivcaiv` | 신 `_sivcaiv_links` |
|---|---:|---:|
| `g` 불일치 | 211 | **0** |
| E 불일치 | 1,119 | **0** |
| 매핑 비항등 | 2이온 | **0이온** |
| 존재분 Υ 부재 | 9,860 | **19,962** |
| 존재분 σ 부재 | 147 | **148** |

- R3a: R1이 Co II 판본 차이 211건을 전부 흡수했습니다. 잔여 목록은 공집합이며 저준위 불일치도 0건입니다.
- R3b: 20,749준위 전수에서 임계 실패 0건. 최대 `|ΔE|=4.033×10⁻⁷ cm⁻¹`로 물리적으로 무해하여 추가 수리하지 않았습니다.
- 추가 덱/sbatch는 불필요하므로 만들지 않았습니다.
- Υ 증가는 신규 준위와 링크 판본의 희소한 collision table 때문이며 R3c로 남겼습니다.

산출물:

- [전체 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_R3A_GMISMATCH.md)
- [전수 감사기](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/audit_r3_level_equivalence.py)
- [fixture](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/r3_level_equivalence_fixture.py)

구 기준값 다섯 개를 정확히 재현했으며, 최종 전수 감사와 fixture, `py_compile` 모두 PASS했습니다. GPU·모델·덱 생성·commit은 실행하지 않았고 기존 덱과 `src/`, 회귀 대장은 건드리지 않았습니다.