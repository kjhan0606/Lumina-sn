T5를 완료했습니다. 전체 보고서는 [CODEX_UV_T5.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_UV_T5.md:7)에 있습니다.

핵심 결과:

- 현재 행렬은 E12와 다른 세대입니다: `iteration=11`, 468,330 edges, SHA `08ff3312…`입니다. E12는 `iteration=10`, 473,045 edges, SHA `2b65dba6…`였습니다.
- rank-1의 B0/B1 상대오차는 두 판본 모두 12.3% 이하, B2→B0는 10.9%로 원문 §9.4의 15% 기준을 통과합니다.
- 따라서 원문 사전등록 기준으로는 **PASS → E10–E13 종결**입니다.
- 이번 과업문의 광학 변화량 기준으로는 상대오차 24.8–25.5%로 **FAIL → 노선 유지**입니다.
- 과업문의 광학 `−0.79%`는 E12 artifact의 실제 `−7.9017%`와 불일치합니다. 상충하는 기준 중 하나를 사후 선택할 수 없어 최종 무조건적 처분은 **UNRESOLVED**로 기록했습니다.
- 광학 수준 자체(`E12/E9`)는 오차 2.2% 이하로 rank-1과 사실상 구별되지 않습니다.
- SVD rank-1 에너지 비율은 대각 포함 59.80%, 제외 62.46%. B0/B1은 잘 재현하지만 B4에는 20%대 full-R 차이가 남습니다.

재현 도구는 [emiss_t5_rank1.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_t5_rank1.py:259), 기계 판정은 [verdict.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_t5/verdict.json)입니다. 세대 fail-closed, 엄격 `2e-13` 가드, 3회 byte-identical stage31, clamp/fallback/nonfinite 0을 검증했습니다. 생산 코드 수정·GPU/model run·커밋은 하지 않았습니다.