발주 L1-A-R1 구현을 완료했습니다.

핵심 결과:

- 선택 덱 대 실제 CMFGEN osc·PHOT·F_TO_S·col authority 비교 구현
- I12 하드코딩 PASS 제거, 실제 level/rank 및 line-bit 검증 결과 기록
- I19 current/legacy distance와 retention 계산값 적용
- sigma를 선택 덱 대 CMFGEN PHOT 평가로 교체
- P01–P10 및 C01–C12 개별 음성대조 구현
- 20개 metric 완결성 및 golden/runtime binding 적용
- 예상 `UNVERIFIABLE/PROVENANCE` 여섯 건 사전등록
- §11 실행 명령에서 `/usr/bin/time` 제거
- `src/`, 덱, CMFGEN 입력은 수정하지 않음
- commit·push·PR 및 실데이터 full-run은 수행하지 않음

최종 검증은 `py_compile` 통과, P01–P10과 C01–C12 모두 `EXPECTED_FAIL`, 세 엔진 음성대조 모두 `PASS`였습니다.

상세 구현 위치, 전체 변경 파일, 음성대조 출력과 운전석용 정확한 명령은 [CODEX_IMPL_L1A_REPAIR.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_IMPL_L1A_REPAIR.md)에 정리했습니다.