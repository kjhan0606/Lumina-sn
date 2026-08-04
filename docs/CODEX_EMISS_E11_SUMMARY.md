구현을 완료했습니다. 전체 결과와 재현 명령은 [CODEX_EMISS_E11.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E11.md)에 정리했습니다.

핵심 결과:

- 수송 중 직접 누적하는 uncapped 형광 행렬 추정기 구현
- 전역 + 대표 셸군 3개 행렬, 총 device 상한 30.52 MiB
- `LUMINA_FLUOR_MATRIX_DUMP=<path>` gate 및 `LFMAT001-v1` little-endian sparse dump
- 내부 SHA-256 기반 `<path>.sha256` sidecar
- 전체·k-packet 에너지 장부와 4종 미분류 카운트
- `(2→0)` edge 7배 왜곡 fixture가 closure 1.2로 정확히 검출
- E10 소비자에 `auto|prefix|formal` 경로 추가
- 기존 prefix 결과 SHA-256 `e64a59…e79fd` byte-identical 유지
- formal identity fixture application/full-source closure 모두 0
- CUDA 강제 rebuild 성공
- 신규 clamp, normalization repair, fallback 모두 0

관련 파일:

- 추정기: [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4127)
- 판독기: [emiss_e11_fluor_matrix.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e11_fluor_matrix.py)
- 음성 대조: [emiss_e11_seeded_fixture.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e11_seeded_fixture.py)
- OFF capture 판독기: [emiss_e11_off_byte_check.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e11_off_byte_check.py)
- E10 소비자: [emiss_e10_apply_redistribution.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e10_apply_redistribution.py:71)
- rung 장부: [e11_expected_changes.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/e11_expected_changes.txt)

규율대로 신규 모델/GPU transport run과 commit은 수행하지 않았습니다. 오프라인 OFF oracle은 2,560 bytes가 byte-identical이었지만, pre-patch와 patched production binary의 실제 spectrum byte 대조는 보고서의 운전석 capture 명령으로 남겨 두었습니다. 따라서 물리 행렬 수치와 최종 production OFF 실증 상태는 명시적으로 `CAPTURE-PENDING`입니다.