구현 및 CUDA 컴파일 검증을 완료했습니다.

- D6 배너를 미구현 상태가 드러나는 residual 문구로 정정.
- pure-CMFGEN에서도 `[FB-EDGE]`가 정확히 1회 출력되도록 공통 가드 적용.
- 기존 `[FORMAL-CONS]` 줄을 유지하고 `L_total_in`·`ratio_total` 줄 추가.
- 기준 deposition 합계: `7.787639096650e+42 erg/s`; `L_inj` 포함 총량 약 `1.08824e43 erg/s`.
- `make cuda`: exit 0.
- 기본 `make`: 기존 `M_PI`/C11 문제로 exit 2이며 범위 밖이라 수정하지 않음.
- `lumina_cuda.withParity*` 26개 해시 불변. GPU 실행·큐 제출 없음.

상세 보고서: [CODEX_TIER1_INSTRUMENT_BATCH.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_TIER1_INSTRUMENT_BATCH.md)