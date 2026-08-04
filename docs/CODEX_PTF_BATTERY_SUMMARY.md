구현과 소급 분석을 완료했습니다.

- 배터리 CLI: [ptf_gated_metrics.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/ptf_gated_metrics.py:1)
- 전체 보고서: [CODEX_PTF_BATTERY.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_PTF_BATTERY.md:1)

핵심 판정:

- relT3 it46–50: 게이트 MAXCH `3.21e3–9.33e3%`, E1 5/5 PASS.
- probe1 it51–55: `1.15e5–4.53e5%`, E1 FAIL.
- E2 bulk 물리 지표는 유계.
- E3는 미충족: 주인이 d21–31로 이동·감쇠하더라도 종분율 `1e-15–1e-17` 수준이라 실물 주인으로 보기 어렵습니다.
- it41–45는 STEQ가 소실되어 물리 지표만 계산하고 게이트값은 정직하게 N/A 처리했습니다.

검증:

- STEQ 10개 block과 OUTGEN extrema/returned 전부 일치.
- CORRECTION_LINK 마지막 block 전수 대조.
- SCRTEMP↔RVTJ 최대 상대오차 `4.2e-8`.
- Markdown/JSON/CSV, STEQ 결측 경로, rundir 내부 출력 거부를 시험했습니다.

신규 런·소스/rundir 수정·커밋은 수행하지 않았습니다. 398961 착지 후 재실행 명령은 보고서 §6에 포함했습니다.