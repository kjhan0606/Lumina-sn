1단 구현을 완료했습니다.

- 전체 보고서: [CODEX_STAGE32_RUNG1.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE32_RUNG1.md)
- 미적용 패치: [stage32_rung1_readonly_lambda.patch](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/stage32_rung1_readonly_lambda.patch)
- 사전등록 변경집합: [stage32_rung1_expected_changes.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/stage32_rung1_expected_changes.txt)

핵심 결과:

- EPAY 이후 side-band disposition census로 출력 비폐기를 입증했습니다.
- `rate_shape_replaced` 9행과 에너지까지 payload에 보존되며 폐기 행은 0입니다.
- beta 결함 음성 대조에서 `KA-3.2.3 FAIL`을 확인했습니다.
- CPU 빌드와 model-free selftest는 PASS했습니다.
- clamp/floor/cap/fallback 및 2단 이후 기능은 없습니다.
- 패치는 작업 트리에 적용하지 않았고 commit도 만들지 않았습니다.
- GPU·실제 모델은 실행하지 않았으며, 필요한 한 줄 명령을 보고서에 기록했습니다.

패치 SHA-256: `db400a22907f32b126fa9007972de4be8dbf76ad3297b8ff5bee99c04bf65bb8`