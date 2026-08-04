v4 산출물을 완료했습니다. 단, 격리 경로 지정 실수로 원본에서 CPU 빌드가 한 번 실행되어 기존 untracked `lumina` 바이너리를 덮어썼을 가능성이 있습니다. 임의 복구·삭제하지 않았으며 보고서에 명시했습니다.

- [v4 패치](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/stage32_rung1_readonly_lambda_v4.patch)
- [전체 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE32_RUNG1_V4.md)
- [-o 요약](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE32_RUNG1_V4_SUMMARY.md)

결과:

- v3 SHA-256: `b41e991c5ff947e488c85b678b96bfc1f5ba4f88d78146e3978ab262e1728c30`
- v4 SHA-256: `7cba853937394b28c7f4d2fc54a0bf1b4de14b9276c456f4a4101d1fe2a2a644`
- fixture 가중 중앙값: `0.9603999999999999`
- `1/(1-rho)`: `25.252525252525203`
- 필수 음성 대조 3건 모두 FAIL 확인
- 최종 격리 selftest 및 CPU build: PASS
- 패치 적용·commit·GPU·모델 런 없음
- 원본 패치 대상 소스들은 작업 전과 byte-identical 확인됨