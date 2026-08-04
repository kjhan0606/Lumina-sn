수리 완료했습니다.

- v2 패치: [stage32_rung1_readonly_lambda_v2.patch](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/stage32_rung1_readonly_lambda_v2.patch)
- SHA-256: `60bc65c172f95da5c10a07f37743b1f51b007ab9645b503aa0a7ff1d028655be`
- 전체 보고: [CODEX_STAGE32_RUNG1_REPAIR.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE32_RUNG1_REPAIR.md)

검증 결과:

- `git apply --check`: PASS
- CPU 빌드: PASS
- fixture/selftest: PASS
- `acc_w` 누락 주입: 체커 FAIL 확인
- opacity-share 및 얇은 선 분자 결함 주입: 체커 FAIL 확인
- v1과 사전등록은 변경하지 않음
- 패치 적용·커밋·GPU/모델 실행 없음

절차상 실제 작업공간의 ignored `lumina` 바이너리를 한 번 CPU rebuild한 사실과 production `rho_local` 계측 미실행 상태도 보고서에 명시했습니다.