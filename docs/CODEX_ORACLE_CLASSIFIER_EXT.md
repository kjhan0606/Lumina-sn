role 분류기 확장을 완료했습니다.

- [cmfgen_oracle_contract.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_oracle_contract.py:168)
  - `provenance` role 추가
  - CMFGEN transient·검증 문서·설정 sidecar·디렉터리 규약 기반 분류
  - 알 수 없는 이름은 계속 `unclassified`, rc=15
  - `PROVENANCE.txt` 등 일반 provenance 파일은 SHA-256 대상
- [검증 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_A2_00_ROLE_CLASSIFIER_EXTENSION.md:1)
  - 11개 role과 실제 확인 근거
  - jnu4의 기존 4개 count 불변 확인법
  - modern/jnu4 운전석 명령과 기대 rc
  - 음성 대조표

`src/`, negative-control 러너, `/gpfs`는 변경하지 않았고 대용량 파일을 열거나 해시하지 않았습니다. 실행 검증은 지시대로 grammar-debug 운전석용으로 남겼습니다.