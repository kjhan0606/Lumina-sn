H-TRANSFORM과 GEN-GUARD 두 계약만 구현했습니다. commit·push 및 본 계산은 수행하지 않았습니다.

변경 파일 3개:

1. [verify_h_transform.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_h_transform.py:128) — 700-zone/90-depth 세 갈래 검증, 적분질량·선불투명도 proxy·X56Ni 일관성·상한 판정.
2. [build_toy06_epoch.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_toy06_epoch.py:73) — 출력 필수화와 정본·별칭·동일/기존 출력 사전 거부.
3. [CODEX_IMPL_L0_H_GENGUARD.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_IMPL_L0_H_GENGUARD.md:1) — 수치 대조, 음성대조 실물, 운전석 실행 명령, 미실행 항목과 위험.

독립 재산출값은 모두 기존 상한 이내였습니다.

- 최대 `|ΔX|`: `5.14565825215918e-6 ≤ 6e-6`
- `ΣX/A` 상대변화: `1.1999856001754239e-5 ≤ 1.3e-5`
- floor-only proxy: `1.6793177959328887e-9 ≤ 2e-9`
- floor+재규격화 proxy: `1.1999856002198328e-5 ≤ 1.3e-5`

검사 결과:

- Python 문법검사 통과
- exact-zero floor 위반 검출 통과
- off-sum 명시적 실패·별도 투영 대조 통과
- 출력 생략, 입력=출력, 정본, 정본 symlink, 기존 출력 모두 종료코드 2
- 정본 트리 해시 전후 동일
- `src/`와 `/gpfs` 파일은 건드리지 않음

H 정식 증거 JSON/CSV 생성과 GEN 성공 전체 생성은 운전석 실행 제약에 따라 수행하지 않았으며, 정확한 명령은 보고서 §5에 기재했습니다.