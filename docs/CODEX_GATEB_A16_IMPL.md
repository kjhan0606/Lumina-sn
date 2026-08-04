Phase 1.6 구현을 완료했습니다.

- C1/C2 `iter=10` → rate/J̄ 소비 `iter=11`로 정렬하고 1-iteration lag를 fail-closed 검증합니다.
- 비-Si 6이온은 원 raw-J̄ 결손과 생산 C1-fallback replay를 분리했습니다. 계산 가능한 bb/collisional 값은 수치화하고, 유실된 raw-J̄만 unavailable로 유지했습니다.
- parity50의 thermal 등록 4종은 실제 소비 gate가 OFF임을 확인해 NULL 상태를 정확히 복원했습니다.
- `LUMINA_GATEB_ORACLE_CAPTURE=1`로 향후 8이온 raw-J̄와 셸별 MA destruction heating을 덤프할 수 있습니다.
- strict comparison `79→89`, Lumina unavailable `241→171`, 수치 보유 행 `305→411`.
- B15 행번호를 `4421/4423/211`로 고쳤고, s43 snapshot 차이를 보고서에 명시했습니다.
- CPU 스모크 2회 모두 3개 CSV `cmp=0`. 비교자 산출 8종도 byte-identical입니다.
- OFF object도 byte-identical, oracle symbol 0개입니다.
- `git`과 GPU는 사용하지 않았습니다. GPU 금지에 따라 신규 CUDA capture observer의 실제 GPU 실행은 하지 않았습니다.

주요 산출물:

- [Phase 1.6 구현 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/GATE_B_PHASE1_6_CODEX_A_REPORT.md)
- [Phase 1.6 검증 디렉터리](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6)
- [Coverage disposition](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6/coverage_disposition.csv)
- [2회 스모크 해시](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6/smoke_sha256.txt)
- [B15 정정 기록](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_GATEB_B15_TEST.md)