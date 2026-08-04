Gate B dual oracle Phase 1.6 구현과 산출물 생성을 완료했다.

주요 결과:

- CMFGEN `n_e` 단위를 RVTJ header, writer의 `WRITE ED`, `ED(:)`의
  `#/cm^3` 선언, 원문 identity 왕복으로 실증했다.
- 범위 밖 `s45`를 CMFGEN 범위 안의 최외곽 `s43`으로 교체했다.
- 마지막 rate 소비 iter 11을 C1/C2 생산 iter 10과 정렬하고 1-iteration
  lag를 fail-closed 계약으로 만들었다.
- Si 외 6이온은 원 raw-Jbar 결손과 생산 C1-fallback replay를 분리해
  계산 가능한 대표 J/bb/collisional 값을 수치화했다.
- thermal 등록 입력 4종은 parity50의 실제 소비 gate가 모두 OFF임을
  확인하고 NULL 등록을 정확히 복원했다.
- full-ion Jbar와 per-shell MA line-destruction rate를 위한
  `LUMINA_GATEB_ORACLE_CAPTURE` observer gate를 추가했다.
- 생산 `simul_r1` 경로에서 thermal ledger를 실측했다.
- PRRR exact-N_SL/ND, OUT NLEV/ND, GENCOOL 전-depth parser를 추가해
  Γ/α, `n_ion`, `b_k`, bf/ff/coll/net 커버리지를 확장했다.
- 모든 결손을 `coverage_disposition.csv`에 분류하고 미등록 사유를
  fail-closed한다. strict compared는 79→89, Lumina unavailable은
  241→171이다.
- 최종 oracle 두 독립 실행에서 s0/s8/s43 모두 byte-identical이었다.
- macro OFF object 두 형태가 byte-identical이고 oracle symbol은 0개다.
- Git 명령과 GPU는 사용하지 않았다.

산출물:

- [상세 구현·검수 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/GATE_B_PHASE1_6_CODEX_A_REPORT.md)
- [Frozen-cell 하네스](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c)
- [CMFGEN REPORT-ONLY 비교자](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py)
- [Phase 1.6 산출 디렉터리](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6)
- [s0 oracle](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6/lumina_oracle_cell_s0.csv)
- [s8 oracle](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6/lumina_oracle_cell_s8.csv)
- [s43 oracle](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6/lumina_oracle_cell_s43.csv)
- [CMFGEN 비교표](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6/oracle_vs_cmfgen.csv)
- [coverage 원장](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6/coverage_disposition.csv)
- [n_e 및 source 근거](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6/cmfgen_source_evidence.csv)
- [parser 왕복표](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6/cmfgen_parser_roundtrip.csv)
- [셸-depth 대응표](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_6/shell_cmfgen_depth_map.csv)
