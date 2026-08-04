실행 완료했습니다. 결론은 **③ 캡처 커버리지 PASS, ④ 부분 PASS·exact-double 폐합 보류**입니다.

| 지표 | Phase 1 | Phase 1.6 | parity59 최종 |
|---|---:|---:|---:|
| strict 비교 | 33/484 (6.82%) | 89/582 (15.29%) | **99/582 (17.01%)** |
| 수치 보유 | — | 411 (70.62%) | **421 (72.34%)** |
| 正결손 | — | 158 (27.15%) | **158 (27.15%)** |
| archive 유실 | — | 13 (2.23%) | **3 (0.52%)** |

주요 결과:

- 49.27GB·683,984,500행을 전체 로드하지 않고 스트리밍 선별했습니다.
- iter 11의 s0/s8/s43, 8개 이온만 추출: 285MB·4,103,907행.
- s49 sentinel 대조에서 7개 실재 이온 line-id 서명 완전 일치.
- Fe IV는 모든 셸 0행으로, 유실이 아닌 frozen topology 正결손입니다.
- raw-J̄ 유실 10행 전부 복구되어 strict 비교가 89→99행으로 증가했습니다.
- oracle 2회: 세 CSV 모두 byte-identical.
- 비교자 2회: 8개 산출물 모두 byte-identical.
- 소비값 재현:
  - raw-J̄ 로드: **14/14 exact**
  - β 재계산: **14/14**, 최대 상대오차 `2.223e-16`
  - 직접 mode-3 생산 J: **11/11 exact**
  - mode-0 C1 fallback 3건은 원 생산 per-line J가 미보존되어 독립 exact 비교 불가
- OFF object: 기본/명시 OFF `cmp=0`, oracle symbol 각각 0개.
- GPU와 git은 사용하지 않았습니다.

잔여 유실 3행은 parity59 디렉터리에 `lumina_ma_line_destruct.csv`가 없기 때문입니다. stdout에는 전역 terminals/destroyed만 있으며 셸 ownership과 체적 정규화는 복원할 수 없습니다. 또한 C1/C2/J̄ writer가 여전히 6자리이므로 원 생산 double의 bit-exact 재현과 독립 C2 생산 `R_bf` 대조는 증명할 수 없습니다.

상세 보고서: [CODEX_GATEB_PARITY59_AB_REPORT.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_GATEB_PARITY59_AB_REPORT.md)

핵심 증빙:

- [oracle_vs_cmfgen.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/parity59/oracle_vs_cmfgen.csv)
- [production_replay_consistency.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/parity59/production_replay_consistency.csv)
- [coverage_disposition.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/parity59/coverage_disposition.csv)
- [jbar_capture_manifest.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/parity59/jbar_capture_manifest.csv)
- [frozen_input_sha256.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/parity59/frozen_input_sha256.txt)