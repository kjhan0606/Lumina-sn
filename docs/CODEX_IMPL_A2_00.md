A2-00만 구현했습니다. `src/`, 덱, CMFGEN 원장에는 쓰지 않았고 commit/push도 하지 않았습니다.

주요 산출물:

- [manifest 생성기·검증기](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_oracle_contract.py:388)
- [6종 결함 주입 러너](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/a2_00_oracle_negative_controls.py:159)
- [O-PHYS 기계 profile](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/A2_00_OPHYS_PROFILE.json)
- [전체 구현 보고서·운전석 명령·회귀 대장](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_A2_00_ORACLE_ELIGIBILITY.md:1)

핵심 판정:

- 362개 전수 분류: `oracle-data=72`, `oracle-metadata=145`, `run-log=28`, `scratch=117`, `unclassified=0`
- 22:37/22:38 파일: `jnu_918_1290_formingshells.csv`, `gamma_feiii_coiii_formingshells.csv`
- generation: `UNDECIDABLE_WITH_CURRENT_EVIDENCE`
- `CMFGEN_NONLINEAR_CONVERGENCE=FAIL`
- `CMFGEN_PHYSICAL_ORACLE=INELIGIBLE`
- 파일 무결성과 replay 자격은 대용량 hash·`FINISH_REC` 확인 전이므로 `PENDING_DRIVER_EXECUTION`
- O-PHYS 부재 7개를 재확인했으며 `check --profile ophys` 기대 rc는 16

실제 `_INFO`에는 record 수·단위·frame·iteration ID가 없다는 포맷 결손도 확인했습니다. 도구는 이를 추정값으로 위장하지 않고 명시적으로 기록합니다.

grammar-debug 실행은 Codex sandbox의 SSH 제한으로 수행하지 못했습니다. 따라서 음성 대조 observed 값과 §11 실행 필드는 날조하지 않고 `PENDING_DRIVER_EXECUTION`으로 두었으며, 복사 가능한 nested SSH 명령과 기대 종료코드는 [보고서 §8](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_A2_00_ORACLE_ELIGIBILITY.md:235)에 정리했습니다.