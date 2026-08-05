구현을 완료했습니다.

- [NE-NAMING checker](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/check_ne_naming.py)와 독립 5종 [음성대조](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_ne_naming_controls.py)
- 처분 A builder 사전 gate와 manifest 기록: [build_toy06_epoch.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_toy06_epoch.py:114)
- [DECK-FOSSIL checker](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/check_deck_fossil.py), 독립 5종 [음성대조](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_deck_fossil_controls.py), [atomic writer](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_generation_atomic.py)
- NE legacy seal과 [fossil quarantine 레코드](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/manifests/deck_fossil_toy06_19p48d_quarantine.json)
- 운전석 명령·기대 rc·대조표·폐합상태·위험: [인수 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_L0_NE_DECK_CHECKERS.md)

정적 검수 결과:

- 두 checker 사이 import 없음
- 등록 marker 10개와 rc가 계약 표와 일치
- `+56.41%` 근거 제거, 영향은 `UNQUANTIFIED_PENDING_CLEAN_ZBAR`
- 금지된 맞춤 상수는 코드에 없음
- 정본 4개 SHA-256·size·mtime 불변
- `git status --short -- data/` 출력 없음
- `git diff --check` 이상 없음
- `src/`, CMFGEN, 덱 파일은 수정하지 않음

로그인 노드 실행 금지에 따라 Python 음성대조와 회귀는 실행하지 않았습니다. 보고서의 grammar-debug 명령으로 각각 5/5와 지정 회귀를 확정해야 하며 현재 상태는 `PENDING_DRIVER_EXECUTION`입니다.