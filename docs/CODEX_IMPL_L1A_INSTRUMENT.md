L1-A(v3) 계측기 구현을 완료했습니다.

- CLI/엔진: [l1a_instrument.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/l1a_instrument.py)
- 교차제약: C01–C12, 정확히 **12개**
- 음성 대조: 세 엔진 및 교차제약 12/12 PASS
- 1% fixture: JSONL 18행, exit 0
- 자원: wall 0.85초, peak RSS 62,160,896 bytes
- 구 덱과 `_ftos` 양방향 실행 확인
- `src/` 기존 변경은 그대로 보존
- full-run, commit, push, PR은 수행하지 않음

Golden 값과 명령, 남은 위험을 포함한 전체 보고서는 [CODEX_IMPL_L1A_INSTRUMENT.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_IMPL_L1A_INSTRUMENT.md)에 있습니다. Golden manifest는 [L1_GOLDEN_MANIFEST.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/L1_GOLDEN_MANIFEST.json)입니다.