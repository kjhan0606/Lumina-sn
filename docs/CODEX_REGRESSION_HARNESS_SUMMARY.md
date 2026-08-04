구현 완료했습니다.

- 하니스: [regression_ledger.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/regression_ledger.py)
- 계산 노드 백필 런처: [backfill_regression_ledger.sh](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/backfill_regression_ledger.sh)
- 전체 보고서: [CODEX_REGRESSION_HARNESS.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_REGRESSION_HARNESS.md)
- append-only 대장: [ledger.jsonl](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/regression_ledger/ledger.jsonl)

검증 결과:

```text
NEGATIVE CONTROL: FAIL (expected): injected uv_fraction=1.5
PASS fixture metrics
PASS fixture census
PASS append-only
PASS missing-input fixture
PASS payload-only fixture
PASS --self-test
```

69개 런을 정확히 발견하는 것도 확인했습니다. 실제 백필과 5개 실런 census 대조는 로그인 노드 금지에 따라 실행하지 않았으며 대장은 현재 비어 있습니다. 계산 노드에서:

```bash
srun --ntasks=1 bash scripts/backfill_regression_ledger.sh
```

런처는 GPU를 숨기고, 로그인 노드·불완전한 allocation·69개 미만 inventory를 거부합니다. `src/`와 지정된 금지 파일은 이번 작업에서 수정하지 않았고 commit도 하지 않았습니다.