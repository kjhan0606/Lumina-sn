수리를 완료했습니다.

- [regression_ledger.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/regression_ledger.py:319): EDDFACTOR `FL`을 `10^15 Hz` 단위로 판독하고 `λ=2997.92458/FL` 적용
- 총 에너지는 참조처럼 전체 native grid 적분
- FUV `[918,1290] Å`, EUV `[450,918] Å` 및 대역비 산출
- `ledger_schema_version`: `1 → 2`
- 기존 대장: 69행, 32,594,156 bytes, SHA-256 불변
- 참조 구현, `src/`, 보호 문서는 수정하지 않음
- commit, 모델 런, GPU, 전량 재백필 없음

검증 게이트:

| 런 | `u_CMFGEN` | `u_mc` | `u_cs` | `u_mc/u_CMFGEN` |
|---|---:|---:|---:|---:|
| capture 188932 s0 | 694.7485728426198 | 1749.0679041201702 | 2675.6022754810087 | 2.517555231475637 |
| 07-15 gphall s0 | 694.7485728426198 | 400.2107555763917 | 463.65637047738164 | 0.5760512093445506 |

자기검사 결과:

```text
NEGATIVE CONTROL EDDFACTOR: FAIL (expected): FL treated as Hz -> ... EDDFACTOR has fewer than two samples in euv
PASS fixture EDDFACTOR: FL decoded as 10^15 Hz; energy and FUV/EUV metrics defined
PASS b_k dual weighting
PASS append-only
PASS missing-input fixture
PASS --self-test
```

두 실런의 `--no-append`도 성공했습니다. 속도 oracle 범위를 벗어난 s44–s49는 외삽 없이 `UNDEFINED`로 남아 두 metric은 44/50셸 `PARTIAL`입니다.

전체 보고서는 [CODEX_HARNESS_EDDFACTOR.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_HARNESS_EDDFACTOR.md)에 있습니다.

운전석 재백필 명령:

```bash
srun --ntasks=1 bash scripts/backfill_regression_ledger.sh
```