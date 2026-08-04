R1 준비를 완료했습니다. GPU·모델·덱 생성·commit은 실행하지 않았고 기존 두 덱도 그대로입니다.

### 실재 확인

기존 구현을 실제 코드로 확인했습니다.

- `CMFGEN_LINKS` 파싱·상호배타·링크 강제: [expand_atomic_data_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:317)
- `levels.csv.configuration`·판본 매니페스트: [expand_atomic_data_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:700)
- 충돌강도 source manifest: [build_cmfgen_coldata_all.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_cmfgen_coldata_all.py:555)
- 재결합 source manifest: [build_ma_radrecomb_target.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/build_ma_radrecomb_target.py:111)
- R1 덱 driver: [deck_regen_r1_vintage_driver.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen_r1_vintage_driver.py:64)

새로 작성했습니다.

- [verify_deck_r1_vintage.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_deck_r1_vintage.py)
- [r1_vintage_fixture.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/r1_vintage_fixture.py)
- [sbatch_deck_r1_vintage.sh](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_deck_r1_vintage.sh)
- [CODEX_R1_VINTAGE.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_R1_VINTAGE.md)

### 판본 변화

기존 `_fullcov` 자동선택 → R1 링크 선택:

- Si IV: `19apr23 → 5dec96`
- S III–V: `19apr23 → 3oct00`
- Ca III–V: `19apr23 → 10apr99`
- Fe IV–VI: `19apr23 → 18oct00`
- Co II–VI: `19apr23 → 18oct00`
- Ni II–VI: `19apr23 → 18oct00`

동일 판본 7이온은 Si II·III·V, S II, Ca II, Fe II·III입니다.

### 검증기 게이트

- 게이트 1, 전 이온 coverage: [186행](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_deck_r1_vintage.py:186)
- 게이트 2, 비항등 이온 0: [201행](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_deck_r1_vintage.py:201)
- 게이트 3, 동일판본 `f_lu/A_ul/λ` 비트동일: [254행](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_deck_r1_vintage.py:254)
- 게이트 4, σ·Υ 확대 공간 및 sidecar 정합: [358행](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_deck_r1_vintage.py:358)

기존 `_fullcov` 음성 대조에서는 정확히 S V 1,255선, Co II 44,000선만 결손으로 잡혔고, 동일판본 7이온은 mismatch 0이었습니다.

### Fixture 실제 출력

```text
S II : 3/324 lev ... (19apr23; auto)
S V  : 3/216 lev ... (3oct00; links)
NEGATIVE absent-link fallback: S II selected auto/19apr23 PASS
POSITIVE linked vintage force: S V selected links/3oct00 PASS
NEGATIVE linked-no-latest-leak: ... PASS
FIXTURE VERDICT: PASS
```

`py_compile`, `bash -n`도 통과했습니다. 신규 `_links` 덱은 생성되지 않았습니다.

### 운전석 명령

```bash
sbatch --export=ALL,REPO_ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_deck_r1_vintage.sh
```

남은 UNRESOLVED는 신규 덱 생성과 생산 4-gate 결과, 실제 링크 판본의 Υ mapped 총행 수입니다. R2·R3도 범위 밖으로 남아 있습니다.