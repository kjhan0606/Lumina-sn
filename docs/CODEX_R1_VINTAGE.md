# CODEX R1 VINTAGE — 링크 정본 덱 준비 기록

작성일: 2026-08-03  
발주: `docs/ATOMIC_EQUIV_PLAN.md` R1

## 결론

R1 링크 정본 덱을 만드는 코드·읽기 전용 4-gate 검증기·Slurm 제출 스크립트와
소규모 fixture를 준비했다. GPU, 모델 런, 덱 생성, commit은 실행하지 않았다. 신규 덱
`data/tardis_reference_toy06_19p48d_sivcaiv_links/`는 아직 없으므로 생산 4-gate는 모두
**NOT RUN**이다. fixture만 실제 실행해 PASS를 확인했다.

기존 두 덱은 건드리지 않았다.

- `data/tardis_reference_toy06_19p48d_sivcaiv/`: directory mtime
  `2026-07-29 12:51:19.207328 +0900`
- `data/tardis_reference_toy06_19p48d_sivcaiv_fullcov/`: directory mtime
  `2026-08-03 15:33:06.827606 +0900`

## 실재 확인

직전 미완 보고를 파일과 코드로 다시 확인한 결과는 다음과 같다.

| 보고 항목 | 실제 상태 | 구현 위치 |
|---|---|---|
| `CMFGEN_LINKS` 입력·4종 링크 파싱 | 있음 | `expand_atomic_data_cmfgen.py:317`, `:354-395` |
| `CMFGEN_VINTAGE_MATCH` 상호배타 | 있음 | `expand_atomic_data_cmfgen.py:397-400` |
| 링크 이온 강제·미링크 자동선택 | 있음 | `expand_atomic_data_cmfgen.py:454-489` |
| osc/f_to_s/phot/col provenance | 있음 | `expand_atomic_data_cmfgen.py:579-590` |
| `levels.csv.configuration` writer | 있음 | `expand_atomic_data_cmfgen.py:700-708` |
| `atomic_vintage_manifest.csv` writer | 있음 | `expand_atomic_data_cmfgen.py:711-740`, `:1674-1676` |
| 충돌강도 builder의 source manifest | 있음 | `build_cmfgen_coldata_all.py:555-570`, `:663-693` |
| 재결합 target builder의 source manifest | 있음 | `build_ma_radrecomb_target.py:111-130`, `:243` |
| 신규 덱 driver | 있음 | `deck_regen_r1_vintage_driver.py:14`, `:64-85` |
| sbatch 초안 | 다른 이름으로 있음 | `sbatch_deck_regen_r1_vintage.sh` |
| 요청한 정확한 sbatch 이름 | 없었음 → 새로 작성 | `sbatch_deck_r1_vintage.sh` |
| R1 검증기 | 없었음 → 새로 작성 | `verify_deck_r1_vintage.py` |
| fixture | 없었음 → 새로 작성 | `r1_vintage_fixture.py` |
| 이 문서 | 없었음 → 새로 작성 | `docs/CODEX_R1_VINTAGE.md` |

기존 `fullcov` 덱은 새 writer가 들어가기 전에 생성됐으므로 실제
`atomic_vintage_manifest.csv`와 `levels.csv.configuration` 열은 없다. 둘은 신규 `_links`
덱 생성 시 처음 물질화된다. 보고의 “적용됨”은 생성 코드에 적용됐다는 뜻으로 확인했다.

이번 이어가기에서는 이미 있던 생성기·sidecar builder·driver를 다시 쓰지 않았다.

## 판본 선택 변화

아래의 “기존”은 `_sivcaiv_fullcov`를 만든 최신 날짜 자동선택이고, “R1”은
`atomic_links.txt` 정본 선택이다. R1에서 더 오래된 날짜로 돌아가는 것은 의도된 결과다.

| 이온 | 기존 자동선택 | R1 링크 선택 |
|---|---|---|
| Si IV | 19apr23 | 5dec96 |
| S III | 19apr23 | 3oct00 |
| S IV | 19apr23 | 3oct00 |
| S V | 19apr23 | 3oct00 |
| Ca III | 19apr23 | 10apr99 |
| Ca IV | 19apr23 | 10apr99 |
| Ca V | 19apr23 | 10apr99 |
| Fe IV | 19apr23 | 18oct00 |
| Fe V | 19apr23 | 18oct00 |
| Fe VI | 19apr23 | 18oct00 |
| Co II | 19apr23 | 18oct00 |
| Co III | 19apr23 | 18oct00 |
| Co IV | 19apr23 | 18oct00 |
| Co V | 19apr23 | 18oct00 |
| Co VI | 19apr23 | 18oct00 |
| Ni II | 19apr23 | 18oct00 |
| Ni III | 19apr23 | 18oct00 |
| Ni IV | 19apr23 | 18oct00 |
| Ni V | 19apr23 | 18oct00 |
| Ni VI | 19apr23 | 18oct00 |

같은 판본을 유지하는 7이온은 Si II, Si III, Si V, S II, Ca II, Fe II, Fe III이며 모두
19apr23이다. 게이트 3의 bit-identity 대상은 이 7이온이다.

## 읽기 전용 검증기

`scripts/verify_deck_r1_vintage.py`는 실패를 모아 1로, 입력·계약 오류를 2로 종료하며
덱에 쓰지 않는다.

| 게이트 | 구현 | 판정 |
|---|---|---|
| 1. 전 이온 CMFGEN coverage 1.000000000 | `:186-198` | `MODEL_SPEC` NF 안의 링크 osc 활성선이 신규 덱에 전부 존재해야 함 |
| 2. 매핑 비항등 이온 0 | `:201-251` | manifest의 네 source가 링크와 같고, NF 전 rank의 level number/E/g/configuration이 링크 osc와 항등이어야 함 |
| 3. 동일 판본 비트동일 | `:254-293` | 위 7이온의 line identity와 `f_lu`, `A_ul`, `wavelength_cm` float bits가 `_fullcov`와 같아야 함 |
| 4. σ·Υ 동반 확대 유지 | `:296-385` | sigma header/크기/flag와 collision manifest/binary header가 신규 전체 준위 공간에 맞고, addressable extent와 σ-present가 기존 덱보다 확대돼야 함 |

게이트 4의 Υ mapped 행 수는 출력하되 증가 조건으로 쓰지 않는다. 링크 정본의 오래된
`col_guess.dat`에는 tabulated Υ가 0인 경우가 많아서 행 수 감소 자체는 결함이 아니라
CMFGEN 선택의 결과일 수 있다. 대신 모든 이온의 `n_levels_ref`, 모든 `status=OK` binary의
Z/ion/nlevel/ntransition header, 매니페스트를 상호검증한다. 대체 Υ나 clamp는 넣지 않았다.

sbatch는 검증기를 마지막에 실행하며 `PIPESTATUS[0]`을 명시적으로 `exit`해 검증기의
종료코드가 job 결과가 되게 했다 (`scripts/sbatch_deck_r1_vintage.sh:53-60`).

## Fixture 자기검사

`scripts/r1_vintage_fixture.py`는 `/tmp` 아래 임시 원자 트리를 만들고 실제 CMFGEN S II,
S V 소스에 symlink한다. S II는 링크에서 의도적으로 빼고, S V는 최신 19apr23이 존재하는
상태에서 3oct00 네 입력을 명시 링크한다. 실행 명령과 실제 출력은 다음과 같다.

```text
$ python3 scripts/r1_vintage_fixture.py
FIXTURE parse: represented ions=[(16, 5)], kinds=osc/f_to_s/phot/col PASS
=== Phase 1: parse CMFGEN ions ===
  S  II  :    3/  324 lev,      3/  8527 trn, phot=Y col=Y  (19apr23; auto)
  S  V   :    3/  216 lev,      1/  3462 trn, phot=Y col=Y  (3oct00; links)
NEGATIVE absent-link fallback: S II selected auto/19apr23 PASS
POSITIVE linked vintage force: S V selected links/3oct00 PASS
NEGATIVE linked-no-latest-leak: S V latest=19apr23 but all linked inputs stay 3oct00 PASS
FIXTURE VERDICT: PASS
```

종료코드는 0이었다. 별도 음성 검사로 존재하지 않는 신규 덱을 검증기에 주었을 때
`ERROR: new deck absent: /tmp/r1_missing_deck`와 종료코드 2를 확인했다.

정적 검사는 다음이 모두 통과했다.

- `py_compile`: 생성기, 두 sidecar builder, R1 driver, 검증기, fixture
- `bash -n`: 새 `sbatch_deck_r1_vintage.sh`와 기존 이름의 sbatch 초안
- 기존 덱 mtime 재확인 및 `_links` 덱 부재 확인

검증기 로직의 읽기 전용 음성 대조로 기존 `_fullcov`를 게이트 1 입력에 넣자 정확히
S V `911/2166`(1,255선 부재)와 Co II `17986/61986`(44,000선 부재)만 FAIL했고, 나머지
25이온은 `1.000000000`이었다. 같은 덱끼리 게이트 3을 대조하면 동일판본 7이온 모두
`f/A/lambda bits PASS`, mismatch 0이었다. 이는 신규 덱의 생산 PASS가 아니라 알려진
R1 결손과 검증기 판별력을 재현한 음성 대조다.

## 운전석 제출 명령

```bash
sbatch --export=ALL,REPO_ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_deck_r1_vintage.sh
```

`REPO_ROOT`를 export하므로 Slurm이 스크립트를 `/var/spool/slurmd/scripts`로 복사해도 repo
탐색이 깨지지 않는다. 로그는 `/gpfs/kjhan/lumina_runner2/slurm/%x_%j.{out,err}`이다.

## 남은 UNRESOLVED

1. 신규 `_links` 덱 자체와 생산 4-gate 결과는 **NOT RUN**이다. 위 sbatch 완료 전에는
   R1을 PASS로 판정할 수 없다.
2. 실제 링크 판본에서의 Υ mapped 총행 수는 덱 생성 뒤 게이트 4 출력으로 처음 확정된다.
   이 값은 정본 충실성 진단값이지 튜닝 대상이 아니다.
3. R2(Co IV Υ 대용)와 R3(g/E/Υ 부착 문제)는 이 R1 발주의 범위 밖이며 그대로 남는다.

`src/`, `validation/regression_ledger/`, `scripts/regression_ledger.py`, 기존 두 덱에는 이
작업으로 변경을 가하지 않았다.
