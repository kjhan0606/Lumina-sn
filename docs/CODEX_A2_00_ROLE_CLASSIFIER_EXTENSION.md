# A2-00 role 분류기 확장 보고

작성일: 2026-08-05. 변경 범위는 `scripts/cmfgen_oracle_contract.py`의 role
분류와 그에 따른 디렉터리 hash-exclusion 설명뿐이다. `src/`, CMFGEN 런 디렉터리,
negative-control 러너는 변경하지 않았다. 로그인 노드에서는 대용량 파일 내용을 열거나
hash하지 않았다.

## 11개 항목 판정

| 항목 | role | 실제 확인 근거 |
|---|---|---|
| `BAMAT` | `scratch` | stat상 2,397,604,608 B인 일반 파일이다. 본문은 열지 않았다. `INTENDED_DIFF_MANIFEST.txt` C절이 NF/NS 변경 때문에 재사용 불가능한 cold-start 잔여물로 `BAMAT*`를 `SCRTEMP`, `POINT*`, `CSCRATCH*`, `BA_ASCI*`와 함께 열거한다. jnu4 manifest의 `BAMATPNT`와 `BA_ASCI_N_D*`도 `scratch`다. |
| `INTENDED_DIFF_MANIFEST.txt` | `provenance` | base 런/validated stint-1 deck, 동일·변경 파일, 126개 symlink의 vintage 변경, `atomic_local` repair를 기록한 계보 manifest다. 일반 파일이므로 SHA-256 대상이다. |
| `MODEL_SPEC.base_reference` | `provenance` | 실제 내용은 base의 8개 dimension key와 27개 ion ISF 행을 보존한 1,173 B 조상 설정 스냅샷이며, modern `MODEL_SPEC` diff의 기준이다. 일반 파일이므로 SHA-256 대상이다. |
| `PHOT_PRESCAN.txt` | `run-log` | `phot_prescan.py`의 검증 출력이다. 28개 photoionization 파일별 NF/term/used/violation을 기록하고 마지막에 `0 VIOLATIONS`를 보고한다. |
| `PREFLIGHT.txt` | `run-log` | CMFGEN startup check를 offline 재현한 출력이다. 27개 ion이 전부 PASS, `sum NS=1792`, `sum NF=24542`, dangling link 0이다. |
| `PROVENANCE.txt` | `provenance` | base 런의 정체, 20개 ion의 19apr23 vintage 전환, `f_to_s` 선택 정책, cold-start 이유, S III/Co II local repair의 근거와 검증을 담은 13,221 B 계보 문서다. `run-log`가 아니며 SHA-256 대상이다. |
| `RUNTIME_ESTIMATE.txt` | `oracle-metadata` | base-run 시간 측정, modern/base 크기 scaling, 자원·반복시간 예상, 제출/monitor 명령과 first-iteration gate를 담은 실행 계획 문서다. 현재 run의 실행 로그는 아니다. |
| `SIGMA_REPAIR_CHECK.txt` | `run-log` | S III와 Co II repair 전후/이웃 level의 광이온화 단면을 수치 비교한 검증 출력이다. |
| `__pycache__` | `scratch` | 디렉터리이며 내부에는 20,426 B `phot_prescan.cpython-313.pyc` 하나가 있다. |
| `atomic_local` | `oracle-metadata` | 디렉터리이며 active symlink가 가리키는 run-local atomic input overlay다. 내부에는 수정된 `COB/II/.../phot_data_A`(1,499,567 B)와 `SUL/III/.../phot_data_A`(1,070,091 B)가 있다. immediate-child 계약이므로 디렉터리 존재/role만 봉인되고 하위 내용 hash는 이 manifest의 범위 밖이다. |
| `seq_logs` | `run-log` | 디렉터리이며 `modern1948_slurm-394403.out`, `modern1948_slurm-394502.out` 두 실행 로그(각 130 B)를 담는다. |

## 규칙 변경

- CMFGEN transient 규약을 `BAMAT(PNT)`, `BA_ASCI_N_D<depth|ND>`,
  `CSCRATCH<digits>`, `fort.<unit>` 패턴으로 묶었다. 특정 depth나 새 이름 목록을
  추가하지 않았다.
- 새 `provenance` role은 `PROVENANCE[...].{txt,md,json}`,
  `*_DIFF_MANIFEST.{txt,json,csv}`, ancestral `MODEL_SPEC.(base|baseline)[_reference]`
  규약에만 적용한다.
- 검증 로그는 제한된 `PHOT*_PRESCAN`, `SIGMA*_CHECK`, `PREFLIGHT*` 규약,
  계획 문서는 `RUNTIME*_ESTIMATE` 규약으로 분류한다. 임의의 `.txt`/`.json`을
  포괄하는 fallback은 없다.
- 디렉터리는 객체 종류까지 확인한다. Python cache는 `scratch`, `atomic_local` 계열은
  `oracle-metadata`, `seq`/`batch`/`run`/`slurm` log 디렉터리는 `run-log`다. 같은
  철자의 일반 파일은 이 디렉터리 규칙에 들어가지 않는다.
- 모든 규칙 밖의 이름은 계속 `unclassified/no-rule`이며 rc=15다. 예를 들어
  `A2_UNKNOWN_PAYLOAD`, `RANDOM_CHECK.txt`, `BAMAT.notes`, 일반 파일인 `atomic_local`은
  흡수되지 않는다.

## jnu4 불변 확인

기존 운전석 산출물
`validation/a2_00_oracle/toy06_19.48d_jnu4.manifest.json`과
`write.stdout.txt`에서 362개와 다음 네 count를 확인했다.

```text
oracle-data       72
oracle-metadata  145
run-log           28
scratch          117
```

새 규칙은 jnu4 항목의 role을 바꾸지 않는다. 새 manifest에는 `provenance: 0`과
`unclassified: 0`도 기록된다. 운전석은 grammar-debug에서 다음으로 재생성해 네 count가
정확히 같은지 확인한다.

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
python3 scripts/cmfgen_oracle_contract.py write \
  /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 \
  --manifest /tmp/toy06_19.48d_jnu4.role-extension.manifest.json \
  --profile snapshot
jq '.role_counts' /tmp/toy06_19.48d_jnu4.role-extension.manifest.json
```

기대 rc는 0이며 위 네 count, `provenance=0`, `unclassified=0`이어야 한다.

## modern 운전석 명령

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
python3 scripts/cmfgen_oracle_contract.py write \
  /gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern \
  --manifest validation/a2_00_oracle/toy06_19p48d_modern.manifest.json \
  --profile snapshot
```

기대 rc는 0, `unclassified=0`이다. manifest에서 `PROVENANCE.txt`를 포함한 세
`provenance` 일반 파일의 `sha256`은 non-null이어야 한다. `BAMAT`과
`__pycache__`/`atomic_local`/`seq_logs` 세 디렉터리는 명시적 hash exclusion으로 남는다.

## 음성 대조

`scripts/a2_00_oracle_negative_controls.py`는 수정하지 않았다. 기존 운전석 결과는
7/7 PASS였으며, 변경 후 grammar-debug 재실행의 기대값은 다음과 같다.

| 대조 | 기대 rc | 핵심 marker |
|---|---:|---|
| 파일 삭제 | 11 | `MISSING_PATH` |
| `EDDFACTOR` 1024 B 절단 | 14 | `RECORD_SCHEMA_FATAL` |
| 다른 run의 동일 크기 `EDDFACTOR` | 13 | `HASH_MISMATCH` |
| mtime만 변경 | 0 | `MTIME_CHANGED_IGNORED` |
| `_INFO` ND + 1 | 14 | `RECORD_SCHEMA_FATAL` |
| 새 `A2_UNKNOWN_PAYLOAD` 추가 | **15** | `UNCLASSIFIED_EXTRA` |
| current-like fixture에 `--profile ophys` | 16 | `MISSING_REQUIRED_FILE:NETRATE` |

```bash
python3 scripts/a2_00_oracle_negative_controls.py \
  | tee validation/a2_00_oracle/negative_controls.txt
```

러너 기대 rc는 0, 마지막 줄은 `SUMMARY controls_passed=7/7 failures=0`이다.
