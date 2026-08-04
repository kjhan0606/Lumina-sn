# A2-00 구현 보고 — CMFGEN 원장 자격

- 단계: `A2-00` 하나만
- 계약: 원장 자격(파일 집합·generation·판정 자격)
- 대상: `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/`
- 작성일: 2026-08-04
- 현재 상태: **구현 완료, grammar-debug 운전석 실행 대기**
- 물리 비교: 수행하지 않음
- `src/`·덱·대상 CMFGEN 디렉터리 변경: 0

## 1. 산출물과 변경 범위

신규 파일은 다음 네 개뿐이다.

1. `scripts/cmfgen_oracle_contract.py`
2. `scripts/a2_00_oracle_negative_controls.py`
3. `docs/A2_00_OPHYS_PROFILE.json`
4. `docs/CODEX_A2_00_ORACLE_ELIGIBILITY.md`

`cmfgen_oracle_contract.py`는 기존 `kshape_contract.py`와 같이 `write`와 `check` 두
모드를 제공한다. manifest는 원장 밖에만 쓸 수 있고, 심볼릭 링크를 따라가지 않는다.
mtime은 기록하지만 모든 판정에서 제외한다.

운전석 실행 뒤 생길 예정인 산출물은 다음이다. 현재 존재한다고 주장하지 않는다.

- `validation/a2_00_oracle/toy06_19.48d_jnu4.manifest.json`
- `validation/a2_00_oracle/negative_controls.txt`
- `validation/a2_00_oracle/write.stdout.txt`
- `validation/a2_00_oracle/check_snapshot.stdout.txt`
- `validation/a2_00_oracle/check_ophys.stdout.txt`

## 2. 로그인 노드에서 허용 범위 안의 재측정

대용량 파일 내용은 열거나 해시하지 않았다. 최상위 이름·종류·크기·mtime과 허용된
소형 텍스트(`*_INFO`, `run_jnu4.info`, `OUTGEN`, `MEANOPAC`)만 판독했다.

### 2.1 362개 전수 분류

세는 명령은 `find "$target" -mindepth 1 -maxdepth 1 -printf '%f\n'`의 결과를
manifest와 같은 이름 규칙으로 분류한 뒤 role을 `sort | uniq -c`한 것이다. 출력은:

```text
     72 oracle-data
    145 oracle-metadata
     28 run-log
    117 scratch
```

합계는 362이며 별도 `rg '^unclassified\t'` 출력은 없었다. 종류별 독립 계수는:

```text
362
      3 d
    233 f
    126 l
```

최종 권위 계수는 운전석의 `write` 출력이다. 기대 판정 지점은
`entries=362 unclassified=0`과 위 role count이다. 임의의 새 이름은 기본 role로
흡수하지 않고 `unclassified`가 되어 rc=15로 거부된다.

scope는 사용자가 제시한 362개, 즉 **최상위 immediate child**이다. 세 백업 디렉터리의
하위는 재귀하지 않으며 디렉터리 자체가 명시적 `scratch` 제외 항목으로 manifest에 남는다.

### 2.2 22:37/22:38 파일 특정

두 파일은 다음으로 특정됐다.

```text
gamma_feiii_coiii_formingshells.csv  1167 B     2026-07-18 22:38:01.678470541 +0900
jnu_918_1290_formingshells.csv       1983980 B  2026-07-18 22:37:46.369192338 +0900
```

둘 다 CMFGEN 원시 oracle 산출물이 아니라 후처리 진단이므로 `run-log`이다. 이 mtime은
generation 증거로 쓰지 않았다.

### 2.3 record schema 실측과 포맷 정정

실제 `_INFO` line 3/4에는 다음 여섯 필드만 있다.

```text
ND RECL WORD_SIZE UNIT_SIZE INT_SIZE LIT_END
```

즉 `_INFO` 자체는 record 수·단위·frame·iteration ID를 **선언하지 않는다**. 도구는
없는 선언을 만들어내지 않고 네 필드를 `NOT_DECLARED_BY_INFO`로 기록한다. record 수는
writer 수식과 실제 content header로 다음처럼 유도한다.

| 파일 | `_INFO` 선언 | record 수 유도 | 기대 바이트 | 실제 바이트 | 정적 결과 |
|---|---|---|---:|---:|---|
| `EDDFACTOR` | ND=90, RECL=728, WORD=8, UNIT=1, INT=4, little=T | `14 + OUTGEN NCF(196185) = 196199` | `196199×728 = 142832872` | 142832872 | 일치 |
| `JH_AT_CURRENT_TIME` | ND=90, RECL=1456, WORD=8, UNIT=1, INT=4, little=T | data record 3의 `ST_IREC=6, NCF, ND`; `6+1+196185=196192` | `196192×1456 = 285655552` | 285655552 | 일치 |

CMFGEN source 근거는 실제 존재를 확인한 다음 파일이다.

- `/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/eddfac_rec_defs_mod.f`
- `/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/open_rw_eddfactor.f`
- `/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_j_blank.f`
- `/gpfs/kjhan/cmfgen_src/cur_cmf/plane/out_jh.f`

`FINISH_REC`는 `EDDFACTOR` record 5의 float64 값이다. 로그인 노드에서는 대용량 파일을
열지 않았으므로 그 값은 **PENDING_DRIVER_EXECUTION**이다. `write`가 grammar-debug에서
정확히 `1.0`을 읽지 못하면 rc=14 FATAL이다. manifest에는 성공하더라도 다음 의미가
고정된다.

```text
status=FILE_COMPLETE
is_physical_convergence=false
semantics=file-complete-only-not-physical-convergence
```

## 3. 해시 대상과 제외 대상

모든 `oracle-data`, `oracle-metadata`, `run-log` 일반 파일은 파일 바이트 SHA-256 대상이다.
심볼릭 링크는 target을 따라가지 않고 link text 자체를 SHA-256한다. 각 entry에는 경로,
object type, role, 바이트 크기, SHA-256 또는 명시적 `null`, 제외 사유, 정보용 mtime이
들어간다.

제외는 `scratch`와 최상위 백업 디렉터리뿐이다. 현재 117개이며, 90개 `CSCRATCH####`,
`SCRTEMP`, `STEQ_VALS`, `BA_ASCI_*`, `BAMATPNT`, `POINT*`, `fort.*`, `J_COMP`, `JEW`,
`CFDAT_OUT`, `CUR_MODEL_DATA`, `CMFGEN_PID`, 세 백업 디렉터리 등이 manifest의
`hash_exclusions`에 항목별 사유와 함께 남는다. 제외된 경로도 inventory 존재와 role은
검사하므로 조용히 누락되지 않는다.

## 4. generation 판정

### 현재 판정

```text
UNDECIDABLE_WITH_CURRENT_EVIDENCE
```

근거를 부풀리지 않았다.

- `_INFO`에는 generation/iteration ID가 없다.
- `OUTGEN`은 great iteration 63, 64, 65, 66을 기록하지만 각 산출물의 마지막 writer
  iteration을 파일별로 결박하지 않는다.
- `RVTJ`/`POP*`의 content timestamp, `R/T/n_e/v` 공통 벡터, EDD/JH의 공통 주파수와
  `R²J` 관계는 대용량 content scan 전에는 확인할 수 없다.
- `OBS_FREQ`는 시작 때 만들어질 수 있고 `OBSFLUX`와 같은 격자를 가져도 그것만으로
  EDD/POP generation에 결박되지 않는다.
- mtime은 어느 방향으로도 증거가 아니다.

운전석 `write`는 내용으로 다음을 전수 검사해 manifest에 넣는다.

1. EDD/JH의 R·V byte equality
2. 196185개 전 주파수 byte equality
3. 90×196185개 전 `JH.RSQ_J` 대 `EDD.J×R²` 관계
4. `RVTJ`와 6개 `POP*`의 content completion token·ND
5. `RVTJ`와 27개 `*PRRR`, `GENCOOL`의 R·V·T·n_e를 각 파일의 선언 출력 정밀도 안에서 비교
6. `OBSFLUX`의 선언 주파수 벡터와 `OBS_FREQ`의 1열 비교

공통 content가 다르면 `MIXED_GENERATION_PROVEN`이다. 모두 맞아도 OBS 그룹을 한 great
iteration에 결박하는 content ID가 없으면 그대로 `UNDECIDABLE_WITH_CURRENT_EVIDENCE`다.
`SAME_GENERATION_PROVEN`은 O-PHYS attestation이 모든 대상 파일에 하나의 content-derived
iteration ID를 제공하고 위 독립 교차검사가 맞을 때만 가능하다.

## 5. 자격 4필드 재측정

운전석 대용량 실행 전에 확정 가능한 값과 대기 값을 분리한다.

| 필드 | 현재 값 | 한 줄 근거 |
|---|---|---|
| `CMFGEN_FILE_INTEGRITY` | `PENDING_DRIVER_EXECUTION` | `_INFO`/stat 크기식은 정확히 일치했지만 금지된 `EDDFACTOR` record 5와 SHA-256은 grammar-debug `write`가 확인해야 한다. 성공 manifest의 값은 `PASS`, 아니면 rc=14/비PASS다. |
| `CMFGEN_SNAPSHOT_REPLAY` | `PENDING_DRIVER_EXECUTION` | `FINISH_REC=1`과 전 hash가 확인돼야 `ELIGIBLE`; 이는 단일 파일 replay 자격이며 cross-file 물리 자격이 아니다. |
| `CMFGEN_NONLINEAR_CONVERGENCE` | `FAIL` | `OUTGEN:197,239,288`의 마지막 세 최대 population increase가 `7.97e5%, 3.69e5%, 3.52e5%`로 1%를 압도하며, line 144/186/228/277은 T fixed를 기록한다. |
| `CMFGEN_PHYSICAL_ORACLE` | `INELIGIBLE` | `FIX_T=T`, 비선형 FAIL, generation 미결, O-PHYS 필수 파일 7개 부재다. |

발주서 값을 복사하지 않고 실제 소형 파일에서 다음을 다시 확인했다.

- `run_jnu4.info:7`: `T [FIX_T]`
- `run_jnu4.info:28`: `POINT1: 62 62 1 -1000 F`
- `run_jnu4.info:26`: `4 [NUM_ITS]`
- `run_jnu4.info:27`: `T [DO_LAM_IT]`
- `OUTGEN:115,165,207,249`: iteration 63–66 네 번
- `OUTGEN:142,184,226,275`: 매번 `LAMBDA iteration used`
- `OUTGEN:155,197,239,288`: 최대 증가 7.50e6, 7.97e5, 3.69e5, 3.52e5%

`iteration 67 NaN 전 중단` 문구는 `run_jnu4.info:2,26`에 실제로 있고, `OUTGEN`이 66에서
끝나는 것도 확인됐다. 다만 현재 `OUTGEN` 자체에는 iteration 67 NaN record가 없다.
따라서 “67에서 NaN이 실제 재현됐다”까지 확대하지 않고 “그 전 중단 계획과 66 종료가
확인됨”으로만 기록한다.

## 6. O-PHYS 기계 profile과 현재 gap

`docs/A2_00_OPHYS_PROFILE.json`이 `check --profile ophys`의 요구명세다. 현재 부재 7개는:

```text
NETRATE
TOTRATE
CHI_DATA
CHI_DATA_INFO
ETA_DATA
ETA_DATA_INFO
LINEHEAT
```

메타데이터 조회로 7/7 `MISSING`을 재확인했다. 반대로 `*PRRR` 27개, 매칭 ion `*OUT`
27개, 여섯 `POP*`, EDD/JH/RVTJ/OBS/GENCOOL/OUTGEN의 존재를 확인했다.

profile은 파일 존재 외에도 `CMFGEN_ORACLE_ATTESTATION.json`을 요구한다. 여기에는
code revision, `IN_ITS`/`MODEL_SPEC`/`SN_HYDRO_DATA`/`VADAT` hash, atomic-data hash,
단위·frame·record schema, matching ion stems, 상·하향 rate 분리, 모든 freeze의 이름과
population/rate/opacity/emissivity 기여율, T solve 상태, NaN/Inf count, 마지막 세 반복
`Jν/T_e/ion` 변화, 최대 population 보정, 열수지 잔차, content generation proof가
들어가야 한다.

임계값은 §5.2 그대로 1%, 1%, `10^-3`이다. 현재 snapshot은 파일 7개와 attestation이
없으므로 rc=16으로 반드시 실패한다. `MEANOPAC`, `NEG_OPAC`, `GENCOOL`은 각각
`CHI_DATA`/`ETA_DATA`의 대용품으로 인정하지 않는다.

## 7. 주입 결함 대조

`a2_00_oracle_negative_controls.py`는 실제 원장을 복사하지 않는다. 작은 direct-access
EDD/JH와 텍스트 파일을 새 `/tmp/a2_00_oracle_controls_*`에 만들고, baseline manifest를
공유한 여섯 독립 사본에서 검사한다. 실제 scratch 경로는 첫 출력
`SCRATCH_COPY_ROOT=...`에 남는다.

현재 Codex sandbox의 SSH는 전역 config 권한 오류 뒤 socket 금지로 grammar-debug에
도달하지 못했다. 로그인 노드 실행 금지 규약을 우회하지 않았으므로 observed 결과는
정직하게 `PENDING_DRIVER_EXECUTION`이다.

| # | 주입 | 기대 rc | 기대 판정 marker | observed |
|---:|---|---:|---|---|
| 1 | `POPCAL` 삭제 | 11 | `MISSING_PATH POPCAL` | `PENDING_DRIVER_EXECUTION` |
| 2 | `EDDFACTOR` 끝 1024 B 절단 | 14 | `SIZE_MISMATCH`, `HASH_MISMATCH`, `RECORD_SCHEMA_FATAL` 세 검사가 각각 발화 | `PENDING_DRIVER_EXECUTION` |
| 3 | 동일 크기의 다른 synthetic run `EDDFACTOR`로 교체 | 13 | 크기는 통과, `HASH_MISMATCH EDDFACTOR` | `PENDING_DRIVER_EXECUTION` |
| 4 | `POPCAL` mtime만 +1일 | **0** | `MTIME_CHANGED_IGNORED` 뒤 전체 PASS | `PENDING_DRIVER_EXECUTION` |
| 5 | `_INFO`가 선언하는 ND(레코드당 depth 값 수)를 +1 | 14 | `RECORD_SCHEMA_FATAL`; 실제 포맷에는 별도 record-count 필드가 없다는 정정 포함 | `PENDING_DRIVER_EXECUTION` |
| 6 | `A2_UNKNOWN_PAYLOAD` 추가 | 15 | `UNCLASSIFIED_EXTRA` | `PENDING_DRIVER_EXECUTION` |

러너는 추가 positive control로 current-like fixture의 `--profile ophys` rc=16도 검사한다.
전체 성공 출력은 `SUMMARY controls_passed=7/7 failures=0`, 러너 rc=0이다.

## 8. 운전석 실행 명령

아래 명령은 모두 grammar-debug에서 실행되고 `/usr/bin/time`과 lageunha를 쓰지 않는다.
대상 CMFGEN 디렉터리에는 쓰지 않는다.

### 8.1 문법 검사와 소형 주입 대조

```bash
ssh grammar "ssh grammar-debug 'set -o pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; mkdir -p validation/a2_00_oracle; python3 -m py_compile scripts/cmfgen_oracle_contract.py scripts/a2_00_oracle_negative_controls.py; python3 scripts/a2_00_oracle_negative_controls.py | tee validation/a2_00_oracle/negative_controls.txt'"
```

- 기대 종료코드: 0
- 판정 지점: 여섯 `CONTROL ... PASS`, mtime row `observed_rc=0`, O-PHYS positive
  control `observed_rc=16`, 마지막 `controls_passed=7/7 failures=0`
- scratch 사본: 출력 첫 줄의 `/tmp/a2_00_oracle_controls_*`

### 8.2 실제 manifest 생성

```bash
ssh grammar "ssh grammar-debug 'set -o pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; mkdir -p validation/a2_00_oracle; python3 scripts/cmfgen_oracle_contract.py write /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 --manifest validation/a2_00_oracle/toy06_19.48d_jnu4.manifest.json --profile snapshot | tee validation/a2_00_oracle/write.stdout.txt'"
```

- 기대 종료코드: 0. record/schema/FINISH/hash 결함이면 기대와 달리 rc=14 등으로 실패하며
  그 실패를 숨기지 않는다.
- 판정 지점: `entries=362`, `unclassified=0`,
  `role_counts={"oracle-data":72,"oracle-metadata":145,"run-log":28,"scratch":117,"unclassified":0}`
- generation 판정 지점: 같은 줄의 `generation=...`와 manifest
  `generation_consistency.verdict`
- manifest hash 기록:

```bash
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; sha256sum validation/a2_00_oracle/toy06_19.48d_jnu4.manifest.json'"
```

### 8.3 unchanged snapshot 대조

```bash
ssh grammar "ssh grammar-debug 'set -o pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; python3 scripts/cmfgen_oracle_contract.py check /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 --manifest validation/a2_00_oracle/toy06_19.48d_jnu4.manifest.json --profile snapshot | tee validation/a2_00_oracle/check_snapshot.stdout.txt'"
```

- 기대 종료코드: 0
- 판정 지점: `PASS CMFGEN_ORACLE_CONTRACT entries=362 unclassified=0 ... mtime_used=false`

### 8.4 O-PHYS positive control

```bash
ssh grammar "ssh grammar-debug 'set -o pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; python3 scripts/cmfgen_oracle_contract.py check /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 --manifest validation/a2_00_oracle/toy06_19.48d_jnu4.manifest.json --profile ophys | tee validation/a2_00_oracle/check_ophys.stdout.txt'"
```

- 기대 종료코드: **16**
- 판정 지점: 일곱 `MISSING_REQUIRED_FILE:*`,
  `MISSING_ATTESTATION:CMFGEN_ORACLE_ATTESTATION.json`, 마지막
  `FAIL CMFGEN_ORACLE_CONTRACT exit_code=16`

## 9. 종료코드 계약

| rc | 의미 |
|---:|---|
| 0 | PASS |
| 10 | manifest/profile 구문 또는 예상 밖 I/O 오류 |
| 11 | 경로 삭제·추가·종류/role 변경 |
| 12 | 바이트 크기 불일치 |
| 13 | SHA-256 불일치 |
| 14 | direct-access record schema, 크기식, FINISH_REC FATAL |
| 15 | `unclassified` 존재 |
| 16 | O-PHYS 요구 미충족 |

여러 결함이 동시에 발화하면 fail-closed 우선순위는 15, 14, 16, 11, 12, 13이다. rc 하나로
다른 검사가 사라지지 않으며 모든 marker를 함께 출력한다.

## 10. §11 단계 회귀 대장 — A2-00

```text
stage_id: A2-00
contract: oracle eligibility
source_tree_hash: git_head=47bfa2001deba1154f0aea0808a04ab06428b443; implementation_bundle_sha256=ca2bdf76766c496abb9628a54902c40cafe8be4aa792c8f9acd561f12c4f3b0b
input_manifest_hash: PENDING_DRIVER_EXECUTION
oracle_id: /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/; manifest=validation/a2_00_oracle/toy06_19.48d_jnu4.manifest.json (PENDING_CREATION)
node: grammar-debug
command: docs/CODEX_A2_00_ORACLE_ELIGIBILITY.md §8.1-§8.4
exit_status: PENDING_DRIVER_EXECUTION
new_layer_status: IMPLEMENTED_AWAITING_DRIVER_EXECUTION
all_previous_layer_statuses: NOT_APPLICABLE_FIRST_STAGE
negative_control_status: PENDING_DRIVER_EXECUTION (expected 6/6 plus O-PHYS positive control)
coverage: preflight immediate children=362/362; unclassified=0; manifest content coverage=PENDING_DRIVER_EXECUTION
metric_values: record-size equations measured; SHA-256/FINISH/content-generation=PENDING_DRIVER_EXECUTION
changed_output_allowlist: scripts/cmfgen_oracle_contract.py; scripts/a2_00_oracle_negative_controls.py; docs/A2_00_OPHYS_PROFILE.json; docs/CODEX_A2_00_ORACLE_ELIGIBILITY.md
guard_hits: PENDING_DRIVER_EXECUTION
fallback_hits: PENDING_DRIVER_EXECUTION
rng_seed: NOT_APPLICABLE_NON_MC
mc_confidence: NOT_APPLICABLE_NON_MC
artifact_paths: scripts/cmfgen_oracle_contract.py; scripts/a2_00_oracle_negative_controls.py; docs/A2_00_OPHYS_PROFILE.json; docs/CODEX_A2_00_ORACLE_ELIGIBILITY.md; validation/a2_00_oracle/* (PENDING_DRIVER_EXECUTION)
driver_signoff: PENDING_DRIVER_EXECUTION
```

## 11. 남은 위험과 A2-01 인계

1. 운전석 hash/content scan 전에는 `CMFGEN_FILE_INTEGRITY`와 snapshot replay를 PASS로
   승격하지 않는다.
2. 현재 포맷은 모든 파일을 한 iteration에 묶는 내부 ID를 제공하지 않는다. O-PHYS
   attestation 또는 writer 측 content generation ID가 올 때까지 cross-file layer gate는
   열리지 않는다.
3. `_INFO`의 단위·frame·NCF·iteration ID 부재는 실제 schema 결손이다. source 추론은
   current replay 설명에는 쓰되 O-PHYS의 명시 선언을 대체하지 않는다.
4. 고정된 Si/S/Ca/Fe/Ni/Co stage는 `run_jnu4.info:15-24`에 이름이 있지만 기여율이 없다.
   현재 physical oracle은 계속 INELIGIBLE이다.
5. A2-01로 넘기는 것은 발주서의 157개 `Jν` 소유권 census뿐이다. A2-00에서 `src/`, 덱,
   runtime owner, 물리 비교를 선행 수정하지 않았다.
