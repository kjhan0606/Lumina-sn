# O-PHYS CMFGEN 물리 원장 런 실행 명세

상태: **발주 승인 완료 / 실행 전 명세**  
대상: `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys`  
운전 경계: 이 문서와 스크립트는 런을 준비할 뿐이다. `sbatch` 실행과 운전 판단은 운전석만 한다.

## 0. 인수 정의

이 런은 A-2의 L-1bb·L-3·L-4·L-5·L-6 및 L-0/2/7/8 물리 최종 판정에 쓰일 **물리 원장 후보**다. 다음 조건이 모두 참일 때만 인수한다.

1. 마지막 네 번의 독립 1-iteration capture가 모두 `FIX_T=F`인 full coupled iteration이며, 그 세 쌍의 변화량이 각각 `Jν ≤ 0.01`, `T_e ≤ 0.01`, 이온분율 `≤ 0.01`이다.
2. 마지막 active population 보정 `≤ 0.01`, 전 깊이 최대 정규화 열수지 잔차 `≤ 1e-3`, NaN/Inf 수가 0이다.
3. 이온·준위·전자밀도·온도에 숨은 freeze가 0이고, 입력 및 출력으로 이를 증명한다.
4. `NETRATE`, `TOTRATE`, `LINEHEAT`, `CHI_DATA`, `ETA_DATA`를 포함한 O-PHYS 필수 산출물이 같은 최종 세대에 묶인다.
5. `CMFGEN_ORACLE_ATTESTATION.json`과 외부 manifest가 생성되고 아래 명령이 `rc=0`을 반환한다.

```bash
python3 scripts/cmfgen_oracle_contract.py check \
  /gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys \
  --manifest /gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys.manifest.json \
  --profile ophys
```

현재 snapshot의 `rc=16`은 `docs/A2_00_OPHYS_PROFILE.json`이 요구하는 물리 산출물/attestation의 부재를 정확히 거부한 양성 대조다. `EDDFACTOR`의 `FINISH_REC=1`은 파일 완결만 뜻하며 물리 수렴을 뜻하지 않는다.

## 1. 결정 1 — `_modern`을 베이스로 채택

**결정:** `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern`의 iteration-40 restart를 부모로 사용한다. 원본은 읽기만 하며 준비 스크립트가 새 디렉터리만 만든다.

근거는 다음과 같다.

- 부모 `PROVENANCE.txt`는 27개 모델 이온 모두를 19apr23 의도에 맞춘 계보, 80개 심링크, 각 `f_to_s` 선택, S III와 Co II의 run-local `phot_data_A` 복구를 공개한다. canonical atomic tree는 고치지 않았다.
- A-2의 이 런은 Lumina 수송/솔버를 검증한다. 따라서 CMFGEN 쪽 원자자료도 Lumina와 정렬하여 원자자료 vintage 차이를 층 1 비교의 몫으로 남기는 것이 통제된 대조다.
- `MODEL_SPEC` 크기가 달라진 SV/Co2/CoIII를 포함해 `_modern` checkpoint는 바로 그 원자 구성으로 작성되어 warm restart와 차원이 맞는다.
- 부모는 OMP=16으로 fixed-T 40회를 정상 종료한 운전 실적과 완전한 provenance를 가진다. 준비 스크립트는 `SCRTEMP`, `POINT1/2`, `EDDFACTOR(_INFO)` 등 최소 호환 restart를 복사하고 BA/scratch 및 부모의 최종 물리 산출물은 복사하지 않는다.

**중요 반론 및 판단:** 부모의 40회는 “11 h 안팎의 완료 실적”이지 수렴 실적이 아니다. 부모 `OUTGEN` iteration 40에는 매우 큰 population 보정이 남아 있고 온도도 고정되어 있다. 후속 rel-T 시험에서 첫 full solve가 열수지/광도를 악화시킨 사례도 있다. 그러므로 `_modern`은 좋은 **계보와 초기조건**일 뿐 원장 자격을 승계하지 않는다. 이 위험 때문에 본 명세는 작은 full-linearization trust radius, 반복 capture, 실패 폐쇄형 연장을 사용한다. 수렴하지 않으면 베이스를 바꿔 합격시켜서는 안 되며 O-PHYS는 미인수다.

## 2. 결정 2 — iteration 41부터 온도 해제

**결정:** 별도의 새 fixed-T/Lambda 배치를 더 두지 않는다. 부모가 이미 40회 fixed-T preconditioning을 했으므로 O-PHYS의 첫 continuation부터 `VADAT: F [FIX_T]`, `F [FIX_T_AUTO]`, `IN_ITS: F [DO_LAM_IT]`로 full coupled 온도 방정식을 요청한다.

CMFGEN 구현 근거:

- `new_main/mod_subs/solve_for_pops.f:86-89`에서 `FIX_T`는 전 깊이의 온도 보정을 잠근다.
- `new_main/cmfgen_sub.f:603-611`의 Lambda iteration은 온도를 고정한다. 따라서 Lambda-only 단계는 최종 열수지 자격을 만들 수 없다.
- `new_main/subs/solve_for_pops.f:278-308`은 큰 보정 때 `LAM_VAL`/`NUM_LAM`에 따라 안전 Lambda step으로 후퇴했다가 full solve를 다시 시도한다. 즉 `DO_LAM_IT=F`는 첫 요청을 full solve로 만들되 내부 안전장치를 없애지 않는다.

추가 fixed-T 단계를 택하지 않은 이유는 이미 40회가 수행됐고, 기존 pure-Lambda 연장이 bounded stall을 보였으며 열수지를 풀 수 없기 때문이다. 대신 clone의 입력만 다음처럼 보수화한다.

- `MAX_LIN=1.01`: full linearization 보정을 1% trust radius로 제한한다.
- `MAX_LAM=1.10`: 안전 Lambda step은 10%로 제한한다.
- `MAX_dT=0.01`: `rd_control_variables.f:1009-1014`와 `solve_for_pops.f:137-141`의 별도 온도 보정 상한을 기본 20%에서 1%로 낮춘다.
- `NUM_LAM=2`: 부모와 기존 rel-T 입력의 안전 step cadence를 유지한다. 근거 없는 새 cadence를 도입하지 않는다.
- 기존 `LAM_VAL=400`은 유지한다. 이는 large-correction fallback 판정의 부모 계보를 보존한다.

위 값도 수렴 보장은 아니다. 특히 capture `OUTGEN`에 `Temperature held fixed at all depths.`가 한 번이라도 있거나 `RVTJ`가 `Was T fixed?: F`를 선언하지 않으면 packaging은 거부한다. 자동 fallback 상태에서 겉보기 population 수치만 좋아진 것을 합격으로 오인하지 않는다.

## 3. 결정 3 — 실제 소스 키와 산출 경로

추측한 키를 사용하지 않았다. `/gpfs/kjhan/cmfgen_src/cur_cmf`의 현재 소스에서 확인한 매핑은 다음과 같다.

| 산출물 | clone 입력 | writer 근거 | 판단 |
|---|---|---|---|
| `NETRATE`, `TOTRATE`, `EWDATA`, `LINEHEAT` | `VADAT: T [WRITE_RATES]` | `new_main/mod_subs/rd_control_variables.f:670-671`; open은 `new_main/cmfgen_sub.f:1312-1316`, write 블록은 1952, 2735, 3651, 3909 이후 | `LINEHEAT` 독립 키는 없다. `WRITE_RATES`가 묶어서 켠다. |
| `JH_AT_CURRENT_TIME(_INFO)` | `VADAT: T [WRITE_JH]` | `rd_control_variables.f:672-674` | SN 기본값도 T이나 provenance를 위해 명시한다. |
| `CHI_DATA(_INFO)`, `ETA_DATA(_INFO)` | `CMF_FLUX_PARAM: T [WR_ETA]` | `obs/rd_cmf_flux_controls.f:219`; `obs/cmf_flux_sub_v5.f:1832-1852` | 주 CMFGEN VADAT 키가 아니다. 수렴 capture 후 `cmf_flux.exe` formal 단계에서 생성한다. |
| `OBSFLUX`, `OBS_FREQ` | `cmf_flux.exe`, 기존 full-key control | 부모의 observer 산출 유지 | formal 출력으로 최종 세대에 다시 묶는다. |
| 불필요한 `FLUX_FILE` | `CMF_FLUX_PARAM: F [WR_FLUX]` | `WR_FLUX`는 `WR_ETA`와 별도 control | O-PHYS 필수품이 아니므로 끈다. |

`COMP_F=F`도 명시하여 최종 capture의 수렴된 `EDDFACTOR`를 formal 단계가 새로 계산해 세대 binding을 흐리지 않게 한다. `cmf_flux_sub_v5.f:1206-1209`의 `OPEN_RW_EDDFACTOR(..., COMPUTE_EDDFAC, ...)` 경로가 이 control을 사용한다. CHI/ETA 계산과 observer flux는 그대로 수행한다.

기존 `RVTJ`, `POP*`, `*OUT`, `*PRRR`, `GENCOOL`, `EDDFACTOR(_INFO)`은 유지한다. `CHI_DATA`를 `MEANOPAC`으로, `ETA_DATA`를 `GENCOOL`로 대체하지 않는다.

현재 실행 파일은 아래 둘이며 packaging이 실제 bytes와 CMFGEN git revision을 다시 해시한다.

- `exe/cmfgen_dev.exe`: 관측된 SHA-256 `f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1`
- `exe/cmf_flux.exe`: 관측된 SHA-256 `21c969be6f3c43094246feef105046f84215470ec2590e1049a02d76ba767a30`
- 판독 시 source revision: `dd1660af039e10f8f8c61d892c4e226e1cd6438c`; packaging 시점 값이 최종 원장이다.

## 4. 결정 4 — 반복, NG, 수렴 및 연장

### 4.1 주 solve

한 `solve` job은 warm restart에서 최대 80회를 허용한다. Slurm walltime은 `72:00:00`이다. 부모 fixed-T 40회가 약 11 h였지만 `docs/CMFGEN_BUILD_RUN_GUIDE.md` §5의 보수적 실측은 iteration당 40–60분이므로 80회 상단은 약 53 h다. free-T BA linearization, 안전 step, grammar 변동과 약 1.5 h의 기존 `cmf_flux` 실적을 더해 72 h를 잡았다. 이는 scheduling 상한이지 수렴 주장이 아니다.

NG는 부모 설정을 유지한다: `DO_NG=T`, `BEG_NG=5%`, `IBEG_NG=30`, `BW_NG=10`, `ITS/NG=20`, `CHK_NG=T`. 즉 초기 큰 보정에서 NG를 강제하지 않고 5% 아래의 후기 수렴 구간에서만 후보가 되며, `CHK_NG`가 나쁜 extrapolation을 거부한다. 첫 런 중간에 NG 설정을 바꾸지 않는다. 반복되는 유효성 거부가 명백하면 운전석이 별도 `DO_NG=F` branch를 승인할 수 있으나, 이는 새 VADAT hash와 새 capture/attestation을 요구한다.

### 4.2 마지막 세 수치의 측정

`capture` job은 같은 allocation 안에서 `NUM_ITS=1`, `DO_LAM_IT=F`를 네 번 연속 수행하고 각 결과를 `seq_logs/captures/capture_<job>_<1..4>/`에 보존한다. 세 변화량은 `(1→2, 2→3, 3→4)`로 계산한다.

- `Jν`: `JH_AT_CURRENT_TIME`의 공통 CMF frequency/depth grid 전부에서 `|new-old| / max(|new|, |old|, DBL_MIN)`의 최대값.
- `T_e`: `RVTJ`의 모든 depth에서 같은 대칭 상대차 최대값.
- 이온분율: 각 `POP*`로부터 원소별 이온 population을 합산/정규화한 뒤 모든 포함 이온과 depth의 같은 상대차 최대값.
- active population 보정: 각 capture의 CMFGEN nonlinear correction 보고에서 물리적으로 active인 population의 최대 absolute fractional correction. 단순 trace/terminal-level 제외가 필요하면 제외 규칙과 population/rate/opacity/emissivity 기여도를 rate-audit 파일에 전부 공개해야 하며, freeze로 바꾸면 안 된다.
- 열수지: 마지막 capture의 모든 depth에서 `|heating-cooling| / max(|heating|, |cooling|, DBL_MIN)` 최대값. `GENCOOL`/`LINEHEAT`와 CMFGEN 열수지 출력이 같은 값을 주는지 교차 확인한다.
- 텍스트와 읽은 부동소수 배열 전체에서 NaN/Inf를 센다.

분석자는 이 결과, 계산 코드/명령, 단위·frame 근거를 reviewed evidence JSON과 `rate_audit.evidence_files`에 남긴다. packaging은 수치 길이/유한성/임계값, 네 capture의 free-T 여부, 최종 capture와 root 파일의 SHA-256 동일성을 재검사한다. 사람이 임계값을 반올림해 내리지 않는다.

### 4.3 미달 처분

어느 하나라도 미달하면 `formal`과 `package`를 인수 경로로 진행하지 않는다. `OPHYS_MODE=solve`를 다시 제출하면 Slurm payload가 `IN_ITS`를 80회 continuation으로 명시적으로 되돌리고 현재 checkpoint에서 연장한다. 이후 새 4-capture와 새 evidence를 만든다. walltime 종료, 비정상 종료, NaN/Inf, 영구 Lambda fallback도 모두 **미인수**이며 이전 capture를 재사용하지 않는다.

수렴이 계속 실패하면 물리 입력, atom, source 또는 freeze를 조용히 바꾸지 않는다. 원인 분석과 새 발주 승인을 별도 branch로 요청한다.

## 5. 결정 5 — freeze 0 증명

입력 측 증명은 준비와 packaging 양쪽에서 fail-closed로 수행한다.

- 모든 이온/원소 `FIX_*` population control은 숫자 0이어야 한다.
- `FIX_NE=F`, `FIX_IMP=F`, `FIX_T=F`, `FIX_T_AUTO=F`, `TAU_SCL_T=0`이어야 한다.
- `XzV_IN`, `XzV_IN_*`, `POP*_IN` 외부 population/freeze vector가 없어야 한다.
- `FIX_BA`는 방정식 성분을 고정하는 freeze가 아니라 BA 행렬 재사용 임계값이므로 별도 공개하되 freeze count에 넣지 않는다.

출력 측에서는 최종 네 capture의 `OUTGEN`에 온도 고정 문구가 없어야 하고, `RVTJ`가 `Was T fixed?: F`여야 한다. `MODEL`/`OUTGEN`의 echoed control과 population correction/rate audit에서 고정된 species/level이 없음을 분석자가 다시 확인한다. attestation은 `freezes.undisclosed_count=0`, `components=[]`로 기록하며 이 자동/수동 증거를 함께 해시한다.

## 6. 결정 6 — attestation과 동일 세대 봉인

`scripts/package_cmfgen_ophys_attestation.py`는 수렴을 추정하지 않는다. 운전석이 검토한 evidence JSON을 받아 다음을 검증하고 패키징한다.

- CMFGEN source git revision, tracked worktree 변경 공개, 두 실행 파일 hash
- `IN_ITS`, `MODEL_SPEC`, `SN_HYDRO_DATA`, `VADAT` hash
- 모든 atomic symlink의 **해결된 file bytes** hash와 `atomic_local/` 전 파일 hash
- freeze-zero controls와 외부 vector 부재
- 네 free-T capture, 임계값, rate audit, 단위/frame 선언
- 마지막 capture와 root의 CMFGEN 산출물 content identity
- `cmf_flux` formal provenance와 CHI/ETA/OBS hash
- attestation 자기 자신을 제외한 run root 아래 **전 파일의 재귀 SHA-256**. 심링크는 link text도 별도 기록하고 hash는 실제 읽힌 target bytes에 대해 계산한다.

자기참조 파일인 attestation은 자신의 `file_sha256`에 넣을 수 없으므로, 그 파일 자체는 run 밖의 oracle manifest가 해시한다. Slurm stdout/stderr도 hash 도중 변하지 않도록 run 밖 `/gpfs/kjhan/cmfgen_runs/slurm_logs`에 둔다.

evidence 최소 골격은 다음과 같다. 문자열 `TODO/TBD/UNKNOWN/UNDECLARED/PLACEHOLDER`는 packaging이 거부한다. 단위는 분석자가 CMFGEN writer와 reader에서 확정한 실제 선언으로 채운다.

```json
{
  "schema": "lumina-cmfgen-ophys-evidence-v1",
  "iteration_id": "great-iteration-<FINAL>",
  "final_capture_dir": "seq_logs/captures/capture_<SLURM_JOB_ID>_4",
  "reviewer": {
    "name": "<reviewer>",
    "method": "<analysis script, revision, command and exclusions>"
  },
  "convergence": {
    "jnu_last3_max_fraction": [0.0, 0.0, 0.0],
    "te_last3_max_fraction": [0.0, 0.0, 0.0],
    "ion_last3_max_fraction": [0.0, 0.0, 0.0],
    "active_population_max_correction_fraction": 0.0,
    "max_normalized_heat_residual": 0.0,
    "nan_count": 0,
    "inf_count": 0
  },
  "rate_audit": {
    "upward_downward_separated": true,
    "evidence_files": ["seq_logs/OPHYS_RATE_AUDIT.json"]
  },
  "record_schemas": {
    "EDDFACTOR": {"units": "<source-derived exact units>", "frame": "comoving"},
    "JH_AT_CURRENT_TIME": {"units": "<source-derived exact scaled-moment units>", "frame": "comoving"},
    "CHI_DATA": {"units": "<source-derived exact opacity units>", "frame": "comoving"},
    "ETA_DATA": {"units": "<source-derived exact emissivity units>", "frame": "comoving"}
  }
}
```

0.0 예시는 실제 수치가 아니다. 측정하지 않은 값을 0으로 두면 허위 attestation이다. `reviewer.method`에는 네 snapshot reader, 상대차 정의, active 판정, 열수지 정규화와 원본 결과 파일을 식별해야 한다.

`scripts/cmfgen_oracle_contract.py`에는 실제 `WRITE_RATES`/`cmf_flux`가 추가로 만드는 `EWDATA`, `FLUX_FILE`, `CMF_FLUX_PARAM`, `CMF_FLUX_STDIN`, `OBSFRAME`, `OUT_FLUX`, `OUT_PARAMS`, `CMFFLUX_PID`, `HYDRO`의 엄격한 역할 분류를 추가했다. 또한 `write`가 명시한 `--profile-file`을 실제로 사용하게 하고, O-PHYS gap이 0일 때만 자격 4필드의 nonlinear을 `PASS`, physical oracle을 `ELIGIBLE`로 승격한다. 이 변경은 CMFGEN source나 deck을 수정하지 않으며, 알려지지 않은 이름은 계속 `rc=15`로 거부한다.

## 7. 준비 스크립트의 불변성과 결과

준비 파일은 `scripts/prepare_cmfgen_ophys_run.sh`다.

- target가 이미 존재하면 즉시 거부한다.
- `/gpfs/kjhan/cmfgen_runs/.toy06_19p48d_ophys.prepare.*` 임시 디렉터리에서만 조립한 뒤 원자적으로 rename한다.
- 부모의 입력/계보/두 local atomic repair와 최소 restart만 복사한다. 부모 output 또는 scratch를 수정/삭제하지 않는다.
- `setup_links.sh`의 두 run-local target만 새 run으로 재지정하고 80개 계보를 보존한다.
- clone VADAT/IN_ITS와 `CMF_FLUX_PARAM`만 이 명세대로 변경한다. CMFGEN source, canonical atom, 부모 deck은 불변이다.
- 스크립트는 `sbatch`, `srun`, CMFGEN 실행을 포함하지 않는다.

## 8. Slurm 자원 결정

`scripts/submit_cmfgen_ophys.slurm`의 고정 조건:

```text
--ntasks=1
--cpus-per-task=16
--mem=256G
--exclusive
--time=72:00:00
--exclude=grammar072,grammar078,grammar080
OMP_NUM_THREADS=16
OMP_DYNAMIC=FALSE
OMP_PROC_BIND=close
OMP_PLACES=cores
OMP_STACKSIZE=512M
```

OMP=16은 선택값이 아니다. 부모의 `comp_opac.f` OpenMP reduction에서 더 큰 thread 수의 합산 순서가 외곽의 작은 net opacity 부호를 뒤집어 LOGMON NaN을 만든 전력이 있고, 16만 clean 운전 실적이 있다. Slurm CPU와 `OMP_NUM_THREADS`가 모두 16이 아니면 payload가 실행 전에 거부한다.

## 9. 운전석용 복사 가능 절차

### 9.1 준비와 정적 확인 — 제출 없음

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
bash -n scripts/prepare_cmfgen_ophys_run.sh
bash -n scripts/submit_cmfgen_ophys.slurm
python3 -m py_compile scripts/package_cmfgen_ophys_attestation.py \
  scripts/cmfgen_oracle_contract.py

# 이 한 줄은 새 디렉터리 준비만 한다. 기존 run은 건드리지 않고 제출도 안 한다.
bash scripts/prepare_cmfgen_ophys_run.sh
```

### 9.2 free-T solve 제출/감시

```bash
mkdir -p /gpfs/kjhan/cmfgen_runs/slurm_logs
JOB_SOLVE=$(sbatch --parsable --export=ALL,OPHYS_MODE=solve \
  /gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/submit_cmfgen_ophys.slurm)
echo "$JOB_SOLVE"
squeue -j "$JOB_SOLVE" -o '%.18i %.9P %.24j %.8T %.10M %.6D %R'
tail -F /gpfs/kjhan/cmfgen_runs/slurm_logs/ophys-cmf-ophys-${JOB_SOLVE}.out
```

종료 후:

```bash
sacct -j "$JOB_SOLVE" --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocCPUS
rg -n 'NaN|Inf|Temperature held fixed|Maximum %|Current great iteration' \
  /gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/OUTGEN | tail -80
```

Slurm filename은 site의 `%x/%j` 확장 형식에 따라 확인한다. `State=COMPLETED, ExitCode=0:0`은 계산 종료일 뿐 수렴 합격이 아니다.

### 9.3 네 번 capture

```bash
JOB_CAPTURE=$(sbatch --parsable --dependency=afterok:${JOB_SOLVE} \
  --export=ALL,OPHYS_MODE=capture \
  /gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/submit_cmfgen_ophys.slurm)
echo "$JOB_CAPTURE"
sacct -j "$JOB_CAPTURE" --format=JobID,State,ExitCode,Elapsed,MaxRSS
ls -ld /gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/seq_logs/captures/capture_${JOB_CAPTURE}_{1,2,3,4}
```

네 archive에서 §4.2 수치를 산출·검수한다. 미달이면 `JOB_SOLVE=$(sbatch ... OPHYS_MODE=solve ...)`로 연장하고 새 capture를 만든다. 이전 job ID의 수치를 섞지 않는다.

### 9.4 formal CHI/ETA/OBS 생성

수렴 검토가 합격한 뒤에만:

```bash
JOB_FORMAL=$(sbatch --parsable --dependency=afterok:${JOB_CAPTURE} \
  --export=ALL,OPHYS_MODE=formal \
  /gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/submit_cmfgen_ophys.slurm)
echo "$JOB_FORMAL"
sacct -j "$JOB_FORMAL" --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

formal은 기존 CHI/ETA가 있으면 덮어쓰지 않고 거부한다. 이후 solve를 연장하면 payload가 이전 formal 산출을 `seq_logs/stale_formal_<job>/`로 이동시켜 stale 혼합을 막는다.

### 9.5 attestation, manifest, 최종 check

검토한 evidence를 `/gpfs/.../seq_logs/OPHYS_EVIDENCE.json`에 두고, 그 안의 `final_capture_dir`가 반드시 같은 `$JOB_CAPTURE`의 `_4`를 가리키게 한다. packaging과 전 파일 hash도 계산이므로 Slurm에서 수행한다.

```bash
EVID=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/seq_logs/OPHYS_EVIDENCE.json
JOB_PACKAGE=$(sbatch --parsable --dependency=afterok:${JOB_FORMAL} \
  --export=ALL,OPHYS_MODE=package,OPHYS_EVIDENCE="$EVID" \
  /gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/submit_cmfgen_ophys.slurm)
echo "$JOB_PACKAGE"
sacct -j "$JOB_PACKAGE" --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

`package` mode는 attestation 작성 후 동일 allocation에서 manifest `write --profile ophys`와 `check --profile ophys`를 연속 실행한다. 최종 인수는 package job `ExitCode=0:0` 및 로그의 다음 행으로만 선언한다.

```text
PASS CMFGEN_ORACLE_CONTRACT ... profile=ophys mtime_used=false
```

재수렴 후 기존 attestation을 교체해야 할 때만 새 capture/evidence를 확인한 뒤 `OPHYS_PACKAGE_FORCE=1`을 추가한다. 기존 attestation을 단지 check 통과용으로 덮어쓰는 것은 금지한다.

## 10. 금지 사항

- 로그인 노드에서 CMFGEN, `cmf_flux`, 수렴 분석, 전 파일 hashing을 실행하지 않는다.
- 부모 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern`, base run, CMFGEN source, canonical atom을 수정하지 않는다.
- freeze, source patch, atom vintage 변경, 미공개 level 제외로 수렴 숫자를 만든다거나 서로 다른 capture/formal 세대를 섞지 않는다.
- `--no-generation-scan`, mtime, `FINISH_REC`, Slurm `COMPLETED`를 물리 합격의 대체물로 쓰지 않는다.
- 이 발주 범위에서 commit/push하지 않는다.
