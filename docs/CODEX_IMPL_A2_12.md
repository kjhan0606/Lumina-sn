# A2-12 구현 보고서 — GPU canonical mirror lifecycle (개정 11)

## 판정

구현·정적 census·CPU 빌드·CUDA 컴파일은 완료했다. 구현자 권한으로 GPU 실행은 하지
않았으므로 단계 판정은 `PASS`가 아니라 **`UNVERIFIED_GPU_NODE`**다. 운전석이 아래 SLURM
명령으로 실제 H200→H100 lifecycle positive/N1~N9를 실행해야 한다.

- 기준 HEAD: `068fb36dc182b00757f4b4e6a1a4cd56a1f25500`
- 명세 SHA-256: `cf202e27ebfe296f8c3dc1db2d1bf0a27463a0b03bca46ce23ca7864d283797a`
- A2-01 census: `rows=157 completed=20 unclassified=0`, rc=0
- A2-12 frozen CUDA census: `86+13+14+10+7=130`, 전 행 고유 처분, rc=0
- 물리 kernel rate/opacity/emissivity 산식 변경: 0
- 전체 gate battery: 사용자 지시에 따라 실행하지 않음

## 구현 범위

`src/gpu_radiation_field.cu`는 CPU `RadiationFieldView`와 `LineJbarView`를 한 candidate
transaction으로 할당·H2D 복사·D2H 전수 attestation·event synchronize한 뒤에만 `READY`로
게시한다. edge, field value/validity, line ID/Jbar/validity/count/SE, fixed-width metadata를
모두 포함한다. public generation은 성공 commit 한 번에만 갱신하며 실패 candidate는 전부
폐기한다. reset은 `gpu_committed_generation=0`과 `DIRTY`를 먼저 게시한다.

`src/gpu_radiation_field_contract.c`는 CUDA 비의존 상태·카운터·보존식을 소유한다. raw pointer
getter나 독립 generation setter는 제공하지 않는다. A2-13~15 이전 물리 launch는 공통
`gpu_rf_block_unmigrated()`로 각각 `GPU_RATE_NOT_MIGRATED`,
`GPU_OPACITY_NOT_MIGRATED`, `GPU_EMISSIVITY_NOT_MIGRATED` nonzero 상태가 된다.

## 130행 전수 처분

기계 판독 가능한 130개 고유 행은
`validation/a2_12/cuda_consumer_census.json`의 `C001`~`C130`에 source-line SHA-256과 함께
있다.

| build-authoritative TU | 행 | 처분 |
|---|---:|---|
| `src/lumina_cuda.cu` | 86 | scalar allocation/upload/free 제거 또는 no-op; legacy global-J/line/blue-wing은 producer-only로 분리; transport 2 launch는 A2-15 상태로 차단 |
| `src/lumina_bf_gemm.cu` | 13 | `d_T_rad/d_W` 독립 allocation/free 제거; compute/fine compute를 A2-14 상태로 차단 |
| `src/lumina_nlte_assemble.cu` | 14 | raw `d_J_nu/d_W` refresh 및 BB launch를 A2-13 상태로 차단; 수식 무수정 |
| `src/lumina_cmf_solve.cu` | 10 | CMF producer buffer로만 유지; OOM/CUDA 실패 nonzero, CPU 대체 금지 |
| `src/lumina_nlte_gemm.cu` | 7 | raw FP32 `nlte->J_nu` staging/GEMM을 A2-13 상태로 차단 |

신규 3군은 다음처럼 종결했다.

- blue-wing `d_jblue_line`: canonical cache가 아닌 GPU producer accumulator로만 분류.
- NLTE rate-GEMM `d_J_nu`: canonical mirror로 가장하지 않고 A2-13까지 H2D/GEMM 차단.
- CMF `d_J/d_Jnew`: CMF producer 전용. 실패 시 같은 attempt의 CPU solve/publish 금지.

archival/untracked `.cu` 다섯 개는 전수 발견했으나 `git ls-files '*.cu'`와 Makefile 입력이
아니므로 판정에서 제외했다: `backup_groupA_1422` 두 파일과 `impl_withParityAA/W/Y/orig`
세 파일. 신규 authoritative backend `src/gpu_radiation_field.cu`는
`DISCOVERED_OUTSIDE_CENSUS`가 아니라 A2-12 자체 mirror owner TU로 manifest에 별도 기록했다.

## 고정 원장 19 ID

누락·병합 없이 다음과 같이 종결했다.

- `GL01`, `GL02`: BF-GEMM scalar owner allocation 제거, A2-14 차단.
- `GL03`, `GL04`: BF-GEMM scalar free 제거, transactional mirror free로 대체.
- `GL05`: transport `d_T_rad` allocation 제거.
- `GL06`, `GL07`: lazy scalar check/allocation 제거.
- `GL08`: scalar free 제거, mirror는 READY stamp 선무효화 후 free.
- `GT01`, `GT02`: Planck sampling 산식 무수정, A2-15 차단.
- `GT03`: raw scalar pointer plumbing은 qualified descriptor가 아니며 launch 차단.
- `GT04`~`GT07`: 두 실제 transport launch를 같은 공통 A2-15 차단 gate가 지배.
- `GT08`, `GT09`: host packet-source tier 무수정, qualified GPU lane 차단.
- `GT10`, `GT11`: 동일 출력 진단식의 두 occurrence를 각각 허용 잔류로 계수.

## B2 — CMF GPU 실패 처분

`src/lumina_cmf_solve.cu`의 memory guard는 더 이상 “CPU fallback”을 출력하지 않는다. OOM은
`BLOCKED_GPU_FALLBACK_FORBIDDEN fallback_attempts=1 physical_launches=0`을 출력하고 nonzero를
반환한다. `src/lumina_cmfgen.c`는 rc를 받으면 cleanup으로 이동하고 CPU loop를 실행하지 않으며,
top-level `cmfgen_run()`은 nonzero를 반환하여 commit/publication 전에 끝난다.

`LUMINA_CMF_SOLVE_GPU=2` A/B도 GPU-first로 재배열했다. GPU가 성공한 뒤에만 동일 입력의 CPU
비교를 실행하므로 GPU 실패 attempt에 CPU solver가 실행·게시될 수 없다.

## 카운터와 root-cause 보존식

매 counter summary는 다음을 검사한다.

```text
sync_attempts = sync_commits + sync_failed_attempts
sync_failed_attempts = sum(sync_root_cause[status])
ready_checks = ready_passes + ready_failures
launch_attempts = physical_launches + blocked_launches
```

한 sync attempt는 root cause 하나만 증가시킨다. 구현 순서는
`GPU_CPU_CHANGED_DURING_UPLOAD`, stale CPU, stale line, CPU/GPU generation, shape/hash,
line ID, profile/Q-set, invalid cell, allocation, partial, first-copy, event, not-ready 순이다.
N5는 `partial_upload_failures=1`, `copy_failures=0`, `sync_failed_attempts=1`로 selftest가
검사한다. `tests/a2_12_contract_selftest.c`와 Z runner의 신규 10번째 행이 보존식을 CPU에서
검증했고 rc=0이었다.

## Z 독립 링크 배선

`src/gpu_radiation_field_contract.c`는 `scripts/run_gate_battery.py`의 `Z-validator`, `Z-tau`,
`Z-population`, `Z-canonical` 네 링크에 각각 정확히 한 번 추가했다. `Z-a2-12` binary,
`Run("Z", ...)`의 `--a2-12-contract`, `run_zinert_selftest.py`의 required CLI·존재성 검사·
definition도 함께 추가했다. Z-only serial 실행 결과는 **10/10 PASS**다. 전체 D/K/Z/CP
battery는 실행하지 않았다.

## 검증 결과

| 검증 | 결과 |
|---|---|
| `make lumina` | PASS |
| `make lumina_cuda` (CUDA 13.0.88, sm_80/86/90) | PASS |
| `make selftest_a2_12_gpu_lifecycle` | compile/link PASS, 실행 `UNVERIFIED` |
| `python3 scripts/a2_01_census_contract.py check` | PASS, 157행 |
| `python3 scripts/a2_12_static_census.py --write` | PASS, 130/130 |
| A2-03 byte parity fixture | PASS |
| A2-04 commit callgraph/replay synthetic | PASS/PASS |
| A2-05 BF rate fixture | PASS |
| A2-06 line-Jbar/dual-commit | PASS/PASS |
| A2-06 L1BB gate | rc=0, upstream `BLOCKED_MISSING_RATE_EXPORT` 보존 |
| A2-07 population fixture/census/gate/classic sweep | PASS |
| Z-only runner | PASS 10/10 |

기존 selftest inventory에서 다음 upstream 실패를 숨기지 않았다.

- `scripts/a2_03_callgraph_audit.py`: A2-05가 도입한 `radiation_field_read_view`를 여전히
  “consumer-like public API”로 금지하는 stale audit, rc=1.
- `scripts/cmf_linepop_roundtrip_selftest.py --skip-build`: `invalid eta_line`, rc=1.
- `scripts/run_ioniz_saha_selftest.sh`: wrapper rc=0이나 A/B/C/D 내부 lane 모두 rc=1
  (`abundances.csv` shell-column shape mismatch).
- 일부 Wave-3.2 raw binaries는 독립 runner용 인자 없이 직접 실행할 때 기존 nonzero를 냈다.

## 운전석 GPU 명령

Lifecycle/selftest는 정확히 다음 자원으로 제출한다.

```bash
mkdir -p validation/a2_12
sbatch --parsable --job-name=a2-12-life \
  --partition=h200,h100 --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=32G --gres=gpu:1 --time=01:00:00 \
  --output=validation/a2_12/a2_12_lifecycle-%j.out \
  --error=validation/a2_12/a2_12_lifecycle-%j.err \
  scripts/run_a2_12_gpu_lifecycle.slurm
```

필요한 경우 full-NLTE lifecycle-only integration은 80 GB 미만과 A40을 거부한다.

```bash
mkdir -p validation/a2_12
sbatch --parsable --job-name=a2-12-full \
  --partition=h200,h100 --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --mem=64G --gres=gpu:1 --time=06:00:00 \
  --output=validation/a2_12/a2_12_full_nlte-%j.out \
  --error=validation/a2_12/a2_12_full_nlte-%j.err \
  scripts/run_a2_12_full_nlte.slurm
```

GPU job이 종결되지 않았으므로 현재 `slurm_job_ids=[]`, GPU name/UUID/memory는
`UNVERIFIED`다. scheduler가 `CANCELLED/TIMEOUT/NODE_FAIL/OUT_OF_MEMORY`로 끝나면 명세
11.3의 `BLOCKED_GPU_UNAVAILABLE` 형식으로 기록해야 한다.

## 남은 위험과 A2-13~15 인계

- 실제 CUDA allocation/copy/event, non-default stream, amended-shape peak bytes, N1~N9 rc와
  marker는 운전석 GPU job 전까지 `UNVERIFIED`다.
- CUDA failure cleanup의 device leak 여부와 double-buffer peak는 sanitizer/profiler가 아니라
  실제 lifecycle job byte ledger로 최종 확인해야 한다.
- A2-13은 `lumina_nlte_gemm.cu`와 `lumina_nlte_assemble.cu`의 rate 소비를 opaque checked
  descriptor로 이관하고, 그때까지의 `GPU_RATE_NOT_MIGRATED`를 제거한다. rate 수식은 이번
  diff에서 바꾸지 않았다.
- A2-14는 BF opacity consumer를 이관하고 `GPU_OPACITY_NOT_MIGRATED`를 제거한다.
- A2-15는 transport Planck/re-emission raw scalar argument를 제거하고
  `GPU_EMISSIVITY_NOT_MIGRATED`를 제거한다. `GT01`~`GT09` 수식 소유권은 그대로 A2-15다.
- 운전석은 GPU evidence가 모두 PASS하기 전 `new_layer_status=PASS`로 바꾸면 안 된다.

## 전달 방식

원 `.git`은 sandbox에서 read-only여서 `index.lock` 생성이 실패했다. 따라서 별도 임시
repository에서 seal+구현 커밋을 만들고 bundle로 전달한다. 최종 bundle 경로, commit ID,
SHA-256은 handoff 응답에 기록한다. push는 수행하지 않는다.
