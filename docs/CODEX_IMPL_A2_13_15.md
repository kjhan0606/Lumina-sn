# A2-13~15 구현 보고 — GPU rate·opacity·emissivity 이관 계약

기준 HEAD: `2aaf6c7968494840a9ea66da8c48dc4e1639584d`

최종 상태: **`BLOCKED_PRODUCTION_NOT_MIGRATED`**  
marker: `A2_13_15_BLOCKED_PRODUCTION_NOT_MIGRATED`  
구현자는 GPU 실행을 하지 않았다. A2-12 fixture의 CPU-side commit 실패는 수리했지만
운전석 재실행과 production call-site 이관이 남았으므로 BF, BB, opacity, emissivity 중
어느 단계도 최종 PASS로 승격하지 않았다.

## 구현 범위

- `src/gpu_physics_contract.{c,h}`: 명세의 validity 9상태, 필수 counter 전부,
  fallback-zero 검사, A2-13 BF/BB 논리곱, A2-14/A2-15 독립 판정, N13/N14/N15의
  16개 marker/child-rc 정본을 구현했다.
- `gpu_radiation_field_device_view()`는 READY/generation/Q-set/profile 검사를 통과한
  mirror에서만 canonical field와 `LineJbarCache`의 read-only device descriptor를 낸다.
- `src/gpu_physics_kernels.cu`: canonical global edge/J를 쓰는 BF partial-bin 적분과
  line-ID checked cache lookup 기반 BB `Jbar/B_lu/B_ul/A_ul` kernel을 구현했다. BB API에는
  coarse/fine grid 인자가 없다.
- `src/gpu_opacity_kernels.cu`: signed ES/BB/BF/FF/total과 nonnegative event measure를
  별도 field로 계산한다. `abs`, zero clamp, floor가 없다.
- `src/gpu_emissivity_kernels.cu`: 5개 component 합, bin-width CDF, shell별 1 RNG 입력
  sampling을 구현했다.
- `tests/a2_13_gpu_oracle.cu`는 동일 mirror 입력에 대해 BF CPU A2-05 식, BB analytic
  oracle, signed opacity, component/CDF/RNG를 독립 대조한다. CUDA 13.0.2에서 sm_80/86/90
  compile/link PASS이며 GPU 실행은 운전석 몫이다.
- `tests/a2_13_15_contract_selftest.c`: BF만 PASS/BB FAIL과 그 역방향을 모두 전체
  FAIL로 판정하는 §5.4-7 음성대조를 실행한다. `EXACT_ZERO`만 수치 0으로 허용하고
  `UNSAMPLED`는 publish 없이 blocked counter로 귀결되는 것도 검사한다.
- 신규 C TU는 `run_gate_battery.py`의 Z-validator, Z-tau, Z-population,
  Z-canonical 네 hard-coded build에 직접 링크했고 별도 Z binary/runner row도 추가했다.
- `scripts/a2_13_15_static_census.py`는 원장 25행과 지정된 `.cu` 5개 전부를 읽고,
  원장 밖 raw hit를 파일/줄/token/text로 `validation/a2_13_15/static_census.json`에 남긴다.

중요: 기존 GPU production consumer에는 A2-12의 `GPU_*_NOT_MIGRATED` fail-closed guard가
남아 있다. 새 kernel은 `lumina_cuda`에 링크됐지만 `lumina_bf_gemm`, `lumina_nlte_assemble`,
transport call site가 아직 descriptor API로 교체되지 않았다. guard 제거는 금지했으며
physical production 이관 완료를 주장하지 않는다.

## 원장 25행 처분

| ID | 파일 | 처분 |
|---|---|---|
| G13-01~G13-08 | `src/lumina_cuda.cu` | A2-12 미폐합 동안 `GPU_RATE_NOT_MIGRATED`/`GPU_EMISSIVITY_NOT_MIGRATED` launch guard 뒤에 보존; production publish 0, GPU 검증 후 실이관 필요 |
| G14-01~G14-10 | `src/lumina_bf_gemm.cu` | `GPU_OPACITY_NOT_MIGRATED`가 coarse/fine 양 경로를 launch 전 차단; scalar owner/refresh는 미처분 잔여로 기록, PASS 금지 |
| G14-11~G14-13 | `src/lumina_nlte_assemble.cu` | `GPU_RATE_NOT_MIGRATED`가 refresh/pair launch 전 차단; dilute Planck/`d_W`/`T_rad[0]` 잔여로 기록, PASS 금지 |
| G15-01~G15-04 | `src/lumina_cuda.cu` | transport가 `GPU_EMISSIVITY_NOT_MIGRATED`로 launch 전 차단; Planck 재표본 잔여로 기록, PASS 금지 |

정적 census 결과는 원장 `25/25`, 지정+신규 CUDA TU `8/8`, 원장 밖 raw hit `340`, Z 직접
링크 occurrence `5`, rc `0`이다. raw hit는 삭제됐다고 세탁하지 않고 JSON에 전수 보존했다.
`src/lumina_cmf_solve.cu`도 목록에 포함했으며 누락 파일은 checker rc 2다.

## BF/BB 독립 CPU-oracle 대조 설계

BF lane은 A2-05와 같은 선택 전역 edge/hash, threshold partial-bin, 보간·합산 순서로
`Gamma`의 CPU/GPU 배열을 route×shell 단위 비교한다. truth-side 99.9% 활성집합에서
`f_cov>=0.95`, flow `E1<=0.10`, active `E_sym` P95 `<=0.25`, photoionization/stimulated/
spontaneous 합계 오차 각각 `<=0.10`을 적용한다. fine 진단 grid는 입력 manifest에 들어갈
수 없고 edge-shift/stim-off poison은 각각 rc 41/42여야 한다.

BB lane은 A2-06 `LineJbarCache`의 `(generation,shell_id,line_id,profile_id,q_set_hash)`
checked view만 입력으로 허용한다. coarse `J_nu` 재적분과 fine grid는 counter가 1이라도
rc 5다. `Jbar`, upward, stimulated downward 각각 `E1<=0.10`, active `E_sym` P95
`<=0.25`, truth flow coverage `>=0.95`, `A_ul` 상대오차 `<=1e-10`을 독립 계산한다.
line shuffle/stale/coarse/fine poison은 rc 43~46이다.

최종 A2-13 판정기는 BF PASS와 BB PASS의 논리곱이다. selftest에서 BF-only와 BB-only를
각각 `A2_13_NEG_HALF_ORACLE_FAIL`(rc 47/48 상당)로 거부했다. GPU 실행이 없으므로 현재
두 lane은 모두 `GPU_RUNNER_COMPILED_NOT_RUN`이며, production 단계는
`BLOCKED_PRODUCTION_NOT_MIGRATED`다. 한쪽 결과로 다른 쪽을 덮지 않았다.

## 빌드·selftest·CPU 불변

- `python3 scripts/a2_01_census_contract.py check`: PASS, rows=157.
- `make lumina`: PASS(기존 warning 보존).
- `module load cuda && make lumina_cuda`: PASS, CUDA 13.0.2, sm_80/sm_86/sm_90.
- 모든 Makefile `selftest*` target 28개 build PASS. CUDA target
  `selftest_a2_12_gpu_lifecycle`, `selftest_nlte_assemble`,
  `selftest_a2_13_gpu_oracle`은 실행하지 않았다.
- 무인자 직접 실행이 유효한 CPU fixture 중 계약 selftest와 A2-03/05/06/07/12 등은
  PASS했다. 인자/전용 runner가 필요한 fixture를 무인자로 실행한 결과는 검증 증거로
  쓰지 않았다. 요청에 따라 full battery는 실행하지 않았다.
- 기존 CPU production TU diff: 0줄. 상세 명령은 `validation/a2_13_15/cpu_invariance.json`.

## 운전석 SLURM 명령과 자원

runner는 반드시 `repo_root="$SLURM_SUBMIT_DIR"`로 시작하고 산출 디렉터리를
`/gpfs/$USER/lumina/a2_13_15/$SLURM_JOB_ID`로 둬야 한다. `/home` 산출은 금지한다.
full-NLTE는 GPU memory 80 GB 이상을 검증하고 A40을 거부해야 한다.

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
sbatch --partition=h200 --gres=gpu:h200:1 --mem=96G --time=04:00:00 \
  --export=ALL,A2_ARTIFACT_ROOT=/gpfs/$USER/lumina/a2_13_15 \
  scripts/run_a2_13_15_gpu.slurm
# h200가 종결 실패/불가일 때만:
sbatch --partition=h100 --gres=gpu:h100:1 --mem=96G --time=04:00:00 \
  --export=ALL,A2_ARTIFACT_ROOT=/gpfs/$USER/lumina/a2_13_15 \
  scripts/run_a2_13_15_gpu.slurm
```

runner `scripts/run_a2_13_15_gpu.slurm`은 A2-12 lifecycle을 먼저 실행하고 80 GB 미만 및
H200/H100 이외 GPU를 거부한다. 이어 micro-oracle을 실행·해시하지만 production guard가
남아 있으므로 결과가 모두 맞아도 rc 3과 위 BLOCKED marker를 기록한다.

## 남은 위험과 A2-16/17 인계

- A2-12 수리본 lifecycle 전체 PASS 재실행이 먼저 필요하다.
- production CUDA의 scalar/coarse/fine 잔여 340 raw hit를 분류하고 실제 checked mirror
  consumer로 교체해야 한다. 현재 micro-oracle/runner는 있으나 full-NLTE integration과
  GPU-side N13/N14/N15 poison 실행은 남았다.
- allowlist JSON/sidecar는 source 수정 전에 만들었지만 `.git/index.lock`이 read-only라
  seal commit/blob 3중 hash를 만들지 못했다. 상태는 implementation manifest에 기록했다.
- 최종 전달물은 tar가 아니라 계약별 4커밋을 포함한 git bundle이며, 파일명과 SHA-256은
  최종 handoff에 기록한다(자기 자신을 hash하는 순환을 피하기 위해 bundle 내부에는 미기록).
- A2-16과 A2-17은 계속 `BLOCKED_UPSTREAM_NOT_CLOSED`다. 이 보고의 internal selftest
  PASS를 upstream closure로 해석하면 안 된다.
