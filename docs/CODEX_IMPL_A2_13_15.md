# A2-14/15 production 이관 구현 보고 (개정 11)

기준 HEAD는 `93ff153`이며 push는 수행하지 않았다. 원본 `.git`이 read-only라 임시 git
metadata에서 계약별 커밋을 만들고 최종 git bundle로 전달한다.

## 판정

- A2-14 production 이관: **CLOSED**. CPU A2-08 publication의 signed
  `chi_es/chi_bb/chi_bf/chi_ff/chi_total`과 별도 nonnegative packet event measure를
  generation-bound device view로 게시한다. signed 값에 `abs`, zero clamp, floor를 적용하지
  않는다.
- A2-15 production 이관: **CLOSED**. CPU A2-09 publication의 component eta와 CDF를
  generation-bound device view로 게시하며, packet 재표본은 shell당 전달된 uniform draw
  하나로 같은 CDF를 이진 탐색한다. Planck/`d_T_rad` fallback은 없다.
- A2-13 BF/BB checked canonical view 소비는 선행 커밋 `d47a596`을 그대로 재사용했다.
- 사용자 제공 SLURM 218079 결과에 따라 BF, BB, opacity, emissivity micro-oracle은 모두
  PASS로 기록했다. 구현자는 GPU job을 추가 실행하지 않았다.
- A2-08 L4는 `BLOCKED_MISSING_CHI_DATA`, A2-09 L3/L5는
  `BLOCKED_MISSING_ETA_DATA` 상태를 보존한다. 이 외부 truth gate를 production 이관 PASS로
  세탁하지 않는다. full-NLTE integration도 이번 라운드에는 실행하지 않았다.

## production 배선

`gpu_opacity_production_bind/view/release`는 CPU opacity publication, A2-12 radiation-field
generation, A2-06 line generation과 shape/status를 함께 검증한 뒤에만 device descriptor를
낸다. BF population은 A2-07 committed NLTE population 또는 같은 `T_e`의 정확 LTE
population만 사용한다. CUDA GEMM 실패는 fatal이며 CPU 계산으로 빠지지 않는다.

`gpu_emissivity_production_bind/view/release`는 CPU emissivity publication의 component,
total, CDF와 radiation/line/opacity/emissivity 세대를 함께 결박한다. transport launch 직전
두 checked view를 만들고, opacity event measure와 emissivity CDF descriptor를 kernel에
전달한다. 이전 `GPU_OPACITY_NOT_MIGRATED`와 `GPU_EMISSIVITY_NOT_MIGRATED` production
guard는 제거했다.

coarse `J_nu` 재적분은 추가하지 않았다. legacy opt-in `bf_gemm_compute_fine` ABI는 남겼지만
호출 즉시 fatal로 종료하므로 fine grid 생성과 interpolated-opacity fallback 모두 불가능하다.
CPU transport/solver 계산식은 변경하지 않았다.

## seal과 커밋

물리 source 수정 전에 A2-14와 A2-15 production allowlist 및 SHA-256 sidecar를 각각
봉인했다.

- A2-14 seal: `19c8ab0`, allowlist SHA-256
  `c755c8948cdcd99ad24bec1426cbd296c6387856264e59a15fc70a09ee322230`
- A2-15 seal: `1178878`, allowlist SHA-256
  `eba79a48c8f9ad0330d41f55e17701d3da304e3926a423c9faf9dde4c0e63b32`
- A2-14 production: `65498e1`
- A2-15 production: `84a1481`
- A2-14 fine-grid fail-closed 보강: `c65800e`

신규 translation unit는 만들지 않았으므로 Z 링크 4곳 및 `run_zinert_selftest` 추가 배선
의무는 발생하지 않았다. 기존 GPU kernel TU와 기존 Z occurrence는 census에서 유지된다.

## 검증

- `make lumina -j2`: rc 0.
- `module load cuda && make lumina_cuda -j2`: rc 0; 기존 미사용 항목 warning만 보존.
- `selftest_a2_13_15_contract`: rc 0. BF-only/BB-only 양방향 half-oracle 음성대조를 모두
  거부했다.
- `selftest_a2_08_signed_opacity`: binary/wrapper rc 0; wrapper는 L4를
  `BLOCKED_MISSING_CHI_DATA`로 명시한다.
- `selftest_a2_09_emissivity`: binary/wrapper rc 0; wrapper는 L3/L5를
  `BLOCKED_MISSING_ETA_DATA`로 명시한다.
- 전체 Makefile `selftest*` target 28개를 build 대상으로 재검증했다. CUDA binary는
  컴파일만 하고 실행하지 않았다. gate wrapper의 의도된 BLOCKED rc는 별도 기록한다.
- `scripts/a2_13_15_static_census.py check`: rc 0, ledger 25/25, CUDA files 8/8,
  Z occurrence 5.
- `scripts/a2_01_census_contract.py check`: rc 0, 원장 157행, unclassified 0.
- static read-trace: A2-14/15 guard 0, production BF scalar read 0, target transport
  Planck/`d_T_rad` sample 0, fine opacity runtime 0.

GPU battery와 full-NLTE integration은 요청에 따라 실행하지 않았다.

## A2-16/17 판정

A2-13~15 production closure가 성립하므로 A2-16의
`BLOCKED_UPSTREAM_NOT_CLOSED` 시작 장벽은 해제했다. 그러나 현재 정적 trace는 source
48개에서 production scalar-read 후보 109개와 미격리 generation-0 `T_e` target 1개를
검출하며, 필수 `validation/a2_16/A2_16_TWO_SEED_MANIFEST.json`도 없다. 따라서 A2-16은
**`BLOCKED_A2_16_IMPLEMENTATION_PREREQUISITES`**로 기록했고 source 구현을 시작하지 않았다.
필수 입력 manifest 없이 native seed schema/coverage를 가정하거나 109개 read를 부분
철거하는 것은 명세상 완결이 아니다.

A2-17은 A2-16 closure가 시작 장벽이므로 계속 **`BLOCKED_UPSTREAM_NOT_CLOSED`**다. 원장
157행은 A2-14/15 해당 행만 terminal completion으로 갱신했고 A2-16/17 행을 선행 폐합하지
않았다.

## 운전석 명령

추가 확인이 필요할 때 runner는 submit directory를 repo root로 사용하고 산출은 `/gpfs`에
둔다.

```bash
sbatch --partition=h200 --gres=gpu:h200:1 --mem=96G --time=04:00:00 \
  --export=ALL,A2_ARTIFACT_ROOT=/gpfs/$USER/lumina/a2_13_15 \
  scripts/run_a2_13_15_gpu.slurm
```

H200을 사용할 수 없을 때만 같은 규약으로 H100 partition/GRES를 사용한다.
