# Codex A-S31 round 6B — 80-digit KA2 oracle 실행 준비

날짜: 2026-08-01  
상태: **PREPARED / production oracle NOT RUN / Slurm NOT SUBMITTED**

## 1. 범위와 규율

원설계 `docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md` §6.2 Equation (15)를 정본으로 고정했다. acceptance는 다음과 같으며 완화하지 않았다.

- oracle 전 단계 `mpmath` 80 decimal digits
- Gauss–Legendre Nyström 및 `r=r'` logarithmic singularity subtraction
- `Nref=2048/4096`
- 512개 shell center `(i+0.5)/512`에서 두 reference의 상대 L2 `<1e-9`
- KA2/rung11: finest J relative L2 `<=1e-4`, max scaled error `<=3e-4`, `1.7<=p_obs<=2.3`, source residual `<=1e-10`, transport residual `<=1e-4`, energy closure `<=1e-4`, 500회 안 수렴, clamp/negative/sign-uncertain/non-finite 모두 0

이번 작업에서 로컬로 실행한 수치 문제는 `Nref=64` smoke 한 번뿐이다. `N>=512` 계산, model/GPU run, `sbatch` 제출, acceptance 수정, 커밋은 하지 않았다.

## 2. full-80-digit oracle

작성 파일은 `scripts/s31_ka2_oracle_hp.py`다. 기본값 또는 production 명령은 다음과 같다.

```bash
python3 scripts/s31_ka2_oracle_hp.py \
  --nref 2048 4096 \
  --workers 32 \
  --out docs/s31_results/ka2_oracle_hp.json
```

수치 경로는 다음과 같다.

1. Legendre recurrence와 Newton iteration으로 Gauss–Legendre node/weight를 80 dps에서 생성한다.
2. Equation (15)의 `E1(|r-r'|)-E1(r+r')` dense kernel을 `mp.mpf`로 조립한다.
3. 대각의 발산항은 `integral E1(|r-r'|) dr'`의 analytic primitive로 subtract/add-back한다.
4. 완전한 dense operator 행을 worker별 retained block으로 보관한다. NumPy/SciPy/binary64 matrix storage는 없다.
5. `(I-0.8 Lambda)J=0.2 Lambda[1]`을 twice-reorthogonalized, unrestarted MGS-GMRES로 80 dps에서 푼다. solve tolerance는 `1e-60`; residual norm도 80 dps다.
6. Nyström node 해를 production radii와 512 shell center에 80-digit natural cubic으로 평가한다.
7. `Nref=2048/4096` 상대 L2를 80 dps에서 계산한 뒤에만 JSON으로 직렬화한다.

고비용 단계마다 timestamp progress log를 stdout과 `OUT.progress.log`에 동시 기록한다. 각 order가 완전히 끝나면 `OUT.checkpoints/N{order}.pickle`을 atomic rename으로 저장한다. 재실행은 schema, order, dps, physics parameter, tolerance, target hash가 모두 같은 완료 체크포인트만 자동 재개한다. 행렬 조립 중간 상태는 재개하지 않으며, 완료된 2048 뒤 4096이 중단된 경우 2048만 재사용한다.

기존 `docs/s31_results/ka2_oracle_rung10R_attempt1.json`은 새 스크립트의 산출물이 아니다. 그 full-precision 선행 시도는 관측점 평가 경로에서 `1.6000e-7`로 Nref gate를 실패했으므로 이번 PASS 근거로 사용하지 않았다. 새 경로의 production pair 결과는 실제 grammar job이 끝나기 전까지 미정이다.

## 3. `ka2_oracle_hp.json` 규격

최상위 schema는 `s31-ka2-oracle-hp-v1`이다. production 파일은 다음 필드를 갖는다.

| field | 계약 |
|---|---|
| `status` | production pair가 있고 원문 gate를 통과하면 `PASS`, 상대차가 실패하면 `FAIL`; single-order run은 `SMOKE` |
| `contract` | Equation (15), 80 dps, required Nref, `<1e-9`, `chi0R=1`, `epsilon=0.2`, `B0=1` |
| `requested_nref` | 실제 요청 order 목록 |
| `arithmetic_audit` | node/weight, E1 assembly, subtraction, dense storage/solve, target evaluation, norm의 80-digit boolean |
| `self_check` | 두 order, 512 center, 80-digit `relative_l2`, threshold, boolean |
| `solver_diagnostics` | order별 linear residual `<1e-50`; 원문에 새 문턱을 추가하지 않기 위해 acceptance와 분리 |
| `targets.values` | `i/512` 513점과 `(i+0.5)/512` 512점의 정렬 union, 80-significant-digit decimal string |
| `references.{2048,4096}` | quadrature/operator/solver 설명, iterations, residuals, matvecs, quadrature audits, elapsed time, target `J` strings 및 canonical SHA-256 |
| `oracle_qualified` | full-80-digit audit와 원문 Nref self-check의 conjunction |
| `runtime` | Python/mpmath/host/workers/time, resumed orders, log/checkpoint 경로, `model_or_gpu_run=false` |

Acceptance에 쓰이는 실수는 JSON float로 낮추지 않고 80-significant-digit decimal string으로 보존한다. `elapsed_seconds` 같은 실행 metadata만 Python float다.

## 4. N=64 smoke와 시간 외삽

실행 명령은 다음과 같았다. 모든 산출물은 `/tmp`에 두었다.

```bash
python3 scripts/s31_ka2_oracle_hp.py \
  --nref 64 --workers 1 \
  --out /tmp/s31_ka2_oracle_hp_smoke64.json \
  --checkpoint-dir /tmp/s31_ka2_oracle_hp_smoke64.checkpoints \
  --log /tmp/s31_ka2_oracle_hp_smoke64.progress.log \
  --no-resume
```

결과:

| quantity | value |
|---|---:|
| process wall | `1.36 s` |
| oracle arithmetic stage | `0.984821 s` |
| max RSS | `27,648 KiB` |
| GMRES iterations / dense matvecs | `32 / 35` |
| linear relative residual | `1.9347503912e-61` |
| source fixed-point residual | `4.5082421839e-61` |
| target J SHA-256 | `0e95ca8988fd2b4aa8794f16df6171e12018eb69704e2f99c8a8487a00fe3133` |
| status | `SMOKE`, `oracle_qualified=false` |

GMRES iteration 수가 smoke와 선행 실행에서 약 32로 고정되어 있어 지배 비용은 dense assembly/matvec의 `O(N^2)`다. 1-worker smoke arithmetic time을 사용하고 32-worker 이득을 전혀 산입하지 않은 보수 외삽은 다음과 같다.

```text
T(2048) = 0.984821 * (2048/64)^2 = 1008.46 s = 16.81 min
T(4096) = 0.984821 * (4096/64)^2 = 4033.83 s = 67.23 min
serial total                              = 84.04 min
```

실제 32-worker grammar 실행은 더 빠를 가능성이 있지만 time request에는 반영하지 않았다. `scripts/sbatch_s31_oracle_hp.sh`는 `--time=03:00:00`으로 보수 외삽 합계 대비 2.14배 여유를 둔다.

메모리는 `4096^2`개의 Python/mpmath mpf object와 worker/process overhead가 지배한다. 초안은 `--mem=32G`, `--cpus-per-task=32`다. GPU GRES/partition은 지정하지 않아 grammar의 default CPU partition을 사용한다. 제출 위치의 실제 default partition과 32 GiB 정책은 운전석에서 `sinfo/scontrol`로 확인해야 하며, 이 작업에서는 조회 가능한 grammar scheduler가 없어 partition 이름을 추측해 박지 않았다.

## 5. rung10/11 judge

`scripts/s31_ka2_judge.py`는 `--oracle`을 필수로 요구한다.

```bash
python3 scripts/s31_ka2_judge.py \
  --oracle docs/s31_results/ka2_oracle_hp.json \
  --out docs/s31_results/ka2_oracle_hp_judgment.json
```

기본 solver 입력은 `docs/s31_results/ka2_rung10.json`과 `scattering_rung11.json`이다. judge는 다음을 수행한다.

- rung10: 80 dps, exact Nref contract, 전 단계 arithmetic audit, `<1e-9`를 fail-closed 재판정한다.
- rung11: 저장된 primitive metrics에서 원문 수치 문턱과 counter를 다시 계산하고, rung10 oracle qualification을 최종 gate로 결합한다.
- binary64 구 oracle의 Nref self-agreement `3.6445269209240967e-10`, matrix storage, strict status를 HP80 결과와 나란히 기록하며 absolute metric difference와 ratio를 낸다.
- oracle JSON이 없거나 malformed/SMOKE이면 PASS하지 않는다.

중요한 한계가 있다. 기존 solver JSON은 raw 513-point J vector를 삭제하고 binary64 oracle에 대해 이미 계산된 `J_oracle_relative_l2`와 `max_scaled_error` scalar만 보존했다. 따라서 judge는 두 scalar의 문턱을 다시 적용하되 이름에 `legacy_binary64_basis`를 명시한다. HP oracle과 solver J의 pointwise 재계산을 했다고 위장하지 않는다. 출력의 binary64 차이도 “Nref self-agreement metric 차이”이며 oracle vector의 pointwise 차이가 아니다. 향후 raw solver J가 제공되기 전에는 이 두 scalar의 basis 한계가 남는다.

가짜 입력 자기시험 결과:

| case | rung10 | rung11 | overall | exit |
|---|---|---|---|---:|
| valid HP oracle (`5e-10`) + valid solver | PASS | PASS | PASS | 0 |
| invalid HP oracle (`2e-9`) + valid solver | FAIL | FAIL | FAIL | 1 |
| `--oracle` 누락 | 입력 거부 | — | — | 2 |
| valid fake HP oracle + 기존 두 solver JSON | PASS | PASS/PASS | PASS | 0 |

마지막 행은 judge wiring 시험일 뿐 실제 oracle 판정이 아니다. fake oracle을 `docs/s31_results`에 쓰지 않았다.

## 6. 산출물과 현재 판정

- `scripts/s31_ka2_oracle_hp.py`: full-80-digit production/smoke oracle
- `scripts/s31_ka2_judge.py`: strict rung10/11 재판정기
- `scripts/sbatch_s31_oracle_hp.sh`: grammar CPU 제출 초안, 미제출
- `docs/CODEX_STAGE31_IMPL6B.md`: 본 보고서

현재 판정은 **PREPARED**다. `docs/s31_results/ka2_oracle_hp.json`은 아직 존재하지 않으며, rung10/11의 새 최종 PASS를 선언하지 않는다. 다음 권한 있는 동작은 grammar 운전석에서 sbatch 초안을 검토·제출하고, 완료 JSON을 judge에 넣는 것이다.
