# 회귀 하니스 EDDFACTOR 판독 수리

## 결론

`scripts/regression_ledger.py`의 CMFGEN EDDFACTOR 판독을 수리했다. 모델 런과 GPU는
사용하지 않았고, 실데이터 전량 재백필도 실행하지 않았다. 검증은 생성 fixture
`--self-test`와 지정된 두 실런의 CPU-only `--no-append` 판독으로만 수행했다.

캡처 `instr_capture_188932`의 s0에서 다음 필수 게이트를 재현했다.

| 값 | 재현값 |
|---|---:|
| `u_CMFGEN` | `694.7485728426198 erg/cm^3` |
| `u_mc` | `1749.0679041201702 erg/cm^3` |
| `u_cs` | `2675.6022754810087 erg/cm^3` |
| `u_mc/u_CMFGEN` | `2.517555231475637` |

07-15 런 `logs/coevolve_consume_a10_kx_gphall`의 s0에서는
`u_mc=400.2107555763917 erg/cm^3`, `u_CMFGEN=694.7485728426198 erg/cm^3`,
`u_mc/u_CMFGEN=0.5760512093445506`을 재현했다.

## 결함 원인

EDDFACTOR의 14개 메타데이터 record 다음 각 data record에서 마지막 열은 Hz가 아니라
CMFGEN `FL`, 즉 **10^15 Hz 단위의 주파수**다. 기존 판독기는 이 열을 Hz로 직접 사용해
`lambda_A=c/nu`를 계산했다. 따라서 실제 파장보다 10^15배 큰 파장이 만들어졌고,
FUV/EUV 및 기존 all-band 선택에 유효 샘플이 남지 않아 69런 모두
`EDDFACTOR has fewer than two samples in all`로 끝났다.

읽기 전용 참조인 `validation/chain_replay_parity59/common.py`의 계약은
`lambda_A=2997.92458/FL`이다. 참조 구현은 수정하지 않았다.

## 수리 내용과 정의

- `FL`을 `nu_Hz=FL*1e15`로 변환해 `J_nu dnu`를 적분한다.
- 대역 선택 파장은 참조 구현과 같은 `lambda_A=2997.92458/FL`로 계산한다.
- EDDFACTOR_INFO의 `ND`, record length, word size, endian 표식과 data record 시작점 14는
  기존대로 사용한다. 유한한 전 depth의 `J_nu`와 양의 유한 `FL`인 record만 사용한다.
- 총 radiation energy density는 참조 `trapping_audit`과 같이 각 source의 유효 native
  주파수 격자 전체를 trapezoid 적분한다. CMFGEN depth별 적분값은 셸 중간속도에서
  log-linear 보간한다.
- 대역비는 FUV `[918,1290] Å`, EUV `[450,918] Å`의 native bin-center 표본을 각각
  적분한다. 분모 0, 표본 부족, 비유한 값, 속도 범위 밖은 보정하지 않고 이유 있는
  `UNDEFINED`다.
- fixture EDDFACTOR도 실제 파일과 같은 `FL` 단위로 기록하도록 고쳤고, 총 에너지 및
  FUV/EUV가 모두 정의되는 수치 게이트를 추가했다.

수치 floor, clamp, endpoint 발명, 외삽, 대체 oracle 또는 fallback은 추가하지 않았다.
실런 50개 셸 중 CMFGEN RVTJ 속도 범위 안의 44개 셸은 두 metric 모두 산출됐다.
외곽 s44--s49는 목표 속도가 CMFGEN 범위 `[1024.9710054,35975.288045] km/s` 밖이므로
기존 계약대로 행을 유지한 채 `UNDEFINED`이며, metric 전체 상태는 `PARTIAL`이다.

## FUV/EUV 실데이터 확인

두 런의 s0에서 다음 값이 `--no-append` 결과에 기록됐다.

| 런 | `u_mc_EUV` | `u_mc_FUV` | `u_CMFGEN_EUV` | `u_CMFGEN_FUV` | `mc_FUV/EUV` |
|---|---:|---:|---:|---:|---:|
| capture 188932 | `106.30413396933915` | `211.18251593428877` | `20.71544209557125` | `83.60055560454931` | `1.986587990973138` |
| 07-15 gphall | `0.2986483993232127` | `1.9222695861645278` | `20.71544209557125` | `83.60055560454931` | `6.436564168837712` |

단위는 네 energy density 열 모두 `erg/cm^3`다.

## 음성 대조와 자기검사

EDDFACTOR 음성 대조는 fixture oracle에 과거 결함을 그대로 주입한다. 즉 `FL`을 Hz로
간주하고 파장도 `c/FL`로 계산한다. 정상 판독에서 통과한 필수 EDDFACTOR gate가 이
주입에서는 반드시 실패해야 한다.

실행 명령:

```bash
python3 scripts/regression_ledger.py --self-test
```

출력과 종료 코드는 다음과 같았다.

```text
NEGATIVE CONTROL: FAIL (expected): injected uv_fraction=1.5 -> UV fraction outside [0,1]: 1.5
NEGATIVE CONTROL EDDFACTOR: FAIL (expected): FL treated as Hz -> fixture EDDFACTOR metrics did not define: EDDFACTOR has fewer than two samples in euv
PASS fixture metrics: all 8 metric objects present and strict-JSON valid
PASS fixture EDDFACTOR: FL decoded as 10^15 Hz; energy and FUV/EUV metrics defined
PASS b_k dual weighting: ordinary and n_k-weighted medians remain distinct
PASS fixture census: levelpop and census paths agree exactly
PASS append-only: first JSONL prefix preserved; recomputed_at added on second measurement
PASS missing-input fixture: row retained; unavailable run-side values are UNDEFINED
PASS payload-only fixture: chi_es/chi_tot and epsilon_eff defined without plasma T_e
PASS --self-test
```

종료 코드는 `0`이다. 기존 UV 음성 대조, b_k 무가중/n_k-가중 중앙값 분리,
append-only prefix와 `recomputed_at`, 결손 입력 행 유지 검사를 모두 계속 통과한다.

## 소수 런 `--no-append` 확인과 대장 불변성

실행한 명령은 다음과 같다. 출력은 대장이 아닌 `/tmp`에만 저장했다.

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/regression_ledger.py --no-append \
  /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932 \
  logs/coevolve_consume_a10_kx_gphall \
  > /tmp/regression_ledger_eddfactor_noappend.jsonl
```

두 JSON 행은 모두 `ledger_schema_version=2`였고, 각 행의
`radiation_energy_density`와 `band_energy_ratio`는 50개 중 44개 셸이 정의됐다.

`validation/regression_ledger/ledger.jsonl`은 작업 전후 모두 다음과 같다.

- 행 수: `69`
- byte 수: `32594156`
- SHA-256: `955a96d6dfd47fb535b40bd1b0d0e0c391361934d1921dec877c6b20aa39c983`

즉 기존 69행을 수정하거나 삭제하지 않았다.

## `ledger_schema_version` 처리

버전을 `1`에서 `2`로 올렸다. 이유는 단순한 byte-reader 수정에 더해 총 에너지 정의를
실제 필수 게이트 및 작동 참조와 일치하도록 `100<=lambda_A<20000` 선택 적분에서
각 source의 전체 유효 native frequency grid 적분으로 명시적으로 변경했기 때문이다.
대역 정의에도 EDDFACTOR `FL` 단위를 명시했다. 버전 1의 69행은 그대로 보존한다.

운전석이 같은 run path를 재백필하면 새 버전 2 행만 append되며, 기존 행이 있으므로 새
행에 `recomputed_at`과 `prior_measurement_count`가 붙는다.

## 운전석 재백필 명령

계산 노드 allocation의 job step 안에서 저장소 루트 기준으로 실행한다. GPU는 요청하지
않으며 launcher가 `CUDA_VISIBLE_DEVICES`를 빈 값으로 고정한다.

```bash
srun --ntasks=1 bash scripts/backfill_regression_ledger.sh
```

새 CPU job 예시는 다음과 같다.

```bash
sbatch --job-name=lumina-ledger --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --mem=32G --time=04:00:00 \
  --wrap='cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && bash scripts/backfill_regression_ledger.sh'
```

이번 수리에서는 위 전량 재백필 명령을 실행하지 않았다.
