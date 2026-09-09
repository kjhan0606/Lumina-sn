# CMFGEN twin-map 개정 3 — 구현 착지 후 운전자 검수

2026-09-09 · 저자: 운전자 · **REWORK / SAFE_FOR_RUN: NO**.
이는 구현 검수이며 Opus5 묶음 종료감사가 아니다. GPU 지출·등록 판정런은 여전히 0이다.
통과한 직접 시험을 실물 실행 자격이나 전체 물리 비교의 검증으로 승격하지 않는다.

## 착지와 대상

sol 세션 `01a084b2-1b88-7730-98b6-60b6033fd565`가 정상 종료(rc=0)했다.
작업 디렉터리는 `/gpfs/kjhan/lumina/cmfgen_map/worktree`이며 두 파일은 미커밋으로 보존했다.
메인 저장소로 코드 통합·실물 staging·Slurm 제출은 하지 않았다.

| 대상 | SHA-256 |
|---|---|
| 계약 `docs/PREREG_CMFGEN_TWIN_MAP_2026-09-08.md` | `b367ae0ec3b629b4f14fecc870b4cf582488efc96f1d0223f22ef4299b122b5f` |
| `scripts/cmfgen_twin_map.py` (2,174행) | `73a9042b5b77646a4488319ce2e238678b265609a0ef701cc95b9062ec686ad5` |
| `scripts/stage_cmfgen_twin_map.sh` (842행) | `ee3f2ca2cf9f2c6c9ee21969609b7e59a2722030fb71f30ffcfa422fe763ec38` |

발주·착지·재현 증거 루트: `/gpfs/kjhan/lumina/cmfgen_map/impl_rev3`.
수정 전 파일은 `cmfgen_twin_map.py.before`, `stage_cmfgen_twin_map.sh.before`다.
`SOL_REPORT.md` 해시: `71d0dbc3bf23d92b628841a88e61c8a35fdaedee8503b88b7e4d852deccc44f3`.

## 재실행한 직접 시험

아래 명령은 메인 저장소에서 실행했고 실제 CPU 작업은 `grammar-debug`에서 수행했다.
두 명령 모두 rc=0이다. 사설 fixture staging은 있었으나 봉인 BASE의 실물 staging은 없었다.

```bash
scripts/gd "cd /gpfs/kjhan/lumina/cmfgen_map/worktree && python3 -B scripts/cmfgen_twin_map.py --selftest > /gpfs/kjhan/lumina/cmfgen_map/impl_rev3/operator_mapper_selftest.log 2>&1"
scripts/gd "cd /gpfs/kjhan/lumina/cmfgen_map/worktree && bash scripts/stage_cmfgen_twin_map.sh --selftest > /gpfs/kjhan/lumina/cmfgen_map/impl_rev3/operator_stager_selftest.log 2>&1"
```

- 분석기: `CMFGEN_TWIN_MAP_SELFTEST_PASS observations=29`.
- 스테이저: `CMFGEN_TWIN_MAP_STAGER_SELFTEST_PASS positive=11 negative=25`.
- 실제 원자료 준비 시험: `anchors=50 base_e_z_cells=50 pending=8`.
- 위 8개 미확인 항목은 실제 loader CPU 소비, 최종 공개 세대 연결, 6원소의 최고 단계 정의다.

재현 상태: **reproduced (해당 시험 범위만)**.

## 실행을 막는 발견 — 한 묶음으로 처리

### R1. 비유한 NLTE 한 값이 전체 지도 출력을 중단함 — 구현 결함, reproduced

판독기 1536행은 앞에서 잡은 비유한 population을 `R_SN` 산출에서 다시 `decimal_token()`으로
읽으며 예외를 처리하지 않는다. 정상 사설 입력은 `analyze rc=0, rows=74`였으나
`shell=0, Z=20, ion=1, level=0`의 `n_pop` 하나를 `NaN`으로 바꾸면
`NEGATIVE_OR_NONFINITE_VALUE NaN` 예외로 중단한다. 기존 지도·summary 해시는 그대로다.
즉 실패 원값·행을 남긴 새 결과가 아니라 이전 결과만 남는다. 성공을 잘못 보고한 CLI는 아니지만,
계약의 부분 관측·실패 보존 의무를 충족하지 못한다. 정상·실패를 같은 실제 분석 경로에서 시험했다.

### R2. 닫힘 일치 표기가 실제 잔차를 검사하지 않음 — 구현 결함, reproduced

판독기 1589행의 `consistent_with_printed_reference_interval`은 계산한 닫힘 잔차가 아니라
참조 행 합의 `includes_one`을 그대로 복사한다. 사설 Co 입력에서
`residual=-0.0497715090751741709434584523`인데도 이 필드는 `true`다.
참조 합 검사와 Saha 닫힘 대조를 분리해야 한다. 이는 전체 자동 PASS가 아니라 **잘못된 국소 일치 표기**다.
해당 분율 행은 별도로 `ELEMENT_DENOMINATOR_MISMATCH`를 기록하므로 전체 실패가 은폐됐다고 확대하지 않는다.

### R3. 생산 원자량과 분석 A=56이 충돌함 — 설계/기준 결함, reproduced

실제 BASE:
`/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260906T234024Z_dc1_ne_closure_on`.
그 `input/model/atom_masses.csv`는 Fe=55.845, Co=58.933194, Ni=58.6934다.
대응 소스 `/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/src/lumina_plasma.c`
2781·2808·2981·2990행은 이 원자량으로 `N_Z=rho*X/(mass*AMU)`를 계산하고 생산 단계를 정규화한다.
분석기 641행만 A=56으로 바꾸면 정규화된 생산 단계 합을 새 분모로 나눈 값은 다음과 같다.

| 원소 | 생산 단계 합 / A=56 분석 분모 |
|---|---|
| Fe | 1.002775539439520100277553944 |
| Co | 0.9502284909248258290565415477 |
| Ni | 0.9541106836543802199225125823 |

위 숫자는 **봉인 소스의 정규화식에서 도출한 값이며 새 GPU 덤프의 실측값이 아니다**.
같은 생산 정규화를 넣은 사설 fixture에서도 세 원소 모두 `ELEMENT_DENOMINATOR_MISMATCH`를 재현했다.
코드는 개정 3의 A=56 지시를 따랐지만, 설계가 봉인 생산 분자와 새 분석 분모의 불일치를 놓쳤다.
기존 설계감사 PASS를 이 발견의 반증으로 사용하지 않는다.

권고(아직 미적용): 생산 단계 분율의 분모와 A=56 조성 비교용 분모를 명시적으로 구분한다.
생산 원자량·바이너리·BASE는 보존하고 두 분모의 비를 조건 차이로 남긴다.
`⟨Z⟩` 명칭·해석과 닫힘식까지 함께 정리해야 하므로, 조용한 코드 우회가 아니라 계약 수정이 필요하다.
다른 선택인 생산 질량 입력 수정은 T-only 비교를 바꾸므로 이번 권고에 포함하지 않는다.

### R4. H100/H200 실행 경로는 여전히 미적격 — 실행/증거 미결, reproduced

실제 파일은 `BASE/input/job.slurm`이다. 헤더에는 h200/h100/a100이 있지만 내부에는
A100 하드웨어·a100 파티션 제한이 남아 있다. 현 스테이저의 실제 변환 함수에 이 파일을 읽히면:

```text
REAL_RUNNER_REWRITE_RC 1 OUTPUT_CREATED False
INPUT_PENDING: legacy A100-only runner is not qualified for H200/H100
```

기존 실행 경로의 미결이 지속되는 것이며 새 구현이 만들어낸 하드웨어 문제는 아니다.
제출 옵션만 바꾸거나 가드를 무검증 삭제하지 않았다. 실제 loader CPU 소비, 최고 단계 원문 정의,
runtime·memory·하드웨어·저장 자격도 여전히 별도 실물 근거가 필요하다.

## 결정적 재현과 증거 identity

```bash
scripts/gd "python3 -B /gpfs/kjhan/lumina/cmfgen_map/impl_rev3/operator_reproduce.py > /gpfs/kjhan/lumina/cmfgen_map/impl_rev3/operator_reproduce.log 2>&1"
```

전체 재현 명령 rc=0. R1 예외와 R4 변환 rc=1을 의도적으로 포착하여 기록한다.
R1·R2·R3 fixture는 **생산 함수의 결함 재현**이며 독립 물리 검증으로 세지 않는다.
R3 수치의 `56/mass` 계산은 생산 분석 산술을 사용하지 않는다.
테스트 파일은 전용 `TemporaryDirectory`에서 생성·정리했고 BASE에는 쓰지 않았다.

| 증거 루트 내 파일 | SHA-256 |
|---|---|
| `operator_reproduce.py` | `486138d72c2129639187435ff28096be5dc090af5e6d93a8448fb8d6f5a2c1b4` |
| `operator_reproduce.log` | `66b39495515f0a92294ab7c80fcc58189f941cbb405f361615e677aa44ccaffe` |
| `operator_mapper_selftest.log` | `93b0c62fd4baae0806823a14839296d9d31ad1a1f5fc8b6ebc500e8a632cf79b` |
| `operator_stager_selftest.log` | `279c013ac4fdb4a76a3832c68e42835a2776a58934c5fbdc9b53f10664d95967` |

실물 질량표 해시: `4861116af3b94750905c36b4528ccf477d8a35250c4cb7e49ea1c64d7f62825c`.
실물 job 해시: `d20afbc8cafacb6b7bb2938ef3de40a4c68404d4b924fbc10e7e4074f046143e`.
대응 plasma 소스 해시: `90fd73cc51cae34649533fc8ed87f84a39e08521b4fdd6fb8a57b107b959210d`.

## 다음 행동

현재 판본은 보존하되 통합·실물 staging·GPU 제출로 넘기지 않는다.
R3 분모·해석의 최소 계약 수정 방향을 사용자에게 확인한 뒤 운전자가 Q-GOAL·Q-LEAN을 포함해 설계하고,
R1·R2와 실행 경로 정리를 sol에 **한 변경묶음**으로 전달한다. 작은 수정마다 별도 모델 감사를 추가하지 않는다.
묶음 산출이 없는 지금 Opus5 종료감사를 호출하거나 그 PASS를 주장하지 않는다.
