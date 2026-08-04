## 부검 결론

397217의 옵션은 모두 정상 파싱됐습니다. 열 위치·옵션 순서 문제는 없습니다. 실패 원인은 “modern의 3.46×10³% 상태에서 LAMBDA를 재개해 발산”한 것이 아니라, modern restart를 삭제하고 LTE cold start에서 첫 스텝부터 LAMBDA를 강제한 별도 궤적입니다.

또한 modern 종점의 `3.46×10³%`는 실제 분기용 `MAXCH`가 아닙니다. 그때 반환된 `MAXCH`는 감소 보정 때문에 `1.00×10⁷%`였습니다.

### ① IN_ITS 옵션 파싱

정상 파싱됐습니다.

- 입력: [IN_ITS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT/IN_ITS:1)
- 실행 echo: [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT/MODEL:50)

Echo 값은 다음과 같습니다.

```text
40 [NUM_ITS]
 T [DO_LAM_IT]
 T [DO_LAM_AUTO]
 T [DO_T_AUTO]
```

소스는 `IN_ITS`를 읽은 뒤 세 키를 각각 `RD_STORE_LOG`로 조회합니다: [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:551).

파서는:

- 줄 어디에서든 `[`와 `]`를 찾아 키를 추출하고
- 키 왼쪽 문자열을 값으로 읽으며
- 조회 순서와 입력 파일 순서가 달라도 검색합니다.

근거: [rd_store_log.f](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/rd_store_log.f:68), [rd_store_log.f](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/rd_store_log.f:156).

따라서 현재 형식은 열 위치·순서에 민감한 형식 오류가 아닙니다. 키 철자는 정확히 일치해야 하는데 현재 모두 일치합니다.

`run_modern.info`의 `DO_T_AUTO present (must be 0): 0`은 파싱 실패 증거가 아닙니다. 제출 스크립트가 잘못된 파일인 `VADAT`에서만 `DO_T_AUTO`를 grep한 결과입니다: [slurm_cmfgen_modern.sh](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT/slurm_cmfgen_modern.sh:45). 이 옵션의 올바른 위치는 `IN_ITS`입니다.

### ② 실제 MAXCH 시계열

397217의 40회 모두:

- `LAMBDA iteration used`
- `Temperature held fixed at all depths`

였습니다. 즉 full iteration 구간은 한 번도 없었습니다.

`SOLVEBA_V13`이 실제 반환한 `MAXCH` 전체 시계열은 다음과 같습니다.

```text
 it  1–10: 2.80e20 7.15e19 1.33e19 2.57e19 1.36e22
           4.62e23 1.70e25 2.70e26 1.16e28 4.80e28

 it 11–20: 2.89e29 1.99e29 1.66e29 9.40e28 2.55e28
           3.08e28 7.02e27 6.17e26 1.03e26 5.70e24

 it 21–30: 1.20e24 1.60e22 6.47e19 1.28e18 5.64e15
           6.23e13 1.25e12 1.01e10 8.42e08 2.36e09

 it 31–40: 3.35e06 1.00e07 1.77e07 3.86e05 4.50e05
           3.29e05 2.64e06 1.66e07 1.50e06 1.69e06
```

판정:

- 1→3회는 하강.
- 4→11회는 역행하여 `2.89×10²⁹`까지 상승.
- 12→29회는 대체로 큰 폭 하강.
- 30회 이후에는 반복적인 역행/진동.
- 최저점은 36회 `3.288×10⁵%`.
- 이후 37–38회에 `2.64×10⁶ → 1.66×10⁷%`로 재역행.
- 종점은 `1.686×10⁶%`.

최저점도 50% 문턱의 약 6,576배이고, 종점은 약 33,715배입니다. 문턱에 근접한 적이 없습니다. 마지막 구간은 [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT/OUTGEN:2925)과 [종점](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT/OUTGEN:3058)에서 확인됩니다.

### ③ 3.46×10³%에서 1.69×10⁶%로 간 이유

먼저 두 값은 연속 실행의 전후가 아닙니다.

제출 스크립트가 시작할 때 다음 restart 자료를 삭제했습니다.

```text
SCRTEMP POINT1 POINT2 EDDFACTOR* BAMAT* ...
```

근거: [slurm_cmfgen_modern.sh](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT/slurm_cmfgen_modern.sh:37). 실제 출력도 `POINT1/POINT2` 읽기 실패 후 `Starting a new model`이라고 명시합니다: [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT/OUTGEN:44).

따라서 relT는 modern 종점에서 LAMBDA를 “재개”한 것이 아니라 LTE cold start를 새로 시작했습니다.

두 cold start의 결정적 차이는 첫 스텝입니다.

- modern: `DO_LAM_IT=F`
  - iteration 1은 full BA/TRIDIAG
  - iteration 2–40은 LAMBDA
- relT: `DO_LAM_IT=T`
  - iteration 1–40 전부 LAMBDA/DIAG

소스는 LAMBDA일 때 `TEMP_CHAR='DIAG'`, 아니면 `METH_SOL`—현재 `TRIDIAG`—을 사용합니다: [solve_for_pops.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:125). 즉 relT는 modern에서 유일하게 수행됐던 초기 full coupled correction을 생략하고 LTE 상태에 곧바로 diagonal LAMBDA correction을 적용했습니다.

그 뒤 `RD_LAMBDA=T`가 자동 순환 로직을 덮어씁니다: [solve_for_pops.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:278). `DO_LAM_AUTO`가 이를 풀려면 실제 `MAXCH<50`이어야 하지만 한 번도 충족하지 못했습니다: [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4756).

따라서 근거가 지지하는 기전은:

> modern 종점에서의 LAMBDA 재개가 아니라, restart 삭제 + immediate forced-LAMBDA cold start로 인해 초기 full coupled step이 빠진 별도 수렴 궤적.

부수 오류가 주원인이라는 증거는 없습니다. NaN/negative-opacity 종료가 없었고 정상 exit 0이었으며, `MOM_J_REL_V9 excessive iteration` 횟수도 relT 18회로 modern 35회보다 적었습니다.

#### modern의 3.46×10³% 주의점

modern iteration 40의 출력은:

```text
Maximum % increase = 3.46e3
Maximum % decrease = 1.01e2
MAXCH returned      = 1.00e7
```

입니다: [modern OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:3094).

`SOLVEBA_V13`은 감소율이 99.999% 이상이면 이를 사실상 무한 감소로 보고 `MAXCH=10⁷`로 치환합니다: [solveba_v13.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:201). 따라서 자동 전환 판단에는 3.46×10³이 아니라 1.00×10⁷이 사용됐습니다.

### ④ VADAT MAX_dT

정상 파싱됐습니다.

- 입력 `5.0D-02 [MAX_dT]`: [VADAT](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT/VADAT:632)
- echo `5.00000E-02 [MAX_dT]`: [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT/MODEL:317)

기본값은 0.2이고 옵션이 있으면 덮어쓰므로, echo의 0.05는 명백한 파싱 증거입니다.

다만 397217에서는 모든 스텝에서 T가 고정됐기 때문에 실효가 없었습니다. `MAX_dT`는 온도 보정 항에만 제한을 거는 값입니다: [solveba_v13.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:311). `DO_T_AUTO`도 full step에서 `MAXCH<50`일 때만 `FIX_T`를 해제하는데 해당 조건이 한 번도 성립하지 않았습니다: [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4712).

## 재시도 처방 2안

### A안 — 권고: modern의 검증된 초기 분기 복원 후 자동 전환

- `DO_LAM_IT=F`
- `DO_LAM_AUTO=T`
- `DO_T_AUTO=T`
- `FIX_T=T`, `LAM_VAL=400` 유지
- 40회보다 긴 fixed-T stint로 제출하되 실제 `Maximm changes as returned`를 기준으로 판단

근거는 modern이 initial full/TRIDIAG 한 번을 거친 뒤 relT보다 훨씬 나은 감소 궤적을 보였다는 점입니다. `RD_LAMBDA=F`이면 소스의 full/LAMBDA 순환 로직이 다시 작동합니다. 이후 non-LAMBDA full step에서 실제 `MAXCH<50`이 확인돼야만 T가 자동 해제됩니다.

형식 수정 재제출은 필요 없습니다. 바꿔야 할 것은 형식이 아니라 `DO_LAM_IT`의 의미 설정입니다.

### B안 — 수동 2단: modern 종점에서 full-only fixed-T 수렴 시험

1단:

- relT 종점이 아니라 modern iteration-40의 continuation 자료 사용
- `SCRTEMP/POINT1/POINT2`를 삭제하지 않는 정상 continuation
- `DO_LAM_IT=F`
- `DO_LAM_AUTO=F`
- `DO_T_AUTO=F`
- `FIX_T=T`
- `LAM_VAL`을 현재 실제 MAXCH보다 충분히 크게 두어 자동 LAMBDA 재진입 차단
- full/TRIDIAG만 수행하며 실제 반환 `MAXCH<50`까지 확인

2단:

- 같은 continuation에서 `FIX_T=F`
- `MAX_dT=0.05`를 유지해 온도 보정을 5%로 제한
- variable-T full solve 제출

modern과 relT의 `MODEL_SPEC`은 MD5가 동일하고 scratch record 크기도 같아 구조적 호환성 근거가 있습니다. 다만 modern의 실제 종점 `MAXCH`도 `10⁷`이므로, 3.46×10³만 보고 즉시 T를 풀면 안 됩니다. 1단 full-only에서 반환 `MAXCH<50`을 실제로 관측한 뒤에만 2단으로 넘어가야 합니다.

어떤 파일도 수정하거나 실행·제출·git 작업을 하지 않았습니다.