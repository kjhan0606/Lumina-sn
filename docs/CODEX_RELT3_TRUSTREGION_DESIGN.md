# relT3 설계 조사 보고서

## 결론

1. **우선 패치가 필요하지 않다.** `MAX_LIN/MAX_LAM=M`은 실제로 인구 갱신비를 \(1/M\le n_{\rm new}/n_{\rm old}\le M\)로 자르는 네이티브 trust radius다. 따라서
   \[
   |\Delta\ln n|\le\ln M
   \]
   이며, relT2의 `M=10`은 \(\rho=\ln10=2.303\), 즉 한 step에 10배 증감까지 허용해서 너무 느슨했다. `M=1.10–1.25`면 \(\rho=0.095–0.223\)의 실질적인 인구 trust-region이 된다. [solveba_v13.f:292](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:292), [fiddle_pop_corrections_v2.f:184](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:184)

2. `SCALE_OPT=MAJOR`에서도 **trace/고준위가 무제한인 것은 아니다.** `10^-10 n_e`보다 큰 변수만 depth 공통 scale 산정에 참여하지만, 최종 per-variable clip은 `J=1..NT` 전부에 적용된다. 따라서 작은 Si III 고준위와 terminal-ion 변수도 `MAX_LAM/MAX_LIN`의 factor cap을 받는다. [fiddle_pop_corrections_v2.f:150](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:150), [fiddle_pop_corrections_v2.f:184](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:184)

3. 권고 P0는 `MAJOR + MAX_LAM=1.10 → 1.20 + MAX_LIN=1.05 → 1.10`, 강제 LAMBDA·fixed-T·NG off다. 패치는 네이티브 cap으로도 안정화되지 않거나, **trace/major별 서로 다른 radius와 제한 발화 감사 로그**가 필요할 때만 사용한다.

---

## 1. 네이티브 correction 제한 장치

### 1.1 correction과 MAXCH의 의미

선형해 `c=STEQ(J,I)`의 부호와 갱신은

\[
n_{\rm new}=n_{\rm old}(1-cS)
\]

이다. `c>0`은 감소, `c<0`은 증가다. [solveba_v13.f:155](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:155), [solveba_v13.f:320](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:320)

중요하게도 `CORRECTION_SUM`, 화면 극값, 반환 `MAXCH`는 **실제 scaling 전에 raw `STEQ`로 계산**된다. 실제 scaling은 그 뒤에 시작한다. 그러므로 `MAX_LAM=10`이 적용돼도 OUTGEN에는 `10^7–10^9%` raw correction이 그대로 보일 수 있다. 이것은 cap 미작동의 증거가 아니다. [solveba_v13.f:143](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:143), [solveba_v13.f:201](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:201), [solveba_v13.f:278](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:278)

감소에 대한 반환값은

\[
100\,{c\over1-c}
\]

이며 `c≥0.99999`이면 `10^7%` sentinel로 치환된다. [solveba_v13.f:201](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:201)

또한 큰 correction이 소수일 때 `DO_LEVEL_CHK`가 켜지면 가장 큰 값이 아니라 **10번째 증가/감소**로 `MAXCH`를 다시 계산한다. 소수의 terminal/Si outlier가 반환 MAXCH에서 가려질 수 있으므로 relT3 관문은 MAXCH만 보면 안 된다. [solveba_v13.f:218](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:218), [solveba_v13.f:261](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:261)

소스 주석은 이때 outlier를 3배 완화한다고 쓰지만, 실제 V13 구현에서 `DO_LEVEL_CHK`의 유일한 후속 사용은 MAXCH 재계산이다. update/scaling에는 연결되지 않는다. 즉 해당 주석은 현재 코드와 불일치한다. [solveba_v13.f:255](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:255), [solveba_v13.f:266](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:266)

### 1.2 `MAX_LIN/MAX_LAM`의 정확한 산식

full이면 `CHANGE_LIM=MAX_LIN`, LAMBDA이면 `MAX_LAM`이 전달된다. [solve_for_pops.f:125](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:125)

\(M=\mathrm{CHANGE\_LIM}>1\)에 대해 코드가 정의하는 한계는

\[
B={M-1\over M},\qquad L=1-M.
\]

최종 적용 correction \(x\)가 \(L\le x\le B\)이면

\[
1-B={1\over M},\qquad 1-L=M,
\]

따라서 \(n_{\rm new}/n_{\rm old}\in[1/M,M]\)이다. [solveba_v13.f:292](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:292), [solveba_v13.f:297](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:297)

| `M` | 네이티브 \(\rho=\ln M\) | step당 인구 범위 |
|---:|---:|---:|
| 10.0 | 2.3026 | ×0.1…×10 |
| 1.25 | 0.2231 | ×0.80…×1.25 |
| 1.20 | 0.1823 | ×0.833…×1.20 |
| 1.10 | 0.0953 | ×0.909…×1.10 |
| 1.05 | 0.0488 | ×0.952…×1.05 |

따라서 relT2의 `MAX_LIN=MAX_LAM=10`은 기능적으로 작동했더라도 발산 mode를 억제하기에는 너무 넓었다. 실제 입력값은 `10/10`이었다. [relT2 MODEL:315](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:315)

### 1.3 `SCALE_OPT` 네 경로

#### `MAJOR` — 권고

depth \(i\)에서 `POPS(J,I)>10^-10*POPS(NT-1,I)`인 변수만 scale 극값 산정에 포함한다. `NT-1`이 전자밀도라는 것은 scratch reader에서도 명시된다. [fiddle_pop_corrections_v2.f:153](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:153), [read_seq_time_file_v1.f:132](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/read_seq_time_file_v1.f:132)

공통 scale은

\[
S_i=\min\left({B\over\max(B,\max_{\rm major}c)},\,
              {L\over\min(L,\min_{\rm major}c)}\right)
\]

이고 T correction에 의해 더 작아질 수 있다. 이후 각 변수에 대해 \(x=cS_i\)를 다시 `[DPTH_LIT_LIM, DPTH_BIG_LIM]`으로 clip한다. 따라서 scale 산정에서 제외된 trace/high-level도 최종 factor cap을 받는다. [fiddle_pop_corrections_v2.f:153](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:153), [fiddle_pop_corrections_v2.f:184](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:184)

즉 “minimum value of scale for Major species”는 정확히는 **major 변수로 계산한 depth 공통 scale의 최솟값**이다. 최종 trace clip 횟수나 실제 최소 per-variable factor는 출력하지 않는다. [fiddle_pop_corrections_v2.f:194](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:194)

#### `LOCAL`

각 depth에서 모든 비-T 변수의 최대/최소 raw correction으로 하나의 scale을 정해 해당 depth 전체에 곱한다. T에는 `MAX_dT`와 `T_MIN`을 적용한다. [solveba_v13.f:301](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:301), [solveba_v13.f:311](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:311)

trace outlier 하나가 전체 depth를 지나치게 느리게 만들 수 있어 relT3 P0에는 `MAJOR`가 낫다.

#### `NONE`

모든 비-T 변수를 독립적으로 `[L,B]`에 clip하므로 가장 직접적인 전준위 log trust-region이다. 다만 T는 입력 `MAX_dT`를 사용하지 않고 20%를 hard-code한다. T 해제 단계에는 부적합하다. [solveba_v13.f:329](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:329), [solveba_v13.f:344](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:344)

#### `GLOBAL`

전 depth의 모든 correction으로 단일 scale을 만든다. 소스 자체가 “probably obsolete”라고 표시한다. T 제한도 입력 `MAX_dT`가 아니라 20% hard-code이고, `T3` loop가 최대값을 누적하지 않아 사실상 마지막 depth 값만 남는다. 사용 비권고다. [solveba_v13.f:278](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:278), [solveba_v13.f:360](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:360), [solveba_v13.f:377](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:377)

### 1.4 LAMBDA 선행 limiter

`LAM_SCALE_OPT='LIMIT'`이면 LAMBDA에서 비-T correction `c>1.1`을 `0.999`로 바꾼다. 증가 방향의 큰 음수 correction은 건드리지 않고, `1.0<c≤1.1`도 통과한다. relT2 terminal 값 `1.0064`에는 발화하지 않는다. [solveba_v13.f:125](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:125), [solveba_v13.f:129](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:129), [modern terminal correction](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/STEQ_VALS:1952752)

이 limiter는 raw diagnostics 전에 `STEQ` 자체를 바꾸므로 관문 계측을 일부 가린다. relT3에서는 `LAM_SCALE_OPT=NONE`을 권고한다. 파서는 문자열을 제한하지 않고, 코드도 첫 5자가 `LIMIT`일 때만 동작한다. [rd_control_variables.f:1000](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/rd_control_variables.f:1000), [solveba_v13.f:129](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:129)

### 1.5 `ADJUST_CORRECTIONS`

`SCALE_OPT=MAJOR`일 때만 읽히며 다음 on-the-fly 옵션이 있다. [fiddle_pop_corrections_v2.f:89](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:89)

| 키 | 코드 기본값 | 실제 기능 |
|---|---:|---|
| `L_ST/L_END` | `1/ND` | 적용 depth |
| `T_LIM` | `MAX_dT` | depth별 T fractional cap |
| `RELAX` | `1` | scale이 정확히 1일 때만 scale을 대체 |
| `MAX_CHNG` | `100*M` | depth별 인구 factor cap; VADAT의 동명 종료 문턱과 다름 |
| `CONSIS_CNT` | `0` | 0이면 비활성 |

기본 설정은 [fiddle_pop_corrections_v2.f:69](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:69), 입력 키는 [fiddle_pop_corrections_v2.f:95](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:95)에 있다.

하지만 LAMBDA에서는 `RELAX=1`과 `POP_LIM=100*M`으로 강제 재설정된다. 즉 `RELAX`과 `ADJUST_CORRECTIONS[MAX_CHNG]`는 이번 LAMBDA 폭주를 제어하지 못한다. [fiddle_pop_corrections_v2.f:145](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:145)

또한 `RELAX`은 correction 전체에 항상 곱하는 언더릴랙세이션이 아니라 `SCALE==1`일 때만 대체된다. 이미 native scale이 1보다 작으면 무효다. [fiddle_pop_corrections_v2.f:174](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:174)

`CONSIS_CNT`는 현재 호출 인수 순서가 서브루틴 선언과 반대다. `BAD_INCREASE=-10^10`, `BAD_DECREASE≈1`을 반대로 전달하므로 호출 즉시 validation error로 반환되는 구조다. 그대로는 사용 비권고다. [fiddle_pop_corrections_v2.f:200](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:200), [set_depth_consistency.f:5](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/set_depth_consistency.f:5), [set_depth_consistency.f:40](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/set_depth_consistency.f:40)

### 1.6 제어 노브와 기본값 전수

`L_TRUE`로 읽는 값은 필수 키라 코드 기본값이 없다. 키가 없으면 parser가 중단한다. [rd_store_log.f:180](/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/rd_store_log.f:180)

| 범주 | 키 | 코드 기본/필수 | relT2 값 및 비고 |
|---|---|---|---|
| scaling | `SCALE_OPT` | 필수 | `MAJOR` |
| correction | `MAX_LIN`, `MAX_LAM` | 필수, `>1` | `10`, `10` |
| LAMBDA prelimit | `LAM_SCALE_OPT` | `LIMIT` | 입력 생략→`LIMIT` |
| T | `MAX_dT` | `0.2` | `0.05`; MAJOR/LOCAL에서만 입력값 사용 |
| T floor | `T_MIN` | `0` | `0.5=5000 K` |
| LAMBDA 강제 | `DO_LAM_IT` | 필수 | `F` |
| 강제 LAMBDA 자동 해제 | `DO_LAM_AUTO` | `T` | `T`; MAXCH<50에서 해제 |
| 일반 auto-LAMBDA | `LAM_VAL` | 필수 | `400%` |
| LAMBDA 횟수 | `NUM_LAM` | 필수 | `2` |
| T 자동 해제 | `DO_T_AUTO` | 실제 할당 기본 `F` | relT2 `T`; full MAXCH<50에서 해제 |
| T 고정 | `FIX_T` | 필수 | `T` |
| 부분 T 고정 | `FIX_T_AUTO`, `TAU_SCL_T` | 필수 | `F`, `0` |
| NG | `DO_NG` | 필수 | `T` |
| NG validity | `CHK_NG` | 필수 | `T` |
| NG 관문 | `BEG_NG`, `IBEG_NG` | 필수 | `5%`, it30 |
| NG 폭/주기 | `BW_NG`, `ITS/NG` | 필수 | `10`, `20`; 주기 최소 4 |
| oscillation 평균 | `DO_AV/NOSC_AV/ITS/AV` | `F/4/8` | 기본 |
| 자동 smoothing | `AUTO_SMOOTH` | `F` | 개발 중 코드 |
| outer undo | `DO_UNDO` | `F` | depth 1–5 직전 값 복구 |
| 안전 종료 | `MAX_CHNG` | 필수 | `10^100`; limiter가 아니라 사후 STOP |
| 수렴 | `EPS_TERM` | 필수 | `0.1%`; 변경 금지 |

파서와 기본값은 [rd_control_variables.f:943](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/rd_control_variables.f:943), [rd_control_variables.f:967](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/rd_control_variables.f:967), [rd_control_variables.f:998](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/rd_control_variables.f:998), [rd_control_variables.f:1143](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/rd_control_variables.f:1143)에 있고, relT2 실제값은 [MODEL:303](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:303), [MODEL:308](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:308), [MODEL:313](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:313), [MODEL:326](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:326), [VADAT:662](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/VADAT:662), [IN_ITS:1](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/IN_ITS:1)에 확인된다.

`DO_T_AUTO`는 파일 머리 주석에는 기본 `TRUE`로 변경됐다고 쓰였지만 실제 할당은 `.FALSE.`다. 실행 의미는 실제 할당을 따라야 한다. [cmfgen_sub.f:46](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:46), [cmfgen_sub.f:565](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:565)

일반 auto-LAMBDA는 full의 MAXCH가 `LAM_VAL` 이상이면 다음 step을 LAMBDA로 바꾸며, LAMBDA MAXCH가 `10^5%`를 넘으면 `NUM_LAM`을 초과해도 LAMBDA를 유지한다. 강제 `DO_LAM_IT=T`인 경우 별도 자동 해제는 MAXCH<50 하나만 본다. [solve_for_pops.f:278](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:278), [solve_for_pops.f:298](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:298), [cmfgen_sub.f:4756](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4756)

`AUTO_SMOOTH`, `DO_AV`, `DO_UNDO`는 trust-region이 아니라 각각 outer-depth 보간, 진동 변수 평균, depth 1–5 rollback이다. 초기 판정 실험에서는 모두 꺼 두어 인과를 분리해야 한다. [smooth_pops_as_we_iterate.f:35](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/smooth_pops_as_we_iterate.f:35), [ave_flips.f:72](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/ave_flips.f:72), [undo_it.f:60](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/undo_it.f:60)

---

## 2. 최소 패치 설계

### 2.1 패치 발동 조건

다음 중 하나일 때만 패치로 승격한다.

- `MAX_LAM=1.05–1.10` 네이티브 cap에서도 Si III/terminal 지표가 두 step 연속 악화.
- major와 trace에 서로 다른 radius가 필요.
- 실제 제한 발화 수를 코드 내부에서 완전 계측해야 함.

그 외에는 native-only가 판정 코드 오염 위험이 가장 작다.

### 2.2 최소 변경 위치

relT3가 `SCALE_OPT=MAJOR`를 고정한다는 전제에서 수정 대상은 두 파일이다.

1. `SOLVEBA_V13`의 `FIDDLE_POP_CORRECTIONS_V2` 호출에 이미 보유한 `MAIN_COUNTER`를 전달. [solveba_v13.f:355](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:355)
2. `FIDDLE_POP_CORRECTIONS_V2`에서 native scale과 fractional clip을 끝낸 직후, 실제 `POPS` update 직전에 Δln limiter 적용. 정확한 지점은 현재 `T1` clip과 update 사이인 [fiddle_pop_corrections_v2.f:187](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:187)이다.

최소 diff 개념은 다음과 같다. 실제 수정은 수행하지 않았다.

```fortran
! New optional ADJUST_CORRECTIONS keys; <=0 means disabled.
DLNN_MAJOR=0.0_LDP
DLNN_TRACE=0.0_LDP
CALL RD_STORE_DBLE(DLNN_MAJOR,'DLNN_MAJOR',L_FALSE,
1                  'Delta ln n radius for major populations')
CALL RD_STORE_DBLE(DLNN_TRACE,'DLNN_TRACE',L_FALSE,
1                  'Delta ln n radius for trace populations')

N_TR_MAJOR=0
N_TR_TRACE=0
N_TR_INC=0
N_TR_DEC=0
N_NATIVE_HI=0
N_NATIVE_LO=0

...
DO J=1,NT
  T1=STEQ(J,I)*SCALE

  IF(T1 .GT. DPTH_BIG_LIM)THEN
    T1=DPTH_BIG_LIM
    N_NATIVE_HI=N_NATIVE_HI+1
  END IF
  IF(T1 .LT. DPTH_LIT_LIM)THEN
    T1=DPTH_LIT_LIM
    N_NATIVE_LO=N_NATIVE_LO+1
  END IF

  IF(J .LT. NT)THEN
    IF(POPS(J,I) .GT. 1.0E-10_LDP*POPS(NT-1,I))THEN
      RHO=DLNN_MAJOR
    ELSE
      RHO=DLNN_TRACE
    END IF

    IF(RHO .GT. 0.0_LDP)THEN
      ! Safe: native DPTH_BIG_LIM is strictly below 1.
      DLN_RAW=LOG(1.0_LDP-T1)
      DLN_USE=MAX(-RHO,MIN(RHO,DLN_RAW))
      IF(DLN_USE .NE. DLN_RAW)THEN
        IF(POPS(J,I) .GT. 1.0E-10_LDP*POPS(NT-1,I))THEN
          N_TR_MAJOR=N_TR_MAJOR+1
        ELSE
          N_TR_TRACE=N_TR_TRACE+1
        END IF
        IF(DLN_RAW .GT. 0.0_LDP)N_TR_INC=N_TR_INC+1
        IF(DLN_RAW .LT. 0.0_LDP)N_TR_DEC=N_TR_DEC+1
        T1=1.0_LDP-EXP(DLN_USE)
      END IF
    END IF
  END IF

  POPS(J,I)=POPS(J,I)*(1.0_LDP-T1)
END DO
```

모든 iteration에서 발화가 0이어도 다음을 OUTGEN과 append-only `POP_TRUST_SUM`에 출력한다.

```text
POP_TRUST it=... enabled=T
 native_hi=... native_lo=...
 trust_major=... trust_trace=...
 trust_inc=... trust_dec=...
 max_abs_dln_raw=... max_abs_dln_applied=...
```

`LAM_SCALE_OPT=LIMIT`도 유지한다면 기존 `COUNT(1)`을 경고문에 포함시켜야 한다. 현재는 발화 여부만 출력하고 수를 버린다. [solveba_v13.f:129](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:129)

### 2.3 패치의 수학적 성질

\[
d_{\rm raw}=\ln(1-x),\qquad
d_{\rm use}=\operatorname{clip}(d_{\rm raw},-\rho,\rho),
\qquad
x_{\rm use}=1-e^{d_{\rm use}}.
\]

따라서 실제 갱신은 \(n_{\rm new}=n_{\rm old}e^{d_{\rm use}}\)이고 항상 양수다. \(x\to0\)이면 limiter가 비활성이고 원래 update와 완전히 같아지므로 고정점과 기존 `EPS_TERM` 판정은 변하지 않는다.

또한 raw `STEQ`는 수정하지 않고 local `T1`만 바꾸므로 기존 `CORRECTION_SUM`, 반환 MAXCH, `STEQ_VALS`, `CORRECTION_LINK`는 계속 raw correction을 기록한다. `CORRECTION_LINK`는 solve 후 raw `SOL`을 받아 종·준위 정보를 쓰는 구조다. [solve_for_pops.f:239](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:239), [solve_for_pops.f:245](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:245), [sum_steq_sol.f:45](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/sum_steq_sol.f:45)

전준위 limiter는 `DLNN_MAJOR=DLNN_TRACE`; trace 강화형은 예를 들어 `0.15/0.075`로 설정한다. trace 정의는 기존 MAJOR 기준과 동일한 `n≤10^-10 n_e`이므로 새 분류 체계를 도입하지 않는다.

---

## 3. relT3 사전등록 프로토콜

### 3.1 재시작 기준점

**relT2 it54가 아니라 modern it40 checkpoint에서 분기한다.**

modern의 `POINT1/2`는 record와 great iteration 모두 40이고, relT2는 54다. [modern POINT1:2](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/POINT1:2), [relT2 POINT1:2](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/POINT1:2)

보존 대상:

- `SCRTEMP`
- `POINT1`, `POINT2`
- `EDDFACTOR` 및 호환 radiation scratch
- `BAMAT`, `BAMATPNT`
- 동일한 원자·물리 입력

`SCR_READ_V2`는 pointer가 없거나 잘못되면 `NEWMOD=T`로 내려가 fresh-model 경로로 돌아간다. `IREC_RD=0`은 pointer가 가리키는 마지막 record를 읽는다. [scr_read_v2.f:85](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/scr_read_v2.f:85), [scr_read_v2.f:123](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/scr_read_v2.f:123), [scr_read_v2.f:183](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/scr_read_v2.f:183)

CMFGEN은 continuing model에서 POPS를 scratch로부터 복원하고 EDDFACTOR도 별도로 연다. [cmfgen_sub.f:907](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:907), [cmfgen_sub.f:957](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:957), [cmfgen_sub.f:1341](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:1341)

각 단계는 stable checkpoint를 복제한 별도 branch에서 수행한다. 실패한 full probe가 기록한 새 SCRTEMP record를 stable branch에 합치지 않는다.

### 3.2 P0 입력

#### VADAT

```text
MAJOR     [SCALE_OPT]
NONE      [LAM_SCALE_OPT]

1.10      [MAX_LAM]     ! 첫 단계; rho=0.0953
1.05      [MAX_LIN]     ! 최초 full probe; rho=0.0488
0.05      [MAX_dT]      ! T 고정 중에는 비활성

T         [FIX_T]
F         [FIX_T_AUTO]
0.0       [TAU_SCL_T]
0.5       [T_MIN]

F         [DO_NG]
T         [CHK_NG]      ! DO_NG=F이므로 비활성
F         [AUTO_SMOOTH]
F         [DO_AV]
F         [DO_UNDO]

T         [COMP_BA]
0         [N_FIX_BA]
T         [STORE_BA]

0.1       [EPS_TERM]    ! 변경 금지
```

full probe에서 `COMP_BA=T`, `N_FIX_BA=0`을 쓰는 이유는 continuing full model이 성공적으로 저장된 BAMAT을 읽으면 `COMP_BA` 입력에 따라 재사용할 수 있기 때문이다. LAMBDA는 자체적으로 BA recomputation을 강제한다. [cmfgen_sub.f:1212](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:1212), [solve_for_pops.f:298](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:298)

#### IN_ITS — LAMBDA 연마

```text
5         [NUM_ITS]
T         [DO_LAM_IT]
F         [DO_LAM_AUTO]
F         [DO_T_AUTO]
```

`DO_LAM_IT=T`는 LAMBDA와 fixed-T를 즉시 강제한다. `DO_LAM_AUTO=F`이므로 MAXCH<50만으로 full로 넘어가지 않는다. [cmfgen_sub.f:602](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:602), [cmfgen_sub.f:4756](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4756)

### 3.3 계측 정의

매 great iteration마다 다음 네 scalar를 저장한다.

1. \(M_k\): `SOLVEBA_V13` 반환 MAXCH.
2. \(C_{100,k}\): `CORRECTION_SUM`의 100% column을 모든 depth에 합산.
3. \(E_{\rm SiIII,k}\): Si III에 매핑되는 모든 unique STEQ equation/depth의 \(\max |c|\).
4. \(E_{\rm term,k}\): 각 ion block 직후 terminal equation의 모든 depth에 대한 \(\max |c|\).

`CORRECTION_SUM`의 raw correction count 산식과 100%…0.0001% column은 [solveba_v13.f:162](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:162)에 있다. level→STEQ 매핑 파일은 [wr_level_links.f:15](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/wr_level_links.f:15)가 생성한다. terminal row가 다음 ion의 SL1로 기술되는 규칙은 [sum_steq_sol.f:31](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/sum_steq_sol.f:31)에 있다.

`CORRECTION_LINK`는 전 배열이 아니라 극값 depth의 top-5만 출력하므로 단독 관문 자료로는 부족하다. [sum_steq_sol.f:47](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/sum_steq_sol.f:47)

### 3.4 단계 스케줄과 full 관문

#### L0 — 강한 damping

- `MAX_LAM=1.10`, 최소 5 LAMBDA steps.
- 첫 step부터 \(M_k\le10^7\%\): relT2의 `10^8–10^9%` 재현 금지.
- 다섯 step 동안 네 지표가 모두 non-increasing.
- 각 연속 구간에서 최소 한 번은 strict 감소.
- \(M\), \(E_{\rm SiIII}\), \(E_{\rm term}\)은 구간 시작 대비 끝이 30% 이상 감소.

반환 MAXCH가 `10^7` sentinel에 붙어 평평할 때는 성공으로 계산하지 않는다. 단, 그동안 \(C_{100}\)과 두 target extreme가 감소하면 L0를 계속할 수 있다. sentinel의 정보 손실은 소스의 decrease 치환 때문이다. [solveba_v13.f:201](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:201)

#### L1 — 완화

L0 관문을 통과하면 stable checkpoint에서:

- `MAX_LAM=1.20`
- 동일한 5-step 관문 반복
- 절대 full 진입 조건:
  - \(M<50\%\)
  - \(C_{100}=0\)
  - \(E_{\rm SiIII}<1\)
  - \(E_{\rm term}<1\)
  - 최근 5 step 모두 non-increasing

`50%`는 CMFGEN의 강제-LAMBDA 자동 해제 문턱과 맞추되, `C100`과 target 조건을 추가한 보수적 설계다. [cmfgen_sub.f:4756](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4756)

#### F0 — full 한 step probe

stable LAMBDA checkpoint를 복제한 시험 branch에서:

```text
1       [NUM_ITS]
F       [DO_LAM_IT]
F       [DO_LAM_AUTO]
F       [DO_T_AUTO]
```

그리고 `MAX_LIN=1.05`, `FIX_T=T`, `COMP_BA=T`.

수락 조건:

- NaN/Inf 및 solver failure 없음.
- 네 지표 모두 마지막 LAMBDA step 이하.
- \(C_{100}=0\).
- Si III/terminal extreme가 각각 `<1`.
- 새 far-outer depth에 correction 집중이 생기지 않음.

하나라도 실패하면 probe branch를 폐기하고 직전 LAMBDA checkpoint에서 `MAX_LAM=1.10`, 필요하면 `1.05`로 되돌린다. 성공할 때만 full branch를 승격한다.

#### F1 — fixed-T full 안정화

- `MAX_LIN=1.10`.
- 최소 5개의 수락된 full steps.
- `DO_NG/DO_AV/AUTO_SMOOTH/DO_UNDO=F` 유지.
- 최근 5 step:
  - \(M<10\%\)
  - \(C_{100}=0\)
  - \(E_{\rm SiIII},E_{\rm term}<0.1\)
  - 전 지표 non-increasing
  - 패치 사용 시 trust limiter 발화 수 `0`

### 3.5 T 해제

위 F1 조건을 모두 충족한 뒤에만 수동으로:

```text
F       [FIX_T]
F       [DO_T_AUTO]
0.02    [MAX_dT]
```

첫 3–5 T-active full steps가 같은 관문을 통과하면 `MAX_dT=0.05`로 복귀한다. `MAX_dT`는 MAJOR 경로에서 T correction에 직접 들어가며, T가 `T_MIN` 아래로 내려가려 할 때 scale을 더 줄인다. [fiddle_pop_corrections_v2.f:163](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:163)

`DO_T_AUTO=T`는 full MAXCH<50 하나만 보고 `FIX_T`를 해제하므로 relT3의 다중 관문을 구현하지 못한다. 계속 `F`로 두고 단계 경계에서만 수동 해제한다. [cmfgen_sub.f:4709](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4709)

최종 CMFGEN 수렴 문턱 `EPS_TERM=0.1%`는 그대로 둔다. source termination도 반환 MAXCH와 이 값을 비교한다. [cmfgen_sub.f:4718](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4718)

### 3.6 사전등록 기대와 조기 실패 기준

**주 기대**

- relT2의 `10^8–10^9%` 재폭주 없이, seed의 반환 MAXCH `10^7%` 이하에서 시작.
- 첫 5-step 구간부터 MAXCH non-increasing.
- MAXCH sentinel plateau 중에는 \(C_{100}\), Si III, terminal 극값이 감소.
- full probe와 T 해제 뒤에도 새 증가 spike 없이 동일 추세 유지.
- 최종적으로 limiter가 발화하지 않는 영역에서 `MAXCH<EPS_TERM`.

**즉시 실패**

- NaN/Inf, 음수/zero population 유발 메시지, CMF/BA solve failure.
- 반환 MAXCH가 한 step에서 직전의 2배 또는 `10^7%`를 초과.
- \(C_{100}\), Si III, terminal 중 하나가 두 step 연속 증가.
- Si III 또는 terminal 극값이 한 step에서 2배 증가.
- full probe가 마지막 LAMBDA 지표보다 악화.
- 패치 사용 시 trust 발화 수가 5 step 동안 감소하지 않거나 trace 발화가 외곽에서 계속 증가.
- exit 137은 물리 실패로 자동 분류하지 않고 자원 실패 가능성으로 별도 중단.

---

## 4. 메모리·노드 배치

relT2는 한 노드, 한 task, 16 OpenMP thread, `--mem=200G`였고 `--exclusive`는 없었다. thread당 stack 설정은 `512M`이었다. [slurm script:4](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/slurm_cmfgen_modern.sh:4), [slurm script:20](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/slurm_cmfgen_modern.sh:20), [slurm script:35](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/slurm_cmfgen_modern.sh:35)

로그에는 단지 `Killed`, exit 137만 있고 OOM 표시는 없다. [Slurm out:1](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/seq_logs/modern1948_slurm-397936.out:1), [batch.log:1](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/batch.log:1)

권고:

- 가능하면 `--mem=256G` 또는 사이트의 다음 단일-node memory tier.
- `--exclusive`로 co-tenant를 제거.
- `--nodes=1 --ntasks=1 --cpus-per-task=16`과 OMP 16은 유지. 기존 script는 다른 thread count가 수치 경로를 깨뜨렸다고 명시한다. [slurm script:20](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/slurm_cmfgen_modern.sh:20)
- 각 job 뒤 `State, ExitCode, MaxRSS, ReqMem, NodeList` accounting을 저장.
- 실제 `MaxRSS`가 충분히 200G 아래면 추가 메모리 증설을 물리 처방으로 해석하지 않는다.

`200G→256G`는 안전 여유 권고이지 OOM 판정이 아니다. kill 주체와 원인은 계속 **UNRESOLVED**다.

---

## 5. 패치 사용 시 심판 독립성 검증

패치가 사용됐다면 최종 채택 전에 반드시 다음 한 번의 stock verification을 수행한다.

1. patched run의 최종 checkpoint와 입력을 immutable 복사하고 hash 기록.
2. 원본 `solveba_v13/fiddle_pop_corrections_v2`로 빌드된 무패치 executable 사용.
3. trust 입력/`ADJUST_CORRECTIONS` 비활성.
4. NG·AV·smoothing·undo는 계속 끈 채 **full great iteration 1회**.
5. 다음을 모두 요구:
   - 반환 `MAXCH<EPS_TERM=0.1%`
   - `C100=0`
   - NaN/Inf 없음
   - Si III/terminal spike 없음
   - stock step 후 population/spectrum 변화가 기존 공식 수렴 허용범위 이내
6. 이 stock step이 생성한 checkpoint를 최종 과학 산출물의 시작점으로 채택.

이는 patched 경로가 stock CMFGEN의 국소 고정점에 도달했다는 검증이다. 다만 서로 다른 basin의 다른 해가 없다는 전역적 유일성까지 증명하지는 못하므로 **전역 경로 독립성은 UNRESOLVED**다.

---

## 최종 권고 순위

1. **P0 네이티브:** modern it40 restart 보존 + `MAJOR` + `MAX_LAM=1.10→1.20` + `MAX_LIN=1.05→1.10` + forced LAMBDA/fixed-T + NG off.
2. 5-step 다중 지표 관문 뒤에만 one-step full probe.
3. fixed-T full 안정화 뒤에만 수동 T 해제; `MAX_dT=0.02→0.05`.
4. 네이티브 실패 또는 감사 로그 필요 시에만 Δln patch.
5. 패치 사용 시 stock full 1회 재확인 필수.
6. 256G급 여유 메모리와 exclusive node는 권고하되, exit 137의 OOM 원인은 **UNRESOLVED**로 유지.

조사 중 CMFGEN 실행·컴파일·파일 수정은 수행하지 않았다.