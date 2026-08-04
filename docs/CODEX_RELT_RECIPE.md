결론부터 말하면, 385770은 “수렴 fixed-T 해에서 FIX_T만 해제한 시험”이 아니었습니다. 실제 초기 체크포인트는 fixed-T에서도 `MAXCH=1.24×10^7%`, 표면광도 `7.70×10^11 L☉`인 강한 limit-cycle 상태였습니다. 반면 현재 `modern`도 정상 종료·finite이기는 하지만 마지막 `MAXCH=3.46×10^3%`이므로 `EPS_TERM=0.1%` 수렴본은 아닙니다. 따라서 추천안은 modern을 먼저 fixed-T로 더 연마한 뒤, CMFGEN의 내장 `<50%` 자동 게이트로 T를 해제하고 `MAX_dT=0.05`로 제한하는 방식입니다.

어떤 파일도 변경하거나 제출하지 않았습니다.

## ① 385770 발산의 유력 기전

근거가 강한 순서입니다.

1. 출발점이 이미 심하게 비수렴이었다.

385770의 실제 seed인 `unconv_stint1_fixT_bak`은 iteration 40에서:

- `MAXCH=1.24×10^7%`
- `L(d=1)=7.700×10^11 L☉`
- 직전 iteration들도 `10^6–10^7%` limit cycle

상태였습니다. [seed OUTGEN](/gpfs/kjhan/cmfgen_runs/unconv_stint1_fixT_bak/OUTGEN:3092)

현재 modern은 훨씬 낫지만 마지막 상태가:

- `MAXCH=3.46×10^3%`
- `L(d=1)=5.805×10^10 L☉`
- `Temperature held fixed at all depths`

입니다. [modern OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:3065)

즉 “완주”는 맞지만 `EPS_TERM=0.1%` 의미의 수렴은 아닙니다.

2. abrupt full release와 대규모 population correction이 결합됐다.

385770 첫 released iteration 41은:

- full BA step
- `+4.52×10^6% / −5.07×10^5%`
- correction scale 최솟값 `7.22×10^-4`
- `L(d=1)=7.842×10^11 L☉`

였습니다. [385770 OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:115)

그 뒤 `MAXCH>LAM_VAL=400`이라 LAMBDA iteration으로 전환됐으며, LAMBDA에서는 소스상 T가 다시 고정됩니다. [solve_for_pops.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:281)

따라서 로그는 “매 iteration T 보정이 무제한 폭주했다”는 해석을 지지하지 않습니다. 직접적인 촉발점은 첫 full released-T BA step과 그에 따른 opacity/population/field 재배치일 가능성이 큽니다.

3. 직접 사망 위치는 T 방정식보다 radiation-field solve 쪽이다.

- iteration 41–46: `DTDR` finite
- iteration 47: 표면 luminosity가 먼저 NaN
- 이어서 전 깊이 `CMF_BLKBAND_V3` 실패
- iteration 49: tau·DTDR까지 NaN, 그 후 GREY 실패

[iteration 47](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:379), [GREY 실패](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:836)

따라서 GREY 실패는 원인이 아니라 이미 NaN이 퍼진 뒤의 후속 증상입니다.

4. 385770은 finite EDDFACTOR도 이어받지 않았다.

시작부에 `Unable to open EDDFACTOR_INFO — will compute new f`가 있습니다. [385770 OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:115)

반면 modern에는 현재 finite run에서 생성된 `SCRTEMP/POINT1/POINT2/EDDFACTOR/JH_AT_CURRENT_TIME`이 모두 있습니다. 일반 restart는 `SCRTEMP`를 읽으며, new-model 경로가 아니면 초기 grey 재생성을 하지 않습니다. [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:908)

결론: 유력 기전은 “나쁜 fixed-T limit cycle에서 전면 T 해제 + 첫 대규모 BA correction + 초기 field/f 재계산”입니다. `MAX_dT` 기본값은 이미 20%였으므로 순수한 무제한 T 보정 폭주라고 단정할 근거는 없습니다.

## 확인된 옵션 의미

- `FIX_T=T`: VARFIXT와 무관하게 전 깊이 T 고정. [solve_for_pops.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:78)
- `FIX_T_AUTO=T`는 내부 변수 `VARFIXT`; `TAU_SCL_T`보다 작은 electron-scattering optical depth의 외곽 깊이를 고정. [rd_control_variables.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/rd_control_variables.f:949), [solve_for_pops.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:91)
- `MAX_dT`: 한 iteration의 최대 fractional T correction. 기본 `0.2`, 허용범위 `0<MAX_dT≤약 0.2`. [rd_control_variables.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/rd_control_variables.f:1009), [solveba_v13.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:311)
- `DO_T_AUTO`는 VADAT가 아니라 `IN_ITS` 옵션. 기본 F. fixed-T, non-LAMBDA 상태에서 `MAXCH<50%`가 되면 코드가 VADAT의 `FIX_T`를 F로 바꿉니다. [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:565), [자동 해제부](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:4712)
- `DO_LAM_IT=T`이면 T가 고정되며, 기본 `DO_LAM_AUTO=T`는 `MAXCH<50%`에서 full iteration으로 전환합니다. [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:603)

## ② 안정화 후보

### A안 — 내장 2단 게이트 자동 해제: 추천

modern checkpoint를 그대로 이어서:

1. pure LAMBDA로 fixed-T population을 `MAXCH<50%`까지 연마
2. CMFGEN이 non-LAMBDA fixed-T 검증 step을 수행
3. 그 step도 `<50%`이면 `DO_T_AUTO`가 `FIX_T=F`로 자동 변경
4. 첫 released step부터 T correction은 5%로 제한

VADAT diff:

```diff
@@
-10.0D0       [MAX_LIN]          !Maximum fractional change allowed for linearization
+3.0D0        [MAX_LIN]          !Released-T Newton population cap

 10.0D0       [MAX_LAM]          !Keep modern fixed-T LAMBDA cap
+5.0D-02      [MAX_dT]           !Maximum fractional T correction = 5%

@@
-T            [DO_NG]            !Use NG acceleration
+F            [DO_NG]            !Disable while crossing the T-release gate
```

`FIX_T=T`, `FIX_T_AUTO=F`, `TAU_SCL_T=0`은 그대로 둡니다. 자동 해제 시 코드가 `FIX_T`만 F로 씁니다.

IN_ITS diff:

```diff
-40           [NUM_ITS]
-F            [DO_LAM_IT]
+60           [NUM_ITS]          !Additional iterations: labels 41..100
+T            [DO_LAM_IT]        !Fixed-T LAMBDA polish first
+T            [DO_LAM_AUTO]      !At MAXCH<50%, take a full fixed-T iteration
+T            [DO_T_AUTO]        !Then release FIX_T only if that full step stays <50%
```

근거:

- 현재 modern의 마지막 네 step은 `4.43e4 → 1.38e4 → 8.98e3 → 3.46e3%`로 하강 중이므로 MAX_LAM=10을 유지할 근거가 있습니다.
- 과거 `MAX_LAM=1.73` deepdamp는 field failure를 막지 못하고 population 연마만 지연시켰으므로 LAMBDA cap을 더 낮추는 것은 1차 레버로 부적절합니다.
- `MAX_LIN=3`은 release 후 full BA step만 제한합니다.
- `MAX_dT=0.05`는 기본 20% 대비 4배 보수적인 운용 선택입니다.

released 상태가 최소 두 번의 full BA iteration 동안 finite이고 `MAXCH<5%`에 들어간 뒤에만 `DO_NG: F→T`를 별도 restart에서 복원합니다.

### B안 — VARFIXT로 외곽을 보호하며 deep-first 해제

modern의 `MEANOPAC`에서 electron-scattering tau는 depth 48에서 `0.085`, depth 49에서 `0.103`입니다. 따라서 `TAU_SCL_T=0.1`은 대략 depth 1–48의 optically thin 외곽을 고정하고 depth 49–90만 먼저 풉니다. [MEANOPAC](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/MEANOPAC:1)

1단계 diff:

```diff
@@ VADAT
-T            [FIX_T]
-F            [FIX_T_AUTO]
-0.0          [TAU_SCL_T]
+F            [FIX_T]
+T            [FIX_T_AUTO]
+1.0D-01      [TAU_SCL_T]        !Hold outer tau_es < 0.1 region

-10.0D0       [MAX_LIN]
+3.0D0        [MAX_LIN]
 10.0D0       [MAX_LAM]
+5.0D-02      [MAX_dT]

-T            [DO_NG]
+F            [DO_NG]
```

```diff
@@ IN_ITS
-40           [NUM_ITS]
+12           [NUM_ITS]          !One inspected partial-release stint
 F            [DO_LAM_IT]
+F            [DO_T_AUTO]
```

첫 12회가 finite이고 두 번의 non-LAMBDA step에서 `MAXCH<50%`이면 2단계:

```diff
-1.0D-01      [TAU_SCL_T]
+1.0D-02      [TAU_SCL_T]        !MEANOPAC상 대략 depth 1..36만 고정
```

다시 같은 gate를 통과하면 전면 해제:

```diff
-T            [FIX_T_AUTO]
-1.0D-02      [TAU_SCL_T]
+F            [FIX_T_AUTO]
+0.0D0        [TAU_SCL_T]
```

장점은 field에 가장 민감한 optically thin 외곽을 마지막까지 고정한다는 점입니다. 단점은 수동 3단계이며, 선택한 tau 경계가 modern의 현재 `MEANOPAC`에 종속된다는 점입니다.

### C안 — 3-iteration short-stint 전면 해제

이 안은 modern을 fixed-T로 먼저 `MAXCH<50%`까지 별도 연마한 뒤에만 사용해야 합니다. 현재 iteration 40 상태에서 바로 적용하는 것은 권하지 않습니다.

```diff
@@ VADAT
-T            [FIX_T]
+F            [FIX_T]

-10.0D0       [MAX_LIN]
+3.0D0        [MAX_LIN]
 10.0D0       [MAX_LAM]
+2.0D-02      [MAX_dT]           !2% T correction cap

-T            [DO_NG]
+F            [DO_NG]
```

```diff
@@ IN_ITS
-40           [NUM_ITS]
+3            [NUM_ITS]          !Three released iterations only
 F            [DO_LAM_IT]
+F            [DO_LAM_AUTO]
+F            [DO_T_AUTO]
```

각 3-iteration stint가 finite일 때만 scratch를 보존해 다시 3회 실행합니다. 두 stint가 안정적이면 `MAX_dT 0.02→0.05`로 완화할 수 있습니다. 가장 보수적이지만 운영 개입이 많습니다.

## ③ 추천안과 중단 기준

추천은 A안입니다. 현재 modern이 아직 `<50%`가 아니므로, “충분히 수렴했다고 가정하고 즉시 release”하지 않고 CMFGEN 자체 기준으로 fixed-T polish → full fixed-T 검증 → release를 순차 적용하기 때문입니다.

조기 경고:

- 자동 release 직후 raw `MAXCH>10^4%`
- correction scale `<10^-2`
- 표면 luminosity가 연속 2–3회 증가
- `MOM_JREL_V9 excessive iteration`가 다수 주파수에서 burst 형태로 증가
- 외곽 tau 또는 `DTDR`가 비정상적으로 급변

즉시 graceful-stop 기준:

- `NaN`, `CMF_BLKBAND_V3`, `Grey solution was NOT`, negative opacity 중 하나라도 발생
- released step에서 `MAXCH≥10^6%` 한 번
- 또는 `MAXCH≥10^5%`가 두 번의 release-capable BA step에서 지속·증가
- `L(d=1)>1.0×10^11 L☉`
- `DTDR` 또는 출력 tau가 nonfinite
- correction scale `<10^-3`이면서 `MAXCH>10^5%`

정지는 실행 중 `IN_ITS`의 `NUM_ITS`를 `0`으로 바꿔 현재 iteration 종료 후 우아하게 끝내는 방식입니다. `kill -9`는 사용하지 않습니다. [런 가이드](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CMFGEN_BUILD_RUN_GUIDE.md:89)

## ④ 복제→편집→sbatch 체크리스트

1. 후보별 독립 clone을 만든다.

```bash
run_src=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern
run_dst=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern_relT_A

test ! -e "$run_dst"
cp -a "$run_src" "$run_dst"
cd "$run_dst"
cp -p OUTGEN OUTGEN.fixedT_it40
cp -p batch.log batch.fixedT_it40.log
```

2. restart 핵심 파일을 확인한다.

```bash
test -s SCRTEMP
test -s POINT1
test -s POINT2
test -s EDDFACTOR
test -s EDDFACTOR_INFO
test -s JH_AT_CURRENT_TIME
strings POINT1 | head
find . -maxdepth 1 -xtype l
```

`POINT1`은 `IREC=NITSF=40`이어야 하고 broken symlink 출력은 0이어야 합니다.

3. VADAT/IN_ITS를 선택안 diff대로 편집하고 키 중복을 검사한다.

```bash
rg -n '\[(FIX_T|FIX_T_AUTO|TAU_SCL_T|MAX_LIN|MAX_LAM|MAX_dT|DO_NG)\]' VADAT
rg -n '\[(NUM_ITS|DO_LAM_IT|DO_LAM_AUTO|DO_T_AUTO)\]' IN_ITS
```

4. 기존 slurm script를 그대로 실행하면 안 된다.

현재 script의 37–40행은 `SCRTEMP`, `POINT*`, `EDDFACTOR*` 등을 삭제해 cold start를 강제합니다. [기존 launcher](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/slurm_cmfgen_modern.sh:37)

복제한 launcher에서 반드시:

- `#SBATCH --output`을 새 디렉터리로 변경
- `DIR=`을 새 clone으로 변경
- 37–40행의 전체 `rm -f ...` 블록 제거
- 대신 위 restart 파일에 대한 `test -s ... || exit 2` 추가
- 다음 설정 유지

```bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

export OMP_NUM_THREADS=16
export OMP_STACKSIZE=512M
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

5. 제출한다.

```bash
ssh grammar \
  'cd /gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern_relT_A &&
   sbatch slurm_cmfgen_relT_A.sh'
```

6. 시작 직후 restart 여부를 확인한다.

- 첫 새 iteration 번호가 `41`이어야 함
- 새 구간에 `Starting a new model`이 없어야 함
- `Error opening POINT1/POINT2`가 없어야 함
- `Unable to open EDDFACTOR_INFO`가 없어야 함
- job/run info의 OMP가 정확히 16이어야 함
- A안에서는 처음에는 `FIX_T=T`; `<50%` gate 통과 후 코드가 VADAT를 `FIX_T=F`로 바꾸는 것이 정상

동일 scratch clone을 가리키는 두 job을 동시에 제출하면 안 됩니다. 후보 비교가 필요하면 반드시 A/B/C 각각 별도 `cp -a` clone을 사용해야 합니다.