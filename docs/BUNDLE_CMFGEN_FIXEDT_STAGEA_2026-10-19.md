# CMFGEN 고정온도 Stage A 묶음 — 2026-10-19

## 목적과 경계

`toy06_19p48d_modern`의 iteration 40 상태에서 새롭고 격리된 CMFGEN 고정온도 연속 계산을 수행해,
full-linearization에서 Si II/III·S population correction과 율 자료가 어떻게 변하는지 판별한다.
이 단계는 **진단용 staging**이다. fixed-T 결과는 oracle, 물리 기준해, 최종 CMFGEN 동질성 PASS가 아니다.
전체 목표는 CMFGEN과 Si II/III 및 S 분포의 물리 동질성을 새 자기일관 계산으로 입증하는 것이며,
중간 단계의 계산·파서 통과로 완료를 선언하지 않는다.

범위는 동일 원자자료·모형·깊이 구조의 복제본에서 full-linearization 19회까지 계산하고,
population correction·상태 변화·최종 반복 율 자료를 보존하는 것이다. 온도 release, CMFGEN 소스 변경,
원자자료 수정, Lumina 실행 및 공개 2일 time-dependent 결과와의 최종 비교는 제외한다.
자동 수치 재시도는 없다. 기존 seed는 불안정한 시작 추정값일 뿐 생성 당시 바이너리 해시가 없으므로,
이를 정답 또는 검증 오라클로 사용하지 않는다.

## 입력·변경·담당

- Seed: `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern`; 기준 상태 포인터 `POINT1/POINT2`는 40/40,
  `SCRTEMP`, `STEQ_VALS`, `RVTJ`, `EDDFACTOR`, `CUR_MODEL_DATA` 등 정확한 입력 해시는 실행 시 기록한다.
- 모델/원자 선택: `MODEL`, `MODEL_SPEC`, `LEVEL_SL_STEQ_LINKS`, 원 run의 `PROVENANCE.txt`,
  `atomic_links.txt`, `setup_links.sh`; 원자자료는 기존 선택 및 run-local repair 두 건을 유지한다.
- 새 run root는 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_<jobid>`로 생성한다.
  기존 run root를 수정하지 않고, stage 복제본에서만 `FIX_T=T`, `FIX_T_AUTO=F`, `DO_T_AUTO=F`,
  steady state `DO_DDT=F`, `TRT_NON_TE=F`, `DO_NG=F`, `MAX_LIN=3.0`, `MAX_LAM=10.0`,
  `DO_LAM_IT=F`, `DO_LAM_AUTO=F`, `NUM_ITS=19`를 적용한다. `INS_DEP=T` 등 나머지
  deposition·경계 물리는 seed 설정에서 변경하지 않고 입력 해시로 봉인한다.
  마지막 반복 율/복사장 증거를 보존하기 위해 `WRITE_RATES=T`, `WRITE_JH=T`를 켠다.
- CMFGEN 실행 바이너리 SHA256은 현재 `f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1`,
  CMFGEN 소스 HEAD는 `dd1660af039e10f8f8c61d892c4e226e1cd6438c`다. 바이너리 자체를 실행 시 다시 해시한다.
- 실행·코딩·직접시험·증거 재현은 운전자가 담당한다. 계획 및 물리 평가 배정은 기존 사용자 지시를 따른다.

Astra의 read-only 계획검토는 seed 포인터·차원/격자 확인, steady-state 제어 고정,
복사장·기여도 자료의 반복별 보존, timeout과 수치 실패의 구별을 요구했다. 이 묶음은 iteration-40
parser preflight와 `POINT1/2`, `EDDFACTOR_INFO` 확인을 포함하고 `DO_DDT=F`, `TRT_NON_TE=F`를
강제한다. 마지막 반복별 `Jν`와 전체 activity 합집합은 아직 복원되지 않았음을 산출물에 명시해
과대판정을 막는다. 이 단계에서 다음 T-release 실행을 자동 승인하지 않는다.

## 판정 기준·증거

계산 산출물은 최대 19회 fixed-T iteration, 반복별 `SCRTEMP`, `STEQ_VALS`, `OUTGEN`(iteration별
luminosity와 correction 포함), 최종 `RVTJ`, `EDDFACTOR`, `NETRATE`, `TOTRATE`, `LINEHEAT`,
`JH_AT_CURRENT_TIME`, population correction/ion-fraction active-subset 지표 및 모든 입력·출력 해시다.
추가 판독 산출물은 저장된 각 iteration·각 depth에서 모델에 포함된 Si/S 이온단계 전부의
population 합과 원소 내 분율이다. 깊이 좌표는 `RVTJ` 속도(km/s), 반경은
`SCRTEMP`의 `10^10 cm` 단위를 쓰며 최종 `SCRTEMP`–`RVTJ` 상태 일치가 확인되지 않으면
추출을 거부한다. `terminal closure` 변수도 버리지 않고 포함 여부를 표시한다. 이는 비교용
증거 추출이며 별도 물리 합격 게이트는 아니다.
현재 binary는 `Jν` full spectrum을 iteration별 별도 파일로 내보내지 않으므로, 마지막 세 iteration의
`Jν` 안정성은 이 묶음에서 `NOT MEASURED`로 남긴다. 전체 active-set 기여도 합집합 역시 별도 분석기
보완 전까지 미확정이다. 둘 중 하나도 ion-only subset 결과로 추정하지 않으며, 따라서 자동 T-release는
없고 이후 단계의 시작 조건으로 사용할 수 없다. Stage A의 완주 자체는 물리 안정화 PASS가 아니다.

NaN/Inf, moment solver 오류, 종료코드 실패 또는 필수 최종 산출물 누락은 실행 실패로 판정한다.
시간제한 종료 또는 마지막 상태 판정에 필요한 증거 누락은 `INCONCLUSIVE`로 남기고 자동 재시도하지 않는다.
기존 parent는 그대로 보존한다.

계산 종료만으로 Stage A 물리 준비도를 PASS하지 않는다. 다음 단계에 대한 물리 평가는 v2 기준의
전체 active-set 합집합(이온분율 `>1e-8` 또는 율·열률·불투명도 기여 `>1e-4`)이 필요하다.
정량 참고 문턱은 active correction p95 `<0.1%`, max `<1%`, 마지막 3회 `T_e,n_e,Jν,L` 변화 `<0.5%`,
moment solver error 0건 및 hidden clamp/NaN/Inf 0건이다. 고정온도 단계에서 RE 온도행이 계산되지 않는
사실은 RE 수렴 증거가 아니다. `MAXCH<50%` 단독은 전환 또는 물리 합격 기준으로 쓰지 않는다.
율/기여도 기반 activity 계측이 아직 전체 구현되지 않은 경우, 낮은 ion-only subset 결과로 release를
승인하지 않고 해당 누락을 미해결로 남긴다.

## 자원·보존·재현

- Slurm: `a10`, CPU 16, 메모리 200 GiB, 14시간, `gpu:0`(CPU-only CMFGEN; GPU 자원 미점유), 노드 고정 없음.
- 저장공간 예산은 8 GiB로 둔다. 기존 동급 O-PHYS의 `NETRATE/TOTRATE/LINEHEAT` 합계가 약 4.75 GiB였고,
  복제 restart와 새 반복 상태에 여유를 더한다. 실행 전 확인에서 GPFS `/gpfs` 가용 공간은 257 TiB였고,
  사용자 hard quota는 설정되지 않은 상태였다.
- 실행 전 입력 해시, 실행 바이너리 및 source HEAD, 모델 제어값, 원자 링크와 그 대상 해시를 기록한다.
- parent 및 모든 기존 출력은 불변 보존한다. 새 run은 job ID별 경로만 사용한다. 실패 partial output도 원인 확인 전 삭제하지 않는다.
- 계획 재현: `sbatch scripts/submit_cmfgen_stageA_fixedT.slurm`
- 결과 재현: `/home/kjhan/miniconda3/bin/python3 scripts/ptf_gated_metrics.py <stageA-run-root> --state-first-it 1 --from-it 41 --format json --output <new-isolated-output-dir>/reproduced.json` (`ptf_gated_metrics.py`는 입력 run root 내부 쓰기를 거부한다.)
- 모든 저장 반복의 Si/S 이온분율 추출: `/home/kjhan/miniconda3/bin/python3 scripts/export_cmfgen_ion_fractions.py <stageA-run-root> --output-csv <stageA-run-root>/metrics/si_s_ion_fractions.csv --output-json <stageA-run-root>/metrics/si_s_ion_fractions.json`
- job 상태 확인: `squeue -j <jobid>` 및 종료 후 `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode`.

## 2026-10-19 운전자 진행 중 기록

- Slurm job `427702`은 현재 `a10` 파티션 `syn01`에서 실행 중이다. 요청 자원은 CPU 16, 메모리 200 GiB, GPU 0, 최대 14시간이며 새 run root는 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702`이다. 실행 바이너리 SHA256과 CMFGEN source HEAD는 계획에 봉인한 값과 일치한다. parent `toy06_19p48d_modern` 및 별도 기존 출력은 수정하지 않았다.
- 실행 시점 확인에서 restart 포인터는 40/40, ND=90이었고 iteration-40 seed 사전진단은 통과했으나, 전체 population correction p95는 `95.5462%`, max는 `3460.3%`였다. seed는 불안정한 초기 추정값이며 oracle이 아니다.
- job 시작 뒤 확인한 포인터는 42/42였고 CMFGEN은 43회차를 수행 중이었다. 42회차 로그에는 최대 증가 `2.48e5%`, `SOLVEBA_V13` 반환 `1e7%`가 기록됐다. 종료 산출물은 아직 없으며 수렴·물리 PASS가 아니다.
- 소스 read-only 대조(`new_main/cmfgen_sub.f`의 LAMBDA 제어부, `new_main/mod_subs/solve_for_pops.f`) 결과, `IN_ITS`의 `DO_LAM_IT=F`는 강제 LAMBDA 시작값 `RD_LAMBDA`를 끄지만, CMFGEN은 큰 correction에서 자체적으로 LAMBDA fallback을 선택한다(특히 `MAXCH>1e5%`일 때 강제). 따라서 로그의 `LAMBDA iteration used`는 현재 복제본의 입력과 모순된다고 단정할 수 없다. 원 계획의 “19회 full-linearization” 표현은 실제 제어 의미보다 강했으며, 이 실행은 fixed-T의 CMFGEN native adaptive correction staging으로 해석하고, 실행 중인 job은 취소·변경하지 않는다.
- Stage A가 끝나면 실제 저장 iteration·입출력 세대 일관성과 active-set 수렴 증거를 판독한다. fixed-T staging 완료만으로 온도 release나 Si/S 동질성을 승인하지 않는다.

## 2026-10-19 live-step 회계 갱신

- job `427702`와 step `427702.0`은 `RUNNING`, `cmfgen_dev.exe`, `syn01`이다. elapsed `03:52:39`, step 누적 CPU `1-13:21:10`, MaxRSS `97,819,832K`(약 93.3 GiB), MaxVMSize `106,795,896K`였다. batch step만 보면 CPU 0/RSS 12M이었지만, 본 계산 step 회계는 실제 계산을 확인한다.
- `seq_logs/cmfgen.stdout`과 `.stderr`가 아직 0 byte이고 `metrics/`는 비어 있다. 출력 미생성만으로 멈춤으로 판단하지 않으며 재시작·취소하지 않는다. 완료 후 반복 포인터·최종 출력 세대·active-set 증거를 검사하기 전까지 수렴 및 물리 판정은 보류한다.

## 2026-10-19 terminal-DI 동일세대 추출 준비

- 마지막 `squeue` 확인: job `427702` `RUNNING`, `syn01`, elapsed `04:45:59 / 14:00:00`. 기존 job은 유지하고 재제출하지 않았다. 격리 run root에서 `SCRTEMP`, `RVTJ`, `OUTGEN` 존재를 확인했으나 완료 출력은 아직 검증하지 않았다.
- Stage A V단 `PHOTSkV_A`와 `PHOTSV_A`는 각각 source photoionization route 1개를 포함한다. SHA256은 `8d6609658da76f99b8aee903e852b4684dd8c47f92efcb81185e356f9bcbc263`, `664aa143032728a4b28d32ed0fa6beeeef883c1d765b4c14e102b28e000edaca`이며 `seq_logs/linked-inputs.sha256`와 일치한다.
- source 확인상 `STEQ_MULTI_V10`의 terminal `DI_S`는 conservation/charge equation에 들어가고, `SUM_STEQ_SOL`은 추가 변수를 다음 이온의 ground state로 표기한다. final `POP{species}` 출력은 `POP_SPECIES`, active ion `XzV_F`, `DXzV_F`를 기록한다. 코드 exporter는 계산 종료 후 독립 원소총수와 terminal `DXzV_F`/`SCRTEMP` 행을 대조하도록 보강했으며, 그 출력은 아직 준비되지 않았다.
- exporter 회귀(`tests/test_export_cmfgen_ion_fractions.py`, `tests/test_ptf_gated_metrics.py`, `tests/test_configure_cmfgen_stageA.py`) `15 passed`. 이것은 추출기 시험일 뿐 현재 CMFGEN run의 수렴 또는 Lumina 동질성 판정이 아니다. V→VI terminal rate 경로와 Lumina owner 연결은 계속 미해결이며 물리 run은 아직 금지 상태다.

## 2026-10-19 사후 산출물·계측기 조사

- job `427702`의 현재 Slurm 상태는 확인되지 않았다. `squeue -j 427702`는 invalid job ID를 반환했고, `sacct -j 427702`에는 job과 연결 근거가 없는 `422019_548`만 나왔다. 이를 `427702`의 완료로 간주하지 않는다. 기존 Stage A run root와 parent는 수정·삭제하지 않았다.
- `SCRTEMP`는 ND=90, NT=1800, 50개 상태이고, `STEQ_VALS`는 50개 배열이다. 실제 `OUTGEN`에는 iteration 41–50의 10개 correction 기록만 있다. 앞서 계측기는 전체 50개 배열을 41–90으로 잘못 배정해 iteration 41에서 `STEQ/OUTGEN` 불일치를 냈다. 가장 최근 10개 `STEQ` 배열을 41–50에 대응하면 증가·감소 correction 모두 10회 전부 OUTGEN의 3자리 반올림 범위와 일치했다.
- 이 증거에 따라 `scripts/ptf_gated_metrics.py`가 OUTGEN의 iteration ID를 최신 대응 STEQ 배열 suffix에 매핑하고, 매핑되지 않은 prefix 개수를 별도 기록하도록 수정했다. 회귀시험 `tests.test_ptf_gated_metrics` 9개 통과, 대상 파일 `git diff --check` 통과. 이는 기존 CMFGEN 출력 해석기의 수정이며 물리모델 변경은 아니다.
- iteration 41–50만 선택하고 `--state-first-it 1`로 재현한 P-TF 판독은 STEQ/OUTGEN 확인을 통과한 뒤 final `RVTJ` 일관성 검사에서 중단됐다. `RVTJ` 전자밀도는 `SCRTEMP` state 40과 최대 상대오차 `3.36e-8`로 일치하지만, state 50과는 `0.131057`(13.1057%) 차이다. 온도는 state 50과 `4.20e-8`, 반경 앵커는 `4.34e-11` 오차다. 따라서 현재 `RVTJ`는 최종 state 50이 아니라 restart 시점 state 40과 일치하며, 서로 다른 출력 세대가 섞여 있다.
- 최종 출력 후보 `POP*`, `NETRATE`, `TOTRATE`, `LINEHEAT`, `JH_AT_CURRENT_TIME`는 run root에서 발견되지 않았다. Stage A가 iteration 51 도중 끝난 듯한 `OUTGEN` 꼬리 기록도 있으나, job accounting으로 종료 원인을 입증하지 못했다. 저장된 이온분율/최종 population 추출 JSON은 생성되지 않았고 격리 재현 디렉터리에도 결과 JSON은 없다.
- **운전자 자체 판정: Stage A `INCONCLUSIVE`; 종료 PASS 아님.** 서로 세대가 일치하는 최종 물리량과 수렴 증거가 없어 CMFGEN oracle로 쓸 수 없다. Si II/III·S 동질성이나 Lumina 물리 결과는 이번 산출물로 판정하지 않았다. 기존 partial raw output은 보존한다.
- 다음은 새 대규모 실행이 아니라, 현재 run의 마지막 iteration과 종료 원인을 식별할 수 있는 원본 Slurm/runner 증거를 read-only로 추적하는 것이다. 새 실행이나 수용기준 변경은 별도 묶음 계획·평가 없이 시작하지 않는다.

## 2026-10-19 runner 종료 경로 추적

- 제출 스크립트는 `srun` 반환 뒤 `seq_logs/stageA-runtime.txt`에 `cmfgen_exit_code`를 덧붙이고 `batch.log`에 `CMFGEN_EXIT`를 쓰도록 되어 있다. 실제 runtime 기록은 `run_start_epoch=1792338672`에서 끝나며 종료 코드는 없다. run root의 `batch.log`도 없다. 따라서 이 실행은 정상적인 runner 후처리 구간에 도달하지 않았다.
- `seq_logs/cmfgen.stdout`과 `seq_logs/cmfgen.stderr`는 각각 0 byte이며 Slurm의 `stageA-fixedT-427702.out/.err`도 각각 0 byte다. 이 파일들에는 종료 신호·오류 원인이 기록되어 있지 않다.
- 입력/출력 세대 혼합은 파일 시각으로도 확인된다. `RVTJ`는 2026-07-30 09:37, `SCRTEMP`와 `STEQ_VALS`는 2026-10-19 06:40, `OUTGEN`은 2026-10-19 06:42에 수정됐다. 이는 앞서 관측한 state-40 RVTJ 대 state-50 SCRTEMP 불일치와 일치한다.
- 확인 당시 `squeue`에는 job `427702`가 없었고 `sacct`에서 해당 ID를 귀속할 수 있는 accounting 행도 찾지 못했다. 다른 job ID의 기록을 이 실행으로 간주하지 않는다. 종료 원인은 증거 부족으로 **미상**이며 OOM, 시간제한, 노드 장애 중 어느 것으로도 단정하지 않는다.
- **갱신 판정: `INCONCLUSIVE` 유지.** 정상 후처리 미도달은 확인했으나 외부 종료 원인은 식별되지 않았다. 원본 partial output은 그대로 보존하며 자동 재제출하지 않는다.

## 2026-10-19 12:17 최종 출력 확인과 사후 재판독

- 위 partial-output 관측 이후 권위 상태가 바뀌었다. 현재 `POINT1/POINT2=59/59`, `SCRTEMP` 59개 상태, `OUTGEN` 41–59회 19개 correction과 최종 `POPSIL/POPSUL/PRRR/RVTJ`가 존재한다. 출력 완료 시각은 12:17 KST이며 `seq_logs/stageA-runtime.txt`에 `cmfgen_exit_code=0`이 추가됐다. 따라서 CMFGEN 본 계산은 19회 continuation을 정상 종료했다. `squeue`에는 현재 해당 계산이 없고, `syn01` 직접 프로세스 조회는 활성 Slurm 할당이 없어 `pam_slurm_adopt`가 거부했으므로 우회하지 않았다. 정상 종료 판단은 실제 포인터·출력·실행 footer에 근거한다. 과거 타 job의 accounting을 이 실행에 귀속하지 않는다.
- 수정된 현재 `ptf_gated_metrics.py`로 41–59회를 다시 판독한 결과를 `/gpfs/kjhan/cmfgen_runs/stageA_427702_recheck_20261019-86Yyb7/metrics.json`에 보존했다. STEQ/OUTGEN 검증과 최종 RVTJ↔state59 매핑을 통과했다. RVTJ 상대차는 전자밀도 `4.66e-8`, 온도 `4.20e-8`, 반경 `4.34e-11`이다. 현재는 이전 state40 RVTJ와 state50 SCRTEMP를 섞던 partial 출력 상태가 아니다.
- **물리 안정화 판정은 실패**다. 최종59회 ion-fraction active subset correction p95 `79.2876%`, max `9348.2%`, 마지막 전자밀도 최대 `|Δln n_e|=.0188877`, depth27 Si III 분율 상대변화 `+26.1361%`, population-weighted step 최대 `.0941167`였다. 마지막 값들이 안정화 조건을 위반하므로 전체 activity 합집합 계측의 추가 여부와 무관하게 T-release를 정당화하지 못한다. 정상 종료와 oracle 자격을 구분하며, fixed-T 출력은 최종 물리 oracle로 인정하지 않는다.
- `export_cmfgen_ion_fractions.py`의 실제 자료 연결에서도 결함을 수정했다. 기존 시험은 단축 파일명 `POPSi/POPS`와 같은 헤더만 사용했지만 실제 CMFGEN은 `POPSIL/POPSUL` 및 `SIL/He`, `SUL/He`를 쓴다. 두 명시적 형식을 처리하고 실제 선택한 파일들을 입력 hash에 기록하도록 수정했다. 같은 원소 후보 파일이 복수이면 모호한 세대 선택을 하지 않는다. 실제 이름 회귀를 추가했으며 exporter/P-TF 시험 `14 passed`였다.
- 실제 1–59회·90깊이·Si/S 전체 이온단계와 terminal VI를 포함한 `53,100`개 분율 행을 같은 격리 경로의 `si_s_ion_fractions.csv`와 `.json`으로 추출했다. 원소총수와 terminal DI는 실제 POP 출력으로 독립 대조했다. CSV SHA256 `fc1f9f4581076e5d3ae49796c2b2c3a64f400b2547b6e1d69abd99c8325987f8`. 원본 run은 수정하지 않았다.
- 별도 Lumina Si/S 묶음에서는 PHOT 평활화 누락과 target SL 열적 분배함수 누락을 수정해 기존 modern 공통상태의 정/역 BF 율 27,180개 셀 수치 기준을 모두 통과했다. 이 수리와 Stage A 59회 완료를 합쳐도 수렴 oracle 및 새 자기일관 이온분포 비교는 아직 미완료다. 다음 계산의 시작 checkpoint와 수치 제어를 현재 비수렴 증거에 근거해 정한다.
