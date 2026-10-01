# 사전등록 — DET-W5-ITER: W5 조건의 반복 확장에 따른 인구비 착지와 공개값 간극 (2026-10-01, 기안)

기안자 = astra, 개정25-r의 이번 직접 발주에 따른 **단계 설계·사전등록 기안**. 판정·구현·발주·실행 승인이 아니다. 파일 수정 없이 본문을 제출한다. 운전자 파일화 목적지: `docs/PREREG_DET_W5_ITER_2026-10-01.md` — **신설**.

## §0. 선행 사실·설계 판단

### 설계 판단

- **N=15를 택한다.** W5의 동일 초기 조건에서 INIT와 논리 반복 k=0–14를 실행한다. 허용된 단일 런 안에서 추가 궤적을 가장 길게 확보한다. 15회 안의 수렴을 보장하거나 예상 효과 크기로 정당화하지 않는다.
- **처음부터 실행한다.** 현재 생산 DET 호출경로는 상태 초기화와 INIT를 거치며, run1의 최종 CSV에서 전체 상태를 복원하는 warm restart 경로가 확인되지 않았다. restart 구현은 이번 범위에 없다.
- **생산 src 편집·재빌드는 없다.** W5 바이너리를 그대로 사용한다. 물리·알고리듬 exports의 변경은 `LUMINA_PURE_CMFGEN_ITER="6"→"15"` 하나다.
- **러너의 반복 의존 항목을 함께 고친다.** ITER 검사뿐 아니라 footer, FIXED_TE 소비 횟수 검사, 오래된 자체시험과 스테이징 메타데이터를 새 계약에 맞춘다.
- **기존 판독기를 확장한다.** 신규 런 15회와 W5 run1·C1의 6회를 분리한다. 기존 MOVE/NONE 판정은 보존하고, 이번 단에는 수렴·착지·간극 판독을 명시적으로 선택하는 모드를 추가한다.
- **선 원장은 모든 호출에 유지한다.** 기존 바이너리의 출력을 그대로 받아 코드·env 변경을 피한다. 예상 약 0.33 GB이므로 기존 파일별 RLIMIT_FSIZE 1 GiB를 유지한다.
- **G-ID에 첫 6회 재현을 포함한다.** 기존 인구·셸·스펙트럼 CSV를 비교한다. 추가 GPU 계산은 없으며, 파일 판독 비용만 든다.
- **수렴과 공개값 근접성을 분리한다.** 등록한 각 셸·권위의 마지막 세 번 변화량으로 운영상 수렴을 판단하고, 착지값과 공개값의 간극을 별도로 보고한다. 외곽 표류는 미수렴 결과이며 자동 수치 실패가 아니다.

### user 결정과 별도 결정 필요 항목

**확정:** user 결정 2026-10-01 「1」에 따라 W5 유지와 12–15회 범위의 반복 확장 런 한 번을 설계한다. 본문은 그 범위에서 N=15를 선택한다.

**현재 기안을 완성하기 위해 남은 user 선택은 없다.** 다음은 발생할 경우 별도 결정이 필요한 항목이다.

- 이번 런 뒤 추가 반복·재실행·OFF 대조 런.
- §3 자원 상한의 확대 또는 다른 GPU 종류로의 전환.
- restart, 물리 src, damping, 내부 solve 허용오차, 인구 권위의 변경.
- 선행 종료감사에서 제기된 문제를 해결하기 위해 위 범위를 넘어서는 변경.

선행 종료감사는 완료된 것으로 간주하지 않는다. 조회 시점의 종료감사 산출 파일은 0 B였다. 결과가 도착하면 운전자가 본 계약의 신원·수치 유효성·주장 상한에 영향을 주는지 확인한다. 영향을 주는 차단 사항은 제출 전에 재기안하며, 단순히 감사가 진행 중이라는 이유로 가역적 준비를 중단하지 않는다.

### 경로와 증거 정체성

```text
P = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
T = /gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source
S = /gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src
W = /gpfs/kjhan/lumina/cmfgen_map/w5_20260930
B = /gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1
C = /gpfs/kjhan/lumina/cmfgen_map/fixall_20260928/runC1
F = /gpfs/kjhan/lumina/cmfgen_map/w5_20260930/impl/readout_demo/referents
U = /gpfs/kjhan/lumina/cmfgen_map/w5_iter_20261001
R = /gpfs/kjhan/lumina/cmfgen_map/w5_iter_20261001/run1
```

U와 R은 **신설 산출 경로**다. 이미 존재하면 덮어쓰지 않고 운전자가 충돌 없는 경로와 대응표를 집행 기록에 확정한다. P와 T는 서로 다른 트리다.

| 확인 대상 | 확인 결과 |
|---|---|
| 프로젝트·역할 | **[저자 실행]** `pwd`로 P 확인. 공유 문맥 전문, `CLAUDE.md`, 선행 사전등록과 판정문 판독. 이번 직접 발주에 따라 기안만 수행한다. |
| 선행 판정 | **[저자 실행]** 판정문과 W/readout/result.json에 W5-NONE, M=`0.16440303084627872`, 게이트 통과가 기록돼 있다. 이는 기록 확인이며 이번 기안자의 재판정이 아니다. |
| run1 시간·종료 | **[저자 실행]** B/RUN_FOOTER.txt: job 409099, H100 두 장, CPU 16, `outer_iterations=6`, `elapsed_seconds=39269`, `terminal_state=SUCCESS`. |
| 실제 GPU | **[저자 실행]** B/gpu_identity.csv: H100 NVL 두 장, 각 95,830 MiB, driver 580.82.07. |
| 바이너리 신원 | **[저자 실행]** footer와 B/input/binary.sha256의 기록은 `7800484fb309addabfa5b2f6b879f3ab09b814db83d3c5db2f39debff3f1b390`. **[미실행]** 바이너리 실물 전체 재해시. |
| 동결 소스 | **[저자 실행]** S/lumina_cmfgen.c 실물 해시는 `fafb396140555d334d8e089de3d65f69882676d8f085072c9210227cb9a60838`, S/env_universe.h는 `62cf0397564889a6149694814bd2fccc6bc09d173ff3bbb1f4c6a110c45c87f5`. APPROVED_SHA 기록과 일치한다. 전체 빌드 manifest 대조는 **[미실행]**이다. |
| 기존 판독기 | **[저자 실행]** P/scripts/w5_cont_readout.py 해시는 `7008f709e4ca49a24823e4f88f452c42448247d8cb5940c1c1670c4e31a7f3cd`, W/JUDGE_FROZEN.txt와 일치한다. |
| 러너 제약 | **[저자 실행]** B/job.slurm: ITER==6 검사, footer의 6, FIXED_TE 소비 횟수 6. 오류 문자열은 `STRICT_TWO_ITERATION_CONTRACT_MISMATCH`이며 실제 검사값을 설명하지 못한다. 자체시험에는 ITER=2 등 이전 상수가 남아 있다. |
| 보조 메타데이터 | **[저자 실행]** B/input/outer_iterations.txt는 2다. 실제 DET 반복 수의 근거는 exports·footer·산출이다. 새 런에 이 값을 그대로 승계하지 않는다. |
| restart | **[저자 실행]** S/lumina_cuda.cu의 transactional 분기와 S/lumina_cmfgen.c의 `cmfgen_run`을 확인했다. 새 `CMFGENState`, radiation generation 0, pass 0 INIT부터 시작하며 이 경로에 checkpoint 복원 입력이 없다. |
| 선 원장 | **[저자 실행]** B/work/w5_lines.csv 실물 크기 144,129,422 B. 기존 판독 기록은 호출당 103,585행, 호출 1–7, 합계 725,095행이다. 행 수는 이번에 전량 재집계하지 않았다. |
| 고정 referent | **[저자 실행]** F의 공개값·선 manifest·신원 기록을 확인했다. 선택 선 400개, 빈 묶음 20개이며 런타임 입력이 아니다. |
| 선행 종료감사 | **[저자 실행]** W/astra_closeout_r1.md 조회 시점 0 B. 감사 결론을 인용하지 않는다. |
| 새 실행·검증 | **[미실행]** 구현, 자체시험, 기계 프리플라이트, 스테이징, GPU 실행, 새 판독. |

조회 재현 명령은 다음과 같다. 위 약칭을 절대경로로 설정한 뒤 사용한다.

```bash
pwd
cat /home/kjhan/.codex/SHARED_CONTEXT.md
cat CLAUDE.md
cat docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md
cat docs/VERDICT_DET_W5_CONT_SOURCE_2026-10-01.md
cat "$B/RUN_FOOTER.txt" "$B/input/binary.sha256"
cat "$B/input/resolved_lumina.exports" "$B/gpu_identity.csv"
cat "$W/APPROVED_SHA.txt" "$W/JUDGE_FROZEN.txt"
rg -n 'ITER|outer_iterations|TE_PUBLICATION|ulimit|SBATCH' "$B/job.slurm"
rg -n 'N_ITER|range\(N_ITER|4, 5|0, 5|3, 5' scripts/w5_cont_readout.py
rg -n 'EXACT-MULTIGPU-TIMING' "$B/stderr.log"
sed -n '8055,8120p' "$S/lumina_cuda.cu"
sed -n '9184,9410p;9440,9495p;9590,9630p' "$S/lumina_cmfgen.c"
sha256sum scripts/w5_cont_readout.py "$S/lumina_cmfgen.c" "$S/env_universe.h"
stat -c '%s %n' "$B/work/w5_lines.csv" "$W/astra_closeout_r1.md"
```

## §1. 물리 계약 하나

**질문:** W5를 유지한 동일 초기 조건·FIXED_TE·수치 설정에서 반복을 15회까지 늘렸을 때, 등록 셸의 S/Si/Ca III/II 인구비는 운영상 수렴하는가? 수렴한 경우 어디에 착지하며 공개 CMFGEN 값과 얼마만큼 떨어져 있는가?

이 단에서 변경하는 계산 조건은 **외부 반복 수 하나**다. W5의 연속 원천 처방, fine 흡수계수, bin/fine 내부 반복 설정, damping, 원자자료, W2 인구 권위, 온도 고정, 경계와 파장 지원은 유지한다.

성과는 다음 결정을 가능하게 하는 완결된 판독이다.

- 현재 W5 조건의 착지값을 후속 기준으로 사용할 수 있는가.
- 공개 간극이 반복 부족만으로 설명될 여지가 남는가.
- 미수렴이라면 어떤 셸·권위가 어느 방향과 속도로 계속 움직이는가.
- 수치·신원 실패라면 어느 증거를 사용할 수 없는가.

**수렴하지 않은 결과도 유효한 측정 결과다.** 다만 그 결과를 착지값 확보 또는 수렴해 획득으로 바꿔 부르지 않는다.

## §2. 설계

### 2.1 N 선택과 시간 근거

N=15, INIT 별도, 논리 반복 k=0–14로 고정한다. 결과를 보고 12회에서 정상 종료하거나 15회 뒤 계속 실행하지 않는다. 정상 종료 전의 중단은 기존 수치 실패·자원 한계·신원 문제에 따른 경우로 한정한다.

run1 stderr의 `caller_total_s` 실측은 다음과 같다.

| fine solve 호출 | 시간, 초 |
|---|---:|
| INIT | 2,666.817 |
| k0 | 2,890.688 |
| k1 | 3,057.736 |
| k2 | 3,083.567 |
| k3 | 3,103.224 |
| k4 | 3,101.428 |
| k5 | 3,105.962 |

이 값은 **fine solve 호출 시간**이다. 물질 갱신·bin 처리·출력 등을 포함한 외부 반복 전체 시간으로 쓰지 않는다.

전체 모델 경과시간은 footer의 39,269초, 약 10.908시간이다. 이를 여섯 외부 반복으로 나눈 **초기화 비용 포함 평균**은 6,544.8초/반복, 약 1.818시간/반복이다. 로그에는 각 외부 반복 전체의 독립 경계 시간이 없으므로 이 평균을 개별 반복 실측으로 가장하지 않는다.

같은 처리율의 단순 예산 환산은 다음과 같다.

| 선택 | 예상 모델 경과시간 | 두 GPU의 합산 시간 |
|---|---:|---:|
| N=12 | 약 21.8 h | 약 43.6 GPU·h |
| **N=15** | **약 27.3 h** | **약 54.5 GPU·h** |

15회는 12회보다 약 5.45시간을 더 사용한다. 단일 런만 허용된 상황에서 마지막 궤적을 추가로 확보하는 선택이다. 후반부 내부 solve 비용이 증가할 수 있으므로 실행 제한은 §3의 36시간으로 둔다. 이 추정은 수렴 보장이나 검정력 계산이 아니다.

### 2.2 처음부터 실행과 restart

현재 경로는 다음 순서다.

```text
동일 봉인 입력 → bootstrap 상태 → cmfgen_init
→ INIT radiation generation 1
→ k0 radiation generation 2
→ …
→ k14 radiation generation 16
→ 최종 publication
```

run1 최종 publication은 인구·온도 등의 산출이지, 다음 반복을 정확히 잇는 전체 상태 checkpoint가 아니다. `CMFGENState`의 복사장·반복 상태, 물질·복사 세대, 관련 소유 상태를 복원하는 진입점도 이번에 확인한 생산 DET 경로에는 없다. fine solve의 내부 warm start는 프로세스 간 restart와 다르다.

| 방법 | 평가 |
|---|---|
| **동일 초기 조건에서 15회** | 기존 바이너리로 가능. 첫 6회가 G-ID 재현 증거가 된다. 이번 선택이다. |
| run1 뒤 9회만 restart | 지원 경로가 없다. 구현·상태 완전성·동등성 검증이 추가로 필요하므로 이번 범위 밖이다. |

지원된 restart가 있었다면 이미 수행한 앞부분의 계산을 줄일 수 있었겠지만, 현재 산출을 재시작 상태로 간주해 비용 절감을 주장하지 않는다.

### 2.3 러너·exports·스테이징의 최소 변경

물리·알고리듬 exports 델타는 다음 하나다.

```text
LUMINA_PURE_CMFGEN_ITER: "6" → "15"
```

경로는 B에서 R로 대응 치환한다. 입력 내용은 동일하게 유지하고, 출력 경로가 B나 C를 가리키지 않는지 확인한다. 경로 치환은 임의 설정 변경의 예외가 아니다.

B/job.slurm을 원본 위치에서 수정하지 않는다. R/job.slurm **신설 사본**에서 다음만 변경한다.

- 등록 반복 수를 러너 내부의 단일 상수 15로 두고 ITER 검사, footer, FIXED_TE 소비 횟수 검사에 사용한다.
- 오류 문자열을 실제 계약을 설명하는 이름으로 고친다.
- 작업 이름·출력 위치·시간 제한을 이번 런에 맞춘다.
- H100 두 장의 자원 검사와 제출 조건을 일치시킨다. run1과 같은 H100 NVL을 사용하며 h200 및 A100으로 자동 전환하지 않는다.
- `module purge`와 GNU 13.2.0·CUDA 13.0.2의 명시적 module load를 둔다. 기존 빌드·실행 기록과 필요한 동적 라이브러리 정합을 확인한다.
- 기존 자체시험의 ITER=2, CPU 32, 이전 GPU 프로필 등 현재 러너와 충돌하는 fixture를 이번 사본의 실제 조건에 맞춘다.

실행 argv의 `1 2 spectrum nlte`에서 2는 이번 DET 반복 수를 결정하지 않는다. 생산 경로가 소비하는 env가 15임을 검사하며, argv까지 불필요하게 바꾸지 않는다.

R/input/outer_iterations.txt, 러너 사본, exports 해시, STAGE_IDENTITY 등 **이번 런의 운용 메타데이터**도 15회와 실제 실행 파일을 가리키게 작성한다. 원래 B의 잘못되거나 오래된 메타데이터를 소급 수정하지 않는다.

### 2.4 판독기 매개변수화와 기준 자료 분리

P/scripts/w5_cont_readout.py를 재사용한다. 최소 인터페이스는 다음으로 등록한다.

- 기존 `readout`에 `--mode cont-source|iter` 추가, 기본값 `cont-source`.
- `--n-iter` 추가, 기본값 6.
- `iter` 모드에는 `--baseline-w5 B`를 필수로 받는다.
- 이번 생산 판독은 `--mode iter --n-iter 15`만 허용한다.
- 기존 `--c1`, `--referents`, `--gates`, `--out`을 유지한다.
- `--stats-only`는 이번 최종 판정에 사용하지 않는다.

판독기 안의 반복 수는 실행별로 전달한다. 전역 `N_ITER`만 15로 바꿔 모든 자료에 적용하는 구현은 불가하다.

| 대상 | 소비 범위 |
|---|---|
| 신규 R | 인구·셸·창 k0–14, 선 INIT와 k0–14, 최종 publication |
| 기준 W5 B | 첫 6회 신원 비교, 기존 k0–5 궤적과 최종 산출 |
| C1 C | 기존 k0–5 산출만 사용 |
| 공개값·선 선택 F | 기존 봉인 referent 그대로 사용 |

특히 다음 의존성을 함께 고친다.

- G-RUN의 파일 범위, W5 태그·exact solve·선 호출 수.
- 선의 세대→논리 반복 매핑과 마지막 반복 표기.
- `levels_final` 대조 대상과 `ion_csv_k5` 같은 고정 열 이름.
- 인구·창 판독에서 실행별 반복 범위.
- 기존 k4·5 효과 통계와 새 수렴 통계의 분리.
- 결과 JSON의 모드·N·마지막 반복·분기 의미.

**C1의 k6–14를 요구하거나 생성하지 않는다.** 해당 대조는 `NO_MATCHED_ITERATION`으로 표시하며 새 런의 입력 누락으로 세지 않는다. R의 말단과 C1 k5의 차이를 싣는 경우 반드시 **서로 다른 반복 수의 종점 비교**라고 표시한다.

F/public_x.csv와 F/line_manifest.csv를 그대로 소비한다. 새 착지 결과에 따라 공개 보간·셸·선 표본을 다시 선택하지 않는다.

기존 `cont-source` 모드의 k4·5, M, 0.30 dex, MOVE/NONE 의미는 보존한다. **이번 iter 모드에는 M을 판정량으로 사용하지 않는다.**

### 2.5 선 원장·저장량·RLIMIT

W5 ON 바이너리가 이미 기록하는 선 원장을 INIT부터 k14까지 유지한다. 호출은 1–16이며, 기존 매핑이 유지되면 다음 규모다.

- 호출당 103,585행.
- 합계 1,657,360행.
- 실측 144,129,422 B에 호출 수 비율 16/7을 적용하면 약 329,438,679 B, **0.329 GB 또는 314 MiB**.
- 선행 계획의 512 B/행 가정을 적용하면 약 848,568,320 B다. 이는 예산 가정이며 실제 행 길이의 강제 상한이 아니다.

기존 `ulimit -f 1048576`, 즉 **파일별 1 GiB 제한**을 유지한다. RLIMIT는 런 전체 저장량의 제한이 아니다. 파일 절단·행 누락을 허용해 제한을 맞추지 않는다.

전 호출 원장을 유지하는 이유는 기존 코드와 exports를 그대로 보존하면서, 고정 선 표본의 말단 궤적과 호출 완비를 같은 판독기로 확인할 수 있기 때문이다. 반복 일부의 출력을 끄기 위한 새 env나 src 변경은 저장 절감에 비해 이번 계약을 더 넓힌다.

판독기는 전량 원장을 스트리밍하고 고정 표본만 결과에 유지한다. 추가 전량 사본·새 fine 배열 덤프는 만들지 않는다.

## §3. 변경집합·산출물·집행 범위

**생산 src 편집, 헤더 변경, env 등록 변경, CUDA 커널 변경, 바이너리 재빌드는 필요 없다.**

### 변경집합 — DET-W5-ITER (표)

| 경로 | 분류 | 허용 변경 |
|---|---|---|
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/w5_cont_readout.py` | 기존 판독기 확장 | §2.4의 실행별 반복 수·모드 분리, §4의 prefix 비교, §5의 수렴·착지·간극과 분기, 이에 직접 필요한 자체시험 |

### 변경집합 끝

새 iter 전용 함수는 `w5_iter_` 접두사를 사용한다. 기존 공통 함수에는 필요한 인수 전달만 추가한다.

R/job.slurm, R/input/resolved_lumina.exports, R/input/outer_iterations.txt와 신원 기록은 **런 디렉토리의 신설 운용 파일**이며 위 소스 변경집합 표 밖의 산출이다. 원본 러너·입력·기존 판독 결과는 보존한다.

| 신설 산출, U 기준 | 목적 |
|---|---|
| `impl/` | 변경 전 판독기 바이트 사본·해시, 패치, 새 판독기·러너 자체시험 |
| `run1/input/`, `run1/job.slurm` | 동일 물리 입력과 새 운용 계약 |
| `run1/work/`, stdout/stderr·footer | 유일한 판정런의 원증거 |
| `readout/prefix_identity.json` | 첫 6회 비교 피연산자·해시·최대 차이·결론 |
| `readout/result.json`, `population.json` | 모드·N·게이트·분기, 셀별 궤적·착지·간극 |
| `readout/lines.json`, `windows.json`, `outer.json` | 기존 보조 판독의 확장 |
| `JUDGE_FROZEN.txt`, `execution_record.json` | 사용 판독기·러너 신원, 명령·rc·분장·중단 사유 |

새 판독기 수정 전에 기존 해시의 파일을 U/impl에 보존한다. W/JUDGE_FROZEN.txt와 기존 판정 산출을 새 판독기에 맞춰 덮어쓰지 않는다. 새 판독기는 GPU 실행 전에 동결한다.

### 판정런과 자원

- **런 한 번**, INIT와 외부 반복 15회.
- H100 NVL 두 장, 단일 노드·태스크, CPU 16, `--mem=64G`.
- **`--time=36:00:00`**, 재큐잉 금지.
- 예상 약 27.3시간·54.5 GPU·h, 할당 상한 **72 GPU·h**.
- 출력 위치는 U 아래 GPFS. 기존 입력·출력 덮어쓰기 금지.
- 신규 실행 출력·판독용 공간은 4 GiB를 준비 예산으로 잡고, 입력 사본 공간은 별도로 확인한다. 이는 실측 총량 또는 추가 파일별 제한이 아니다.
- 기존 입력 manifest에 열거된 파일과 출력 파일의 크기를 대상으로 확인한다. 광범위한 파일시스템 재귀 조사 없이 여유 공간을 확인한다.
- 오프라인 자체시험·판독·해시 배터리는 grammar-debug 또는 허용된 계산 자원에서 수행한다. 로그인 노드는 조회에 한정한다.

실패가 추가 GPU 런·허용오차 완화·바이너리 변경 권한을 만들지 않는다. 실패 산출과 사용 신원을 보존한다.

## §4. 게이트·오프라인 자체시험·음성대조

게이트는 기존 세 묶음을 유지한다. 수렴 자체를 실행 유효성 게이트로 추가하지 않는다.

| 게이트 | 수용 기준 | 소비 |
|---|---|---|
| **G-ID** | §0·§2·§3의 바이너리·입력·exports·실행 조건 정합 및 아래 첫 6회 동일성 | 같은 W5 계산의 반복 확장인가 |
| **G-SELF** | 기존 바이너리의 선행 검증 신원 확인과 아래 변경 부분의 직접시험·음성대조 통과 | 반복 범위와 새 판정식이 올바르게 구현됐는가 |
| **G-RUN** | 아래 정상 종료·호출·세대·필수 자료 완비 | 수렴·간극 분기를 계산할 수 있는가 |

### G-ID: 정적 신원과 첫 6회

정적 신원 피연산자는 다음이다.

- 바이너리: W/cuda_build1/lumina_cuda, B/input/lumina_cuda, R/input/lumina_cuda.
- 소스 신원: W/cuda_build1/source_sha_at_build.txt와 W/APPROVED_SHA.txt에 지정된 파일.
- 입력: B와 R의 model·global_atomic·fixed_te_profile 및 기존 deck·sigma·topion manifest.
- exports: B/input/resolved_lumina.exports와 R/input/resolved_lumina.exports. 등록 경로 대응을 정규화한 뒤 ITER 값 하나만 달라야 한다.
- 판독기·러너: 보존한 변경 전 파일, 실제 변경 파일, 이번 JUDGE_FROZEN과 집행 기록.

실행 후에는 B와 R에서 다음 파일 쌍을 비교한다.

```text
work/physics_comparison/physics_DET_iter0000.ion.csv
…
work/physics_comparison/physics_DET_iter0005.ion.csv

같은 k=0000–0005의 .shell.csv 및 .spectral.csv
```

비교 범위는 **전체 행·전체 열**이다. 등록 대상 셸만 골라 동일성을 주장하지 않는다.

1. 먼저 각 파일 쌍의 바이트 동일성을 검사하고 해시를 기록한다.
2. 바이트가 다르면 CSV 구조·행 키·정수 신원을 정확히 대조한 뒤, 수치 열에 아래 사전 허용오차를 적용한다.
3. 누락·중복 행, 열 차이, 셸·bin·population generation 차이는 실패다.

유한 실수 a, b에 대해:

\[
a=b=0 \quad\text{또는}\quad
\frac{|a-b|}{\max(|a|,|b|)}\le10^{-8}.
\]

분모가 0인 경우는 위의 정확한 zero 동등성으로만 처리한다. 작은 양수나 절대오차 floor를 넣지 않는다. 비유한 필수 물리값은 통과시키지 않는다.

결과는 `BYTE_IDENTICAL`, `WITHIN_REGISTERED_TOLERANCE`, `FAIL`로 구분한다. 두 번째를 byte 동일이라고 쓰지 않는다. \(10^{-8}\)은 이번 출력 신원 비교의 고정 허용오차이며, fine solve residual에서 도출한 물리 오차 보증이 아니다.

stdout/stderr 전체는 시간·경로·마지막 반복 진단 위치가 달라지므로 바이트 비교 피연산자로 삼지 않는다. 최종 publication도 서로 다른 반복 시점이므로 이 비교에서 제외한다.

첫 6회 불일치가 확인되면 G-ID 실패다. 결과를 본 뒤 허용오차를 늘리지 않는다. 이 검사는 추가 GPU 계산 없이 기존 산출을 읽지만, 프로세스의 모든 내부 상태에 대한 비트 동일성을 인증하지는 않는다.

### G-SELF: 변경 부분에 집중

생산 바이너리가 동일하고 선행 검증 신원이 유지되므로 W5 연속 원천 자체시험·OFF GPU 실험을 다시 만들지 않는다. 새 시험은 다음을 하나의 직접시험 묶음으로 수행한다.

| 시험 | 필요한 증거 |
|---|---|
| 기존 모드 보존 | 보존한 구 판독기와 새 `cont-source` 기본 모드가 기존 run1의 M·δ·분기와 필수 표의 의미를 보존한다. 산출은 독립 디렉토리에 쓴다. |
| 실행별 N | 합성 N=15 자료와 N=6 기준 자료를 함께 판독한다. C1 k6–14 부재는 정상이며 R 말단 누락은 거부한다. |
| 러너 | ITER=15와 올바른 소비 횟수의 성한 fixture 통과. ITER=6 또는 FIXED_TE 횟수 부족 fixture 거부. 기존 자원·원본 보호 검사 유지. |
| prefix 비교 | 성한 파일 통과, 등록 오차 경계 검사, 경계 밖의 실제 수치 변경·행 누락·세대 변경 거부. 주입 전후 값이 실제로 다른지 확인한다. |
| 수렴 | 양·음 방향 모두에서 경계 아래·정확 경계·경계 위를 검사한다. 한 셀의 표류와 마지막 창의 왕복 변동도 검출한다. |
| 공개 간극 | 2×와 1/2×의 양쪽 경계, 경계 밖, 한 셀만 간극이 남는 경우를 검사한다. |
| 실패 우선순위 | 필수 행 누락·NaN·비양수 인구·게이트 실패가 수렴 또는 근접 분기로 통과하지 못한다. |

시험은 §5의 식을 구현한 실제 판독 경로를 호출한다. 분기 코드를 수동으로 써 넣은 시험만으로 판독기를 검증하지 않는다.

### G-RUN

N=15에 대해 다음이 필요하다.

- 모델 정상 종료, footer `terminal_state=SUCCESS`, `outer_iterations=15`.
- INIT와 k0–14의 W5 bin·fine 소비 증거, fine exact solve **16회 전부 OK**.
- fine 연산자 순서: INIT Gaussian 한 번, 이후 일반 Sobolev 15회.
- radiation generation은 INIT=1, k의 값=k+2. FIXED_TE publication 증거는 논리 반복마다 존재하며 합계 15회.
- k0–14의 ion·shell·spectral 자료와 최종 publication이 완비된다.
- 선 원장 호출 1–16, 호출별 행 수·신원 정합, 고정 선택 선의 INIT·k0–14 자료 완비.
- 최종 준위합·인구 snapshot·publication의 세대가 실제 마지막 상태를 가리킨다. 기존 k5 표기를 그대로 남기지 않는다.
- §5에 필요한 인구와 공개값이 유효하고, 기존 필수 창·방출 판독이 정의된다.

기존 판독기의 수치 실패 태그 범위를 유지한다. 로그 어디에든 `BLOCKED`라는 단어가 있다는 이유로 새로운 전역 실패 규칙을 만들지 않는다. 선행 종료감사가 기존 증거의 유효성을 문제 삼으면 해당 범위로 재기안한다.

**외곽 인구 표류, 큰 공개 간극, 작은 bin 갱신량 자체는 G-RUN 실패 조건이 아니다.** fine exact solve 성공과 외부 인구비 수렴은 서로 다른 검사다.

## §5. 판정량·기대 관측·등록 분기

### 5.1 인구 권위와 공개 기준

선행 단과 같은 대상을 유지한다.

\[
Z\in\{14,16,20\},\qquad
\mathcal S=\{10,15,19,24,30\},\qquad
A\in\{\mathrm{stage},\mathrm{levels}\}.
\]

\[
x_{Z,A,s,k}
=\frac{n_{Z,A,\mathrm{III},s,k}}
       {n_{Z,A,\mathrm{II},s,k}},
\qquad
y_{Z,A,s,k}=\log_{10}x_{Z,A,s,k}.
\]

판정 대상은 30개 셀이다. 각 인구가 양의 유한값인지 먼저 검사한다. 계산은 가능한 한

\[
y=\log_{10}n_{\mathrm{III}}-\log_{10}n_{\mathrm{II}}
\]

로 수행해 불필요한 비율 overflow를 피한다. 표현 불가능한 선형 비율은 로그와 상태로 보고하며 floor를 넣지 않는다.

공개값은 선행 F/public_x.csv를 유지한다. toy06 19.480 d에서 셸 중심 속도로 II·III 분율을 각각 보간한 뒤 비율을 취한 값이며 외삽하지 않는다. 원소·셸에 대응하는 양의 유한 공개값이 필요하다.

stage, 준위합, 사다리 r은 별도 권위다. r은 기존 자료에서 식별 가능할 때만 보조 기재하며 필수 판정 입력으로 새로 만들지 않는다.

### 5.2 방향 중립 수렴식과 착지값

마지막 **세 번의 갱신**을 고정한다.

\[
\Delta y_{i,k}=y_{i,k}-y_{i,k-1},
\qquad
D_i=\max_{k=12,13,14}|\Delta y_{i,k}|,
\]

여기서 \(i=(Z,A,s)\)다. 따라서 필요한 말단 인구 상태는 k11–14다.

셀별 운영상 수렴:

\[
C_i \iff D_i\le0.01\ {\rm dex}.
\]

정확히 0.01 dex이면 통과다. 원소·권위별 결과는 해당 다섯 셀을 모두 제시하며, 전체 등록 대상 수렴은

\[
C_{\rm all}\iff \bigwedge_i C_i
\]

로 정의한다. 중앙값으로 움직이는 소수 셀을 숨기지 않는다.

0.01 dex는 말단의 반복 안정성을 판독하기 위한 **운영상 기준**이다. 효과 크기, 검정력, 점근 오차상한이 아니다. 마지막 세 번이 작아도 이후의 느린 표류나 다른 고정점 가능성이 배제되는 것은 아니다.

말단 대표값은 마지막 세 상태의 로그 평균이다.

\[
\bar y_i=\frac{y_{i,12}+y_{i,13}+y_{i,14}}3,
\qquad
x_{{\rm tail},i}=10^{\bar y_i}.
\]

- \(C_i\)가 참이면 **등록 기준의 착지 대표값**으로 보고한다.
- \(C_i\)가 거짓이면 같은 값은 **말단 평균**으로만 보고한다.
- 마지막 세 상태의 최소·최대, \(D_i\), 각 \(\Delta y\), k14 값을 함께 싣는다.
- 미수렴 자료에서 지수 감쇠 등을 맞춰 무한 반복 착지값을 대신 판정하지 않는다.

### 5.3 공개값 간극

셀별 부호 있는 간극은

\[
g_i=\bar y_i-\log_{10}x_{{\rm pub},Z,s}.
\]

착지 대표값의 \(x/x_{\rm pub}\), \(g_i\)와 함께 마지막 창의 간극 범위를 보고한다.

공개 2× 안이라는 판정이 평균에 의해 경계 통과를 숨기지 않도록:

\[
H_i=\max_{k=12,13,14}
\left|y_{i,k}-\log_{10}x_{{\rm pub},Z,s}\right|,
\]

\[
P_i\iff H_i\le\log_{10}2.
\]

따라서 마지막 세 상태가 모두 \([x_{\rm pub}/2,\,2x_{\rm pub}]\) 안에 있어야 한다. 정확한 경계는 포함한다. \(\log_{10}2\)를 기존 효과 문턱 0.30으로 대체하지 않는다.

\(C_i\)와 \(P_i\)는 따로 저장한다. stage와 levels의 수렴·간극을 합쳐 평균내지 않는다. 전체 요약에 두 권위의 조건을 모두 사용하더라도 두 권위의 값이 같아야 한다는 조건을 추가한 것은 아니다.

### 5.4 사전 기대와 보조 판독

선행 판정문의 관측에 따르면 W5와 C1의 k4·5 차이는 등록 셀 모두 음수였고, W5의 안쪽 일부 셸은 k5까지 감소 중이었다. 그 관측에서 다음을 **조건부 외삽**한다.

- k5→k14의 아홉 갱신 동안 −0.02∼−0.05 dex/회가 그대로 지속된다면 추가 이동은 약 −0.18∼−0.45 dex다.
- 이는 해당 궤적의 선형 연장 예시다. 모든 원소·권위·셸에 적용되는 기대값이 아니다.
- 이미 +1∼+3 dex인 간극이 그 정도 이동만으로 사라진다고 기대할 근거는 없다.
- 외곽의 평탄화가 유지될지, 안쪽 이동이 둔화할지, 방향이 바뀔지는 이번 관측 대상이다.

이 기대는 판정식에 들어가지 않는다. 감소·증가·교차·진동을 같은 식으로 처리한다. **사전 선택 효과 크기나 검정력을 주장하지 않는다.**

보조 판독은 선행 형식을 유지한다.

- 모든 등록 셀의 k0–14 인구 궤적과 stage–levels 차이.
- sh19/24/30의 stage·준위합·\(n_e\) 궤적, bin 갱신량과 fine solve 상태.
- 기존 고정 선 표본의 반복별 \(J_c,\bar J,\beta,L_l,a_l\). 요약 구간은 k5→14와 k11→14로 고정한다.
- 기존 다섯 창의 J/B와 최종 observer spectrum의 \(L_{\rm grid}\), λ<2500 Å 및 912–2000 Å 분율.
- C1의 k0–5 자료와 W5 run1 산출은 기존 관측 대조로 유지한다.

선 변화는 W5 내부의 반복 변화다. \(J_c,\bar J\)와 인구의 시점을 세대별로 구분하며 같은 k 표기만으로 동시 소비를 가정하지 않는다. C1 선 대조는 계속 `BASE_LINE_FIELD_UNAVAILABLE`다.

### 5.5 등록 분기

우선순위는 **X → DRIFT → GAP/WITHIN2**다.

| 분기 | 판정식 | 허용 해석 | 다음 걸음 |
|---|---|---|---|
| **W5-ITER-WITHIN2** | 게이트 통과, 모든 \(C_i\) 참, 모든 \(P_i\) 참 | 등록 셀·권위가 운영상 수렴하고 말단 창 전체가 공개값 2× 안 | W5 착지 자료를 후속 비교 기준 후보로 보존. 스펙트럼·MC 목적의 달성 여부는 별도 범위로 판단 |
| **W5-ITER-GAP** | 게이트 통과, 모든 \(C_i\) 참, 적어도 하나의 \(P_i\) 거짓 | 등록 기준에서는 안정됐으나 공개 간극이 남음 | 셸·권위별 잔여 간극으로 W2/W3/W4/H1–H4 등 후속 후보를 정리. 추가 반복만으로 해소된다고 주장하지 않음 |
| **W5-ITER-DRIFT** | 게이트 통과, 적어도 하나의 \(C_i\) 거짓 | 15회까지 등록 대상 전체의 착지를 확보하지 못함 | 움직이는 셀·속도·방향·진동을 보고. 추가 반복 또는 다른 접근은 자원과 함께 별도 user 결정 |
| **W5-ITER-X** | 필수 게이트 실패 또는 필수 판정량 미정의 | 수렴·공개 근접성 판정 불가 | 신원·실행·판독 실패를 분류하고 원증거 보존. 재실행 자동 허용 없음 |

미수렴 셀의 큰 간극을 수렴 후 잔여 간극으로 부르지 않는다. 반대로 수렴한 셀의 결과는 전체 DRIFT 요약 속에서도 따로 보고한다.

전체 WITHIN2는 등록 대상에만 적용된다. 모델 전체 셸·전체 이온·스펙트럼의 일치 판정이 아니다.

## §6. 주장 상한

- 선행 W5-NONE은 6회 조건의 등록 효과 통계에 관한 판정이다. 이번 반복 확장 결과로 과거 판정의 의미를 소급 변경하지 않는다.
- 반복 확장은 W5 ON 조건의 궤적이다. **동일 N의 OFF 런이 없으므로 장기 착지의 ON/OFF 인과 차이를 측정한 것이 아니다.**
- W5는 확인된 연속 원천 처방 차이이자 결함 후보라는 지위를 유지한다. 주원인·원인별 dex 몫은 판정하지 않는다.
- W2의 사다리 r, damping된 stage 인구, 준위합은 별도 권위다. 단계 Γ의 LTE 준위 가중을 NLTE 인구 가중으로 해석하지 않는다.
- FIXED_TE에서 얻은 착지는 온도까지 함께 수렴한 자기일관 복사평형해의 인증이 아니다.
- 외곽 표류는 유한 반복 결과로 남긴다. fine exact solve 성공이나 작은 bin 갱신량만으로 외부 물질 수렴을 인증하지 않는다.
- 기존 ff pack과 bin 별도 ff의 중첩을 보존한다. 이번 단은 ff 처방의 물리적 정확성을 인증하지 않는다.
- 원천비·연속장비·선 \(J_c,\bar J\)·인구비는 다른 양이다. 반복 사이에는 β와 \(L_l\)도 변하므로 고정 물질 응답식을 무조건 적용하지 않는다.
- H1–H4, 전하 폐합의 영향, 원자자료·준위 대응 오류 미배제를 유지한다.
- 독립 σ 전수 적분, 동일 조건 fine/bin 장·bf 적분 비교, CMFGEN 내부 J 비교의 선행 재현 BLOCKED를 해소했다고 주장하지 않는다.
- C1 선 원장 공백은 유지한다. W5 내부 궤적이나 bin J로 채우지 않는다.
- 한 번의 DET 반복 확장으로 최종목적인 MC 수송 스펙트럼 재현을 달성했다고 주장하지 않는다.

## §7. Q-GOAL·Q-LEAN 자기답·집행 기록

### Q-GOAL 자기답

**자기답: PASS. 독립 감사 판정이 아니다.**

최종목적은 MC 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현하는 것이다. 이번 단은 W5를 유지한 물질·복사장 반복에서 현재 인구 간극이 지속적인 착지 차이인지, 아직 이동 중인 유한 반복 상태인지를 구분한다.

이를 생략하면 미수렴 종점의 간극을 다음 물리 수리의 근거로 오독할 수 있다. 이번 결과는 추가 반복과 다른 물리 후보 중 무엇을 다음으로 다룰지 결정하는 자료다.

### Q-LEAN 자기답

**자기답: APPROPRIATE. 독립 감사 판정이 아니다.**

생산 소스·바이너리·계측은 유지한다. 새 계산은 승인 범위의 GPU 런 한 번이며, 새 논리는 기존 판독기의 반복 범위와 착지 판정에 한정한다.

| 유지하는 증거 | 소비자·이유 |
|---|---|
| 동일 바이너리·입력·exports | 반복 수 단일 변경 확인 |
| 첫 6회 기존 CSV | G-ID. 추가 GPU 계산 없이 재현 확인 |
| 인구 말단 궤적·공개값 | 수렴·착지·간극 분기 |
| 기존 선 원장 | 기존 구현 보존, 고정 표본의 반복 궤적과 호출 완비 |
| 기존 창·방출·외곽 표 | 착지 결과의 해석 범위와 동반 변화 |
| 변경 부분의 자체시험 | N 혼용·말단 누락·분기 오류 검출 |

restart 구현, OFF 대조 런, 새 전체 배열 덤프, 추가 물리 gate, 자동 반복 연장, 점근 외삽 판정은 포함하지 않는다. 앞 6회의 계산 비용은 지원되지 않는 restart를 새로 만들지 않고 동일 경로를 실행하는 비용이며, 그 산출을 G-ID 증거로 활용한다.

### 집행 기록

분장 장부의 정본은 이 절이다. 운전자가 실제 수행자와 결과를 기록한다.

| 단계 | 담당·현재 상태 | 기록 |
|---|---|---|
| 기안 | astra, read-only 본문 제출 | 모델 ID, 직접 조회와 미실행 구분 |
| 파일화·기계 검사 | 운전자, 미집행 | 문서 해시, 아래 명령·rc |
| 선행 종료감사 인계 | 운전자, 결과 미확정 | 본 계약에 영향을 주는 지적과 처분 |
| 판독기·러너 변경과 직접시험 | 현행 배분에 따라 운전자 발주, 미집행 | 실제 담당자, 허용 변경, 시험·NC |
| 소스·입력·바이너리 신원 | 운전자, 전체 대조 미집행 | §4 피연산자·해시·rc |
| 실행 | 운전자, 미집행 | job·GPU·메모리·시작/종료·시간·파일 크기 |
| 판독·판정 | 운전자, 미집행 | G-ID·G-SELF·G-RUN, 셀별 결과와 전체 분기 |
| 종료 처리 | 현행 배분으로 운전자 기록 | 실제 감사자·재현 상태·미해결 사항 |

### 실행·판독 재현 명령

다음은 **구현 후 운전자가 실행할 인터페이스**이며 현재 실행됐다는 뜻이 아니다. 판독 출력 디렉토리는 존재하면 거부한다.

```bash
python3 -B scripts/w5_cont_readout.py selftest \
  --out "$U/impl/readout_selftest"

bash "$R/job.slurm" --selftest

python3 -B scripts/w5_cont_readout.py readout \
  --mode iter --n-iter 15 \
  --run "$R" \
  --baseline-w5 "$B" \
  --c1 "$C" \
  --referents "$F" \
  --gates "$U/gates.json" \
  --out "$U/readout"
```

기존 모드 회귀 판독은 별도 신규 출력 경로에서 수행한다. 새 결과 JSON에는 문서·판독기·러너 해시, 모드, N, 기준 런, 말단 창, 허용오차, 게이트 증거 위치를 기록한다. G-ID·G-SELF의 PASS 문자열만으로 근거 파일을 대신하지 않는다.

### 기계 프리플라이트

파일화 후 P를 root로, 허용된 오프라인 계산 환경에서 운전자가 실행한다.

```bash
timeout 5m python3 -B scripts/check_prereg_preflight.py \
  docs/PREREG_DET_W5_ITER_2026-10-01.md --root .

timeout 5m python3 -B scripts/check_gate_referents.py \
  docs/PREREG_DET_W5_ITER_2026-10-01.md --root .

timeout 5m python3 -B scripts/check_doc_canonical.py \
  docs/PREREG_DET_W5_ITER_2026-10-01.md \
  docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md \
  docs/VERDICT_DET_W5_CONT_SOURCE_2026-10-01.md \
  CLAUDE.md \
  /gpfs/kjhan/lumina/cmfgen_map/w5_20260930/astra_closeout_r1.md
```

종료감사 파일에 대한 정본 검사 통과는 감사 완료 또는 내용 승인이라는 뜻이 아니다.

**어느 명령이든 rc≠0이면 기안 재발주다.** 운전자가 수렴식·문턱·범위·분기를 임의 수정해 통과시키지 않는다. 구현 발주에는 기존 dispatch preflight를 별도로 적용한다.

아래 선언은 **셀 코드의 전체 분기 집계**를 검사한다. 코드 0=X, 1=수렴·2× 안, 2=수렴·간극 잔존, 3=미수렴이다. 실제 인구에서 코드가 올바르게 계산되는지는 §4의 직접시험으로 검증한다. 선언의 변경집합 검색은 해당 판독기 파일 하나로 한정한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-W5-ITER (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "/home/kjhan/BACKUP/Eunha\\.A1/Claude/Lumina-sn/scripts/w5_cont_readout\\.py",
    "symbol": "w5_iter_",
    "roots": [
      "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/w5_cont_readout.py"
    ],
    "expected_extra": [
      "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/w5_cont_readout.py"
    ]
  },
  "branches": {
    "regimes": [
      [0.0, 0.0],
      [1.0, 1.0],
      [2.0, 2.0],
      [3.0, 3.0]
    ],
    "metrics": {
      "count": "len(v)",
      "bad": "sum(1 for x in v if x not in (0.0, 1.0, 2.0, 3.0))",
      "failed": "sum(1 for x in v if x == 0.0)",
      "within": "sum(1 for x in v if x == 1.0)",
      "gap": "sum(1 for x in v if x == 2.0)",
      "drift": "sum(1 for x in v if x == 3.0)"
    },
    "rules": [
      {
        "name": "W5-ITER-X",
        "predicate": "count != 30 or bad > 0 or failed > 0"
      },
      {
        "name": "W5-ITER-DRIFT",
        "predicate": "count == 30 and bad == 0 and failed == 0 and drift > 0"
      },
      {
        "name": "W5-ITER-GAP",
        "predicate": "count == 30 and bad == 0 and failed == 0 and drift == 0 and gap > 0"
      },
      {
        "name": "W5-ITER-WITHIN2",
        "predicate": "count == 30 and bad == 0 and failed == 0 and within == 30"
      }
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "all-within", "mix": [[1.0, 30]]},
      {"name": "one-gap", "mix": [[1.0, 29], [2.0, 1]]},
      {"name": "all-gap", "mix": [[2.0, 30]]},
      {"name": "one-drift", "mix": [[1.0, 29], [3.0, 1]]},
      {"name": "drift-over-gap", "mix": [[2.0, 29], [3.0, 1]]},
      {"name": "failure-over-drift", "mix": [[3.0, 29], [0.0, 1]]},
      {"name": "missing-cell", "mix": [[1.0, 29]]},
      {"name": "extra-cell", "mix": [[1.0, 31]]},
      {"name": "invalid-code", "mix": [[1.0, 29], [4.0, 1]]}
    ],
    "residual": "W5-ITER-X"
  },
  "references": [
    {"path": "CLAUDE.md"},
    {"path": "docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md"},
    {"path": "docs/VERDICT_DET_W5_CONT_SOURCE_2026-10-01.md"},
    {"path": "scripts/check_prereg_preflight.py"},
    {"path": "scripts/check_gate_referents.py"},
    {"path": "scripts/check_doc_canonical.py"},
    {
      "path": "scripts/w5_cont_readout.py",
      "flags_existing": [
        "W5_READOUT_V1",
        "BASE_LINE_FIELD_UNAVAILABLE",
        "def load_pops",
        "def g_run",
        "def cmd_selftest"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/job.slurm",
      "flags_existing": [
        "STRICT_TWO_ITERATION_CONTRACT_MISMATCH",
        "outer_iterations=6",
        "ulimit -f 1048576"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/RUN_FOOTER.txt",
      "flags_existing": [
        "job_id=409099",
        "outer_iterations=6",
        "elapsed_seconds=39269",
        "terminal_state=SUCCESS"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/input/resolved_lumina.exports",
      "flags_existing": [
        "LUMINA_PURE_CMFGEN_ITER",
        "LUMINA_CMF_FINE_CONT_NLTE_SOURCE",
        "LUMINA_FIXED_TE_PROFILE"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/input/binary.sha256"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/input/outer_iterations.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/gpu_identity.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/APPROVED_SHA.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/JUDGE_FROZEN.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1/source_sha_at_build.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1/compiler.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/readout/result.json"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/impl/readout_demo/referents/referents_manifest.json"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/impl/readout_demo/referents/public_x.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/impl/readout_demo/referents/line_manifest.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_cuda.cu",
      "flags_existing": [
        "LUMINA_PURE_CMFGEN_ITER",
        "det_transactional_requested",
        "cmfgen_final_writer"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_cmfgen.c",
      "flags_existing": [
        "int cmfgen_run",
        "uint64_t rad_generation = 0",
        "INIT_SEED_MATERIAL_PREDICTOR",
        "w5_cont_rad_generation"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/env_universe.h",
      "flags_existing": [
        "LUMINA_ENV_UNIVERSE_COUNT 534",
        "LUMINA_CMF_FINE_CONT_NLTE_SOURCE"
      ]
    }
  ]
}
```
