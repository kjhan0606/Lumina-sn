# CODEX_DECK_REGEN — CMFGEN full-coverage 덱 자원 산정·생성 기록

작성일: 2026-08-03  
대상 설계: [`DECK_REGEN_DESIGN.md`](DECK_REGEN_DESIGN.md:1)

## 결론

- **1단계 자원 관문: PASS.** 실제 새 준위 집합은 macro-atom 준위 **36,355**, 물리선
  **3,406,111**, edge **10,218,333**이다. 현재 대비 edge 배수는 **1.318087×**이며,
  설계의 상한 3.6×가 아니다. 산정 규칙은 생성기와 동일하게 `NF` 이내 양 끝 준위와
  `lambda != 0`을 적용했다([검증기](../scripts/verify_deck_regen_fullcov.py:62),
  [생성기](../scripts/expand_atomic_data_cmfgen.py:431)).
- 80,000 MiB 기준선을 edge 비로 전부 확대하는 보수 산정은
  **110,569,158,168 byte = 105,446.97 MiB**이다. H200의
  **150,754,820,096 byte = 143,771 MiB**보다
  **40,185,661,928 byte = 38,324.03 MiB** 작으므로 **H200에 들어간다**.
  H200 실물 용량은 캡처 기록에 있다
  ([job 188932](</gpfs/kjhan/lumina_runner2/slurm/instr_capture_188932.out:2>));
  80,000 MiB 관문은 캡처 제출기의 검사값이다
  ([sbatch_instr_capture.sh](../scripts/sbatch_instr_capture.sh:45)).
- **2단계 생성: 미실행.** CPU 계산 노드 제출을 시도했으나 현재 샌드박스가 Slurm
  컨트롤러 소켓을 차단해 `Unable to contact slurm controller`로 끝났다. 규율에 따라
  호스트 `syntax`에서 무거운 생성을 실행하지 않았다. 목표 경로
  `data/tardis_reference_toy06_19p48d_sivcaiv_fullcov/`는 **생성되지 않았고**, 기존
  `_sivcaiv` 덱은 변경·삭제하지 않았다.
- **3단계 네 게이트: 모두 NOT RUN/미판정.** 신규 덱이 없으므로 PASS로 기록하지 않는다.
  다만 현행 생성기를 그대로 사전 열거하면 S V·Co II의 CMFGEN vintage 차이 때문에
  게이트 1 실패가 예상된다. 이를 통과시키기 위한 조정은 하지 않았다.
- **Co IV Υ 대용: 해소되지 않는다.** 생성 경로가 여전히 `19apr23/col_data`를 읽으며,
  Co IV 원본 스스로 Fe III 대용임을 명시한다
  ([Co IV col_data](../data/atomic/cmfgen/COB/IV/19apr23/col_data:4),
  [참조 문구](../data/atomic/cmfgen/COB/IV/19apr23/col_data:12)).

## 1단계 — 직접 계수

### 현재 수치의 코드상 산출 근거

캡처가 실제로 읽은 값은 물리선 2,584,132, macro-atom 준위 26,592,
transition/edge 7,752,396이다
([stdout](</gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:146>),
[macro topology](</gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:149>)).
생성기는 물리선 하나마다 emission, internal-down, internal-up 세 edge를 만든다
([expand_atomic_data_cmfgen.py](../scripts/expand_atomic_data_cmfgen.py:600)). 따라서

```text
7,752,396 = 3 × 2,584,132
```

이며 transition type별 행 수도 각각 2,584,132이다. 준위별 edge block의 끝은 실제로
세 버킷을 합산해 기록한다
([expand_atomic_data_cmfgen.py](../scripts/expand_atomic_data_cmfgen.py:651)).

### full coverage 집합

`MODEL_SPEC`의 27개 `NF`와 `atomic_links.txt`의 27개 `F_OSCDAT`를 읽고, 각 전이의
`min(i,j)`, `max(i,j)`가 `1..NF`이며 파장이 0이 아닌 것을 직접 모았다. 이 로직은
[검증기 69–105행](../scripts/verify_deck_regen_fullcov.py:69)에 고정했다. 결과는 다음과
같다.

| 양 | 현재 덱 | CMFGEN 활성 집합 | 공통 | full union |
|---|---:|---:|---:|---:|
| 물리선 identity | 2,584,132 | 1,703,064 | 881,085 | **3,406,111** |
| 준위 identity | 26,592 | 20,749 | 10,986 | **36,355** |
| macro-atom edge | 7,752,396 | — | — | **10,218,333** |

CMFGEN 활성선·준위와 공통 수의 독립 원장은
[`DECK_REGEN_DESIGN.md`](DECK_REGEN_DESIGN.md:8)에 있다. union은 추정식이 아니라
packed identity 집합의 합집합 크기다. 마지막 edge만 생성기의 실제 3-edge 규칙을 새
물리선 집합에 적용한 `3 × 3,406,111`이다.

참고로 `CMFGEN_FULL_LEVELS=1`, `CMFGEN_SUPER_LEVELS=1`로 **현행 최신-vintage 생성
소스**를 메모리에서 직접 열거하면 준위 36,355, 물리선 3,361,364, edge 10,084,092다.
준위 수는 닫히지만 full union보다 물리선이 44,747 적다. 생성기가 최신 날짜를 고르는
경로([expand_atomic_data_cmfgen.py](../scripts/expand_atomic_data_cmfgen.py:316))와 CMFGEN
런의 S V·Co II 링크 vintage가 다르기 때문이다. 기존 커버리지 원장은 이 차이를
S V 1,255 + Co II 44,000 = **45,255개의 CMFGEN 활성선 미포함**으로 분해한다
([CODEX_COVERAGE_SCOPE_SUMMARY.md](CODEX_COVERAGE_SCOPE_SUMMARY.md:149)). 총 행 차이
44,747과 활성선 결손 45,255의 차이 508은 새 최신-vintage 고준위 identity가 full
union 밖에 추가되기 때문이다.

## GPU 메모리 산정

기준 캡처는 50 shell이다
([stdout](</gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:148>)).
`L=물리선`, `E=edge`, `N=macro 준위`, `S=50`으로 놓으면 덱 크기에 따라 변하는 주요
device 배열은 다음과 같다.

| 배열군 | 코드식 | 현재 byte | full byte | 증가 byte |
|---|---:|---:|---:|---:|
| transition probability + line destruction | `16ES` | 6,201,916,800 | 8,174,666,400 | 1,972,749,600 |
| transition type/destination/line id | `12E` | 93,028,752 | 122,619,996 | 29,591,244 |
| Sobolev τ | `8LS` | 1,033,652,800 | 1,362,444,400 | 328,791,600 |
| Jbar/count/Jblue | `20LS` | 2,584,132,000 | 3,406,111,000 | 821,979,000 |
| line ν와 line 정수 3열 | `20L` | 51,682,640 | 68,122,220 | 16,439,580 |
| k-packet 2표 | `16NS` | 21,273,600 | 29,084,000 | 7,810,400 |
| ion-up destination/probability | `4N+8NS` | 10,743,168 | 14,687,420 | 3,944,252 |
| macro block reference | `4(N+1)` | 106,372 | 145,424 | 39,052 |
| BF σ(float)와 level 보조열 | `4,224N` | 112,324,608 | 153,563,520 | 41,238,912 |
| **식별된 합계** |  | **10,108,860,740** | **13,331,444,380** | **3,222,583,640** |

근거 allocation은 기본 line/edge/level 배열
([lumina_cuda.cu](../src/lumina_cuda.cu:174)), k-packet
([lumina_cuda.cu](../src/lumina_cuda.cu:219)), line-destruction
([lumina_cuda.cu](../src/lumina_cuda.cu:250)), Jbar/Jblue
([lumina_cuda.cu](../src/lumina_cuda.cu:7967)), BF σ
([lumina_bf_gemm.cu](../src/lumina_bf_gemm.cu:120))이다.

준위² 배열은 macro topology가 아니라 batched NLTE 행렬이다. 코드는
`batch × max_N² × 8` byte와 rhs/pivot을 할당한다
([lumina_cuda.cu](../src/lumina_cuda.cu:599)). 캡처의 `max_N=6,664`, `batch=50`에서
행렬만 **17,763,558,400 byte**, solver 전체는 **17,767,557,800 byte**이며 로그도
16,943.2 MiB로 확인된다
([stdout](</gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:325>)).
full coverage가 추가하는 IV–VI 준위는 현재 최대 pair인 Co II+Co III를 바꾸지 않으므로
이 준위² 항은 증가하지 않는다. `max_N`은 코드가 명시적 ion pair 중 최대를 찾아 정한다
([lumina_cuda.cu](../src/lumina_cuda.cu:7719)).

식별 합계에는 packet/event/CMF workspace 같은 고정·상태 의존 allocation이 빠져 있다.
따라서 최종 판정에는 더 보수적으로 **현행 80,000 MiB 전체를 edge 비
`10,218,333/7,752,396 = 1.318087079`로 확대**했다. 그 결과가 위 결론의
105,446.97 MiB이며, H200에서 38,324.03 MiB가 남는다. 즉 식별 증가분
3.00 GiB만 더하는 경우보다도 훨씬 보수적인 판정이다.

## CPU 메모리·덱 크기·로딩 시간

### CPU 메모리

상주하는 크기 의존 host 배열을 코드 선언과 allocation에서 바이트 단위로 합쳤다.
line CSV는 정수 4열+double 6열, 즉 `64L`을 읽는다
([lumina.h](../src/lumina.h:366), [loader](../src/lumina_atomic.c:649)); opacity는 τ,
source, transition probability와 topology를 가진다
([lumina.h](../src/lumina.h:204), [loader](../src/lumina_atomic.c:465)); estimator 두 표는
`16LS`다([lumina_atomic.c](../src/lumina_atomic.c:588)); host σ는 double
`8×1000×N`이다([lumina.h](../src/lumina.h:431)).

- 식별된 크기 의존 상주분: **14,530,712,000 → 19,164,792,743 byte**,
  증가 **4,634,080,743 byte = 4.316 GiB**.
- 여기에 같은 solver의 host matrix/rhs/pointer가 **17,767,557,800 byte** 상주한다
  ([lumina_cuda.cu](../src/lumina_cuda.cu:615)). 이를 합친 확인 가능 subtotal은
  **32,298,269,800 → 36,932,350,543 byte = 30.080 → 34.396 GiB**다.
- 충돌 lookup에는 이온별 `4×n_level²` 임시 dense map도 있다
  ([lumina_plasma.c](../src/lumina_plasma.c:796)); 표별 map은 line mapping 뒤 해제된다.
  event log와 pandas 생성 peak 등 고정/임시 workspace를 포함한 실제 MaxRSS는 아직
  측정되지 않았다. 그래서 계산 배치는 **64 GiB**를 요청한다
  ([sbatch_deck_regen_fullcov.sh](../scripts/sbatch_deck_regen_fullcov.sh:3)).

### 덱 파일 크기

현재 핵심 8파일의 실측 합은 **5,229,887,203 byte = 4.871 GiB**다. NPY와 σ binary는
layout으로 정확히 계산하고, CSV 네 개는 현재 파일의 bytes/row를 새 행 수에 적용했다.
σ layout은 32-byte header, level별 flag/padding, `N×1000` double이다
([expand_atomic_data_cmfgen.py](../scripts/expand_atomic_data_cmfgen.py:1245),
[writer](../scripts/expand_atomic_data_cmfgen.py:1383)); τ와 transition-probability는 각각
`L×50`, `E×50` double이다
([finalize_cmfgen_ref_npy.py](../scripts/finalize_cmfgen_ref_npy.py:210)).

| 핵심 파일군 | 현재 byte | full 예상 byte |
|---|---:|---:|
| levels.csv | 825,334 | 1,128,348 |
| line_list.csv | 390,625,345 | 514,878,220 |
| macro_atom_data.csv | 469,527,728 | 618,878,432 |
| macro_atom_references.csv | 861,532 | 1,177,835 |
| cmfgen_sigma_bf.bin | 212,762,624 | 290,876,392 |
| tau_sobolev.npy | 1,033,652,928 | 1,362,444,528 |
| transition_probabilities.npy | 3,100,958,528 | 4,087,333,328 |
| line2macro_level_upper.npy | 20,673,184 | 27,249,016 |
| **합계** | **5,229,887,203** | **6,903,966,099 (6.430 GiB)** |

증가는 **1,674,078,896 byte = 1.559 GiB**, 배수는 **1.320098×**다. 충돌 bin/H5와
작은 companion 파일은 신규 매핑 수가 생성되어야 확정되므로 이 합계 밖이다. 값을
채우거나 대체해 크기를 맞추지 않았다.

### 로딩 시간

현행 C loader는 `line_list.csv`를 열별로 10회 읽고
([lumina_atomic.c](../src/lumina_atomic.c:649)), macro topology도 별도 열로 읽은 뒤
대형 NPY를 순차 로딩한다([lumina_atomic.c](../src/lumina_atomic.c:482)). 같은 노드·스토리지
캐시 조건의 byte-bound 예측은 따라서 **현재 로딩 시간의 약 1.32×**다. 모델 런 금지와
계산 노드 제출 차단 때문에 절대 초 단위와 실제 MaxRSS는 측정하지 않았으며
`UNRESOLVED`로 남긴다.

## 2단계 — 생성 상태

계산 노드용 파이프라인은 준비했다.

- [deck_regen_fullcov_driver.py](../scripts/deck_regen_fullcov_driver.py:17): 정확한 신규 경로,
  gate 두 개 필수, 기존 경로와 기존 출력 overwrite 거부, rebuilt 파일은 새로 만들고
  나머지 companion은 실제 파일 복사.
- [sbatch_deck_regen_fullcov.sh](../scripts/sbatch_deck_regen_fullcov.sh:1): GPU/GRES 요청 없음,
  `CUDA_VISIBLE_DEVICES` 제거, `CMFGEN_FULL_LEVELS=1`, full topology/NPY/σ/충돌표/
  multiplicity/recombination target 생성 후 검증기를 마지막에 실행.
- 실행 명령: `sbatch scripts/sbatch_deck_regen_fullcov.sh`
- 제출 결과: exit 1, `Batch job submission failed: Unable to contact slurm controller
  (connect failure)`.

신규 디렉터리는 존재하지 않는다. 따라서 생성 완료라고 표기하지 않으며, 기존 덱에는
어떤 write도 하지 않았다. `src/`, `validation/regression_ledger/`,
`scripts/regression_ledger.py`도 수정하지 않았고 commit·GPU·모델 런도 하지 않았다.

## 3단계 — 검증 게이트

[verify_deck_regen_fullcov.py](../scripts/verify_deck_regen_fullcov.py:142)는 읽기 전용이며,
실패를 보고한 뒤 exit 1로 멈춘다. identity 전수 커버리지, 기존 CMF-common
881,085선 포함, 세 float64의
bit pattern, σ flag와 collision manifest의 동반 증가를 각각 검사한다
([gate 1](../scripts/verify_deck_regen_fullcov.py:156),
[gate 2](../scripts/verify_deck_regen_fullcov.py:165),
[gate 3](../scripts/verify_deck_regen_fullcov.py:172),
[gate 4](../scripts/verify_deck_regen_fullcov.py:195)).

| 게이트 | 결과 | 비고 |
|---|---|---|
| 1. 전 이온 coverage 1.0 | **NOT RUN** | 사전 열거상 S V·Co II 합계 45,255 CMFGEN선 미포함으로 FAIL 예상 |
| 2. 기존 CMF-common 881,085선 전부 포함 | **NOT RUN** | 신규 덱 없음 |
| 3. 공통 881,085선 A/f/λ bit 동일 | **NOT RUN** | 신규 덱 없음 |
| 4. σ·Υ 동반 확대 | **NOT RUN** | 신규 σ·collision manifest 없음 |

게이트를 통과시키기 위해 vintage, 값, 필터를 조정하지 않았다. 원자값이 없을 때
clamp/floor/대체값을 넣지 않는다. 충돌표 builder도 mapping 실패 시 bin을 만들지 않고
manifest에 사유를 기록하는 fail-closed 경로다
([build_cmfgen_coldata_all.py](../scripts/build_cmfgen_coldata_all.py:11)).

## Co IV Υ 부수 확인

collision H5는 parse된 `col_data`를 그대로 level pair와 Ω 배열로 쓴다
([expand_atomic_data_cmfgen.py](../scripts/expand_atomic_data_cmfgen.py:1450)). 계산용 collision
builder는 고정 `19apr23` source를 택한다
([build_cmfgen_coldata_all.py](../scripts/build_cmfgen_coldata_all.py:43))하고, 신규 ref의
모든 이온에 같은 builder를 적용한다
([build_cmfgen_coldata_all.py](../scripts/build_cmfgen_coldata_all.py:666)).

원본 비교 결과 Co IV `4,455/4,455` Ω 행이 Fe III Ω 행에 float64 bit 그대로 존재했고,
20점 temperature grid도 bit 동일했다. 이 감사가 배치 검증기에 들어 있다
([audit_coiv_proxy](../scripts/verify_deck_regen_fullcov.py:125)). Co IV 파일 헤더도
`Zha96_FeIII_col`, `Using FeIII values?`라고 명시한다
([col_data](../data/atomic/cmfgen/COB/IV/19apr23/col_data:12)). 따라서 단순 덱 재생성은
I1을 고치지 않는다. **실제 Co IV 충돌강도 원자자료를 별도로 교체하고 provenance를
새로 세우는 수리**가 필요하다.

## `-o` 요약

- **1단계:** 준위 **36,355**, edge **10,218,333**. 보수 GPU 요구량
  **110,569,158,168 byte (105,446.97 MiB)**. H200 143,771 MiB에 **적합**, 여유
  **40,185,661,928 byte (38,324.03 MiB)**.
- **2단계:** **미생성**. 예정 경로
  `data/tardis_reference_toy06_19p48d_sivcaiv_fullcov/`; CPU Slurm 제출이 샌드박스의
  controller socket 차단으로 실패. 기존 `_sivcaiv` 무변경.
- **3단계:** gate 1/2/3/4 모두 **NOT RUN**. gate 1은 현행 vintage 경로상 S V·Co II
  때문에 FAIL 예상이며 조정하지 않음.
- **Co IV Υ:** Fe III 대용 **해소 안 됨**; 별도 원자자료 수리 필요.
- **남은 UNRESOLVED:** 계산 노드에서의 생성 및 네 gate 실측, 절대 CPU MaxRSS·로딩
  초, S V·Co II vintage를 보존하면서 full coverage를 만드는 설계 변경, 실제 Co IV Υ,
  기존 설계의 I14/I15와 super-level cutoff 영향.
