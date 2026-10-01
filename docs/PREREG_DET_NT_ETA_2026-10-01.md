# 사전등록 — DET-NT-ETA: W5 기준 NT 율 축소의 유한 반복 종점 응답 (2026-10-01, 기안)

기안자 = astra, 개정25-r과 이번 직접 발주에 따른 **read-only 계약 기안자**. 구현·발주·실행·판정은 수행하지 않았다. 산출은 이 마크다운 본문이며, 운전자 파일화 목적지는 `docs/PREREG_DET_NT_ETA_2026-10-01.md` — **신설**이다. 판독기 구현은 운전자에게 속한다(개정25-w).

## §0. 선행 사실·설계 판단

### 설계 판단

- **생산 소스 변경 0, 재빌드 0, 판정런 1회**다. W5와 같은 바이너리·입력에서 `LUMINA_NT_FULL="1"`을 **`"0"`으로 명시적으로 변경**한다. 코드에서 `"0"`이 `ETA_NONTHERMAL=0.05` 경로를 선택함을 확인했다.
- **1차 관측은 최종 gen8 단계 배열의 IV/III**다. 반복별 q3가 없으므로 평균·표류·수렴 효과를 주장하지 않는다.
- **2차 관측은 단계·준위합 각각의 II/III**다. 양쪽 런의 k4·k5 로그 평균 차이와 k3·k5 표류로 셀을 네 갈래로 판정한다. 30셀을 모두 보고하며, 전체 분기에는 황 10셀의 **유일 최빈 결과**를 사용한다. 과반 조건은 추가하지 않는다.
- **FUV는 별도 측정량**이다. II/III 분기로 FUV 검출·미검출을 대신하지 않으며, FUV 효과 크기 게이트를 추가하지 않는다.
- **판독기 하나만 신설**한다. 동결된 W5 판독기의 순수 계산 함수를 재사용하되, W5 명령·분기·파일을 수정하지 않는다. 새 GPU 게이트·생산 계측·반복별 q3 덤프는 추가하지 않는다.
- 목표 산출은 **등록된 응답·분기·한계를 재현할 수 있는 증거 묶음**이다. 판독기 자체시험 통과나 런 정상 종료만으로 목표 달성을 선언하지 않는다.

### user 결정과 별도 결정 필요 항목

**확정:** user의 2026-10-01 「승인」에 따라 DET-NT-ETA를 기안·진행한다. 이번 범위 안에서 다시 선택받아야 할 항목은 없다.

다음은 **결과에 따른 후속 계약과 별도 user 결정이 필요한 항목**이다.

- 완전 SF 형 처방 또는 SF 대리 처방의 구현·검증.
- 반복별 q3 관측 추가, 반복 연장, 추가 대조런.
- 실패·시간 초과 후 재제출 또는 이번 자원 상한을 넘는 계산.

W5 판정문에 기록된 반복 연장 선택을 이번 NT 시험에 합치지 않는다. 이번 비교는 **W5 6회 반복 대 NT 시험 6회 반복**으로 고정한다.

### 경로와 증거 정체성

```text
P = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
T = /gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source
D = /gpfs/kjhan/lumina/cmfgen_map/fork_20261001
W = /gpfs/kjhan/lumina/cmfgen_map/w5_20260930
R = /gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1
B = /gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1
N = /gpfs/kjhan/lumina/cmfgen_map/nt_eta_20261001
E = /gpfs/kjhan/lumina/cmfgen_map/nt_eta_20261001/run1
```

N과 E는 **신설 예정 산출 경로**다. 기존 경로가 있으면 덮어쓰지 않고 운전자가 충돌 없는 경로를 확정하여 집행 기록에 남긴다. 경로 변경은 물리 설정 변경을 허용하지 않는다.

| 선행 사실 | 저자 확인 범위 |
|---|---|
| 프로젝트 정체성 | **[저자 실행]** `pwd`로 P 확인. 공유 맥락 전문과 현행 분장 문서를 조회했다. 저장소 P와 동결 생산 소스 T를 구별한다. |
| 갈림길 인계 | **[저자 실행]** D의 FORK·감사 r1–r4·r3 재현 기록을 판독했다. r4 끝의 「기안 인계 목록 확정본」을 본 계약의 변경·관측·분기·해석·증거 조항으로 승계한다. |
| 선행 계약·판정 | **[저자 실행]** `docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md`와 `docs/VERDICT_DET_W5_CONT_SOURCE_2026-10-01.md` 조회. W5-NONE은 선행 판정이며 이번 시험의 결과가 아니다. |
| NT 경로 | **[저자 실행]** T의 `src/lumina_plasma.c`에서 `atoi("0")=0`, `pref=ETA_NONTHERMAL=0.05` 경로와 단계 사다리·NLTE 행렬의 NT 소비를 확인했다. |
| W5 실행 | **[저자 실행]** R의 runner·exports·footer·빌드 기록 조회. job 409099, h100 두 장, CPU 16, 반복 6, footer 경과 39,269초, `terminal_state=SUCCESS`, `fixed_te_loader_consumed=1`. Slurm 독립 재조회는 **[미실행]**이다. |
| W5 바이너리 | **[저자 실행]** R의 `input/binary.sha256`와 footer에 기록된 전체 해시 일치 확인. 바이너리 실물 재해시는 **[미실행]**이다. |
| 출력 권위·세대 | **[저자 실행]** 반복별 ion CSV 머리글·첫 자료행과 최종 발행 manifest 조회. k3/k4/k5 인구 세대는 6/7/8이며 반복별 q3는 없다. 최종 단계 배열은 gen8이다. |
| 판독 선례 | **[저자 실행]** W5 판독기의 인구·창 적분·방출 적분·검증 경로와 프리플라이트 검사 형식을 판독했다. |
| 계산·변경 | **[미실행]** 파일 작성·수정, 전체 소스 manifest 검증, 원자료 독립 재적분, 판독기 구현·NC, 기계 프리플라이트, 빌드, 스테이징, GPU 실행, 신규 결과 판정. |

이번 저자가 직접 확인한 파일 해시:

| 대상 | SHA-256 |
|---|---|
| D/FORK.md | `c70056c572ef48274ed694fcd50aabb089fbedc7ac01ca2dd065e8b2561ded02` |
| D/FORK_r3.md — 보존본 | `e7b9b5fc53590e6a9d02b94db338610467c38b6864913d1563b30512605547c0` |
| D/astra_audit_r1.md | `96a56af025e2d2fc480b64a3d4da638666c9934724c5a1c5f5193fb457f78196` |
| D/astra_audit_r2.md | `170983345b2b9b5513151138db24b544d7e38981c8f74d13675aa486b83e26e9` |
| D/astra_audit_r3.md | `d7bee6f1ef7a61555374cffb0cd5dc2229740131e7599761f5b02f1b5cb7df1f` |
| D/astra_audit_r4.md | `af23cdf1b204e95850b08135d0d34d2b395f12f4ecf9e109f3c0003f5615aaae` |
| D/r3_repro.txt | `741ab6c29cdbdc7861eab1da66932ca00b9b32e2c952e043c4dbee5f22f04dbf` |
| P/scripts/w5_cont_readout.py | `7008f709e4ca49a24823e4f88f452c42448247d8cb5940c1c1670c4e31a7f3cd` |
| T/src/lumina_plasma.c | `d66e34f3a601397b6270559604241c74238188aa2ed66b6e99cb3edffaff20c4` |

W5 기록에서 승계하는 바이너리 해시는 다음이다.

```text
7800484fb309addabfa5b2f6b879f3ab09b814db83d3c5db2f39debff3f1b390
```

운전자는 제출 전에 실물 대조로 G-ID를 완성한다. 선행 감사의 재현 주장과 이번 기안자의 조회를 서로 대체하지 않는다.

## §1. 물리 계약 하나

**질문:** W5와 같은 입력·바이너리·FIXED_TE·6회 반복에서 기존 NT 율 계수만 1에서 0.05로 낮추면, 지정 중간 셸의 최종 IV/III와 반복별 II/III가 등록 방향·크기로 응답하며 FUV와 보조량은 어떻게 동반 변화하는가?

변경은 다음 하나다.

```text
W5:  declare -x LUMINA_NT_FULL="1"
시험: declare -x LUMINA_NT_FULL="0"
```

기존 계산식은

\[
\Gamma_{\rm nt}
=p\,\frac{q_{\rm dep}}{35\,{\rm eV}\,n_{\rm atom}},
\qquad p=1\ \text{또는}\ 0.05
\]

이다. **같은 침적·원자수에서 NT 율을 1/20로 낮추는 감도 시험**이다. 재풀이 뒤 모든 인구비가 1/20이 된다는 뜻은 아니다.

ETA=0.05는 SF 이온화 에너지 분율이 아니다. toy06 해당 셸의 SF 분율이나 CMFGEN 대비 NT 과대 배수를 이 값에서 추론하지 않는다. 이 스위치는 단계 사다리와 NLTE 행렬의 NT 항을 함께 바꾸므로, IV 매개 경로만 격리한 개입도 아니다.

소스·바이너리·원자자료·침적 입력·초기 조건·반복 수·수렴 허용오차·W5 연속 원천 설정·W2 인구 권위·FIXED_TE를 유지한다. MC 계산, 새 생산 계측, 재빌드, 반복 연장은 포함하지 않는다.

## §2. 설계

### 2.1 대조·시험 구성

기준선은 R의 W5 실물이며 C1이나 연장 W5 결과로 교체하지 않는다. E는 W5 봉인 입력에서 새로 스테이징한다. W5 최종 상태를 새 warm start로 사용하지 않는다.

exports를 중복 키 없는 사전으로 읽고 비교한다. 다음 경로 키는 **R→E 접두 경로 이식과 동일한 상대 대상**만 허용한다.

- `LUMINA_MODEL_DIR`
- `LUMINA_CMFGEN_SIGMA_BF`
- `LUMINA_DEPOSITION_FILE`
- `LUMINA_FIXED_TE_PROFILE`
- `LUMINA_IONIZATION_REFERENCE_FILE`
- `LUMINA_TOPION_LEVELS_FILE`
- `LUMINA_PHYSICS_COMPARISON_DIR`

이 이식을 정규화한 뒤 설정 차이는 `LUMINA_NT_FULL: "1"→"0"`뿐이어야 한다. 임의 문자열·경로 차이를 일괄 삭제하여 비교하지 않는다.

특히 다음을 유지한다.

```text
LUMINA_PURE_CMFGEN_ITER="6"
LUMINA_DET_TRANSACTIONAL="1"
LUMINA_CMF_FINE_MGPU_DEVICES="2"
LUMINA_CMF_FINE_CONT_NLTE_SOURCE="1"
LUMINA_ION_NT="1"
LUMINA_POP_DJDN="1"
```

### 2.2 관측 권위와 세대

원소는 S·Si·Ca, 즉 Z=16·14·20이며 셸은 `{10,15,19,24,30}`이다. 파일의 `stage` 및 q 색인은 II=1, III=2, IV=3으로 읽는다.

| 관측 | 입력 | 사용 권위·세대 |
|---|---|---|
| IV/III | 양쪽 런의 `work/lumina_ion_pops.csv` | 최종 단계 배열, population gen8. stage 3/2 |
| II/III | 양쪽 런의 `work/physics_comparison/physics_DET_iter000{3,4,5}.ion.csv` | `Z{Z}_stage_q{1,2}_cm3`, `Z{Z}_nlte_q{1,2}_cm3`를 별도로 사용. gen6/7/8 |
| 세대·완주 | 반복 manifest와 `work/cmfgen_final_publication.json`, 로그·footer | 반복 COMMITTED, 최종 complete, 발행 권위·세대 대조 |
| FUV | 양쪽 런의 `work/lumina_spectrum.csv` | 최종 관측자계 \(L_\lambda\), erg s⁻¹ Å⁻¹ |
| A 대역 J/B | k5 spectral·shell CSV | sh24, radiation gen7와 FIXED_TE |
| D 대역 Γ 몫 | k5 spectral·shell CSV, 최종 준위 CSV, 봉인 σ 입력 | 아래 §5.4의 발행 상태 후처리 진단 |

최종 단계 CSV 자체의 세대와 최종 manifest를 함께 확인한다. 파일명이 `final`이라는 이유만으로 gen8을 가정하지 않는다. 반복별 q3·IV 준위합·III→IV 채널 덤프를 요구하지 않는다. 기존 채널 덤프는 II→III만 담는다.

### 2.3 판독기 최소안

**새 `scripts/nt_eta_readout.py` 하나**를 선택한다. W5 판독기 확장보다 기존 W5 명령·분기·봉인 신원을 건드리는 면적이 작다.

동결된 `scripts/w5_cont_readout.py`에서 `log_planck`, `logsumexp`, `window_ratios`, `trapz_band` 등 필요한 순수 계산을 재사용할 수 있다. 기존 `cmd_readout`·W5 분기·선 선택·전체 선 원장 처리는 호출하지 않는다. 중복 행을 사전으로 덮어쓰는 기존 로더에 검증을 맡기지 않고, 새 판독기의 입력 검사에서 중복·결손·세대를 먼저 확인한다.

D 대역 Γ는 D의 `gamma_bands_v2.py` 중 **광이온화 대역합에 필요한 부분만** 같은 규칙으로 구현한다. 재결합 분모·반사실 평형·III→IV 재적분을 추가하지 않는다. `gamma_bands_ion.py`의 IV 분배함수 결함을 승계하지 않는다.

판독기는 다음 기능을 한 파일 안에서 제공한다.

- 제출 전 G-ID 검사.
- 아래 판독 NC와 정상 대조.
- 완주 후 G-ID·G-RUN 및 필수 입력 유효성 검사.
- §5의 원자료 표·집계·전체 분기·보조량·실패 사유 출력.

단순한 `"PASS"` 문자열을 받아 게이트를 통과시키지 않는다. 각 게이트 결과는 구체 입력·해시·검사 결과와 연결한다.

## §3. 변경집합·산출물·자원

### 변경집합 — DET-NT-ETA (표)

| 경로 | 변경 | 범위 |
|---|---|---|
| `scripts/nt_eta_readout.py` | **신설** | `nt_eta_` 접두의 판독·검증·NC. §4·§5 구현. 운전자 담당 |

### 변경집합 끝

생산 소스 T와 W5 판독기는 변경하지 않는다. 새 C/CUDA 코드·환경 키·계측·GPU 시험·빌드는 없다.

사전등록·판정문·런 전용 exports·runner·증거 파일은 운전자의 문서화·스테이징 산출이다. 기존 W5 스테이징 스크립트와 runner를 제자리 편집하지 않는다. W의 `stage_run1.sh`는 절차 선례이며, E 생성용으로 경로를 바꾸지 않은 채 실행하지 않는다.

### 3.1 산출물과 보존

N 아래에 다음을 만든다. 모두 **신설 산출물**이며 기존 출력 디렉터리에 덮어쓰지 않는다.

| 위치 | 내용 |
|---|---|
| `run1/` | 봉인 입력, 런 전용 job.slurm, 실제 exports, 바이너리·빌드 신원, stdout/stderr, footer, 기존 생산 출력 |
| `reader_selftest/` | 정상 대조·NC 입력과 기대값·실측 분기·실패 사유·rc |
| `evidence/` | 제출 전후 G-ID, Slurm 상태·자원 기록, 계약·판독기·의존 파일 해시, 실제 명령 |
| `readout/result.json` | 게이트, 전체 분기, 사유, 황 셸 수·10셀 집계, 주장 상한 |
| `readout/population34.csv` | IV/III 15셀의 양쪽 인구·로그비·δ₃₄·세대 |
| `readout/population23.csv` | II/III 30셀의 양쪽 k3/k4/k5·평균·표류·D·T·δ₂₃·셀 결과 |
| `readout/auxiliary.json` | FUV 적분·로그 변화, A 대역 J/B, D 대역 Γ 몫과 권위·세대·상태 |
| `readout/evidence_manifest.json` | 실제 소비 파일·판독기·게이트 증거의 절대 경로와 해시 |

최종 판정문은 운전자가 `docs/VERDICT_DET_NT_ETA_2026-10-01.md` — **신설**로 작성한다. 30셀 결과를 전부 싣고 원자료 표를 연결한다.

W5·갈림길 평가·보존본·기존 감사·실패 산출은 보존한다. 실패 시 부분 결과를 삭제하거나 다시 사용하지 않는다. 읽기만 하는 원자료와 쓰기 가능한 판독 출력 루트를 분리한다.

### 3.2 자원과 runner

**판정런 한 번**, 단일 노드, H100 또는 A100 80 GB급 동일 종류 GPU 두 장이다.

```text
--partition=h100,a100
--nodes=1
--ntasks=1
--gres=gpu:2
--cpus-per-task=16
--mem=64G
--time=30:00:00
--no-requeue
--exclude=syn102
```

H200은 금지한다. R의 `job.slurm`에 있는 `h100-a100x2` 실운영 프로필의 H200 제거 상태와 GPU 종류·메모리·장수 검사를 유지한다. 이름이 같은 과거 원형 runner로 바꾸지 않는다.

**시간 근거:** W5 footer의 39,269초는 약 10.9시간이며, H100 두 장 기준 예상 비용은 약 **21.8 GPU·시간**이다. `--time=30:00:00`은 W5 runner의 상한을 유지하여 A100 및 실행시간 변동 여유를 둔 값이다. 예상 실행시간을 30시간으로 주장하지 않는다. 이번 할당의 최대 비용은 **60 GPU·시간**이며, 시간 초과 시 자동 연장·재큐잉·재제출하지 않는다. A100의 10.9시간 완주는 실측되지 않았다.

런 전용 job.slurm은 W5 실행 본체를 유지하고, 작업명·로그 목적지·위 자원 설정과 재현 가능한 모듈 초기화만 운영 변경으로 기록한다. 스크립트 안에서 `module purge` 후 W5 기록의 GNU 13.2.0·CUDA 13.0.2에 맞는 모듈을 명시적으로 로드하고 실제 모듈 목록·라이브러리 해석 결과를 남긴다. 재빌드하지 않는다.

runner의 인자는 **절대 run root 한 개**다. 제출 형식은 다음과 같다. 이 문서에서 제출을 실행한 것은 아니다.

```bash
sbatch \
  --partition=h100,a100 --nodes=1 --ntasks=1 \
  --gres=gpu:2 --cpus-per-task=16 --mem=64G \
  --time=30:00:00 --no-requeue --exclude=syn102 \
  --output="$E/slurm-%j.out" --error="$E/slurm-%j.err" \
  "$E/job.slurm" "$E"
```

경량 판독기 시험·프리플라이트·후처리는 프로젝트 규약의 grammar-debug에서 실행한다. GPU 계산은 Slurm에서만 수행하며, 새 GPU 사전 시험을 추가하지 않는다. 기존 출력 구성을 유지하므로 추가 생산 덤프는 없고, 기존 원장도 임의로 끄지 않는다.

## §4. 게이트

| 게이트 | 수용 기준 | 소비 |
|---|---|---|
| **G-ID** | §0·§2.1·§3의 바이너리·소스·입력·exports·FIXED_TE·자원·판독기 신원 일치 | W5 대비 단일 설정 변경 성립 |
| **G-READOUT** | 아래 정상 대조와 판독 NC의 기대 분류·실패 검출 전부 충족 | 분기 계산과 결손 처리의 신뢰 |
| **G-RUN** | 정상 완주, 등록 반복·세대·필수 파일·필수 수치 유효성 충족 | §5 전체 분기의 적용 가능성 |

제출 전 G-ID와 G-READOUT을 충족한다. 완주 후에는 실행 증거를 포함해 G-ID를 완성하고 G-RUN을 평가한다. 하나라도 실패하거나 증거가 없으면 전체 분기는 **NT-X**이며, 이유를 기록한다. 새 물리 효과 크기 게이트나 GPU 계측 게이트는 없다.

### 4.1 G-ID의 피연산자

- **바이너리:** B의 `lumina_cuda`·`binary.sha256`, R과 E의 `input/lumina_cuda`·`input/binary.sha256`, 각 `input/run_build_identity.json`, stage 기록을 대조한다. 모두 §0의 W5 바이너리 해시에 연결되어야 한다.
- **소스:** B와 R/E의 `source_sha_at_build.txt`를 대조하고, T에서 등록 manifest의 파일을 검증한다. W의 `APPROVED_SHA.txt`에 있는 경로별 해시가 같은 빌드 기록과 일치해야 한다. 이번 작업의 생산 소스 편집·추가·재빌드는 허용하지 않는다.
- **입력:** R/E의 `input/deck.sha256`, `input/topion.sha256`, `input/sigma.sha256`와 그 실물을 대조한다. 모델·원자자료·침적·초기 입력의 내용은 W5와 같아야 한다.
- **exports:** R/E의 `input/resolved_lumina.exports`를 §2.1 방식으로 비교한다. 경로 이식 외 차이는 `LUMINA_NT_FULL` 한 키뿐이며 시험 값은 명시적 `"0"`이다. 상속된 `LUMINA_*`를 지운 뒤 봉인 exports를 소비하는 W5 절차를 유지한다.
- **FIXED_TE:** R/E의 `input/fixed_te_profile.txt`가 동일해야 한다. 실행 후 `fixed_te_loader_consumed=1`, 최종·반복 manifest의 FIXED_T 및 온도 manifest, 실제 발행 \(T_e\)의 프로필 일치를 기존 runner 기준으로 확인한다. 고정 파일 해시와 정규 온도 manifest 해시를 서로 같은 종류의 해시로 비교하지 않는다.
- **실행·판독 신원:** 실제 GPU·장수·노드·재시작 횟수·반복 수, runner 해시, 새 판독기와 동결 의존 파일 해시를 증거에 연결한다.

소스·입력 검증은 등록 manifest의 명시된 파일을 대상으로 한다. 공유 파일시스템 전체를 탐색하지 않는다.

### 4.2 G-RUN과 필요한 파일

정상 완료는 Slurm `COMPLETED`, 모델 rc=0, footer `terminal_state=SUCCESS`, DET 최종 COMPLETE와 기존 exact solve 성공 조건으로 확인한다. 정상 종료와 물리 수렴을 동일시하지 않는다.

양쪽 런에서 다음을 요구한다.

- k0–k5 반복 manifest와 등록 반복의 완주 증거.
- k3/k4/k5 ion CSV: 대상 셸·원소·권위 완비, 중복 없음, 인구 세대 6/7/8.
- 최종 publication manifest와 단계 CSV: complete, population/required population gen8, 대상 III·IV 인구 및 온도 세대 일치.
- 최종 spectrum CSV: §5.4 두 적분 구간 전체 지원.
- k5 spectral·shell CSV: 등록 셸·주파수 격자·온도·세대 확인.
- 최종 준위 CSV와 봉인 σ: D 대역 Γ에 쓰는 준위·인덱스·격자 대응 확인.

필수 로그비의 인구가 비양수·비유한이거나 행·파일·세대가 결손이면 미정의로 표시하고 NT-X로 처리한다. 중복 행을 합치거나 마지막 행으로 덮어쓰지 않는다. 임의 floor·clamp·대체 세대·부분 셸 집계로 결과를 만들지 않는다.

FUV·보조량의 **값이나 부호에는 합격 문턱이 없다**. 다만 등록한 필수 측정을 만들 자료가 결손이면 증거 미완성으로 NT-X다. 이 경우와 실제 FUV 증가·감소·무변화를 구분한다.

### 4.3 판독 NC

NC는 합성 입력 또는 격리된 작은 입력 사본으로 수행한다. 생산 소스·W5 원자료를 변조하지 않는다.

| 묶음 | 필수 시험 |
|---|---|
| 셀 네 분할 | \(3D<0.30\), \(=0.30\), \(>0.30\)에서 \(\delta_{23}=\pm T\), 바로 안쪽·바깥쪽을 검사. 정확 경계는 검출에 포함한다. |
| 과거 중첩 반례 | D=0.15, δ₂₃=−0.60은 (a)만 성립해야 한다. `3D>0.30`만으로 (d)를 붙이는 결함을 검출한다. |
| 양쪽 표류·평균 | W5 쪽 표류만 큰 경우도 D에 반영. 로그 평균 대신 산술 인구비 평균을 쓰거나 k3/k5를 잘못 소비하는 결함을 검출한다. |
| 황 집계 | (a,b,c,d)=(4,3,2,1)은 (a). 최다 동률은 불능. 10셀 건수의 가능한 분포를 검사하며 Si·Ca가 황 분기를 바꾸지 않아야 한다. |
| 1차 경계·전체 분기 | δ₃₄=−0.5 포함. 황 통과 셸 2와 3의 경계, 각 y₂₃ 집계 결과와 전체 분기의 조합을 검사한다. 1차 불성립이면 셀 (b)가 있어도 전체는 NT-X다. |
| FUV 독립성 | 1차 성립, y₂₃=(c), FUV 로그 변화 −0.7인 경우 NT-IV-ONLY를 유지하면서 FUV −0.7을 별도로 출력한다. |
| 결손·신원 | 필수 행·파일 누락, 중복, NaN/Inf, 비양수 로그 인구, 잘못된 세대, 바이너리·소스·FIXED_TE 불일치, exports 추가 차이를 주입하면 NT-X와 해당 사유를 낸다. |
| 보조 적분 | 경계 사이를 가르는 선형 스펙트럼의 알려진 적분, 지원 부족 거부, Γ 대역 경계·가중·격자 불일치 검출을 확인한다. |

정상 대조도 함께 통과해야 한다. 통계 함수에 W5를 양쪽으로 넣으면 δ₃₄=δ₂₃=0이며 **1차 불성립에 따른 NT-X**가 맞다. 이 통계 대조를 실제 노출 설정이 다른 두 런의 G-ID 통과로 표시하지 않는다.

경계 판단에는 반올림 표시값이나 임의 허용오차를 사용하지 않는다. 부동소수 경계 시험은 실제 비교 피연산자의 정확 경계와 인접 표현 가능 수를 사용한다. NC의 성공은 주입 결함을 기대한 사유로 거부했다는 뜻이며, 단순히 하네스가 종료했다는 뜻이 아니다.

## §5. 기대치·판정량·분기

### 5.1 1차: 최종 IV/III

런 \(r\in\{E,R\}\), 원소 Z, 지정 셸 s에 대해

\[
y_{34}^{r}(Z,s)
=\log_{10}\frac{n_{\rm IV}^{r}(Z,s,\mathrm{gen8})}
                  {n_{\rm III}^{r}(Z,s,\mathrm{gen8})},
\qquad
\delta_{34}(Z,s)=y_{34}^{E}(Z,s)-y_{34}^{R}(Z,s).
\]

단계 배열만 사용한다. 전체 15셀을 보고하고, 전체 분기에 들어가는 조건은

\[
I_{34}
=\left[
\#\{s:\delta_{34}(Z=16,s)\le-0.5\}\ge3
\right]
\]

이다. **황 5셸 중 3셸 이상**, 정확히 −0.5인 셸도 포함한다.

선행 고정 장 계산의 황 기대 방향은 감소이며 f=0.05 감도는 약 −0.73…−1.13 dex였다. 이는 새 재풀이 결과의 보증이나 별도 수용 범위가 아니다. 등록 문턱 0.5 dex는 **6회 반복 종점의 응답 크기**다. IV/III 반복 표류와 수렴 효과는 측정하지 않는다.

### 5.2 2차: 반복별 II/III의 네 분할

각 Z·권위 \(A\in\{\mathrm{stage},\mathrm{levels}\}\)·셸 s를 하나의 셀로 둔다. 3원소×2권위×5셸의 **30셀**이다.

\[
y_{23,k}^{r}(Z,A,s)
=\log_{10}\frac{n_{\rm III,k}^{r,A}}{n_{\rm II,k}^{r,A}},
\qquad
\bar y_{23}^{r}=\frac{y_{23,4}^{r}+y_{23,5}^{r}}2,
\]

\[
\delta_{23}=\bar y_{23}^{E}-\bar y_{23}^{R},
\]

\[
d_E=\frac{|y_{23,5}^{E}-y_{23,3}^{E}|}{2},\qquad
d_R=\frac{|y_{23,5}^{R}-y_{23,3}^{R}|}{2},\qquad
D=\max(d_E,d_R),\qquad T=\max(0.30,3D).
\]

D는 등록한 반복 표류 대리량이다. 통계 오차·수렴 보증으로 해석하지 않는다.

유한한 δ₂₃와 D≥0에 대해 다음을 그대로 적용한다.

| 조건 | 셀 결과 |
|---|---|
| \(\delta_{23}\le-T\) | **(a) 기대 방향 응답 검출** |
| \(\delta_{23}\ge T\) | **(b) 반대 방향 응답 검출**, 셀 표지 `NT-REV` |
| \(|\delta_{23}|<0.30\ \land\ 3D\le0.30\) | **(c) 등록 크기 응답 없음** |
| \(3D>0.30\ \land\ |\delta_{23}|<3D\) | **(d) 판정 불능** |

네 조건은 배타적이며 전체를 덮는다. **δ₂₃=±T는 검출**, **3D=0.30의 문턱 미만은 (c)**다. `3D>0.30`만으로 무조건 불능으로 처리하지 않는다.

30셀 각각의 양쪽 y·평균·d·D·T·δ·결과와 전체 결과 분포를 보고한다. 단계와 준위합의 결과를 일치시키거나 유리한 권위를 선택하지 않는다.

### 5.3 황 집계와 전체 분기

전체 분기에 사용하는 표본은 **황의 단계·준위합×5셸 = 10셀**이다. 건수를 \(n_a,n_b,n_c,n_d\)로 둔다.

- 최다 건수를 가진 결과가 하나면 그 결과를 사용한다.
- 최다 건수가 동률이면 **집계 동률에 따른 불능**이다.
- 과반수 조건을 추가하지 않는다. `(4,3,2,1)`은 (a)다.
- 단계·준위합을 독립 확률 표본으로 해석하지 않는다.

게이트와 필수 자료가 유효할 때 전체 분기는 다음과 같다.

| 황 IV/III 조건 | (a) 유일 최빈 | (b) 유일 최빈 | (c) 유일 최빈 | (d) 유일 최빈 또는 최다 동률 |
|---|---|---|---|---|
| 3셸 이상 δ₃₄≤−0.5 | **NT-SUPPORT** | **NT-REV** | **NT-IV-ONLY** | **NT-UNDET** |
| 위 조건 불성립 | **NT-X** | **NT-X** | **NT-X** | **NT-X** |

게이트·필수 자료 결손은 위 표보다 먼저 **NT-X**로 처리한다.

| 분기 | 허용 해석·후속 |
|---|---|
| NT-SUPPORT | NT 의존성 및 사슬 가설 지지, 매개 경로 미분리. FUV를 포함한 사슬 지지 여부는 실제 FUV 결과까지 함께 보고 판단한다. 완전 SF 형 계약·반복별 q3 관측은 후속 결정 대상으로 남긴다. |
| NT-IV-ONLY | **II/III의 등록 크기 응답 미검출; FUV 변화는 별도 보고.** 다른 셀의 검출이 공존할 수 있다. IV 사슬 밖으로 원천을 배제하지 않는다. |
| NT-REV | 반대 방향 응답. 등록 사슬 가설은 미지지로 처리하되 NLTE 행렬 NT 항·전자밀도·재순환 등의 원인은 미분리로 남긴다. |
| NT-UNDET | IV 응답과 II/III 판정 보류를 기록한다. 원인을 **셀 (d)의 유일 최빈**과 **집계 동률**로 구별한다. 동률을 반복 표류 탓이라고 단정하지 않는다. |
| NT-X | 유효 자료에서 1차 불성립이면 고정 장 감도와 재풀이의 괴리를 기록하고 NT 소비 배선·POP_DJDN·분모 검사를 후속 후보로 둔다. 게이트·자료 실패이면 물리 응답 결론을 내리지 않는다. 두 원인을 사유 코드로 구별한다. |

셀의 `NT-REV`와 전체 분기 `NT-REV`는 별도 필드다. 전체 NT-X에서도 확보된 셀 결과는 사유와 함께 보존한다. 어느 분기도 추가 실행을 자동 발주하지 않는다.

### 5.4 FUV·보조량

**FUV:** 최종 스펙트럼에서

\[
F_r=\frac{L_r(912\!-\!1500\ {\rm \mathring A})}
          {L_r(4000\!-\!6000\ {\rm \mathring A})},
\qquad
\Delta_{\rm FUV}=\log_{10}F_E-\log_{10}F_R.
\]

각 구간의 양 끝을 인접 표본으로 **선형 보간**하고, 내부 표본과 함께 **파장에 대한 사다리꼴 적분**을 한다. 구간 전체가 저장 지원 안에 있어야 하며 외삽·부분 지원 적분을 허용하지 않는다. 파장 중복·비유한 수치와 로그를 정의할 수 없는 적분값은 명시적으로 처리한다. 분자·분모 적분, 분율, 로그 변화, 지원 범위를 함께 출력한다.

FUV 감소는 기대 방향으로만 기록한다. 새 검출 문턱을 만들지 않으며, **II/III 분기로 FUV 결론을 대신하지 않는다.** 분율은 형태 비교이지 절대 광도·내부 J의 배수가 아니다.

**A 대역 J/B:** 양쪽 런의 k5·sh24에서 저장 격자 청색 끝부터 531 Å까지 W5의 창 정의를 승계한다.

\[
R_A=
\frac{\sum_bJ_b\,\Delta\nu_{b\cap A}}
     {\sum_bB_\nu(\sqrt{\nu_{b,\rm lo}\nu_{b,\rm hi}},T_e)
             \,\Delta\nu_{b\cap A}}.
\]

주파수 경계 중첩 폭을 사용하고, 양쪽 값·로그 변화·지원 범위·세대를 보고한다. J/B를 총소스나 원천 위치의 판정량으로 바꾸지 않는다.

**D 대역 Γ 몫:** 선행 `gamma_bands_v2.py`의 기본 관측 범위인 **S II·Si II, sh10·sh24, k5**에서 LTE·NLTE 준위 가중을 따로 보고한다. D는 `[912,1500)` Å이며,

\[
f_D^A=
\frac{\sum_l w_l^A\Gamma_{l,D}}
     {\sum_l w_l^A\Gamma_{l,\mathrm{all}}},
\quad
\Gamma_{l,\mathcal B}
=4\pi\!\!\sum_{\substack{b:\lambda_b\in\mathcal B\\
                       \nu_b\ge\nu_{{\rm th},l}}}
 \frac{\sigma_l(\nu_b)J_b}{h\nu_b}\Delta\nu_b.
\]

선행 규칙대로 bin 중심 파장으로 대역을 분류하며, FUV 적분의 경계 보간 규칙과 혼동하지 않는다. LTE 가중은 해당 이온 준위의 \(g_l e^{-E_l/kT_e}\) 정규화, NLTE 가중은 최종 준위 인구의 이온 내 정규화다. 분모는 **전체 저장 지원의 광이온화 Γ**이며 NT·충돌률을 더하지 않는다. 양쪽의 분자·분모·몫·몫의 차이를 보고한다.

이 진단은 **k5 radiation gen7에 최종 population gen8 가중을 적용한 명시적 후처리**다. 솔버가 같은 물질 세대에서 소비한 Γ의 복원이라고 주장하지 않는다. 단계·준위합 인구 판정 권위와 Γ 가중 권위도 구별한다. 새 fine 적분·격자 정제·생산 계측은 하지 않는다.

## §6. 주장 상한

- ETA=0.05는 같은 침적에서 NT 율을 1/20로 낮추는 감도값이다. SF 분율·검증된 SF 근사·toy06의 실제 이온화 에너지 몫으로 부르지 않는다.
- IV/III의 변화는 **6회 반복 gen8 종점 응답**이다. 반복별 q3 표류와 수렴 효과는 미측정이다. II/III의 D도 수렴 인증이 아니다.
- IV/III·II/III·FUV 공동 응답은 NT 의존성과 사슬 가설을 지지할 수 있으나 **IV 매개 재결합 경로를 단독 식별하지 못한다**.
- 미검출·반대 응답·1차 불성립을 NT 무관, 사슬 배제, 원인별 기여율 상한으로 바꾸지 않는다. 상쇄·재순환·유한 반복·동시 NT 소비 경로가 남는다.
- 황 10셀 최빈은 등록된 요약 정책이다. 30셀 전체의 동일 반응이나 독립 표본 통계로 확대하지 않는다. NT-IV-ONLY에도 일부 셀 검출이 공존할 수 있다.
- 원자자료·준위 대응, 외곽 미수렴, W2 가중, H1 bin 혼합 등 선행 미해결 범위를 유지한다. 고정 장 감도는 재풀이 수리 상한이 아니다.
- 독립 σ 전수 재적분, 동일 조건 fine/bin 복사장·bf 적분, CMFGEN 내부 J·준위 인구, 솔버 η 전체 복원의 **재현 BLOCKED 또는 자료 부재**를 유지한다. 이번 보조 판독으로 이를 해소했다고 표시하지 않는다.
- 완전 SF 형 처방과 반복별 q3 관측은 결과에 따른 후속 계약이다. 이 단의 응답만으로 최종목적인 MC 수송 스펙트럼 재현을 달성했다고 주장하지 않는다.

## §7. Q-GOAL·Q-LEAN 자기답·집행 기록

### Q-GOAL 자기답

**자기답: PASS. 독립 감사 판정은 아니다.**

최종목적은 MC 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현하는 것이다. 이번 단은 그 전제인 이온 인구·복사장 계산에서 NT 처방의 감도를 공개 IV/III 간극과 연결된 표적으로 검사한다. 이를 생략하면 기존 NT 처방의 큰 고정 장 감도가 재풀이에서도 나타나는지 모른 채 다음 코드 수리를 선택하게 된다.

이번 목표는 NT 주원인 확정이 아니라 **NT 감도와 한계를 관측하여 후속 처방·관측의 필요성을 판단할 증거를 만드는 것**이다.

### Q-LEAN 자기답

**자기답: APPROPRIATE. 독립 감사 판정은 아니다.**

동일 바이너리·exports 한 키·판정런 한 번을 유지한다. 생산 코드·계측·GPU 게이트·재빌드·반복별 q3 추가를 하지 않는다. 새 판독기 하나에 입력 검사·분기·직접 NC를 묶고, 동결 W5 계산 함수를 재사용한다.

| 남기는 증거 | 소비 |
|---|---|
| 바이너리·소스·exports·W5 입력·FIXED_TE | 단일 변경 비교의 성립 |
| gen8 IV/III 15셀 | 1차 응답과 황 3/5 조건 |
| II/III 30셀·양쪽 표류 | 셀 네 분할, 황 유일 최빈과 동률 |
| FUV·A J/B·D Γ 몫 | 동반 응답의 제한된 해석 |
| 완주·세대·필수 파일 검사 | 결손을 물리 무응답으로 오인하지 않기 |
| 판독 NC | 이미 발생한 경계 중첩·FUV 귀속·집계 오류 방지 |

선 원장 전량 재분석, 새 공개 코드 재적분, III→IV 채널 복원, 원인별 몫 추정은 이번 판독 요구에 추가하지 않는다. 기존 독립 재현의 공백은 그대로 명시한다.

### 집행 기록

| 단계 | 담당·현재 상태 | 남길 기록 |
|---|---|---|
| 표적·진행 결정 | user, 2026-10-01 「승인」 | DET-NT-ETA 진행 |
| 갈림길 평가 종료 | Fable ↔ astra 선행 기록 | FORK 정본·감사 r1–r4·보존 r3·r3_repro의 정체성 |
| 계약 기안 | astra, 이번 read-only 본문 | 모델 ID, 저자 실행/미실행 |
| 파일화·기계 검사 | 운전자, 미집행 | 문서 해시, 아래 명령·rc |
| 계약 사전감사 | 개정25-r에 따른 운전자 발주·기록, 미집행 | 실제 담당·산출·판정. 갈림길 r4를 새 계약 감사로 재명명하지 않음 |
| 판독기 구현·직접 NC | 운전자, 개정25-w, 미집행 | 패치·해시·정상 대조·결함 검출 |
| 스테이징·제출 | 운전자, 미집행 | G-ID, runner 인자, exports 차이, job·자원 |
| 완주·판독·판정 | 운전자, 미집행 | 게이트·전체 분기·30셀·보조량·한계 |
| 종료 처리 | 현행 분장에 따라 운전자 기록 | 실제 감사 여부·저자·재현 상태·보존 경로 |

운전자가 판독기 구현과 판정을 겸한다는 점을 남긴다. 이번 기안자의 자기답과 선행 루프 감사는 독립 실행 검증을 대신하지 않는다.

### 운전자 재현·프리플라이트

다음은 **파일화 후 실행할 명령**이다. 기안자는 실행하지 않았다. P를 작업 디렉터리로 하여 grammar-debug에서 수행한다.

```bash
timeout 5m python3 -B scripts/check_prereg_preflight.py \
  docs/PREREG_DET_NT_ETA_2026-10-01.md --root .

timeout 5m python3 -B scripts/check_gate_referents.py \
  docs/PREREG_DET_NT_ETA_2026-10-01.md --root .

timeout 5m python3 -B scripts/check_doc_canonical.py \
  docs/PREREG_DET_NT_ETA_2026-10-01.md \
  docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md \
  docs/VERDICT_DET_W5_CONT_SOURCE_2026-10-01.md \
  CLAUDE.md \
  /gpfs/kjhan/lumina/cmfgen_map/fork_20261001/FORK.md \
  /gpfs/kjhan/lumina/cmfgen_map/fork_20261001/astra_audit_r1.md \
  /gpfs/kjhan/lumina/cmfgen_map/fork_20261001/astra_audit_r2.md \
  /gpfs/kjhan/lumina/cmfgen_map/fork_20261001/astra_audit_r3.md \
  /gpfs/kjhan/lumina/cmfgen_map/fork_20261001/astra_audit_r4.md
```

구현할 판독기의 재현 인터페이스는 다음으로 둔다. 출력 경로는 매번 새 경로를 사용한다.

```bash
python3 -B scripts/nt_eta_readout.py selftest \
  --w5 "$R" --out "$N/reader_selftest"

python3 -B scripts/nt_eta_readout.py preflight \
  --w5 "$R" --run "$E" --source "$T" --build "$B" \
  --approved "$W/APPROVED_SHA.txt" \
  --out "$N/evidence/g_id_pre"

python3 -B scripts/nt_eta_readout.py readout \
  --w5 "$R" --run "$E" --source "$T" --build "$B" \
  --approved "$W/APPROVED_SHA.txt" \
  --evidence "$N/evidence" --out "$N/readout"
```

preflight는 제출 전 검사 상태만 출력하며 미실행 런에 물리 분기를 부여하지 않는다. readout은 최종 게이트와 분기를 함께 출력한다. 기계 검사 rc≠0 또는 의미·문턱·관측 범위의 변경 필요는 기안 재발주 대상으로 기록한다.

아래 선언의 분기 입력은 **집계 시험용 부호화**다.

- `10` = 유효한 1차 조건 성립, `11` = 유효한 1차 조건 불성립.
- 나머지 열 개 값 `1/2/3/4` = 황 각 셀의 (a)/(b)/(c)/(d).
- 1차 표지 중복·부재, 잘못된 셀 수, `0` 또는 다른 값은 게이트·입력 실패에 해당한다.
- 선언은 전체 분기·최빈·동률의 배타성과 전체 덮음을 검사한다. 원자료에서 δ·D·T를 계산하는 검증은 §4.3 NC가 담당한다.
- `residual`에 기대지 않고 NT-X까지 명시한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-NT-ETA (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "scripts/[A-Za-z0-9_/.]+\\.py",
    "symbol": "nt_eta_",
    "roots": [
      "scripts/nt_eta_readout.py"
    ],
    "expected_extra": [
      "scripts/nt_eta_readout.py"
    ]
  },
  "branches": {
    "regimes": [
      [0.0, 0.0],
      [1.0, 1.0],
      [2.0, 2.0],
      [3.0, 3.0],
      [4.0, 4.0],
      [10.0, 10.0],
      [11.0, 11.0]
    ],
    "metrics": {
      "valid": "len(v) == 11 and all(x in (1.0, 2.0, 3.0, 4.0, 10.0, 11.0) for x in v) and sum(1 for x in v if x in (10.0, 11.0)) == 1",
      "iv": "sum(1 for x in v if x == 10.0) == 1",
      "a": "sum(1 for x in v if x == 1.0)",
      "b": "sum(1 for x in v if x == 2.0)",
      "c": "sum(1 for x in v if x == 3.0)",
      "d": "sum(1 for x in v if x == 4.0)"
    },
    "rules": [
      {
        "name": "NT-SUPPORT",
        "predicate": "valid and iv and a > max(b, c, d)"
      },
      {
        "name": "NT-IV-ONLY",
        "predicate": "valid and iv and c > max(a, b, d)"
      },
      {
        "name": "NT-REV",
        "predicate": "valid and iv and b > max(a, c, d)"
      },
      {
        "name": "NT-UNDET",
        "predicate": "valid and iv and not (a > max(b, c, d) or b > max(a, c, d) or c > max(a, b, d))"
      },
      {
        "name": "NT-X",
        "predicate": "not valid or not iv"
      }
    ],
    "adversarial_fixtures": [
      {
        "name": "support-plurality-not-majority",
        "mix": [[10.0, 1], [1.0, 4], [2.0, 3], [3.0, 2], [4.0, 1]]
      },
      {
        "name": "reverse-plurality",
        "mix": [[10.0, 1], [1.0, 3], [2.0, 4], [3.0, 2], [4.0, 1]]
      },
      {
        "name": "iv-only-with-other-detections",
        "mix": [[10.0, 1], [1.0, 3], [2.0, 2], [3.0, 4], [4.0, 1]]
      },
      {
        "name": "undetermined-cell-plurality",
        "mix": [[10.0, 1], [1.0, 3], [2.0, 2], [3.0, 1], [4.0, 4]]
      },
      {
        "name": "tie-a-b",
        "mix": [[10.0, 1], [1.0, 4], [2.0, 4], [3.0, 2]]
      },
      {
        "name": "tie-c-d",
        "mix": [[10.0, 1], [3.0, 5], [4.0, 5]]
      },
      {
        "name": "three-way-tie",
        "mix": [[10.0, 1], [1.0, 3], [2.0, 3], [3.0, 3], [4.0, 1]]
      },
      {
        "name": "iv-false-a",
        "mix": [[11.0, 1], [1.0, 10]]
      },
      {
        "name": "iv-false-b",
        "mix": [[11.0, 1], [2.0, 10]]
      },
      {
        "name": "iv-false-c",
        "mix": [[11.0, 1], [3.0, 10]]
      },
      {
        "name": "iv-false-d",
        "mix": [[11.0, 1], [4.0, 10]]
      },
      {
        "name": "missing-cell",
        "mix": [[10.0, 1], [1.0, 9]]
      },
      {
        "name": "failed-cell",
        "mix": [[10.0, 1], [0.0, 1], [1.0, 9]]
      },
      {
        "name": "duplicate-primary-marker",
        "mix": [[10.0, 2], [1.0, 9]]
      },
      {
        "name": "invalid-code",
        "mix": [[10.0, 1], [9.0, 1], [1.0, 9]]
      },
      {
        "name": "empty",
        "mix": []
      }
    ]
  },
  "references": [
    {
      "path": "CLAUDE.md"
    },
    {
      "path": "docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md"
    },
    {
      "path": "docs/VERDICT_DET_W5_CONT_SOURCE_2026-10-01.md"
    },
    {
      "path": "scripts/check_prereg_preflight.py"
    },
    {
      "path": "scripts/check_gate_referents.py"
    },
    {
      "path": "scripts/check_doc_canonical.py"
    },
    {
      "path": "scripts/w5_cont_readout.py",
      "flags_existing": [
        "def log_planck",
        "def window_ratios",
        "def trapz_band"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fork_20261001/FORK.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fork_20261001/FORK_r3.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fork_20261001/astra_audit_r1.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fork_20261001/astra_audit_r2.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fork_20261001/astra_audit_r3.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fork_20261001/astra_audit_r4.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fork_20261001/r3_repro.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fork_20261001/gamma_bands_v2.py"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fork_20261001/fork_fuv_yardstick2.py"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_plasma.c",
      "flags_existing": [
        "LUMINA_NT_FULL",
        "#define ETA_NONTHERMAL 0.05",
        "double pref = full ? 1.0 : ETA_NONTHERMAL",
        "gamma_nt_atom",
        "ACM(ground_hi, 0) += R_nt_per_particle"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/stage_run1.sh"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/APPROVED_SHA.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1/source_sha_at_build.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1/compiler.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/job.slurm",
      "flags_existing": [
        "readonly GPU_PROFILE=h100-a100x2",
        "expected_partition=h100,a100; expected_gpu_count=2",
        "ABSOLUTE_RUN_ROOT"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/input/resolved_lumina.exports",
      "flags_existing": [
        "LUMINA_NT_FULL=\"1\"",
        "LUMINA_PURE_CMFGEN_ITER=\"6\"",
        "LUMINA_CMF_FINE_CONT_NLTE_SOURCE=\"1\"",
        "LUMINA_FIXED_TE_PROFILE"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/input/binary.sha256",
      "flags_existing": [
        "7800484fb309addabfa5b2f6b879f3ab09b814db83d3c5db2f39debff3f1b390"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/RUN_FOOTER.txt",
      "flags_existing": [
        "job_id=409099",
        "elapsed_seconds=39269",
        "fixed_te_loader_consumed=1",
        "terminal_state=SUCCESS"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/work/cmfgen_final_publication.json",
      "flags_existing": [
        "\"population_generation\": 8",
        "\"required_population_generation\": 8",
        "\"authority\": \"SAHA_STAGE\""
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/run1/work/physics_comparison/physics_DET_iter0005.manifest.json",
      "flags_existing": [
        "\"transaction_status\": \"COMMITTED\"",
        "\"radiation_generation\": 7",
        "\"population_generation\": 8"
      ]
    }
  ]
}
```
