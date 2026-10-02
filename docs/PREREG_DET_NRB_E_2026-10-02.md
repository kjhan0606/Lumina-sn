# 사전등록 — DET-NRB-E: E 기준 NRB의 생산계 되먹임 응답 (2026-10-02, 기안)

기안자 = astra, 개정25-r 및 이번 직접 발주에 따른 read-only 계약 기안자. 산출은 이 마크다운 본문이며, 운전자가 `docs/PREREG_DET_NRB_E_2026-10-02.md`로 **신설**한다. 판독기 구현은 운전자 담당이다(개정25-w). 이번 기안에서 파일 수정·계산 실행·신규 결과 판정은 수행하지 않았다.

## §0. 설계 판단

**생산 코드 변경 0, 재빌드 0, GPU 판정런 1회, 판독기 신설 1개**로 한다. E의 바이너리·입력·자원 상한을 유지하고, 설정 차이는 `LUMINA_BB_NRB` 부재→`"1"`과 `LUMINA_A210_SPRODUCER_CAPTURE` `"0"`→`"1"` 두 키로 제한한다.

목적은 **E 대비 NRB의 생산계 되먹임 응답 판별**이다. 주원인 확정과 완전 수렴은 판정 범위 밖이다. 진단 루프의 축소 모형 결과는 기대 경향으로 승계하며 생산계 응답의 보증으로 사용하지 않는다.

확정 인계의 관측·집계를 유지한다.

- ① E call 7 기준 선 집합을 고정하고 call 3–7 궤적을 보존한다. 판정값은 call 7이며, 셸 집계는 같은 범주 **3/5**다.
- ② 지정 셸의 J/B는 **보고 전용**이다.
- ③ 양쪽 런의 반복별 ion CSV k3·k4·k5를 사용한다. 변화 부호는 **새 런−E**다. 30셀을 모두 보고하고, 분기에는 황 10셀의 **유일 최빈값**을 사용한다.
- ④ 두 FUV 색비를 각각 보존한다.
- **R5 선분리 → 나머지 R0–R4** 순서를 적용한다. 결손과 유효한 혼합 결과의 증거 상태를 구분한다.

목표 산출은 관측 ①–④, 셸·셀별 결과, 집계, 분기와 해석 한계를 재현할 수 있는 증거 묶음이다. 정상 종료나 판독기 NC 통과만으로 목표 달성을 선언하지 않는다.

### user 결정과 별도 결정 필요 항목

**확정:** user의 2026-10-02 「재개」 결정으로, 09-25의 「이 방향 종료」 결정을 재심하여 DET-NRB-E를 진행한다. 진단서와 인계 목록 작성 당시의 “기존 결정 유지·재심 사안”은 이 후속 사용자 결정으로 해소됐다. 404407의 과거 **X 판정과 저작·집행 이력은 보존**한다.

이번 등록 범위 안에서 재개 승인을 다시 받을 항목은 없다. 다음은 별도 user 결정 대상이다.

- 실패·시간 초과·미판정 뒤 추가 GPU 런, 재제출, 반복 연장 또는 자원 상한 확대.
- NRB의 생산 기준선 채택, 생산 코드·물리 설정 변경.
- 연속 결합 맞춤 검산, 외곽 W2 등 독립 후속 작업의 실제 착수·자원 배정.

연속 결합 맞춤 검산을 이번 런의 선행 필수조건으로 추가하지 않는다.

### 경로와 기준선

```text
P = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
T = /gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source
D = /gpfs/kjhan/lumina/cmfgen_map/diag2_20261002
W = /gpfs/kjhan/lumina/cmfgen_map/w5_20260930
B = /gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1
E = /gpfs/kjhan/lumina/cmfgen_map/nteta_20261001/run1
N = /gpfs/kjhan/lumina/cmfgen_map/nrbe_20261002
R = /gpfs/kjhan/lumina/cmfgen_map/nrbe_20261002/run1
```

E는 기존 기준선, R은 새 NRB 런이다. N·R은 **신설 예정 경로**다. 기안 조회 시 N은 부재했다. 스테이징 시 기존 경로와 충돌하면 덮어쓰지 않고 새 경로를 집행 기록에 확정한다. 경로 변경으로 물리 설정이나 실행 횟수를 바꾸지 않는다.

계약의 선행 근거는 다음이다.

- [진단서 정본 §4](/gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/DIAGNOSIS.md:78)와 [astra 4회차 확정 인계 목록](/gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/astra_audit_r4.md).
- [DET-BB-ALI 사전등록](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_DET_BB_ALI_2026-09-25.md), [판정·XSHELL 계약 결함 기록](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_BB_ALI_2026-09-26.md).
- [DET-NT-ETA 사전등록](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_DET_NT_ETA_2026-10-01.md), [E 판정](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_NT_ETA_2026-10-01.md).

### 저자 확인 범위

**[저자 실행]** 공유 문맥 전문, 프로젝트 정체성·현행 분장 문서, 위 계약·판정·진단·감사, E의 exports·runner·footer·stage 기록·발행 manifest·CSV 머리글, 동결 NRB 소비·원장 할당 경로와 판독기 선례를 조회했다. 기계 검사기의 선언 형식도 조회했다.

직접 확인한 소형 파일 해시는 다음이다.

| 대상 | SHA-256 |
|---|---|
| D/DIAGNOSIS.md | `d6c6ff0768d995b6de87a1cfba88ebf1cc2e152a098a58e57e9c19ef86c31809` |
| D/astra_audit_r1.md | `9e7e052fb24d72d2ccd9273b0f26ceff4523d9772c931cc96c17eeebaa2bfa95` |
| D/astra_audit_r2.md | `1c903f5addd10a27317b63910fca04c614aad044560c9717d0c4a9cda7073431` |
| D/astra_audit_r3.md | `7d5b66078db01ba6bfa57100db4e96ef5337dacd507dff0cfcbfea5c4ba5ced2` |
| D/astra_audit_r4.md | `1cf29cf01e86fe3bb8a99b1045e5876c36c42bd12dd2a7a1e4cc5146caf66153` |
| P/scripts/nt_eta_readout.py | `a4ae3f3ae31f5b28910204bbfe1bbf5ef99292b9b06c1c53a866be08ed20ce3f` |
| P/scripts/w5_cont_readout.py | `7008f709e4ca49a24823e4f88f452c42448247d8cb5940c1c1670c4e31a7f3cd` |

再現 명령:

```bash
sha256sum \
  /gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/DIAGNOSIS.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/astra_audit_r{1,2,3,4}.md \
  scripts/nt_eta_readout.py scripts/w5_cont_readout.py
```

**[기록 조회]** E footer는 job 409603, H100×2, CPU 16, 반복 6, 모델 경과 40,070초, `terminal_state=SUCCESS`, `fixed_te_loader_consumed=1`을 기록한다. Slurm 상태 보존 기록은 COMPLETED·재시작 0이다. 직접 Slurm 재조회와 E MaxRSS 재조회는 **[미실행]**이다.

B의 빌드 기록과 E 기록이 가리키는 바이너리는 다음이다.

```text
7800484fb309addabfa5b2f6b879f3ab09b814db83d3c5db2f39debff3f1b390
```

바이너리 실물·전체 소스 manifest·입력 실물의 전건 재해시, 선 집합 물질화, 원자료 재적분, 판독기 구현·NC·기계 프리플라이트·스테이징·GPU 실행은 **[미실행]**이다. 운전자가 아래 G-ID와 집행 기록으로 완성한다.

## §1. 물리 계약 하나

**질문:** E의 W5 연속 원천·NT_FULL=0·고정 온도·6회 반복 구성을 유지하면서 기존 NRB 경로와 그 필수 원장 캡처를 켜면, 고정된 두꺼운 D 대역 선 소스, 단계·준위합 III/II, 두 FUV 색비가 어떻게 동반 응답하는가?

설정 델타는 다음과 같다.

| 키 | E | R |
|---|---|---|
| `LUMINA_BB_NRB` | 부재 | `"1"` |
| `LUMINA_A210_SPRODUCER_CAPTURE` | `"0"` | `"1"` |

동결 구현은 적용 가능한 Sobolev 선에서 생산자가 저장한 `ct=βJ_c`와 `τ_eff`를 소비하여

\[
R_{\rm absorb}=B_{lu}\,ct,\qquad
R_{\rm stim}=B_{ul}\,ct,\qquad
R_{\rm spont}=A_{ul}\beta
\]

를 사용한다. 이는 선 자기 소스를 새 인구에 연결하는 **반복법 변경**이다. 고정된 전체 \( \bar J \) 행렬의 항등 재작성이나 CMFGEN 선형화 전체와의 동치를 주장하지 않는다.

적용 조건은 동결 구현 그대로다. INIT에서는 비활성이고, 유효한 checked view·쌍 소유 NLTE 선에서 작동한다. E의 `LUMINA_NLTE_SKIP_Z=""`를 유지하므로 과거 404407의 Si 제외 조건을 승계하지 않는다. `SRCE_CHK` 셀은 기존 율 형태를 유지한다.

개입은 관측 선 집합에만 한정되지 않는다. 따라서 공동 응답이 나와도 특정 D 대역 선 집합의 독점적 인과나 주원인 몫을 확정하지 않는다.

## §2. 설계

### 2.1 기준선·스테이징·exports

R은 **E의 봉인 입력**을 새 디렉터리로 복사하여 시작한다. E 최종 인구를 warm start로 넣지 않는다. 스테이징 선례는 `/gpfs/kjhan/lumina/cmfgen_map/nteta_20261001/stage_run1.sh`이며, 원본 스크립트를 제자리 수정하거나 원래 목적지로 재실행하지 않는다.

exports는 중복 없는 키 사전으로 비교한다. 다음 경로 키에만 **E→R 접두 이식과 동일한 상대 대상**을 허용한다.

```text
LUMINA_MODEL_DIR
LUMINA_CMFGEN_SIGMA_BF
LUMINA_DEPOSITION_FILE
LUMINA_FIXED_TE_PROFILE
LUMINA_IONIZATION_REFERENCE_FILE
LUMINA_TOPION_LEVELS_FILE
LUMINA_PHYSICS_COMPARISON_DIR
```

이를 정규화한 뒤 차이는 §1의 두 키와 정확히 일치해야 한다. 부재와 빈 문자열을 같은 값으로 처리하지 않는다. 다른 문자열·경로 차이를 일괄 제거하지 않는다.

MUST_KEEP는 양쪽 런에서 다음과 같다.

```text
LUMINA_PURE_CMFGEN_ITER="6"
LUMINA_DET_TRANSACTIONAL="1"
LUMINA_CMF_FINE_MGPU_DEVICES="2"
LUMINA_CMF_FINE_CONT_NLTE_SOURCE="1"
LUMINA_ION_NT="1"
LUMINA_POP_DJDN="1"
LUMINA_NLTE_SKIP_Z=""
LUMINA_XSHELL_RESPONSE="0"
LUMINA_NT_FULL="0"
```

나머지 설정도 exports 전체 비교로 유지한다. E의 CPU NLTE 조립 경로, JEQB 부재, 원자자료·침적·초기 조건·반복 수·솔버 허용오차를 바꾸지 않는다. 상속된 `LUMINA_*`를 지운 뒤 봉인 exports를 소비하는 기존 runner 절차를 유지한다.

### 2.2 관측 권위·시점

| 관측 | 입력 | 시점·권위 |
|---|---|---|
| ① 선 소스 | E/R의 `work/w5_lines.csv` | 고정 키 `(shell,line_id)`, call 3–7. 판정 call 7 |
| ② J/B | E/R의 k5 spectral·shell CSV | radiation gen7, 고정 \(T_e\), 셸별 대역 적분 |
| ③ III/II | E/R의 k3·k4·k5 ion CSV | 단계와 준위합을 분리. population gen6·7·8 |
| ④ FUV 색비 | E/R의 `work/lumina_spectrum.csv` | 최종 관측자계 \(L_\lambda\) |
| 완주·세대 | 반복 manifest, 최종 publication, 로그·footer | 등록 실행과 저장량의 연결 |

선 원장의 call과 ion CSV의 논리 반복 k는 다른 지수다. call 7 원장은 **복사·물질 gen7**이고, k5 ion CSV는 그 뒤 커밋된 **인구 gen8**이다. 최종 준위 gen8을 사용하여 call 7의 소스를 재구성하지 않는다.

### 2.3 구성상 도달 가능성 자기 점검

이 절은 XSHELL 실사고의 재발을 막는 계약 자기 점검이며, 새 GPU 시험을 만들지 않는다.

| 점검 대상 | 이번 계약의 처리 |
|---|---|
| XSHELL 핀 상수 | E/R 모두 `XSHELL_RESPONSE=0`. XSHELL G-RATE·G-DIR 통과, 과거 Γ 상수 재현 또는 XSHELL 출력 파일을 요구하지 않는다. |
| E의 NRB 비활성 | E에 새 런의 armed/captured 태그를 요구하지 않는다. E의 NRB·캡처 부재는 등록된 기준 상태다. |
| INIT | 새 런도 `operator=1 armed=0`이 정상이다. INIT에 `armed=1`·Sobolev 소스 열을 요구하지 않는다. |
| 첫 응답 시점 | call 3은 최초 응답 가능 시점이다. call 2 변화나 call 3에서 특정 하강폭 달성을 통과 조건으로 요구하지 않는다. |
| 고정 선 집합 | τ 문턱은 **E call 7의 선택 조건**이다. 새 런에서 다시 τ≥1000을 요구하거나 얇아진 선을 제거하지 않는다. 원소·이온별 최소 선 수도 추가하지 않는다. |
| SRCE_CHK·β | `srce_chk_applied=0`이나 전역 `β≤1`을 강제하지 않는다. 동결 코드의 정상 예외와 관측량의 로그 미정의를 구분한다. |
| exact solve | 기존 성공 조건을 요구한다. E와 같은 해·반복 소요 횟수·잔차 수치·스펙트럼을 요구하지 않는다. |
| 물리 응답 | 하강만 정상 결과로 취급하지 않는다. R0·R2·R3·R4와 유효한 R5도 등록된 결과다. ②의 크기·부호는 분기를 바꾸지 않는다. |

**자기 점검 결론:** 등록 설정과 수용 조건 사이에 확인된 구성상 모순은 없다. 새 NRB 조합의 실제 완주 가능성은 아직 미시연이다. 기계 프리플라이트 통과를 이 의미 검사의 대체 증거로 사용하지 않는다.

### 2.4 판독기 최소안

새 `scripts/nrb_e_readout.py` 하나에 신원 검사·판독·집계·NC를 구현한다. 기존 `scripts/nt_eta_readout.py`와 `scripts/w5_cont_readout.py`는 동결한다.

- NT-ETA의 G-ID·G-RUN·셀 판정·유일 최빈·NC 구조를 재사용할 수 있다.
- 기존 NT-ETA의 **NT_FULL 한 키 델타 검사와 전체 분기**를 그대로 호출하지 않는다.
- W5의 Planck·주파수 창·파장 적분 순수 함수를 재사용할 수 있다.
- 기존 진단 스크립트의 선 누락 무시, β 컷, 최소 선 수에 따른 출력 생략을 승계하지 않는다.
- 입력 중복·결손을 먼저 검사한다. 사전 덮어쓰기, 부분 집계, 대체 세대·floor·clamp로 값을 만들지 않는다.

## §3. 변경집합·산출물·자원

### 변경집합 — DET-NRB-E (표)

| 경로 | 변경 | 범위 |
|---|---|---|
| `scripts/nrb_e_readout.py` | **신설** | `nrb_e_` 접두의 신원·판독·집계·직접 NC. 운전자 구현 |

### 변경집합 끝

생산 소스·기존 판독기·빌드 산출은 변경하지 않는다. 새 환경 키·생산 계측·GPU 게이트를 추가하지 않는다. 런 전용 exports·runner·stage 기록과 문서는 운전자의 스테이징·증거 산출이다.

### 3.1 산출물·보존

N 아래에 다음을 **신설**한다.

| 위치 | 내용 |
|---|---|
| `run1/` | 입력 사본, runner·exports·신원 기록, 기존 생산 출력, 로그·footer |
| `reader_selftest/` | 정상 대조·NC 기대값·실측·사유·rc |
| `evidence/preflight/lines_E_call7.csv` | 고정 선 집합, E 원자료 해시·선택식·키·기준값 |
| `evidence/` | G-ID, 실제 제출 명령·Slurm 자원·상태·재시작 기록, 계약·판독기·의존 파일 해시 |
| `readout/line_trajectory.csv` | 고정 선별 양쪽 call 3–7 원장값·세대·상태·재구성 S·E call 7 대비 변화 |
| `readout/source_shells.csv` | 5셸의 Δ₁ 궤적·call 7 범주·집계 |
| `readout/population23.csv` | 30셀의 인구·y₃/y₄/y₅·평균·양쪽 표류·D·T·δ₃·셀 결과 |
| `readout/auxiliary.json` | ② 셸별 J/B와 ④ 두 색비의 양쪽 적분·로그 변화 |
| `readout/result.json` | 게이트·유효성·분기·R5 사유, 황 10셀 건수·집계와 전체 분포 |
| `readout/evidence_manifest.json` | 실제 소비 파일·판독기·의존 파일의 절대 경로와 해시 |

최종 판정문은 `docs/VERDICT_DET_NRB_E_2026-10-02.md`로 **신설**한다. 셸별·30셀 결과를 전부 보고하고 황 집계를 별도로 표시한다.

E·W5·404407·진단·감사 원본과 실패 산출은 보존한다. 입력 사본에 원본과 쓰기를 공유하는 하드링크를 사용하지 않는다. 판독 출력은 읽는 런·입력과 분리하며, 재판독도 새 출력 경로를 사용한다. 실패가 원본 복구·삭제나 자동 재실행의 근거가 되지 않는다.

### 3.2 자원·runner

**단일 노드의 H100 또는 A100 80 GB급 같은 종류 GPU 두 장, 판정런 한 번**이다.

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

H200은 금지한다. E runner의 활성 `h100-a100x2` 프로필과 GPU 종류·장수·메모리 검사를 유지한다.

**시간 근거:** E 모델 footer의 40,070초는 약 11.13시간이다. H100×2 기준 비교용 예상 비용은 약 **22.3 GPU·시간**이다. 30시간은 E의 기존 상한이며, A100 속도와 NRB 이후 실행시간 변동 여유를 포함한다. 최대 할당 비용은 **60 GPU·시간**이다. A100의 동일 완주 시간을 실측했다고 주장하지 않는다. 시간 초과 시 자동 연장·재큐잉·재제출하지 않는다.

**메모리 근거:** E 로그의 `upper_population_cells=129439900`과 동결 할당 코드는 원장이 관측 선 집합이 아닌 전체 선×셸에 할당됨을 보인다. 추가량은

\[
129{,}439{,}900\times(4\times8+1)
=4{,}271{,}516{,}700\ {\rm bytes}
\simeq3.98\ {\rm GiB}
\]

의 **호스트 메모리**다. 기존 64 GiB 상한 안에서 사용한다. 선례 398939의 약 30.9 GiB MaxRSS에 이를 더한 약 35 GiB는 참고 추정이며, E 또는 새 런의 실측 RSS가 아니다. 운전자는 확보 가능한 E 회계 자료와 새 런의 MaxRSS를 증거에 남긴다. 메모리 부족이 발생해도 상한을 자동 확대하지 않는다.

런 전용 runner는 E의 본체·절대 run-root 인자 규약을 유지한다. 작업명·출력 경로·등록 자원만 운영 변경으로 기록하며, 기존 모듈 블록의 `module purge` 후 `gnu13/13.2.0 cuda/13.0.2` 로드와 실제 모듈·라이브러리 기록을 유지한다.

제출 형식:

```bash
sbatch \
  --partition=h100,a100 --nodes=1 --ntasks=1 \
  --gres=gpu:2 --cpus-per-task=16 --mem=64G \
  --time=30:00:00 --no-requeue --exclude=syn102 \
  --output="$R/slurm-%j.out" --error="$R/slurm-%j.err" \
  "$R/job.slurm" "$R"
```

프리플라이트·판독 NC·원장 스트리밍 판독은 grammar-debug에서 실행한다. 생산 계산은 Slurm에서만 수행한다. 새 GPU 사전 시험은 없다.

## §4. 게이트

| 게이트 | 수용 기준 | 소비 |
|---|---|---|
| **G-ID** | §4.1의 바이너리·소스·입력·exports·MUST_KEEP·FIXED_TE·판독 신원 일치 | E 대비 등록 개입의 성립 |
| **G-RUN** | §4.2의 완주·세대·exact solve·NRB 발화·필수 자료 충족 | 관측 ①–④의 실행 증거 |
| **G-READOUT** | §4.3의 정상 대조·NC 전건 충족 | 원자료 판독·집계·분기·결손 처리 |

제출 전 G-ID의 사전 부분과 G-READOUT을 충족한다. 실행 후 G-ID를 완성하고 양쪽 G-RUN을 평가한다. 새 런에 아직 존재할 수 없는 footer를 제출 전 통과 조건으로 요구하지 않는다.

### 4.1 G-ID

1. **바이너리:** B의 `lumina_cuda`·`binary.sha256`, E/R의 `input/lumina_cuda`·`input/binary.sha256`·`input/run_build_identity.json`·`STAGE_IDENTITY.txt`, 실행 후 footer를 §0의 전체 해시에 연결한다.
2. **소스:** B와 E/R의 `source_sha_at_build.txt`가 일치하고, T의 해당 파일 실물 및 W의 `APPROVED_SHA.txt`와 연결돼야 한다. 기록 사본의 일치만으로 실물 검증을 대신하지 않는다. 재빌드는 없다.
3. **입력:** E/R의 `input/deck.sha256`·`input/topion.sha256`·`input/sigma.sha256`와 각각의 실물을 검증한다. 고정 온도·침적·원자자료·초기 입력을 포함하여 내용이 같아야 한다.
4. **exports:** §2.1의 경로 이식 후 차이가 정확히 두 키인지 검사한다. 중복 키·추가 차이·MUST_KEEP 이탈을 거부한다. 실제 소비 exports의 해시를 stage·footer에 연결한다.
5. **FIXED_TE:** 양쪽 `input/fixed_te_profile.txt`의 내용이 같아야 한다. 실행 후 loader 소비, FIXED_T manifest, 온도 manifest 일관성과 실제 발행 \(T_e\)의 프로필 일치를 확인한다. **파일 해시와 정규 온도 manifest 해시는 서로 다른 종류**로 비교한다.
6. **판독·선택 신원:** 새 판독기와 동결 의존 파일 해시를 봉인한다. 고정 선 집합은 E 원자료 해시·선택식·키·기준값에 연결하며, 런 후 임의 재선택하지 않는다.
7. **실행 신원:** runner 해시·제출 인자·실제 GPU·장수·노드·시간·메모리·재시작 횟수를 등록 자원과 대조한다.

해시 검증은 명시된 manifest와 파일만 대상으로 한다. 공유 파일시스템 전체를 탐색하지 않는다.

### 4.2 G-RUN

**공통 완주 조건**

- Slurm COMPLETED, 재시작 0, 모델 rc=0, footer `terminal_state=SUCCESS`.
- 등록 outer iteration 6, DET COMPLETE, k0–k5 manifest의 COMMITTED.
- 반복 k의 radiation gen=`k+2`, population gen=`k+3`, FIXED_T.
- 최종 publication complete, population/required population gen8, 온도·권위 연결.
- 기존 `[cmf_fine][EXACT-MULTIGPU-EPOCH] status=OK` 7회와 등록 solver 성공 조건.
- `[FATAL]`, 실제 BLOCKED·실행 실패 표지가 없어야 한다. 정상 `[R6]` 등 단계 태그 자체를 금지하지 않는다.

**NRB 발화 증인 — R에 적용**

- `[BB-NRB] iter=0 operator=1 armed=0`: INIT 증인.
- 논리 반복 k=0–5 각각 `[BB-NRB] iter=k operator=2 armed=1 captured=1 n_shells=50 skip_z=none`.
- Sobolev 6회에 대응하는 `[cmf_fine][SPRODUCER-CAPTURE] status=CAPTURED`, `n_shells=50`, Sobolev operator.
- 태그의 단순 총개수만 세지 않고 `(iter,operator)` 완전성·중복·모순을 검사한다. INIT에 `captured=1`을 요구하지 않는다.
- `[BB-NRB][FATAL]`과 SPRODUCER 할당·정합 실패는 X다.

E에는 NRB armed·SPRODUCER CAPTURED를 요구하지 않는다. 동결 설정과 모순되는 발화가 발견되면 신원 문제로 기록한다.

**SRCE_CHK 처리**

기존 producer의 반복별 `srce_chk_applied`와 고정 선 원장의 `srce_chk`를 보존한다. 동결 구현에서 provenance bit0 셀은 `NOT_APPLICABLE`로 기존 율을 유지한다. 0xff 등 미정의 provenance는 이 예외로 인정되지 않는다.

SRCE_CHK 발생 자체는 실패 조건이 아니다. 선택 선이 예외에 해당하면 이를 표시하고 집합에서 제거하지 않는다. 그 선의 등록 로그 관측량이 정의되지 않으면 측정 결손으로 처리한다. 항등식 `Jbar=ct+le`만으로 세대 일치를 입증했다고 주장하지 않는다.

**필수 자료**

| 위치 | 필수 내용 |
|---|---|
| 양쪽 로그·footer·Slurm 증거 | 위 실행·발화·실패·자원 정보 |
| `work/physics_comparison/physics_DET_iter000{k}.manifest.json`, k=0–5 | 반복·세대·FIXED_T |
| 같은 디렉터리의 k=3·4·5 ion CSV | ③의 30셀, 중복 없는 셸 행, population gen6·7·8 |
| 같은 디렉터리의 k5 spectral·shell CSV | ②의 지정 셸·격자·온도·세대 |
| `work/w5_lines.csv` | 고정 선의 양쪽 call 3–7, `(call,shell,line_id)` 유일성·operator·복사/물질 세대 |
| `work/lumina_spectrum.csv` | ④ 적분 구간의 전체 지원 |
| `work/cmfgen_final_publication.json` 및 그 참조 파일 | 최종 발행 완전성·권위·세대 |

최종 publication이 참조하는 파일은 기존 `lumina_plasma_state.csv`, `lumina_ion_pops.csv`, `nlte_levels_final.csv`다. 이를 새 반복별 준위 덤프나 III→IV 관측 요구로 확대하지 않는다.

원장의 정상 INIT·`OUTSIDE_FINE_SUPPORT` 행을 전역 결함으로 처리하지 않는다. 고정된 관측 키에서 필요한 값의 누락·중복·비유한·잘못된 세대는 결손이다. ②는 보고 전용이지만 필수 산출이므로 자료 부재를 정상 보고 완료로 처리하지 않는다.

### 4.3 판독 NC

합성 입력 또는 격리된 작은 사본으로 시험하며 원본·생산 소스를 변조하지 않는다. 별도 테스트 파일을 신설하지 않고 판독기 selftest에 둔다.

| 묶음 | 필수 검증 |
|---|---|
| 선 집합·① | E call 7에서만 선택, `(shell,line_id)` 대응, τ=1000 포함, D 대역 경계, 새 런의 τ 변화로 재선택하지 않음. 선별 로그 차이의 중앙값과 중앙값끼리의 차이를 구별하는 입력. call 7 판정·call 3–7 보존. |
| ① 범주·집계 | ±0.30 포함, 바로 안·밖. 같은 범주 2셸/3셸 경계, `(2,2,1)` 혼합. 5셸의 세 범주 건수 분포 21개를 검사. |
| ③ 셀 식 | 양쪽 k3·k4·k5 실제 인구에서 계산. E 쪽 표류만 큰 경우, 로그 평균, 부호 새 런−E. `3D<0.30`, `=0.30`, `>0.30`과 δ=±T·인접값. D=0.15·δ=−0.60은 (a)만 성립. |
| ③ 황 집계 | 황 10셀 건수 분포 286개 전수. `(4,3,2,1)`은 (a), 최다 동률은 혼합. Si·Ca 20셀을 바꿔도 유효 자료에서 황 집계가 바뀌지 않아야 함. |
| 분기·우선순위 | 단계 2의 36조합 전수, 결과 건수 R0/R1/R2/R3/R4=`24/1/2/6/3`. 각 조합에서 ④b를 바꾸어 하강/증가 반대일 때 R5가 먼저 선택됨을 확인. |
| 과거 반례 | ① 전부 하강·③ 전부 (a)·④a=−0.4·④b=+0.4 → R5 하나. ④b만 −0.1 → R1 하나. R5 사유가 겹쳐도 분기는 하나. |
| ②·④ 적분 | 알려진 선형 \(L_\lambda\)의 경계 보간 적분, J/B의 주파수 bin 중첩 폭·Planck 가중, 지원 부족·중복·비유한 검출. ②의 유효한 값·부호 변경은 분기에 비간섭. |
| 신원·실행 | 두 키 델타 정상 대조와 추가 차이·키 중복·MUST_KEEP·FIXED_TE·바이너리·소스 불일치. NRB 태그 삭제·중복·armed/captured 오염, 세대 변경·미완주를 기대 사유로 거부. E의 OFF와 INIT armed=0은 정상으로 수용. |
| 결손·예외 | 파일·선·셀 부재, 중복, NaN/Inf, 비양수 로그 인구·S, 정의되지 않는 나눗셈·적분 → **R5(결손) 및 유효성 X**. 유효한 SRCE_CHK 발생이나 전역 β>1만으로 실행 실패를 만들지 않음. |

정상 대조를 반드시 포함한다. **통계 함수에 E를 양쪽으로 넣으면 Δ₁=δ₃=Δ₄a=Δ₄b=0, ③은 표류에 따라 (c)/(d), 전체 통계 분기는 R0**다. 이를 실제 두 키 개입의 G-ID 통과로 표시하지 않는다.

경계에는 반올림 표시값이나 임의 epsilon을 사용하지 않는다. 실제 비교 피연산자의 경계와 인접 표현 가능 수를 시험한다. NC 성공은 주입 결함을 기대 사유로 검출했다는 뜻이다.

### 4.4 결손·X와 R5의 관계

진단 분기와 증거 유효성을 별도 필드로 둔다.

- `validity=VALID`: 게이트·필수 자료가 유효하다. 혼합·동률·두 색비 반대에 따른 R5도 유효한 관측 결과다.
- **필수 자료 결손:** 진단서 규칙대로 `response_branch=R5`, 동시에 `validity=X`와 결손 사유를 기록한다.
- 신원·실행·NC 실패로 비교 자체가 성립하지 않으면 `validity=X`로 처리하고 물리 분기를 확정하지 않는다. 확보한 원자료·셀 결과는 실패 사유와 함께 보존한다.

X를 R0–R5에 경쟁하는 새 물리 분기로 만들지 않는다. 결손을 불변·혼합의 정상 관측으로 대체하지 않는다.

## §5. 기대치·판정량·분기

변화 부호는 전부 **R−E**다. ①의 call 지수는 아래에서 c, ③의 논리 반복 지수는 k로 구별한다.

### 5.1 ① 고정 선 소스

E의 `work/w5_lines.csv` call 7에서

\[
L=\{(s,\ell):
\tau_{\rm eff,E,7}\ge10^3,\ 
912\le\lambda_\ell<1500\ {\rm \mathring A},\
s\in\{10,15,19,24,30\}\}
\]

를 정한다. 파장 경계는 선행 D 대역의 `[912,1500)` 정의를 승계한다. \(\lambda_\ell=c/\nu_\ell\)로 읽는다. 해당 원장의 매핑 선 집합을 사용하며, 다른 선 목록·상위 N개·가중 표본으로 교체하지 않는다.

제출 전에 L과 E 기준값을 물질화·봉인한다. 셸별 집합 \(L_s\)는 결과와 무관하게 유지한다.

\[
S^r_{s\ell,c}
=\frac{L^r_{\ell,c}}{1-\beta^r_{s\ell,c}},
\]

\[
\Delta_1(s,c)
=\operatorname{median}_{\ell\in L_s}
\left[
\log_{10}S^R_{s\ell,c}
-\log_{10}S^E_{s\ell,7}
\right],
\qquad c=3,\ldots,7.
\]

먼저 동일 선의 로그 차이를 만들고 셸별 중앙값을 취한다. 양쪽 call 3–7 원장값·S·세대를 모두 보존하며 **판정값은 \(\Delta_1(s,7)\)**이다.

call 3은 최초 응답 가능 시점이다. 축소 모형의 약 −0.8…−1.8 dex는 기대 경향이며 판정 조건이 아니다. 분모 0, 비양수·비유한 S 또는 고정 키 결손은 값을 대체하지 않고 결손으로 기록한다.

### 5.2 ② 셸별 J/B — 보고 전용

k5에서 다음을 보고한다.

- D=`[912,1500)` Å: sh15–24, 즉 `{15,16,17,18,19,20,21,22,23,24}`.
- B=`[531,760)` Å: sh19–24, 즉 `{19,20,21,22,23,24}`.

\[
Q^r_{\mathcal B,s}
=\frac{\sum_b J_b^r\,\Delta\nu_{b\cap\mathcal B}}
{\sum_b B_\nu(\sqrt{\nu_{b,\rm lo}\nu_{b,\rm hi}},T_{e,s})
\,\Delta\nu_{b\cap\mathcal B}},
\]

\[
\Delta_2(\mathcal B,s)
=\log_{10}Q^R_{\mathcal B,s}
-\log_{10}Q^E_{\mathcal B,s}.
\]

주파수 경계의 실제 중첩 폭을 사용한다. 분자·분모·양쪽 비·로그 변화·지원·세대를 보존한다. 셸 평균으로 원자료를 대체하지 않으며 **분기 조건으로 사용하지 않는다**.

### 5.3 ③ 반복별 III/II — 30셀과 황 10셀

양쪽 런의

```text
work/physics_comparison/physics_DET_iter0003.ion.csv
work/physics_comparison/physics_DET_iter0004.ion.csv
work/physics_comparison/physics_DET_iter0005.ion.csv
```

를 사용한다. 원소는 Z=14·16·20, 권위는 stage·levels, 셸은 `{10,15,19,24,30}`이다. levels는 `Z{Z}_nlte_q{1,2}_cm3`, stage는 `Z{Z}_stage_q{1,2}_cm3`다. q1=II, q2=III다.

각 셀에서

\[
y_k^r=\log_{10}(q2_k^r/q1_k^r),\qquad
\bar y^r=(y_4^r+y_5^r)/2,
\]

\[
d_R=|y_5^R-y_3^R|/2,\quad
d_E=|y_5^E-y_3^E|/2,\quad
D=\max(d_R,d_E),
\]

\[
T=\max(0.30,3D),\qquad
\delta_3=\bar y^R-\bar y^E.
\]

NT-ETA §5.2의 네 분할을 그대로 적용한다.

| 조건 | 셀 결과 |
|---|---|
| \(\delta_3\le -T\) | **(a)** |
| \(\delta_3\ge +T\) | **(b)** |
| \(|\delta_3|<0.30\ \land\ 3D\le0.30\) | **(c)** |
| \(3D>0.30\ \land\ |\delta_3|<3D\) | **(d)** |

δ=±T는 검출에 포함한다. `3D=0.30`에서 문턱 미만은 (c)다. D는 반복 표류 대리량이며 통계 오차나 수렴 인증이 아니다.

**30셀 전부**의 양쪽 인구·y·평균·표류·D·T·δ·결과를 보고한다. 황 10셀과 Si·Ca 20셀을 구분하며, 권위를 합치거나 유리한 권위를 선택하지 않는다.

### 5.4 ④ 두 FUV 색비

\[
F_a^r=\frac{L_r(1200\!-\!1500)}
{L_r(4000\!-\!6000)},\qquad
F_b^r=\frac{L_r(1500\!-\!2000)}
{L_r(4000\!-\!6000)},
\]

\[
\Delta_{4a}=\log_{10}F_a^R-\log_{10}F_a^E,\qquad
\Delta_{4b}=\log_{10}F_b^R-\log_{10}F_b^E.
\]

파장은 Å다. 최종 \(L_\lambda\)를 파장에 대해 사다리꼴 적분하고 양 끝은 인접 표본으로 선형 보간한다. 전체 구간 지원을 요구하며 외삽·부분 구간 적분은 허용하지 않는다. 각 적분·색비·로그 변화를 보존한다.

### 5.5 범주·집계

유한한 Δ에 대해 다음 범주는 배타적이다.

| 범주 | 조건 |
|---|---|
| 하강 | \(\Delta\le-0.30\) dex |
| 증가 | \(\Delta\ge+0.30\) dex |
| 불변 | \(|\Delta|<0.30\) dex |

집계는 확정 인계대로 적용한다.

- **①:** 5셸을 각각 범주화한다. 같은 범주가 **3/5 이상**이면 그 범주, 아니면 **①혼합**.
- **③:** **황 10셀의 유일 최빈값**을 사용한다. 과반 조건은 없다. `(a,b,c,d)=(4,3,2,1)`은 (a)다. 최빈값 동률은 **③혼합**. Si·Ca 20셀은 보고 전용.
- **④:** Δ₄a·Δ₄b를 각각 범주화한다.
- 파일·셀·선 부재 또는 필요한 값의 비유한·미정의는 **결손**이다.

### 5.6 분기 — 확정 2단계 표

**단계 1: 먼저 분리**

다음 중 하나라도 참이면 **R5 혼합/미판정**이다.

1. 결손.
2. ①혼합.
3. ③혼합.
4. ④a와 ④b가 서로 하강/증가로 **반대**.

여러 사유가 동시에 참이어도 분기는 R5 하나이며 사유를 모두 보존한다.

**단계 2: 나머지에만 적용**

| ① (3/5) | ③ 황 유일 최빈 | ④a | 분기 |
|---|---|---|---|
| 불변 또는 증가 | 임의 | 임의 | **R0 램프 해제 미확인** — 괄호 소비 경로(세대 k−1 제공자 항·SRCE_CHK·provenance)·J_c 선 되먹임·축소 모형 미맞춤 항 재검; 구현 오류 단정 없음 |
| 하강 | (a) | 하강 | **R1 램프 기전 지지**(몫 = 측정값; 주원인 확정 아님) |
| 하강 | (a) | 불변 또는 증가 | **R2 이온화 응답 지지, FUV 결론 없음** |
| 하강 | (c) 또는 (d) | 임의 | **R3 이 창에서 이온화 응답 미확인**(주원인 아님이 아님) |
| 하강 | (b) | 임의 | **R4 반대 방향 응답**(원인 미분리로 기재) |

단계 1 이후의 `① 세 범주 × ③ 네 범주 × ④a 세 범주 = 36조합`은 R0 24, R1 1, R2 2, R3 6, R4 3으로 분할된다. ④b는 단계 1의 반대 검사에만 쓰며 단계 2 분기를 바꾸지 않는다.

확정 반례:

- ① 전부 하강·③ 황 전부 (a)·④a=−0.4·④b=+0.4 → **R5 하나**.
- 같은 조건에서 ④b=−0.1 → **R1 하나**.

어느 분기도 추가 런을 자동 승인하지 않는다.

## §6. 주장 상한

- 결과는 **E 구성·기존 NRB 구현·6회 반복 창의 생산계 되먹임 응답**이다. 완전 수렴·주원인 확정·공개 CMFGEN 재현을 뜻하지 않는다.
- 축소 모형은 생산 충돌 분기·영역 밖 처리를 맞췄지만 연속 결합·초준위·DR·NT 등의 생산계 전체를 재현하지 못했다. 그 하강폭을 새 런의 최소 응답으로 요구하지 않는다.
- J_c에는 남은 선 되먹임이 있다. ① 하강이나 R1을 특정 D 선의 독점적 인과·정확한 원인별 기여율로 바꾸지 않는다.
- R0은 구현 오류 확정이 아니며, R3은 주원인 부정이 아니다. R4의 원인도 미분리로 남긴다.
- 황 최빈은 등록된 요약 정책이다. 단계·준위합을 독립 확률 표본으로 해석하거나 30셀의 동일 반응으로 확대하지 않는다.
- 어느 분기도 **NT·W2의 인과적 배제**로 읽지 않는다. 외곽 누출·산란선 FUV 영향·중간 셸 W2는 현재 자료로 주원인임을 입증하지 못한 후보로 유지한다.
- 연속 결합 맞춤 검산은 독립 작업이다. 이번 런의 선행 조건이나 자동 후속 실행으로 추가하지 않는다.
- 404407의 X 판정·XSHELL 계약 결함·당시 종료 결정은 이력으로 보존한다. 이번 재개는 2026-10-02 사용자 결정에 따른 새 계약이다.

## §7. Q-GOAL·Q-LEAN 자기답·집행 기록

### Q-GOAL 자기답

**PASS — 기안자의 자기답이며 독립 감사 판정이 아니다.**

최종목적은 MC 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현하는 것이다. 이번 표적은 E 이후 남은 중간 셸 II/III 과이온화와 FUV 초과에 연결된 선 소스 이완이다. 산출은 ①–④의 물리 응답이다.

외곽 W2와 생산 연속 결합 맞춤 검산은 대안·후속으로 남아 있다. 이번 단을 생략하면 축소 모형의 기전 지지가 생산계 되먹임에서도 나타나는지 모른 채 다음 수리나 기준선 채택을 선택하게 된다.

### Q-LEAN 자기답

**APPROPRIATE — 기안자의 자기답이며 독립 감사 판정이 아니다.**

GPU 한 번과 판독기 하나로 묶는다. 생산 소스·새 계측·별도 GPU 게이트·OFF 재런은 추가하지 않는다.

| 남기는 항목 | 실제 소비 |
|---|---|
| 바이너리·소스·입력·exports·FIXED_TE | 등록 개입의 신원 |
| 고정 선 집합·call 3–7 | ①의 궤적·종점·셸 집계 |
| 반복별 ion CSV | ③의 평균·표류·30셀·황 집계 |
| k5 spectral·shell | ② 셸별 J/B |
| 최종 spectrum | ④ 두 색비 |
| 기존 NRB·원장·세대·완주 증인 | 켰으나 소비되지 않음, 잘못된 세대·결손 검출 |
| 판독 NC | 실제 발생했던 집계·경계·분기 중첩·결손 오인 방지 |

기존 G-FIRE의 기능은 G-RUN에 합친다. NT-ETA의 IV/III 판정·Γ 대역 재적분·NT 스위치 몫 판독, 새 원인별 급식 계측은 이번 변경집합에 넣지 않는다.

### 집행 기록

| 단계 | 담당·현재 상태 | 기록 |
|---|---|---|
| 재개 결정 | user, 2026-10-02 확정 | 09-25 종료 방향 재심·재개 |
| 진단 루프 | Fable ↔ astra, 종료 | 정본·감사 r1–r4, 최종 차단 0 |
| 기안 | astra, 이번 read-only 본문 | 모델 ID·저자 실행/미실행 |
| 파일화·기계 검사 | 운전자, 집행 | prereg_preflight·gate_referents·doc_canonical rc=0, 커밋 84ed869 |
| 계약 사전감사 | Fable fresh(개정25-r, Q-GOAL·Q-LEAN), 집행 | `nrb2_20261002/preaudit_fable.md` — `claude-fable-5-1`, «폐합 차단» 0, K-1·K-2 반영(00ccde5); ⚠Fable 이 로그인 노드에서 파일 집계를 돌린 이탈 자기 기재 |
| 판독기 구현·직접 NC | 운전자(25-w), 집행 | 00ccde5 동결 d4fdd837…(`JUDGE_FROZEN.txt`, 판정런 종료 전), NC 71/71; 종료감사 C1 수리 후 5b5a68f9…, `reader_selftest_fix2` 91/91 |
| 코드 검사·봉인 접촉 처리 | gpt-6-sol(개정25-v), 집행 | `sol_audit_r1.md`(stderr `model: gpt-6-sol`) Q0 가·«폐합 차단» 9 → 운전자 수리, 재감사 없음 |
| 스테이징·제출 | 운전자, 집행 | G-ID 사전 PASS(`evidence/gid_pre`), 고정 L 171선(`evidence/preflight/lines_E_call7.csv`), job 410042(h100·a100, 1-06:00:00, 64G), runner sha 6db94795… |
| 판독·판정 | 운전자, 집행 | `readout/`(최초) → `readout_fix2/`(C1 수리 후, 분기·통계 동일) · R2 VALID · `docs/VERDICT_DET_NRB_E_2026-10-02.md` |
| 종료 처리 | astra 종료감사 1회, 집행 | `astra_closeout_r1.md`(실행 헤더 `model: gpt-6-astra`; 1차 발주는 stdin 대기로 산출 0 → PID 지목 종료·재발주): R2 원자료 재현, «폐합 차단» 1(G-READOUT §4.3 NC 누락·결손 예외) → 운전자 수리·재판독 분기 동일, 기재 3 반영. ⚠astra 는 이 계약의 기안자(자기 계약 감사); Slurm 은 보존 기록 기준 확인 |

운전자의 판독기 구현·판정 겸임을 남긴다. 선행 진단 루프 종료를 새 계약 감사나 실행 검증으로 재명명하지 않는다.

### 운전자 프리플라이트·재현

다음은 **파일화 후 grammar-debug에서 실행할 명령**이며 기안자는 실행하지 않았다. 작업 디렉터리는 P다.

```bash
timeout 5m python3 -B scripts/check_prereg_preflight.py \
  docs/PREREG_DET_NRB_E_2026-10-02.md --root .

timeout 5m python3 -B scripts/check_gate_referents.py \
  docs/PREREG_DET_NRB_E_2026-10-02.md --root .

timeout 5m python3 -B scripts/check_doc_canonical.py \
  docs/PREREG_DET_NRB_E_2026-10-02.md \
  docs/PREREG_DET_BB_ALI_2026-09-25.md \
  docs/VERDICT_DET_BB_ALI_2026-09-26.md \
  docs/PREREG_DET_NT_ETA_2026-10-01.md \
  docs/VERDICT_DET_NT_ETA_2026-10-01.md \
  CLAUDE.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/DIAGNOSIS.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/astra_audit_r1.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/astra_audit_r2.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/astra_audit_r3.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/astra_audit_r4.md
```

구현할 판독기 인터페이스:

```bash
python3 -B scripts/nrb_e_readout.py selftest \
  --out "$N/reader_selftest"

python3 -B scripts/nrb_e_readout.py preflight \
  --ref "$E" --run "$R" --source "$T" --build "$B" \
  --approved "$W/APPROVED_SHA.txt" \
  --out "$N/evidence/preflight"

python3 -B scripts/nrb_e_readout.py readout \
  --ref "$E" --run "$R" --source "$T" --build "$B" \
  --approved "$W/APPROVED_SHA.txt" \
  --line-set "$N/evidence/preflight/lines_E_call7.csv" \
  --evidence "$N/evidence" --out "$N/readout"
```

preflight는 제출 전 신원 검사와 고정 L을 출력하며 미실행 런의 물리 분기를 만들지 않는다. readout은 최종 게이트·유효성·관측·분기를 함께 출력한다. 검사 실패와 의미·문턱·관측 범위 변경 필요를 구분하여 기록한다.

### 선언 블록의 범위

아래 입력은 원자료에서 얻은 **범주 집계 시험용 부호화**다.

- 다섯 값 `1/2/3`: ① 각 셸의 하강/불변/증가.
- 열 값 `11/12/13/14`: ③ 황 각 셀의 (a)/(b)/(c)/(d).
- 한 값 `21/22/23`: ④a의 하강/불변/증가.
- 한 값 `31/32/33`: ④b의 하강/불변/증가.
- 결손은 `0` 또는 불완전 입력으로 표현한다. 어느 필수 측정의 결손이든 R5로 간다.

선언은 요약 범주에서 R0–R5로 가는 **배타성·전체 덮음**을 검사한다. 36조합은 명시적 fixture로 포함한다. 원자료 계산·정확한 기대 분기·X 처리는 §4.3 NC가 검사한다. `residual`로 공백을 숨기지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-NRB-E (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "scripts/[A-Za-z0-9_/.]+\\.py",
    "symbol": "nrb_e_",
    "roots": ["scripts/nrb_e_readout.py"],
    "expected_extra": ["scripts/nrb_e_readout.py"]
  },
  "branches": {
    "regimes": [
      [0.0, 0.0],
      [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
      [11.0, 11.0], [12.0, 12.0], [13.0, 13.0], [14.0, 14.0],
      [21.0, 21.0], [22.0, 22.0], [23.0, 23.0],
      [31.0, 31.0], [32.0, 32.0], [33.0, 33.0]
    ],
    "metrics": {
      "valid": "len(v) == 17 and all(x in (1,2,3,11,12,13,14,21,22,23,31,32,33) for x in v) and sum(1 for x in v if x in (1,2,3)) == 5 and sum(1 for x in v if x in (11,12,13,14)) == 10 and sum(1 for x in v if x in (21,22,23)) == 1 and sum(1 for x in v if x in (31,32,33)) == 1",
      "one_down": "sum(1 for x in v if x == 1)",
      "one_mixed": "max(sum(1 for x in v if x == t) for t in (1,2,3)) < 3",
      "a": "sum(1 for x in v if x == 11)",
      "b": "sum(1 for x in v if x == 12)",
      "c": "sum(1 for x in v if x == 13)",
      "d": "sum(1 for x in v if x == 14)",
      "three_mixed": "sum(1 for t in (11,12,13,14) if sum(1 for x in v if x == t) == max(sum(1 for x in v if x == u) for u in (11,12,13,14))) != 1",
      "four_a_down": "21 in v",
      "opposite": "(21 in v and 33 in v) or (23 in v and 31 in v)"
    },
    "rules": [
      {
        "name": "R5",
        "predicate": "not valid or one_mixed or three_mixed or opposite"
      },
      {
        "name": "R0",
        "predicate": "valid and not (one_mixed or three_mixed or opposite) and one_down < 3"
      },
      {
        "name": "R1",
        "predicate": "valid and not (one_mixed or three_mixed or opposite) and one_down >= 3 and a > max(b,c,d) and four_a_down"
      },
      {
        "name": "R2",
        "predicate": "valid and not (one_mixed or three_mixed or opposite) and one_down >= 3 and a > max(b,c,d) and not four_a_down"
      },
      {
        "name": "R3",
        "predicate": "valid and not (one_mixed or three_mixed or opposite) and one_down >= 3 and (c > max(a,b,d) or d > max(a,b,c))"
      },
      {
        "name": "R4",
        "predicate": "valid and not (one_mixed or three_mixed or opposite) and one_down >= 3 and b > max(a,c,d)"
      }
    ],
    "adversarial_fixtures": [
      {"name":"R1-down-a-down","mix":[[1,5],[11,10],[21,1],[32,1]]},
      {"name":"R2-down-a-flat","mix":[[1,5],[11,10],[22,1],[32,1]]},
      {"name":"R2-down-a-up","mix":[[1,5],[11,10],[23,1],[32,1]]},
      {"name":"R4-down-b-down","mix":[[1,5],[12,10],[21,1],[32,1]]},
      {"name":"R4-down-b-flat","mix":[[1,5],[12,10],[22,1],[32,1]]},
      {"name":"R4-down-b-up","mix":[[1,5],[12,10],[23,1],[32,1]]},
      {"name":"R3-down-c-down","mix":[[1,5],[13,10],[21,1],[32,1]]},
      {"name":"R3-down-c-flat","mix":[[1,5],[13,10],[22,1],[32,1]]},
      {"name":"R3-down-c-up","mix":[[1,5],[13,10],[23,1],[32,1]]},
      {"name":"R3-down-d-down","mix":[[1,5],[14,10],[21,1],[32,1]]},
      {"name":"R3-down-d-flat","mix":[[1,5],[14,10],[22,1],[32,1]]},
      {"name":"R3-down-d-up","mix":[[1,5],[14,10],[23,1],[32,1]]},

      {"name":"R0-flat-a-down","mix":[[2,5],[11,10],[21,1],[32,1]]},
      {"name":"R0-flat-a-flat","mix":[[2,5],[11,10],[22,1],[32,1]]},
      {"name":"R0-flat-a-up","mix":[[2,5],[11,10],[23,1],[32,1]]},
      {"name":"R0-flat-b-down","mix":[[2,5],[12,10],[21,1],[32,1]]},
      {"name":"R0-flat-b-flat","mix":[[2,5],[12,10],[22,1],[32,1]]},
      {"name":"R0-flat-b-up","mix":[[2,5],[12,10],[23,1],[32,1]]},
      {"name":"R0-flat-c-down","mix":[[2,5],[13,10],[21,1],[32,1]]},
      {"name":"R0-flat-c-flat","mix":[[2,5],[13,10],[22,1],[32,1]]},
      {"name":"R0-flat-c-up","mix":[[2,5],[13,10],[23,1],[32,1]]},
      {"name":"R0-flat-d-down","mix":[[2,5],[14,10],[21,1],[32,1]]},
      {"name":"R0-flat-d-flat","mix":[[2,5],[14,10],[22,1],[32,1]]},
      {"name":"R0-flat-d-up","mix":[[2,5],[14,10],[23,1],[32,1]]},

      {"name":"R0-up-a-down","mix":[[3,5],[11,10],[21,1],[32,1]]},
      {"name":"R0-up-a-flat","mix":[[3,5],[11,10],[22,1],[32,1]]},
      {"name":"R0-up-a-up","mix":[[3,5],[11,10],[23,1],[32,1]]},
      {"name":"R0-up-b-down","mix":[[3,5],[12,10],[21,1],[32,1]]},
      {"name":"R0-up-b-flat","mix":[[3,5],[12,10],[22,1],[32,1]]},
      {"name":"R0-up-b-up","mix":[[3,5],[12,10],[23,1],[32,1]]},
      {"name":"R0-up-c-down","mix":[[3,5],[13,10],[21,1],[32,1]]},
      {"name":"R0-up-c-flat","mix":[[3,5],[13,10],[22,1],[32,1]]},
      {"name":"R0-up-c-up","mix":[[3,5],[13,10],[23,1],[32,1]]},
      {"name":"R0-up-d-down","mix":[[3,5],[14,10],[21,1],[32,1]]},
      {"name":"R0-up-d-flat","mix":[[3,5],[14,10],[22,1],[32,1]]},
      {"name":"R0-up-d-up","mix":[[3,5],[14,10],[23,1],[32,1]]},

      {"name":"R1-three-of-five","mix":[[1,3],[2,2],[11,10],[21,1],[32,1]]},
      {"name":"R0-two-down-three-flat","mix":[[1,2],[2,3],[11,10],[21,1],[32,1]]},
      {"name":"R1-plurality-without-majority","mix":[[1,5],[11,4],[12,3],[13,2],[14,1],[21,1],[32,1]]},
      {"name":"R5-first-mixed","mix":[[1,2],[2,2],[3,1],[11,10],[21,1],[32,1]]},
      {"name":"R5-sulfur-tie","mix":[[1,5],[11,4],[12,4],[13,2],[21,1],[32,1]]},
      {"name":"R5-opposite-before-R1","mix":[[1,5],[11,10],[21,1],[33,1]]},
      {"name":"R5-opposite-before-R0","mix":[[2,5],[11,10],[23,1],[31,1]]},
      {"name":"R5-multiple-reasons","mix":[[1,2],[2,2],[3,1],[11,5],[12,5],[21,1],[33,1]]},
      {"name":"R5-missing-cell","mix":[[1,4],[0,1],[11,10],[21,1],[32,1]]},
      {"name":"R5-missing-report-only-input","mix":[[1,5],[11,10],[21,1],[32,1],[0,1]]},
      {"name":"R5-duplicate-color","mix":[[1,5],[11,10],[21,2],[32,1]]},
      {"name":"R5-empty","mix":[]}
    ]
  },
  "references": [
    {"path":"CLAUDE.md"},
    {"path":"docs/PREREG_DET_BB_ALI_2026-09-25.md"},
    {"path":"docs/VERDICT_DET_BB_ALI_2026-09-26.md"},
    {"path":"docs/PREREG_DET_NT_ETA_2026-10-01.md"},
    {"path":"docs/VERDICT_DET_NT_ETA_2026-10-01.md"},
    {"path":"scripts/check_prereg_preflight.py"},
    {"path":"scripts/check_gate_referents.py"},
    {"path":"scripts/check_doc_canonical.py"},
    {
      "path":"scripts/nt_eta_readout.py",
      "flags_existing":["MUST_KEEP","def gid","def grun","def cell_outcome","def y23_cell","def aggregate"]
    },
    {
      "path":"scripts/w5_cont_readout.py",
      "flags_existing":["def log_planck","def window_ratios","def trapz_band"]
    },
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/DIAGNOSIS.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/astra_audit_r1.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/astra_audit_r2.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/astra_audit_r3.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/astra_audit_r4.md"},
    {
      "path":"/gpfs/kjhan/lumina/cmfgen_map/diag2_20261002/lines_traj.py",
      "flags_existing":["D912-1500","912 <= v[7][5] < 1500"]
    },
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/nteta_20261001/stage_run1.sh"},
    {
      "path":"/gpfs/kjhan/lumina/cmfgen_map/nteta_20261001/run1/job.slurm",
      "flags_existing":["readonly GPU_PROFILE=h100-a100x2","expected_partition=h100,a100; expected_gpu_count=2","module purge && module load gnu13/13.2.0 cuda/13.0.2","REQUEUED_EXECUTION_FORBIDDEN"]
    },
    {
      "path":"/gpfs/kjhan/lumina/cmfgen_map/nteta_20261001/run1/input/resolved_lumina.exports",
      "flags_existing":["LUMINA_A210_SPRODUCER_CAPTURE=\"0\"","LUMINA_NLTE_SKIP_Z=\"\"","LUMINA_XSHELL_RESPONSE=\"0\"","LUMINA_NT_FULL=\"0\"","LUMINA_CMF_FINE_CONT_NLTE_SOURCE=\"1\""]
    },
    {
      "path":"/gpfs/kjhan/lumina/cmfgen_map/nteta_20261001/run1/RUN_FOOTER.txt",
      "flags_existing":["job_id=409603","elapsed_seconds=40070","terminal_state=SUCCESS","fixed_te_loader_consumed=1"]
    },
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/APPROVED_SHA.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1/source_sha_at_build.txt"},
    {
      "path":"/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1/binary.sha256",
      "flags_existing":["7800484fb309addabfa5b2f6b879f3ab09b814db83d3c5db2f39debff3f1b390"]
    },
    {
      "path":"/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_plasma.c",
      "flags_existing":["LUMINA_BB_NRB","bb_nrb_arm","line_net_bracket_rates","[BB-NRB]","LINE_NET_BRACKET_NOT_APPLICABLE"]
    },
    {
      "path":"/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_cmfgen.c",
      "flags_existing":["W5_CONT_LINES_FILE","SPRODUCER-CAPTURE","line_producer_terms_captured","SOBOLEV-LINE-OPERATOR"]
    },
    {
      "path":"/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/line_net_rate.c",
      "flags_existing":["line_net_bracket_rates","LINE_NET_BRACKET_NOT_APPLICABLE","LINE_NET_BRACKET_LEDGER_ABSENT","line_net_cmfgen_exponx"]
    }
  ]
}
```
