# CMFGEN twin — 최종 공개 출력 연결

## 실행 이행 검토의 첫 질문 — 2026-09-09 「다음 묶음 단계 진행」

1. **Q-GOAL — PASS:** 목표는 실제 최종 Saha/NLTE 분율과 공개 전자밀도를 얻어 CMFGEN과
   비교하는 것이다. 출력 연결·빌드·등록 판정런을 생략하면 목표 관측을 얻을 수 없다.
2. **Q-LEAN — APPROPRIATE:** 기존 묶음을 검수→빌드→단일 판정런→판독으로 마무리한다.
   새 획득런·상주 계측·단계별 감사는 추가하지 않는다. 원출력은 기존 mapper가 직접 소비한다.

두 질문을 먼저 판단했으며 기존 수용기준·봉인 BASE·8 GPUh/128 할당 CPUh·16 GiB 상한을
그대로 유지한다. **DESIGN PASS**이며 아래 역사적 구현 미결의 면제나 물리 PASS는 아니다.

2026-09-09 사용자 「승인」: 봉인 BASE 보존, 계산식 변경 없이 최종 Saha/NLTE 출력 추가와 새 바이너리 검증.
기존 twin 묶음의 미완성 출력 연결을 마무리하는 변경이다. 새 물리 질문이나 GPU 런을 추가하지 않는다.
운전자 사전감사 **Q-GOAL=PASS · Q-LEAN=APPROPRIATE · DESIGN=PASS**. 구현/검수/실행 PASS 아님.

## 목적·소유·입력·보존

DET 성공 반환 직전에 최종 공개 상태의 Saha 단계와 NLTE 개별 준위를 기록하고 기존 판독기로 연결한다.
기획·검수·통합·빌드·실행·독립 재현은 운전자, 구현·직접 시험은 GPT-5.6-sol이다.
전체 산출 후 기존 묶음 종료감사를 수행하며 구현마다 별도 모델 감사를 추가하지 않는다.

- BASE와 원 바이너리/원 소스는 runner 묶음문에 고정된 경로와 SHA256 그대로 읽기만 한다.
- 출발 소스는 `/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth`의 **실제 파일**이다.
  git HEAD만으로 복원하지 않는다. 해당 트리에 기존 미커밋 변경이 있으므로 src/tests/Makefile의
  독립 복사와 원본 해시 목록으로 출발 bytes를 고정한다. 원본이나 메인의 다른 dirty 파일을 덮어쓰지 않는다.
- 새 개발/빌드 경로: `/gpfs/kjhan/lumina/cmfgen_map/writer_worktree_20260909`.
  구현·검수 증거: `/gpfs/kjhan/lumina/cmfgen_map/impl_final_writer_20260909`.
- 기존 runner 발주가 끝난 뒤 그 파일과 개정 4 분석기를 새 개발 경로로 통합해 **한 변조 작업**만 진행한다.
- 기존 준비 증거는 유지한다. run 전용 소스/바이너리 identity만 명시적으로 추가하고 BASE identity와 구별한다.
  새 실행용 바이너리는 별도 파일로 빌드/봉인한다. BASE 바이너리 해시를 새 값으로 바꿔치기하지 않는다.

## 변경집합

허용 파일: `src/lumina_cuda.cu`의 실제 DET 성공 반환 직전 호출부, 작은 전용 final writer C/H,
그 직접 CPU selftest, 해당 빌드 연결, `scripts/cmfgen_twin_map.py`,
`scripts/stage_cmfgen_twin_map.sh`, `scripts/run_cmfgen_twin_map.sh`의 직접 연결 부분.
구현이 선택한 C/H·시험의 정확한 경로와 diff는 코드 검수에 고정한다. 기존 계산 함수·자료구조·
원자량·율·iteration·seed·상속 물리 환경은 수정하지 않는다. 새로운 물리 스위치는 만들지 않는다.

## 수용기준

1. 기존 `cmfgen_run()` 성공 뒤, 공개 객체가 살아 있는 DET 반환 전에서만 새 writer를 호출한다.
   private candidate에서 일반 dump 효과를 켜거나 solve를 다시 호출하지 않는다.
   기존 덤프 스위치로 Saha/NLTE 출력을 활성화하며 비활성화 시 기존 실행을 보존한다.
2. Saha는 공개 atom의 모든 stage×shell, NLTE는 실제 계산된 ion의 모든 level×shell을 기록한다.
   기존 판독 필드명/단계 정의를 유지하고 중복·누락 없이 출력한다. NLTE `n_ion_total`은 같은 공개
   atom 값, T/ne는 같은 공개 plasma를 읽는다. 음수/NaN/Inf는 교정·재정규화하지 않고 원값으로 남긴다.
   출력 실패는 계산 성공으로 숨기지 않으며 기존 출력은 덮어쓰지 않는다.
3. 파일명은 최종 공개용 고정 이름으로 정한다(예: `nlte_levels_final.csv`). 호출 카운터를 반복 번호나
   공개 세대로 간주하지 않는다. 동시 공개 기록의 작은 manifest는 경로/실제 행수/권위/실제
   population generation/실제 Te generation/Te manifest를 기록하고 마지막에만 완성 표시한다.
   새 일반 장부가 아니라 세 CSV를 판독하기 위한 원출력 계약이다.
4. **세대 공간 구분:** atom/NLTE committed population generation과 NLTE required generation이
   일치해야 하고 population의 partition Te stamp는 공개 Te publication과 연결돼야 한다.
   Te generation과 population generation이 서로 같은 숫자일 필요는 없다. 둘을 억지로 같게 쓰지 않는다.
   stale/null/missing/부분 출력은 최종 공개 비교로 채택하지 않는다. 이 관계는 데이터 계보 조건이지
   물리 정확도 조건이 아니다. 선언된 source 위치와 실제 snapshot manifest/CSV를 함께 대조한다.
5. CPU selftest는 **생산에서 호출할 동일 writer**를 사용한다. 서로 다른 Te/population 세대의 정상
   fixture, 기대 원값/준위·stage 피복, 출력 활성/비활성, stale stamp/세대, 파일 충돌/쓰기 실패,
   비유한·음수 원값 보존을 시험한다. 모든 공개 객체 및 참조 배열을 출력 전후 memcmp하여 불변을 확인한다.
   기대값은 writer로 만들지 않는다. 기존 DET 성공 경로에 실제 호출이 연결됐는지도 확인한다.
6. mapper/stager/runner는 새 run artifact identity를 명시적으로 소비한다. BASE 선적용 수치와
   raw BASE/source hash는 유지한다. 미리 최종 generation을 꾸며 넣지 않고 등록 경로의 실제
   런 후 manifest를 읽으며, 분석은 사전 기대치 파일을 갱신하지 않는다. 기존 분모/집계/표본은 불변.
7. runner의 실제 입력 소비를 수정한다. 원 프로파일 파일 SHA와 loader의 canonical temperature
   manifest SHA는 다른 값이므로 비교하지 않는다. 실제 50셸 T 공개/소비와 해당 hash를 대조한다.
   과거 DPROXY 진단 BLOCKED를 계산 FATAL로 일괄 취급하지 않고 model rc와 이번 필수 원출력/실제
   DET·fixed-T 종단을 소비한다. 실패 진단은 삭제하지 않는다. 시간/메모리/저장 상한은 계속 집행한다.
8. 직접 writer 시험 및 연결된 mapper/stager/runner 시험을 한 번의 코드 검수로 묶는다.
   가능한 범위에서 새 binary의 GPU code identity를 원본과 대조한다. CPU writer 불변 시험만으로
   전체 GPU 실행의 수치 동등성을 증명했다고 쓰지 않는다. 원본 대비 source 변경은 I/O뿐이어야 한다.

## 자원·실행·재현

GPU 판정런 **1회·2GPU×4h=8GPUh**, CPU32×4h=128 할당 CPUh, mem64G, 총 신규16GiB 유지.
소형 CPU 시험과 빌드는 기존 단일 CPU 누적1CPUh 안에서 실행하며 CPU 시간을 기록한다.
grammar-debug에서 시험한다. CUDA toolkit이 없는 이 노드에서 GPU 실행을 시도하지 않는다.
필요시 syntax의 compute 노드에 **GPU 미요청 단일 CPU 빌드 job**을 제출하고 실제 AllocTRES의
GPU0을 확인한다. 이는 컴파일만 수행하며 모델 실행/획득런은 하지 않는다. 초과 예상 시 재승인 없이 늘리지 않는다.
실제 GPU job은 노드 미고정 H200/H100·no-requeue, 기존 runner 계약 그대로다.
모든 정상/실패 산출물과 source delta/해시를 보존하고 재런·BASE 덮어쓰기는 하지 않는다.
직접 시험 명령/rc/시간/메모리, 정확한 build 명령/환경/산출 해시, 생성 job/exports와 원출력·판독을
기존 묶음의 증거로 연결한다. 검수 후 실제 재현 명령을 이 문서에 추가한다.

## 운전자 설계감사

Q-GOAL: 분율 자료를 실제로 얻기 위한 I/O 수정이다. writer가 없는 상태로 GPU를 돌리면 전자밀도
외 목표 관측을 잃는다. 계산을 재호출하거나 private trial을 공개로 포장하는 대안은 더 큰 물리 변경이다.
공개 성공 반환 직전에 읽기 전용 writer를 붙이는 것이 직접적인 최소 수정이다. **PASS**.
Q-LEAN: 최종 CSV와 그 공개 연결만 추가하고, 기존 analyzer/runner가 곧바로 소비한다.
새 물리 gate·상주 프로세스·획득런·단계별 감사는 만들지 않는다. 누적 dump counter와 사전 가짜
generation 대신 최종 한 snapshot을 사용한다. **APPROPRIATE**. **DESIGN PASS**.

이 문서는 이전의 생산 C/CUDA·새 바이너리 변경 제외 및 BASE/RUN 동일 binary 요구를
**이번 I/O 변경의 명시적 계보 연결에 한해** 대체한다. 물리 산식·BASE 보존·비용 상한은 대체하지 않는다.

## 구현 착지 후 운전자 검수 — 수정 중

sol 세션 `01a08514-a3a8-7982-a549-be109ab4b13f`는 정상 종료했다.
`impl_final_writer_20260909/SOL_REPORT.md`에 writer/consumer/runner/stager/mapper 직접 시험을 기록했다.
이 구현의 검수에서 다음 두 회귀가 확인되어 아직 빌드·실물 staging을 하지 않았다.

- stager가 새 `lumina_cuda`를 0640으로 만들어 runner의 실행 가능 조건을 충족하지 못한다.
- CUDA 호출부에서 기존 무조건 plasma 출력을 대체했으나 writer는 덤프 비활성화 시 아무것도 쓰지 않는다.
  비활성 실행의 기존 plasma 출력 보존이라는 수용기준에 어긋난다.

동일 구현 소유자 sol의 수정 작업 `01a08531-2872-7472-b463-41313b834a41`에 두 회귀와 직접
회귀시험을 맡겼다. 함께 실제 정상 producer→mapper 정규화, public T와 staged profile의 직접
대조, 모델 프로세스의 파일 크기 상한을 마무리한다. 별도 모델 감사나 GPU 런을 추가하지 않는다.
기존 CPU 불변 시험을 전체 GPU 수치 동등성으로 해석하지 않는다.

### 수정본 재현 및 CPU 빌드 이행

수정 sol은 정상 종료했다(`CORRECTION_REPORT.md`). 운전자도 grammar-debug에서 writer,
정상 producer→mapper, stager, mapper를 재실행하여 rc=0을 얻었다(`operator_cpu.log`,
wall 2.199s/user 1.374s/system 0.773s). 원소스 174개가 출발 해시와 그대로 일치하며,
기존 파일의 변경은 Makefile 빌드 연결과 CUDA의 DET 반환 직전 I/O 연결뿐이다.
생산 소스 검수는 CPU 빌드를 허용한다. 빌드 입력은 `source_final.sha256`에 고정했다.

runner의 public T 비교에는 실제 프로파일의 `1.8900E+04`와 writer의 `18900`을 다른
문자열로 오인하는 문제가 남아 sol이 **runner만** 수정 중이다. 수치 허용오차 없이 유한한
양의 binary64 값으로 비교하고 실제 표기 차이를 정상 시험에 넣는다. 이 수정은 봉인한
생산 빌드 소스를 바꾸지 않으며, GPU 제출 전에 직접 재현한다.

### 빌드 완료 및 실행 자원 수용

CPU 빌드 `syntax:337600`은 syn08에서 COMPLETED/rc=0, 72초였다. 실제 AllocTRES는
cpu=1,mem=8G,node=1이며 GPU 할당은 없다. compiler time은 wall71.719/user62.460/system6.746초다.
새 binary는 `impl_final_writer_20260909/build/lumina_cuda`, SHA256
`ca47ff50237f281dac0beb515da6d8b9c1d50e882e5b10ce6dab3a573a902b4b`다.
11개 sm90 cubin 중 10개는 원본과 byte 동일하다. 나머지 `lumina_cuda.21.sm_90.cubin`은
익명 namespace 이름의 `730445bf`→`8ef46168` 78곳을 제외하면 파일 전체가 같다.
따라서 GPU 명령·상수의 동일성을 확인했다(`gpu_code_comparison.log`). 전체 GPU 수치 동등성
또는 물리 PASS 주장은 아니다.

CPU writer/consumer를 강제 재컴파일한 추가 재현도 rc=0이었다. wall0.544314/user0.359048/
system0.161599초, peak RSS36864KiB다. 자원 카드는 이 실측과 기존 동일 반복의 333531 실적,
Slurm 시간·메모리 제한, 모델의 1GiB/파일 제한 및 stager의 쓰기 전후 총량 검사를 근거로 수용했다.
새 T의 runtime/peak memory는 여전히 **미측정**이며 4h 완주를 보증하지 않는다.
로그/기타 런 출력 4GiB, 분석 1GiB, 빌드·재현·감사 2GiB를 예약하고 population dump 공식
368324400 bytes와 전체 신규 16GiB 상한을 유지한다. 이전 resource card/identity/expectations는
`evid_rev4_before_final_writer_20260909`에 보존했다.

등록 RUN은 `/gpfs/kjhan/lumina/cmfgen_map/run_twin_hopper_20260909` 하나다.
실효 입력은 이 RUN의 `input` 아래 독립 복사하며 원 BASE는 수정하지 않는다.
