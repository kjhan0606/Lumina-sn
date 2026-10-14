# Ada 단일 GPU 이식 — 제한 실행 결과·코드 감사

2026-10-14. 감사자: astra. 동일 설계 저자의 좁은 구현·직접시험 검토이며 독립 설계 검증이 아니다. 검토 범위는 Ada 할당 변경 두 파일, 직접시험·빌드 증거와 R2 실행 러너이다. 기존 Newton 물리 설계 전체를 다시 감사하지 않았다.

**Q-GOAL: PASS.** 사용자가 지정한 lageunha Ada에서 기존 FREE_T R2를 동일 방정식·grid·허용오차로 실행하기 위한 변경이다. 이를 생략하면 기존 exact 수송 할당량이 단일 32 GB VRAM을 넘어 실행을 막는다. 이식 성공은 CMFGEN 동등 스펙트럼 달성을 뜻하지 않는다.

**Q-LEAN: APPROPRIATE.** 기존 할당 helper의 명시적 compile-time 분기, 실제 장치 oversubscription 탐침, 같은 장치에서 기존 합성 fixture의 기본/managed 바이트 비교, 그리고 기존 시간 제한을 가진 생산 시도 하나로 충분하다. 추가 batching framework·생산 CPU A/B·역사 전체 재감사는 요구하지 않는다.

**판정: 제한된 생산 시도 1회 투입 가. 본 검토에서 미해결 본질적 코드 차단 사항은 발견하지 않았다.** 실제 실행은 아래 자원 제한으로 감독하며 실패 시 자동 재시도나 CPU 전환을 하지 않는다. 이는 생산 수렴·시간 목표 달성 판정이 아니다.

## 소스와 빌드

- 작업 소스: `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-ada-20261014`.
- 변경은 `src/cmf_exact_multigpu.cu`와 `.h` 두 파일, 29 additions / 2 deletions이다. `LUMINA_CMF_EXACT_MANAGED_MEMORY=1`에서만 `cudaMallocManaged(..., cudaMemAttachGlobal)`을 사용한다. 기본 `cudaMalloc` 경로는 유지된다.
- 각 shard 첫 할당에서 현재 장치의 managed memory와 concurrent managed access capability를 검사한다. 미지원·CUDA 오류는 기존 오류 반환 흐름으로 전달된다. kernel, copy, rounding, halo, envelope, convergence 코드는 변경되지 않았다.
- header는 managed 모드의 reported allocated bytes가 VRAM 상주량이 아니라 할당 주소 공간임을 명시한다.
- 실제 `tools/run_checks_and_build.sh`는 `sm_89`, CUDA 13.0.2, GNU 13.2.0으로 빌드하며 `src/a210_coupled_newton.c`를 명시적으로 연결한다. 빌드 종료값은 0이다. 사용하지 않는 함수 등의 warning이 있으나 이 변경에 따른 링크 실패는 없다.
- 생산 바이너리 SHA256: `6bff25d9336ac4232826e4867439ab2d1724a69483b809674717f8566ba02fba`.

## 직접 증거와 재현 범위

- `managed_probe.log`: visible=1, managed=1, concurrent_managed=1. device total 33,794,359,296 B보다 큰 35,941,842,944 B managed allocation에서 각 4 KiB 페이지 표식 GPU 쓰기·검사가 mismatched_pages=0으로 통과했다. 탐침 소스도 확인했다.
- `build1/bench_default.log`와 `bench_managed.log`: 같은 50-shell / 8192-bin / 66-ray epoch fixture, devices=1, 13 iterations, contexts=1, bounds=7, upper=4, numerical_repairs=0으로 PASS. 결과 hash는 둘 다 `62893f78bb1068ff`; 경과는 각각 약 33.49 s, 33.53 s이다.
- `build1/result_default.bin`과 `result_managed.bin`의 `cmp`를 감사자가 다시 실행해 일치를 확인했다. `binary.sha256` 검증도 다시 통과했다. `source_verify_after.log`는 189개 항목이고 빌드 후 source 검증을 기록한다. 감사자가 현재 소스 manifest 검증을 다시 실행했으며 오류 출력이 없었다.
- GPU 탐침과 fixture 자체는 운전자가 lageunha에서 수행한 증거를 판독했다. 감사자는 GPU 시험을 중복 실행하지 않았고 로그인 노드에서 모델 계산을 하지 않았다.
- fixture의 약 318 MB 할당은 생산 크기 oversubscription을 시험한 것이 아니다. 35.94 GB 탐침도 생산 수송 접근 패턴이나 약 74 GiB 규모의 성능 보증이 아니다. 1GPU와 기존 2GPU의 reduction grouping 차이로 마지막 비트까지 동일하다고 확대하지 않는다.

## 러너와 실행 한계

검토 러너: `tools/run_r2_ada.sh`, SHA256 `dcb3719a3427ae0716832791b5a6cdbdd477573a2d1ced3f462f85bb504f8f77`. `bash -n` 통과. host/GPU UUID·독점 lock·빈 work·한 번만 생성하는 RUN_ATTEMPTED·binary/source/deck 신원 확인을 갖춘다. env를 정리한 뒤 봉인 exports를 읽고 FREE_T, dep=0, NE closure=1, 외부 반복6, exact GPU devices=1을 확인한다. exports의 ppd12, tol1e-8, refinements42와 기존 물리 설정을 유지한다.

첫 완료 manifest는 시작 후 21600 s 이내의 mtime이어야 한다. 60 s polling이므로 실제 중단 통지는 최대 약 60 s 늦을 수 있고, TERM 정리에도 최대 약 60 s가 필요하다. 6 h 이후 만들어진 manifest를 성공으로 인정하지 않는다. 최종 성공은 exit=0, DET-TRANSACTIONAL COMPLETE 1회, R7_MATERIAL_PHASE_COMMITTED 6회 및 iter0005 manifest 존재를 함께 요구한다. 처음 제시된 COMPLETE 6회 조건은 운전자가 의미를 확인해 수정했고, 본 감사는 수정된 러너를 검토했다.

실행 감독 계약은 `systemd-run --user --wait --pipe`와 RuntimeMaxSec=86280, TimeoutStopSec=120, MemoryMax=192G, CPUQuota=1600%, TasksMax=512, KillMode=control-group, Restart=no이다. runtime과 stop 최대 시간을 합쳐 24 h, 단일 GPU 최대 약 24 GPU·h로 제한한다. 실제 unit 생성 직후 활성 상태에서 이 설정의 적용을 운전자가 확인한다. 본 감사는 계획된 설정을 검토했으며 아직 생성되지 않은 생산 unit의 적용을 확인했다고 주장하지 않는다. linger=no 환경에서 SSH 연결 종료 등으로 조기 중단되면 성공으로 처리하거나 자동 재개하지 않는다.

MemoryMax=192G는 호스트 전체 여유와 다른 프로세스 한도이며 생산 peak가 이 안에 든다는 실측은 없다. 실패 시 자원 제한 실패로 기록한다. 관리 메모리 page migration과 Ada double-precision 비용 때문에 첫 외부 반복 6 h·전체 24 h 달성은 미확인이다. 작은 fixture의 약 0.04 s 차이를 생산 속도 추정으로 사용하지 않는다. 기존 H100/O와 성능 비교 시 GPU 종류·개수 및 managed paging 변경을 반드시 명시한다.

생산 성공 판정은 첫 실제 반복과 최종 산출을 본 뒤 운전자가 내린다. 현재 판정은 이 시간·자원 한도 안에서 물리 기준을 바꾸지 않는 1회 시도가 정당하다는 뜻이다.
