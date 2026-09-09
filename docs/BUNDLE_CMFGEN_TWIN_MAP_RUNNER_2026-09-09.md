# CMFGEN twin — H200/H100 실행 전환

2026-09-09 사용자 「그렇게 해요」: 노드 미고정 H200/H100 runner 수정 승인.
기존 DET-CMFGEN-TWIN-MAP 묶음의 실행 부분 확장이다. 물리 질문·비교 기준·실행 횟수를 추가하지 않는다.
운전자 설계감사 **Q-GOAL=PASS, Q-LEAN=APPROPRIATE, DESIGN=PASS**. 구현 검수 전 제출 금지.

## 목적·범위·입력

고정-T 프로파일과 population 덤프를 봉인 바이너리에 공급하는 실행 경로를 완성한다.
부모 계약은 `docs/PREREG_CMFGEN_TWIN_MAP_2026-09-08.md`. 분율·전자비·공개 세대 의미는 개정 4 그대로다.
계획/검수/실행/재현은 운전자, 구현/직접 시험은 GPT-5.6-sol. 전체 출력 뒤에만 기존 종료감사를 한다.

- 출발점 `78fed0d`, 두 코드 파일 identity는 개정 4 묶음문에 고정.
- BASE `/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260906T234024Z_dc1_ne_closure_on`.
- 소스 `/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth`, 기존 바이너리 SHA256
  `ca721b07cb4fef632b1a0a0b2d3c7b11cdaac0e95f74e6cc56b131a7c5449059` 유지.
- 증거 `/gpfs/kjhan/lumina/cmfgen_map/evid_rev4_20260909`. CPU loader 50/50와 참조 열 INDIVIDUAL
  근거를 소비한다. prepare 재실행은 필요 없다.
- RUN은 `/gpfs/kjhan/lumina/cmfgen_map` 아래 신규 독립 경로 하나, staging 직전 절대경로를 고정한다.

## 변경집합·수용기준

`scripts/stage_cmfgen_twin_map.sh` 수정과 필요시 전용 `scripts/run_cmfgen_twin_map.sh` 템플릿 하나 추가.
생산 C/CUDA·원자량·바이너리·분석 산식은 변경하지 않는다. 과거 여러 진단 모드를 이식하지 않는다.

1. 생성 job: 단일 노드/task, GPU2, CPU32, mem64G, time04:00:00, partition h200,h100,
   node 미고정, requeue 금지. GPU 클러스터는 **syntax**다. grammar 제출의 불량 노드 제외 규칙은
   유지하되 syntax에 존재하지 않는 grammar 노드명을 넣지 않는다.
2. scheduler 배치와 CUDA_VISIBLE_DEVICES 두 장을 소비한다. 같은 종류의 H200 또는 H100,
   고유 UUID, 각각 ≥80000 MiB 확인. mask를 0,1로 덮어쓰지 않는다. A100/A40·한 장·중복 GPU·
   잘못된 partition·자원 초과는 모델 시작 전 거부. 봉인 바이너리의 sm90/PTX 호환 근거는 제출 전 확인.
   미지원이면 임의 재빌드 금지.
3. BASE 읽기 전용·새 RUN 독립 복사, 기존 경로/링크/무손상/총량 시험 유지. 생성 runner·실효 exports·
   profile을 staging manifest에 포함. 상속 LUMINA 환경을 제거하고 staged exports만 복원한다.
   허용 delta는 T profile 추가, 균일 진단 T 제거, 세 덤프 스위치=1, BASE→RUN 경로 치환뿐이다.
   실제 argv는 봉인 모델의 두 outer iteration 실행이며 MC 피드백은 활성화하지 않는다.
4. 옛 A100-only/10020 K 강제와 census/SLB 승인문·post-model seal은 새 실행 경로에서 제외한다.
   binary/deck/catalog 동일성·strict env·고정 profile·두 장 사용·두 반복·오류 처리는 보존한다.
5. 실행은 한 번. model rc·종단·경과시간·할당·GPU identity·고정-T 실제 소비 근거를 기존 footer와
   원로그에 남긴다. CPU loader 성공만으로 GPU 소비=1을 쓰지 않는다. 실패 원출력 보존,
   모델 재호출/자동 재제출 금지. 실행 성공은 물리 PASS가 아니다.
6. 부모 §4.3의 최종 공개 연결은 런 후 관측 대상이다. deferred 상태인데도 실제 최종 NLTE 파일명/
   세대를 미리 요구하는 stager 모순은 해소할 수 있다. 가짜 경로/세대/공개 증거 금지.
   source상 예상 기록 경로와 실제 공개 확인을 구별하고 분석기 수정 없이 못 읽는 항목은 결손으로 남긴다.
7. grammar-debug CPU 시험에서 실제 생성/실행 경로를 사용한다. 사설 모델/GPU 조회 stub으로 정상
   H100/H200 두 장, 자원/마스크/identity 오류, 실패 rc, 재실행 거부, T 소비 증거 누락을 확인한다.
   selftest 우회가 production 호출에 노출되면 안 된다. 실제 BASE 입력/exports 구조의 생성 결과도
   판독하되 GPU 바이너리를 CPU 시험에서 실행하지 않는다. 실물 staging/제출은 운전자만 한다.

## 자원·보존·증거

GPU **1회·2장×4h=8 GPUh**, CPU32×4h=128 할당 CPUh, 단일 CPU 오프라인 누적1 CPUh,
신규 총량16 GiB 유지. 기존 동일 반복의 약3h·64G 할당 실적, 실제 덱의 덤프 상한, CPU 시험
시간/메모리와 실제 free bytes로 수용한다. 새 T runtime/peak 미측정을 완료 예측으로 바꾸지 않는다.
출력/로그/분석·재현/임시량을 포함해 쓰기 전 예약하고 기존 저장 통제를 완화하지 않는다.
runner의 로그/덤프 한도와 실패 보존을 검수한다. 부족·초과면 중단, 추가 지출/재런 금지.
실패 변경과 원출력은 보존하고 BASE·다른 dirty 파일은 건드리지 않는다.
증거는 같은 묶음의 diff/해시, 직접 시험 명령/rc/시간/메모리, 생성 job, 자원 카드,
staging identity, job ID/종단/원출력이다. 새 일반 gate·별도 감사 장부는 만들지 않는다.
재현: `bash scripts/stage_cmfgen_twin_map.sh --selftest`; 별도 runner가 있으면 그 `--selftest`도 실행.
검수 후 실제 staging/sbatch 명령을 이 문서에 추가한다.

## 운전자 설계감사

Q-GOAL: 물리 측정인가 정비인가? 이번 수정은 실행 정비, 뒤따르는 등록 twin 1회는 T 교란을
제거한 CMFGEN 분율/전자비 관측이다. C/CUDA 수리는 물리 변경을 섞고, 기존 runner는
A100/균일-T/과거 후처리 때문에 사용할 수 없다. 실행에 직접 필요한 수정이다. **PASS**.
Q-LEAN: 기존 원로그/footer/manifest를 소비하고 과거 진단 모드·중복 runner 변환을 대체한다.
한 구현·한 검수로 제출까지 연결하며 획득런·하드웨어별 복수런·상주 모니터·일반 인증기는
추가하지 않는다. **APPROPRIATE**. 물리 기준과 비용 상한을 보존하므로 **DESIGN PASS**.
구현·자원 수용·제출 자격 PASS는 아직 아니다.

## 실행 전 발견 — writer 경로 미연결, GPU 제출 보류

runner 발주 세션 `01a08504-d8bf-7f71-9c79-a09ab80879ab`, 증거 루트
`/gpfs/kjhan/lumina/cmfgen_map/impl_runner_20260909`. 문서 봉인 `ba2df38`, dispatch preflight rc=0.

운전자가 봉인 소스의 실제 DET 호출을 추적했다. `src/lumina_cuda.cu:8077` 부근의
`det_transactional_requested` 분기는 `cmfgen_run()` 뒤 plasma CSV만 쓰고 **8110에서 return**한다.
Saha 최종 writer **10280**과 per-iter writer **8648**은 그 아래의 우회된 legacy 경로다.
`src/lumina_cmfgen.c:7525`의 seed predictor 및 **7538**의 R7/A210은 private candidate로 풀고,
`src/nlte_population_candidate.c:278`은 explicit effect policy=1, allowed=0을 설정한다.
`src/lumina_plasma.c:20916`의 NLTE dump는 DIAGNOSTIC_FILES 허용 시에만 실행되며,
같은 파일 **21193**의 candidate solve는 일반 덤프를 계속 억제한다.
**일반 dump 스위치를 켜는 runner 변경만으로 최종 Saha/NLTE 파일을 얻을 수 없다.**
이는 GPU 재현 결과가 아니라 exact source control-flow 확인이다. 기존 prepare의 writer token
존재 확인은 실행 경로 도달성 확인이 아니었다. 최종 연결을 단순 런 후 미결로만 둔 판단을 정정한다.
기존 `tests/nlte_population_candidate_selftest.c:160–169`를 봉인 소스에 링크한 CPU 시험도
rc=0으로 재현했다. 이 시험은 실제 candidate begin 후 일반 DIAGNOSTIC_FILES 효과가 꺼져 있음을
검사한다. 로그 `candidate_effects.log` SHA256
`64cf888e9488a2ced8a0c072fa72a03c3e03865bf1a35ac03edddba7370371f8`.
이 시험은 private 효과 정책의 executable 재현이며, 전체 GPU 호출 경로의 실행 재현은 아니다.
추가 참조 source SHA256: `lumina_cmfgen.c`
`12b83b14c0ba06793127fa5d4744b9a17cc400d34bea98d768dd084114fedd6e`,
`nlte_population_candidate.c` `539c3da5787d546a0d7daf5c9724d9e68e75c6d961814d4b1fbd23a385d0ddf9`.
CPU 재현은 grammar-debug에서 아래처럼 실행했다(기존 출력은 재실행 시 별도 임시 경로 사용):

```bash
S=/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth
T=$(mktemp -d /gpfs/kjhan/lumina/cmfgen_map/candidate_repro.XXXXXX)
gcc -O2 -Wall -Wextra -std=c11 -I"$S/src" -o "$T/candidate" \
  "$S/tests/nlte_population_candidate_selftest.c" "$S/src/nlte_population_candidate.c" \
  "$S/src/population_contract.c" "$S/src/opacity_publication.c" "$S/src/emissivity_publication.c" \
  "$S/src/atomic_internal_energy.c" "$S/src/cmfgen_adiabatic.c" -lm
"$T/candidate"
```

새 바이너리/C/CUDA 변경은 현재 승인 범위 밖이다. runner 구현은 보존·검수하되 GPU 제출은
보류하고, 물리 계산식은 그대로 두는 최종 공개 writer 추가와 새 바이너리 검증에 대해 사용자에게
별도 범위 승인을 요청했다. BASE 재빌드·덮어쓰기·추가 획득런은 하지 않는다.

한편 바이너리 `cuobjdump --list-elf`에서 33개 ELF 중 sm90 code image를 확인했다
(`binary_cuda_images_verified.txt`). 기존 A100 job333531은 02:58:58, CPU32/mem64G,
MaxRSS32213028K였다(`base_sacct.txt`). model rc=0이지만 옛 후처리 때문에 Slurm FAILED였으며,
이 실행시간/메모리는 새 T 완주 보장이 아니다. 이 두 근거만으로 writer 부재를 면제하지 않는다.
