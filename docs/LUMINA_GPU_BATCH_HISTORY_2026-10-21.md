# GPU batch/forest로 native90 자체 복사장과 Si/S율 확보

## 실제 판정 — 2026-10-22 00:02

**batch-C의 생산 채택을 기각한다.** 새 root xSx3Z2의 실제GPU 동일mesh/합성2시험은 통과했고 native13003 전ν를 처음 완결했다. 그러나27.1339초와 미세기준해 최대정규화차0.00713852(0.71385%)는 원10초/1e-4 기준을 모두 넘었다. 내부오차8.5472e-10<=tol8.5586e-10가 통과해도 약한UV 요구 정확도를 보장하지 못했다. gpu-local.json SHA256 `503c60e4d6643a300318d6bfdcc4aab146f8e4f620938270ee8008ad4cea4b07`, 근거와3점 절대값은 [22일 인계서](HANDOVER_2026-10-22.md)에 있다.

원 중단선대로 C의 추가 미세튜닝/시간 확대/조건부 전체장은0이다. GPU91초를 한 번 정산하고4GPUh 예약을 해제했다. 잔여약73.6182222GPUh/live수치작업0이다. Astra에 다른 일관 물리·수치 이산화 B의 구체 구조를1건 평가 요청했다(제출01a15efa…). 아래 계획/구현·당시 실행기록은 보존하며 실패를 소급 통과시키지 않는다.

## 목표·선택

최종 목표는 CMFGEN과 자기일관 Si II/III·S 이온분포/스펙트럼/물리량 동질성이다. 이번 묶음 산출물은 native13003 완료 전ν·기존 미세참조·10초 비용 판정, 통과 시 무참조 native90 J1/J2·J1 실제잔차·생산 Si/S율이다. 코드/국소/합성 통과로 묶음을 종료하지 않는다. 현재 최종 active/NOT_MET다.

Astra 후속 평가 제출 `01a15edb-37bc-7e61-ba0a-b52b94fe4b56`는 **현 C의 생산 채택 기각은 유지하되 별도 batch/forest 엔진으로 한 번만 재설계**하기로 했다. 구간0의0.036초 전ν 정합과 refinement의 순차 발행 구조가 근거다. launch 비용이 지배한다는 실측 증명은 아니다. 물리 이산화를 함께 바꾸는 B보다 같은 물리식의 병렬 구조를 우선한다. 새 batch-C도 원10초 비용/정확도/메모리 계약을 못 지키면 기각하고 다른 일관 이산화 B로 전환한다. 이후 C 미세튜닝/시간 연장은 하지 않는다.

## 고정 입력·불변·보존

기존 M70/native90/epoch1683072초/전체1115020ν/완성 계수/원 기하·경계를 유지한다. 입력 정본은 `lumina_native90_pair_20261021-e7DQZB/native-contract.json` SHA256 `0fe9718063d6d1ba46a1dbb32e4ccace4430f8d4f7040ef0be86ee3a025ac598`다. 이전 C 준비/실패는 `lumina_gpu_C_local_20261021-BRDZTA`로 보존한다. 새 실행은 새 root/계약만 허용하고 원본/기본CPU/A/C를 덮어쓰지 않는다.

전체ν 이력·signedχ/zeroη·기존 양의 모멘트·8점 물리구간 입사 재구성·공간χ 표현/허용 기준·반경 보간→ESJ→주파수 순서·경계는 불변이다. 참조 J71/구91셀 J를 초기/경계/산란원으로 쓰지 않는다. 물질/SE/RE/이산화 B 변경은 이번에 함께 하지 않는다.

## 구현 구조

- 여러 독립 슬라브에서 필요한 missing leaf의 coarse/left/right/overlap을 descriptor 묶음으로 만든다.
- 독립 bin 생성→각 tree level 합성→query를 collective launch한다. 트리별 offset을 분리하고 각 트리 내부 시간순을 유지한다. 단순 async 전환/launch 크기 조절은 이번 후보가 아니다.
- 공간 전파는 기존 순서의 replay로 수행한다. 실제 입사가 바뀌면 오차도 재계산하고 과거 unsigned error를 재사용하지 않는다.
- 완결된 배치의 후보/오차/현재최대값으로 전ν global norm을 재판정한다. 현재최대값 감소 시 이전 통과 슬라브도 다시 본다. 중간 값으로 배치를 통과시키지 않는다.
- prefix 차감/역변환 및 전ν×전체leaf 배열은 금지한다. cache+forest+관련 임시 합계1GiB를 넘지 않도록 배치를 나눈다.
- bin/쿼리/트리/padding/replay/재구성/관측의 기존 GPU 작업계수와 상한을 유지한다. CPU scalar formal과 같은 단위로 부르지 않으며 캐시 재방문 때 초기화하지 않는다.

## 직접시험·실제 판정

동일 mesh에서 기존 ordered map/replay 및 독립 기준과 먼저 대조한 뒤 적응 문제에 연결한다. 합성 전체 JHK/저장 자체장 반환·J 재결합·narrow line/partial/blue/signed/underflow를 확인한다. 합성 own-return 잔차2.33418e-8을 최종 자체수렴 증거로 올리지 않는다.

첫 실제 구간0/13003은 같은 J=I=0/전체ν다.0의 기존 출력3e-12,13003의3점 미세참조1e-4와 완료 전ν **10초 이내**를 모두 요구한다. 시간/정확도/메모리 실패 시 batch-C도 기각하고 B로 전환한다. 실패 때 마지막 완료 배치 leaf·오차/tol·작업량만 보존하며 부분 출력은 완성장으로 반환하지 않는다. 국소 시간의 전체 F 선형 외삽은 하지 않는다.

## 자원·실행 소유·다음 연결

운전자가 구현/직접시험/실행/재현한다. Astra는 이번 묶음 선택/평가 담당이며 의뢰는 완료·종료했다. 다른 모델/Fable/Q-GOAL/Q-LEAN은 없다.

- RTX5000 Ada GPU0 1장, CPU4, host/device 각16GiB. **cache+forest+관련 임시 합계1GiB**.
- 준비·검증·실제0/13003 최대30분1회, 국소 각60초/목표10초, 자동retry0.
- 통과할 때만 새 정본의 완성장 캠페인 최대210분1회/F최대2회·각90분, 율 추출 포함. 묶음 합계 최대4GPUh, 새 출력10GiB.
- F당2,000,000NF/구간당200,000NF 상당 GPU 작업 상한과 leaf4096/depth32를 유지한다.
- 시작 전 부하/장치 여유·저장소/입력 신원·바이너리/코드/재현 명령을 새 실행 계약에 고정한다. GPU 시작 때만 예약하고 종료 후 실제할당으로 한 번 정산한다. 현재 잔여약73.6435GPUh/예약0/계산0이다.

23:54 구현 상태: `cmf_forest_cuda.cu/.h`의 트리별 offset·collective bin/각level/query와 다중 슬라브의 순서 보존 replay, `lumina_gpu_forest.py`의 메모리 분할·완결 배치 적응을 연결했다. 초기 전ν 경로/입사 재구성/물리 광선·경계·관측 누적은 재사용한다. forest와 초기 트리는 같은 공간을 재사용하고 맵/replay/ray/descriptor까지 합계1GiB를 사전 제한한다. 여러 슬라브를 선택해도 공간 leaf 순서는 바꾸지 않고 실제 입사로 오차를 다시 계산한다. 마지막 완료 배치의 leaf/error/tol/work만 실패 결과에 남긴다.

`lumina_gpu_forest_build_20261021-bUgCPn/test.log`는 host **237 passed/4 deselected/41.84초/exit0**, SHA256 `f160c41cf47f6b0d8ff385672ddf85060f1c726cf4c728e42b362ca5ffe20318`다. wire ABI/forest 분할·겹치지 않는 offset·메모리 합계·떨어진 현재 최대값 재판정과 기존225시험을 포함한다. 이 root의 CUDA 컴파일도 exit0다. 아직 batch GPU 수치/실제구간 성공 증거는 아니다.

새 실제 준비 root는 `lumina_gpu_batch_local_20261021-xSx3Z2`다. gpu-local-plan.json SHA256 `1865fdfdba8226c267b47a80d902b7a4a7bb2656308cb65b463f09f5f7b81748`, history_cuda.so `8d0aa08ff46b7928c7923bc01b6cb66ba374c7984080ca1d24a33954dba2b722`, driver.so `6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4`다. actual.sh의 단일30분 준비/동일 mesh·합성 GPU시험→실제0/13003 판정이 봉인됐다. GPU 시작 때4GPUh 예약하며 결과 전 native 전체장은0이다. 실패한 직렬 C root를 재시작하지 않는다. 소스/바이너리/입력·명령을 보존하고 새 결과만 출판한다. 이 전체 구현/실행은 한 묶음의 내부 작업이다.

실제 완성장 쌍 뒤에도 고정T/약한UV 정규화·공간 근사·자기일관 SE/RE/스펙트럼 미해결 사항을 분리해 평가한다. 저장장 쌍은 최종 물리 동질성 판정의 대체가 아니다.
