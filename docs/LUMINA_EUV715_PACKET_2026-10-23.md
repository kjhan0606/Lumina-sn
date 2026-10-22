# S III 715Å 실제 CMFGEN formal 광선 — 2026-10-23

## 목적과 판정 경계

고정 물질70/J71의 S III 광흡수율 바깥 native49 초과는
600–750Å에서 주로 나오며, 715Å 단일 점의 동일반경 C F/CMFGEN J는
바깥 약`2.12`, 안쪽 약`0.968`이다. 1450Å 중앙광선의 기하를
외삽한 추정에서는 715Å 주파수 대각 항/물질 χ 깊이가 약
`12.8–13.9`이나, 이는 **715Å 원광선 측정이 아니다**.
따라서 715Å 부근 두 주파수의 실제 CMFGEN packet으로
주파수 유입·유효 source·중앙광선 끝점 강도를 확인한다.
한 광선 또는 두 주파수의 결과를 전체 각도 광흡수율,
자기일관 이온분포·스펙트럼 동질성으로 대체하지 않는다.

## 봉인 실행

- 원 완성 장: 물질70/J71, 전197291 formal 주파수,
  `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429062/field/FORMAL_PRIMARY_FIELDS.bin`,
  SHA256 `d9073ac830d8a70beffc731dc919140b2b97ccff7e8d960060d42a7f7ca60563`.
- 격리 빌드: `/gpfs/kjhan/cmfgen_runs/lumina_euv715_packet_build_20261023-UqNPQK/build`,
  실행파일 SHA256 `9e71841d897e0a9e19063bbe6565f2e5778b4f461ddd1bec67a228ff678e01aa`.
  원 formal 커널과 수송 객체는 재사용하고 계측 선택만
  `outer71, frequency44360–44361`로 바꿨다. 빌드 manifest SHA256
  `25e6a8c2af4db618649cbf557d058f964051262b8b14a267e582963be5237fbe`.
- 수동 실행 root:
  `/gpfs/kjhan/cmfgen_runs/lumina_euv715_packet70_20261023-orvJq8/capture`.
  Lageunha CPU16/OMP16·GPU0, 벽 상한2400초, formal 장 단회,
  원 물질·장 read-only 입력, native 갱신·SE/RE 0.
  2026-10-23 KST 실행 중 확인; 완료 판정은 아직 없다.
- 코드 변경 커밋 `5914822`, 직접시험
  `tests/test_prepare_cmfgen_center_packet.py` 및
  `tests/test_cmfgen_formal_capture.py` **11통과/1.30초**.

## 완료 후 확인 순서

기존 strict 장 SHA 검사는 이전 1450Å 포획처럼 부동소수점
비트 차이로 실패할 수 있다. 실패를 성공으로 바꾸지 않고,
실제 전주파수 장의 기하·주파수·호출수와 J/H/K/N 수치 차이를
별도 판독해 계측 무간섭 범위를 평가한다. 그 다음 두 주파수 packet의
광선합과 JIT2 native49/50 J를 원 장에 대조하고, 실제 중앙광선
χ·주파수 깊이와 청색 유입 source 비중을 계산한다.
원 packet·실패 기록·기존 장은 덮어쓰지 않는다.

현재 최종 목표: `active/NOT_MET`.
