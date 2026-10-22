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
  실행 `1428.97초`, 의도된 `STOP97`, 완료 표식
  `71/197291/197291`. 원 엄격 바이트 해시 검사는 실패로
  보존했다. 새 장 SHA256
  `e5d62825e6e0a38fa9fe264eeb54b9f06c740184439f8d9e52777222b5e51b98`,
  실행 기록 `center-packet-run.json` SHA256
  `1ac14d859cd48d203b91de736e1c484f65ae5b3b04fb9728924027d29928a5b8`.
- 코드 변경 커밋 `5914822`, 직접시험
  `tests/test_prepare_cmfgen_center_packet.py` 및
  `tests/test_cmfgen_formal_capture.py` **11통과/1.30초**.

## 계측 무간섭의 수치 확인

엄격 바이트 해시 차이를 성공으로 재명명하지 않았다. 별도 전체
197291ν 비교에서 기하·주파수·가중치·formal 호출수는 동일하고,
J의 최대 주파수 정규화차는 `2.11e-13`, p95 `5.34e-15`다.
8개 저장 성분의 가장 큰 최대정규화차는 `2.35e-11`이다.
`center-field-comparison.json` SHA256
`e0cc009999bcda57dbad9a35d48385d56424adc1ee9725bd5873af38593c96f1`.

두 주파수의 실제 packet은 각3회(JIT0/1/2)이며, 최종 JIT2
광선합 J/H/K/N은 저장 모멘트와 IEEE 합산 오차 상한 안에서 맞는다.
원 장에 대한 최종 native49/50 J 상대차는 두 주파수 모두
`5.51e-16` 이하이다. `714.98478Å`에서 실제 J는 각각
`1.1638397927833624e-8 / 3.864721577547988e-8`이다.
`center-packet-analysis.json` SHA256
`b1e841827c2b49606514e403f0f5fc89cd00a5a45498f071f60dca3249a4d072`.

## 714.98478Å 실제 중앙광선의 항 분해

0-based native49–50의 양방향 실제 packet과 원 CMFGEN short
characteristic endpoint를 재구성했다. 재구성 끝점 강도는 packet과
일치한다. 이 표의 주파수 항은 이산 이류의 대각 성분이며
물질 흡수 깊이가 아니다.

| 방향 | 물질 `∫χ ds` | 주파수 `(α+3)∫b ds` | 합 유효깊이 | 청색측 source 비중 양 끝 |
| --- | ---: | ---: | ---: | ---: |
| M | 3.23942 | 41.62435 | 44.86377 | 99.05% / 88.80% |
| P | 2.98873 | 41.63217 | 44.62090 | 89.61% / 99.32% |

앞선 1450Å 기하 외삽의 715Å 깊이 수치는 이 **한 중앙광선**에서
실제로 재현됐다. 청색측 source 비중은 이번에 새로 측정했다.
조건부 SC 공간중점 강도는 M `2.81422e-8`, P `2.37305e-8`이고,
같은 endpoint 물질의 선형 공간보간 대안은 각각
`2.51346e-8`, `2.78590e-8`이다. 이는 실제 CMFGEN 중심의
전각도 J가 아니라 **내부 보간 선택의 민감도**다.
정본 `/gpfs/kjhan/cmfgen_runs/lumina_euv715_midray_20261023-h07pKa/midray/midray.json`
SHA256 `098d88707e561ec24a87c60f8d8aa7520867887c3dfec6f790cbf08ad82e8c0c`.

기존 같은 반경의 C F/CMFGEN J는 715Å 바깥 약`2.12`, 안쪽
약`0.968`이다. 이 값은 **각도평균 장**이고 위 표는 중앙광선
한 개이므로 두 숫자를 직접 빼서 수송 오차 비중이라 하지 않는다.
C 연속 특성선은 한 셀에서 약42 주파수 칸을 이동하므로,
두 packet만으로 동일 입사 스펙트럼의 연산자 대조도 불가능하다.
원 packet·실패 기록·기존 장은 보존한다.

현재 최종 목표: `active/NOT_MET`.
