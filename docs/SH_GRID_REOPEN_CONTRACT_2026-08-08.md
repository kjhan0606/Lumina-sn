# SH-GRID 재개방 소비 계약 — 2026-08-08

## 재개방 사유

실제 MC-EVT 덱에는 기존 `NLTE_NU_MIN=1.5e14 Hz` 이하의 양의 BF threshold가 707개
있다. 707개 모두 default-active이고 CMFGEN cross section을 갖는다. 최저 witness는
Ca II level 61, `nu_edge=5.84852771e13 Hz` (`lambda=51259.4747 Å`)다. 따라서 기존
grid 밖 BF event measure를 exact zero로 선언할 수 없다.

## 구현 사전등록

SH-GRID는 다음 순서로 이 707개 edge를 정상 domain 안으로 가져온다.

1. 상한 `3.0e16 Hz`와 기존 log 간격 `dlog=0.00529831736655`를 보존한다.
2. bin 수를 1000에서 1178로 늘리고 정렬된 하한을
   `5.84127859196e13 Hz` (`lambda_max=51323.088478 Å`)로 둔다. 이는 최저 활성 edge보다
   낮아 모든 707개 threshold가 grid 내부가 되며 기존 구간의 해상도를 낮추지 않는다.
3. radiation-field, BF sigma, event-measure, opacity/emissivity publication과 GPU upload를
   모두 새 canonical edge manifest에서 재생성한다. old-grid row padding이나 첫 bin
   대입은 금지한다.
4. `[nu_min_new, nu_min_old)`에서 각 BF edge는 기존 in-domain threshold classifier와
   CMFGEN sigma를 그대로 사용해 photoionization/recombination/event measure 적분에
   들어간다. 새 하한 아래에 active edge가 0개임을 다시 census한 뒤에만 진짜 OOG를
   exact zero로 분류한다.
5. 기대 효과의 부호·크기는 사전 지정하지 않는다. 저주파 excited-level BF 채널 707개가
   새로 장부에 나타나야 하며, total event measure 변화는 그 합과 수치적으로 닫혀야 한다.

## CMFGEN 동종 처리 대조

새 grid flight 전에 동일 population/radiation snapshot으로
`[5.84127859196e13,1.5e14) Hz`의 level별 BF rate 및 emissivity를 Lumina와 CMFGEN에서
별도 적분한다. level identity와 sigma provenance가 같은 cell만 비교하며, 이 band의
Lumina 합이 CMFGEN 합과 닫히기 전에는 MC-EVT를 재개하지 않는다. 현재 707개 census는
재개방의 필요조건 증거일 뿐, 이 수치 대조의 대체물이 아니다.
