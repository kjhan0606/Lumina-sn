구현 완료했습니다. 검증기 자체는 PASS지만 처분 미승인 상태이므로 `TRAD_FIX=UNRESOLVED`로 유지합니다.

신규 파일:

- [verify_trad_fix.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_trad_fix.py)
- [CODEX_IMPL_L0_TRADFIX.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_IMPL_L0_TRADFIX.md)

검수 결과:

- 사전 기대값 9항목 모두 지정 자릿수에서 일치
- 네 상태 × 50셸 × 5개 밴드 JSON 산출
- persistent `T_rad/W`, GPU `d_T_rad/d_W` 접근 157곳 분류, 미분류 0곳
- gate OFF 원본 50행 binary64 동일: PASS
- 독립 W 훼손 검출: PASS
- 합성 CMFGEN Planck fit/log-J 보간: PASS
- 전체 최종 명령 종료코드: `0`

CMFGEN 전량 대조는 구현했지만 현재 `syntax` 노드에서는 143 MB 처리를 실행하지 않았습니다. lageunha용 정확한 명령과 판정 위치는 보고서 §9에 있습니다.

처분 권고는 후보 2 우선, 후보 1은 진단 대조군, 후보 3은 `REJECTED`입니다. `10020 / 10470.093 / 14172.549 K`의 정본 선택은 별도 승인 사항으로 남겼습니다. 근거 없는 `4.93%`는 철회하고 사용하지 않았습니다.

`src/`, 덱, `/gpfs`는 수정하지 않았으며 commit·push도 하지 않았습니다.