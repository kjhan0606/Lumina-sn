완료했습니다. 전체 보고서는 [CODEX_EMISS_E13.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E13.md)입니다.

핵심 판정:

- **색인 불일치 기각:** LFMAT001은 ν 오름차순, LCMFCE01은 ν 내림차순이며 적용기가 정확히 한 번 변환합니다.
- **Mirror 반증:** B2→B0 점유율이 54.92% → 64.82%, B0 `J_det/CMFGEN`이 26.43 → 53.97로 더 악화했습니다.
- **진동수 sanity:** 전체 energy-weighted 평균은 `νout/νin=1.003594`로 소폭 상향입니다. UV 전체는 하향 우세지만 B2는 상향 우세입니다.
- **UNRESOLVED:** edge별 k-packet 표지와 event count가 없어 non-k 및 photon-count 가중 평균은 복원할 수 없습니다.
- **이론 분기:** radiative-only terminal UV-exit은 Fe II 98.40%, Fe III 89.64%, 측정 전체 92.74%입니다. 원자 radiative branch 결함은 입증되지 않았습니다.
- **다음 좌표:** 생산 strong-UV Fe III의 `p_iup≈88%`를 만드는 β/J/population 및 collision·BF·k-packet 확률 조립입니다.
- **ARTIS 20.2%:** 역사 기록은 확인했지만 정확한 자산과 적분 recipe가 없어 독립 재현은 `UNRESOLVED`입니다.

재현 도구는 [emiss_e13_index_audit.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e13_index_audit.py), 핵심 결과는 [index_branch_audit.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e13/index_branch_audit.json)입니다. 컴파일·핵심 수치·해시 assertions를 모두 통과했습니다. 생산 코드 수정, 신규 모델/GPU run, clamp, 커밋은 하지 않았습니다.