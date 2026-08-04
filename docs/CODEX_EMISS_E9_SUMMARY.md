E9 차터를 완료했습니다. 전체 보고서는 [CODEX_EMISS_E9.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E9.md)입니다.

핵심 판정:

- BALL 사전등록 `0.93665×` 대비 산술 `0.93596×`, 재조립 source `0.92791×`, stage31 `J_det=0.93229×`로 모두 적중했습니다. 산란 재순환 기전은 진폭 수준에서 확정입니다.
- B1–B4도 사전등록 대비 −1.49%~+1.65%로 전부 ±10% 이내였습니다.
- 1208 Å trip은 재발하지 않았습니다. clamp, solution-negative, sign-uncertain, nonfinite 모두 0이며 3회 출력 SHA가 동일합니다.
- 다만 B0/B1은 여전히 `8.29×/4.92×`입니다. scalar epsilon은 전체 진폭만 닫으며, 스펙트럼 형상 수리에는 비대각 재분배 행렬이 필요합니다.
- 기존 event prefix에서 305개 입력 빈, 92,287개 희소 edge, 1,856,667쌍을 복원했습니다. 에너지 보존 상대오차는 `7.50e-7`입니다.
- 수리 후 목표는 UV `42.9→23.8%`, blue `5.8→14.5%`로 사전등록했습니다.
- 물리적 결합 고정점은 Stage 4 population/rate ownership 없이는 승인 불가입니다. frozen 진단 solve만 Stage 4 없이 가능합니다.

재현 스크립트와 산출물은 [validation/emiss_e9](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e9)에 있습니다. 신규 모델/GPU 런, clamp/floor, 구조 수리 구현, 커밋은 모두 하지 않았습니다.