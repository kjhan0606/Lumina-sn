# 무죄 확정 리스트 (2026-07-29 초판 — 재검증 대기, 각 항목의 무죄 근거를 명시해 반증 가능하게 함)
기준선 = parity42 산출물. "무죄" = 현 기준선에서 발화 0 실측(존재 자체의 면죄 아님 — 다른 config에서는 발화 가능).

| 항목 | 무죄 주장 근거 (재검증 대상) | 출처 |
|---|---|---|
| LUMINA_NLTE_INV_CEIL=1e4 | 발화 카운터/흔적 0 (stderr 0건) | partA |
| NLTE pair-solve Boltzmann@T_rad 폴백 | [NLTE-FALLBACK] 배너 0건 | partA |
| radeq no-root HOLD | 600/600 root-found, 핀·HOLD 0 | partA |
| 로더 차원 가드("reinitializing zeros") | 이 데이터셋에서 미발화 (차원 일치) | partB |
| INJECT2 τ-floor(2.0) | 미발화 | partB |
| β-소멸(mode-3) | β 분포 98.8%가 >0.99 — 소비 결손은 크로싱 통계 몫 | partA |
| fine 창 내 τ/S 짝맞춤 소비 | S/B 최대 1.07e6이어도 τ~1e-33 ⟹ 기여 ≤3e-33 유계 | partB |
| bisection 브래킷/스텝 | 이 런에서 근 탐색 실패 0 (근쌍 건너뜀 리스크는 정적 잔존) | partA |

재검증 규칙: 각 항목을 원시 산출물에서 독립 재측정. "발화 0"은 **카운터가 실제로 그 경로에 배선돼 있는지**까지 확인해야 무죄(카운터 부재로 0이면 UNVERIFIABLE로 강등).
