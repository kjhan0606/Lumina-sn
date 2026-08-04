E2는 **UNRESOLVED**, 따라서 E3는 **NOT RUN**으로 종료했습니다.

- `chi_total` 상대 L1: `8.226059670973736e-2`
- `eta_fixed` 상대 L1: `3.517015818466757e-1`
- 요구치: `≤1e-10`
- grid 경계와 캡처 `J`: bitwise 항등
- 주원인: 캡처는 iteration-10 입력 상태지만 E1은 iteration-11 종료 CSV로 선 τ, BF와 EPAY를 재구성함
- 캡처에는 pre-EPAY 선/연속 항과 정확한 iteration-10 population이 없어 잔차를 유일하게 역분해하거나 항등 조립할 수 없음
- 권위장 pass-through는 재조립으로 인정하지 않음
- E3 선 방출 교체와 stage31 계산은 관문 규율에 따라 실행하지 않음

산출물:

- [전체 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E2E3.md)
- [E2 감사 스크립트](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_assembly_identity_e2.py)
- [기계 판정 JSON](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e2e3/assembly_identity_audit.json)
- [대역별 항 분해 CSV](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e2e3/term_gap_by_band.csv)

두 chieta 계약 검사와 48행 대역 감사 검증은 PASS했습니다. E1 재생도 약 50초 로컬 실행으로 기존 SHA-256을 재현했습니다. `src` 수정, 신규 모델/GPU 실행, clamp, 커밋은 없었습니다.