전체 보고서를 작성했습니다: [CODEX_EMISS_E4_VERDICT.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E4_VERDICT.md)

최종 판정은 **`UNRESOLVED-SOLVER-GUARD`**입니다.

- A/B 입력 인증 PASS. A payload는 같은-run `chieta_iter10`과 byte-identical.
- 미정의: 전이 2.4214%, line-shell 2.6054%, pre-EPAY A-reference 기여 5.9398%.
- 미정의 전이는 A값으로 폴백하지 않고 직접 B emissivity를 0으로 둠.
- A-lane BALL: `A/CMFGEN = 11.7037914`; frozen 기준은 `11.9770975`.
- B-lane은 1208.743 Å(B1)에서 certified-negative solution으로 중단.
- 동일 실패 3회 stderr SHA까지 일치.
- 따라서 B 대역표와 Γ는 `UNRESOLVED`; 정식화 진범 확정/기각/부분 중 어느 분기도 선언하지 않음.
- 성공한 A 결과: [jdet_A.tsv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e4/jdet_A.tsv)
- 신규 모델/GPU run, clamp, acceptance 변경, commit 없음.