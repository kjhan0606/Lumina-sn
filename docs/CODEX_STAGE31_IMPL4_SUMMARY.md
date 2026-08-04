완료했습니다. 최종 판정은 **rung6 PASS, rung7 KA3 FAIL / STOP**입니다.

- KA1 수치·차수는 기존 결과와 동일하게 PASS.
- KA3 공식 최상위 triple:
  - `p_obs = 2.0011103392` — PASS
  - finest L2 `= 9.5801490e-5` — PASS
  - finest L1 `= 1.0565893e-4` — FAIL
  - sign-uncertain `51036 / 242152 / 1019773` — FAIL
  - finest non-finite enclosure 28 — FAIL
- acceptance 완화, clamp, 기존 `src` 수정, 커밋 없음.
- KA3 실패로 KA2와 산란 rung8·9는 미구현·미실행했습니다.

납품물:

- [전체 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_IMPL4.md)
- [rung6 패치](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/s31_rung6.patch)
- [rung7 패치](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/s31_rung7.patch)
- [KA1 결과](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_results/ka1_rev3.json)
- [KA3 결과](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_results/ka3_rev3.json)
- [KA1 로그](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_logs/rung6_ka1_rev3.log)
- [KA3 로그](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_logs/rung7_ka3_rev3.log)

패치 순차 재생, byte identity, strict C11 빌드와 확장 self-test도 PASS했습니다.