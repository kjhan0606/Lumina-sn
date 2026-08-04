C4 FAIL 4좌표를 모두 협소 수정하고 A5 산출물을 완료했습니다.

- [전체 구현 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE32_A5_IMPL.md)
- [rung 1 — EW I/O fail-closed](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung1_ew_io_fail_closed.patch)
- [rung 2 — atomic 전수 재스윕](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung2_atomic_resweep.patch)
- [rung 3 — iter override 계약](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung3_iter_override_contract.patch)
- [rung 4 — NaN fail-closed](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a5_rung4_nan_fail_closed.patch)

핵심 검증 결과:

- `/dev/full` write/close: RC `-1`; 정상 artifact: RC `0`
- 공유 capture counter 19종: 1/8-thread 각각 `200000`, 누락 없음
- iter=7 무단 우회: RC `1`; 명시 override: `NON-CONTRACT`, RC `2`
- `b[0]=NaN`: residual `inf`, gate pass `0`
- 네 패치 순차 적용 및 최종 byte 비교: PASS
- 통합 seeded-defect, round-trip, Python compile, diff check: PASS

물리 산출 변경, 신규 clamp/floor/cap, 모델/GPU 실행, 커밋은 없습니다.