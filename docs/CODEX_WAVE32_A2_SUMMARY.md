구현과 검증을 완료했습니다. 전체 보고서는 [docs/CODEX_WAVE32_A2_IMPL.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE32_A2_IMPL.md)에 작성했습니다.

핵심 결과:

- R1 `SUPER_LEVELS={0,1}` × s0/s8: 12/12 byte-identical PASS
- COMMIT=1 음성 대조: `requested=1`, `performed=0`, `blocked_by=boundary_gate`
- R3 bf field/GPU 조건을 공유 helper로 단일화
- R5 필수 manifest 카운터 5개 및 runtime hook 3개 실배선
- D3 정책 5좌표 반영, pair 결함은 EW에 복제하지 않음
- `make clean && make`: RC=0, error=0, warning=60
- R7 `LCMFCE01` 왕복: 424 bytes, bitwise PASS, SHA-256 일치
- 신규 clamp/floor/cap, 모델 실행, GPU 실행, 커밋: 모두 0

현 boundary gate 때문에 COMMIT=1 양성 격리는 UNRESOLVED이며, 실제 parity59 χ/η capture도 GPU/모델 실행 금지에 따라 정직하게 UNRESOLVED로 남겼습니다.