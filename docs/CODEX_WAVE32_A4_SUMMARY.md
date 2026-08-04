납품 완료했습니다. 6/6 rung PASS이며 커밋은 생성하지 않았습니다.

- [전체 구현 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE32_A4_IMPL.md)
- [rung 1 — RC semantics](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung1_rc_semantics.patch)
- [rung 2 — atomic counters](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung2_atomic_counters.patch)
- [rung 3 — OOM propagation](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung3_oom_propagation.patch)
- [rung 4 — iter consumer contract](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung4_iter_consumer_contract.patch)
- [rung 5 — M_V independent audits](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung5_mv_independent_audits.patch)
- [rung 6 — runtime deck](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/w32a4_rung6_runtime_deck.patch)

핵심 검증 결과:

- 패치 1→6 순차 적용 및 6→1 역적용 폐합
- 공식 frozen-cell CPU 배터리 12/12 byte-identical
- s8 S `SCOPE_FAIL`은 RC 0으로 계속 진행
- bad config/OOM/I/O 음성은 모두 RC -1
- atomic 1-thread/8-thread 값 동일, OFF 대조 0/0/0
- M_V 정본과 d_k 관련 artifact 불변
- 보존행·route·q coupling 결함 모두 검출
- CPU build와 CUDA source compile 통과
- 신규 모델/GPU 실행, 신규 clamp/floor/cap 없음
- `git diff --check` 통과