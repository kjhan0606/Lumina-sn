Wave-3.2 수리 배치를 구현하고 보고서를 작성했습니다.

- R1: 권위 레인 기반 τ/source 쓰기 제한. s0·s8 armed `COMMIT=0` 비교 6건 모두 byte-identical.
- R3: pair/EW bf 장 선택을 공유 헬퍼 하나로 통합. JEQB에서 estimator 0, Planck 경로 3,081,675 bin 확인.
- R5: 독립 조립 원장, runtime/hot-cold 계측, Kramers 폴백 구현. Fe II Γ 변화는 `+0.0158125 dex`.
- R4: CMFGEN signed DIE 관례를 정확히 복제할 데이터가 없어 UNRESOLVED로 동결.
- R6: Fe V `ma_rr`가 0/200이므로 II–V 창은 미구현하고 경계-질량 설계안만 제출.
- frozen oracle 빌드와 Saha selftest 통과. 신규 모델/GPU 런 및 커밋 없음.
- COMMIT=1 실런과 seeded-defect 음성 대조는 명세대로 B 단계 잔여 검증으로 명시했습니다.

주요 산출물:

- [전체 구현 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_WAVE32_A_IMPL.md)
- [plasma 구현](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:374)
- [element-wide 구현](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c:272)
- [R1 byte 불변식 검사](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/wave32_r1_byte_invariant.py)
- [R3/R5 provenance 비교](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/wave32_r35_compare.py)
- [Fe V 데이터 실측](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/wave32_fe5_inventory.py)

기존 dirty worktree의 사용자 변경은 보존했으며, Makefile과 다른 `src/` 파일은 이번 배치에서 수정하지 않았습니다.