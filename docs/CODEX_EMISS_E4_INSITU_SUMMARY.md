E4 구현과 검증을 완료했습니다. 최종 상태는 **코드 준비 완료 / 실제 50×1000 운전석 캡처 대기**입니다.

핵심 납품물:

- 전체 보고서: [CODEX_EMISS_E4_INSITU.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E4_INSITU.md)
- B-lane 조립·상태 해시·결손 장부: [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:530)
- 런타임 게이트: [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7415)
- seeded 음성 fixture: [emiss_ab_insitu_fixture.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/tests/emiss_ab_insitu_fixture.c)
- stage31·대역·Γ 판정기: [emiss_ab_insitu_e4.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_ab_insitu_e4.py:37)
- rung 변경집합: [e4_expected_changes.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/e4_expected_changes.txt)

검증 결과:

- E4 전/후 OFF production payload와 sidecar bitwise 동일
- OFF payload SHA: `869ec4d2...e347dc9`
- clean B와 `n_u×2` seeded B가 서로 다른 SHA로 검출
- 결손 선의 B 방출 무단 폴백 없음
- `make -B lumina_cuda` 통과—GPU 실행 없음
- stage31 driver `-Werror` 빌드 통과
- 5개 rung 순차 적용 및 whitespace 검사 통과
- 신규 clamp/floor 0
- 모델/GPU 캡처 및 물리 판정 수치는 생성하지 않음
- 커밋하지 않았고 기존 무관 변경은 건드리지 않았습니다.