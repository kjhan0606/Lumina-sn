Stage 3.1 상세 설계 초안을 작성했습니다.

문서: [CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md)

핵심 결정:

- 균일 `ln ν` 격자, blue→red BDF2 + 선형 short-characteristics
- shell별 Gauss-Legendre `μ`를 p-ray로 변환
- 3.1은 formal solver/J와 무가속 coherent-scattering 반복까지만 담당
- KA ①~③의 정확해, 3-grid Richardson 절차와 개별 문턱 명시
- parity59에는 총 `χ_ν,η_ν`가 없어 판별 벤치를 `UNRESOLVED-INPUT-1`로 판정
- Wave-3.2용 최소 binary dump schema와 정확한 계기 지점 제안
- s8의 600–3000 Å 6대역 비교 및 기존 Gamma 재계산 코드 연결 정의
- 모든 물리 clamp/floor 금지, 발견 시 좌표를 포함해 UNRESOLVED 처리
- 구현 규모 9–14 PM 및 단계별 검증 게이트 수록

이번 작업에서 추가한 것은 이 설계 문서 한 개뿐이며, `src/`, `scripts/`, `Makefile`은 수정하지 않았습니다.