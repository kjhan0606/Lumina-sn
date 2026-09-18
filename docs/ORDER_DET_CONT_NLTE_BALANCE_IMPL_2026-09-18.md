# 발주서 — DET-CONT-NLTE-BALANCE 구현 (Opus 4.8, 2026-09-18)

앞겹 = 사전등록 `docs/PREREG_DET_CONT_NLTE_BALANCE_2026-09-18.md` **전문**. 이 발주서는 계약의 어떤 값도 다시 적지 않는다.

## 0. 경위 (이 절의 값은 요구가 아니다)
DET-LTEUNMAP-EPS 폐합(LU-A) 뒤 남은 중간 셸 LTE 고정을 겨냥한다. user 결정으로 ITER 10 레인·비열적 전강도·한 계약.

## 1. 집행 조건
- 작업 트리 = 사전등록 §3 이 지정한 동결 트리 복사본(직전 계약의 변경집합이 이미 들어 있음). §3 변경집합 표의 파일만 수정/생성. repo src/tests 수정 금지. 봉인 런 루트 읽기만.
- 컴파일·selftest 는 로그인 노드 금지 — `scripts/gd "<cmd>"` 만. 산출은 `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/selftest2/` 아래.
- fail-closed·byte-동일·census 형식은 §3 원칙 그대로. 설정 키(§3 「설정」)는 코드 변경 대상이 아니다.
- 산출 보고: 첫 줄 모델 ID → 파일별 diff 요약·동결 원본 대비 `diff -rq`(직전 계약의 변경집합 파일 + 이번 §3 표의 파일 외에 없음) → selftest 명령·rc·항목 PASS/FAIL → §3 항목 → 구현 위치 표 → 미해결·위험(정직 기재).
