# 발주서 2 — DET-LTEUNMAP-EPS 구현 (개정 2 반영, Opus 4.8, 2026-09-17)

앞겹 = 사전등록 `docs/PREREG_DET_LTEUNMAP_EPS_2026-09-17.md` **전문**(개정 2 적용본, 집행 기록 1·2 포함). 이 발주서는 계약의 어떤 값도 다시 적지 않는다. 발주서 1(`docs/ORDER_DET_LTEUNMAP_EPS_IMPL_2026-09-17.md`)은 이 문서로 대체된다.

## 0. 경위 (이 절의 값은 요구가 아니다)
1차 구현은 §3 변경집합 중 fine 경로 몫을 완결했고 selftest 를 통과했다. 구현자가 신고한 위험(연산자 패스 미deposit)을 운전자가 소스로 확인해 개정 2 가 커밋 경로(bin 조립기)를 변경집합에 추가했다. 이 발주는 그 추가분의 구현이다.

## 1. 집행 조건
- 작업 트리·수정 허용 파일·봉인 규칙·실행 규칙(로그인 노드 금지, `scripts/gd` 전용)은 발주서 1 과 동일하며 사전등록 §3 이 정한다.
- 1차 구현분(fine 경로·`used_nlte` 발행·`radeq_line_table_ensure`·selftest)은 **유지**하고, §3 「개정 2 — 커밋 경로 적용」 항목 전부와 census 형식 변경(조립기 행)을 추가 구현한다. 게이트 OFF 경로에 새 동작을 넣지 않는다.
- selftest 를 §3 표의 항목대로 확장한다(조립기 경로: 게이트 OFF byte-동일 · unmapped 선 `el` 원값 · mapped 선 `el=1` 유지 · 표 부재 시 시작 BLOCKED · −1 fail-closed · 조립기 census 행 형식).
- 산출 보고: 첫 줄 모델 ID → 파일별 diff 요약(1차 대비 증분과 동결 원본 대비 총계, `diff -rq` 결과 첨부) → selftest 명령·rc·항목별 PASS/FAIL → §3 항목(개정 2 포함) → 구현 위치 표 → 미해결·위험(정직 기재).
