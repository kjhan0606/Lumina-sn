# 발주서 — DET-LTEUNMAP-EPS 구현 (Opus 4.8 fresh, 2026-09-17)

앞겹 = 사전등록 `docs/PREREG_DET_LTEUNMAP_EPS_2026-09-17.md` **전문**(개정 1 적용본, 집행 기록 1 포함) — 구현자는 그 문서를 읽고 그대로 따른다. 이 발주서는 계약의 어떤 값도 다시 적지 않는다.

## 0. 경위 (이 절의 값은 요구가 아니다)
사전감사(집행 기록 1)가 «폐합 차단» 1건을 냈고 개정 1 로 해소됐다. 구현은 개정 1 이 적용된 사전등록 §3 을 따른다.

## 1. 집행 조건
- 작업 트리 = 사전등록 §3 이 지정한 **동결 트리 복사본**. repo(`~/BACKUP/Eunha.A1/Claude/Lumina-sn`)의 src/tests 는 **수정 금지**(읽기 참조만).
- 변경 파일 = §3 변경집합 표의 행 전부, 그 밖 파일 byte 불변. 봉인 런 루트(`consistent_ionization_state_20260911/…`, `superlevel_b_…`)는 **읽기만**.
- 컴파일·selftest 실행은 로그인 노드 금지 — 반드시 `scripts/gd "<cmd>"`(grammar-debug) 로만. 재현 레시피는 `consistent_ionization_state_20260911/operator_reproduce.sh` 의 gcc 플래그·소스 목록을 참조.
- fail-closed·byte-동일·census 형식·반환 의미는 §3 원칙 그대로. 게이트 OFF 경로에 한 바이트도 새 동작을 넣지 않는다.
- 산출 보고: 첫 줄 모델 ID 자기보고 → 변경 파일별 diff 요약(행 수) → selftest 명령·rc·출력 요약 → 사전등록 §3 의 각 항목이 어디에 구현됐는지 행 번호 표 → 구현자가 판단한 미해결·위험(있으면).
