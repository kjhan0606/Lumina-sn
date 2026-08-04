전체 보고서를 [CODEX_EPS_THIN.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EPS_THIN.md)에 작성했습니다.

-o 요약:

- 조사 1: `C/A=8.07–77.45`; 네 선 모두 tabulated CMFGEN 충돌강도이며 van Regemorter 미사용. Fe IV·Ni IV는 정상 물리, Fe III 자료를 대용한 Co IV는 `UNRESOLVED`.
- 조사 2: 80.9%는 과거 CAP128M 이벤트 원장의 `13.6314688 / 16.8389519 = 0.809520027`. `mc_J` 덤프가 아니며 방출선 이온축이다.
- 조사 3: 496,950건·에너지 3.69273247의 분포와 전체 방출 대비 몫은 `UNRESOLVED`. 비-LINE 합법 경로는 존재하나 지배 경로 판정에는 8GB 원장 통과가 필요하다.
- 잔여 작업: 계산 노드에서 [codex_unpaired_emissions_offline.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/codex_unpaired_emissions_offline.py)를 `--run-heavy`로 한 번 실행. GPU·49GB 덤프는 불필요하다.
- 검증: 조사 1 `eps_l` 4개 재생 최대오차 0.0, 조사 2 재현 PASS, 조사 3 합성·결함주입 자기시험 PASS.
- commit하지 않았고 금지된 파일은 수정하지 않았습니다.