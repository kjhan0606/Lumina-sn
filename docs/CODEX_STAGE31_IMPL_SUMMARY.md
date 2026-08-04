KA1 acceptance 실패로 규율에 따라 구현을 중단했습니다. 골격은 PASS했으며 KA3·KA2·산란 반복은 실행하지 않았습니다.

핵심 KA1 결과:

| χR | h4 I rel L2 | h4 J rel L2 | max error | p_obs | 판정 |
|---:|---:|---:|---:|---:|---|
| 1e-3 | 4.88e-5 | 3.56e-5 | 3.01e-7 | 1.503 | FAIL |
| 1 | 6.64e-5 | 4.40e-5 | 2.41e-4 | 1.504 | FAIL |
| 100 | 1.09e-4 | 5.23e-5 | 1.14e-3 | 0.464 | FAIL |

`χR=100`에서는 설계가 강제한 outer half-cell constant extension의 optical depth가 h4에서도 `0.390625`여서 boundary layer가 비점근적입니다. 최악 좌표와 상세 분석은 보고서에 기록했습니다. Clamp/negative/non-finite는 모두 0이고 3회 출력 해시는 동일했습니다.

산출물:

- [전체 구현 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_IMPL.md)
- [KA1 수치 JSON](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_results/ka1.json)
- [rung1 골격 패치](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/s31_rung1_skeleton.patch)
- [rung2 KA1 패치](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/patches/s31_rung2_ka1_pure_absorption.patch)
- [rung2 자기검증 로그](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/s31_logs/rung2_ka1_pure_absorption.log)
- [CMF solver 헤더](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.h)
- [CMF solver 구현](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c)

판별 벤치는 capture 부재를 `UNRESOLVED-INPUT-1`로 fail-closed 기록했으며 C1/C2로 η를 추측하지 않았습니다. 모델/GPU 실행, 기존 `src/` 수정, Makefile 수정, 커밋은 하지 않았습니다.