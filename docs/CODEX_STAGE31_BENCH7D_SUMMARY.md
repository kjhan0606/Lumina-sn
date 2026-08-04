7D를 완주했습니다. 전체 결과는 [CODEX_STAGE31_BENCH7D.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_STAGE31_BENCH7D.md)에 정리했습니다.

핵심 결과:

- 가드 계수: `C=27.80641753160013`, KA 실측 최대, 안전배수 `1.0`
- seeded 시험:
  - `-0.1` → `LCMF_ENEGATIVE`
  - `-1e-4` → 기록 후 raw-bit 불변 진행
  - `NaN` → 즉시 `LCMF_ENONFINITE`
- KA1·KA3·KA2+10R: 모두 최종 PASS, 7C 수치 불변
- 생산 solve:
  - certified-negative sub-truncation `90,761`
  - excess/sign-uncertain/nonfinite/clamp `0/0/0/0`
  - transport residual `9.420355153e-7`
  - 3회 J_det SHA 동일
- 판별:
  - 전체 `J_det/J_MC=0.977181`
  - B2만 CMFGEN 반대 방향
  - Fe III `D/B=1.5713`, S II `D/B=0.521386`
  - 최종 판독: **UNRESOLVED-MIXED — 대역·Γ별 혼합 원인**

구현은 [lumina_cmf_field.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.c:1188), 계수·통계 API는 [lumina_cmf_field.h](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_field.h:115)에 있습니다. strict build와 byte-identical J_det 재검증도 통과했습니다.

신규 모델/GPU 런, clamp/floor, acceptance 수치 문턱 변경, 커밋은 없습니다.