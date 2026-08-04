E1 캠페인을 완료하고 전체 보고서를 작성했습니다: [CODEX_EMISS_E1.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E1.md)

핵심 결과:

- 600–3000 Å `J_det/CMFGEN`
  - 권위 A: **11.7038**
  - A 재생: **11.6578**
  - CMFGEN-pop B: **11.4055**
- 인구 스왑이 제거한 과잉은 **2.79%**뿐입니다.
- 방향성은 강하게 **조립식 운반자**를 가리킵니다.
- 다만 A 재생 항등이 `χ` 8.23%, `η_fixed` 35.17% 차이로 실패했고, B 커버리지도 54.13%이므로 공식 판정은 **UNRESOLVED**로 유지했습니다.

추가 확인:

- `*OUT → Saha–Boltzmann → POP*` 왕복은 최대 약 `1.18e-5 dex`로 검증됐습니다.
- 완전 교체 가능 이온은 9개, 교체된 line-shell cell은 61.51%입니다.
- 외곽 s44–s49는 RVTJ 범위 밖이라 외삽 없이 A를 유지했습니다.
- B J_det 수치 guard의 clamp/sign/nonfinite 계수는 모두 0입니다.
- CMFGEN의 직접 `A_ul n_u` emissivity/profile 처리와 Lumina의 `eps_l B(T_e)`·coarse-bin·EPAY 재형상 사이 구조 차이를 원소스 행 단위로 정리했습니다.

재현 코드와 전체 산출물은 [validation/emiss_e1](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e1)에 있습니다. 두 frozen artifact는 독립 checker를 통과했으며 커밋은 만들지 않았습니다.