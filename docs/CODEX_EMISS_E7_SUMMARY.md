전체 보고서를 작성하고 산술 재현까지 검증했습니다: [CODEX_EMISS_E7.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E7.md)

핵심 판정:

- E6의 s8 BALL `11.9773×` 중 99.98094%는 자기 `J`를 다시 읽은 coherent 항입니다. `S_fixed/CMFGEN=0.00228248`, 즉 고정 source는 오히려 438× 부족합니다.
- UV opacity의 CMFGEN 대비 결손량은 UV-bin `chi_nu` 부재로 **UNRESOLVED**입니다. 다만 s8 opacity의 epoch-proxy 선 기여는 97.1–99.6%여서 광범위한 선 누락은 기각됩니다.
- s8 대표 Fe III/Co III `b_u=2.32–4.86`으로 초열성만으로 12×를 설명할 수 없습니다. 과이온화와 s8에서는 동행하지만 s0에서는 동행하지 않아 통계적 상관도 **UNRESOLVED**입니다.
- 남은 물리적 `J` 과잉 후보는 경계조건/산란 operator, EPAY 재형상, 빈 폭·선 투영 순으로 지정했습니다.

재현 소비기와 결과:

- [emiss_e7_arithmetic.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e7_arithmetic.py)
- [summary.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e7/summary.json)
- [band_shell_fixed_opacity.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e7/band_shell_fixed_opacity.csv)
- [line_departure_proxies.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e7/line_departure_proxies.csv)

최종 검증은 300개 band-shell 행, 60개 line-shell 행 및 canonical 수치 assertions를 모두 통과했습니다. 신규 런·수송 solve·clamp·commit은 수행하지 않았습니다.