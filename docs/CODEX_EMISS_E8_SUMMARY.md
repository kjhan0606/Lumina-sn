완료했습니다. 전체 보고서는 [CODEX_EMISS_E8.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8.md)에 있습니다.

핵심 판독:

- s8 UV `chi_coherent/chi_total = 97.7713%`; B1–B4는 99.88–99.994%.
- `chi_coherent`는 전자산란뿐 아니라 `(1−eps_l)·chi_line`을 포함합니다.
- s8 `eps_eff=1.90567×10⁻⁴`, 재순환 이득은 5247.49×.
- 필요한 이득 5247.41×와 0.00152% 이내로 일치하여 UV 11.977× 과잉의 amplitude를 완전히 설명합니다.
- MC 열적 파괴율은 0.243682%로 결정론 `eps_eff`의 12.787×.
- 이벤트 prefix에서 실제 coarse-bin coherence 파괴율은 95.1164%입니다. 현 ALI가 이 형광 재분배를 같은 빈 `chi_coherent·J`로 되돌리는 것이 직접 수리 표적입니다.
- CMFGEN 등가 `eps`와 배율은 `ETAL/CHIL` depth-frequency dump 부재로 `UNRESOLVED` 처리했습니다.

재현 자료:

- [분석 스크립트](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e8_recycling.py)
- [요약 JSON](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e8/summary.json)
- [대역·셸 결과](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e8/band_shell_recycling.csv)
- [MC 열적 파괴](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e8/macro_thermal_destruction_iter10.csv)
- [형광 분기 통계](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e8/event_fluorescence_branch_iter11.csv)

신규 런·수송 solve·신규 clamp·커밋은 수행하지 않았습니다. CSV 행 수, JSON 유한값, 50개 셸 census 항등 및 이벤트 분기 보존도 검증했습니다.