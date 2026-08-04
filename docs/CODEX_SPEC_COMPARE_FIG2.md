수리 및 재작도 완료했습니다.

- ARTIS 선별: `escape_time ∈ [18.6903, 20.3031) d`
- 선택 패킷: 52,549개 — 1,000개 이상이므로 진단 히스토그램 조건 미해당
- 파장: `λ_obs = c / nu_rf`
- ARTIS 광도: `9.531667e42 erg/s`
- CMFGEN 경고 범례와 프레임 차이 각주 반영
- Lumina 선 유지
- `git` 미사용

| Spectrum | 912–2000 Å | 2000–2500 Å | 2500–5000 Å | 5000–10000 Å |
|---|---:|---:|---:|---:|
| ARTIS | 9.1132% | 15.5109% | 47.5006% | 21.1035% |
| CMFGEN | 96.9408% | 0.1803% | 1.4198% | 0.4498% |
| Lumina | 5.2175% | 11.9794% | 71.7978% | 8.1057% |

산출물:

- [재작도 PNG](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/artis_ts1948_vs_cmfgen_spectrum.png)
- [대역 점유율 CSV](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/artis_ts1948_vs_cmfgen_spectrum_band_occupancy.csv)
- [수정 스크립트](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/plot_spec_compare_ts1948.py)