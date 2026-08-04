완료했습니다.

- 그림: [artis_ts1948_vs_cmfgen_spectrum.png](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/artis_ts1948_vs_cmfgen_spectrum.png)
- 재현 스크립트: [plot_spec_compare_ts1948.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/plot_spec_compare_ts1948.py)
- 수치 CSV: [band_occupancy.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/artis_ts1948_vs_cmfgen_spectrum_band_occupancy.csv)

PNG는 3900×2760, 300 dpi입니다.

대역 점유율은 각 스펙트럼의 native-domain 총 광도를 분모로 했습니다.

| 스펙트럼 | 총 광도 [erg/s] | 912–2000 Å | 2000–2500 Å | 2500–5000 Å | 5000–10000 Å |
|---|---:|---:|---:|---:|---:|
| ARTIS ts27 | 1.051909e39 | 43.6849% | 0.0000% | 56.3151% | 0.0000% |
| CMFGEN OBSFLUX | 9.825171e44 | 96.9408% | 0.1803% | 1.4198% | 0.4498% |
| LUMINA formal | 1.966714e44 | 5.2175% | 11.9794% | 71.7978% | 8.1057% |

핵심 단서:

- ARTIS는 `t_arrive=escape_time−pos·dir/c`, ts27 `[18.6903,20.3031)` d, `e_rf/8/Δt/Δλ`를 적용했습니다.
- 지정 스냅숏이 ts27 종료에서 끝나므로 관측자 도착창이 완전히 채워지지 않았습니다. 실제 선별량은 8 packets뿐이며, 이 때문에 ARTIS 절대광도와 shape가 통계적으로 대표성이 없습니다.
- 패킷 계산 광도 `1.051909e39`는 `light_curve.out`의 274,938 L☉=`1.051913e39 erg/s`와 일치합니다.
- CMFGEN은 OBSFLUX의 `10^15 Hz`, `Jy at 1 kpc` 규약으로 직접 \(L_\lambda\) 변환했습니다. 공식 배포 문서·소스는 [CMFGEN 배포 페이지](https://sites.pitt.edu/~hillier/web/CMFGEN.htm_old)에서 확인할 수 있습니다.
- CMFGEN 앵커는 [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/RVTJ:11)의 fixed-T 상태이며, 최종 반복 변화가 3.46×10³%라 조건부·미수렴 스냅숏으로 표시했습니다.
- LUMINA CSV는 `erg/s/cm`에서 `erg/s/Å`로 `×1e−8` 변환했습니다.
- git 명령은 사용하지 않았습니다.