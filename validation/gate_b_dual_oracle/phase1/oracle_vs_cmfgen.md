# Gate-B Lane C 비교 (REPORT-ONLY)

이 문서는 임계값·PASS/FAIL 판정을 포함하지 않는다.

## 속도 대응

| Lumina 셸 | v_L [km/s] | CMFGEN depth(1-based) | v_C [km/s] | Δv | 범위 내 |
|---:|---:|---:|---:|---:|:---:|
| s0 | 4264.000 | 67 | 4394.182 | +130.182 | True |
| s8 | 10088.000 | 54 | 10163.506 | +75.506 | True |
| s45 | 37024.000 | 1 | 35975.288 | -1048.712 | False |

s45는 CMFGEN RVTJ 최고 속도 밖에 있으므로 가장 가까운 외곽 depth를 대응시켰으며 외삽하지 않았다.

## 수치가 있는 비교행

| 셸 | 분류 | 수량 | Z:stage | Lumina | CMFGEN | L/C | 앵커 |
|---:|---|---|---:|---:|---:|---:|---|
| s0 | state | T_e | 0:-1 | 2.120268337e+04 | 1.853588700e+04 | 1.143872067e+00 | RVTJ:Temperature (10^4K)*1e4 |
| s0 | state | n_e | 0:-1 | 4.621814000e+09 | 4.852872100e+09 | 9.523873502e-01 | RVTJ:Electron density |
| s0 | bf | Gamma_photoion_total | 14:1 | 0.000000000e+00 | 2.079745202e-02 | 0.000000000e+00 | Sk2PRRR:sum(PR)/Ion Density |
| s0 | bf | Gamma_photoion_total | 16:1 | 0.000000000e+00 | 1.274938583e-02 | 0.000000000e+00 | S2PRRR:sum(PR)/Ion Density |
| s0 | bf | Gamma_photoion_total | 26:1 | 8.330218723e+05 | 7.624200139e-03 | 1.092602315e+08 | Fe2PRRR:sum(PR)/Ion Density |
| s0 | bb | jbar_representative | 26:1 | 1.501866504e-03 | 9.379098742e-04 | 1.601290854e+00 | EDDFACTOR:record_frequency=1.258150482e+15Hz |
| s0 | bb | jbar_representative | 26:2 | 1.793868012e-03 | 8.027568538e-04 | 2.234634315e+00 | EDDFACTOR:record_frequency=1.581590788e+15Hz |
| s0 | bb | jbar_representative | 27:2 | 3.226170349e-04 | 9.231811683e-05 | 3.494623222e+00 | EDDFACTOR:record_frequency=3.192526570e+15Hz |
| s8 | state | T_e | 0:-1 | 1.200772780e+04 | 1.032347600e+04 | 1.163147742e+00 | RVTJ:Temperature (10^4K)*1e4 |
| s8 | state | n_e | 0:-1 | 7.496125000e+08 | 7.341043400e+08 | 1.021125280e+00 | RVTJ:Electron density |
| s8 | bf | Gamma_photoion_total | 14:1 | 4.993938812e+01 | 1.968552105e-03 | 2.536858841e+04 | Sk2PRRR:sum(PR)/Ion Density |
| s8 | bb | jbar_representative | 14:1 | 2.034428000e-05 | 1.459557936e-05 | 1.393865875e+00 | EDDFACTOR:record_frequency=1.955062180e+15Hz |
| s8 | bb | jbar_input_raw | 14:1 | 2.034428000e-05 | 1.459557936e-05 | 1.393865875e+00 | EDDFACTOR:record_frequency=1.955062180e+15Hz |
| s8 | bb | jbar_representative | 14:2 | 7.490040000e-05 | 7.959250361e-05 | 9.410484229e-01 | EDDFACTOR:record_frequency=1.179078297e+15Hz |
| s8 | bb | jbar_input_raw | 14:2 | 7.490040000e-05 | 7.959250361e-05 | 9.410484229e-01 | EDDFACTOR:record_frequency=1.179078297e+15Hz |
| s8 | bf | Gamma_photoion_total | 16:1 | 1.725658118e+00 | 2.002659369e-03 | 8.616832923e+02 | S2PRRR:sum(PR)/Ion Density |
| s8 | bb | jbar_representative | 16:1 | 2.193058971e-06 | 1.092832775e-07 | 2.006765372e+01 | EDDFACTOR:record_frequency=3.305787835e+15Hz |
| s8 | bb | jbar_representative | 16:2 | 1.077846238e-05 | 1.790743976e-06 | 6.018985698e+00 | EDDFACTOR:record_frequency=2.496300635e+15Hz |
| s8 | bf | Gamma_photoion_total | 26:1 | 9.398721146e+02 | 1.141949575e-03 | 8.230416958e+05 | Fe2PRRR:sum(PR)/Ion Density |
| s8 | bb | jbar_representative | 26:1 | 6.640636433e-05 | 6.789423149e-05 | 9.780855145e-01 | EDDFACTOR:record_frequency=1.258150482e+15Hz |
| s8 | bb | jbar_representative | 26:2 | 4.033624221e-05 | 3.084237016e-05 | 1.307819146e+00 | EDDFACTOR:record_frequency=1.581590788e+15Hz |
| s8 | bb | jbar_representative | 27:2 | 3.326616389e-05 | 2.209465584e-05 | 1.505620369e+00 | EDDFACTOR:record_frequency=1.702989105e+15Hz |
| s45 | state | T_e | 0:-1 | 1.189706591e+04 | 1.649763900e+04 | 7.211374856e-01 | RVTJ:Temperature (10^4K)*1e4 |
| s45 | state | n_e | 0:-1 | 1.017327000e+05 | 2.004388500e+05 | 5.075498088e-01 | RVTJ:Electron density |
| s45 | bf | Gamma_photoion_total | 14:1 | 4.094554828e-01 | 6.651445554e-07 | 6.155887159e+05 | Sk2PRRR:sum(PR)/Ion Density |
| s45 | bb | jbar_representative | 14:1 | 6.524693000e-07 | 4.964944612e-06 | 1.314152223e-01 | EDDFACTOR:record_frequency=1.955062180e+15Hz |
| s45 | bb | jbar_input_raw | 14:1 | 6.524693000e-07 | 4.964944612e-06 | 1.314152223e-01 | EDDFACTOR:record_frequency=1.955062180e+15Hz |
| s45 | bb | jbar_representative | 14:2 | 1.034507000e-07 | 2.987722140e-06 | 3.462527475e-02 | EDDFACTOR:record_frequency=2.484850976e+15Hz |
| s45 | bb | jbar_input_raw | 14:2 | 1.034507000e-07 | 2.987722140e-06 | 3.462527475e-02 | EDDFACTOR:record_frequency=2.484850976e+15Hz |
| s45 | bf | Gamma_photoion_total | 16:1 | 1.317759645e-03 | 6.446397152e-07 | 2.044180050e+03 | S2PRRR:sum(PR)/Ion Density |
| s45 | bb | jbar_representative | 16:1 | 1.697185642e-07 | 1.515018823e-06 | 1.120240631e-01 | EDDFACTOR:record_frequency=2.380247636e+15Hz |
| s45 | bb | jbar_representative | 16:2 | 1.425776769e-07 | 2.786116394e-06 | 5.117434333e-02 | EDDFACTOR:record_frequency=2.496300635e+15Hz |
| s45 | bf | Gamma_photoion_total | 26:1 | 0.000000000e+00 | 3.357747661e-07 | 0.000000000e+00 | Fe2PRRR:sum(PR)/Ion Density |

## 완결성 census

- `compared`: 33
- `lumina_unavailable`: 162
- `unavailable`: 289

모든 미비교 행과 사유는 `oracle_vs_cmfgen.csv`에 보존했다.

## 파서·단위 근거

- RVTJ: 기존 `cmp_rvtj_T_ne_vs_published.py`의 ND/블록 리더 재사용; Temperature 헤더의 `10^4K`만 명시적으로 ×1e4.
- EDDFACTOR: 기존 `gamma_coiii_alllevel.read_eddfactor` 재사용; `EDDFACTOR_INFO`의 RECL/WORD/endian과 완료 레코드를 검사.
- PRRR Γ: 기존 검증식 `PR=n_SL R_SL [cm^-3 s^-1]`, `sum(PR)/Ion Density`만 수치 비교에 사용. α 블록은 라벨은 있지만 인라인 단위와 검증 파서가 없어 원값만 읽고 비교하지 않음.
- GENCOOL free-free는 해당 라벨에 인라인 단위가 없고 검증된 파서가 없어 비교하지 않았다. 단위를 추측하지 않았다.
- 원시 RVTJ 행 대조표: `cmfgen_parser_roundtrip.csv`.
