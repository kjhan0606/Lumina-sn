# Wave 3 판별 측정 — Γ 삼중 대조

오프라인 재생만 수행했으며 신규 transport/NLTE 런과 `src/` 수정은 없었다.

## 결론

- **Fe III C48 lump**: **동결장 내용이 진범 (구조·산술 무죄)** — A≈B and C/B collapses by 1.1920 dex.
- **S II SL4**: **동결장 내용이 진범 (구조·산술 무죄)** — A≈B and C/B collapses by 1.8269 dex.

## A/B/C 수치

C는 field falsifier가 되도록 frozen bfr까지 포함한 **전체 장**을 CMFGEN Jν로 치환했다. 즉 모든 유효 bin에서 `pref*J_CMFGEN`을 사용하며, frozen positive/fallback mask는 분해 표시에만 쓴다. positive-bfr을 그대로 두고 fallback J만 바꾸는 literal-branch hybrid는 판독에 쓰지 않고 아래에 감도값으로 병기한다.

| 준위 | A [s⁻¹] | B [s⁻¹] | C [s⁻¹] | log10(A/B) dex | log10(C/B) dex | 판독 |
|---|---:|---:|---:|---:|---:|---|
| Fe III C48 lump | 4.367637485e+02 | 4.367637485e+02 | 2.807174861e+01 | -0.000000 | -1.191977 | 동결장 내용이 진범 (구조·산술 무죄) |
| S II SL4 | 3.187513635e+01 | 3.187513635e+01 | 4.748076950e-01 | +0.000000 | -1.826934 | 동결장 내용이 진범 (구조·산술 무죄) |

## B positive-bfr / fallback 분리

소비 횟수는 코드와 같이 threshold 통과·σ>0 bin-route 평가 횟수다. oracle 전역 s8 비율은 2,762,679/(5,159,471+2,762,679) = **34.873%** (요청의 34.9% 반올림값과 일치)다. 준위별 비율은 문턱/σ support가 달라 전역값과 같을 필요가 없다.

| 준위 | B positive [s⁻¹] | B fallback [s⁻¹] | fallback rate share | positive eval | fallback eval | fallback eval share |
|---|---:|---:|---:|---:|---:|---:|
| Fe III C48 lump | 4.367625804e+02 | 1.168103636e-03 | 0.0003% | 469,931 | 382,200 | 44.8523% |
| S II SL4 | 3.187407471e+01 | 1.061642689e-03 | 0.0033% | 69 | 273 | 79.8246% |

## C 장 치환 세부

Lumina geometry의 s8 midpoint는 **10088.0 km s⁻¹**다. jnu4 RVTJ의 9610.017–10163.506 km s⁻¹ 두 depth 사이에서 주파수별 log(Jν)를 보간(w=0.863582)하고, 1000개 Lumina log-bin에 적분 평균했다. `Σ Jbar Δν / ∫Jνdν = 1.000000000000000`이다.

| 준위 | C on positive-mask [s⁻¹] | C on fallback-mask [s⁻¹] | fallback-only hybrid [s⁻¹] | log10(hybrid/B) dex | threshold range [eV] |
|---|---:|---:|---:|---:|---:|
| Fe III C48 lump | 2.807173929e+01 | 9.320042595e-06 | 4.367625897e+02 | -0.000001 | 1.967399–17.517820 |
| S II SL4 | 4.747989816e-01 | 8.713413949e-06 | 3.187408342e+01 | -0.000014 | 20.291398–20.291398 |

## 입력·identity·산술 검증

- 동기화: EW consumer iter 11, field producer iter 10, lag 1. 따라서 B는 C1/C2 `iter=10, shell=8`을 사용했다.
- 실제 σ 소스: `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/cmfgen_sigma_bf.bin` → `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen_sigma_bf_superlev_ionfix_ddc15strat_sivcaiv.bin`. EW stdout의 26087/26592 coverage와 production stdout의 `LUMINA_CMFGEN_SIGMA_BF`가 같은 모델 링크를 지목한다.
- 소스 계약: `lumina_element_wide.c` estimator L328, fallback L330, SL weight L370; `lumina_plasma.c` estimator L15632, fallback L15637.
- frozen grid `nu_mid` max relative mismatch 4.987e-07; C1 dump 반올림값으로 재구성한 coarse-integral max relative residual 5.708e-04.
- jnu4 schema: ND=90, good frequency records=196,185, ν=3.499e+12–1.000e+18 Hz, FINISH=1; FL↔Hz max relative round-trip 2.220e-16, ν↔Å 2.220e-16. CGS Jν 단위 독립 sanity `u_inner/(aT⁴)=0.944769`.
- Fe III C48 lump identity: matrix 201, sl_id 100, members 1400, Σ within-SL fraction=1.0000000000000016, direct Boltzmann max|Δf|=8.327e-17, route count=1400.
- S II SL4 identity: matrix 4, sl_id 4, members 1, Σ within-SL fraction=1.0000000000000000, direct Boltzmann max|Δf|=0.000e+00, route count=1.

### 사용 입력 실측

| 입력 | data rows/records | bytes | schema |
|---|---:|---:|---|
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_element_wide.c` | 1,323 | 65,928 | text |
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c` | 18,469 | 1,007,945 | text |
| `/tmp/w31_on_a.JuCpDY/lumina_oracle_cell_s8.csv` | 194 | 28,131 | OK |
| `/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59/lumina_c1_bins.csv` | 14,400 | 880,945 | OK |
| `/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59/lumina_c2_bfr_dump.csv` | 600,000 | 30,634,495 | OK |
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/levels.csv` | 26,592 | 825,334 | OK |
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/ionization_energies.csv` | 53 | 1,038 | OK |
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv` | 50 | 3,359 | OK |
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/cmfgen_sigma_bf.bin` | 26,592 | 212,762,624 | OK |
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/ma_radrecomb_target.bin` | 14,063 | 106,384 | OK |
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/trapping_audit/audit_u_cmfgen.py` | 114 | 5,107 | text |
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/trapping_audit/VERDICT.md` | 164 | 11,337 | text |
| `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ` | 4,698 | 604,183 | text |
| `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR_INFO` | 4 | 284 | text |
| `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR` | 196,185 | 142,832,872 | OK |
| `/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59/stdout.log` | 38,204 | 2,970,607 | text |
| `/tmp/w31_on_a.JuCpDY/stdout.txt` | 71 | 3,501 | text |
| `/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z26_s008_provenance.csv` | 16,477 | 2,848,712 | OK |
| `/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z26_s008_identity.csv` | 303 | 43,547 | OK |
| `/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z26_s008_manifest.csv` | 44 | 1,233 | OK |
| `/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z26_s008_solution.csv` | 4,701 | 371,216 | OK |
| `/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z16_s008_provenance.csv` | 15,337 | 2,654,228 | OK |
| `/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z16_s008_identity.csv` | 303 | 21,365 | OK |
| `/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z16_s008_manifest.csv` | 44 | 1,232 | OK |
| `/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z16_s008_solution.csv` | 1,201 | 93,885 | OK |

## 사전등록 판독표 적용

동등성 경계는 명시된 **0.1 dex**다. A/B가 이를 넘으면 첫 행, A/B가 통과하고 C/B가 −0.1 dex 미만이면 두 번째 행, A/B와 B/C가 모두 0.1 dex 이내면 세 번째 행을 적용했다. C가 오히려 +0.1 dex 넘게 증가하는 경우는 사전등록 표에 없으므로 UNRESOLVED 규칙이다.

## UNRESOLVED

- 없음. 요청한 두 준위 모두 σ row, target route, within-SL weight, 장 단위·격자가 확정됐다.

## 재현

`python3 scripts/w3_gamma_triple_compare.py --report docs/CODEX_W3_GAMMA_TRIPLE_COMPARE_2026-07-31.md`
