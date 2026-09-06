# 판독 — DET-DUAL-AUTH-2 arm (Slurm 332463) 대 base 331050 (2026-09-06, 판정 아님)

저자 = 운전석(Fable). **판독이지 판정이 아니다** — 분기 라벨은 Fable fresh 판정이 붙인다. 계약 = `docs/PREREG_DUAL_AUTHORITY_2026-09-05.md`(개정 2). 산출 = `/gpfs/kjhan/lumina/dproxy_audits/readout_331050_fable/out_arm/`. 런루트 무변경.

## 0. 착지
Slurm 332463 `FAILED 70:0` 02:58:47 @ syn102 — **model.rc = 0**, publication 2, `work/lumina_spectrum.csv` 1,235행, physics_comparison iter 0·1 존재. FAILED 는 예상된 post-model seal 오판(`unsealed complete marker at stderr line 2202`, base 와 동일 계급). 가드·치명 표지 0. `stderr.log` sha256 `0eb04f529d3a4531…`. arm 런루트 = `/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260905T090712Z_dproxy_provenance_k42_ion1_rescale`.

## 1. 게이트 (사실만)
| 게이트 | 결과 | 값 |
|---|---|---|
| G-0 env 단일 차이 | ✔ | sorted diff = `RESCALE "0"→"1"` 한 쌍 + 런루트 경로 치환 행 |
| G-1 봉인 입력 | ✔ | `resolved_lumina.sha256` cmp 동일, registry `c96f74f0…`, binary `a4990a59…` |
| G-2 gen-1 bit-동일 | ✔ | base·arm 모두 `iterations=45 residual=9.6662782724980344e-09` |
| G-3 gen-2 상이 | ✔ | base `7.8976358675842993e-09` / arm `7.8947378821034078e-09` (gen-3: 8.5619e-09 / 8.5697e-09) |
| G-4 완주·338 짝 | **337/338** | registry 선 `1139990`(shell 0) 이 arm publication 1 에서 상위 90% 선택 밖으로 빠짐(publication 0 에는 있음, log line 1970). 나머지 337 짝 유효, provenance OK, D2 산포 1.3e-15, D3 T_e 동일 |
| G-5 역산 정합 | ✔ | 같은 하준위 공유 선 간 산포 최대 1.26e-9 ≤ 1e-8 |

## 2. 계측
### E-1 (작동 확인) — 바닥 준위 Δlog n(iter0→1), arm
| | Fe II | Co II | Ni II |
|---|---|---|---|
| base | +3.62% | +0.49% | +4.09% |
| **arm** | **−0.056%** | **−0.023%** | **+0.321%** |
공통모드는 핀 아래에서 사라졌다(구성상). Ni II 만 (0.3%, 1%] 구간.

### 이온 내 재배분 (arm, iter0→1)
| 블록 | base Y 평균 | arm Y 평균 ± sd | arm 럼프 |
|---|---|---|---|
| Fe (26 SL) | +3.3% | **−0.4% ± 1.2%** [−6.1%, +0.2%] | SL 465 **−6.05%** |
| Co (40) | +0.7% | +0.2% ± 0.4% | SL 869 −0.9% |
| Ni (32) | +4.3% | +0.5% ± 1.1% [−0.7%, +3.6%] | — |
`rho_SL`(arm) = +0.0076. 상준위 중앙 Δlog n_upper: Fe **−6.05%**, Co +0.08%, Ni −0.05%.

### E-4 (교차런 비, iter 0) — `n(arm)/n(base)` = `n_II^Saha / n_II^NLTE` (shell 0)
| | Fe II | Co II | Ni II |
|---|---|---|---|
| iter 0 바닥 | **0.8836** | **1.0231** | **0.9549** |
| iter 0 하준위 40/29/15 중앙(산포) | 0.8838 (6e-3) | 1.0231 (3e-4) | 0.9549 (2e-4) |
| iter 0 상준위 90/60/32 중앙(산포) | 0.8811 (3e-3) | 1.0231 (4e-4) | 0.9550 (2e-4) |
| iter 1 바닥 | 0.8518 | 1.0178 | 0.9196 |
⟹ 준위 전체가 이온별 **단일 비**로 스케일(시드 solve 행렬해 동일, post-solve 스케일만 다름 — 계약 E-4 의 전제 그대로). NLTE 분배는 Saha 보다 **Fe II 13%(iter 1: 17%)·Ni II 5%(9%) 많고 Co II 2% 적다.**

### E-3 (formal 스펙트럼, 1,234 bin, 격자 bit-동일) — `Δlog F = ln(F_arm/F_base)`
| 대역 (eV) | bin | 중앙 | 최대 | 최소 |
|---|---|---|---|---|
| [0,2) | 399 | −0.17% | +0.70% | −7.0% |
| [2,4) | 131 | +0.53% | +7.1% | −2.1% |
| **[4,6)** | 76 | **+1.37%** | +4.2% | −0.5% |
| [6,12) | 131 | +0.02% | +0.59% | −1.0% |
| [12,∞) | 497 | ~0 | +49% | −7.2% |
계약 §5 규칙 「어느 대역 \|Δlog F\| 중앙 ≥ 1%」 에 [4,6) eV 가 해당.

## 3. 재현
```
# grammar-debug, cd /gpfs/kjhan/lumina/dproxy_audits/readout_331050_fable
python3 -B readout_dproxy_sl_331050.py --run-root <ARM> --out-dir out_arm
python3 -B probe_lower_any.py <ARM>            # E-1, G-5
python3 -B cross_run_ratio.py <BASE> <ARM>     # E-4
python3 -B <repo>/scripts/dproxy_readout/spectrum_contrast.py <BASE> <ARM> > out_arm/spectrum_contrast.json
```
스크립트 저장소 사본 `scripts/dproxy_readout/`.

## 4. 분장 장부 (사전등록 §6 지목)
판독 = 운전석. 판정 = Fable fresh(발주). 감리 = 미발주.
