# SH-GRID 1178-bin 이행 판정 — 2026-08-08

## 판정

**SH-GRID 구조·원자단면·동일-snapshot 저주파 물리 폐합은 PASS**다.
DET 수렴 flight를 준비할 수 있다. MC-EVT는 DET 수렴과 비교 dump 판정 뒤에 재개한다.

## 정본 격자

| 항목 | 값 |
|---|---:|
| BF/NLTE bins | 1178 |
| `nu_min` | `5.8412785919616062e13 Hz` |
| `nu_max` | `3.0e16 Hz` |
| dlog | `0.0052983173665480362` (구 1000-bin dlog와 bit 동일) |
| canonical bins | 3866 |
| canonical offsets | `[-1398,2468]` |
| BF→canonical refinement | K=2 |
| canonical edge SHA-256 | `8388614bbfcaf2f01d101216301ef12c680f15d09ad50fc05b521429e4b75def` |

`selftest_grid_roundtrip` 결과는 `max_abs=0`, `max_rel=0`이다. 구 canonical 절대
범위를 유지하면서 BF anchor가 178칸 아래로 이동했다.

## CMFGEN sigma 재생성

활성 덱 `data/tardis_reference_toy06_19p48d_sivcaiv_active` 의 봉인된
`active_ions.csv`, `levels.csv`, `DECK_PROVENANCE.json`, `atomic_links.txt` 해시를 먼저
대조했다. 그런 뒤 `expand_atomic_data_cmfgen.py` 의 CMFGEN photo evaluator로 전
24,542 level×1178 bin을 새로 평가했다.

- 최초 구조 이행 자산: 231,308,384 bytes, SHA-256
  `4772cdad1ad75f6a409e1e38732b8b94f1ba741921f716330a2c02e37089847e`.
- 구 1000-bin 행 padding: **0**. first-bin fill: **0**.
- `has_cmfgen[24542]`: 구 자산과 byte 동일.
- 전 격자: finite, nonnegative.
- 구 자산은
  `quarantine/sh_grid_migration_2026-08-08/cmfgen_sigma_bf.pre_sh_grid_1000.bin`에
  복구 가능하게 보존했다.
- loader는 bin 수만이 아니라 양 끝점, flag, padding, 정확한 file extent를 검사하며
  stale/corrupt 자산을 Kramers로 조용히 바꾸지 않고 중단한다.

동일-snapshot 물리 비교에서 최초 자산이 CMFGEN type 2/3/8/9에 역사적
`params[0]-as-sigma0` stand-in을 사용했음이 드러났다. grid geometry나 구 1000-bin
보존 오류가 아니라 기존 evaluator 의미론의 문제였다. rebaker를
`CMFGEN_EXACT_HYD=1` fail-closed로 바꾸고 CMFGEN `SUB_PHOT_GEN` 동형 evaluator로
7,266행을 재평가했다.

- 현재 canonical SHA-256:
  `c65d6fcf12c61952855599b317562f45a1b3cc816ce66cc801f2a7a3f14d2675`.
- 이전 stand-in 1178-bin 자산은
  `quarantine/exacthyd_promotion_2026-08-08/cmfgen_sigma_bf.pre_exacthyd_standin_1178.bin`
  에 복구 가능하게 보존했다.
- promotion은 same-filesystem staging·fsync·SHA 검증 후 원자 교체하며, 교차
  파일시스템 실패가 남긴 유일한 안전 중간상태도 fail-closed로 재개한다.
- active root 55개 봉인 파일의 size/SHA 검증 PASS.

## 저주파 edge 장부

새 하한 아래 default-active 양의 threshold는 **0개**다. 종래 하한 아래였던
707개는 모두 정상 domain에 들어왔다. 현재 exact-Hydrogen 자산은 raw CMFGEN phot
data 재구성과 최대 상대오차 `1.054706e-10`으로 일치한다.

- 707행 중 706행은 신규 178 bins에서 양의 `sigma*dnu`.
- 1행 `(Z=16, ion=1, level=304)`은 CMFGEN evaluator 자체가 해당 band에서 exact zero.
동일한 `EDDFACTOR/RVTJ/POP*` snapshot에서 707 level×90 depth를 raw CMFGEN-native
적분과 production bin-average 소비로 독립 평가했다.

- depth 합계 photo-rate max rel: `2.858299e-4`.
- depth 합계 spontaneous Milne eta max rel: `2.854924e-5`.
- significant ion-depth photo/eta max rel: `1.119969e-3` / `4.699186e-5`.
- level-weighted L1 photo/eta max rel: `7.084012e-4` / `4.547908e-5`.
- finite/nonnegative, POP roundtrip, EDD complete, 707-level coverage: 전부 PASS.

단일 level-cell 최대 상대오차는 임계면의 매우 얇은 한 빈에서 약 1.04이나, 이는
서로 따로 평균된 sigma와 J만으로 sub-bin 공분산을 복원할 수 없는 진단치다. 합계,
ion 합, 기여도 가중 L1을 수용 게이트로 사용했으며 모두 사전 한계 안에 닫혔다.

## 소비자·회귀

- GPU injection CDF의 1024-bin 정적 배열과 clamp를 `NLTE_N_FREQ_BINS` 권위로 교체.
- full-bin average sigma에 physical threshold를 다시 적용하던 이중 절단을 CPU canonical
  rate, NLTE GEMM, Milne/recombination, opacity/emissivity, element-wide 적분 경로에서 제거.
  실제 packet frequency를 소비하는 event/transport의 sharp-edge 검사는 유지.
- A2-05의 centre-below-threshold partial-bin 회귀시험 PASS.
- radiation field, BF event measure, opacity/emissivity, GPU BF upload은 모두 새 상수로
  CPU/OpenMP/CUDA sm_80·sm_86·sm_90 compile+link PASS. GPU/model binary는 login node에서
  실행하지 않았다.
- `selftest_sh_grid_loader`: stale range·trailing byte 거부 PASS.
- 현재 active 덱 전체 battery: D 19/19, K 7/7, Z 12/12, CP 4/4 PASS.
- active-only quarantine 정책과 tau 기준선 `active_lines=2588798`,
  `FNV64=6c53c2f89ad53e47`, bit difference 0을 C/Python 양쪽에 fail-closed 등록.
- active 덱 quarantine: Q1/Q2/Q3/Q5 PASS. Q4의 종래 I20 직렬화 대 CMFGEN
  원값 불일치 2,588,793건은 그대로이며 SH-GRID와 독립이다.

## 다음 차단선

동일-snapshot 저주파 폐합이 끝났으므로 다음 차단선은 **계산 노드 DET 수렴**이다.
active deck, exact-Hyd canonical SHA, binary SHA, 필수 comparison dump와 fail-closed
환경을 batch manifest에 고정한다. DET가 수렴한 뒤 shell별 `T_e`, `n_e`, 이온분율,
`u_atom`, 단열 네 항, `chi/eta`, `H-C`를 CMFGEN과 비교한다. 그 판정 전 MC-EVT는
재개하지 않는다.
