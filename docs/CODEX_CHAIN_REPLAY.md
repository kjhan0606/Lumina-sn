# 캠페인 「확정 인과 사슬」 parity59 캡처 재생

작성일: 2026-08-03  
현행 입력: `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/`  
역사 기준선: `logs/coevolve_consume_a10_kx_gphall/` (2026-07-15)  
CMFGEN 기준: `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/`

## 최종 판정

**07-15의 확정 인과 사슬은 현행 parity59 캡처에서 성립하지 않는다.** 종점만 바뀐 것이
아니다. s0의 EUV/FUV 결손 자체가 초과로 역전됐고, 1500 Å 집중은 지배적 pile에서 약한
국소 봉우리로 축소됐으며, 냉각 root는 CMFGEN보다 뜨거운 root로 역전됐다. 따라서
`EUV/FUV 기근 → 복사 가열 고사 → 냉각 root`는 현행 상태의 설명이 될 수 없다.

다만 `광학 trapping 부족`이라는 대안의 **반증은 유지**된다. 현행 s0는 CMFGEN보다
bolometric `mc_J` 에너지밀도가 2.52배이고 전자산란 광학깊이도 CMFGEN보다 작지 않다.
정확한 현행 `tau_FUV`는 필수 파일 부재 때문에 `UNRESOLVED`이므로, 과거의 `tau_FUV≈70`
자체를 현행 사실로 승계하지는 않는다.

| 사슬 고리 | 07-15 주장 | 캡처 188932 | 판정 |
|---|---|---|---|
| s0 RADEQ 상태가 실제 root인가 | HOLD/pin 가능성을 배제 못함 | 12회 s0 모두 `root-found`; 전체 600/600 `root-found`, pin 0 | **변화** |
| 심부 온도 종점 | 13,119.875 K, 진리보다 5,640.125 K 낮음 | 21,227.639 K, 진리보다 2,467.639 K 높음 | **역전** |
| 심부 bolometric bath | `u_mc/u_CMFGEN=0.576` | `2.518` | **역전** |
| EUV/FUV 기근 | xuv 0.014×, FUV 0.023× CMFGEN | xuv 5.132×, FUV 2.526× CMFGEN | **역전** |
| 1508 Å 한-bin funnel | s0 에너지의 41.6% (~42%) | 15.9%; s2/s4 최대 bin은 2244/2924 Å로 이동 | **변화** |
| 1526 Å MC/deterministic 분리 | `mc_J/cs_J=39.04` | `3.93` | **변화** |
| trapping 부족 반증 | `u` 결손은 작고 `tau_es`, `tau_FUV`가 큼 | `u`는 오히려 초과, `tau_es`도 CMFGEN 이상 | **유지** (현행 `tau_FUV` 제외) |
| CMFGEN J 주입 coupled root | 18,277.377 K | 역사 추정기를 parity59 state에 적용하면 18,385.799 K | **RESOLVED** (production solver와 구분) |
| Co IV가 심부 방출의 83%, 평균 1553 Å | 8 GB 이벤트 전량 통계 | 이번 경량 재생에서 전량 통과 금지 | **UNRESOLVED** |

## 출처와 계산 정의

아래 표의 수치는 다음 파일/필드에 직접 연결된다.

| ID | 파일 | 필드/정의 |
|---|---|---|
| O-FIELD | `logs/coevolve_consume_a10_kx_gphall/lumina_coevolve_field.csv` | `shell, bin, wavelength_A, cs_J, mc_J` |
| C-FIELD | `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/lumina_coevolve_field.csv` | 같은 필드 |
| O-PLASMA | `logs/coevolve_consume_a10_kx_gphall/lumina_plasma_state.csv` | `shell_id, n_e, T_e` |
| C-PLASMA | `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/lumina_plasma_state.csv` | 같은 필드 |
| C-LOG | `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log` | `[SIMUL ...]`, `[TEHOLD] ... radeq_root` |
| C-ENV | 캡처의 `PARITY59_INSTR.env` | RADEQ 및 field gate. `docs/PARITY59_INSTR.env`와 SHA256 `3ee3a446...392594a`로 동일 |
| CMF-J | CMFGEN `EDDFACTOR`, `EDDFACTOR_INFO`, `RVTJ` | `J_nu`; shell velocity에서 주파수별 log-J 보간 |
| CMF-TAU | CMFGEN `MEANOPAC` | `Tau_Ross`, `Tau_es`; 4264 km/s 등에 보간 |
| GEO | `data/tardis_reference_toy06_19p48d/geometry.csv` | `r_inner`, `r_outer` |
| O-VERDICT | 역사 3개 분석 디렉터리의 `VERDICT.md` 및 CSV | 07-15 판정과 이벤트 유래 값; 수정하지 않음 |

정의는 다음과 같다.

- `u = (4 pi/c) integral J_nu dnu`. Lumina는 native 1000-bin `wavelength_A, mc_J`
  또는 `cs_J`; CMFGEN은 `EDDFACTOR:J_nu`를 사용했다.
- band `u`도 같은 적분을 표의 파장 구간에 제한했다.
- `tau_es = sum_from_shell_to_surface(n_e sigma_T delta_r)`,
  `sigma_T=6.6524587e-25 cm2`.
- 1508 Å 집중도는 역사 `taskA_sed_shape.py`와 같이 100–19933 Å를 40개 동일
  log-lambda bin으로 나누고, 각 bin을 독립 적분한 뒤 그 40개 합을 분모로 삼았다.
- 1526 Å 비는 목표 1526.17 Å에 가장 가까운 native bin 1526.087 Å의
  `mc_J/cs_J`다. 보간값이나 floor를 쓰지 않았다.

경량 사본으로 역사 입력도 다시 계산해 `u_mc/u_CMFGEN=0.576051`, 1508-bin
`0.416061`, 1526 Å `mc_J/cs_J=39.035893`을 재현했다. 즉 아래의 변화는 정의
변경이 아니라 입력 상태 변화다.

## 1. RADEQ ledger 재판정

### 관측 가능한 종점

| 양 | 07-15 | 캡처 188932 | 판정 | 출처 |
|---|---:|---:|---|---|
| s0 `T_e` | 13,119.874754 K | 21,227.639444 K | **역전** | O/C-PLASMA:`T_e` |
| `T_e - 18,760 K` | -5,640.125246 K | +2,467.639444 K | **역전** | 위 값과 O-VERDICT의 CMFGEN truth |
| s0 `n_e` | 4.426076e9 cm^-3 | 4.627433e9 cm^-3 | 변화 | O/C-PLASMA:`n_e` |
| s0 `u_mc` | 400.210756 erg cm^-3 | 1749.067904 erg cm^-3 | 변화 | O/C-FIELD:`wavelength_A,mc_J` |
| s0 `u_cs` | 463.656370 erg cm^-3 | 2675.602275 erg cm^-3 | 변화 | O/C-FIELD:`wavelength_A,cs_J` |
| `(u_mc/a)^0.25` | 역사 ledger 15,166 K | 21,927.514 K | 변화 | O-VERDICT; C-FIELD:`mc_J` |

캡처의 root 이력은 `11183→12988→13451→15170→16999→18197→19228→20071→20714→21081→21226→21228 K`다.
12개 `[SIMUL]` 요약은 모두 `pins hi=0 lo=0`, 600개 shell-record는 전부
`radeq_root=root-found`다(C-LOG). 따라서 07-15 감사의 “s0가 pin/HOLD일 수 있다”는
불확실성은 이 캡처에는 전파되지 않는다.

### coupled-root lever 표 (후속 가산성 감사로 갱신)

| 역사 lever | 07-15 값 | 캡처 값 | 판정/사유 |
|---|---:|---:|---|
| committed → own `cs.J` root | +3,496.683 K | +1,573.768 K (`22801.408 K`) | 변화; 초과를 악화 |
| own `cs.J` root → CMFGEN-J root | +1,660.820 K | **−4,415.609 K** (`18385.799 K`) | **역전** |
| CMFGEN-J root → truth | +482.623 K | +374.201 K | 변화 |
| endpoint 전체 | +5,640.125 K의 가열이 필요 | **-2,467.639 K의 냉각이 필요** | **역전** |

역사 문서의 표시값 `3400+1660+480=5540 K`는 제목의 5640 K와 100 K 차이가 난다.
root 원값 `13120→16617→18277→18760`을 쓰면 약 `3497+1660+483=5640 K`로 닫힌다.
이 보고서는 표시값과 root 원값을 모두 보존하고 차이를 숨기지 않았다.

후속 감사 `docs/CODEX_LEVER_PARITY59.md`는 발주된 07-19 analytic estimator 사본을 먼저
baseline gate한 뒤 parity59 captured state에 적용했다. 따라서 위 현행값의 정확한 이름은
“역사 추정기의 parity59-state root”다. 캡처 production solver는 C-ENV에서
`RADEQ_DB_FB=1`, `BF_RATE_POPS=1`, `FB_COOL_KT=1`, `TE_STEP_CLAMP=1`,
`ETLA_ALLOW_HEAT=1`이고 final CSV에는 trial-T별 `Gph/Hex/emit_bf/ETLA` 표가 없으므로,
18,385.799 K를 production solver의 정확 counterfactual root라고 재명명하지 않는다.

### 인과 해석

현행 s0는 450–918 Å에서 CMFGEN의 5.13배, 918–1290 Å에서 2.53배이고,
bolometric bath도 2.52배다(C-FIELD + CMF-J). 따라서 “EUV/FUV가 굶어서 가열이
죽었다”는 전제가 거짓이며 종점도 냉각 root가 아니다. **이 고리는 역전**이다.

다른 서사는 필요하지만, 현재 자료만으로 “FUV 초과가 직접 과열을 만들었다”고 확정할
수도 없다. photoheating은 threshold-weighted 적분이고 line exchange는 solver가 실제 소비한
field와 trial population에 의존한다. 현행 정확 항 ledger가 없으므로 허용되는 결론은
`field-starvation/cold-root 서사는 폐기`까지다. 과열의 세부 원인 배분은 `UNRESOLVED`다.

## 2. trapping audit 재판정

| 양 | 07-15 | 캡처 188932 | 판정 | 출처 |
|---|---:|---:|---|---|
| s0 `u_mc` | 400.211 | 1749.068 erg cm^-3 | 변화 | O/C-FIELD:`mc_J` |
| s0 `u_mc/u_CMFGEN` | 0.576051 | 2.517555 | **역전** | O/C-FIELD + CMF-J |
| s0 `tau_es` | 1.799943 | 1.553726 | 변화 | O/C-PLASMA:`n_e` + GEO |
| s0 `tau_es/tau_es,CMFGEN` | 1.181702 | 1.020055 | **유지** (`>=1`) | 위 값 + CMF-TAU:`Tau_es` |
| s0 `tau_FUV` | 69.8 | `UNRESOLVED` | **UNRESOLVED** | O-VERDICT; 캡처 `lumina_line.csv` 부재 |
| s0 Lumina Rosseland line+es depth | 5.83 | `UNRESOLVED` | **UNRESOLVED** | 같은 누락 branch |

현행 `tau_es`는 CMFGEN보다 겨우 2.0% 크므로 과거의 “모든 척도에서 1.2–1.4배 더
불투명”이라는 정량 문구는 유지되지 않는다. 그러나 energy density가 부족하지 않고
전자산란 바닥도 부족하지 않으므로 “photons escape too easily → deep energy shortage”는
여전히 반증된다. 즉 **반증 결론은 유지, 과거의 tau_FUV≈70 근거는 미승계**다.

캡처 stdout은 현재 line list `2,584,132`개를 로드했다고 기록한다(C-LOG `Lines:` 및
`[RADEQ] collisional line-cooling table`). super-cutoff가 transport line list를 제거한다는
과거 대안도 다시 지지되지 않는다. 다만 line별 현행 expansion opacity 수치는 만들지 않았다.

## 3. reddening localization 재판정

### s0 band 대조

아래 `mc/CMF`는 band-integrated `u_mc/u_CMFGEN`, `mc/cs`는 같은 band의
`integral mc_J dnu / integral cs_J dnu`다. 모든 값의 원천은 O/C-FIELD의
`wavelength_A,mc_J,cs_J`와 CMF-J의 `J_nu`다.

| band (Å) | 07-15 mc/CMF | 캡처 mc/CMF | 07-15 mc/cs | 캡처 mc/cs | 고리 판정 |
|---|---:|---:|---:|---:|---|
| 100–300 | 9.03e-19 | 0.0462 | 4.06e-14 | 0.00123 | 변화; 에너지 비중 극소 |
| 300–450 | 0.000224 | **10.463** | 0.330 | 0.930 | **역전** |
| 450–918 | 0.0144 | **5.132** | 0.562 | 1.280 | **역전** |
| 918–1290 | 0.0230 | **2.526** | 0.277 | 0.650 | **역전** |
| 1290–2000 | 0.941 | **3.108** | 1.224 | 0.895 | 결손→초과 |
| 2000–3000 | 0.107 | **1.577** | 0.127 | 0.337 | **역전** |
| 3000–4500 | 0.604 | **1.875** | 1.358 | 0.616 | **역전** |
| 4500–7000 | 1.384 | 2.246 | 2.000 | 0.812 | CMF 초과 유지, MC/cs 관계 변화 |
| 7000–10000 | 5.758 | 19.498 | 1.200 | 2.741 | CMF 초과 유지 |
| 10000–19933 | 9.954 | 15.924 | 1.503 | 2.145 | CMF 초과 유지 |

과거에는 s0 에너지의 51.4%가 1290–2000 Å였지만 캡처는 38.8%다. 더 중요한 차이는
다른 UV/optical band도 함께 밝아져, narrow pile이 “굶은 양옆 사이의 유일한 저장소”가
아니라는 점이다. band fraction 출처는 각 replay의 `taskA_band_table.csv:frac_lumina`다.

### 1508/1526 Å 및 방출 평균

| 양 | 07-15 | 캡처 | 판정 | 출처 |
|---|---:|---:|---|---|
| s0 최대 log-bin 중심 | 1508.45 Å | 1508.45 Å | 위치 유지 | replay `taskA_logbin_concentration.csv` |
| 그 bin의 s0 `u` 비중 | 0.4161 (~42%) | 0.1586 | **변화** | 같은 파일:`lumina_u_fraction` |
| 같은 bin CMFGEN 비중 | 0.0920 | 0.0920 | 기준 고정 | 같은 파일:`cmfgen_u_fraction_same_bin` |
| s2/s4 최대 bin 중심 | 1508/1508 Å | 2244/2924 Å | **변화** | 같은 파일:`peak_logbin_mid_A` |
| s0 1526.087 Å `mc_J/cs_J` | 39.0359 | 3.93277 | **변화** | replay `taskA_mc_cs_1526.csv` |
| s0-2 방출 에너지 가중 평균 lambda | 1553 Å | `UNRESOLVED` | **UNRESOLVED** | O-VERDICT; C-EVENT 미통과 |

따라서 “1500 Å Co IV funnel이 심부 bath를 지배한다”는 고리는 유지되지 않는다.
s0에는 1508 Å 봉우리가 남았지만 지배율과 MC/deterministic 분리가 크게 줄었고, 더 바깥
심부 shell에서는 최대점 자체가 장파장으로 이동했다. Co IV 방출 비중, line forest의 net
band flow, 평균 1553 Å는 event-derived이므로 현행에서는 판정하지 않았다.

## UNRESOLVED 목록

1. **현행 `tau_FUV`, Rosseland line+es depth.** 요청에서 필수로 지정한
   `/gpfs/.../instr_capture_188932/lumina_line.csv`가 없다. 다른 line list나 level-pop 조합으로
   값을 지어내지 않았다. 참고로 현재 workspace의 역사 `audit_t_expop.py`는 실제로
   `data/.../line_list.csv`와 `lumina_levelpop.csv`를 읽어 요청 설명과 불일치한다. 이번 작업은
   요청 규율을 우선해 그 경로를 대체 계산으로 사용하지 않았다.
2. **현행 이벤트 forensic.** `lumina_events.bin`은 8,000,000,032 bytes(400M record)여서
   전량 통과 금지에 따라 실행하지 않았다. 따라서 Co IV 83/84%, pile emission 96.3%,
   심부 평균 1553 Å, upconversion, band net-flow의 현행값은 모두 `UNRESOLVED`다.
3. **production solver 자체의 counterfactual ladder.** 발주된 역사 estimator의 parity59
   대응값과 R/J/O 가산성은 후속 감사에서 `RESOLVED`됐다. 다만 현행 solver는 역사 offline
   estimator와 식이 다르고 final CSV에는 trial-T 내부항이 없다. 따라서 production solver
   자체의 정확 ladder는 여전히 분리된 `UNRESOLVED`다.

## 운전석의 계산 노드 잔여 작업

### 즉시 실행 가능한 8 GB 이벤트 재생

아래 스크립트는 5M-record chunk로 한 번 전량 통과하며 원본 event 정의를 보존한다.

```bash
python3 validation/chain_replay_parity59/reddening_localization/taskB_event_forensics_compute_node.py \
  --input-dir /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932 \
  --output-dir validation/chain_replay_parity59/reddening_localization/results_event
```

출력: `taskB_band_ledger.csv`, `taskB_upconversion.csv`, `taskB_emission_color.csv`,
`taskB_top_ions.csv`, `taskB_coverage.csv`. 실행 뒤 본 보고서의 event-derived 고리만 갱신해야 한다.

### 현 캡처만으로 즉시 닫을 수 없는 작업

- `tau_FUV`: `lumina_line.csv` producer/schema가 복구되어야 한다. 이번 캡처에 없는 값을
  계산 노드 성능으로 해결할 수는 없다.
- production-solver coupled-root levers가 추가로 필요하면 현행 `simul_r1`을 그대로 mirror하는
  offline evaluator와 trial-T별 `Gph/Hex/emit_bf/ETLA` 재구성이 먼저 필요하다. 역사 estimator
  lever 및 가산성 결과는 `docs/CODEX_LEVER_PARITY59.md`에서 이미 닫혔다.

## 재현 산출물

- `validation/chain_replay_parity59/comparison_summary.csv`: 핵심 07-15/캡처 대조.
- `validation/chain_replay_parity59/trapping_audit/`: parameterized replay와 old/new CSV.
- `validation/chain_replay_parity59/radeq_ledger_audit/`: 관측값, gate snapshot, root history,
  정확히 닫히지 않는 lever 목록.
- `validation/chain_replay_parity59/reddening_localization/`: Task-A parameterized replay,
  old/new band 및 concentration CSV, compute-node event script.

원본 3개 `VERDICT.md`, `src/`, 모델/캡처 파일은 수정하지 않았다. 새 모델 런, GPU 작업,
commit, 8/49 GB 전량 통과도 수행하지 않았다.
