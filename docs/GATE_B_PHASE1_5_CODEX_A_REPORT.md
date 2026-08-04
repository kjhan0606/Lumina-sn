# Gate B dual oracle Phase 1.5 — Codex A 구현 보고서

## 1. 범위와 결론

`docs/CODEX_GATEB_C_REVIEW.md`의 FAIL 4건을 대상으로 Phase 1 oracle을
보강했다. 비교자는 Phase 1과 동일하게 REPORT-ONLY이며 임계값이나 gate
판정을 구현하지 않는다.

구현 결과:

1. CMFGEN `n_e`는 RVTJ 헤더, RVTJ writer의 `WRITE ... ED`, `ED(:)`의
   `#/cm^3` 소스 선언, 원문 physical line의 identity 왕복을 모두 산출했다.
2. 범위 밖 `s45`를 `s43`으로 교체했다. `s43`은 중심 속도 35568 km/s로
   CMFGEN 최고 속도 35975.288 km/s 안에 있으며 depth 10에 대응한다.
3. PRRR 전 이온의 Γ/α, `n_ion`, OUT `b_k`, GENCOOL bf/ff/coll/net, 생산
   thermal ledger를 추가했다. 동일 수량이 아닌 값은 context-only 또는
   unavailable로 남기고 사유를 기록했다.
4. 동결 run의 실제 C2 `bf_rate_estimator`, 실제로 기록된 raw Jbar, 외부
   deposition을 적재하고 생산 `nlte_assemble_rate_matrix`와
   `compute_radiative_equilibrium_te/radeq_simul_all/simul_r1` 소비 지점에서
   관측했다.

## 2. 구현

### 2.1 관측 전용 경로

- `LUMINA_FROZEN_ORACLE` 전용 probe만 추가했다.
- rate matrix와 opacity의 산술 결과는 기존 생산 함수 내부에서 읽는다.
- thermal은 `radeq_simul_all`이 shell 입력과 line/collision table을 모두
  구성한 뒤, 기록된 `T_e`에서 생산 `simul_r1`을 한 번 호출해 그 함수가
  이미 계산한 항목별 shadow를 읽는다.
- oracle build에서는 이 1회 관측 뒤 root solve와 plasma commit 전에
  빠져나오므로 동결 입력 상태를 변경하지 않는다.
- 정상 build에는 oracle 심볼과 분기가 전처리 단계에서 제거된다.

### 2.2 동결 생산 입력 재현

세 셀 모두 마지막 iteration 11을 사용했다.

| 입력 | 처리 |
|---|---|
| `lumina_plasma_state.csv` | 기록된 `W`, `T_rad`, `n_e`, `T_e` |
| `lumina_ion_pops.csv` | 기록된 ion density |
| `lumina_levelpop.csv` | 기록된 level population과 비-sentinel `b_k` |
| `lumina_c1_bins.csv` | 24-bin W/T를 생산 1000-bin `J_nu`로 복원 |
| `lumina_c2_bfr_dump.csv` | 1000/1000 bin의 실제 `bfr` 적재; bin 중심 주파수 검사 |
| `lumina_jbar_dump.csv` | 기록된 line Jbar/count/beta 적재 |
| `deposition_cmfgen.csv` | shell별 heating rate 적재 후 생산 nonthermal 등록 함수 호출 |

C2가 fallback으로 대체되지 않았음을 소비 지점에서 계수했다.

| 셀 | C2 bins loaded/expected | positive estimator 소비 | fallback 소비 |
|---:|---:|---:|---:|
| s0 | 1000/1000 | 5,968,990 | 1,953,160 |
| s8 | 1000/1000 | 5,119,009 | 2,803,141 |
| s43 | 1000/1000 | 4,344,072 | 3,578,078 |

fallback 소비는 C2 값이 0인 bin에서 생산 코드가 의도적으로 C1을 쓰는
분기이며, 하네스가 C2 배열 전체를 0으로 만든 이전 실행과 구별된다.

raw-Jbar 동결 writer는 설정
`LUMINA_JBAR_DUMP_IONS=14:1,14:2` 때문에 Si II/III만 기록했다. 따라서
Si는 raw Jbar 기반 생산 경로를 재현한다. S/Fe/Co의 실제 production-memory
Jbar는 파일에 없으므로, C1 fallback 결과를 raw-Jbar 실측으로 가장하지 않고
bb 및 대표 collisional 행을 unavailable로 남겼다.

### 2.3 production thermal ledger

각 셀에서 다음 행을 생산 `simul_r1` 산술로 채웠다:

- `heating_photoion`
- `heating_deposition`
- `cooling_ff`
- `cooling_bf`
- `cooling_bf_net = cooling_bf - heating_photoion`
- `cooling_bb_collisional`
- `cooling_adiabatic`
- `thermal_net = heating - cooling`

`heating_MA_LINE_DESTRUCT`만 unavailable이다. 이 채널은 transport 중
macro-atom energy를 k-packet으로 보내지만, 원 동결 run은 셀별 volumetric
rate ledger를 쓰지 않았다. 임의 재구성이나 0 치환을 하지 않았다.

## 3. CMFGEN 단위와 parser 실증

### 3.1 `n_e` 단위

`cmfgen_source_evidence.csv`에 다음 근거의 실제 파일과 physical line을
기록했다.

| 근거 | 소스 physical line | 내용 |
|---|---:|---|
| RVTJ 헤더 writer | `cmfgen_sub.f:4421` | `Electron density` |
| RVTJ 값 writer | `cmfgen_sub.f:4423` | `WRITE ... ED` |
| 변수 단위 선언 | `mod_cmfgen.f:211` | `ED(:) !Electron density (#/cm^3)` |

`cmfgen_parser_roundtrip.csv`는 각 선택 depth에서 RVTJ 원문 line, raw 값,
post-conversion 값을 보존한다. `n_e` 변환은 `identity #/cm^3`이며 추측
배율이 없다. T만 RVTJ 헤더의 `(10^4K)`에 따라 1e4를 곱한다.

RVTJ와 각 PRRR의 `Electron Density`도
`cmfgen_snapshot_consistency.csv`에서 대조했다. s0/s8의 16행은 최대 상대차
약 `5.91e-6`로 `same_snapshot`이다. s43의 8행은 상대차
`-1.8805437716e-3`이며 모두
`different_snapshot_or_output_generation`으로 공개된다. 따라서 세 셀 전체가
동일 snapshot이라는 주장은 하지 않는다.

### 3.2 PRRR, OUT, GENCOOL

- PRRR는 CMFGEN run과 함께 보존된 `MODEL_SPEC`의 `N_SL`을 읽는다.
  모든 10-depth chunk에서 정확히 `N_SL`개의 photo-rate row를 요구하고,
  ND=90의 모든 chunk를 요구한다.
- Γ는 `sum(PR)/Ion Density`, α는 PRRR scalar block을 읽는다. α 단위는
  `wrrecomchk_v3.f`의 `TOTRR=TOTRR/ED/DHYD` 소스 식으로 `cm^3/s`를
  실증했다.
- ion OUT은 파일 헤더의 `NLEV`를 읽고 90개 depth block마다 정확히
  `NLEV`개의 departure coefficient를 요구한다. Lumina `level_num=0`은
  OUT block의 첫 coefficient에 대응하는 0-based mapping이다.
- GENCOOL은 90 depth를 모두 요구하고 모든 ion의 bound-free,
  collisional, free-free를 합산하며 explicit `Net Cooling Rate`도 읽는다.
  Fortran의 생략형 3자리 지수(예: `4.0938-101`)도 엄격히 복원한다.

`cooling_bf_net`, collisional, 부호를 H-C로 맞춘 `thermal_net`은 동일
ledger 의미로 비교한다. Lumina의 `cooling_ff`/`cooling_bf` 단독 행은
emission-only인 반면 GENCOOL 행은 emission-minus-absorption의 net이므로
양쪽 수치는 보존하되 `context_only_nonidentical`로 표시하고 ratio를 만들지
않는다.

## 4. 셸 대응

| Lumina 셸 | v_L [km/s] | CMFGEN depth | v_C [km/s] | Δv [km/s] | 범위 내 |
|---:|---:|---:|---:|---:|:---:|
| s0 | 4264.000 | 67 | 4394.182 | +130.182 | 예 |
| s8 | 10088.000 | 54 | 10163.506 | +75.506 | 예 |
| s43 | 35568.000 | 10 | 35497.710 | -70.290 | 예 |

`s44` 중심은 36296 km/s로 CMFGEN 상한 밖이다. 따라서 `s43`이 CMFGEN
범위 안에 남는 가장 바깥 Lumina 셀이다.

## 5. 수량 커버리지

세 oracle은 각각 182 data row이며 category별 행 수가 동일하다.

| category | 셀당 행 |
|---|---:|
| state | 28 |
| input provenance | 12 |
| bf | 64 |
| ff | 5 |
| bb | 48 |
| collisional | 16 |
| thermal | 9 |

CMFGEN comparator 전체 546행의 census:

- `compared`: 79
- `context_only_nonidentical`: 9
- CMFGEN identical anchor 없음: 217
- Lumina 원자료/생산 topology상 unavailable: 241

주요 확장:

- `T_e`, `n_e`: 각 3 비교
- `n_ion`: 24 비교
- representative `b_k`: 14 비교, positive population/non-sentinel level이
  없는 10행은 명시적 unavailable
- Γ 및 α total: 각각 9 비교. NLTE 구성에 다음 ion pair가 없는 상위
  stage 15행은 0이 아니라 unavailable
- `cooling_bf_net`, `cooling_bb_collisional`, `thermal_net`: 각 3 비교
- GENCOOL ff/bf emission-vs-net 문맥값: 각각 3

spontaneous/stimulated α split, monochromatic χ/η, representative `C_lu/C_ul`,
complete-element CMFGEN ion fraction처럼 CMFGEN이 동일 수량을 내지 않는
행은 사유를 보존했다.

## 6. 자기 스모크와 OFF 검증

최종 바이너리로 결과 디렉터리와 별도 `/tmp` 디렉터리에 독립 실행했다.
세 oracle 모두 `cmp` 성공했다.

| 파일 | SHA-256 (두 실행 동일) |
|---|---|
| `lumina_oracle_cell_s0.csv` | `526f490fee030ce9573ec1d00267c706991a8865539ebe9c432eefd5efd94d7f` |
| `lumina_oracle_cell_s8.csv` | `8210eef19a569c452acb6b7cea1b7d29c9c3022efa73f8baccdfced90683d052` |
| `lumina_oracle_cell_s43.csv` | `4f7819f51ffb04b02fa06768755179517ddd7aa5252a1f458d0afe8609f15e07` |

추가 확인:

- `make bench_frozen_oracle`: 성공
- CSV schema/status/unavailable-note 검사: 성공
- comparator `py_compile` 및 실제 RVTJ/PRRR/OUT/GENCOOL/EDDFACTOR 실행: 성공
- 일곱 frozen input 파일의 SHA-256:
  `validation/gate_b_dual_oracle/phase1_5/frozen_input_sha256.txt`
- macro 미정의 object와 명시적 `-ULUMINA_FROZEN_ORACLE` object:
  byte-identical (`ed9973e4...dd55`)
- 정상 object의 `lumina_oracle`/`g_oracle` symbol: 0
- Git 명령과 GPU 실행: 사용하지 않음

## 7. 산출물

- 구현: `bench_frozen_oracle.c`, `src/lumina_plasma.c`, `src/lumina.h`
- comparator: `scripts/oracle_compare_cmfgen.py`
- 상세 산출: `validation/gate_b_dual_oracle/phase1_5/`
- 요약: `docs/CODEX_GATEB_A_IMPL.md`

Phase 1 산출물은 삭제하거나 덮어쓰지 않고 `phase1_5`에 분리 보존했다.

## 8. Codex B 재검증 절차

Git과 GPU 없이 다음 순서로 독립 검증할 수 있다.

```sh
make bench_frozen_oracle
d1=$(mktemp -d /tmp/gateb15_B1.XXXXXX)
d2=$(mktemp -d /tmp/gateb15_B2.XXXXXX)
./bench_frozen_oracle logs/coevolve_consume_parity50 \
  data/tardis_reference_toy06_19p48d_sivcaiv "$d1"
./bench_frozen_oracle logs/coevolve_consume_parity50 \
  data/tardis_reference_toy06_19p48d_sivcaiv "$d2"
for s in 0 8 43; do
  cmp "$d1/lumina_oracle_cell_s${s}.csv" \
      "$d2/lumina_oracle_cell_s${s}.csv"
done
python3 -m py_compile scripts/oracle_compare_cmfgen.py
python3 scripts/oracle_compare_cmfgen.py
```

확인점은 `s43`의 `in_cmfgen_range=True`, C2 1000/1000 적재와 positive
소비 횟수, thermal 9행의 8 available/1 explained-unavailable,
`cmfgen_source_evidence.csv`의 n_e 3중 근거, 모든 unavailable 행의 비어
있지 않은 note다.
