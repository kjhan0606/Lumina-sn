# Wave 3 D8 falsifier — EW 무장이 오프라인 pair 기준선을 바꾸는가

날짜: 2026-07-31  
범위: parity59 frozen oracle replay, CPU single thread, 신규 모델 런 0, `src/` 수정 0

## 판정

**D8은 이번 Wave 3 shadow 측정의 `p_pair`에 영향을 주지 않았다.** s8과 s0 모두 armed/unarmed pair 산출이 byte-identical이며, 양수 수치의 최대 차이는 `0 dex`다. 따라서 사전등록 판독에 따라 `docs/CODEX_WAVE3_B2_TEST.md` §4.3/§4.4의 `p_pair`, `D(pair)`, improvement 분모는 유효하다. unarmed 기준 improvement 재산정은 필요 없고 B2 값이 그대로 유지된다.

이 결론은 **이 frozen replay와 shadow 분모**에 한정된다. EW 무장이 전역 33-slot layout을 선택한다는 live-run 구조 우려는 별도 잔여 위험이다.

| 셀/대상 | 전체 oracle CSV | pair ion fraction | pair-owned level population | 최대 절대/상대/dex 차이 |
|---|---|---|---|---:|
| s8 S+Fe | byte-identical | 6/6행 byte-identical | 4902/4902행 byte-identical | `0 / 0 / 0 dex` |
| s0 Fe | byte-identical | 3/3행 byte-identical | 4198/4198행 byte-identical | `0 / 0 / 0 dex` |

독립적으로 전체 oracle도 B2 당시 armed 산출과 byte-identical이다.

- s8: 현 armed = 현 unarmed = `/tmp/w31_on_a.JuCpDY` B2 oracle, SHA-256 `2ee175c6be4a0ab6ae07034722180c6312f0a6e55e1009fcdda23b72310693eb`.
- s0: 현 armed = 현 unarmed = `/tmp/w31_s0_fe.C6wf6v` B2 oracle, SHA-256 `7a79f4f345d5c8500b48530a7f859bbe05a57771f43e52f983639ef9a0839381`.

## 1. 복원한 Wave 3 replay 절차

`docs/CODEX_WAVE3_A_IMPL.md`가 연결하는 상세 구현 보고서와 `bench_frozen_oracle.c`의 실제 loader를 기준으로 다음을 복원했다.

- frozen run: `/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59`
- model projection: `data/tardis_reference_toy06_19p48d_sivcaiv`
- rate consumer: iteration 11 (`lumina_jbar_dump.csv`)
- lagged field producer: iteration 10 (`lumina_c1_bins.csv`, `lumina_c2_bfr_dump.csv`)
- 셀: s8 S/Fe, s0 Fe
- pair authority: 원래 순서의 16개 II–III pair assembler 호출
- armed candidate: `COMMIT=0`, 즉 EW solve/dump만 수행하고 pair authority를 유지
- 실행 파일: `bench_frozen_oracle`, SHA-256 `5ba234a613aeef19c4487899aa15156950ebda2f6a0c077a129feebb2205fe78`

셀 selector와 EW gate의 shell 값이 scalar라서 셀당 armed/unarmed 한 쌍, 총 4개 독립 프로세스를 실행했다. 모두 exit 0이다. 이 benchmark는 archived final population을 로드해 rate matrix를 재조립하는 oracle이며 새 transport/plasma 모델을 진화시키지 않는다.

### pair dump 정의

- `pair_ion_fractions.csv`: archived `lumina_ion_pops.csv`의 II/III/IV와 모든 recorded stage의 원소합을 사용한다. 이것이 B2 `p_pair`의 분율 정의다.
- `pair_level_populations.csv`: legacy pair owner가 실제 소유하는 II/III full-level의 `(Z,ion,level_num,n_k,n_ground,b_k)`를 identity 순서로 덤프한다. armed-only IV slot은 EW observer 소유이므로 pair dump에서 제외했다.
- oracle은 pair matrix를 **assemble만 하고 다시 solve하지 않으므로**, archived `lumina_levelpop.csv`에서 identity로 로드된 final population이 이 replay의 pair 최종 준위인구다.

덤프/비교 helper는 `scripts/wave3_d8_pair_dump.py`다.

## 2. RESOLVED env 스냅샷

각 프로세스는 parity59 `stdout.log`의 `=== RESOLVED CONFIG`에서 117개 설정을 동일 loader로 복원하고 `OMP_NUM_THREADS=1`로 덮어썼다. 중요한 공통 실측 입력은 `LUMINA_SUPER_LEVELS=1`이다. 즉 이번 w31 환경에서는 unarmed도 이미 super-level 요청이 켜져 있다.

| 실행 | shell selector | EW 변수 | 전체 snapshot / SHA-256 |
|---|---:|---|---|
| armed s8 | `8` | `ELEMENT_WIDE=1`, `Z=16,26`, `SHELL=8`, `COMMIT=0`, `DUMP=1`, `DUMP_DIR=...armed_s8...` | [/tmp/w3_d8_armed_s8.5ZcbNC/resolved_env.txt](/tmp/w3_d8_armed_s8.5ZcbNC/resolved_env.txt), `d6b11e9ca8a69950dfbc50ce494977246fceaad532526ad7c59fecc127897505` |
| unarmed s8 | `8` | 여섯 EW 변수 모두 **미설정** | [/tmp/w3_d8_unarmed_s8.Wqar0J/resolved_env.txt](/tmp/w3_d8_unarmed_s8.Wqar0J/resolved_env.txt), `e818d84537ced6be6898d6413115e8c2fe1bb0689c079ec608e79a8c69056727` |
| armed s0 | `0` | `ELEMENT_WIDE=1`, `Z=26`, `SHELL=0`, `COMMIT=0`, `DUMP=1`, `DUMP_DIR=...armed_s0...` | [/tmp/w3_d8_armed_s0.0nj9y1/resolved_env.txt](/tmp/w3_d8_armed_s0.0nj9y1/resolved_env.txt), `b3bb3dc081b93af9635c725cf7b278daf41b03ae3e1dcdc5ddb9221241739c6a` |
| unarmed s0 | `0` | 여섯 EW 변수 모두 **미설정** | [/tmp/w3_d8_unarmed_s0.bDPrCR/resolved_env.txt](/tmp/w3_d8_unarmed_s0.bDPrCR/resolved_env.txt), `40b25f43dfd0e81bcd357ad216e4ab1eb8b00541c231eb3c26bed3625677ebd6` |

armed/unarmed snapshot의 `comm -3`는 각 셀에서 위 여섯 EW 행만 출력한다. unarmed는 `env -i`에서 시작했으므로 `ELEMENT_WIDE=0`도 주지 않은 완전 미설정 조건이다.

## 3. 배너 실측

추론 대신 각 실행의 실제 stdout/stderr를 인용한다.

armed s8 stdout:

```text
[ORACLE] recovered 117 production settings; CPU single-thread path
  [NLTE] Total NLTE levels: 21432
  [EW] CPU pilot layout: 33 slots; Fe/S II-IV are contiguous
  [NLTE] Super-levels: ACTIVE (21432 FL -> 3030 SL across ions)
```

armed s8 stderr:

```text
[EW] Stage-2A CPU path armed: Z=16,26 shell=8 commit=0 dump=1
```

armed s0 stdout/stderr도 같은 유효값을 인쇄했다.

```text
  [NLTE] Total NLTE levels: 21432
  [EW] CPU pilot layout: 33 slots; Fe/S II-IV are contiguous
  [NLTE] Super-levels: ACTIVE (21432 FL -> 3030 SL across ions)
[EW] Stage-2A CPU path armed: Z=26 shell=0 commit=0 dump=1
```

unarmed s8/s0 stdout:

```text
  [NLTE] Total NLTE levels: 21038
  [NLTE] Super-levels: ACTIVE (21038 FL -> 2828 SL across ions)
```

따라서 armed의 layout slot 수는 배너 실측 `33`, super-mode 유효 상태는 배너 실측 `ACTIVE`다. unarmed도 `ACTIVE`이며, armed/unarmed 차이는 이번 생산 env에서 super-mode on/off가 아니라 layout 크기다. 원문은 [armed s8 stdout](/tmp/w3_d8_armed_s8.5ZcbNC/stdout.txt), [armed s0 stdout](/tmp/w3_d8_armed_s0.0nj9y1/stdout.txt), [unarmed s8 stdout](/tmp/w3_d8_unarmed_s8.Wqar0J/stdout.txt), [unarmed s0 stdout](/tmp/w3_d8_unarmed_s0.bDPrCR/stdout.txt)에 보존했다.

## 4. byte/수치 diff

### 4.1 최종 ion fraction

armed와 unarmed의 다음 값이 byte-identical이다.

| 셀 | 원소 | II | III | IV |
|---|---|---:|---:|---:|
| s8 | S | `1.76884325333996617e-02` | `9.72337092292610006e-01` | `9.95489874500191455e-03` |
| s8 | Fe | `8.74934763309054075e-06` | `9.39933080160507917e-01` | `5.96946519256283531e-02` |
| s0 | Fe | `7.66378178899405437e-07` | `4.19589421452688446e-03` | `9.82050900459393805e-01` |

artifact:

- s8 [armed](/tmp/w3_d8_armed_s8.5ZcbNC/pair_ion_fractions.csv) / [unarmed](/tmp/w3_d8_unarmed_s8.Wqar0J/pair_ion_fractions.csv): SHA-256 `c2b977bd42f45b0d9763c9d70df0848b822059b3270212381dc723f30c6bd337`.
- s0 [armed](/tmp/w3_d8_armed_s0.0nj9y1/pair_ion_fractions.csv) / [unarmed](/tmp/w3_d8_unarmed_s0.bDPrCR/pair_ion_fractions.csv): SHA-256 `29e56721aa17a9e9d561ead771f391aa3930fc992108da82965e4ad5d17ad683`.

### 4.2 final pair-owned level population

- s8 S II/III + Fe II/III: 4902 data rows, [armed](/tmp/w3_d8_armed_s8.5ZcbNC/pair_level_populations.csv) / [unarmed](/tmp/w3_d8_unarmed_s8.Wqar0J/pair_level_populations.csv) byte-identical, SHA-256 `13662c9881f209d64e4a4dd60807daf01209cab4b118cfad9334caa851c4472e`.
- s0 Fe II/III: 4198 data rows, [armed](/tmp/w3_d8_armed_s0.0nj9y1/pair_level_populations.csv) / [unarmed](/tmp/w3_d8_unarmed_s0.bDPrCR/pair_level_populations.csv) byte-identical, SHA-256 `38154f7d4cd3fb40abed0601c44099eaf70ef3d22a1a2a7b1be9e516d9147fd6`.
- `n_k`, `n_ground`, `b_k`의 max absolute diff = `0`, max relative diff = `0`, 양수 항 max `|log10(armed/unarmed)| = 0 dex`.

원자료와 자동 수치 집계는 [s8 comparison](/tmp/w3_d8_armed_s8.5ZcbNC/comparison_summary.csv), [s0 comparison](/tmp/w3_d8_armed_s0.0nj9y1/comparison_summary.csv)에 있다.

## 5. B2 분모 영향

armed `p_pair`와 unarmed `p_pair`가 동일하므로 `delta d_k(pair)=0 dex`이고 B2의 수치를 바꾸지 않는다.

| 셀/원소 | B2 `d_k(pair)` II/III/IV | B2 `D(pair)` | D8 후 상태 |
|---|---|---:|---|
| s8 S | `0.1045 / 0.00208 / 2.1668` | `0.75779` | 유지 |
| s8 Fe | `0.5563 / 0.02560 / 1.3398` | `0.64057` | 유지 |
| s0 Fe | `4.8875 / 1.1385 / 0.00306` | `2.0097` | 유지 |

따라서 B2 improvement도 s8 S `-139.8%`, s8 Fe `-93.2%`, s0 Fe aggregate `+58.13%`로 유지된다. 이는 B2의 다른 scope/numerics 판정을 승격하지 않으며, 오직 D8에 의한 분모 오염 가설을 기각한다.

## 6. 재현 명령

아래는 기존 binary와 frozen archive를 그대로 쓰며 model run을 시작하지 않는다.

```bash
D8_FROZEN=/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59
D8_MODEL=data/tardis_reference_toy06_19p48d_sivcaiv
D8_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
D8_A8=$(mktemp -d /tmp/w3_d8_armed_s8.XXXXXX)
D8_U8=$(mktemp -d /tmp/w3_d8_unarmed_s8.XXXXXX)
D8_A0=$(mktemp -d /tmp/w3_d8_armed_s0.XXXXXX)
D8_U0=$(mktemp -d /tmp/w3_d8_unarmed_s0.XXXXXX)

env -i PATH="$D8_PATH" LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=16,26 LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$D8_A8" \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 \
  ./bench_frozen_oracle "$D8_FROZEN" "$D8_MODEL" "$D8_A8" \
  >"$D8_A8/stdout.txt" 2>"$D8_A8/stderr.txt"

env -i PATH="$D8_PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 \
  ./bench_frozen_oracle "$D8_FROZEN" "$D8_MODEL" "$D8_U8" \
  >"$D8_U8/stdout.txt" 2>"$D8_U8/stderr.txt"

env -i PATH="$D8_PATH" LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=26 LUMINA_NLTE_ELEMENT_WIDE_SHELL=0 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$D8_A0" \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=0 \
  ./bench_frozen_oracle "$D8_FROZEN" "$D8_MODEL" "$D8_A0" \
  >"$D8_A0/stdout.txt" 2>"$D8_A0/stderr.txt"

env -i PATH="$D8_PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=0 \
  ./bench_frozen_oracle "$D8_FROZEN" "$D8_MODEL" "$D8_U0" \
  >"$D8_U0/stdout.txt" 2>"$D8_U0/stderr.txt"

python3 scripts/wave3_d8_pair_dump.py \
  --frozen "$D8_FROZEN" --armed-dir "$D8_A8" --unarmed-dir "$D8_U8" \
  --shell 8 --z 16,26
python3 scripts/wave3_d8_pair_dump.py \
  --frozen "$D8_FROZEN" --armed-dir "$D8_A0" --unarmed-dir "$D8_U0" \
  --shell 0 --z 26

cmp "$D8_A8/lumina_oracle_cell_s8.csv" "$D8_U8/lumina_oracle_cell_s8.csv"
cmp "$D8_A8/pair_ion_fractions.csv" "$D8_U8/pair_ion_fractions.csv"
cmp "$D8_A8/pair_level_populations.csv" "$D8_U8/pair_level_populations.csv"
cmp "$D8_A0/lumina_oracle_cell_s0.csv" "$D8_U0/lumina_oracle_cell_s0.csv"
cmp "$D8_A0/pair_ion_fractions.csv" "$D8_U0/pair_ion_fractions.csv"
cmp "$D8_A0/pair_level_populations.csv" "$D8_U0/pair_level_populations.csv"

rg "CPU pilot layout|Super-levels:|Total NLTE levels" \
  "$D8_A8/stdout.txt" "$D8_U8/stdout.txt" \
  "$D8_A0/stdout.txt" "$D8_U0/stdout.txt"
```

## 7. 규율 확인

- `src/` 수정 없음.
- GPU 실행 없음.
- 신규 model/transport run 없음.
- 기존 parity59 frozen oracle replay만 4프로세스 실행.
- scratch와 수치 산출물은 모두 `/tmp/w3_d8_*` 아래에 보존.
- 신규 저장소 산출물은 이 보고서와 offline dump helper뿐이다.
