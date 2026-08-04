# parity59 coupled-root 및 lever 가산성 감사

작성일: 2026-08-03  
역사 입력: `logs/coevolve_consume_a10_kx_gphall/` (2026-07-15)  
현행 입력: `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/`  
CMFGEN: `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/`

## -o 요약

- **baseline 게이트 PASS.** 파라미터화 사본이 07-15 원본과 같은
  `own cs.J=16616.557659 K`, `CMFGEN-J=18277.377195 K`를 냈다. 반올림 endpoint
  차이는 발주 기준 `+3497 / +1660 / +483 K`와 모두 일치한다.
- parity59 상태에 같은 역사 추정기를 적용하면 `own cs.J=22801.407876 K`,
  `CMFGEN-J=18385.799234 K`다. committed `21227.639444 K`에서 own root는 오히려
  `+1573.768432 K` 더 뜨겁고, root 안에서 CMFGEN J로 바꾸면 `-4415.608642 K`
  내려간다. CMFGEN-root는 진리보다 `374.200766 K` 낮다.
- **가산성 기각.** parity59의 독립 단독 레버가 원래 `2467.639444 K` 초과를 닫는
  방향의 합은 `-1947.969198 K`이다. 즉 초과를 닫지 않고 더 키운다.
  `sum(단독)-전체=+4415.608642 K`(signed-temperature 정의), 또는 초과를 닫는
  방향으로 쓰면 `sum(단독 closure)-|전체|=-4415.608642 K`다.
- 6개 누적 순서는 모두 최종적으로 진리에 도달하여
  `누적 최종-전체=0 K`지만, 이것은 endpoint 차이의 망원경 합일 뿐 가산성 증거가
  아니다. R과 J의 순서별 귀속 폭은 각각 `4415.608642 K`다.
- 판정: 단일 원소의 독립 오차가 아니라 **root 연산 R과 field 교체 J의 비가산 결합**, 즉
  연산자 분할/고정점 문제다. parity59의 `R x J=-4415.608642 K`가 전체 초과
  `2467.639444 K`보다 1.79배 크다.

## 1. 무엇을 계산했는가

`validation/cmfgen_toy06_19p48d/analysis/radeq_ledger_audit/radeq_coupledroot.py`는
수정하지 않았다. 사본
`validation/chain_replay_parity59/radeq_ledger_audit/radeq_coupledroot.py`가 다음 경로를
인자로 받는다: run/capture directory, atomic-model directory, deposition CSV, CMFGEN
jtable, CMFGEN directory, output directory.

추정기 정의는 07-19와 같다. s0의 committed `n_ion` 인접-stage 비를 기준으로
`r_j(T)=r_j(T_committed)*(T/T_committed)^0.8`을 만들고, 각 trial T에서 ion ladder,
line exchange, analytic free-bound, free-free, adiabatic 항을 다시 계산한다. 3500–140000 K의
24개 기하 간격을 위로 훑어 처음 만나는 `r(T)>0`에서 `r(T)<=0` 교차를 40회 이분한다.
line pump만 own `cs_J`와 CMFGEN jtable 사이에서 교체한다. `H_photo=7.2e-7
erg cm^-3 s^-1`은 원 추정기의 고정 scalar 그대로이며 field와 함께 재계산하지 않는다.

원본에 이미 들어 있던 VR_STD collision-strength minimum과 grid 밖 `J=1e-30` 관례는
baseline 코드 동일성 때문에 보존했다. 이는 캡처의 결측치를 채우거나 결과를 맞추려고
새로 넣은 floor/대체값이 아니다. 새 사본에서는 지수 cap과 분모 floor를 쓰지 않고,
정의되지 않은 분모·partition·root는 예외로 종료한다. 이번 두 입력에는 예외가 없었다.

중요한 범위 구분: 아래 수치는 **07-19 역사 analytic estimator를 parity59의 final captured
state에 적용한 값**이다. parity59 production solver는 `RADEQ_DB_FB=1`,
`BF_RATE_POPS=1`이므로 이 수치를 production solver의 정확 counterfactual root라고
재명명하지 않는다.

## 2. baseline 재현 게이트

게이트는 정수 온도라는 역사 표의 관례를 그대로 썼다. 먼저 endpoint를 반올림한 뒤
차이를 냈다. 미세값은 결과를 07-15에 맞춰 조정하지 않은 raw root다.

| 게이트 | 기대 | 재현 | 결과 |
|---|---:|---:|---|
| CMFGEN-J coupled root | 18,277 K | 18,277.377195 K → 18,277 K | PASS |
| committed→own `cs.J` | +3,497 K | 16,617−13,120 = +3,497 K | PASS |
| own `cs.J`→CMFGEN-J | +1,660 K | 18,277−16,617 = +1,660 K | PASS |
| CMFGEN-J→truth | +483 K | 18,760−18,277 = +483 K | PASS |

지속 산출물은 `radeq_ledger_audit/baseline_0715_results/baseline_gate.csv`이며 네 행 모두
`PASS`다. 원본을 별도로 실행한 root와 사본 root는 표시된 전체 double 값까지 같았다.

## 3. 07-15와 parity59 coupled-root/lever 대조

아래의 delta는 모두 오른쪽 endpoint에서 왼쪽 endpoint를 뺀 signed temperature 변화다.
따라서 parity59 전체 `truth-committed`는 냉각이 필요하므로 음수다.

| endpoint 또는 순차 lever | 07-15 | capture 188932 | 변화 |
|---|---:|---:|---|
| committed s0 `T_e` | 13,119.874754 K | 21,227.639444 K | 진리 대비 부호 역전 |
| own `cs.J` coupled root | 16,616.557659 K | 22,801.407876 K | +6,184.850217 K |
| CMFGEN-J coupled root | 18,277.377195 K | 18,385.799234 K | +108.422039 K |
| CMFGEN truth anchor | 18,760.000000 K | 18,760.000000 K | 동일 |
| R: committed→own root | +3,496.682905 K | +1,573.768432 K | 크기 감소; 캡처 초과를 악화 |
| J: own→CMFGEN-J root | +1,660.819536 K | **−4,415.608642 K** | 부호 역전 |
| O: CMFGEN-J root→truth | +482.622805 K | +374.200766 K | 감소 |
| 전체: truth−committed | +5,640.125246 K | **−2,467.639444 K** | 부호 역전 |

출처는 `coupled_roots.csv`와 `historical_levers.csv`의 각 행에 파일·필드·정의 문자열로
기록했다. 18,760 K campaign anchor의 CMFGEN `RVTJ: Temperature (10^4K)` 선형 보간값은
18,760.319891 K이고, 표는 발주에서 고정한 18,760 K를 사용한다.

## 4. 독립 가산성 검정

역사 3구간을 그대로 더하면 정의상 endpoint가 소거되어 언제나 전체와 같다. 이를 독립성
검정으로 쓰지 않았다. 대신 상태 함수 `F(R,J,O)`를 명시했다.

- `R=0`: CSV의 committed T를 유지. `R=1`: coupled root를 푼다.
- `J=0`: own `cs.J`; `J=1`: CMFGEN J. 단, `R=0`이면 root를 호출하지 않으므로 J는
  committed T를 바꾸지 않는다.
- `O=1`: 역사 추정기의 CMFGEN-root→truth residual `truth-CMFroot`를 더한다.
- 실제 식:
  `F=(committed if R=0 else [own-root if J=0 else CMF-root]) + O*(truth-CMFroot)`.

### 단독 레버

| 값 | 07-15 ΔT | capture ΔT | capture의 초과를 닫는 양 |
|---|---:|---:|---:|
| R alone: `F(1,0,0)-F(0,0,0)` | +3,496.682905 | +1,573.768432 | −1,573.768432 |
| J alone: `F(0,1,0)-F(0,0,0)` | 0 | 0 | 0 |
| O alone: `F(0,0,1)-F(0,0,0)` | +482.622805 | +374.200766 | −374.200766 |
| **단독 합** | **+3,979.305710** | **+1,947.969198** | **−1,947.969198** |
| **전체 truth−committed** | **+5,640.125246** | **−2,467.639444** | **+2,467.639444** |
| **sum(단독)−전체** | **−1,660.819536** | **+4,415.608642** | **−4,415.608642** |

캡처에서 R과 O는 단독으로 모두 과열을 키운다. 필요한 냉각은 J가 R과 함께 켜졌을 때만
나타난다. 고전적인 2-factor 상호작용
`F(1,1,0)-F(1,0,0)-F(0,1,0)+F(0,0,0)`은 07-15에서
`+1660.819536 K`, 캡처에서 `-4415.608642 K`다. 따라서 단독 합이 전체를 못 닫는
정도가 작지도 않고, 캡처에서는 방향마저 반대다.

### 누적 순서 6개

| dataset | R의 순서별 marginal 범위 | J의 범위 | O의 범위 | 귀속 폭 max | 누적 최종−전체 |
|---|---:|---:|---:|---:|---:|
| 07-15 | +3,496.683 … +5,157.502 | 0 … +1,660.820 | +482.623 | 1,660.820 | 0 |
| capture | −2,841.840 … +1,573.768 | −4,415.609 … 0 | +374.201 | **4,415.609** | 0 |

예를 들어 캡처의 `R→J→O` 귀속은 `+1573.768, -4415.609, +374.201 K`지만,
`J→R→O`는 `0, -2841.840, +374.201 K`다. 두 순서 모두 최종 변화는
`-2467.639 K`로 같지만 R/J 중 어느 원소에 얼마를 귀속할지는 4,415.609 K나 바뀐다.
전체 18개 step은 `cumulative_orders.csv`, 폭은 `order_dependence.csv`에 있다.

## 5. 판정

**비가산·비선형 결합이다.** capture에서 독립 단독 레버의 방향성 closure 합은
`-1947.969 K`, 닫아야 할 전체 초과는 `+2467.639 K`다. 차이 `-4415.609 K`는
전체의 179%이며, 6개 순서의 R/J 귀속 폭도 정확히 같은 4,415.609 K다.

따라서 `own field`, `CMFGEN field`, `residual formula` 중 하나를 독립적인 단일 원인으로
지목할 수 없다. field 교체 J는 root 연산 R 밖에서는 효과가 0이고 안에서는 전체 gap보다
큰 효과를 낸다. 측정된 민감도는 구성요소 자체의 기여도가 아니라
`S→J→T_e→n_i→chi,eta→S` 고리가 다시 닫힐 때의 고정점 이동이다.

## 6. 출처와 산출물

수치별 원천은 각 results directory의 `provenance.csv`에 고정했다.

- committed `T_e`, `n_e`: 각 `lumina_plasma_state.csv:T_e,n_e`.
- own pump: 각 `lumina_coevolve_field.csv:bin,cs_J`.
- ion ladder calibration: 각 `lumina_ion_pops.csv:Z,stage,n_ion`.
- atomic terms: 역사에는 `data/tardis_reference_toy06_19p48d/`, 캡처에는 실제 env의
  `data/tardis_reference_toy06_19p48d_sivcaiv/` 아래
  `line_list.csv`, `levels.csv`, `ionization_energies.csv`.
- deposition: 해당 model directory의 `deposition_cmfgen.csv:heating_rate`; 두 파일은
  SHA256가 같고 s0는 `1.506865e-3 erg cm^-3 s^-1`다.
- CMFGEN pump: `data/cmfgen_jtable_toy06_19p48d.bin:J_nu`. sidecar JSON은 이 table의
  source를 CMFGEN `EDDFACTOR:J_nu`와 `RVTJ:Velocity`로 명시한다.
- truth cross-check: CMFGEN `RVTJ:Velocity (km/s), Temperature (10^4K)`.
- `MEANOPAC`은 존재 gate만 했다. 이 coupled-root 식에는 opacity-table 항이 없어 수치에
  소비하지 않았으며, 소비했다고 표기하지 않았다.
- `lumina_levelpop.csv`도 필수 캡처 존재 gate만 했다. 역사 추정기는 LTE partition과
  `lumina_ion_pops.csv`를 사용하므로 level-pop 값을 대체 입력으로 쓰지 않았다.

실행한 재현 명령은 다음과 같다.

```bash
python3 validation/chain_replay_parity59/radeq_ledger_audit/radeq_coupledroot.py \
  --input-dir logs/coevolve_consume_a10_kx_gphall \
  --model-dir data/tardis_reference_toy06_19p48d \
  --deposition-file data/tardis_reference_toy06_19p48d/deposition_cmfgen.csv \
  --cmfgen-jtable data/cmfgen_jtable_toy06_19p48d.bin \
  --cmfgen-dir /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 \
  --output-dir validation/chain_replay_parity59/radeq_ledger_audit/baseline_0715_results \
  --label baseline_0715 --baseline-gate

python3 validation/chain_replay_parity59/radeq_ledger_audit/radeq_coupledroot.py \
  --input-dir /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932 \
  --model-dir data/tardis_reference_toy06_19p48d_sivcaiv \
  --deposition-file data/tardis_reference_toy06_19p48d_sivcaiv/deposition_cmfgen.csv \
  --cmfgen-jtable data/cmfgen_jtable_toy06_19p48d.bin \
  --cmfgen-dir /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 \
  --output-dir validation/chain_replay_parity59/radeq_ledger_audit/results \
  --label capture_188932
```

모든 경로와 필드 사용 여부는 results의 `provenance.csv`에도 남아 있다.
핵심 산출물은 다음과 같다.

- `baseline_0715_results/{baseline_gate,coupled_roots,historical_levers}.csv`
- `baseline_0715_results/{standalone_additivity,cumulative_orders,order_dependence}.csv`
- `results/`의 같은 파일들과 `summary.json`
- `validation/chain_replay_parity59/comparison_summary.csv`의 갱신된 root/lever 네 행

## UNRESOLVED

1. **현 production solver의 정확 counterfactual root.** 이번 발주의 역사 추정기 결과는
   닫혔지만, parity59 production solver의 DB free-bound/BF population-rate trial table은
   캡처에 없다. 새 런 없이 둘을 동일시할 수 없다.
2. **가산성의 더 세밀한 물리 factorization.** 이번 R/J/O cube는 역사 세 lever의 중첩을
   검정한다. `Gph`, `Hex`, DB free-bound emissivity, ETLA, population owner를 각각 독립
   factor로 나누려면 trial-T 항별 persistent table 또는 검증된 current-solver mirror가
   필요하다.

## 잔여 작업

- 이번 발주 범위에서 필요한 baseline gate, 캡처 root, 단독/누적/순서 가산성 계산은 완료.
- production solver 자체의 항별 cube가 필요하면 새 모델 런 대신 먼저 캡처만 소비하는
  current-solver offline mirror를 별도 검증해야 한다. 현재 캡처만으로는 정확 closure를
  보증할 수 없어 수치를 만들지 않았다.

새 모델 런·GPU 작업·commit은 하지 않았다. 원본 `radeq_*.py`, 원본 `VERDICT.md`, `src/`는
수정하지 않았다.
