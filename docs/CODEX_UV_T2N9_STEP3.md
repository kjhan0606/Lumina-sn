# UV T2/N9 3단 — 판정과 대장 기재

작성일: 2026-08-02 (Asia/Seoul)  
판독 정본: `validation/uv_t2n9/PREREG.md`  
2단 산출물: `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_t2n9_offline/`

## -o 요약

- **BALL 구성 가능 여부: 불가.** `t2_C_population_coverage.json`의
  `nonpositive_required_rows=28,949`는 writer가 600–3000 Å 선만 기록한 LINEPOP
  행들이다. 따라서 28,949행은 전부 고정 BALL 내부이고, `outside_BALL_rows=0`은
  writer 선택 술어에서 정적으로 따라온다. 이 행들의 population-native `chi_line`
  기여는 기록된 population으로 정의되지 않는다. 기존 산출물만으로 인증 가능한
  recorded-A 기여분율 상한은 비음수 opacity의 자명한 **1.000000**뿐이며, 더 조밀한
  셸별 수치와 0/음수 분리는 추가한 forensic 1회 판독 전까지 정의되지 않는다.
- **사전등록 갈래:** C와 양·유한 BALL 분모를 구성하지 못했으므로 네 갈래의 비율
  판정에는 진입하지 않았다. 경계나 판독량을 바꾸지 않고
  **`UNRESOLVED-FAIL-CLOSED`를 유지**한다. 이는 네 번째 수치 갈래
  `UNRESOLVED-OUTSIDE-PREREG`를 사후로 대신 적용한 것이 아니라, C 구성 선행 gate에서
  멈춘 것이다.
- **nonpositive 인구 0/음수:** 현재 2단 JSON은 합집합 28,949만 보존하고 값별 분리를
  보존하지 않았다. 따라서 0행 수, raw 음수행 수, 그중 writer의 `-1` 미정의 sentinel과
  실제 솔버 음수의 분리는 **현재 산출물에서 정의 불가**다. 이를 대신 추정하지 않았다.
  `scripts/uv_t2n9_offline.py --forensics-only`가 세 분류를 별도로 출력하도록 보강했다.
- **N9 판정:** s>=5 셀의 `rate_shape_replaced`는 `34304/45000 =
  0.7623111111111112`; shell 8 BALL 방출에너지 분율은
  `0.9956303809148374`, s>=5의 B1–B4는 모든 셸에서 정확히 `1.0`이다. 선 성분은
  `eta_rate_line=chi_line_th*B_nu(T_e)`를 최대 상대오차
  `2.220446049250313e-16`, 1 ULP로 만족한다. 따라서 해당 셀에서는 population-native
  `eta_line`이나 형광 행렬을 바꾸어 비열적/형광 선원 형상을 새길 수 없다. 인구·원자
  데이터는 `chi_line_th`와 처분/수송을 바꾸는 경로로는 형상을 바꿀 수 있으므로
  “모든 개입이 무효”로 확대하지 않는다. 직접 선원 형상을 바꾸려면 EPAY 처분·정규화,
  `T_e`, bf 항 또는 수송 연산자를 바꾸는 별도 개입이 필요하다.
- **추가 실행 명령 한 줄:** 아래 §7의 forensic-only 명령 하나다. 수송·GPU·모델은
  실행하지 않는다.

## 1. 입력 자격과 2단 상태

기존 산출물은 다음 사전등록 자격을 통과한 상태다.

| 항목 | 판독 |
|---|---:|
| LINEPOP SHA-256 | `84d1849dafd1c796dac77c4037b19683e3ef1d5ddb72dd0e6bf701490b05a1cc` |
| iteration / field generation | `10 / 10` |
| 격자 / 셸 | `50×1000`, 선택 셸 `[0,8,16,20,45]` |
| 실제 / 예상 bytes | `137151032 / 137151032` |
| A complete bins | 303 |
| A `chi_line`, `chi_line_th`, `eta_line` | 각 1,515 cells bitwise PASS |
| 음성 대조 | expected FAIL = observed FAIL |
| clamp / fallback / nonfinite | `0 / 0 / 0` |

2단 실행의 최종 상태는 `failure.json`의
`UNRESOLVED-FAIL-CLOSED: C population-native coverage is incomplete; substitution is
forbidden`이다. N9 산출물은 이 실패 전에 완성됐고, C payload·수송표·T2 band table은
생성되지 않았다.

## 2. BALL 구성 판정

### 2.1 28,949행의 BALL 안팎

`t2_C_population_coverage.json`은 전체 1,169,145행 중 다음을 기록한다.

| 항목 | 행 수 |
|---|---:|
| `missing_population_rows` | 28,949 |
| `missing_population_source_rows` | 28,949 |
| `nonpositive_required_rows` | 28,949 |
| `stimulated_adjustment_rows` | 0 |
| `tau_population_bitwise_rows` | 1,140,196 |

writer `src/lumina_cmfgen.c:777-804`는 `tau_used>1e-12`인 선을 재생하다가 선 주파수가
캡처 경계 `nu_lo..nu_hi`, 즉 600–3000 Å 밖이면 행 기록 전에 `continue`한다. 고정
헤더와 manifest도 동일 창을 선언한다. 따라서 행 테이블에 존재하는 모든 행의 선
파장은 BALL 안이다. 특히 nonpositive 28,949행은

```text
BALL rows         = 28,949
outside BALL rows =      0
```

이다. 이는 band 경계를 새로 고른 결과가 아니라 동결된 BALL과 writer의 기록 술어를
직접 합친 결과다.

### 2.2 기여 몫과 상한

해당 행의 recorded-A 기여 `w`는 유한·비음수라는 gate를 이미 통과했으므로, 그 행들이
recorded-A BALL `chi_line`에 차지하는 분율의 현재 인증 상한은 **1.000000**이다. 그러나
이는 유용한 조밀 상한이 아니라 0≤부분합≤전체합에서 오는 보편 상한이다. 기존 JSON은
28,949행의 `w`, shell, bin을 집계하지 않아 더 조밀한 숫자를 보존하지 않았다.

population-native C 기여의 유한 상한은 **정의되지 않는다**. 필요한 `n_lower`,
`n_upper`, `S_l_pop` 자체가 비양수/미정의이므로 recorded A의 `tau_used`나 `w`를 C에
넣으면 금지된 대체가 된다. 이 보고서는 A 기여분율과 미관측 C 기여를 혼동하지 않는다.

보강한 forensic 모드는 exact frequency-edge BALL weights로 선택 셸마다

```text
sum(w_bad * BALL_overlap[bin]) / sum(chi_line_A * BALL_overlap[bin])
```

을 계산하고, 그중 최댓값을 숫자 상한으로 쓴다. 출력은
`t2_nonpositive_population_forensics.json`, 파장·셸 표는
`t2_nonpositive_wavelength_shell.csv`, 원소·이온·준위 순위는
`t2_nonpositive_level_rank.csv`다. 이 1회 판독 전에는 더 정밀한 숫자를 쓰지 않는다.

### 2.3 네 갈래 적용

사전등록 순서는 그대로다.

1. `abs(C/A-1)<=0.05`: operator-only
2. 위가 아니고 `C/CMFGEN>3`: assembly+operator
3. 위가 아니고 `1/3<=C/CMFGEN<=3`: assembly-only
4. 나머지/양·유한 분모 부재: `UNRESOLVED-OUTSIDE-PREREG`

이번에는 C 자체가 구성되지 않아 `C/A`와 `C/CMFGEN`가 모두 정의되지 않았다. 따라서
수치 네 갈래보다 앞선 데이터 자격 gate에서 **`UNRESOLVED-FAIL-CLOSED`**다. 5%,
factor-3, BALL band-mean 어느 것도 변경하지 않았고 B0–B4 다수결이나 B2 대체값도
사용하지 않았다.

## 3. nonpositive 28,949행의 정체

### 3.1 현재 확정 가능한 것과 불가능한 것

현재 JSON은 네 필드
`tau_from_pops`, `n_lower`, `n_upper`, `S_l_pop` 중 하나 이상이 비양수인 행의 합집합만
기록했다. 값별 0/음수 histogram은 없다. 따라서 아래는 현재 상태다.

| 요구 판독 | 현재 판정 |
|---|---|
| 비양수 합집합 | 28,949행, 확정 |
| 0인 population 행 | 정의 불가 — 미집계 |
| raw 음수 population 행 | 정의 불가 — 미집계 |
| 실제 솔버 음수 population 행 | 정의 불가 — `-1` sentinel과 미분리 |
| 원소·이온·준위 순위 | 정의 불가 — 미집계 |
| B0–B4 및 선택 셸 분포 | 정의 불가 — 미집계 |

writer는 `src/lumina_cmfgen.c:811-840`에서 NLTE level lookup이 실패하면 해당
`n_lower` 또는 `n_upper`에 **`-1.0` sentinel**을 기록한다. 그러므로 raw 음수를 곧바로
솔버 음수라고 부르면 안 된다. 실제 솔버 결함 판정은 `nlte_lower/upper>=0`인데 대응
population이 `<0`인 행만 세어야 한다. 추가한 모드는 다음을 각각 분리한다.

- `raw_negative_population_rows`
- `raw_zero_population_rows`
- `undefined_minus_one_sentinel_rows`
- `actual_solver_negative_population_rows`
- `solved_zero_population_rows`
- `zero/negative S_l_pop`, `tau_from_pops` 부호별 수

어느 수에도 floor나 0 대체를 적용하지 않는다.

### 3.2 `tau_used`가 유한한 이유

LINEPOP writer의 데이터 흐름은 다음과 같다.

1. `src/lumina_plasma.c:2582-2681`의 bulk/nebular opacity writer가 이온 밀도,
   `T_rad`, dilution `W`, Boltzmann level population, `f_lu`, `lambda`로
   `opac->tau_sobolev`를 먼저 만든다.
2. `src/lumina_plasma.c:16987-17079`의 NLTE writer는 line map과 양쪽 NLTE level
   mapping이 모두 있을 때만 population-native tau/source를 덮어쓴다. lookup이 없으면
   기존 bulk tau가 남을 수 있다. 이번 캡처의 `LUMINA_NLTE_SKIP_Z`는 빈 값이므로
   SKIP_Z 보존을 이 건의 일반 설명으로 사용하지 않았다.
3. `src/lumina_cmfgen.c:777`은 이 이미 조립된 `opac->tau_sobolev`를 `tau_used`로
   읽고, `:783-784`에서 expansion-opacity `w`를 계산한다.
4. 이후 `:811-840`에서 별도로 NLTE population mapping을 시도한다. mapping이 없으면
   `n=-1`, `tau_from_pops=-1`인 반면 `tau_used`와 `w`는 앞 단계 bulk 값으로 유한할 수
   있다. `:865-872`가 둘을 나란히 직렬화한다.

즉 “population이 비양수인데 opacity가 유한”은 산출물 대체가 아니라 **두 writer
권위가 다른 상태**를 같은 행에 기록한 결과다. 다만 28,949행이 전부 이 sentinel
경로인지, 일부가 실제 zero/solver-negative인지의 수치 판정은 forensic 출력 전까지
미정이다.

### 3.3 발견 판정

현재 확정 가능한 결함은 “C에 필요한 population-native 권위가 28,949 BALL 행에서
없는데, 생산 조립 권위의 유한 `tau_used`는 존재한다”는 계기/권위 불일치다. 솔버 음수
결함은 아직 선고하지 않는다. 수리안은 본 과업 범위 밖이다.

## 4. N9 확정 판정문

전역 disposition은 `legacy_source=5000`, `thick_exempt=10696`,
`rate_shape_replaced=34304`, `scalar_rescaled=0`으로 manifest와 정확히 일치한다. s>=5
45,000셀 중 rate-shape는 `34304/45000=0.7623111111111112`다. 에너지 정의는 사전등록한
`eta_fixed_post_EPAY * exact_frequency_overlap * shell_volume`이고 coherent 반환은
제외됐다. shell 8 BALL은 `0.9956303809148374`가 rate-shape 셀에서 나오며, s>=5의
B1–B4 에너지 분율은 모든 셸에서 `1.0000`이다. 참고로 EPAY 적용 전인 s0–4는 이
문장에 포함되지 않으며, 전 셸을 합친 BALL 분율은 `0.1793726021322533`이다.

선 성분 직접 재생은
`eta_rate_line=chi_line_th*B_nu(T_e)`,
`S_rate_line=eta_rate_line/chi_line_th` 항등식을 최대 상대오차
`2.220446049250313e-16`, 최대 1 ULP로 만족해 사전등록 한계 `2^-48`, 8 ULP를
통과했다. 양성 셀은 shell 8/16/20/45에서 각각 301/300/302/304개였고 상태는 PASS다.
clamp, fallback, nonfinite는 모두 0이며, 1-bit 음성 대조는 expected FAIL = observed
FAIL이다.

**판정문.** parity59의 s>=5, 특히 shell 8 BALL과 B1–B4에서는 UV fixed 선 방출의
실질 전부가 구성상 `chi_line_th*B_nu(T_e)`로 재형상된다. 그러므로 population-native
`eta_line=A_ul n_u`를 바꾸거나 형광 행렬을 바꾸는 개입은 이 셀의 비열적/형광 선원
형상을 바꿀 수 없다. 원자데이터·인구 개입은 `chi_line_th`, opacity, EPAY 처분과
정규화, 그리고 그 뒤 수송을 바꿀 수 있으므로 UV 결과 전체에 “아무 효과도 없다”고
확대할 수는 없다. 그러나 **EPAY 뒤 선원 형광 형상 자체**를 바꾸는 세 개입 중
population-native emissivity와 형광 행렬에는 유효 경로가 없고, 원자데이터/인구는
opacity 경로만 남는다. 형광 선원 형상을 직접 바꾸려면 EPAY rate-shape 재조립 자체,
그 정규화 장부, `T_e`/bf 항 또는 수송 연산자를 별도 단일인자로 다뤄야 한다.

## 5. 대장 기재용 문안 2건

`docs/VERIFICATION_REGISTERS.md`는 직접 수정하지 않았다. 현재 형식상 두 건 모두
“B. 검증 대상 — 정합성”에 넣을 후보이며, 아래 문안을 그대로 옮길 수 있다.

### B 후보 — EPAY-REPLAY-001

> **▲EPAY-REPLAY-001 — LCMFLP01-v1 EPAY 정규화 재생 불능(계기 결함):** payload는
> per-cell disposition은 보존하지만 rate-shape/scalar EPAY의 셸별 정규화 `wn`과 그
> 입력 `acc_abs`, `acc_dep`, `acc_w`를 직렬화하지 않는다. `acc_abs`는 assemble 당시
> lagged J로 계산되고 dump 전 solve/damping 뒤 J와 동일하지 않으므로 같은 세대
> payload만으로 population-native opacity 반사실의 EPAY scale을 exact 재조립할 수
> 없다. 근거=`n9_summary.json: epay_scale_not_reproducible=true`, disposition
> `5000/10696/34304/0`, clamp/fallback/nonfinite `0/0/0`; 재현=
> `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_t2n9_offline/n9_summary.json`.
> 처분=scale과 세 누산량의 per-shell 직렬화가 필요한 계측 부채(수리안 채택은 별도).

### B 후보 — LINEPOP-POPAUTH-001

> **▲LINEPOP-POPAUTH-001 — BALL population 권위 결손:** LCMFLP01-v1의 선택
> 1,169,145행 중 28,949행은 `tau_from_pops/n_lower/n_upper/S_l_pop` 요구량이
> 비양수이고 population/source flags도 미정의다. writer 기록 술어상 이 행들은 전부
> 600–3000 Å BALL 안(`outside=0`)인데 생산 `tau_used`/`w`는 별도 bulk/nebular
> opacity 권위에서 유한할 수 있어, population-native C를 대체 없이 구성할 수 없다.
> T2=`UNRESOLVED-FAIL-CLOSED`; 고정 네 갈래 미진입. 0/음수·`-1` sentinel·실제 솔버
> 음수 분리와 원소/이온/준위 순위는
> `t2_nonpositive_population_forensics.json` 1회 판독으로 완결하며, 그 전에는 솔버
> 음수 결함을 선고하지 않는다. 근거=`t2_C_population_coverage.json`, writer
> `lumina_cmfgen.c:777-872`, bulk/NLTE writers `lumina_plasma.c:2582-2681,
> 16987-17079`; 처분=권위 결손 발견 등록, 수리안은 별도 과업.

두 번째 문안은 추가 판독 뒤 `0/음수/sentinel/solver-negative` 실제 수와 상위 순위를
괄호 안에 덧붙여야 완전 기재가 된다. 현재 수를 추정해 채우지 않았다.

## 6. 스크립트 보강과 경량 검증

`scripts/uv_t2n9_offline.py`에 `--forensics-only`를 추가했다. 이 모드는 다음만 한다.

- 기존 137,151,032 B LINEPOP의 header/sidecar/SHA/배치 gate 재확인
- 28,949행의 raw zero/raw negative/`-1` sentinel/actual solver-negative 분리
- 원소·이온 및 NLTE level index·통계중량·에너지 조합 순위
- B0–B4/BALL 안팎과 선택 셸 분포
- recorded-A BALL `chi_line` 영향분율의 exact-overlap 셸별 값과 수치 상한
- `tau_used`, `tau_from_pops`, `w` 부호·유한성 장부

수송 driver compile/실행, CMFGEN 결과 재계산, GPU, 모델은 호출하지 않는다. 분석에
clamp, floor, fallback, 대체를 추가하지 않았고 결과 JSON에 네 카운터를 0으로 고정해
위반 시 정상 결과로 보이지 않게 했다. Python bytecode 문법 검사와 합성 self-test는
PASS했다. 합성 검사는 header 152 B, row 76 B, line 80 B, 반복 직렬화 항등 및 1-bit
결함의 expected FAIL을 확인했다. 큰 LINEPOP 판독은 이 세션에서 실행하지 않았다.

## 7. 추가 실행 명령 — 한 줄

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && python3 scripts/uv_t2n9_offline.py --forensics-only --linepop /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/linepop_iter10 --outdir /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_t2n9_offline
```

이 명령의 산출물이 생겨도 BALL 안에 affected row가 하나라도 있으면 T2 상태는 그대로
`UNRESOLVED-FAIL-CLOSED`다. 경계 완화나 A 값 대체로 C를 만들지 않는다.

## 8. 준수 사항

- `validation/uv_t2n9/PREREG.md` 미수정
- `src/` 미수정
- GPU/모델/CMFGEN/Stage-3.1 수송 미실행
- 무거운 LINEPOP 재판독 미실행
- clamp/floor/fallback/대체 미도입
- `docs/VERIFICATION_REGISTERS.md` 미수정
- git commit 미수행
