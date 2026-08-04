# 상시 회귀 계측 하니스

## 결과

`scripts/regression_ledger.py`가 런 디렉터리 하나 이상을 오프라인으로 읽어 런당 JSON 1행을
`validation/regression_ledger/ledger.jsonl`에 추가한다. Lumina 실행 파일, 새 모델, GPU를
호출하는 코드는 없다. 대장은 `O_APPEND` + 배타 잠금 + `fsync`로만 쓰며 기존 바이트를 여는
수정 경로가 없다. 같은 `run_path`의 기존 행이 있으면 새 행에 `recomputed_at`과
`prior_measurement_count`를 넣는다.

이번 작업에서는 실데이터 백필을 실행하지 않았다. 현재 파일명 census만 한 가벼운 inventory는
60개 `logs/coevolve_consume_a10_kx_*` + 9개 scratch = 69개이며, levelpop은 logs 53개 +
scratch 5개 = 58개다. 5개 scratch 런에는 levelpop과 census가 모두 있다. 58개 levelpop 약
4 GB의 전수 판독과 5개 실런 census 대조는 로그인 노드 금지 규율에 따라 계산 노드 백필에
남겼다.

## 대장 스키마

최상위 필드는 다음과 같다.

| 필드 | 내용 |
|---|---|
| `ledger_schema_version` | 현재 `1`; 정의 변경 시 올리고 옛 행은 보존 |
| `run_path` | 절대·정규화 런 경로 |
| `run_kind` | `logs`, `scratch`, 판별 불가 시 `UNDEFINED` |
| `run_directory_mtime` | epoch seconds와 UTC ISO-8601 |
| `binary_identifier` | 인증된 run-local/log sha256 또는 이름+sha 불가 사유 |
| `gate_set` | 실제 footer 우선, 차선 `.env`; 정렬 `[key,value]`, 개수, SHA-256, 출처 |
| `measured_at` | UTC ISO-8601 |
| `recomputed_at` | 동일 `run_path`의 선행 행이 있을 때만 새 행에 존재 |
| `prior_measurement_count` | 재계측 전 같은 런의 행 수 |
| `metric_definitions` | 아래 8개 동결 정의 전문 |
| `metric_definitions_sha256` | 정렬된 정의 객체의 SHA-256 |
| `model_geometry` | 셸 반경·중간속도의 권위 파일 |
| `cmfgen_oracle` | EDDFACTOR/MEANOPAC 디렉터리와 표준 진리 디렉터리 |
| `input_inventory` | 필수/선택 입력별 `PRESENT`/`MISSING`, 크기, mtime |
| `metrics` | 아래 8개 객체; 계산 불가는 `status=UNDEFINED`와 `reason` |

모든 행은 strict JSON이다. NaN/Infinity, 수치 floor/cap/clamp, 가짜 0, 대체 oracle은
허용하지 않는다. 부분 산출이 가능하면 metric은 `PARTIAL`이고, 해당 하위 값만
`UNDEFINED`다. 런 자체는 건너뛰지 않는다.

## 동결한 8개 정의 문자열

1. `uv_quotient`

   `formal UV fraction = integral_[500,3500) F_lambda d_lambda / integral_[500,20000) F_lambda d_lambda from lumina_spectrum_formal.csv; trapezoids join the native bin-center samples selected inside each half-open wavelength band; no boundary extrapolation or invented endpoint is used`

2. `electron_temperature`

   `per shell: Lumina T_e from lumina_plasma_state.csv; CMFGEN truth is the TIME=19.48 d block of data/standart_data1/toy06/phys_toy06_cmfgen.txt linearly interpolated in shell midpoint velocity from model geometry.csv; difference_K=Lumina-CMFGEN and ratio=Lumina/CMFGEN`

3. `radiation_energy_density`

   `per shell and lane mc/cs/CMFGEN: u=(4*pi/c)*integral J_nu dnu over 100<=lambda_A<20000; Lumina uses lumina_coevolve_field.csv and CMFGEN uses EDDFACTOR depth integrals log-linearly interpolated in midpoint velocity; trapezoids join native bin-center samples selected inside the wavelength band and use no boundary extrapolation`

4. `band_energy_ratio`

   `per shell and lane mc/cs/CMFGEN: u_FUV/u_EUV with FUV=[918,1290) A and EUV=[450,918) A, where u=(4*pi/c)*integral J_nu dnu; also record each Lumina band divided by the matching CMFGEN band; trapezoids join selected native bin centers only, and zero denominators are UNDEFINED`

5. `optical_depth`

   `per shell outward tau_es=sum_(j>=shell) n_e[j]*sigma_T*(r_outer-r_inner)[j]; when an LCMFCE01 payload exists, Lumina tau_Ross is the outward sum of local Rosseland harmonic-mean chi_tot at local T_e times shell width; CMFGEN Tau(es) and Tau(Ross) are read from MEANOPAC and linearly interpolated in midpoint velocity`

6. `thermalization_and_clamps`

   `when an LCMFCE01 payload exists, per shell chi_es/chi_tot=(sum chi_es*dnu)/(sum chi_tot*dnu) and epsilon_eff=[sum(eta_fixed/chi_tot)*dnu]/[sum(eta_total/chi_tot)*dnu] over its full frequency grid; clamp firings are the exact sum of recognized run-log counters (FLOORM clamped levels and fine-solver clamped lines), reported by counter family without inferred zeros`

7. `ionization_fractions`

   `per shell, element, and 0-based stage: Lumina fraction=n_ion/sum_stages(n_ion) from lumina_ion_pops.csv; CMFGEN truth is the matching ionfrac_*_toy06_cmfgen.txt TIME=19.48 d fraction linearly interpolated in midpoint velocity; record absolute difference and Lumina/CMFGEN only for a nonzero CMFGEN denominator`

8. `departure_coefficients`

   `b_k=(n_k/n_ground)/[(g_k/g_ground)*exp(-(E_k-E_ground)/(k_B*T_e))]; T_e is the local lumina_plasma_state electron temperature, ground is the first recorded level of the same (Z,0-based ion), and the LTE reference is the same-ion ratio of Saha-Boltzmann populations (the ion's Saha factor cancels); no T_rad, dilution W, g-weighted aggregation, floor, cap, or replacement is used; per (shell,Z,ion), median_bk_unweighted is the ordinary median of all finite recorded b_k values and median_bk_population_weighted is the first sorted b_k whose cumulative n_k reaches >=50% of total n_k; max/min and frac outside [0.1,10] include nonpositive recorded b_k, count_bk_le_0 is explicit, and undefined/negative weights make only the weighted median UNDEFINED`

각 metric 객체에도 같은 `definition`을 다시 박는다. FUV는 발주대로 918–1290 Å,
EUV는 450–918 Å의 반열린 구간이다.

## b_k 이중가중과 census 대조

`median_bk_unweighted`는 각 명시적 level을 1표로 센다. 준위 수가 많은 희박한 고준위의
꼬리를 그대로 보여준다. `median_bk_population_weighted`는 `n_k`를 가중치로 삼아 실제
population 질량의 중앙값을 보여준다. `g`는 level의 LTE 기준을 만드는 식 안에만 있고
집계 가중치로 다시 쓰지 않는다. 따라서 과거 66× g-가중 아티팩트를 재생산하지 않으며,
두 중앙값이 벌어질 때 그 차이 자체가 기록된다. 0 population이면 가중 중앙값에 0의
가중치로만 참여하지만 무가중 중앙값·최솟값·범위이탈률·`count_bk_le_0`에는 기록값 그대로
남는다.

levelpop과 census가 함께 있는 실런 5개는 다음과 같다.

- `chieta_capture_parity59_188605`
- `emiss_ab2_capture_188766`
- `emiss_ab_capture_188747`
- `fluormat_capture_188828`
- `instr_capture_188932`

백필 때 `(shell,Z,ion,level_num)` 키로 census 전 행을 찾아 `E_eV,g,n_k,n_ground,b_k`의
CSV 문자열을 exact 비교한다. 결과는 각 행의
`metrics.departure_coefficients.census_crosscheck`에 `PASS`/`FAIL`, 기대/비교/누락 키 수,
전체 mismatch 수와 앞 20개 예시로 남는다. fixture 경로 결과는 **PASS**다. 실 5런 결과는
아직 **UNRESOLVED — 계산 노드 백필 미실행**이며, 실패 시 하니스는 행을 보존하되 해당
crosscheck를 `FAIL`로 명시한다.

## 자기검사와 음성 대조

실행한 명령:

```bash
python3 scripts/regression_ledger.py --self-test
```

실제 출력:

```text
NEGATIVE CONTROL: FAIL (expected): injected uv_fraction=1.5 -> UV fraction outside [0,1]: 1.5
PASS fixture metrics: all 8 metric objects present and strict-JSON valid
PASS fixture census: levelpop and census paths agree exactly
PASS append-only: first JSONL prefix preserved; recomputed_at added on second measurement
PASS missing-input fixture: row retained; unavailable run-side values are UNDEFINED
PASS payload-only fixture: chi_es/chi_tot and epsilon_eff defined without plasma T_e
PASS --self-test
```

fixture는 formal/plasma/field/ion/levelpop/census, 작은 EDDFACTOR·RVTJ·MEANOPAC,
LCMFCE01+manifest, env footer와 clamp 로그만 임시 디렉터리에 생성한다. append-only 검사는
첫 행의 전체 byte prefix가 두 번째 append 뒤에도 그대로인지 확인한다. 음성 대조는 1번
지표의 `uv_fraction`을 1.5로 변조해 validator가 반드시 FAIL하는 것을 보인다.
빈 런 fixture는 입력이 하나도 없어도 런 행과 8개 metric 객체를 버리지 않고, 없는
run-side 값을 이유 있는 `UNDEFINED`로 남기는지 확인한다(CMFGEN-only 값은 남을 수 있다).
payload-only fixture는 plasma가 없어 Rosseland 경로가 불가능해도 T_e가 필요 없는
`chi_es/chi_tot`·`epsilon_eff`는 정의되는지 확인한다.

## 계산 노드 백필

이미 계산 노드 allocation 안이라면 저장소 루트에서 다음 한 줄이다.

```bash
srun --ntasks=1 bash scripts/backfill_regression_ledger.sh
```

새 Slurm job으로 보낼 예시는 다음과 같다. GPU 요청은 없다.

```bash
sbatch --job-name=lumina-ledger --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G --time=04:00:00 --wrap='cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && bash scripts/backfill_regression_ledger.sh'
```

런처는 `SLURM_JOB_ID`가 없으면 거부하고, allocation 변수만 있고 slurmd/job-step marker가
없어도 `srun`을 요구하며 거부한다. hostname에 `login`이 있으면 다시 거부한다.
`CUDA_VISIBLE_DEVICES`를 빈 값으로 고정한다. 디렉터리만 골라 정확히 69개인지 확인한 뒤
한 Python 프로세스에 모두 넘기므로 143 MB `EDDFACTOR`는 한 번만 읽는다. 예상 개수가
다르면 silent partial backfill 대신 실패한다. 의도적으로 다른 inventory를 재려면 계산
노드에서만 `REGRESSION_LEDGER_EXPECTED_RUNS=<N>`을 명시한다.

## 남은 UNRESOLVED

- 실 69런 백필과 5런 census exact 대조: 계산 노드에서 위 명령 실행 대기.
- 옛 a10 런에는 LCMFCE01 `chi_tot/chi_es/eta` payload가 없다. 해당 런의 Lumina
  `tau_Ross`, `chi_es/chi_tot`, `epsilon_eff`는 surrogate 재구성 없이 이유와 함께
  `UNDEFINED`; `tau_es`, CMFGEN 두 tau, clamp counter는 계속 기록된다.
- stdout/footer에 인증된 binary SHA가 없고 실행 파일이 run-local로 보존되지 않은 옛 런은
  현재 workspace binary를 소급 해시하지 않는다. 실행 파일 이름과 “historical artifact가
  없어 sha 불가” 사유를 남긴다.
- 알려진 두 clamp 계기(FLOORM, fine line-source) 외의 cap/floor가 숫자 counter를 인쇄하지
  않은 런은 0으로 추정하지 않고 해당 counter family를 기록하지 않는다.
- 표준 CMFGEN ion 지상진리는 파일이 있는 Ca/S/Si/Fe/Co/Ni만 정의된다. 없는 원소는
  `UNDEFINED`이며 다른 원소로 대체하지 않는다.
