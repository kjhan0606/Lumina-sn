# A2-03 구현 보고 — canonical `RadiationField` CPU shadow

- 구현일: 2026-08-05
- 기준 커밋: `bca476a7b675790cf393871b860b785a9ebd7860`
- 범위: A-2의 A2-03 하나만. A2-04 소유권 이관, A2-06 line-cache 생산,
  A2-12 GPU 미러는 하지 않았다.
- gate: `LUMINA_RADFIELD_SHADOW`, 기본값 **OFF**. 선택적 출력 전용 gate는
  `LUMINA_RADFIELD_SHADOW_DUMP=<path>`다.
- 판정: 구현·빌드·정적 호출그래프·마이크로 생산경로 parity는 PASS.
  production 2000×2 parity와 D/K 등 운전석 전량 회귀는
  `PENDING_DRIVER_EXECUTION`이며 이 문서는 이를 production PASS로 주장하지 않는다.

## 1. 정본 구조와 A2-02 결박

정의 위치는 `src/radiation_field.h:89-102`다. 직접 멤버는 아래 10개뿐이며 정본
발주서 §2.1과 이름까지 1:1이다.

| 순서 | §2.1 이름 | 코드 멤버 | 구현 의미 |
|---:|---|---|---|
| 1 | `shell_boundaries` | `shell_boundaries` | 속도 경계 `[cm s^-1]`, `n_shells+1` |
| 2 | `frequency_bin_edges` | `frequency_bin_edges` | amended-union log edge, 4001개 |
| 3 | `J_nu[shell][bin]` | `J_nu` | 2차원 4000-bin 빈 평균 |
| 4 | `units` | `units` | `erg s^-1 cm^-2 Hz^-1 sr^-1` 고정 enum |
| 5 | `frame` | `frame` | 셸 공이동계 고정 enum |
| 6 | `epoch` | `epoch` | 폭발 후 초 단위 epoch |
| 7 | `generation` | `generation` | `required_generation/computed_generation` |
| 8 | `provenance` | `provenance` | 생산자·union/edge hash·기여 원장 통계 |
| 9 | `validity` | `validity` | `VALID/EXACT_ZERO/UNSAMPLED/OUT_OF_GRID/STALE` |
| 10 | `estimator_count_or_variance` | `estimator_count_or_variance` | 이번 MC lane은 실제 path-estimator 기여 count |

크기 정보는 멤버를 늘리지 않고 axis/grid/statistics 타입 내부에 둔다. amended A2-02
결박은 `src/radiation_field.h:7-15`다.

- `N=4000`
- `nu_min=1.4402928950097124e12 Hz`
- `nu_max=4.032418413741097e16 Hz`
- union SHA-256 `1443c069...1321c`
- 4001-edge SHA-256 `ec3f94d9...8f76`

이는 `validation/a2_02c/A2_02C_FREQUENCY_UNION.json`과
`validation/a2_02c/a2_02c_global_resolution_result.json`의 확정값이다. 192,922/196,510
fine 진단 격자는 런타임 shape로 사용하지 않았다.

`LineJbarCache` holder는 `src/radiation_field.h:104-129`, 정본과 holder를 함께 소유하는
컨테이너는 `src/radiation_field.h:139-144`다. holder에는 generation, shell/line/profile
ID와 profile hash, value, validity, sample count, variance/standard error, `Q_g` hash,
units/frame/provenance 자리가 있다. A2-03에서는 비할당·비생산 상태이며 독립 setter나
조회 API가 없다. production 잡음 자산 **median 12% @ sample_count 66**은 A2-06 damping과
통계 게이트의 입력으로 회귀 대장에 인계했다.

## 2. 생산 지점 병행 배선

이 세션에서 `nl -ba`로 다시 확인한 위치다.

| 파일:행 | 역할 |
|---|---|
| `src/lumina_main.c:286` | NLTE와 함께 shadow lifecycle을 초기화. OFF면 할당 0 |
| `src/lumina_main.c:396` | 세대 `iter+1` required 선언, 속도 shell edge와 epoch 결박 |
| `src/lumina_main.c:427` | CPU thread-local 4000-bin accumulator 생성 |
| `src/lumina_transport.c:115-120` | 기존 1000-bin estimator를 기록한 같은 `update_base_estimators()`에서 공이동 `epsilon*dl`을 shadow에도 기록 |
| `src/lumina_main.c:477` | thread-local raw/count를 owner accumulator로 reduce |
| `src/lumina_main.c:485` | 기존 경로에 손대지 않고 shadow만 validate/commit |
| `src/radiation_field.c:259-309` | `raw/(4*pi*V*dt*Delta_nu)`로 빈 평균 정규화, validity/count commit |
| `src/radiation_field.c:312-350` | 명시적으로 요청된 shadow CSV만 읽는 출력 전용 진단 |

중심 주파수의 `J(nu_center)`는 계산에 등장하지 않는다. 누적량은 기존 CPU 수송이 만든
셸 공이동계 path-length measure이고, commit에서 정확한 edge 차 `Delta_nu`로 나눈다.
따라서 기록값은 §2.1/§6.2의 빈 평균이다. observer-frame 주파수나 observer spectrum은
shadow producer에 들어오지 않는다.

`generation`은 K-FRESH와 같은 두 단계다. 초기값은 required=1/computed=0이고 전 빈은
STALE다(`src/radiation_field.c:52-53,90`). 세대 시작은 required만 전진시키고 STALE로
돌린 뒤(`:109-155`), owner 검증이 끝난 commit만 computed=required로 공개한다
(`:302-309`). 다음 세대가 직전 computed+1이 아니면 실패한다.

MC count는 `(shell, canonical-bin)`에 더해진 실제 path-estimator 항 수다
(`src/radiation_field.c:185-207`). count=0인 셀은 `J_nu=0`을 유지하되 의미는
`UNSAMPLED`; count>0/raw=0은 `EXACT_ZERO`; count>0/raw>0만 `VALID`다. historical
`src/lumina_plasma.c:14683-14688`의 `nlte->J_nu=1e-30` 배열은 “기존 결과 불변” 때문에 이 단계에서 제거하지 않았지만,
shadow는 그것을 읽거나 복사하지 않는다. 제거/소유권 이관은 A2-04다.

## 3. 소비자 무접촉 증명

단순 문자열 검색 대신 `scripts/a2_03_callgraph_audit.py`가 8개 production translation
unit을 GCC `-fdump-ipa-cgraph`로 컴파일해 호출자 집합을 판정한다. 결과는
`validation/a2_03/a2_03_callgraph_audit.json`이다.

- 유일한 data producer call: `update_base_estimators -> radiation_field_accumulator_add`
- lifecycle/reduce/commit call: `main`만
- owner validation과 dump: commit 내부만
- rate/population/opacity/emissivity/transfer physics caller: **0**
- 공개 get/read/query/lookup/sample/consume API: **0**

따라서 ON에서도 물리 소비자는 shadow를 읽지 않는다. dump는 §2.2의 출력 전용 진단이며
`LUMINA_RADFIELD_SHADOW_DUMP`를 별도로 설정해야만 열린다.

## 4. validity와 음성대조

`tests/a2_03_radiation_field_selftest.c`와 owner validator
`src/radiation_field.c:225-257`가 다음을 검사한다.

| §13 경로 | 주입 | 기대·관측 |
|---:|---|---|
| 9 | committed field의 frame을 observer로 변경 | owner validation 거부, PASS |
| 10 | sampled positive cell을 `UNSAMPLED`로 바꿈; exact-zero와 missing을 교환 | count/value/validity 불변식 위반으로 거부, PASS |
| 11 | unsampled cell에 `1e-30` 삽입 | `count=0 => value==0` 위반으로 거부, PASS |

추가로 required/computed 불일치, amended union/edge hash, 단위, bin 수를 fail-closed로
검사한다. self-test marker는 다음이며 정상 rc는 0이다.

```text
A2_03_RADIATION_FIELD_SELFTEST PASS negative_9=PASS negative_10=PASS negative_11=PASS fields=10 bins=4000
```

## 5. 기존 결과 불변 검증

`tests/a2_03_producer_parity_fixture.c`는 production 함수
`update_base_estimators()`에 동일한 공이동 path-length 항을 넣는다. OFF와 ON을 Pool
2청크로 동시에 실행하고 `scripts/a2_03_byte_parity.py`가 stdout, stderr, 기존 estimator
binary를 전부 SHA-256 비교한다. ON의 shadow CSV 하나만 allowlist다. 러너는 진행률을
stderr로 내며 아래처럼 `tee`로 보존한다.

관측 결과(`validation/a2_03/a2_03_byte_parity.json`):

```text
off_rc=0 on_rc=0 compared_files=3 differing_files=0
guard_hits=0 fallback_hits=0
unsampled_bins=3998 unsampled_nonzero_bins=0 historical_1e-30-values=0
verdict=PASS
```

이는 마이크로 생산경로 증거다. production 2000×2 전량 parity는 아래 lageunha 명령의
운전석 실행 전까지 `PENDING_DRIVER_EXECUTION`이다.

## 6. 운전석 복사 명령과 기대 rc

### 6.1 grammar-debug: schema/callgraph/fixture와 앞 단계 경량 gate

전체 블록 기대 rc는 **0**이다. 각 하위 명령도 0이어야 한다.

```bash
ssh grammar-debug 'bash -s' <<'EOF' 2>&1 | tee /tmp/a2_03_grammar_debug.log
set -euo pipefail
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
make selftest_a2_03_radiation_field selftest_a2_03_producer_parity_fixture
./selftest_a2_03_radiation_field
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_03_callgraph_audit.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_03_byte_parity.py \
  --binary ./selftest_a2_03_producer_parity_fixture --fixture \
  --parallel --progress-seconds 1
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_01_census_contract.py check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_01_read_trace.py selftest
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_01_oracle_compat_selftest.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_capture_gate_selftest.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_frequency_union.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02_prepare_fine_dump.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02_resolution_ladder.py self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_segment_replay.py self-test
EOF
```

기대 marker는 fields=10/bins=4000/negative 9·10·11 PASS, callgraph
`physics_consumer_callers=[]`, parity differing=0, A2-01 157/157, A2-02C
capture/union/prepare/ladder/replay PASS다.

### 6.2 lageunha: CUDA header syntax, production parity, 전 회귀

production parity 두 lane은 10분 초과 판정 도구이므로 runner 자체가
`ThreadPoolExecutor(max_workers=2)` 청크 Pool을 쓰고 30초마다 진행률을 낸다. `tail`
pipe를 쓰지 않고 `tee`로 전 로그를 보존한다. 고정 RNG의 packet-to-thread 배정을
보존하려고 각 lane은 `OMP_NUM_THREADS=1`이다. 전체 블록 기대 rc는 **0**이다.

```bash
ssh lageunha 'bash -s' <<'EOF' 2>&1 | tee /tmp/a2_03_lageunha.log
set -euo pipefail
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda
nvcc -std=c++14 -x cu -Isrc -c \
  tests/a2_03_radiation_field_selftest.c \
  -o /tmp/a2_03_cuda_header_syntax.o
gcc -O2 -Wall -Wextra -std=c11 -o /tmp/lumina_a2_03 \
  src/lumina_main.c src/lumina_transport.c src/a2_02c_segment_capture.c \
  src/radiation_field.c src/lumina_plasma.c src/lumina_element_wide.c \
  src/lumina_atomic.c src/lumina_cmfgen.c -lm
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_03_byte_parity.py \
  --binary /tmp/lumina_a2_03 \
  --data data/tardis_reference_toy06_19p48d_sivcaiv_ftos \
  --packets 2000 --iterations 2 --parallel --progress-seconds 30
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_00_oracle_negative_controls.py \
  --scratch-root /tmp
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_ne_naming_controls.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_deck_fossil_controls.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_config_prec_negative_controls.py \
  --binary /tmp/lumina_a2_03
bash scripts/run_zinert_selftest.sh
/home/kjhan/.lumina_scratch/run_dbuild_gates.sh
/home/kjhan/.lumina_scratch/run_cls_verify.sh
EOF
```

기대 marker는 production parity `verdict=PASS`, differing=0, guard/fallback=0;
A2-00 7/7; NE 5/5; DECK 5/5; CONFIG-PREC 7/7; Z-INERT 전항; D 19/19;
K 7/7; classifier 7/7이다. CUDA는 syntax만 수행하며 device allocation/upload/mirror는
없다. `/usr/bin/time`과 `/gpfs` 쓰기는 없다.

## 7. §11 회귀 대장

정확히 한 행은 `validation/a2_03/a2_03_regression_ledger.json`에 있다. 구현 source 묶음
SHA-256은 `5a2d4922...7fa7f`, A2-02 입력 결합 hash는
`a4c6f261...cced7`이다. local exit status는 0, allowlist는
`radiation_field_shadow.csv` 하나, guard/fallback은 0이다. `driver_signoff`는 전량 명령이
실행되기 전까지 `PENDING_DRIVER_EXECUTION`으로 남겼다.

## 8. 비변경선, 위험, A2-04 인계

- 기존 `NLTE_N_FREQ_BINS=1000`, `nlte->J_nu`, `(W,T_rad)`, `bf_rate_estimator`,
  `jbar_line/j_blue`, floor/cap 및 §2.3 제거 대상은 삭제·대체하지 않았다.
- 덱, `data/`, `/gpfs`, CUDA source/device 배열을 변경하지 않았다. commit/push도 하지 않았다.
- `src/lumina_main.c`는 A2-01의 물리 줄번호 결박을 유지하려고 총 874행과 기존 census
  위치를 보존했다. 이 세션의 A2-01 census 157/157 및 trace self-test가 PASS했다.
- CPU MC producer만 실제 4000-bin shadow 값을 commit한다. pure-CMFGEN의 현 `cs.J`
  단순 복사는 bin-center/legacy-grid 양을 정본 bin average로 가장할 위험이 있으므로 하지
  않았다. A2-04는 MC와 pure-CMFGEN을 한 원자적 commit API로 생산하고 L-0 replay로
  이 공백을 닫아야 한다.
- A2-04는 legacy `1e-30` floor overwrite를 제거하되, 이번 shadow의
  `UNSAMPLED/EXACT_ZERO/count` 의미를 그대로 정본으로 승격해야 한다. missing을 0으로,
  0을 missing으로 바꾸면 안 된다.
- A2-04 dual-view commit은 `RadiationField`와 아직 빈 `LineJbarCache` holder의 generation을
  함께 원자적으로 전진시킬 schema를 유지해야 한다. 실제 cache 생산은 A2-06이다.
- 60 CPU thread에서 ON의 thread-local raw+count 메모리는 약 192 MB다. OFF는 추가 할당
  0이다. production driver는 memory/elapsed를 로그에 남겨야 한다.
- A2-12 전에는 GPU가 shadow를 보거나 업로드하면 계약 위반이다.

## 9. 이 세션의 검증 요약

```text
CPU full build                         PASS (기존 warning만)
CUDA header syntax                    PASS (nvcc 13.0.2)
RadiationField self-test              PASS
compiler callgraph 8/8 TU             PASS, physics consumer 0
fixed-output producer parity          PASS, 3 files byte-identical
A2-00 negative controls               PASS 7/7
A2-01 census / trace / oracle compat  PASS
A2-02C capture/union/prepare/ladder/replay self-tests PASS
NE / DECK controls                    PASS 5/5 / 5/5
git diff --check                      PASS
production parity + D/K/Z/CONFIG/CLS PENDING_DRIVER_EXECUTION
```
