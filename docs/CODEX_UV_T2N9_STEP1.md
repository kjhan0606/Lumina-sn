# UV T2/N9 Step 1 — 오프라인 소비기와 사전등록

작성일: 2026-08-02 (Asia/Seoul)  
단계 상태: **SCRIPTED + PREREGISTERED / HEAVY MEASUREMENT NOT RUN**

## 0. 결론

차터의 1단계 산출물 두 개를 작성했고 측정 경계는 실제 payload 판독 전에 동결했다.

- 분석 소비기: `scripts/uv_t2n9_offline.py`
- 사전등록: `validation/uv_t2n9/PREREG.md`

생산 `src/`는 수정하지 않았고, 137 MB LINEPOP 파싱·재조립, Stage 3.1 수송,
GPU/모델/CMFGEN 계산을 실행하지 않았다. git commit도 하지 않았다. 스크립트는 기존
CMFGEN 결과를 읽고 기존 CPU formal operator를 호출하도록만 작성했다.

합성 `--self-test`는 수십 KB 이하이지만 로그인 노드 실행 금지 때문에 로컬에서 돌리지
않았다. 규약 경로인 grammar-debug 접속은 첫 시도에서
`Bad owner or permissions on /etc/ssh/ssh_config.d/50-redhat.conf`, `-F /dev/null`
재시도에서는 sandbox의 `socket: Operation not permitted`로 **Python 실행 전에** 차단됐다.
따라서 이 단계의 검증 상태는 정적 판독이며, 문법/합성 자기검사는 아래 grammar-debug
명령으로 운전석이 수행해야 한다.

## 1. LCMFLP01 레코드 배치 — writer 확정

배치는 문서나 크기에서 역추정하지 않고 `src/lumina_cmfgen.c` writer를 권위로 삼았다.

### 1.1 endian, 고정 크기, 레코드 필드

- 513–523행: `u32`, `i32`, `f64`를 명시적으로 **little-endian** byte로 pack한다.
- 525–526행: 행은 76 B, 선 정적 레코드는 80 B다.
- 939–964행: magic `LCMFLP01`, endian marker `0x01020304`, version 1과 고정 헤더를
  쓴다. 고정 헤더는 152 B다.
- 843–860행: 선 정적 레코드의 실제 offset과 타입을 쓴다.
- 862–873행: 행 레코드의 실제 offset과 타입을 쓴다.

| 구역 | offset | field/type |
|---|---:|---|
| line static | 0 | `line_id u32` |
|  | 4 | `bin u32` |
|  | 8,12 | `Z i32`, `ion i32` |
|  | 16,20 | `g_lower i32`, `g_upper i32` |
|  | 24,28 | `nlte_lower i32`, `nlte_upper i32` |
|  | 32,40,48,56,64,72 | `nu_l, lambda_cm, A_ul, f_lu, E_lower_eV, E_upper_eV`, 각 `f64` |
| row | 0,4,8 | `line_slot u32`, `shell_slot u32`, `flags u32` |
|  | 12,20 | `tau_used f64`, `tau_from_pops f64` |
|  | 28,36 | `n_lower f64`, `n_upper f64` |
|  | 44,52 | `S_l_pop f64`, `S_l_used f64` |
|  | 60,68 | `eps_l f64`, `w f64` |

flags는 writer 528–535행의 여섯 bit
`NLTE_ION, POPS_DEFINED, SL_POP, SL_FALLBACK, STIM_CLAMPED, TAU_ROUNDTRIP`다.
소비기는 알려지지 않은 bit를 거부하고, C 구성에서는 population/source 미정의 및
`STIM_CLAMPED` 행을 대체하지 않고 거부한다.

### 1.2 파일 구역 순서와 크기 불일치의 이유

writer 965–985행의 순서는 다음과 같다.

```text
152 B fixed header
selected shell ids                  u32 * n_sel
(T_e,T_rad,n_e,dr)                  f64 * 4*n_sel
nu, dnu                             f64 * 2*n_bins
chi_line, chi_line_th, eta_line     f64 * 3*n_sel*n_bins
EPAY disposition                    u8 * n_shells*n_bins
line-static table                   80 B * selected_lines
rows                                76 B * rows
```

따라서 manifest의 `rows*row_bytes`는 **마지막 행 테이블의 크기만** 나타낸다.
약 48 MB로 보인 부속물은 추정 대상이 아니라 writer가 984행에서 행 앞에 쓰는
`selected_lines × 80 B` 선 정적 테이블이다.

이번 count를 대입하면 다음처럼 정확히 닫힌다.

| 구역 | bytes |
|---|---:|
| 고정 헤더 | 152 |
| 선택 셸 ID | 20 |
| 선택 셸 상태 | 160 |
| `nu+dnu` | 16,000 |
| 세 집계 배열 | 120,000 |
| EPAY disposition | 50,000 |
| `601371×80` 선 정적 테이블 | 48,109,680 |
| `1169145×76` 행 테이블 | 88,855,020 |
| **예상 파일 크기** | **137,151,032** |

소비기는 sidecar와 152 B 헤더를 먼저 읽어 이 식과 `stat` 크기를 확인한다. 이어
streaming SHA-256이 고정값
`84d1849dafd1c796dac77c4037b19683e3ef1d5ddb72dd0e6bf701490b05a1cc`와
일치한 뒤에만 두 큰 테이블을 mmap한다. trailing byte, 부분 파일, 다른 count는 모두
fail-closed다.

## 2. 사전등록 판독

`validation/uv_t2n9/PREREG.md`는 reassembly·transport·N9 측정 전 상태로 동결했다.

### 2.1 T2 경계

주 판독량은 shell 8의 BALL(600–3000 Å) band-mean `J_det`이다. B0–B4도 전부 내지만
사후 다수결을 막기 위해 최종 세 갈래는 BALL 하나로 판정한다.

1. `abs(C/A - 1) <= 0.05`: **OPERATOR_ONLY**. C가 A를 5% 이내 재현한다.
2. 위가 아니고 `C/CMFGEN > 3`: **ASSEMBLY_AND_OPERATOR**.
3. 위가 아니고 `1/3 <= C/CMFGEN <= 3`: **ASSEMBLY_ONLY**.
4. 나머지는 `UNRESOLVED-OUTSIDE-PREREG`; 사후 경계를 추가하지 않는다.

5%는 0.0212 dex로 수치 반복성보다 넓되 물리적으로 같은 진폭이라 부를 수 있는
작은 차이다. CMFGEN 수준의 factor-3 경계는 기존 T2 사전등록
`docs/CODEX_UV_T1T2.md`의 `[1/3,3]`을 그대로 유지했다. branch 2의 기여분은

```text
log10(A/CMFGEN) = log10(A/C) + log10(C/CMFGEN)
```

으로 assembly 제거분과 operator 잔여분을 낸다. branch 3이면 E6/T1과의 모순 해명을
필수 flag로 출력한다.

### 2.2 고정 수송 좌표와 대역

- Stage 3.1 CPU formal operator, shell 8, `nmu=16`, `T_inner=10020 K`,
  `bb_scale=1`, OMP 최대 4 threads.
- B0 `[600,1000)`, B1 `[1000,1500)`, B2 `[1500,2000)`,
  B3 `[2000,2500)`, B4 `[2500,3000]`, BALL `[600,3000]` Å.
- 대역 평균과 N9 energy에는 exact frequency-edge overlap을 쓴다.
- B2는 같은 캡처 디렉터리의 `emiss_ab_iter10.B2`만 허용한다. 누락 시 다른 E-계열
  산출물을 찾지 않는다.

## 3. 스크립트 계약

### 3.1 사전검사

`scripts/uv_t2n9_offline.py`는 다음을 무거운 row 처리 전에 검사한다.

- LINEPOP fixed header/schema/endian/version, iteration/generation 10,
  50×1000, 선택 셸 `[0,8,16,20,45]`, count, gate, 정확한 파일 크기와 SHA-256.
- CHIETA sidecar/SHA, iteration/generation 10, post-damping/coherent-frozen,
  LINEPOP과 뒤집었을 때 bitwise 같은 `nu,dnu,t_exp`.
- manifest/payload disposition exact census, 알려진 flags, row slot 범위와 writer의
  shell→line 순서, 모든 필수 배열의 유한성·부호.
- 같은 디렉터리 A/B2의 lane tag 및 geometry/grid/opacity/J bitwise 공통성.

비유한값이나 음수 물리량을 고쳐서 진행하는 경로는 없다.

### 3.2 A, 음성 대조, 결정론

A 검사는 두 층이다.

1. LINEPOP 선택창이 bin 전체를 덮는 내부 bin에서 기록 행을 writer 순서로 합산해
   `chi_line`, `chi_line_th`, `eta_line` 집계 배열과 각각 bitwise 비교한다.
2. LCMFCE01 header와 9개 배열을 같은 순서로 직렬화해 `chieta_iter10` 전체 bytes와
   비교한다.

정상 A가 두 gate를 지난 뒤 재합산 `chi_line`의 첫 byte에 XOR `0x01`을 주입하고,
동일 bitwise gate가 **FAIL**해야 한다. FAIL이 발화하지 않으면 정상 실행도 중단한다.
`negative_control.json`에는 expected/observed FAIL을 남긴다.

C payload 조립은 같은 입력으로 두 번 직렬화해 SHA가 같아야 한다. A/B2/C 수송 표도
각각 두 번 계산해 byte-identical SHA를 요구한다.

### 3.3 C 재조립

선별 행마다 writer와 같은 expansion-opacity 식을 사용한다.

```text
w_pop       = (1-exp(-tau_from_pops))*nu_l/(c*t_exp*dnu_bin)
chi_C       = sum(w_pop)
chi_th,C    = sum(eps_l*w_pop)
eta_native  = sum(w_pop*S_l_pop)
eta_fixed,C = sum(eps_l*w_pop*S_l_pop)
chi_coh,C   = chi_C - chi_th,C
eta_coh,C   = chi_coh,C * J_producer
```

`TAU_ROUNDTRIP` 행은 이미 `tau_from_pops`와 `tau_used`가 bitwise 같음을 writer가
확인했으므로 기록된 `w`를 그대로 사용해 다른 libm의 1 ulp 차이가 intervention으로
섞이지 않게 한다. 나머지는 `tau_from_pops`에서 직접 계산한다.

선택 밖 45개 셸과 선택창 밖 선은 A로 동결한다. 따라서 C의 엄밀한 범위는 전 모델
global replacement가 아니라 **선택 셸 `[0,8,16,20,45]`의 600–3000 Å intervention**이다.
이 제한은 결과 manifest와 미해결 항목에 남긴다.

### 3.4 EPAY fail-closed

disposition 0/1은 A 선택선의 fixed η를 C의 값으로 교체한다. disposition 2는 line
source가 폐기되므로 fixed η를 A로 유지한다. 다만 C가 그 셀의 `chi_line_th`를 바꾸면
셸 `wn`도 달라지므로 중단한다. 또한 dump의 `n_e`, 집계 `chi_line`, CHIETA
`chi_total`에서 `chi_abs`를 복원해 old/new thick predicate를 비교하고, 셀의 1↔2
소속이 바뀌면 중단한다. disposition 3에서 η 또는 `chi_line_th`가 바뀌어도 중단한다.

즉 저장되지 않은 EPAY scale을 추정하거나 기존 source를 새 opacity로 나누는 우회는 없다.

## 4. N9 정의

### 4.1 셀 수와 에너지 가중

셀 수는 bin 중심 파장을 반개구간 B0–B4에 한 번만 넣는다. 에너지 분율은

```text
E_cell = eta_fixed(post-EPAY) * exact_band_overlap * shell_volume
f_rate = sum(E_cell | disposition=2) / sum(E_cell | all dispositions)
```

로 정의했다. `eta_fixed`는 EPAY 뒤 실제 fixed source이고 coherent 반환은 새 방출
에너지가 아니므로 제외한다. 공통 4π 인자는 분율에서 소거된다. 결과는
`n9_disposition_shell_band.csv`와 `n9_energy_shell_band.csv`에 셸별·대역별로,
`n9_summary.json`에 전역으로 기록한다.

전역 exact gate는 s>=5의 45,000셀 중 34,304셀,
`0.7623111111111111`이 rate-shape여야 한다.

### 4.2 `B(T_e)` 기전

writer의 disposition은 904–918행에서 정해지고, 실제 EPAY rate-shape는
`src/lumina_cmfgen.c:1720-1722`에서

```text
w = eta_bf + chi_line_th * B_nu(T_e)
S_fixed = wn*w/chi_total
```

로 다시 쓰인다. population `S_l_pop`은 이 경로에 없다. 소비기는 선택 셸의
disposition-2 UV cell에서 dump의 `chi_line_th`, `T_e`, `nu`로 line term을 직접 재생해

```text
eta_rate,line = chi_line_th*B_nu(T_e)
S_rate,line   = eta_rate,line/chi_line_th
```

가 `B_nu(T_e)`와 최대 상대오차 `2^-48`, 최대 8 ulp 이내인지 검사한다. 양의
`chi_line_th` cell이 없으면 PASS로 세지 않고 중단한다. 이는 line component 확인이며,
전체 `S_fixed`에는 BF term과 저장되지 않은 `wn`이 함께 있다는 한계를 숨기지 않는다.

## 5. `epay_scale_not_reproducible=true`

정체는 source에서 확정된다.

- 1649–1676행: assemble 당시 **lagged `cs->J`**로 `acc_abs`를 누적하고,
  `acc_w`, `acc_dep`도 같은 pass에서 누적한다.
- 1689–1724행: `wn=(acc_abs+acc_dep)/acc_w`를 계산해 thin-bin source 전체를 다시 쓴다.
- LINEPOP dump는 assemble 뒤 `cmfgen_solve_J`와 J damping 뒤에 호출된다
  (writer 설명 494–500행). dump 시점 `J`는 acc_abs가 읽은 J가 아니며, 세 누산량과
  `wn` 자체도 payload에 없다.

따라서 “재현 불가”는 부동소수점 정밀도 문제가 아니라 **필요 상태 미직렬화**다.
대장 후보 문안은 사전등록과 스크립트에 다음처럼 고정했다.

> **EPAY-REPLAY-001** — LCMFLP01-v1은 disposition은 보존하지만 rate-shape/scalar
> EPAY의 셸별 정규화와 그 입력 장부(`acc_abs`, `acc_dep`, `acc_w`)를 보존하지 않는다.
> 따라서 population-native opacity가 정규화 장부를 바꾸는 반사실은 같은 세대
> payload만으로 exact 재조립할 수 없다. scale과 세 누산량의 per-shell 직렬화를
> 계측 부채로 등재한다.

## 6. 실행 명령과 예상 자원

운전석은 grammar-debug에서 다음을 실행한다.

```bash
python3 scripts/uv_t2n9_offline.py --self-test

python3 scripts/uv_t2n9_offline.py --linepop /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/linepop_iter10 --chieta /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/chieta_iter10 --outdir /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_t2n9_offline
```

예상 상한은 wall 10분, RSS 2 GiB, OMP 4 threads다. LINEPOP은 mmap하며, 가장 큰
임시 배열은 1,169,145행의 population-native 재조립 벡터들이다. 수송은 A/B2/C 각 2회,
기존 50×1000 CPU Stage 3.1 formal solve다. 신규 모델·CMFGEN·GPU 실행은 없다.

## 7. 미해결 항목

1. **실행 검증 미실시.** grammar-debug SSH가 sandbox/시스템 SSH 설정 때문에 Python
   시작 전에 막혔다. `py_compile`, 합성 self-test, 실 payload, 수송 모두 NOT RUN이다.
2. **C coverage는 측정 전 미상.** `POPS_DEFINED`, `SL_POP`, 양의 population/source가
   1,169,145행 모두에서 닫히지 않거나 `STIM_CLAMPED`가 하나라도 있으면 C는
   fail-closed한다. 값을 유지하거나 0으로 두지 않는다.
3. **C는 다섯 선택 셸에 국지화된다.** 나머지 45개 셸은 A다. full-radial native C가
   필요하면 전 셸 LINEPOP 재캡처가 필요하다. 현재 결과는 이 국지 범위를 넘겨 일반화할
   수 없다.
4. **B2와 C의 intervention 범위 차이.** 기존 B2는 캡처된 전 셸 lane이고 C는 선택 셸
   lane이다. B2는 요구된 기존 음성/참조 판본으로 함께 풀지만 A↔C가 주 단일인자다.
5. **EPAY scale 부재.** C가 rate-shape의 `chi_line_th`, scalar-rescaled η, 또는
   thick/thin membership을 바꾸면 exact C를 만들 수 없고 스크립트는 중단한다.
6. **기존 B2 수송 guard 위험.** 이전 E5에서 B2 Stage 3.1이 certified-negative로
   중단된 전례가 있다. 이번에도 발화하면 acceptance를 완화하지 않고 T2 수송 판독은
   UNRESOLVED로 남는다.
7. **N9 `B(T_e)` 확인 범위.** dump로 line term은 직접 재생할 수 있지만 BF term과
   `wn`은 직렬화되지 않았다. 전체 post-EPAY source의 독립 재생에는 EPAY-REPLAY-001
   계측이 필요하다.

## 8. 규율 장부

| 항목 | Step 1 결과 |
|---|---|
| 스크립트 작성 | 완료 |
| 측정 전 숫자 경계 사전등록 | 완료 |
| writer에서 레코드 배치 확정 | 완료 |
| `rows*row_bytes` 크기 차 설명 | 완료, 예상 137,151,032 B |
| 주입 결함 음성 대조 | 스크립트 내 의무화; 실행 NOT RUN |
| 동일 입력 반복 결정론 | payload/수송 각 2회 의무화; 실행 NOT RUN |
| 비유한값 대체 또는 수치 보정 | 없음 |
| `src/` 수정 | 없음(기존 dirty worktree 보존) |
| 로그인 노드 연산 | 없음 |
| 무거운 계산/GPU/모델/CMFGEN 실행 | 없음 |
| git commit | 없음 |
