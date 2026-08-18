# UV T2/N9 사전등록 — 측정 전 동결

상태: **FROZEN BEFORE REASSEMBLY, TRANSPORT, AND N9 MEASUREMENT**  
동결일: 2026-08-02 (Asia/Seoul)  
범위: `scripts/uv_t2n9_offline.py`의 grammar-debug 오프라인 실행. 이 문서는 실제
`linepop_iter10` 또는 `chieta_iter10` 수치 판독 전에 작성되었다.

## 1. 입력·세대 계약

- `linepop` SHA-256은
  `84d1849dafd1c796dac77c4037b19683e3ef1d5ddb72dd0e6bf701490b05a1cc`여야 한다.
- 두 payload 모두 `iteration=10`, `field_generation=10`, `n_shells=50`,
  `n_bins=1000`이어야 한다. CHIETA는 post-damping/coherent-frozen/내림차순 주파수여야
  하며, LINEPOP은 오름차순 주파수여야 한다. 두 격자는 뒤집었을 때 bitwise 같아야 한다.
- LINEPOP 선택은 셸 `[0,8,16,20,45]`, 파장 `[600,3000]` Å, `eps_phys=1`,
  `src_nlte=0`, `epay=2`, `epay_smin=5`, `epay_taubin=10`, `epay_hotf=0`으로
  동결한다. manifest의 `clamp`와 `fallback`은 모두 0이어야 한다.
- B2는 LINEPOP과 같은 디렉터리의 `emiss_ab_iter10.B2`를 사용한다. lane tag는
  `B2-Aul-nu-retain-A-undefined`여야 하고, A와 geometry/grid/opacity/J가 bitwise
  같아야 한다. 누락되거나 세대·좌표가 다르면 대체 자료를 찾지 않고 중단한다.
- 해시·sidecar·고정 헤더·예상 파일 크기를 먼저 검사한 뒤에만 행 테이블을 mmap한다.
  비유한값, 음수 opacity/emissivity, 알 수 없는 flag/disposition, 불완전 population은
  세지 않고 즉시 중단한다.

## 2. writer에서 동결한 LCMFLP01-v1 배치

근거는 `src/lumina_cmfgen.c`의 writer 자체다.

- little-endian packer: 513–523행; 행 76 B·선 정적 레코드 80 B: 525–526행.
- 행 필드 offset: 862–873행. 순서는
  `line_slot:u32, shell_slot:u32, flags:u32, tau_used:f64,
  tau_from_pops:f64, n_lower:f64, n_upper:f64, S_l_pop:f64,
  S_l_used:f64, eps_l:f64, w:f64`이다.
- 선 정적 레코드 offset: 843–860행. 순서는
  `line_id:u32, bin:u32, Z:i32, ion:i32, g_lower:i32, g_upper:i32,
  nlte_lower:i32, nlte_upper:i32, nu_l:f64, lambda_cm:f64, A_ul:f64,
  f_lu:f64, E_lower_eV:f64, E_upper_eV:f64`이다.
- 파일 쓰기 순서: 939–985행. 152 B 고정 헤더 뒤에 선택 셸 ID, 셸별
  `(T_e,T_rad,n_e,dr)`, `nu`, `dnu`, 세 집계 배열, 전 셸 EPAY disposition,
  선 정적 테이블, 행 테이블이 이어진다.

따라서 manifest의 `rows*row_bytes`는 **파일 전체 크기가 아니라 마지막 행 테이블만**
뜻한다. 이번 고정 count에서 예상 크기는 다음과 같다.

```text
fixed header                                      152
selected shell ids                     5*4 =       20
selected shell state                  5*4*8 =      160
nu + dnu                         2*1000*8 =   16,000
chi_line + chi_line_th + eta_line 3*5*1000*8 = 120,000
EPAY disposition                  50*1000 =   50,000
line-static table               601371*80 = 48,109,680
rows                           1169145*76 = 88,855,020
-------------------------------------------------------
expected file bytes                         137,151,032
```

실제 크기가 이 식과 다르면 offset을 추정하거나 trailing bytes를 무시하지 않고 중단한다.

## 3. T2 구성과 수송 좌표

공통 수송 좌표는 기존 Stage 3.1과 같은 `shell=8`, `nmu=16`,
`T_inner=10020 K`, `bb_scale=1`, canonical 50×1000 격자, CPU formal operator다.
OMP는 4 threads 이하로 둔다. 대역은 B0 `[600,1000)`, B1 `[1000,1500)`,
B2 `[1500,2000)`, B3 `[2000,2500)`, B4 `[2500,3000]`, BALL `[600,3000]` Å다.
대역 평균에는 exact frequency-edge overlap을 쓴다.

- A: CHIETA를 읽고 같은 header/array 순서로 다시 직렬화한다. 원본과 byte-for-byte
  같지 않으면 중단한다.
- B2: 동시 캡처된 기존 B2 payload다. 다른 lane과 같은 operator에 투입한다.
- C: 기록된 행 순서로 `tau_from_pops`에서 `w_pop`을 만들고,
  `chi_line=sum(w_pop)`, `chi_line_th=sum(eps_l*w_pop)`,
  `eta_line,fixed=sum(eps_l*w_pop*S_l_pop)`을 조립한다. coherent 몫은
  `(chi_line-chi_line_th)*J_producer`로 두어 A와 같은 split/operator를 유지한다.
  population/`S_l_pop`이 정의되지 않은 행이나 stimulated-emission 보정이 잘린 행은
  값을 대체하지 않고 중단한다.
- disposition 0/1에서는 A의 선택 선 기여를 C 기여로 정확히 교체한다. disposition 2는
  population line source를 폐기하므로 fixed η를 A 그대로 유지하되, C가
  `chi_line_th`를 바꾸면 EPAY 정규화도 달라져 exact 재현이 불가능하므로 중단한다.
  disposition 3도 η 또는 `chi_line_th`가 바뀌면 셸 scale이 필요하므로 중단한다.

A 직렬화 gate가 PASS한 직후 payload의 첫 수치 byte 1 bit를 의도적으로 뒤집어 같은
gate가 FAIL하는 것을 확인한다. 이 FAIL이 발화하지 않으면 정상 실행도 실패다. A/B2/C
payload 조립과 각 transport table은 각각 2회 수행하며 SHA-256이 같아야 한다.

## 4. 측정 전 T2 판독 경계

주 판독량은 BALL의 band-mean `J_det`이다. 모든 B0–B4 값도 함께 표로 내지만 세 갈래
판정은 BALL 하나로 고정해 사후 다수결을 금지한다. 아래 규칙은 위에서 아래 순서다.

1. **operator-only**: `abs(C/A - 1) <= 0.05`. 즉 C가 A를 **5% 이내**로 재현한다.
2. **assembly+operator**: 1이 아니고 `C/CMFGEN > 3`.
3. **assembly-only**: 1이 아니고 `1/3 <= C/CMFGEN <= 3`.
4. 위 어디에도 들지 않거나 양/유한 분모가 없으면 `UNRESOLVED-OUTSIDE-PREREG`로
   중단하며 새 경계를 고르지 않는다.

5%는 0.0212 dex로, byte 결정론과 수송 residual 기준보다 훨씬 넓지만 물리적으로
“같은 진폭”이라 부를 수 있는 작은 차이다. 반면 기존 T2 사전등록은 CMFGEN 수준을
factor-3 (`[1/3,3]`)로 동결했다. 두 scale 사이를 의도적으로 넓게 벌려 수치 잡음과
물리적 붕괴를 혼동하지 않는다. branch 2에서는 로그 장부를
`log10(A/CMFGEN) = log10(A/C) + log10(C/CMFGEN)`으로 나눠 assembly 제거분과
operator 잔여분을 정량한다. branch 3이면 E6/T1과의 모순을 별도 필수 항목으로 남긴다.

## 5. N9 측정 정의와 경계

- 셀 수 분율은 각 bin 중심 파장을 위 반개구간에 한 번만 배정한다.
- 에너지 가중 분율은 coherent 반환을 제외한 post-EPAY fixed emissivity
  `eta_fixed * exact_band_overlap * shell_volume`로 정의한다. 공통 `4pi` 인자는 분율에서
  소거된다. 셸별·대역별 및 전 셸·대역별
  `rate_shape_replaced / all dispositions`를 모두 낸다.
- 전역 disposition count는 manifest와 exact 일치해야 하며, 특히 s>=5의 45,000셀 중
  rate-shape 34,304셀, 즉 `34304/45000 = 0.7623111111111111`이어야 한다.
- 선택 셸의 disposition-2 UV cell에서 dump의 `chi_line_th`, `T_e`, `nu`로
  `eta_rate,line=chi_line_th*B_nu(T_e)`와
  `S_rate,line=eta_rate,line/chi_line_th`를 직접 재생한다. 양의 `chi_line_th` 셀이
  하나도 없으면 중단한다. `S_rate,line/B_nu(T_e)`의 최대 상대오차는 `2^-48`
  (`3.552713678800501e-15`) 이하이면서 최대 8 ulp 이하여야 한다.
- `epay_scale_not_reproducible=true`는 조립 당시 lagged `J`를 쓰는
  `acc_abs`(1649–1676행)와 `wn=(acc_abs+acc_dep)/acc_w`(1689–1724행)가 dump 호출 전
  solve/damping으로 갱신된 `J`로는 재생되지 않는다는 뜻으로 판독한다. 다음 문안을
  결함 대장 후보로 그대로 출력한다.

> EPAY-REPLAY-001: LCMFLP01-v1은 disposition은 보존하지만 rate-shape/scalar EPAY의
> 셸별 정규화와 그 입력 장부(acc_abs, acc_dep, acc_w)를 보존하지 않는다. 따라서
> population-native opacity가 정규화 장부를 바꾸는 반사실은 같은 세대 payload만으로
> exact 재조립할 수 없다. scale과 세 누산량의 per-shell 직렬화를 계측 부채로 등재한다.

## 6. 자원·금지 사항

실행 장소는 grammar-debug 전용이다. 예상 상한은 wall 10분, RSS 2 GiB,
OMP 4 threads이며 GPU/모델/CMFGEN 실행은 없다. 기존 CMFGEN 결과를 읽고 Stage 3.1 CPU
formal solve만 수행한다. 생산 `src/` 수정, clamp/floor/fallback, 비유한값 대체,
acceptance 완화, git commit은 금지한다.
