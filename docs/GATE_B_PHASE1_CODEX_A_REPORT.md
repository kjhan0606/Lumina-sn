# Gate-B Phase 1 구현 보고서 — Codex A

작성일: 2026-07-31  
명세 정본: `docs/GATE_B_DUAL_ORACLE_SPEC.md`

## 1. 결론

직전 작업자의 잔존 편집은 **전면 폐기하지 않고 검수 후 재사용·완성**했다.
`src/lumina_plasma.c`의 관측 지점이 실제 production 함수
`compute_bf_opacity()`와 `nlte_assemble_rate_matrix()` 내부의 계산 완료 로컬값을
읽고 있었고, 모든 관측 코드와 `src/lumina.h` 선언이
`LUMINA_FROZEN_ORACLE`로 닫혀 있어 기본 경로에 피드백하지 않는 구조였기
때문이다.

다만 잔존 상태는 Phase 1 완성본이 아니었다. 실행 하네스와 비교자가 없었고,
누락 항목을 행으로 보존하지 않았으며, free-free cooling 및 입력 provenance가
없었다. 또한 조립하지 않은 상위 이온의 bf rate를 물리적 0으로 출력할 수 있는
오류가 있었다. 이 부분은 `unavailable`로 고쳤다.

튜닝, 임계값 판정, population solve, 물리식 수정은 하지 않았다. 커밋과 push도
하지 않았다.

## 2. 잔존 편집 전수 검수

### 재사용한 부분

- 두 파장(1000 Å, 5000 Å)의 이온별 `chi_bf`, `eta_bf`를 production opacity
  누산 지점에서 읽는 probe.
- production rate assembler가 실제 소비한 `J`, Sobolev `beta`, radiative 및
  collisional line rate를 읽는 probe.
- level-population 가중
  `Gamma = sum(n_l R_bf,l) / sum(n_l)`과 production Milne 적분에서 나온
  `alpha_total/spont/stim`.
- `LUMINA_FROZEN_ORACLE` compile gate와 헤더 선언.

이 값들은 모두 이미 계산된 값을 복사할 뿐이며 matrix, opacity, population,
RNG 또는 제어 흐름을 변경하지 않는다.

### 발견하고 보완한 결함

- 호출자와 CSV writer가 없어 잔존 코드만으로는 실행 불가.
- 대표선이 없는 이온과 원자료가 없는 수량이 누락됨.
- Jbar writer에 기록되지 않은 count=0 행을 원자료 0으로 오인할 수 있었음.
- 조립하지 않은 상위 이온(Si III, S III, Fe III/IV, Co III)의 bf rate가
  계산된 0처럼 보일 수 있었음. 실제 lower member로 조립된 이온만 rate를
  `available`로 표시하도록 `bf_rate_seen`을 추가함.
- ff emissivity grid의 `4 pi integral(eta_nu dnu)`가 없었음.
- thermal/state 범주와 producer/provenance가 불완전했음.
- 초기 하네스 검수 중 parity50 resolved config에 없는
  `LUMINA_BF_NLTE_POPS=1` 강제 설정을 발견하여 제거함. 최종 산출물에는
  기록된 설정 외 물리 gate를 추가하지 않음.

## 3. 구현 골자

- `bench_frozen_oracle.c`
  - `stdout.log`의 resolved config 116개를 복원하고 OMP만 1 thread로 고정.
  - s0/s8/s45를 각각 one-cell shell 0에 이식.
  - plasma state, ion population, NLTE level population, C1 radiation-field
    fit, Jbar/Beta를 로드.
  - production `compute_bf_opacity()`와 `nlte_assemble_rate_matrix()`를 직접
    호출. 조립 matrix는 solve하지 않고 버림.
  - 여섯 범주 `bf/ff/bb/collisional/thermal/state`를 모두 출력하며 얻을 수
    없는 항목도 사유와 함께 보존.
- `src/lumina_plasma.c`
  - compile-time observer와 production 소비 지점 probe.
  - 대표선 선택은 실제 upward population flow `n_l R_lu` 최대선.
  - ff grid cooling, raw Jbar availability, 상위-stage bf availability 구분.
- `src/lumina.h`
  - gate 내부 observer API 선언. 기존 기본 빌드에는 선언도 노출되지 않음.
- `Makefile`
  - 독립 CPU 타깃 `bench_frozen_oracle`만 추가. 기존 `withParity*` 파일은
    수정하지 않음.
- `scripts/oracle_compare_cmfgen.py`
  - REPORT-ONLY. threshold/PASS/FAIL/verdict 없음.
  - 기존 RVTJ 및 EDDFACTOR parser를 재사용.
  - PRRR Gamma는 기존 검증식
    `sum(n_SL R_SL [cm^-3 s^-1]) / Ion Density`만 사용.
  - 단위 또는 동일 수량 대응이 검증되지 않은 값은 수치 비교하지 않고 사유를
    남김.

## 4. 입력 실측

parity50 입력은 `logs/coevolve_consume_parity50`에서 읽었다.

| 파일 | 헤더 포함 행 수 | 실측 크기 | 용도 |
|---|---:|---:|---|
| `lumina_plasma_state.csv` | 51 | 512 B | W, T_rad, n_e, T_e |
| `lumina_levelpop.csv` | 1,051,901 | 67 MiB | n_k, b_k, sigma 존재 여부 |
| `lumina_ion_pops.csv` | 3,401 | 80 KiB | n_ion, ion fraction |
| `lumina_c1_bins.csv` | 14,401 | 864 KiB | 마지막 iteration C1 W/T/mode |
| `lumina_jbar_dump.csv` | 2,935,501 | 203 MiB | line Jbar/count/beta |
| `lumina_c2_bfr_dump.csv` | 600,001 | 30 MiB | 존재 확인; 이번 CPU C1 assembler에는 주입하지 않음 |

`lumina_ion_pops.csv`는 명세의 네 기본 파일 목록에는 없지만 production writer가
만든 보조 입력이며, 명세가 요구한 `n_ion`과 ion fraction을 추측 없이 만들기
위해 사용했다.

선택 셀의 마지막 C1 iteration은 모두 11이었다. mode census는 다음과 같다.

- s0: fit 14, pin 6, empty 4
- s8: fit 14, pin 4, empty 6
- s45: fit 14, pin 2, empty 8
- 세 셀 모두 degen 0

셀 상태 실측:

| 셀 | T_e [K] | n_e [cm^-3] | T_rad [K] | W |
|---|---:|---:|---:|---:|
| s0 | 2.120268337e4 | 4.621814000e9 | 1.047009324e4 | 2.978587262e-1 |
| s8 | 1.200772780e4 | 7.496125000e8 | 1.047009324e4 | 3.887582090e-2 |
| s45 | 1.189706591e4 | 1.017327000e5 | 1.047009324e4 | 2.781713000e-3 |

## 5. 산출물 및 완결성

Oracle CSV:

| 파일 | 데이터 행 | available | unavailable | SHA-256 |
|---|---:|---:|---:|---|
| `lumina_oracle_cell_s0.csv` | 158 | 93 | 65 | `e8c6e300b3ed68411ac6137e1f08605462283a04d785fc424ba6b37957a725cd` |
| `lumina_oracle_cell_s8.csv` | 166 | 127 | 39 | `d63c7225012d177b249224d93b7a398e28354158d9e94ca88048a186ad18d8e0` |
| `lumina_oracle_cell_s45.csv` | 160 | 102 | 58 | `8129bc1ab71039d2ea9081e5c1424c169126855576161edba3c7dc5dd7be03ee` |

모든 CSV는 `csv.DictReader` 기준 malformed 행 0건이며 각 셀에 다음 행 수가
존재한다: bf 64, ff 5, state 28, collisional 16, thermal 7, bb 38/46/40.

CMFGEN 비교 산출:

- `oracle_vs_cmfgen.csv`: 전체 484행, `compared` 33,
  `lumina_unavailable` 162, CMFGEN-side `unavailable` 289.
- `oracle_vs_cmfgen.md`: 수치 비교행과 parser/unit 근거 요약.
- `shell_cmfgen_depth_map.csv`: 속도 기반 대응.
- `cmfgen_parser_roundtrip.csv`: RVTJ 원문 physical line과 변환 후 값의 9행
  왕복 대조.

## 6. 셸 ↔ CMFGEN depth 대응

RVTJ velocity에 대한 최근접 매칭이며 interpolation/extrapolation은 하지 않았다.

| Lumina 셸 | v_L [km/s] | CMFGEN depth (1-based) | v_C [km/s] | delta v [km/s] | CMF 범위 내 |
|---:|---:|---:|---:|---:|:---:|
| s0 | 4264.000 | 67 | 4394.1823505 | +130.1823505 | 예 |
| s8 | 10088.000 | 54 | 10163.5057500 | +75.5057500 | 예 |
| s45 | 37024.000 | 1 | 35975.2880450 | -1048.7119550 | 아니오 |

s45는 RVTJ 최고 속도 밖이다. 표에는 가장 가까운 외곽 depth를 명시하되
외삽하지 않았다.

## 7. 단위·파서 처분

- RVTJ는 기존 `cmp_rvtj_T_ne_vs_published.py`를 재사용했고 헤더
  `Temperature (10^4K)`에 근거해서만 1e4를 곱했다.
- EDDFACTOR는 기존 `gamma_coiii_alllevel.read_eddfactor`를 재사용했다.
  `_INFO`의 RECL, WORD size, endian과 finish record 검사를 그대로 거친다.
- PRRR Gamma만 기존 parser의 차원 검증식을 재사용했다.
- PRRR의 `Radiative Recombination Coefficient...` 블록은 숫자와 라벨은
  읽을 수 있으나 인라인 단위 및 검증된 alpha parser가 없어 비교를 보류했다.
- GENCOOL `(ion) Free-Free Cooling`도 인라인 단위와 기존 검증 parser가 없어
  ff cooling 비교를 보류했다.
- monochromatic chi/eta, 개별 C_lu/C_ul, alpha spontaneous/stimulated split은
  동일 수량 CMFGEN anchor가 없어 `unavailable`로 남겼다.

## 8. 빌드 및 자기 스모크

- `make bench_frozen_oracle`: 성공. 기존 소스의 warning은 있으나 신규 하네스
  warning/error는 없음.
- 현재 최종 바이너리로 기본 출력 디렉터리와 `/tmp/gateb_oracle_exact2`에 각각
  독립 실행. s0/s8/s45 모두 `cmp` 성공, 위 SHA-256 3개가 2회 동일.
- `python3 -m py_compile scripts/oracle_compare_cmfgen.py`: 성공.
- comparator 실제 RVTJ/PRRR/EDDFACTOR 입력 실행: 성공.
- gate-OFF `src/lumina_plasma.c` object를 별도 compile 후 `nm` 검사:
  `lumina_oracle`/`g_oracle` symbol 0개.
- `git diff --name-only | rg -i withParity`: 결과 없음.

전체 `make lumina`는 oracle과 무관한 현재 dirty worktree의 선행 결함 때문에
완료되지 않았다. strict C11에서 `src/lumina_plasma.c`와
`src/lumina_cmfgen.c`의 기존 `M_PI`가 노출되지 않고,
`-D_GNU_SOURCE`를 준 뒤에는 CPU 타깃이 기존
`cmf_solve_J_gpu`/`bf_gemm_compute_fine` 심볼을 링크하지 못한다. 이 파일과
해당 링크 구조는 이번 범위에서 수정하지 않았다. 따라서 full production
binary의 실행 비교는 Codex B에 넘긴다.

## 9. 미해결 및 제한

- 사전 oracle 편집 이전의 신뢰 가능한 binary/output snapshot이 없어 실제
  production 결과의 OFF-before/OFF-after byte 비교는 이 작업트리만으로
  만들 수 없다. compile-time 제거와 symbol 부재는 확인했지만, 명세의 최종
  byte 회귀 판정은 별도 clean baseline이 필요하다.
- thermal ledger 중 `cooling_ff_grid`만 production emissivity로 계산 가능하다.
  heating_photoion, deposition, MA line destruction, cooling_bf,
  cooling_bb_collisional, cooling_adiabatic, net은 이 frozen production call이
  노출하지 않아 명시적 unavailable이다.
- parity50 Jbar writer는 Si II/III만 원자료를 기록했다. 다른 이온의
  representative J는 production C1 fallback 소비값이지만 raw Jbar 행은
  unavailable이다.
- baseline NLTE 구성에 stage-IV pair가 없어 상위 stage의 photoion/recomb
  rate는 임의의 0 대신 unavailable이다.
- s45는 CMFGEN 속도 범위 밖이다.
- C2 bfr dump는 존재하지만 이번 frozen CPU assembler가 사용하는 production
  C1 inline 경로와 다른 입력이므로 혼합하지 않았다.

## 10. Codex B 테스트 지침

1. 현재 dirty tree를 보존하고 먼저 `git status --short`와
   `git diff --name-only | rg -i withParity`를 기록한다.
2. `make bench_frozen_oracle`을 실행한다.
3. 서로 다른 빈 출력 디렉터리에 하네스를 두 번 실행한다.

   ```sh
   ./bench_frozen_oracle logs/coevolve_consume_parity50 \
     data/tardis_reference_toy06_19p48d_sivcaiv /tmp/gateb_B_run1
   ./bench_frozen_oracle logs/coevolve_consume_parity50 \
     data/tardis_reference_toy06_19p48d_sivcaiv /tmp/gateb_B_run2
   sha256sum /tmp/gateb_B_run1/*.csv /tmp/gateb_B_run2/*.csv
   cmp /tmp/gateb_B_run1/lumina_oracle_cell_s0.csv /tmp/gateb_B_run2/lumina_oracle_cell_s0.csv
   cmp /tmp/gateb_B_run1/lumina_oracle_cell_s8.csv /tmp/gateb_B_run2/lumina_oracle_cell_s8.csv
   cmp /tmp/gateb_B_run1/lumina_oracle_cell_s45.csv /tmp/gateb_B_run2/lumina_oracle_cell_s45.csv
   ```

4. CSV schema/완결성은 각 셀에 여섯 category가 모두 존재하는지, `status`가
   available/unavailable 중 하나인지, unavailable에 note가 있는지 검사한다.
5. `python3 -m py_compile scripts/oracle_compare_cmfgen.py` 후 comparator를
   실행한다. 출력에 PASS/FAIL/verdict/threshold가 없는지 확인한다.
6. `cmfgen_parser_roundtrip.csv`에 적힌 RVTJ physical line을 원문에서 최소
   3건 직접 확인한다. 특히 T_e raw 값에만 1e4 변환이 적용되는지 본다.
7. s45의 `in_cmfgen_range=False` 및 depth 1 nearest-only 처분을 확인한다.
8. OFF byte 회귀는 oracle 이전 clean snapshot 또는 별도 control worktree를
   확보한 뒤 동일 compiler/flags/input/thread/RNG 조건으로 실행해야 한다.
   현재 선행 `lumina_cmfgen.c` CPU 링크 결함을 oracle 변경과 분리하여 먼저
   처리하거나, 이미 검증된 production binary를 control로 사용한다.
9. 어떤 비교에서도 unavailable을 0으로 치환하지 말고, comparator 수치 자체로
   gate verdict를 내리지 않는다.
