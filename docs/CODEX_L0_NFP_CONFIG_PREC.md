# L-0 잔여 계약 NE-NAMING · DECK-FOSSIL · CONFIG-PREC

- 작성일: 2026-08-05 KST
- 기준 실측: `docs/OUTSIDE_LOOP_POOL.md`의
  `★★★0-N/0-F/0-P 실측 (2026-08-04 23:30, 운전석)`
- 상태:
  - **NE-NAMING — 처분 A checker/builder guard 구현 완료, grammar-debug 음성 대조 대기**
  - **DECK-FOSSIL — fossil quarantine checker/atomic writer 구현 완료, grammar-debug 음성 대조 대기**
  - **CONFIG-PREC — 계약 저작 및 구현 완료, grammar-debug 운전석 검증 대기**
- 실행 상태 표기: 이 문서를 쓰는 세션에서는 로그인 노드 연산 금지를 지켰다. 파일 조회,
  `rg`, `stat`, `sha256sum`, diff 판독만 수행했다. 아래 빌드·Python·음성 대조·회귀 값은
  **`PENDING_DRIVER_EXECUTION`**이며 실행 전 PASS로 승격하지 않는다.
- 순서 봉인: 세 계약이 모두 닫히기 전에는 A2-04 생산자 commit을 시작하지 않는다.

---

## 1. 선행 CONFIG-PREC 변경집합

CONFIG-PREC 구현 세션이 더한/고친 파일은 다음 다섯 개다. 그 세션 시작 전부터 이 중 세 `src` 파일을
포함한 worktree가 크게 dirty였으므로 기존 변경은 되돌리거나 재정렬하지 않고 국소 hunk만
삽입했다.

| 파일 | 종류 | 이번 계약의 변경 |
|---|---|---|
| `src/lumina_atomic.c` | 수정 | 공통 CONFIG-PREC 판정, strict gate, `T_inner` source resolve/log |
| `src/lumina_main.c` | 수정 | CPU argv/config source 로그, 중복 env `T_inner` resolve 제거 |
| `src/lumina_cuda.cu` | 수정 | CUDA argv/config source 로그, 중복 env `T_inner` resolve 제거 |
| `scripts/run_config_prec_negative_controls.py` | 신규 | `/tmp` 사본만 쓰는 실제 바이너리 음성 대조 4건 |
| `docs/CODEX_L0_NFP_CONFIG_PREC.md` | 신규 | 본 계약·조사·구현·운전석 인수 문서 |

후속 NE-NAMING/DECK-FOSSIL checker 구현 변경집합은
`docs/CODEX_L0_NE_DECK_CHECKERS.md`에 별도로 기록한다. 모든
`data/tardis_reference_*`, `/gpfs/kjhan/cmfgen_runs/**`, `src/`,
`LUMINA_TRAD_COLOR_FIX` 블록은 그 후속 작업에서 수정하지 않았다. commit/push도 하지 않았다.

---

## 2. 공통 용어와 판정량

현재 toy06 덱에 대해 다음을 정의한다.

\[
T_{\rm cfg}\equiv {\tt config.json:T\_inner\_K},\qquad
T_{{\rm color},s}\equiv {T_{{\rm rad},s}\over W_s^{1/4}}
\]

\[
\Delta_T=|T_{\rm cfg}-T_{{\rm color},0}|,\qquad
\delta_T={\Delta_T\over |T_{\rm cfg}|},\qquad
S_T=\max_sT_{{\rm color},s}-\min_sT_{{\rm color},s}.
\]

`plasma_state.csv`는 현재 builder가 명시한
`T_rad = T_inner * W**0.25`를 정확히 따르므로 `T_color,s`가 파일에 내재한 경계 color
선언이다. 현재 값은 `T_cfg=10020 K`, `T_color=14172.549003 K`,
`Delta_T=4152.549003 K`, `delta_T=0.414426...`이다.

등급의 공통 의미는 다음과 같다.

- **FATAL**: 정본성이나 물리 경계가 결정되지 않아 생산·A-2 입력으로 소비할 수 없다.
  구현된 CONFIG-PREC 실행에서는 rc=1이다.
- **WARN**: 결함을 승인한 legacy/진단 경로만 계속 실행한다. PASS나 정본 승격이 아니며
  rc=0이어도 로그 marker를 반드시 남긴다.

---

## 3. 계약 NE-NAMING — 광학깊이 전자밀도의 진실성

### 3.1 단일 물리 계약

전자산란 광학깊이로 photosphere를 정하는 `n_e(v)`는 그 물리 출처와 근사를 명시해야 하며,
placeholder가 해를 푼 전자밀도인 것처럼 생산 경계조건을 조용히 결정해서는 안 된다.
이 계약의 처분은 **개명**이 아니다. 또한 현 시점에 CMFGEN `<Z>(v)`로 덱을 즉시
재생성하라는 계약도 아니다.

### 3.2 금지와 요구

금지한다.

1. `n_e=n_atom*1.0`을 provenance 없는 물리 전자밀도로 소비하는 것.
2. placeholder를 허용하는 명시적 legacy/diagnostic 승인 없이 그 값으로 `i_phot`,
   `v_inner`, `r_inner`, `T_inner`, 50-shell grid를 생산하는 것.
3. `electron_densities.csv`만 바꾸고 geometry/config/plasma를 그대로 두는 부분 교체.
4. `4.005038`의 덱 계보가 해결되기 전에 CMFGEN `<Z>`를 현 정본 덱에 바로 이식하는 것.
5. 변수·파일의 개명만으로 물리 계약이 복구됐다고 판정하는 것.

요구한다.

1. placeholder 경로는 최소한 `electron_density_mode=PLACEHOLDER_ZBAR_ONE`,
   수식, 적용 zone, builder hash, 입력 hash, `tau_phot`, 승인 disposition을 machine-readable
   manifest와 로그에 기록한다.
2. production/default 경로는 provenance가 없거나 placeholder 승인 토큰이 없으면
   `i_phot` 계산 전에 fail-closed 한다.
3. 참값 경로는
   \(n_e(v)=\sum_Z\sum_q q\,n_{Z,q}(v)\)를 같은 epoch·composition·velocity frame에서
   구성하고, 원본 CMFGEN `RVTJ`, 단위, ND, interpolation, 중복/비단조 처리, coverage와
   격자 밖 정책을 명세한다. coverage 밖 값을 조용히 1로 두지 않는다.
4. 어느 경로를 즉시 이행할지는 처분 승인으로 고른다.
   - A: 현 legacy 덱을 hash로 봉인하고 placeholder builder를 production에서 차단한다.
   - B: 위 참값 경로를 구현하고 DECK-FOSSIL 폐합 뒤 새 generation으로 전체 덱을 재생성한다.

### 3.3 위반 판정식과 등급

zone별

\[
\bar Z_s={n_{e,s}\over n_{{\rm atom},s}}
\]

를 기록하고, 경계 사슬 영향은

\[
\tau_i=\sum_{j=i}^{N-2}{1\over2}(n_{e,j}+n_{e,j+1})\sigma_T
(r_{j+1}-r_j),\qquad
i_{\rm phot}=\max\{i:\tau_i\ge\tau_{\rm phot}\}
\]

로 재현한다.

| 조건 | 등급 | 근거 |
|---|---|---|
| `n_e` provenance/mode/승인 중 하나라도 없는데 production boundary를 생성 | FATAL | placeholder가 경계조건을 결정하지만 소비자가 이를 식별할 수 없음 |
| mode가 placeholder인데 승인 없이 canonical/production 출력 요청 | FATAL | provenance 없이 placeholder가 경계 사슬을 결정함; 영향 크기는 `UNQUANTIFIED_PENDING_CLEAN_ZBAR` |
| 참값 경로에서 epoch/frame/unit/coverage/격자 밖 정책 불명 | FATAL | 같은 배열처럼 보여도 다른 물리 좌표를 섞을 수 있음 |
| geometry/config/plasma 중 일부만 새 `n_e`와 결합 | FATAL | `r_inner`와 Stefan–Boltzmann 경계가 서로 다른 generation이 됨 |
| 정확한 기존 덱 hash에 묶인 승인된 legacy 진단 실행 | WARN | 재현 보존은 가능하나 물리 정본 주장은 불가 |

현재 `build_toy06_epoch.py`는 처분 A gate를 `tau_i/i_phot` 계산 앞에 두어 기본 production
호출을 차단한다. 현 덱을 즉시 교체하는 처분은 아니다. case A의 3900 km/s가 실제 config와
정확히 일치한다는 양성 대조는 placeholder가 단순 이름 문제가 아니라 현재 덱 경계의
생산자였음을 보인다. 오염되지 않은 `<Z>`가 없으므로 그 영향의 크기는 정량화하지 않는다.

### 3.4 사전등록 음성 대조

NE 구현 단계에서 만들 checker의 종료코드는 PASS/WARN=0, FATAL=1로 고정한다.

| 주입 결함 | 기대 marker | 기대 rc |
|---|---|---:|
| scratch builder manifest에서 `electron_density_mode` 삭제 | `NE-NAMING][FATAL] missing mode` | 1 |
| `PLACEHOLDER_ZBAR_ONE`인데 production 출력과 승인 토큰을 함께 요청하지 않음 | `NE-NAMING][FATAL] unapproved placeholder` | 1 |
| CMFGEN 참값 manifest의 epoch를 19.48 d가 아닌 값으로 변경 | `NE-NAMING][FATAL] epoch mismatch` | 1 |
| 새 `electron_densities.csv`와 구 geometry/config/plasma를 혼합 | `NE-NAMING][FATAL] generation mismatch` | 1 |
| 승인된 legacy hash와 placeholder mode를 함께 명시 | `NE-NAMING][WARN]` | 0 |

이 표대로 `scripts/check_ne_naming.py`와 `scripts/run_ne_naming_controls.py`를 구현했다.
실행 전 상태는 **PENDING_DRIVER_EXECUTION**이며 모든 fixture는
`/tmp/lumina_ne_naming_controls_*` 사본에서만 만들어진다.

### 3.5 기대 변경집합

바뀌어야 하는 것:

- 처분 A면 builder의 mode/provenance/production guard, manifest writer, checker, fixture, 문서.
- 처분 B면 위 항목에 더해 charge-balance 입력/보간기와 전체 덱 generation writer.
- builder 출력에는 `i_phot`, `v_inner`, `r_inner`, `tau_total`, `<Z>` 통계와 입력 hash가
  남아야 한다.

바뀌면 안 되는 것:

- 승인 전 현재 `data/tardis_reference_toy06_19p48d/**` 바이트.
- `sigma_T`, `tau_phot`, photosphere index 정의, shell resampling을 이 계약에 끼워 바꾸는 것.
- `4.005038`을 설명하지 못한 채 luminosity/config/plasma를 재생성하는 것.
- 수송·NLTE·A-2 `J_nu` 생산자 코드.

### 3.6 폐합 조건

1. A/B 처분이 승인되고 machine-readable mode/provenance가 구현된다.
2. case A와 CMFGEN `<Z>` case B/C 재현 수치가 manifest에 묶인다.
3. 위 음성 대조 전부 기대 rc를 내고, 승인된 양성 대조가 rc=0을 낸다.
4. DECK-FOSSIL과 CONFIG-PREC가 닫혀 `n_e -> i_phot -> r_inner -> T_inner/plasma`
   generation 하나를 가리킨다.
5. 폐합은 A2-04 이전이어야 하며, 그 전까지 L-0 legacy 음성 대조는 의미 확정 전이다.

---

## 4. 계약 DECK-FOSSIL — 덱 generation 원자성·재현성

### 4.1 단일 물리 계약

`config.json`, `geometry.csv`, `electron_densities.csv`, `plasma_state.csv`와 그 생산 입력은
한 writer transaction/generation에서 나와야 하며, 선언된 명령으로 같은 물리 스칼라를
재현할 수 있어야 한다.

### 4.2 금지와 요구

금지한다.

1. mtime, 유효자릿수 또는 `1/4`에 맞춘 서사만으로 producer를 추정해 정본화하는 것.
2. config를 다른 세 파일보다 뒤에 손으로/별도 스크립트로 바꾸고 generation 표식을 남기지
   않는 것.
3. untracked 덱을 재현 가능한 canonical deck이라고 부르는 것.
4. 원인 미상 `L_cfg`를 현 builder의 출력이라고 선언하거나, 현 builder 출력으로 덱을
   덮어 원인을 지우는 것.

요구한다.

1. manifest가 writer 경로+SHA-256, 전체 argv/env, 입력 hash, epoch, 상수, 단위,
   여섯 config key, companion hash, generation ID, 생성 시작/commit을 기록한다.
2. writer는 임시 디렉터리에 전체 집합을 쓴 뒤 검증 성공 시 한 generation으로 commit한다.
3. registered writer replay가 선언한 `L`, `r_inner`, `T_inner`, `W`, `T_rad`, `n_e`를
   허용오차 안에서 재현한다.
4. 현 덱은 원인이 해결될 때까지 exact SHA-256에 묶인 legacy fossil로만 보존하며
   A-2 production seed의 무조건 정본으로 승격하지 않는다.

### 4.3 위반 판정식과 등급

등록 writer replay에 대해

\[
R_L={L_{\rm replay}\over L_{\rm deck}},\quad
\epsilon_L=|R_L-1|,
\]

\[
T_{\rm SB}=\left({L_{\rm deck}\over4\pi r_{\rm inner}^2\sigma_{\rm SB}}\right)^{1/4},
\quad \Delta_{\rm SB}=|T_{\rm cfg}-T_{\rm SB}|.
\]

`epsilon_L <= 10^-6`, `Delta_SB <= 5 K`를 replay 합격선으로 사전등록한다. 5 K는
현 builder가 `T_inner`를 10 K 단위로 반올림하는 데 필요한 최대 반폭이다. 더 큰 물리
허용오차가 필요하면 원인과 단위를 먼저 계약 개정해야 하며 덱 자체가 완화할 수 없다.

| 조건 | 등급 | 근거 |
|---|---|---|
| writer/입력/명령/generation manifest 부재인데 canonical/production 주장 | FATAL | 재현 대상 자체가 정의되지 않음 |
| companion hash/generation 중 하나라도 불일치 | FATAL | 서로 다른 경계 사슬의 혼합 |
| 등록 writer replay가 `epsilon_L>1e-6` 또는 `Delta_SB>5 K` | FATAL | 물리 스칼라를 재생산하지 못함 |
| 현 덱을 exact hash로 제한한 승인 legacy 재현 실행 | WARN | 보존은 되지만 producer 정본성은 없음 |

현재 builder를 producer라고 가정하면 `R_L=4.005038`, `epsilon_L=3.005038`이므로
FATAL이다. 반면 config 내부의 `Delta_SB=1.65 K`는 통과한다. 즉 내부 산술 정합이
generation 재현성을 대신하지 못한다.

### 4.4 사전등록 음성 대조

DECK 구현 단계 checker의 종료코드는 PASS/WARN=0, FATAL=1로 고정한다.

| 주입 결함 | 기대 marker | 기대 rc |
|---|---|---:|
| scratch 덱에서 generation manifest 삭제 | `DECK-FOSSIL][FATAL] missing manifest` | 1 |
| config만 다른 generation 사본으로 교체 | `DECK-FOSSIL][FATAL] generation mismatch` | 1 |
| `plasma_state.csv` 한 바이트 변경 | `DECK-FOSSIL][FATAL] companion hash mismatch` | 1 |
| manifest writer hash 또는 argv 하나 변경 | `DECK-FOSSIL][FATAL] writer replay mismatch` | 1 |
| 승인된 fossil hash와 read-only legacy mode | `DECK-FOSSIL][WARN]` | 0 |

이 표대로 `scripts/check_deck_fossil.py`, `scripts/deck_generation_atomic.py`,
`scripts/run_deck_fossil_controls.py`를 구현했다. 실제 시연은
**PENDING_DRIVER_EXECUTION**이고 fixture는 `/tmp/lumina_deck_fossil_controls_*`만 사용한다.

### 4.5 기대 변경집합

바뀌어야 하는 것:

- generation manifest schema, atomic writer/commit, replay checker, scratch fixtures, 문서.
- 원인이 실제로 발견되면 그 writer/입력 경로와 재현 명령.
- 원인을 영구 복구할 수 없다면 명시적 fossil quarantine와 승인된 새 canonical generation.

바뀌면 안 되는 것:

- 원인 조사/승인 전 현 canonical과 파생 덱 바이트 및 mtime.
- CMFGEN run 디렉터리.
- NE placeholder 물리, CONFIG precedence, 수송/복사장 생산자.
- 숫자에 맞춘 임의 `0.249686` 상수나 3557 Å cutoff.

### 4.6 폐합 조건

1. 실제 producer가 발견되어 등록 명령으로 전 companion을 재현하거나, producer가
   회복 불가능하다는 처분과 exact-hash fossil quarantine가 승인된다.
2. `4.005038`이 재현 가능한 계산으로 설명되거나, 설명 불능 상태가 fossil quarantine에
   명시되어 더 이상 canonical production 입력으로 쓰이지 않는다.
3. 양성/음성 대조와 hash manifest가 모두 통과한다.
4. 새 generation이면 NE의 `r_inner`와 CONFIG 경계온도가 같은 generation임을 증명하고,
   fossil legacy면 NE/CONFIG가 같은 exact companion hash seal을 가리킨다.
5. A2-04 전에 닫는다.

---

## 5. 계약 CONFIG-PREC — 단일 유효 경계온도와 출처 우선순위

### 5.1 단일 물리 계약

한 run에서 inner-boundary 온도에는 하나의 유효값과 한 출처가 있어야 한다. 다른 입력
채널의 같은 물리 선언은 조용히 공존하거나 후순위 값으로 앞선 덱 무결성 결함을 가릴 수
없다.

### 5.2 우선순위 정본

지원되는 동일 field에 대한 일반 우선순위는 다음과 같다.

```text
명시적 argv > 명시적·유효한 env override > config.json > compiled default
```

`plasma_state.csv`는 override 채널이 아니다. 현재 legacy diluted-boundary schema에서는
`T_rad/W^0.25`가 config 선언과 맞는지를 증언하는 consistency witness다. field별 실제
해석은 다음과 같다.

| 물리량 | 높은 순서 | 로그 |
|---|---|---|
| 초기/고정 `T_inner` | `LUMINA_T_INNER_FIX` > `config.json:T_inner_K`; argv 없음 | 공통 loader가 값과 source 출력 |
| `n_packets`, `n_iterations` | `argv[2]/argv[3]` > config | CPU/CUDA simulation banner가 source 출력 |
| runtime `T_inner` evolution | diffusion mode가 켜지면 `L_cfg,r_inner`의 SB 값; 아니면 유효한 `LUMINA_T_INNER_FIX`; 아니면 controller | 기존 iteration 로그가 매회 mode/value 출력 |
| plasma seed | raw `plasma_state.csv` | override가 아니라 config와 먼저 대조 |

env `LUMINA_T_INNER_FIX`는 strict check 전에 config 대신 대입되지 않는다. 따라서 env를
`14172.549003`으로 맞춰도 fossil config/plasma 불일치를 숨길 수 없다. 유효한 env 값은
무결성 판정 뒤 effective value가 되고, 숫자 전체를 소비했으며 finite/positive인 경우만
허용된다.

`LUMINA_TINNER_COLOR`는 CUDA의 명시적 색 재배치 진단이고 별도 banner가 이미 출처를
남긴다. production 정본으로 합칠지는 A-2 대상이다. 이 계약은 그 게이트를 고치지 않는다.

### 5.3 판정식, 허용오차, 등급

구현 상수는 다음과 같다.

\[
\tau_{\rm decl}=5.0\ {\rm K}+10^{-9}\max(|T_{\rm cfg}|,|T_{{\rm color},0}|),
\]

\[
\tau_{\rm profile}=0.01\ {\rm K}+10^{-9}
\max(|\min T_{\rm color}|,|\max T_{\rm color}|).
\]

다음 중 하나면 mismatch다.

\[
N_W\ne N_{T_{rad}}\ \lor\ N_W\ne N_{shell}
\]

\[
\exists s:\neg(0<W_s\le1)\ \lor\ \neg(T_{{\rm rad},s}>0)\ \lor\
\text{non-finite}
\]

\[
S_T>\tau_{\rm profile}\ \lor\ \Delta_T>\tau_{\rm decl}.
\]

5 K는 builder의 10 K 반올림 반폭이다. profile 0.01 K는 CSV 왕복 반올림보다 넓고 현재
실측 spread `3.6e-12 K`보다 충분히 넓다. 상대항은 큰 값에서 부동소수점 비교만 흡수하며
물리 오차 허용치로 쓰지 않는다.

| 조건 | gate OFF(기본) | gate ON |
|---|---|---|
| 유한한 두 선언의 mismatch/profile 불인증 | `[CONFIG-PREC][WARN]`, 계속(rc=0) | `[CONFIG-PREC][FATAL]`, loader rc=-1, 프로그램 rc=1 |
| row 수 불일치/null 배열/비양의 `T_cfg` | FATAL rc=1 | FATAL rc=1 |
| gate가 정확히 `0/1`이 아님 | FATAL rc=1 | 해당 없음 |
| `LUMINA_T_INNER_FIX`가 finite positive 전체 문자열이 아님 | FATAL rc=1 | FATAL rc=1 |
| 두 선언 합격 | PASS rc=0 | PASS rc=0 |

현재 덱은 `Delta_T=4152.549003 K > tau_decl≈5.000014 K`이므로 gate ON에서 반드시
FATAL이다. `delta_T=41.4426%`라서 계약이 실측 결함을 놓치지 않는다.

### 5.4 gate 이름과 기본값

- 이름: **`LUMINA_CONFIG_PREC`**
- 허용값: unset, `0`, `1`만. 다른 문자열은 FATAL.
- 제안 기본값: **OFF(unset 또는 0)**.

기본 OFF 근거는 현재 canonical 덱 자체가 새 판정에서 FATAL이고 즉시 기본 ON으로 바꾸면
DECK-FOSSIL/NE 처분 전에 캠페인 전체가 정지하기 때문이다. 그러나 OFF는 침묵이 아니다.
현재 덱은 매 load마다 수치와 `[CONFIG-PREC][WARN]`을 남긴다. A2-04 진입과 새 canonical
generation 인수에서는 gate ON을 의무화한다. 이는 단계적 이행이지 결함의 PASS 승격이
아니다.

### 5.5 `LUMINA_TRAD_COLOR_FIX` 비처분 기록

이 게이트의 기존 블록은 수정하지 않았다. 실측상

\[
10470.093240 = 14172.549003\times0.297858726^{1/4}.
\]

따라서 s0 값은 독립 color가 아니라 s0 희석값이다. 기존 게이트가 전 shell에 복사하는
값도 이 희석값이므로 “keeps the photospheric COLOR”라는 주석과 실제 값은 어긋난다.
CONFIG-PREC는 raw 파일을 이 블록보다 먼저 검사할 뿐, 블록의 값·분기·기본값을 바꾸지
않는다. 처분은 A-2에 남긴다.

### 5.6 구현 위치

- `src/lumina_atomic.c:515-659`: strict switch, env number parser, 판정/로그.
- `src/lumina_atomic.c:723-731`: raw `T_cfg` 보존.
- `src/lumina_atomic.c:757-769`: raw W/T_rad를 읽은 직후, TRAD-FIX 전에 공통 판정.
- `src/lumina_atomic.c:770-783`: 기존 TRAD-FIX 블록은 내용 그대로 유지.
- `src/lumina_main.c:152-156,210-224`: argv가 packet/iteration을 이기는 기존 동작을
  source까지 로그하고, 중복 T-inner env parse를 제거.
- `src/lumina_cuda.cu:7094-7097,7343-7361`: CPU와 같은 resolve/log.
- 두 backend의 이후 `LUMINA_T_INNER_FIX` controller pin은 그대로이며 공통 loader가
  이미 유효성을 검증한 값을 소비한다.

### 5.7 구현된 음성 대조

`scripts/run_config_prec_negative_controls.py`는 필수 네 파일만
`/tmp/lumina_config_prec_controls_*`에 `copy2`하고 지정된 실제 바이너리를 호출한다.
운전석 명령은 GPU가 없는 grammar-debug에서도 공통 loader를 검사할 수 있도록 CPU
바이너리를 지정한다.
원본 덱이나 GPFS에는 쓰지 않는다.

| case | 주입 | child 기대 rc | runner 기대 |
|---|---|---:|---|
| `canonical_mismatch` | 현 10020 대 14172.549, gate ON | 1 + FATAL marker | PASS |
| `env_cannot_waive_deck_mismatch` | env를 14172.549로 맞추되 raw 덱은 유지 | 1 + FATAL marker | PASS |
| `split_inferred_color_profile` | config를 s0 color에 맞춘 뒤 row 1 `T_rad`를 2% 변경 | 1 + FATAL marker | PASS |
| `invalid_gate_value` | `LUMINA_CONFIG_PREC=true` | 1 + invalid-switch FATAL | PASS |

네 child가 모두 rc=1과 marker를 내면 runner 자체는 rc=0이다. 현재 상태는
`PENDING_DRIVER_EXECUTION`; 아직 4/4 PASS로 쓰지 않는다.

### 5.8 기대 변경집합

바뀌어야 하는 것:

- mismatch gate/log와 common source resolution.
- packet/iteration argv/config source banner.
- scratch-only 음성 대조와 이 문서.
- gate ON run은 현 덱에서 atomic/plasma load 이후로 진행하지 않아야 한다.

바뀌면 안 되는 것:

- gate OFF에서 warning/source banner 외 기존 물리값, RNG draw, output byte.
- 모든 deck/CMFGEN 파일.
- NE builder와 `n_e`.
- `LUMINA_TRAD_COLOR_FIX`, `LUMINA_TINNER_COLOR`, A-2 seed/producer 규칙.
- D/K/Z-INERT의 기존 판정과 결과.

### 5.9 폐합 조건

1. CPU 및 CUDA 전체 빌드 rc=0.
2. 공통 loader 음성 대조가 4/4이고 각 child rc=1.
3. gate OFF 양성 실행은 현재 덱에서 WARN을 남기고 기존 실행을 계속한다.
4. D 19/19, K 7/7, Z-INERT selftest가 새 소스에서 다시 rc=0.
5. 새 canonical 덱은 gate ON PASS가 인수조건이다.
6. NE와 DECK이 함께 닫히기 전에는 CONFIG 구현만으로 A2-04를 열지 않는다.

---

## 6. `4.005038` 출처 조사

### 6.1 판정

**UNRESOLVED.** 현재 증거로 `3.092725510802548e+42`를 산출해 현
`config.json`에 쓴 계산/명령을 재현하지 못했다. 따라서 출처를 찾았다는 명령은 없다.

### 6.2 확인된 사실

1. 현 파일:
   - `config.json` mtime `2026-06-29 19:19:58 KST`.
   - `plasma_state.csv`, `electron_densities.csv` mtime
     `2026-06-29 14:54:10 KST`.
   - canonical config는 `git ls-files --error-unmatch` rc=1인 untracked 파일이다.
2. 현재 `scripts/build_toy06_epoch.py:179-185`는 19.48 d CMFGEN spectrum을 적분해
   `L_inner`와 `T_inner`를 만들고, `:236-250`에서 plasma와 config를 함께 쓴다.
   현재 실측 `L_gen/L_cfg=4.005038`과 mtime 순서는 이 스크립트를 현 config의 입증된
   producer로 만들지 못한다.
3. 저장소의 직접적인 six-key writer 후보 `scripts/export_tardis_reference.py:288-303`는
   TARDIS model의 계산된 `luminosity_inner`를 float로 기록하므로 긴 유효자릿수를 만들 수
   있다. 그러나 입력은 `data/sn2011fe/sn2011fe.yml`, 출력은
   `data/tardis_reference`로 고정되어 toy06 경로와 다르고, plasma/electron/config를 같은
   실행에서 쓴다. 실제 그 출력의 config는 `T_inner=10521.519456990323 K`,
   `L=9.44e42`, 30 shell이고 세 companion mtime도 4 ms 안에 모여 있어 현 toy06 덱과
   다르다. 현 4시간 25분 mtime 분리를 설명하는 invocation 증거도 없다.
4. `scripts/slurm_*.sh` 다수는 실제로 config/plasma/electron을 지우고 다시 쓴다.
   예를 들어 `scripts/slurm_ddc15_A4.sh:45-58`은 DDC15 source config의 `L_inner`를 읽고,
   `:88-100`에서 새 `T_inner`, config, plasma를 쓴다. 이 계열은 source deck의 L을
   상속하며 현 toy06 숫자 literal이나 canonical target writer 증거가 아니다.
5. `10020.0`과 정확한 L이 함께 있는 config는 canonical, `_sivcaiv` symlink,
   `_sivcaiv_ftos`, `_sivcaiv_fullcov`, `_sivcaiv_links`뿐이다. 네 regular config의
   SHA-256은 모두
   `cf61ab7c880243ffa94bba95b55c3bb4c88e526bcdf1d9b76bd81f44ff81293b`이다.
   `_sivcaiv/config.json`은 canonical 절대경로 symlink이고, downstream regen driver는
   companion을 `copy2`한다(`scripts/deck_regen_fullcov_driver.py:71-81`). 이는 공통
   독립 writer가 아니라 복사 계보를 지지한다.
6. exact L literal 검색은 위 config 복사본과 이를 인용하는 문서 밖의 계산 코드/입력에서
   producer를 찾지 못했다. shell history 검색에서도 exact literal/target writer 명령을
   찾지 못했다.
7. StaNdaRT CMFGEN bolometric 표의 19.48 d 행은
   `1.27878e43`, deposition `1.06134e43`이며 둘 다 `L_cfg`가 아니다
   (`data/standart_data1/toy06/lbol_edep_toy06_cmfgen.txt:29`).
8. 운전석이 이미 검사한 epoch 오선택, 3557 Å 절단, 정확한 1/4 상수, git 계보 네 가설도
   모두 기각된 상태다.

### 6.3 말할 수 있는 범위

16자리 L은 계산/직렬화 산물일 가능성이 높고 `export_tardis_reference.py`는 그런 형식의
writer 한 종류다. 그러나 이것은 **형식적 가능성**이지 현 파일의 출처 증명이 아니다.
현재 정직한 결론은 `UNRESOLVED`이며, 숫자에 맞는 새로운 계산을 발명해 producer로
소급하지 않는다.

---

## 7. A-2 단계 의존

| A-2 단계/게이트 | 세 계약에서 필요한 전제 | 미폐합 시 처분 |
|---|---|---|
| A2-01 소유권 census | CONFIG-PREC가 effective scalar/source와 plasma witness를 구별 | census에는 `UNRESOLVED`로 기록 가능하나 정본 승격 불가 |
| A2-02 좌표·격자 | NE가 `i_phot/v_inner/r_inner` 생산 근거를 확정하고 DECK이 같은 generation을 증명 | grid oracle 확정 불가 |
| A2-03 `RadiationField` shadow | DECK generation과 CONFIG boundary source가 field provenance에 들어감 | interface 초안만 가능, 인수 불가 |
| L-0 legacy 음성 대조 | `W B_nu(T_rad)`가 어느 boundary/generation에서 왔는지 세 계약 모두 확정 | s0 다섯 대역 FAIL이어도 의미 미정, PASS 자격 없음 |
| **A2-04 producer commit** | **NE · DECK · CONFIG 세 계약 완전 폐합** | **HARD BLOCK** |
| A2-04 이후 replay | 새 seed commit 규칙과 구 legacy negative를 generation으로 구분 | 구 ambiguous profile을 새 규칙의 음성 대조로 재사용 금지 |

핵심은 L-0의 “기존 deck `W B_nu(T_rad)`를 넣으면 s0 다섯 대역 모두 FAIL”이 단순
수치 테스트가 아니라는 점이다. `T_rad`의 boundary와 generation이 확정돼야 무엇을
기각했는지가 정의된다. 반대로 A2-04 뒤에는 producer/seed 규칙이 바뀌므로 세 계약을
나중에 닫으면 legacy negative의 정체를 복구할 수 없다.

---

## 8. 운전석 실행 명령과 기대 종료코드

모든 명령은 로그인 노드 계산을 피하는 nested ssh이고, `lageunha`와
`/usr/bin/time`을 쓰지 않는다. 로그와 주입 fixture는 `/tmp`다. §8.4만 Makefile의 정상
target인 저장소 루트 `lumina_cuda`를 다시 빌드하며 deck/GPFS에는 쓰지 않는다.

### 8.1 CPU build, Python grammar, CONFIG 음성 대조

```bash
ssh grammar "ssh grammar-debug 'set -euo pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; mkdir -p /tmp/lumina_config_prec_build; gcc -O2 -Wall -Wextra -std=c11 -o /tmp/lumina_config_prec_build/lumina src/lumina_main.c src/lumina_transport.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c src/lumina_cmfgen.c -lm; python3 -m py_compile scripts/run_config_prec_negative_controls.py; python3 scripts/run_config_prec_negative_controls.py --binary /tmp/lumina_config_prec_build/lumina --deck data/tardis_reference_toy06_19p48d | tee /tmp/lumina_config_prec_cpu_negative.txt'"
```

- 기대 최종 rc: 0.
- 기대 marker: 네 줄 모두 `child_rc=1 ... verdict=PASS`, 마지막
  `CONFIG_PREC_NEG_SUMMARY passed=4 total=4`.
- runner가 출력한 `scratch=/tmp/lumina_config_prec_controls_*`를 실제 fixture 경로로
  회귀 대장에 복사한다.

### 8.2 gate OFF 양성/경고 대조

```bash
ssh grammar "ssh grammar-debug 'set -euo pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; mkdir -p /tmp/lumina_config_prec_build; gcc -O2 -std=c11 -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections -o /tmp/lumina_config_prec_build/kshape_harness scripts/kshape_harness.c src/lumina_atomic.c src/lumina_element_wide.c src/lumina_plasma.c src/lumina_cmfgen.c -lm; env -u LUMINA_CONFIG_PREC -u LUMINA_T_INNER_FIX /tmp/lumina_config_prec_build/kshape_harness data/tardis_reference_toy06_19p48d_sivcaiv | tee /tmp/lumina_config_prec_gate_off.txt'"
```

- 기대 rc: 0.
- 기대 marker: `[CONFIG-PREC][WARN]`, `gate=OFF`,
  `source=config.json:T_inner_K`, 마지막 K-shape harness rc=0.

### 8.3 gate ON 직접 FATAL 대조

아래 명령 자체의 기대 rc는 **1**이다.

```bash
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; env LUMINA_CONFIG_PREC=1 /tmp/lumina_config_prec_build/kshape_harness data/tardis_reference_toy06_19p48d_sivcaiv > /tmp/lumina_config_prec_gate_on.txt 2>&1'"
```

- 기대 rc: 1.
- `/tmp/lumina_config_prec_gate_on.txt` 기대 marker:
  `[CONFIG-PREC][FATAL] boundary-temperature declarations disagree`.

### 8.4 CUDA full build

```bash
ssh grammar "ssh grammar-debug 'set -euo pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; make -B -j2 cuda | tee /tmp/lumina_config_prec_cuda_build.txt'"
```

- 기대 최종 rc: 0, CUDA full build rc=0.
- CUDA main은 현재 공통 loader 전에 `cudaGetDevice`를 호출하므로 GPU가 없는
  grammar-debug에서 그 바이너리로 음성 대조를 실행하지 않는다. 판정 구현은 CPU/CUDA가
  함께 링크하는 `src/lumina_atomic.c` 한 곳이고, §8.1의 실제 CPU 바이너리와 §8.2의
  loader harness가 그 경로를 검사한다.

### 8.5 기존 D/K/Z-INERT 회귀

```bash
ssh grammar "ssh grammar-debug 'set -euo pipefail; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; /home/kjhan/.lumina_scratch/run_dbuild_gates.sh | tee /tmp/lumina_config_prec_dk.txt; bash scripts/run_zinert_selftest.sh | tee /tmp/lumina_config_prec_zinert.txt'"
```

- 기대 최종 rc: 0.
- D/K 기대: `DBUILD_GATE_REPASS = PASS (D rc=0, K 7/7)`.
- Z-INERT 기대: 기존 양성 검사 전부 PASS, phantom 음성 대조 child nonzero, 스크립트 rc=0.

### 8.6 diff·금지 대상 확인

```bash
ssh grammar "ssh grammar-debug 'set -e; cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn; git diff --check -- src/lumina_atomic.c src/lumina_main.c src/lumina_cuda.cu scripts/run_config_prec_negative_controls.py docs/CODEX_L0_NFP_CONFIG_PREC.md; git status --short -- src/lumina_atomic.c src/lumina_main.c src/lumina_cuda.cu scripts/run_config_prec_negative_controls.py docs/CODEX_L0_NFP_CONFIG_PREC.md data/tardis_reference_toy06_19p48d'"
```

- 기대 rc: 0.
- `git diff --check` 출력 없음.
- canonical deck 아래 이번 작업으로 생긴 수정 없음.

---

## 9. 음성 대조 총괄

| 계약 | 등록 수 | 실제 구현 | 기대 종료코드 | 현재 상태 |
|---|---:|---|---|---|
| NE-NAMING | FATAL 4 + WARN 1 | 독립 checker/runner 구현 | FATAL 1, WARN 0 | `PENDING_DRIVER_EXECUTION` |
| DECK-FOSSIL | FATAL 4 + WARN 1 | 독립 checker/runner + atomic writer 구현 | FATAL 1, WARN 0 | `PENDING_DRIVER_EXECUTION` |
| CONFIG-PREC | FATAL 4 | 구현됨, CPU/CUDA가 공유하는 loader | child 1, runner 0 | `PENDING_DRIVER_EXECUTION` |

로그인 노드에서는 두 checker를 실행하지 않았으므로 실제 injection 결과를 PASS로 꾸미지
않는다. grammar-debug 운전석 시연 결과가 각 계약의 실행 폐합 조건이다.

---

## 10. 자기 검수와 남은 위험

1. 계약은 정확히 세 개이고 각각 `n_e` 경계 진실성, deck generation, 단일 effective
   boundary-temperature precedence라는 물리 계약 하나만 다룬다.
2. NE 처분을 개명으로 쓰지 않았고 즉시 CMFGEN `<Z>` 덱 재생성도 요구하지 않았다.
3. `4.005038`은 재현 producer가 없으므로 `UNRESOLVED`로 남겼다.
4. CONFIG 판정은 현재 덱의 4152.549 K 차이를 약 5 K 한계와 비교하므로 실제로 걸린다.
5. 세 계약 모두 주입 결함, marker, 기대 rc를 등록했다. NE/DECK도 독립 runner를 구현했고
   운전석 실행 전 상태를 명시했다.
6. 본문에 쓴 파일명과 줄번호는 이 세션에서 `nl -ba`/`rg`로 확인했다.
7. 실행 명령은 nested ssh이며 `/usr/bin/time`, lageunha, GPFS/deck write가 없다.

남은 위험은 다음과 같다.

- CPU/CUDA compile과 음성 대조, D/K/Z 회귀는 운전석 실행 전이다.
- 기본 OFF 동안 현 덱은 계속 실행된다. WARN을 수집하지 않는 launcher는 결함을 놓칠 수
  있으므로 A2-04와 새 덱 인수에서는 ON을 강제해야 한다.
- legacy `plasma_state.csv`에는 temperature semantics/generation ID가 없다. 현재 식은
  builder와 50-shell 상수 profile로 입증됐지만 새 schema에는 이를 명시해야 한다.
- CONFIG-PREC는 `LUMINA_TRAD_COLOR_FIX`의 잘못된 color 주석/동작을 고치지 않는다.
- `4.005038` producer가 끝내 회복되지 않으면 현 덱은 fossil quarantine을 벗어날 수 없다.
- NE와 DECK이 미폐합이므로 CONFIG 구현 완료만으로 L-0 legacy negative나 A2-04를
  진행할 수 없다.
