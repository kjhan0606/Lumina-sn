# A2-08 구현 명세 V2 — signed CPU opacity 게시와 정직한 소비자 차단

- 개정: 11
- 저작: Codex
- 검수: fable
- 기준 HEAD: `694d9cdc297c97082d2c1fa731c5a9fc7ba591ce`
- 단계: A-2 캠페인의 A2-08 하나
- V1 판정: L2 교차 검수 `BLOCK`(BLOCKER 6건)
- 정본: 이 문서가 `docs/SPEC_A2_08_V1.md`를 전부 대체한다.
- 최종 물리 상태: **`BLOCKED_MISSING_CHI_DATA`**

## 0. 규범 우선순위와 개정 11의 확정 처분

이 문서는 다음 존재 파일을 현 HEAD에서 직접 읽어 작성했다.

- `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md`
- `docs/ORDER_OPHYS_RUN_BY_CODEX.md`
- `docs/A2_00_OPHYS_PROFILE.json`
- `docs/A2_01_DISPOSITION_LEDGER.md`, `docs/A2_01_DISPOSITION_LEDGER.json`
- `scripts/a2_01_census_contract.py`, `scripts/run_gate_battery.py`
- `docs/SPEC_A2_06_V5.md`, `validation/a2_06/A2_06_CLOSURE.md`
- `tests/zinert_canonical_tau_fixture.c`
- 이 문서의 census에 적힌 `src` 파일들

상위 ORDER의 단계표는 `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:663-666`에서
A2-08을 CPU opacity `chi_nu`, A2-09를 CPU emissivity `eta_nu`와 재분배,
A2-11을 CPU formal transfer로 나눈다. 개정 11은 이 경계를 문자 그대로 따른다.

> **A2-08의 계약은 signed opacity component와 그 validity/status를 같은 세대로
> 게시하는 데까지다. A2-08은 maser 수송 방정식을 풀지 않는다. signed 값을 처리할
> 수 없는 소비자는 값을 바꾸지 말고 `BLOCKED_NEGATIVE_OPACITY_SEMANTICS`로 경로를
> 막는다.**

따라서 아래 여섯 처분은 재해석할 수 없는 규범이다.

1. `chi`, `tau`, `dtau`, `line_source_S`, `stim`, `corr` 계열의 CPU 소비지점을
   전수 census하고 각 지점을 이관, 잔류-허용, BLOCKED 중 정확히 하나로 처분한다.
2. signed publish와 maser transport solution을 분리한다. `abs`, 0, 양의 floor로
   음수를 조용히 삼키는 행위는 모든 경로에서 금지한다.
3. changed-output allowlist는 구현 전에 봉인하는 불변 manifest다.
4. `A2-01:old7897:T_rad`와 `eta_reemit` producer/CDF builder는 A2-09로 환원한다.
5. formal solver 내부 수식은 바꾸지 않는다. A2-08은 signed 입력 게시까지만 한다.
6. A2-01 census는 수리 완료된 HEAD `694d9cd`의 canonical renderer 규약을 따른다.

## 1. 범위와 성공 정의

### 1.1 A2-08에서 반드시 구현할 것

1. CPU `chi_nu`를 `es`, `bb`, `bf`, `ff`, `total` component로 분리해 signed 값과
   cell별 validity를 게시한다.
2. 두 Sobolev producer의 stimulated-emission clamp와 `1e-100` floor를 제거하고,
   population difference를 직접 쓰는 단일 signed helper로 합친다.
3. BF의 signed net coefficient와 packet 선택용 nonnegative gross event measure를
   서로 다른 배열·타입·API로 분리한다.
4. A2-06이 넘긴 line-source fallback 4행을 checked `RadiationFieldView`와
   status-bearing line-source view로 이관한다. 숫자 `<=0`을 missing sentinel로 쓰지 않는다.
5. CMFGEN replay가 `CMFGEN_REPLAY` provenance의 line block을 `J_nu`와 같은
   transaction에서 원자적으로 commit하도록 배선한다.
6. signed component/line 값이 실제 음수인 고유 `(line,shell)`과
   `(route,shell,bin)` 수를 계수하고 identity artifact를 남긴다.
7. 아래 54개 소비지점 census와 raw grep universe를 정적 checker로 고정하고
   목록 밖 소비지점 0건을 증명한다.
8. L-4 내부 component closure와 truth 입력 부재 상태를 서로 다른 status/rc로 기록한다.

### 1.2 성공 상태

A2-08 완료는 두 상태를 동시에 기록한다.

- `INTERNAL_SIGNED_OPACITY_PUBLISH=PASS`: signed component/status publish, 원자성,
  self-closure, census, selftest, N1-N8, 전 회귀가 PASS한다.
- `L4.CHAIN=L4.ORACLE_INPUT=BLOCKED_MISSING_CHI_DATA`, rc 3: 현 workspace에
  `CHI_DATA`와 `CHI_DATA_INFO`가 없으므로 물리 L-4를 PASS로 쓰지 않는다.

`docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:475-488`의 L-4 기준은 최종 인수조건이다.
내부 closure는 truth-side coverage나 물리 비교를 대신하지 않는다.

### 1.3 비범위

- maser amplification의 Monte Carlo event law, signed optical-depth draw, saturation,
  seed/boundary condition: A2-08 비범위.
- CPU formal transfer의 `dtau`, attenuation, source integration, boundary condition:
  A2-11.
- `eta_reemit`, BF Planck 재방출 교체, full `eta_nu`, macro-atom redistribution,
  L-3/L-5: A2-09.
- 복사평형과 heating/cooling의 signed-opacity 해법, `T_e` root solve: A2-10.
- CUDA/GPU opacity와 transport: A2-12/A2-14/A2-15.
- native J seed와 scalar lifecycle 제거: A2-16/A2-17.
- CMFGEN source, deck, `/gpfs` 원본 수정.

### 1.4 후속 maser arm 제안

ORDER 개정 승인을 전제로 **A2-11M — CPU negative-opacity semantics**를 A2-11의
선행 arm으로 신설한다. A2-11M은 CPU MC의 signed line/continuum event law와 line-overlap
전파를 소유하고, A2-11은 formal 방정식·경계조건을 소유한다. A2-11M이 승인되지 않으면
이 문서에서 A2-11M으로 표기한 경로는 A2-18 통합 전에 별도 단계로 반드시 재귀속해야
하며, 그 전까지 `BLOCKED_NEGATIVE_OPACITY_SEMANTICS`를 유지한다.

## 2. signed 게시 자료형과 물리식

### 2.1 `CpuOpacityPublication`

구현 이름은 저장소 관례에 맞게 조정할 수 있으나 의미는 다음과 같아야 한다.

```text
CpuOpacityPublication {
    required_generation, committed_generation
    epoch, shell_geometry_hash, frequency_edge_hash, atomic_model_hash
    radiation_generation, line_jbar_generation
    population_generation, partition_generation, within_sl_generation
    te_generation, ne_generation, tau_generation

    chi_es[s,b], chi_bb[s,b], chi_bf[s,b], chi_ff[s,b], chi_total[s,b]
    chi_validity[component,s,b]

    tau_sobolev[line,s]
    tau_validity[line,s]
    line_source_S[line,s]
    line_source_validity[line,s]

    bf_net_route[route,s,b]
    bf_event_measure[route,s,b]
    bf_route_validity[route,s,b]
}
```

주파수 edge는 정본 `RadiationField`의 edge를 빌리고 단위는 `cm^-1`, frame은
`comoving`이다. 기존 1000-bin `BFOpacity` 배열은 정본이 아니라 generation-bound
projection이다. candidate 배열과 status는 비공개 버퍼에서 전부 만든 뒤 한 번에
commit한다. cell 하나가 invalid이거나 필수 consumer declaration이 누락되면 새
generation의 일부만 공개할 수 없다. 실패 뒤 public pointer와 committed generation은
이전 세대 그대로다.

### 2.2 validity와 EXACT_ZERO

최소 상태는 서로 다른 enum 값이어야 한다.

```text
VALID
EXACT_ZERO
UNSAMPLED
OUT_OF_GRID
MISS
STALE_GENERATION
QHASH_MISMATCH
PROFILE_MISMATCH
INVALID_POPULATION
INVALID_PARTITION
INVALID_TE
INVALID_NE
NONFINITE
SOURCE_CANCELLATION_SINGULAR
EVENT_MEASURE_UNAVAILABLE
BLOCKED_NEGATIVE_OPACITY_SEMANTICS
FORBIDDEN_FALLBACK
```

`VALID`와 `EXACT_ZERO`만 수치 입력이다. `EXACT_ZERO`의 payload는 정확한 `+0.0`이며
missing sentinel이 아니다. `UNSAMPLED`, `MISS`, OOG, stale를 0으로 대입할 수 없다.
`line_source_S<=0`, `tau<=0`, NULL pointer를 validity로 재해석할 수 없다. dump/checksum도
checked snapshot을 읽고 status를 함께 직렬화한다.

### 2.3 bound-bound와 signed Sobolev

선 `l`의 population difference와 optical depth는 다음 식 하나를 쓴다.

\[
D_l=n_lB_{lu}-n_uB_{ul}
   =B_{lu}\left(n_l-{g_l\over g_u}n_u\right),
\]

\[
\tau_l=K_{Sob} f_{lu}\lambda_l t_{exp}
       \left(n_l-{g_l\over g_u}n_u\right).
\]

`D_l<0`, `tau_l<0`은 유효한 inversion이다. `n_l=0,n_u>0`도 정확한 음수다.
`n_l(1-ratio)` 형태를 쓰지 말고 차이를 직접 계산해 0·무한대 cancellation을 피한다.
`tau=0`은 EXACT_ZERO, NaN/Inf는 NONFINITE다. `1e-100` floor와 stimulated clamp는 없다.
K-FRESH의 required/computed/first-consumer generation 규약은 유지한다.

빈 component는 정규화된 공이동 profile로 보존 적분한다.

\[
\bar\chi^{bb}_{s,b}={1\over\Delta\nu_b}
 \sum_l\int_b {h\nu_l\over4\pi}D_{l,s}\phi_l(\nu)\,d\nu.
\]

이 식은 signed **게시식**이지 transport 해법이 아니다. profile 적분은
`|int phi dnu-1|<=1e-12`여야 한다.

### 2.4 bound-free: signed net과 nonnegative event measure 분리

route별 stimulated bracket은 자르지 않는다.

\[
\chi^{net}_{r,s}(\nu)=\chi^{gross}_{r,s}(\nu)-\chi^{induced}_{r,s}(\nu)
=\chi^{gross}_{r,s}(\nu)\left[1-r_{mod}
 e^{-h(\nu-\nu_{edge})/(kT_e)}\right].
\]

`chi_net`은 signed component에만 들어간다. packet event CDF에는 독립적으로 증명된
`chi_gross>=0`만 들어간다. `max(0,chi_net)`, `abs(chi_net)`, net을 event measure로
재사용하는 것은 금지한다. gross route 자료가 없으면 signed publish는 유지하되 event
consumer만 `EVENT_MEASURE_UNAVAILABLE` 또는
`BLOCKED_NEGATIVE_OPACITY_SEMANTICS`로 막는다.

A2-05의 부분-빈 적분과 weighted-missing 규약을 유지한다. edge 아래는 EXACT_ZERO,
threshold 빈은 실제 교집합 구간만 적분한다. `J_nu`는 radiation-dependent rate/audit에
한 번만 곱하며 opacity coefficient에 다시 곱지 않는다. 이것이 J 이중계상 금지 규약이다.

### 2.5 ES, FF, total과 closure

\[
\chi^{es}_{s,b}=n_{e,s}\sigma_T.
\]

FF는 같은 generation의 `T_e`, `n_e`, ion population과
`-expm1[-hnu/(kT_e)]`를 사용한다. `n_i approximately n_e`, `Z^2 approximately 1`
fallback은 금지한다. candidate total은 고정 순서

```text
total = ((es + bb) + bf) + ff
```

로 한 번 계산한다. 독립 closure는

\[
E_{close}={|\chi_{total}-\sum_c\chi_c|\over
\max(\sum_c|\chi_c|,\mathrm{DBL\_MIN})}
\]

이며 모든 cell과 등록 대역에서 `max E_close<=1e-10`이어야 한다. 모든 component가
EXACT_ZERO면 `E_close=0`이다. component invalid인데 total만 유효한 publish는 금지한다.

### 2.6 line source의 값/상태 분리

population-native line pair는

\[
\eta_l={h\nu_l\over4\pi}n_uA_{ul}\phi_l,\qquad
\chi_l={h\nu_l\over4\pi}(n_lB_{lu}-n_uB_{ul})\phi_l
\]

를 같은 population generation에서 만든다. `chi_l!=0`일 때만 `S_l=eta_l/chi_l`을
materialize한다. `chi_l=0,eta_l!=0`은 숫자 cap이나 fallback이 아니라
`SOURCE_CANCELLATION_SINGULAR`다. finite negative `S_l`도 값으로는 VALID할 수 있다.
consumer는 status를 먼저 보고, `S_l<=0`을 missing으로 간주하지 않는다.

### 2.7 material owner와 scalar 금지

CPU opacity의 물질 입력은 A2-07 publication과 같은 generation의
`PlasmaState.n_electron`, `PlasmaState.T_e`, ion/level population, partition view다.
`OpacityState.electron_density`를 독립 owner나 fallback으로 읽을 수 없다. ABI 때문에
잠시 남기면 const non-owning alias와 generation stamp만 허용하며 allocation/copy/update는
0건이어야 한다. `OpacityState.t_electrons`를 `T_rad`로 초기화해 matter temperature처럼
쓰는 것도 금지한다.

ES/FF/BF/BB builder는 checked material view가 invalid이면 candidate publication을 막고,
이전 `n_e`, deck seed, LTE/Saha, `T_rad`, 0으로 대체하지 않는다. A2-11 formal consumer의
기존 mirror read는 A2-08에서 수식을 고치지 않고 formal entry를 BLOCKED로 유지한 채
A2-11에 넘긴다.

## 3. 소비자 capability 선언과 공통 BLOCK 규약

signed `chi/tau`를 받는 모든 소비자는 manifest에서 다음 중 정확히 하나를 선언한다.

1. **SIGNED_EQUATION**: signed 처리 방정식, overflow domain, 초기조건과 경계조건을
   문서화하고 negative fixture를 통과한다.
2. **SEPARATE_NONNEG_EVENT_MEASURE**: 확률/Poisson event에는 별도 nonnegative gross
   measure를 쓰고 signed net은 계수·회계에만 쓴다. 두 값의 물리 분해 근거와 closure를
   기록한다.
3. **BLOCK_UNSUPPORTED**: 첫 negative identity를 포함한
   `BLOCKED_NEGATIVE_OPACITY_SEMANTICS`로 해당 경로를 실행 전에 막고 후속 단계를 기록한다.

어느 선언에서도 `abs`, `fmax(0,...)`, `x<0?0:x`, 양의 epsilon/floor, positive-only
skip으로 음수를 정상 값처럼 소비할 수 없다. BLOCK은 candidate publication을 지우지
않는다. producer는 signed 값과 카운터를 게시한 뒤, 그 값을 못 받는 **consumer 경로만**
막는다. 동일 process에서 다른 안전한 진단/closure 경로는 실행할 수 있다.

production에서 다음 fallback/config는 설정 시 `FORBIDDEN_FALLBACK`, rc 5다.

- `W B_nu(T_rad)`, `B_nu(T_inner)`, 임의 `B_nu(T_e)` source fallback
- raw/coarse/이전 generation J 또는 Jbar, numeric `line_source_S` fallback
- `chi<0 -> 0`, `fabs(chi)`, `tau<epsilon -> epsilon`, positive-only line drop
- signed net BF를 packet event probability로 사용
- invalid population/material view를 LTE, Saha, 이전 값, 0으로 대체
- `LUMINA_OPACITY_SKIP_Z`로 production channel을 제거
- unsupported signed path에서 nonnegative legacy path로 자동 복귀

## 4. signed 소비지점 전집 — 54개 1:1 처분

아래 행 하나가 semantic consumer site 하나다. 처분 값은 `이관`, `잔류-허용`,
`BLOCKED` 중 정확히 하나다. `BLOCKED` 행의 runtime reason은 전부
`BLOCKED_NEGATIVE_OPACITY_SEMANTICS`이며, 음수가 없는 lane의 기존 동작까지 무조건
막으라는 뜻은 아니다. 각 행은 negative identity가 하나라도 입력될 때 그 경로를
fail-closed하라는 조건부 capability 선언이다.

### 4.1 CPU transport — 3지점

| ID | 현행 witness | 소비 의미 | 처분 | capability / 사유·후속 |
|---|---|---|---|---|
| `T01` | `src/lumina_transport.c:178-260` | 양의 exponential draw와 누적 Sobolev tau 비교 | BLOCKED | ③; signed event law 미정, A2-11M |
| `T02` | `src/lumina_transport.c:562-594`의 `bf_get_event_chi` branch | BF packet event와 Thomson 선택 | 이관 | ②; `bf_event_measure`만 사용, signed net과 타입/API 분리 |
| `T03` | `src/lumina_transport.c:565-569`의 `bf_get_chi` fallback | signed BF net을 continuum event opacity로 사용 | BLOCKED | ③; event measure 없는 fallback 금지, A2-11M |

### 4.2 CMF field solver — 6지점

| ID | 현행 witness | 소비 의미 | 처분 | capability / 사유·후속 |
|---|---|---|---|---|
| `F01` | `src/lumina_cmf_field.c:224-240` | total/coherent chi nonnegative validator | BLOCKED | ③; A2-11 |
| `F02` | `src/lumina_cmf_field.c:301-328,349-382,438-507` | `delta_tau>=0` short-characteristic kernels | BLOCKED | ③; 방정식·경계조건을 A2-11에서 정의 |
| `F03` | `src/lumina_cmf_field.c:710-785` | `chi>0`에서 `eta/chi` source 생성 | BLOCKED | ③; A2-11 |
| `F04` | `src/lumina_cmf_field.c:945-1675` | ray/segment residual과 attenuation | BLOCKED | ③; A2-11 |
| `F05` | `src/lumina_cmf_field.c:1908-1937` | coherent chi를 event/source iteration에 결합 | BLOCKED | ③; A2-11 |
| `F06` | `src/lumina_cmf_field.c:2181-2186` | frozen field nonnegative 재검증 | BLOCKED | ③; signed schema가 생길 때까지 A2-11 |

### 4.3 `lumina_cmfgen.c` — 23지점

| ID | 현행 witness | 소비 의미 | 처분 | capability / 사유·후속 |
|---|---|---|---|---|
| `G01` | `src/lumina_cmfgen.c:281-299` | CHI/ETA dump nonnegative validator·분해 | 이관 | signed component와 validity를 dump; 진단이 값의 부호를 바꾸지 않음 |
| `G02` | `src/lumina_cmfgen.c:736-840` | line-pop dump, stim clamp/floor 재계산 | 이관 | publisher의 signed tau/status를 round-trip; 재계산 clamp 삭제 |
| `G03` | `src/lumina_cmfgen.c:1159-1211` | Stage32 R1 positive tau/chi 검증 | BLOCKED | ③; emissivity/transition 증거이므로 A2-09 |
| `G04` | `src/lumina_cmfgen.c:1277-1300` | R1 writer의 tau cutoff와 raw line source fallback | BLOCKED | ③; A2-09 |
| `G05` | `src/lumina_cmfgen.c:1616-1618` | raw tau/source/BF assembly-input hash | 이관 | status-bearing checked snapshot의 value+status+generation hash |
| `G06` | `src/lumina_cmfgen.c:1787-1799` 중 tau-to-chi | expansion line opacity `1-exp(-tau)` | 이관 | ① 게시식; signed 결과와 overflow status만 생성, solver 아님 |
| `G07` | `src/lumina_cmfgen.c:1795-1838` 중 source/thermal split | raw `line_source_S`, beta, eta 생성 | BLOCKED | ③; eta/source 의미는 A2-09 |
| `G08` | `src/lumina_cmfgen.c:1976-2129` 중 BF/FF/line/total chi 조립 | component producer의 음수 clamp | 이관 | signed component publication; `chi_bf<0`/`chi_ff<0` clamp 삭제 |
| `G09` | `src/lumina_cmfgen.c:2084-2200`의 `S_fixed`, EPAY | `chi_total>0`에서 eta/chi와 thermal source | BLOCKED | ③; eta publication A2-09, formal consumption A2-11 |
| `G10` | `src/lumina_cmfgen.c:2459-2487` | coarse formal `dtau<0 -> 0` | BLOCKED | ③; 내부 수식 변경 금지, A2-11 |
| `G11` | `src/lumina_cmfgen.c:2575-2646` | ALI/scattering `chi_es/chi_total` 비 | BLOCKED | ③; A2-11 |
| `G12` | `src/lumina_cmfgen.c:2747-2804` | tau/source/line-heating 출력 진단 | 잔류-허용 | output-only; A2-11 단계, negative는 raw value+status로만 출력하고 생산 무기여 |
| `G13` | `src/lumina_cmfgen.c:2880-2905` | ray formal `dtau<0 -> 0` | BLOCKED | ③; A2-11 |
| `G14` | `src/lumina_cmfgen.c:2990-2998` | observer continuum `chi0`, `dtau` clamp | BLOCKED | ③; A2-11 |
| `G15` | `src/lumina_cmfgen.c:3073-3098` | interpolated total/es/abs chi clamp와 source | BLOCKED | ③; A2-11 |
| `G16` | `src/lumina_cmfgen.c:3114-3180` | Sobolev tau cutoff 및 raw J/Jbar/source fallback | BLOCKED | ③; signed 입력만 A2-08이 게시, formal은 A2-11 |
| `G17` | `src/lumina_cmfgen.c:3232-3258` | solver/formal chi ratio 진단 | BLOCKED | ③; 실행 중 solver state를 소비하므로 A2-11 |
| `G18` | `src/lumina_cmfgen.c:3571-3654` | CMF field input과 `chih>0` source division | BLOCKED | ③; A2-11 |
| `G19` | `src/lumina_cmfgen.c:3776-3798` | fine formal `dtau<0 -> 0` | BLOCKED | ③; A2-11 |
| `G20` | `src/lumina_cmfgen.c:3843-3858` | formal이 raw `line_source_S` pointer 보유 | BLOCKED | ③; A2-11 |
| `G21` | `src/lumina_cmfgen.c:4110-4350` | fine BF/line deposition, tau skip, source clamp, `ct>0` division | BLOCKED | ③; formal input assembly 내부이므로 A2-11 |
| `G22` | `src/lumina_cmfgen.c:4447-4565` | line-source/tau 진단 dump | 잔류-허용 | output-only; checked snapshot value+status, production 무기여, A2-11 |
| `G23` | `src/lumina_cmfgen.c:5177-5222` | replay line-resolved J/overlap에서 positive tau/source만 사용 | BLOCKED | ③; nonnegative lane의 atomic line-block wiring은 A2-08, maser 의미는 A2-11M |

### 4.4 `lumina_plasma.c` 및 기타 CPU — 18지점

| ID | 현행 witness | 소비 의미 | 처분 | capability / 사유·후속 |
|---|---|---|---|---|
| `P01` | `src/lumina_plasma.c:2148-2169` | Z-INERT tau/source validator와 positive candidate 수 | 이관 | signed/zero/status count로 교체; Z-INERT exact zero는 유지 |
| `P02` | `src/lumina_plasma.c:2985-2995` | bulk Sobolev stim clamp와 `1e-100` floor | 이관 | 단일 signed difference helper |
| `P03` | `src/lumina_plasma.c:7063-7105` | route `fmax(0,1-stim)`과 event CDF | 이관 | ② gross event measure와 signed net을 분리 |
| `P04` | `src/lumina_plasma.c:7673-7797` | BF grid corrfactor clamp와 net 누적 | 이관 | signed route/component publish; event 배열에는 gross만 누적 |
| `P05` | `src/lumina_plasma.c:7116-7142` | BF net/event getter가 같은 scalar 모양 | 이관 | signed-net view와 nonnegative-event view를 다른 타입/API로 분리 |
| `P06` | `src/lumina_plasma.c:4505-4510` | macro-atom transition probability의 escape beta | BLOCKED | ③; transition/redistribution은 A2-09 |
| `P07` | `src/lumina_plasma.c:5683-5970,6821-6884` | tau 순위·band·debug 출력 | 잔류-허용 | output-only; signed histogram/identity를 추가하고 생산 분기 무기여 |
| `P08` | `src/lumina_plasma.c:8854-8900,11284-11288` | RADEQ epsilon/beta의 positive tau 전제 | BLOCKED | ③; A2-10 |
| `P09` | `src/lumina_plasma.c:12051-12078` 중 tau | blanketed-heating negative-tau drop | BLOCKED | ③; signed heating 의미는 A2-10 |
| `P10` | `src/lumina_plasma.c:12064-12078,12122`의 source/base field | line source fallback과 raw heating J | 이관 | checked source/status와 `RadiationFieldView.J_nu`; invalid 시 fallback 없이 block |
| `P11` | `src/lumina_plasma.c:12281-12310` | RADEQ line cooling/response escape beta | BLOCKED | ③; A2-10 |
| `P12` | `src/lumina_plasma.c:13952-13981` | coupled RADEQ escape beta | BLOCKED | ③; A2-10 |
| `P13` | `src/lumina_plasma.c:11497-12175,14046-14176` | registered chi ratio와 line-RE/heating response | BLOCKED | ③; signed energy equation은 A2-10 |
| `P14` | `src/lumina_plasma.c:15532-15616` | MALI/rate beta와 raw lagged line source | BLOCKED | ③; inversion line-rate semantics는 A2-11M |
| `P15` | `src/lumina_plasma.c:17495-17542` | NLTE tau floor와 line-source raw write | 이관 | signed tau + source validity를 같은 transaction에 게시 |
| `P16` | `src/lumina_plasma.c:18289-18339` | overlap correction의 positive tau skip/ratio | BLOCKED | ③; negative가 있으면 feature path 차단, A2-11M |
| `P17` | `src/lumina_plasma.c:18655-18755` | Sobolev formal cutoff, source fallback, attenuation | BLOCKED | ③; formal 내부 변경 금지, A2-11 |
| `P18` | `src/lumina_plasma.c:19020-19109` | CMF formal BF/tau positive-only deposit와 `dt<=0` drop | BLOCKED | ③; A2-11 |

### 4.5 재전수에서 추가된 CPU 소비군 — 4지점

| ID | 현행 witness | 소비 의미 | 처분 | capability / 사유·후속 |
|---|---|---|---|---|
| `E01` | `src/lumina_element_wide.c:1783-1788` | raw signed `tau`에 `fabs`를 적용해 `tau_all`에 누적 | 이관 | ②; signed `tau`와 별도 타입/API의 nonnegative `tau_interaction_measure=abs(tau)` builder로 분리하고 validity·generation을 함께 검사한다. 이 measure는 수송 opacity가 아니라 경계-stage 표현 범위를 재는 총변동(total-variation) 강도다. |
| `E02` | `src/lumina_element_wide.c:2242-2255` | raw `tau_sobolev`를 읽어 전체/경계 measure와 `boundary_opacity` 비율 생성 | 이관 | ②; checked signed snapshot에서 유도한 별도 nonnegative measure view만 누적한다. 흡수·inversion의 상쇄를 허용하면 경계-stage line 강도 coverage를 과소평가하므로 절댓값 총변동 분리가 물리적으로 정당하다. A2-08 publication/consumer migration 단계. |
| `E03` | `src/lumina_element_wide.c:2299-2303` | `boundary_opacity`를 boundary gate와 commit 판정에 투입 | 잔류-허용 | ②; A2-08 consumer-gate 단계에서 분리된 nonnegative measure의 경계-stage 점유율만 읽는다. transfer 계수·event probability·signed closure에는 사용하지 않으며, signed `tau`의 validity/generation 실패 시 gate를 fail-closed한다. |
| `M01` | `src/lumina_main.c:720-733` | raw `tau_sobolev` validation CSV dump | 이관 | output-only dump를 checked snapshot의 `value,validity,generation` 직렬화로 이관한다. 생산 분기에는 무기여하며 A2-08 dump migration 단계다. |

### 4.6 census 불변식

처분 분포는 정확히 다음과 같다.

```text
semantic_consumer_sites = 54
migrate                 = 16
keep_allowed            = 4
blocked                 = 34
unclassified            = 0
duplicate_disposition   = 0
```

`잔류-허용` 네 행 중 기존 세 행은 output-only이며 producer/solver/gate 결과에 무기여다.
`E03`은 예외적으로 ②의 별도 nonnegative total-variation measure만 소비하는
consumer-gate다. signed `tau` 자체를 `abs`로 바꾸는 허용이 아니며, measure의 독립
타입/API·validity·generation과 transfer/event/closure 무기여가 모두 성립해야 한다.
파일 전체, 함수 전체 wildcard, “diagnostic” 한 단어만으로 허용할 수 없다. 각 allow row는
파일, 함수, token occurrence, output artifact 또는 gate metric, 물리적 분리 사유, 후속
단계가 있어야 한다.

## 5. 정적 checker로 목록 밖 0건 증명

구현은 새 `scripts/a2_08_signed_consumer_census.py`와 생성 artifact
`validation/a2_08/A2_08_SIGNED_CONSUMER_CENSUS.json`을 만든다. 이 두 경로는 현재
존재 파일 인용이 아니라 구현 시 생성할 필수 산출물이다.

checker의 lexical universe는 CUDA를 제외한 `src/*.{c,h}` 전체이고, 최소 다음 identifier
family를 case-sensitive token으로 수집한다.

```text
chi, chi_*, *_chi, chi_total, chi_tot, chi_abs, chi_es, chi_line, chi_bf
tau, tau_*, *_tau, tau_sobolev, dtau, delta_tau
line_source_S
stim, stim_*, *_stim, stim_corr, stimfactor
corr, corr_*, *_corr, correction, corrfactor
bf_get_chi, bf_get_event_chi
```

단순 substring hit는 `consumer`, `producer/write`, `lifecycle`, `declaration`, `comment`,
`selftest`, `non-opacity homonym` 중 하나로 전부 분류한다. consumer만 §4의 정확히 한 ID에
귀속한다. registry row schema는 다음과 같다.

```text
id, path, function, anchor_token, occurrence, line_at_manifest
family, access_kind, semantic_site_id
classification
disposition
capability
reason, followup_stage
source_sha256
```

합격 조건은 다음 전부다.

```text
raw_hits == classified_hits
unknown_hits == 0
consumer_sites == 54
migrate_sites == 16
keep_allowed_sites == 4
blocked_sites == 34
consumer_hits_without_site == 0
sites_without_live_hit == 0
duplicate_site_dispositions == 0
silent_abs_zero_floor_hits == 0
raw_line_source_numeric_sentinel_consumers == 0
```

줄번호만 고정하지 않는다. `path + function + anchor_token + 1-based occurrence`와 source
SHA-256을 함께 결박한다. line shift 뒤 token이 다른 의미로 이동하면 FAIL한다. 새 hit를
allowlist에 자동 추가하거나 결과를 본 뒤 classification을 완화할 수 없다. `check`는
human JSON과 checker 내부 canonical registry의 byte-equivalent rendering을 비교한다.

## 6. line-source 4행과 replay line block

### 6.1 A2-06 인계의 정확한 범위

`docs/A2_01_DISPOSITION_LEDGER.md:229-232`와
`validation/a2_06/A2_06_CLOSURE.md:15-19,44`가 A2-08에 넘긴 것은 다음뿐이다.

- `A2-06:old11908:W`
- `A2-06:old11908:T_rad`
- `A2-06:old11915:W`
- `A2-06:old11915:T_rad`
- replay lane의 line block production wiring

blanketed field의 base는 같은 generation의 `RadiationFieldView.J_nu[s,b]`다. line source는
value와 validity를 분리한 checked view로 읽는다. source가 EXACT_ZERO이면 실제 0이고,
invalid이면 `W B_nu(T_rad)`, `B_nu(T_e)`, raw J/Jbar로 내려가지 않는다. `P09` 때문에
negative tau가 있으면 heating 경로는 A2-10까지 BLOCKED지만 signed data publish와
checked input 이관은 완료할 수 있다.

### 6.2 replay 원자성

A2-06의 query set에 `Q_line_source`를 합쳐 line id를 결과를 보기 전에 정렬·deduplicate하고
Q hash를 만든다. CMFGEN replay는 fine deterministic field를 같은 profile로 적분해
`line_jbar`, validity, line id, Q hash, profile id/hash를 채운다. J 후보와 line 후보는 한
commit으로만 전이한다. 어느 후보든 실패하면 둘 다 이전 generation이며, 성공 뒤
`RadiationFieldView`와 `LineJbarView`를 둘 다 refresh한다.

negative tau가 없는 lane에서는 이 wiring을 끝낸다. negative identity가 있으면 `G23`에
따라 replay overlap 소비만 BLOCKED하고, 후보에 signed tau/status가 있었다는 사실과
identity count는 보존한다.

## 7. old7897와 A2-09 환원

`docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:663-664`는 A2-08=CPU opacity,
A2-09=CPU emissivity와 redistribution이다. `docs/A2_01_DISPOSITION_LEDGER.md:165`도
`A2-01:old7897:T_rad`를 A2-09 `REPLACE_PLANCK_REEMISSION_SOURCE`로 둔다. 현 구현은
`src/lumina_plasma.c:8035-8044`에서 BF absorption 뒤 Planck frequency를 표본한다.

따라서 다음은 **A2-09 인계 목록일 뿐 A2-08 구현 대상이 아니다.**

- `A2-01:old7897:T_rad`
- `bf_absorption_event`의 `sample_planck_frequency`
- `eta_reemit` producer
- re-emission weight/CDF builder와 sampler
- 그 경로의 RNG draw-count 변경

A2-08은 이 코드를 수정하거나 re-emission CDF를 새 owner에 넣거나 A2-09에서 다시
재배치하는 addendum을 작성할 수 없다. N5는 이 경계 침범을 잡는 static negative다.

### 7.1 A2-01/A2-06 고정 migration ID의 1:1 처분

canonical 157행 자체를 손으로 고치지 않고 §13.2의 A2-08 MD addendum에 다음을 기록한다.

#### `opacity_rate` 9행

| 고정 ID | 현행 witness | V2 처분 |
|---|---|---|
| `A2-01:old2435:T_rad` | `src/lumina_plasma.c:2624` | dead nebular opacity/rate shadow 제거 |
| `A2-01:old2437:W` | `src/lumina_plasma.c:2626` | dead nebular opacity/rate shadow 제거 |
| `A2-01:old2498:T_rad` | `src/lumina_plasma.c:2695` | dead zeta shadow 제거 |
| `A2-01:old2499:T_rad` | `src/lumina_plasma.c:2696` | dead Te/Trad shadow 제거 |
| `A2-01:old2500:W` | `src/lumina_plasma.c:2697` | dead dilution shadow 제거 |
| `A2-01:old2501:T_rad` | `src/lumina_plasma.c:2698` | dead ratio shadow 제거 |
| `A2-01:old2502:W` | `src/lumina_plasma.c:2699` | dead non-meta dilution shadow 제거 |
| `A2-01:old2503:T_rad` | `src/lumina_plasma.c:2700` | 제거 또는 output-only checked diagnostic으로 격리 |
| `A2-01:old2504:W` | `src/lumina_plasma.c:2701` | 제거 또는 output-only checked diagnostic으로 격리 |

이 9행의 production successor는 checked radiation/material/population view다. 새
`T_color[J]`, fitted W 같은 압축 scalar를 발명할 수 없다.

#### `opacity` 3행

| 고정 ID | 현행 witness | V2 처분 |
|---|---|---|
| `A2-01:old908:T_rad` | `src/lumina_cmfgen.c:908` | opacity publication의 scalar regime 선택 제거; eta/source 선택은 A2-09 BLOCKED |
| `A2-01:old2144:T_rad` | `src/lumina_cmfgen.c:2144` | 같은 처분; formal 소비는 A2-11 BLOCKED |
| `A2-01:old18010:T_rad` | `src/lumina_plasma.c:18314` | formal thermal width이므로 A2-11로 재귀속; A2-08 수식 diff 0 |

#### A2-06 재배치 4행과 emissivity 1행

| 고정 ID | 현행 witness | V2 처분 |
|---|---|---|
| `A2-06:old11908:W` | `src/lumina_plasma.c:12064-12066` | P10 checked line-source/status view |
| `A2-06:old11908:T_rad` | 같은 위치 | P10; Trad fallback 0 |
| `A2-06:old11915:W` | `src/lumina_plasma.c:12073-12078` | P10 base=`RadiationFieldView.J_nu` |
| `A2-06:old11915:T_rad` | 같은 위치 | P10; Trad fallback 0 |
| `A2-01:old7897:T_rad` | `src/lumina_plasma.c:8043-8044` | **A2-09 유지**, A2-08 diff 0 |

합계 9+3+4+1행은 각각 정확히 한 처분을 가진다. 특히 old7897을 A2-08로 옮기는 문구와
old18010의 formal 수식 변경 문구는 V2 addendum에 존재하면 안 된다.

## 8. formal transfer와 A2-11 경계

A2-08은 formal solver에 입력할 signed component/status view를 게시한다. 그 다음은
무조건 다음 순서다.

```text
signed publication committed
-> formal consumer capability check
-> negative identity count == 0 이면 기존 nonnegative lane 진입
-> 하나라도 음수이면 BLOCKED_NEGATIVE_OPACITY_SEMANTICS, rc 3
```

`G10-G11`, `G13-G21`, `F01-F06`, `P17-P18`의 `dtau`, `eta/chi`, attenuation,
source integration, cutoff, boundary condition은 A2-08에서 바꾸지 않는다. 특히
`src/lumina_cmfgen.c:2460,2895,2994,3077-3097,3115,3788`,
`src/lumina_cmf_field.c:227`, `src/lumina_plasma.c:18656,19057`은 clamp/skip을
signed 공식으로 고쳐 실행시키는 것이 아니라, entry capability check 뒤 도달 불가능해야
한다. formal source raw read `src/lumina_cmfgen.c:3153,3159`도 A2-11 인계다.

## 9. 불변 changed-output allowlist

### 9.1 경로, 생성 시점과 봉인

manifest의 정확한 경로는 다음이다.

```text
validation/a2_08/A2_08_CHANGED_OUTPUT_ALLOWLIST.json
```

구현자는 **첫 `src` 수정 전에** baseline scan으로 이 파일을 만들고 canonical JSON
(UTF-8, sorted keys, indent 2, LF, 마지막 LF 1개)으로 직렬화한다. SHA-256은
`validation/a2_08/A2_08_CHANGED_OUTPUT_ALLOWLIST.sha256`에 GNU sha256sum 형식으로
기록한다. 두 파일만 담은 pre-implementation seal commit의 commit id와 JSON blob id를
`A2_08_IMPLEMENTATION_START.json` 및 회귀 대장의 `input_manifest_hash`에 기록한다.

모든 gate는 다음 세 값을 동시에 비교한다.

1. 현재 JSON의 SHA-256
2. `.sha256`에 봉인된 SHA-256
3. seal commit의 해당 path blob을 읽어 계산한 SHA-256

하나라도 다르거나 seal commit이 첫 `src` diff보다 늦으면 rc 2 FAIL이다. 구현 뒤 JSON과
sidecar를 함께 고치는 것도 seal commit 비교에서 실패한다. 결과를 본 뒤 새 identity를
추가하는 것은 허용되지 않는다. 예상 밖 변화는 allowlist 갱신이 아니라 회귀 FAIL이다.

### 9.2 schema

```json
{
  "schema": "lumina-a2-08-changed-output-allowlist-v1",
  "stage": "A2-08",
  "baseline_head": "40-hex",
  "created_before_source_edit": true,
  "canonicalization": "utf8-sorted-keys-indent2-lf-final-lf",
  "entries": [
    {
      "id": "stable-id",
      "artifact_or_symbol": "exact path or symbol",
      "identity_kind": "line_shell|route_shell_bin|shell_bin|buffer|diagnostic",
      "identity": {},
      "before_value_or_sha256": "exact value/hash/status",
      "allowed_change_kind": "enum",
      "expected_after_constraint": "machine-checkable predicate",
      "reason_code": "enum",
      "owner_stage": "A2-08"
    }
  ],
  "forbidden_scopes": ["whole_spectrum", "all_opacity", "physics_changed"]
}
```

`identity`는 line/route/shell/bin의 정수 key를 모두 가진다. 범위 wildcard와 결과 의존
predicate는 금지한다.

### 9.3 허용 범주

최소 허용 범주는 다음과 같다.

- inversion `(line,shell)`의 signed tau와 BB chi.
- stimulated BF clamp가 발화하던 `(route,shell,bin)`의 signed BF/total chi.
- **`1e-100` floor 제거로 정확히 `1e-100 -> 0`이 되는 `(line,shell)`**. reason code는
  `REMOVE_NUMERIC_FLOOR_EXACT_ZERO`로 고정한다.
- status-bearing line source publication과 기존 fallback이 발화하던 exact identity.
- A2-06 line-source 4행의 checked J/source provenance.
- replay line block generation/hash/status.
- component/status diagnostic artifact와 owner pointer provenance.

`tests/zinert_canonical_tau_fixture.c:52-61`은 현재 exact-zero tau도 `1e-100`으로 만들므로,
이 범주는 inversion line이나 active-nonzero-tau 범주에 합치면 안 된다. pre-change offline
evaluation으로 unfloored expression이 정확히 0인 identity를 먼저 열거한다.

A2-09 재방출, A2-10 `T_e`, A2-11 spectrum/formal output, packet fate와 bolometric
luminosity 변화는 A2-08 allowlist에 넣을 수 없다.

## 10. 카운터, status와 종료코드

### 10.1 필수 카운터

thread-local 집계 후 deterministic reduction하고 `nlte_free`에서 정확히 한 줄
`[A2-08][SIGNED-OPACITY]`로 보고한다.

```text
generation_required, generation_committed
shells_attempted, shells_published, cells_attempted, cells_published
es_terms, bb_terms, bf_terms, ff_terms
exact_zero_es, exact_zero_bb, exact_zero_bf, exact_zero_ff
negative_tau_line_shells
negative_bb_line_shells
negative_bf_route_shell_bins
negative_bf_shell_bins
negative_total_shell_bins
blocked_negative_transport, blocked_negative_formal
blocked_negative_heating, blocked_negative_transition
blocked_stale, blocked_unsampled, blocked_oog, blocked_miss
blocked_profile, blocked_qhash, blocked_population, blocked_te, blocked_ne
source_valid, source_exact_zero, source_negative
source_cancellation_singular, event_measure_unavailable
closure_failures, nonfinite_failures
fallback_attempts, abs_attempts, zero_clamp_attempts, floor_attempts
raw_view_attempts, partial_publish_attempts
replay_line_blocks_attempted, replay_line_blocks_committed
```

`negative_*`는 loop hit가 아니라 고유 identity 수다. identity CSV/JSON의 행 수와 정확히
같아야 한다. 실제 음수 fixture에서 `negative_tau_line_shells>0`과
`negative_bf_route_shell_bins>0`을 각각 요구한다. 음수 count가 0이어야 PASS라는 조건을
두지 않는다. counter와 path status/rc는 별개다.

### 10.2 종료코드

| rc | 의미 |
|---:|---|
| 0 | 요청한 internal gate PASS 또는 negative wrapper가 기대 child 실패를 확인 |
| 2 | 사용법, I/O, schema, manifest/hash, census/renderer 오류 |
| 3 | truth/upstream 또는 negative semantics의 정직한 `BLOCKED_*` |
| 4 | 계산 완료 후 metric/self-closure/negative expectation FAIL |
| 5 | forbidden fallback, raw/stale read, silent clamp/floor, partial publish 위반 |

모든 gate JSON은 `status`, `reason_code`, `child_rc`, `wrapper_rc`를 가진다.
internal PASS와 physical/BLOCKED status를 합치지 않는다.

## 11. L-4 gate 사전등록

### 11.1 생성 artifact

구현은 `validation/a2_08/` 아래 최소 다음을 생성한다.

```text
A2_08_OPACITY_COMPONENTS.npz
A2_08_OPACITY_COMPONENTS_MANIFEST.json
A2_08_COMPONENT_INTEGRALS.csv
A2_08_NEGATIVE_LINE_SHELLS.csv
A2_08_NEGATIVE_ROUTE_SHELL_BINS.csv
A2_08_SIGNED_CONSUMER_CENSUS.json
A2_08_SELFTEST.json
A2_08_L4_GATE.json
A2_08_REGRESSION_LEDGER.jsonl
```

component manifest는 grid edge, shell boundary, units, frame, signed/net 규약, summation
order, generation/hash, validity count를 가진다. 등록 대역은 wavelength 기준
`450-918`, `918-1290`, `1290-2000`, `2000-10000`, `10000-25000` Angstrom이다.

### 11.2 CHAIN과 ORACLE_INPUT

| lane | 입력 | 내부 판정 | 최종 L-4 |
|---|---|---|---|
| CHAIN | current committed RF/LineJbar + A2-07 population/Te/ne | signed publish/closure | `BLOCKED_MISSING_CHI_DATA` |
| ORACLE_INPUT | 같은 checked API로 commit한 CMF J/Jbar/population/Te/ne | opacity 층 분리 | `BLOCKED_MISSING_CHI_DATA` |

ORACLE_INPUT도 raw 배열을 직접 주입하지 않는다. upstream BLOCKED가 있으면 더 구체적인
reason을 병기하되 CHI 부재를 가리지 않는다.

### 11.3 truth-side 활성집합과 `f_cov`

CHI_DATA가 도착하면 Lumina 결과를 열기 전에 writer schema로 component inclusion,
units, frame, depth/frequency order와 sign convention을 고정한다. truth component의
absolute contribution을 내림차순 누적해 99.9% 최소 접두 활성집합을 만들고 경계 동률을
전부 포함한다.

\[
f_{cov}={\sum_{active\cap matched}|\chi_C|\Delta\nu
\over\sum_{active}|\chi_C|\Delta\nu}.
\]

stale/unsampled/OOG/unmatched는 truth 분모에서 빠지지 않는다. 합격선은 다음과 같다.

- component truth coverage `f_cov>=0.95`
- 공통 셸별 signed total `E_1<=0.15`, 분모 `sum Delta_nu |chi_C|`
- 다섯 대역 각각 signed `E_B<=0.15`
- CPU component-sum/total self-closure `<=1e-10`
- CMFGEN negative-active interval sign mismatch 0

MC 영향 metric의 95% CI half-width는 합격폭의 1/3 이하여야 한다. 통계오차를 물리오차에서
빼지 않는다.

### 11.4 truth 부재와 schema

`docs/A2_00_OPHYS_PROFILE.json:16-19`는 CHI/ETA exact file을 요구하고 `:68-71`은
units/frame을 요구하며 `:89-91`은 MEANOPAC/GENCOOL/NEG_OPAC 대체를 금지한다.
`docs/ORDER_OPHYS_RUN_BY_CODEX.md:67`은 CHI/ETA writer가 `CMF_FLUX_PARAM: T [WR_ETA]`
formal 단계임을 기록한다. component inclusion schema가 증명되지 않으면 파일이 있어도
`BLOCKED_CHI_SCHEMA`다. `NEG_OPAC`은 sign 진단일 뿐 값 truth가 아니다.

## 12. N1-N8 음성 대조

모든 poison은 baseline과 별도 child process, 한 poison만, 고유 marker를 사용한다.
marker 미발화, child rc 불일치, poisoned PASS, baseline FAIL은 wrapper FAIL이다.

| ID | poison | marker | 기대 거부 | child/wrapper rc |
|---|---|---|---|---|
| N1 | stimulated correction 제거 | `A2_08_NEG_STIM_OFF` | signed coefficient/rate identity 또는 L-4 sign FAIL | `4/0` |
| N2 | BF edge 하나를 canonical bin 하나 이동 | `A2_08_NEG_BF_EDGE_SHIFT` | partial-edge digest/EB FAIL | `4/0` |
| N3 | opacity channel 하나 제거 | `A2_08_NEG_CHANNEL_DROP` | required-component/analytic total FAIL | `4/0` |
| N4 | negative chi/tau를 abs/0/floor | `A2_08_NEG_CHI_CLAMP` | sign digest와 silent-clamp checker FAIL | `5/0` |
| N5 | old7897/eta_reemit/CDF를 A2-08 diff에 포함 | `A2_08_NEG_A209_SCOPE` | ownership/static scope checker FAIL | `5/0` |
| N6 | raw Jbar 또는 numeric `line_source_S` fallback | `A2_08_NEG_RAW_JBAR` | checked-view/census/generation FAIL | `5/0` |
| N7 | RF/line/pop stamp 하나 stale | `A2_08_NEG_STALE_SOURCE` | publish 0, stale counter >0 | `5/0` |
| N8 | replay commit에서 line block 제거 | `A2_08_NEG_REPLAY_LINELESS` | dual-view atomicity FAIL | `5/0` |

N3는 total까지 재합산해 closure만 통과하는 우회를 required-component presence와 analytic
oracle로 잡는다. 각 poison은 실제 witness identity, before/after hash, 발화 count를
JSON에 남긴다.

## 13. selftest와 회귀 전판

### 13.1 필수 selftest

1. ES 단위와 exact-zero `n_e`.
2. two-level BB normal/zero/inversion과 signed tau bit round-trip.
3. `1e-100 -> EXACT_ZERO` fixture와 별도 manifest category.
4. profile integral, bin 경계 line의 보존 적분.
5. piecewise-linear BF threshold partial bin과 OOG.
6. BF bracket이 음수인 fixture, signed net/gross event closure, clamp poison.
7. FF의 `T_e`, `n_e`, ion charge sum과 invalid material view.
8. cell/대역 component closure `<=1e-10`.
9. VALID/EXACT_ZERO/missing/stale/profile/Qhash 전파.
10. 중간 shell 실패 시 partial publish 0.
11. line source zero/negative/singular와 numeric-sentinel 금지.
12. replay J+line atomic success와 양방향 failure injection.
13. T01/F01/G16/P09/P17의 negative input이 정확히 BLOCKED, 값은 게시된 채 유지.
14. 고유 negative identity counter와 artifact 행 수 일치.
15. N1-N8 marker와 rc.

### 13.2 A2-01 canonical renderer 양립 규약

현 HEAD에서 `python3 scripts/a2_01_census_contract.py check`는 rc 0이며
`rows=157`, `completed=20`, `unclassified=0`이다. 구현 회귀의 첫 항목으로 유지한다.

`scripts/a2_01_census_contract.py:372-393`의 renderer는 canonical 157행 표를 먼저 만들고,
기존 addenda를 뒤에 붙인다. `:396-408`은 첫 `## ADDENDUM (`부터 EOF까지를 추출하고
필수 A2-05/06/07 heading을 확인한다. `:513-527`의 check는 JSON이 canonical SITE_DATA와
같고 MD 전체가 `markdown(document, addenda)`와 byte-exact인지 검사한다.

따라서 A2-08 원장 보강은 canonical 157행이나 JSON row를 손으로 고치지 않고, 기존
A2-07 addendum **뒤 EOF**에 다음 heading으로 append한다.

```text
## ADDENDUM (A2-08 구현, YYYY-MM-DD) — signed opacity 소비지점 54행 처분
```

addendum은 UTF-8/LF이며 끝의 whitespace를 제거하고 마지막 LF 정확히 하나를 둔다.
그 상태에서 `write`가 추출한 addendum과 다시 붙인 addendum이 byte-exact하다. 구현 시
`REQUIRED_ADDENDA`에 A2-08 heading을 추가하되 renderer/extract 규약을 바꾸지 않는다.
별도 A2-08 checker가 54행을 검증하고, A2-01 canonical row count 157은 그대로다.

`scripts/run_gate_battery.py:25-40,369-373`은 이 A2-01 check를 build 전 preflight로
영구 실행한다. A2-08 driver가 이를 우회하거나 post-build로 늦출 수 없다.

### 13.3 배터리 Z 배선

새 CPU opacity translation unit을 만들면 현재 hard-coded인 네 Z compile list 모두에
직접 link한다.

- Z-validator: `scripts/run_gate_battery.py:139-145`
- Z-tau: `scripts/run_gate_battery.py:148-153`
- Z-population: `scripts/run_gate_battery.py:156-162`
- Z-canonical: `scripts/run_gate_battery.py:165-173`

`scripts/run_zinert_selftest.py:57-88`에 A2-08 binary 인자와 definition을 추가하고 결과
row를 추가한다. 현재 `scripts/run_gate_battery.py:22`의 Z=6은 Z=7, 전체 36은 37로
올린다. `cpu_link_sources()`를 쓰는 K/CP에 우연히 link되는 것은 Z 네 곳 배선을 대신하지
않는다.

현재 canonical expectation은 `scripts/run_zinert_selftest.py:16-19`의

```text
active_lines=2211572
active_tau_bit_differences=0
active_tau_fnv64=1cfbc8dba0b0f23f
```

이다. signed/floor 제거로 허용된 identity만 pre-sealed allowlist에 따라 변경할 수 있다.
Z-INERT exact-zero, active line set, allowlist 밖 active tau는 변하면 안 된다.

### 13.4 회귀 목록

최소 회귀는 다음이다.

1. `python3 scripts/a2_01_census_contract.py check` preflight.
2. `make lumina`.
3. A2-02/02c resolution/union/replay 기존 gates.
4. A2-03 radiation-field, parity, byte-parity, callgraph gates.
5. A2-04 commit/replay, L0 replay, classic debt sweep.
6. A2-05 BF rate selftest/gates와 기존 BLOCKED 상태 보존.
7. A2-06 Jbar/dual-commit/gates와 기존 BLOCKED 상태 보존.
8. A2-07 population selftest/census/gates/classic sweep.
9. A2-08 census, publication, closure, BLOCK surface, N1-N8, L-4 blocked assertion.
10. `python3 scripts/run_gate_battery.py` 37 case와 serial/parallel table equivalence.

## 14. 구현 순서와 산출 보고

1. pre-change grep registry와 54-site canonical census를 고정한다.
2. changed-output allowlist를 첫 `src` 수정 전에 만들고 seal commit/hash를 기록한다.
3. signed line-opacity helper와 validity를 만들고 두 producer/dump를 이관한다.
4. ES/BB/BF/FF/total component owner와 atomic publish를 구현한다.
5. BF signed net/nonnegative event measure를 타입/API 수준에서 분리한다.
6. A2-06 line-source 4행을 checked view로 이관한다.
7. replay J+line block의 dual-view atomic commit을 배선한다.
8. §4의 34개 unsupported consumer entry에 공통 capability check와 정직한 BLOCK을 단다.
   formal 내부 수식은 수정하지 않는다.
9. counter, identity artifact, selftest, N1-N8, static checker를 구현한다.
10. A2-01 MD addendum을 canonical renderer 호환 방식으로 append한다.
11. Z 네 build와 `run_zinert_selftest`를 배선하고 회귀 전판을 실행한다.
12. L-4를 `BLOCKED_MISSING_CHI_DATA`, rc 3으로 기록한다.

구현 보고서는 최소 다음을 포함한다.

- 기준/최종 HEAD와 source diff.
- 54 ID의 구현 후 anchor와 처분 변화 0 증명.
- raw grep hit 전량 분류, unknown 0.
- signed 식·unit·frame·grid·generation/status schema.
- negative `(line,shell)` 및 `(route,shell,bin)` 실제 count와 artifact hash.
- self-closure worst identity와 값.
- consumer별 capability와 BLOCK 실제 reason/rc.
- raw/fallback/clamp/floor/partial-publish count 0.
- allowlist seal commit/blob/SHA 검증.
- replay dual-view atomicity.
- N1-N8 actual child/wrapper rc.
- A2-01 renderer byte-exact check와 battery preflight.
- Z 네 build + Z=7 + total=37 증거.
- CHAIN/ORACLE_INPUT의 L-4 BLOCKED 상태.
- A2-09/A2-10/A2-11/A2-11M 인계 목록.

## 15. 단계 회귀 대장

`validation/a2_08/A2_08_REGRESSION_LEDGER.jsonl`에는 정확히 한 JSON object를 남기고
`docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:689-710`의 필드를 모두 포함한다.

```text
stage_id=A2-08
contract=docs/SPEC_A2_08_V2.md
source_tree_hash
input_manifest_hash={allowlist_sha256,seal_commit,blob_id}
oracle_id
node
command
exit_status
new_layer_status={
  INTERNAL_SIGNED_OPACITY_PUBLISH:PASS,
  L4:{CHAIN:BLOCKED_MISSING_CHI_DATA,ORACLE_INPUT:BLOCKED_MISSING_CHI_DATA}
}
all_previous_layer_statuses
negative_control_status={N1,N2,N3,N4,N5,N6,N7,N8}
coverage={truth_f_cov:null,reason:BLOCKED_MISSING_CHI_DATA}
metric_values={cell_closure,band_closure,negative_identity_counts,replay_atomicity}
changed_output_allowlist
guard_hits
fallback_hits
rng_seed
mc_confidence
artifact_paths
driver_signoff={author:Codex,reviewer:fable}
```

`truth_f_cov:null`인 상태에서 L-4를 PASS로 쓸 수 없다.

## 16. fable 자기검수 체크리스트

- [ ] semantic consumer 54행이 이관 16/잔류-허용 4/BLOCKED 34로 정확히 분류됐다.
- [ ] CPU family grep raw hit의 unclassified와 목록 밖 consumer가 0이다.
- [ ] signed component와 status가 같은 generation으로 원자 publish된다.
- [ ] EXACT_ZERO가 missing/invalid와 구별된다.
- [ ] 두 Sobolev producer와 dump가 단일 signed helper/view를 쓴다.
- [ ] BF signed net과 gross event measure가 별도 타입/API이고 closure가 있다.
- [ ] 실제 negative line-shell과 route-shell-bin count가 0으로 숨겨지지 않는다.
- [ ] unsupported consumer가 abs/0/floor 대신 정확한 BLOCK reason/rc를 낸다.
- [ ] formal 내부 `dtau`, eta/chi, attenuation, boundary 수식 diff가 0이다.
- [ ] A2-06 line-source 4행과 replay line block만 이관됐다.
- [ ] old7897/eta_reemit/CDF/RNG diff가 0이고 A2-09 인계에 있다.
- [ ] allowlist가 구현 전 seal되었고 JSON/sidecar/git blob 세 hash가 일치한다.
- [ ] `1e-100 -> 0`이 독립 allowlist category다.
- [ ] N1-N8이 고유 marker와 실제 witness/rc를 가진다.
- [ ] A2-01 check가 build 전 rc 0이고 MD renderer가 byte-exact다.
- [ ] 새 TU가 Z compile 네 곳과 Z runner에 모두 연결됐다.
- [ ] internal PASS와 L-4 `BLOCKED_MISSING_CHI_DATA`, rc 3이 분리됐다.

## 17. 후속 단계 인계

### A2-09

- `A2-01:old7897:T_rad`, BF Planck re-emission, `eta_reemit`, CDF builder/sampler.
- `G03`, `G04`, `G07`, `G09`, `P06`의 eta/source/transition semantics.
- full eta components, redistribution, L-3/L-5.

### A2-10

- `P08`, `P09`, `P11-P13`의 signed heating/cooling와 RADEQ 해법.
- signed opacity를 포함한 `T_e` root와 energy boundary conditions.

### A2-11

- `F01-F06`, `G10-G11`, `G13-G22`, `P17-P18`의 formal 방정식,
  boundary condition, observer/output.
- A2-08 signed input/status view를 유지하고 clamp로 되돌리지 않는다.

### A2-11M 제안 arm

- `T01`, `T03`, `G23`, `P14`, `P16`의 CPU MC event law, overlap, inversion rate 의미.
- saturation/seed/boundary condition과 finite amplification domain.

## 18. 개정 시 실측

측정 HEAD는 `694d9cdc297c97082d2c1fa731c5a9fc7ba591ce`이며, 다음은 그
작업트리에서 `rg -n`, `nl -ba`와 checker 실행으로 확인했다.

1. CUDA를 제외한 `src/*.{c,h}`에서 `chi/tau/dtau/line_source_S/stim/corr` identifier
   family lexical grep은 **1,945행**이다. 파일별 주요 hit는
   `lumina_plasma.c` 755, `lumina_cmfgen.c` 585, `lumina_cmf_selftest.c` 273,
   `lumina_cmf_field.c` 111, `lumina.h` 49행이다.
   재현 명령은 다음과 같다.

   ```bash
   rg -n -i '\b(dtau|tau|chi|stim|corr)\w*|\w*(dtau|tau|chi|stim|corr)\b|line_source_S' \
     src --glob '*.{c,h}' --glob '!*.cu'
   ```

2. direct signed-array/API 후보 grep은 **202 expression**이며, 생산/수명/선언/주석/
   selftest/homonym을 제외해 §4의 **54 semantic consumer site**로 정규화했다.
3. V1 8지점 밖 positive-only witness를 실제 확인했다:
   `src/lumina_cmfgen.c:281,1617,1788,2460,2895,3115,3788`,
   `src/lumina_plasma.c:2166,12058,18656,19057`.
4. 가장 깊은 solver 전제는 `src/lumina_transport.c:178-260`의 양의 exponential draw,
   `src/lumina_cmfgen.c:2089`의 `chi_total>0` source division,
   `src/lumina_cmf_field.c:224-240`의 nonnegative validator다.
5. formal clamp는 `src/lumina_cmfgen.c:2459-2487,2894-2905,2990-2998,
   3073-3098,3787-3798`, `src/lumina_plasma.c:18655-18656,19056-19057`에서 확인했다.
6. raw line source 소비는 `src/lumina_cmfgen.c:785-786,1297,1617,1795,3855,
   4274,4313,4464,4557,5215`와 `src/lumina_plasma.c:2166-2167,12064-12065,
   15604-15605,18689-18690,19084-19085`에서 확인했다.
7. ORDER 단계 근거는 `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:653-673`, L-4 기준은
   `:475-488`, 회귀 대장 필드는 `:689-710`이다.
8. old7897의 현 위치는 `src/lumina_plasma.c:8035-8044`, 원장 귀속은
   `docs/A2_01_DISPOSITION_LEDGER.md:165`, ORDER 귀속은
   `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:663-664`다.
9. A2-06 인계 근거는 `docs/A2_01_DISPOSITION_LEDGER.md:219-243`과
   `validation/a2_06/A2_06_CLOSURE.md:15-19,44`다.
10. `tests/zinert_canonical_tau_fixture.c:52-61`은 stim clamp 뒤 모든 `tau<1e-100`을
    `1e-100`으로 만들며, current active expectation은 같은 파일 `:66-68`과
    `scripts/run_zinert_selftest.py:16-19`에서 확인했다.
11. `python3 scripts/a2_01_census_contract.py check` 실측은 rc 0,
    `rows=157 completed=20 unclassified=0`이다. canonical renderer/addendum 규약은
    `scripts/a2_01_census_contract.py:372-408,513-527`, battery preflight는
    `scripts/run_gate_battery.py:25-40,369-373`에서 확인했다.
12. 현 battery는 `scripts/run_gate_battery.py:22`에서 D19/K7/Z6/CP4=36이고,
    Z hard-coded build는 `:139-173`의 네 곳이다.
13. `docs/A2_00_OPHYS_PROFILE.json:16-19,68-71,89-91`은 CHI/ETA exact file,
    units/frame, 대체 금지를 요구한다. 현 `docs`/`validation`에는 이름이 정확히
    `CHI_DATA`, `CHI_DATA_INFO`인 파일이 **0개**다.
14. 누락 재검수로 다음 다섯 파일을 위와 같은 identifier-family 정규식으로 파일 전체
    재전수했다. lexical line hit는 `src/lumina_element_wide.c` **32**,
    `src/lumina_main.c` **21**, `src/lumina_cuda.cu` **383**,
    `src/lumina_transport.c` **26**, `src/lumina_cmf_field.c` **111**이다.
    `lumina_element_wide.c`에서는 ionization-potential `chi` homonym과 declaration/selftest를
    제외하고 `E01-E03` 세 소비군, `lumina_main.c`에서는 lifecycle/comment를 제외하고
    `M01` 한 소비군을 추가했다. `lumina_transport.c`의 소비는 기존 `T01-T03`,
    `lumina_cmf_field.c`의 소비는 기존 `F01-F06`에 전부 귀속되어 추가 누락은 0건이다.
    CUDA 383행은 §1.3과 §5의 CPU lexical universe 밖이며 A2-12/A2-14/A2-15까지
    잔류-허용하는 GPU producer/consumer·declaration·diagnostic 군으로 확인했다. 이 CUDA
    결과는 54-site CPU 처분 분포에는 넣지 않는다.
15. 이 명세 개정 중 `src`, script, test, 원장, deck은 수정하지 않았다. 새 산출물은
    `docs/SPEC_A2_08_V2.md` 하나다.
